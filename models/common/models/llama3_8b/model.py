# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.1-8B Transformer model.

Model:
    Llama3Transformer1D — pure forward methods, no input/output processing

Executor wrappers live in models/common/models/llama3_8b/executor.py.

Architecture:
    Llama3Transformer1D (1D only — non-TG)
    ├── Embedding1D
    ├── RotarySetup1D
    ├── TransformerBlock1D × n_layers
    │   ├── RMSNorm1D  (attention_norm)
    │   ├── Attention1D
    │   ├── RMSNorm1D  (ff_norm)
    │   └── MLP1D
    ├── RMSNorm1D  (final norm)
    ├── LMHead1D
    └── Sampling1D (optional)

Loop policy functions (run_teacher_forcing, run_perf_benchmark) are in
models/common/models/executor.py.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.common.device_utils import get_device_name
from models.common.lightweightmodule import LightweightModule
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _compute_kernel_config_hifi2
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _create_dram_sharded_mem_config
from models.common.modules.rmsnorm.rmsnorm_1d import SHARD_HEIGHT, RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import TT_CCL, default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_out_subblock_w, nearest_32, num_to_core_range_set, pad_dim_to_size

# =============================================================================
# Runtime Config
# =============================================================================


class Llama31DecoderPrecision:
    """Per-decoder tensor dtype and math-fidelity selection."""

    _DTYPES = {
        "bfp4": ttnn.bfloat4_b,
        "bfp8": ttnn.bfloat8_b,
        "bf16": ttnn.bfloat16,
        None: None,
    }

    @classmethod
    def from_string(cls, optimizations: str):
        if optimizations == "performance":
            return cls.performance
        if optimizations == "accuracy":
            return cls.accuracy
        raise ValueError(
            f"Invalid optimization configuration: {optimizations}. Allowed values are 'performance' or 'accuracy'"
        )

    @classmethod
    def performance(cls, num_decoders: int, model_name: str):
        inst = cls(num_decoders, model_name, cls._performance_settings(model_name))
        if model_name == "Llama-3.1-8B-Instruct" and num_decoders > 31:
            inst._tensor_precision[31]["ff1_ff3"] = "bfp8"
            inst._op_fidelity[31]["li_ff1_ff3"] = "hifi2fp16"
            inst._update_full_name()
        inst.__name__ = "performance"
        return inst

    @classmethod
    def accuracy(cls, num_decoders: int, model_name: str):
        inst = cls(num_decoders, model_name, cls._accuracy_settings(model_name))
        inst.__name__ = "accuracy"
        return inst

    def __init__(self, num_decoders: int, model_name: str, settings: dict | None = None):
        self.model_name = model_name
        default_tensor_precision, default_op_fidelity = self._default_settings()
        settings = settings or {}
        default_tensor_precision.update(settings.get("tensor_precision", {}))
        default_op_fidelity.update(settings.get("op_fidelity", {}))
        self._tensor_precision = {decoder_id: dict(default_tensor_precision) for decoder_id in range(num_decoders)}
        self._op_fidelity = {decoder_id: dict(default_op_fidelity) for decoder_id in range(num_decoders)}
        self._update_full_name()

    @staticmethod
    def _base_model_name(model_name: str):
        for suffix in ("-Instruct", "-instruct"):
            if model_name.endswith(suffix):
                return model_name[: -len(suffix)]
        return model_name

    @classmethod
    def _accuracy_settings(cls, model_name: str):
        base_model_name = cls._base_model_name(model_name)
        if base_model_name.startswith("Llama-3") or base_model_name.startswith("Meta-Llama-3"):
            return {
                "tensor_precision": {
                    "wqkv": "bfp8",
                    "kv_cache": "bfp8",
                    "wo": "bfp8",
                },
                "op_fidelity": {
                    "li_ff1_ff3": "hifi2fp16",
                    "li_ff2": "hifi2fp16",
                },
            }
        return {
            "tensor_precision": {
                "wqkv": "bf16",
                "kv_cache": "bf16",
                "wo": "bf16",
            },
            "op_fidelity": {
                "li_qkv_decode": "hifi4",
                "li_qkv_prefill": "hifi4",
                "sdpa_decode": "hifi4",
                "sdpa_prefill": "hifi4",
                "li_o_decode": "hifi4",
                "li_o_prefill": "hifi4",
            },
        }

    @classmethod
    def _performance_settings(cls, model_name: str):
        return {
            "tensor_precision": {"ff1_ff3": "bfp4"},
            "op_fidelity": {"li_ff1_ff3": "lofi"},
        }

    @staticmethod
    def _default_settings():
        return (
            {
                "ff1_ff3": "bfp8",
                "ff2": "bfp8",
                "wqkv": "bfp8",
                "wo": "bfp8",
                "kv_cache": "bfp8",
                "activation": None,
            },
            {
                "li_ff1_ff3": "hifi2fp16",
                "li_ff2": "hifi2fp16",
                "li_qkv_decode": "hifi2",
                "sdpa_decode": "hifi2",
                "li_o_decode": "hifi2",
                "li_qkv_prefill": "hifi2",
                "sdpa_prefill": "hifi4",
                "li_o_prefill": "hifi2",
                "accuracy": "hifi4fp32",
            },
        )

    def get_tensor_dtype(self, decoder_id: int, tensor: str, prefetcher: bool = False):
        effective_decoder_id = 0 if prefetcher else decoder_id
        value = self._tensor_precision.get(effective_decoder_id, {}).get(tensor)
        if prefetcher and value is None and tensor != "activation":
            return ttnn.bfloat8_b
        return self._DTYPES.get(value)

    def get_math_fidelity(self, decoder_id: int, op: str, configuration):
        kernel_lookup = {
            "lofi": configuration.compute_kernel_config_lofi,
            "hifi2": configuration.compute_kernel_config_hifi2,
            "hifi2na": configuration.compute_kernel_config_hifi2_na,
            "hifi2fp16": configuration.compute_kernel_config_hifi2_fp16,
            "hifi2nol1acc": configuration.compute_kernel_config_hifi2_nol1acc,
            "hifi4": configuration.compute_kernel_config_hifi4,
            "hifi4fp32": configuration.compute_kernel_config_hifi4_fp32,
        }
        return kernel_lookup[self._op_fidelity[decoder_id][op]]

    def _update_full_name(self):
        self._full_name = " | ".join(
            f"Decoder {decoder_id}: precision_cfg = {self._tensor_precision[decoder_id]}, fidelity_cfg = {self._op_fidelity[decoder_id]}"
            for decoder_id in self._tensor_precision
        )


def _base_model_name(model_name: str) -> str:
    for suffix in ("-Instruct", "-instruct"):
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


# =============================================================================
# TransformerBlock1D
# =============================================================================


@dataclass
class TransformerBlock1DConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Attention1DConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtype: ttnn.DataType | None = None


class TransformerBlock1D(LightweightModule):
    """Single transformer block for 1D topologies (N150, N300, T3K).

    Happy path (takes pre-built sub-modules):
        block = TransformerBlock1D(attn_norm, attention, ff_norm, mlp)

    Power-user path (builds from config):
        block = TransformerBlock1D.from_config(config)
    """

    def __init__(
        self,
        attention_norm: RMSNorm1D,
        attention: Attention1D,
        ff_norm: RMSNorm1D,
        feed_forward: MLP1D,
        decode_residual_memcfg: ttnn.MemoryConfig | None = None,
        prefill_residual_memcfg: ttnn.MemoryConfig | None = None,
        activation_dtype: ttnn.DataType | None = None,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.ff_norm = ff_norm
        self.feed_forward = feed_forward
        self.decode_residual_memcfg = decode_residual_memcfg
        self.prefill_residual_memcfg = prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtype = activation_dtype

    @classmethod
    def from_config(cls, config: TransformerBlock1DConfig):
        return cls(
            attention_norm=RMSNorm1D.from_config(config.attention_norm_config),
            attention=Attention1D.from_config(config.attention_config),
            ff_norm=RMSNorm1D.from_config(config.ff_norm_config),
            feed_forward=MLP1D.from_config(config.mlp_config),
            decode_residual_memcfg=config.decode_residual_memcfg,
            prefill_residual_memcfg=config.prefill_residual_memcfg,
            activation_dtype=config.activation_dtype,
        )

    def decode_forward(self, x: ttnn.Tensor, current_pos, rot_mats, page_table) -> ttnn.Tensor:
        residual = x

        x = _all_gather_rmsnorm_tensor(
            self.attention_norm, x, memory_config=self.attention_norm.config.decode_memory_config
        )
        attn_in = self.attention_norm.decode_forward(x)
        attn_out = self.attention.decode_forward(attn_in, current_pos, rot_mats, page_table=page_table)
        attn_out = ttnn.to_memory_config(attn_out, self.decode_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.decode_residual_memcfg)
        residual = hidden_states

        hidden_states = _all_gather_rmsnorm_tensor(
            self.ff_norm, hidden_states, memory_config=self.ff_norm.config.decode_memory_config
        )
        hidden_states = self.ff_norm.decode_forward(hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.decode_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.decode_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        rot_mats,
        user_id,
        page_table,
        chunk_page_table,
        chunk_start_idx,
        batch_size: int = 1,
        chunk_start_idx_tensor=None,
    ) -> ttnn.Tensor:
        residual = x

        attn_in = self.attention_norm.prefill_forward(x)
        attn_in = _all_gather_rmsnorm_tensor(self.attention_norm, attn_in)
        if batch_size > 1:
            attn_in = ttnn.reshape(attn_in, [batch_size, 1, attn_in.shape[-2] // batch_size, -1])
        attn_out = self.attention.prefill_forward(
            attn_in,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
        )
        if batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])
        attn_out = ttnn.to_memory_config(attn_out, self.prefill_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.prefill_residual_memcfg)
        residual = hidden_states
        x.deallocate(True)

        hidden_states = self.ff_norm.prefill_forward(hidden_states)
        hidden_states = _all_gather_rmsnorm_tensor(self.ff_norm, hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.prefill_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.prefill_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        batch_size: int = 1,
        chunk_start_idx_tensor=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                user_id,
                page_table,
                chunk_page_table,
                chunk_start_idx,
                batch_size,
                chunk_start_idx_tensor,
            )
        return self.decode_forward(x, current_pos, rot_mats, page_table)


# =============================================================================
# Llama3Transformer1D
# =============================================================================


@dataclass
class Llama31_8BPagedAttentionConfig:
    block_size: int
    max_num_blocks: int


@dataclass
class Llama3Transformer1DConfig:
    """Full TTTv2 model config."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    dim: int
    num_devices: int
    mesh_device: ttnn.MeshDevice

    # Sub-module configs
    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    block_configs: list[TransformerBlock1DConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None

    # Model-level memory configs
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None

    # Per-layer activation dtypes (from decoders_optimizations)
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)

    # CCL
    tt_ccl: TT_CCL | None = None

    # Weight cache path
    cache_path: "str | None" = None


class Llama3Transformer1D(LightweightModule):
    """TTTv2 Llama 3.1-8B Transformer.

    Constructor takes a config and builds everything internally:
        model = Llama3Transformer1D(config)

    Public sub-modules (accessible by executor for trace support):
        - embedding: Embedding1D
        - rope_setup: RotarySetup1D
        - layers: list[TransformerBlock1D]
        - norm: RMSNorm1D (final)
        - lm_head: LMHead1D
        - sampling: Sampling1D | None

    Forward methods take pre-embedded tensors. The executor handles
    embedding, input preparation, and output processing.
    """

    def __init__(self, config: Llama3Transformer1DConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config

        tt_ccl_inst = config.tt_ccl
        if tt_ccl_inst is None and config.num_devices > 1:
            tt_ccl_inst = get_tt_ccl(config.mesh_device)

        self.embedding = Embedding1D.from_config(config.embedding_config)
        self.rope_setup = RotarySetup1D.from_config(config.rope_config)

        self.layers = [
            TransformerBlock1D.from_config(config.block_configs[i])
            for i in tqdm(range(config.n_layers), desc="Building layers")
        ]

        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)

        self.sampling = None
        if config.sampling_config is not None:
            self.sampling = Sampling1D.from_config(config.sampling_config)
        self.supports_on_device_sampling = self.sampling is not None

        self.mesh_device = config.mesh_device
        self.tt_ccl = tt_ccl_inst
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers

    # =========================================================================
    # KV Cache binding
    # =========================================================================

    def iter_executor_named_modules(self):
        """Yield named submodules that declare executor input contracts."""
        if not hasattr(self, "layers"):
            return

        for i, layer in enumerate(self.layers):
            for suffix, submodule in (
                ("attn_norm", getattr(layer, "attention_norm", None)),
                ("attention", getattr(layer, "attention", None)),
                ("ff_norm", getattr(layer, "ff_norm", None)),
                ("mlp", getattr(layer, "feed_forward", None)),
            ):
                if submodule is not None:
                    yield f"layer[{i}].{suffix}", submodule

        if hasattr(self, "norm"):
            yield "final_norm", self.norm
        if hasattr(self, "lm_head"):
            yield "lm_head", self.lm_head

    def set_kv_cache(self, kv_cache: list | None):
        """Bind or unbind the static KV-cache pool transactionally."""
        if kv_cache is None:
            for layer in self.layers:
                layer.attention.config.kv_cache = None
                if hasattr(layer.attention, "kv_cache"):
                    layer.attention.kv_cache = None
            return

        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers")

        cache_pairs = []
        for i, value in enumerate(kv_cache):
            try:
                cache_pair = tuple(value)
            except TypeError as error:
                raise TypeError(f"kv_cache layer {i} must provide an iterable K/V tensor pair") from error
            if len(cache_pair) != 2:
                raise ValueError(f"kv_cache layer {i} must contain exactly two K/V tensors")
            cache_pairs.append(cache_pair)

        for layer, cache_pair in zip(self.layers, cache_pairs):
            layer.attention.config.kv_cache = cache_pair
            if hasattr(layer.attention, "kv_cache"):
                layer.attention.kv_cache = cache_pair

    # =========================================================================
    # Forward methods — take pre-embedded tensors
    # =========================================================================

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Decode forward. x_embed is already embedded, unsqueezed, and in decode_residual_memcfg."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            x = ttnn.to_memory_config(x, self.decode_residual_memcfg, self.activation_dtypes[i])

            x = layer.decode_forward(x, current_pos, rot_mats, page_table)

        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        x = self.lm_head.forward(x)
        return x

    def prefill_forward(
        self,
        x_embed: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
        batch_size: int = 1,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Prefill forward. x_embed is already embedded and unsqueezed to 4D."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None and x.dtype != activation_dtype:
                old = x
                x = ttnn.typecast(x, activation_dtype)
                ttnn.deallocate(old)

            x = layer.prefill_forward(
                x,
                rot_mats,
                user_id,
                page_table,
                chunk_page_table,
                chunk_start_idx,
                batch_size,
                chunk_start_idx_tensor,
            )

        if last_token_index is not None and last_token_slice is None:
            raise ValueError("last_token_index is required with a runtime last_token_slice")
        if get_last_token == -1 and last_token_slice is None:
            return x

        old = x
        if last_token_slice is None:
            get_last_token_floor = (get_last_token // 32) * 32
            x = ttnn.slice(
                x,
                (0, 0, get_last_token_floor, 0),
                (1, 1, get_last_token_floor + 32, x.shape[-1]),
            )
        else:
            x = ttnn.slice(
                x,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(x.shape[2]) // 32,
            )
        ttnn.deallocate(old)

        if last_token_index is not None:
            if x.dtype != ttnn.bfloat16:
                old = x
                x = ttnn.typecast(x, ttnn.bfloat16)
                ttnn.deallocate(old)
            old = x
            x = ttnn.embedding(last_token_index, x, layout=ttnn.TILE_LAYOUT)
            x = ttnn.unsqueeze_to_4D(x)
            ttnn.deallocate(old)

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def post_process_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Convert traced prefill hidden states into logits for the last token block."""
        if last_token_slice is None:
            get_last_token_floor = (last_token_idx // 32) * 32
            x = ttnn.slice(
                hidden_states,
                (0, 0, get_last_token_floor, 0),
                (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
            )
        else:
            x = ttnn.slice(
                hidden_states,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(hidden_states.shape[2]) // 32,
            )

        if last_token_index is not None:
            if x.dtype != ttnn.bfloat16:
                old = x
                x = ttnn.typecast(x, ttnn.bfloat16)
                ttnn.deallocate(old)
            old = x
            x = ttnn.embedding(last_token_index, x, layout=ttnn.TILE_LAYOUT)
            x = ttnn.unsqueeze_to_4D(x)
            ttnn.deallocate(old)
        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def post_process_batched_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx_list: list[int],
        padded_batch: int,
        prefill_seq_len: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Convert batched prefill hidden states into one logits row per slot."""
        x = self.norm.prefill_forward(hidden_states)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        x_split = ttnn.split(x, prefill_seq_len, dim=2)
        if last_token_slice is None:
            selected = [
                x_user[:, :, last_token_idx : last_token_idx + 1, :]
                for x_user, last_token_idx in zip(x_split, last_token_idx_list)
            ]
        else:
            if last_token_index is None:
                raise ValueError("last_token_index is required with a runtime last_token_slice")
            selected = []
            for x_user in x_split[: len(last_token_idx_list)]:
                block = ttnn.slice(
                    x_user,
                    last_token_slice[0],
                    last_token_slice[1],
                    slice_dim=2,
                    num_devices=prefill_seq_len // 32,
                )
                row = ttnn.embedding(last_token_index, block, layout=ttnn.TILE_LAYOUT)
                row = ttnn.unsqueeze_to_4D(row)
                ttnn.deallocate(block)
                selected.append(row)
        x = ttnn.concat(selected, dim=2)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token: int = -1,
        batch_size: int = 1,
        chunk_start_idx_tensor=None,
        last_token_slice=None,
        last_token_index=None,
    ) -> ttnn.Tensor:
        """Dispatcher for backward compatibility. Llama 3.1-8B has no local rope."""
        rot_mats = rot_mats_global
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
                batch_size=batch_size,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                last_token_slice=last_token_slice,
                last_token_index=last_token_index,
            )
        return self.decode_forward(
            x,
            current_pos,
            rot_mats,
            page_table=page_table,
        )

    # =========================================================================
    # Embedding + output processing helpers (called by executor)
    # =========================================================================

    def prepare_prefill_rot_mats(self, position_indices: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Gather prefill RoPE rows from runtime device position indices."""
        self.rope_setup.load_device_weights()
        cos = None
        sin = None
        try:
            cos = ttnn.embedding(position_indices, self.rope_setup.cos_matrix, layout=ttnn.TILE_LAYOUT)
            sin = ttnn.embedding(position_indices, self.rope_setup.sin_matrix, layout=ttnn.TILE_LAYOUT)
            return ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)
        except BaseException:
            for tensor in (sin, cos):
                if tensor is not None:
                    try:
                        ttnn.deallocate(tensor)
                    except BaseException:
                        pass
            raise

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens and prepare for decode. Returns tensor in decode_residual_memcfg."""
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        x = ttnn.to_memory_config(x, self.decode_residual_memcfg)
        return x

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens for prefill. Returns tensor in DRAM interleaved."""
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return x

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather logits across devices and untilize for host argmax."""
        if self.num_devices > 1:
            logits = ttnn.experimental.all_gather_async(
                logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                memory_config=logits.memory_config(),
                topology=default_topology(self.mesh_device),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor):
        """Increment decode position counters on device."""
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)


# =============================================================================
# RMSNorm gather helpers
# =============================================================================


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=tt_ccl.get_num_links(),
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


def build_llama3_transformer_1d_config(
    *,
    mesh_device,
    instruct: bool,
    max_batch_size: int,
    max_seq_len: int,
    model_name: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_layers: int,
    head_dim: int,
    hidden_dim: int,
    vocab_size: int,
    norm_eps: float,
    padded_vocab_size: int,
    rope_cos,
    rope_sin,
    model_cache_path: str | Path,
    state_dict,
    optimizations="performance",
    weight_cache_path=None,
    dtype=None,
    paged_attention_config=None,
    pad_logits_to_power_of_2=False,
) -> Llama3Transformer1DConfig:
    """Build explicit TTTv2 module configs from Llama-3.1-8B construction data."""
    num_devices = mesh_device.get_num_devices()
    dram_grid_size = mesh_device.dram_grid_size()
    device_name = get_device_name(mesh_device)
    cluster_shape = list(mesh_device.shape)
    cluster_type = ttnn.cluster.get_cluster_type()
    is_galaxy_cluster = cluster_type in (
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    )
    if num_devices == 32:
        raise ValueError("Llama3Transformer1D only supports 1D mesh topologies.")

    use_paged_kv_cache = paged_attention_config is not None

    if optimizations is None:
        decoder_precision = Llama31DecoderPrecision.performance(n_layers, model_name)
    elif isinstance(optimizations, str):
        decoder_precision = Llama31DecoderPrecision.from_string(optimizations)(n_layers, model_name)
    else:
        decoder_precision = optimizations

    assert n_heads % cluster_shape[1] == 0
    assert n_kv_heads % cluster_shape[1] == 0

    tile_padded_batch_rows = ttnn.TILE_SIZE * int(math.ceil(max_batch_size / ttnn.TILE_SIZE))
    qkv_size = head_dim * (2 * n_kv_heads + n_heads)
    min_kv_prefill_shard_seqlen = (ttnn.TILE_SIZE * 8 * 8) / (n_kv_heads // cluster_shape[1])
    prefill_len_cutoff = 512 if ttnn.device.is_blackhole(mesh_device) else 1024
    if _base_model_name(model_name) == "Llama-3.1-8B" and device_name in ("N150",):
        prefill_len_cutoff = 512

    compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    compute_kernel_config_hifi4_fp32 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=False,
    )
    compute_kernel_config_hifi2_na = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    compute_kernel_config_hifi2_nol1acc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def ccl_topology():
        current_cluster_type = ttnn.cluster.get_cluster_type()
        if current_cluster_type in (
            ttnn.cluster.ClusterType.P300_X2,
            ttnn.cluster.ClusterType.P150_X4,
            ttnn.cluster.ClusterType.P150_X8,
        ):
            return ttnn.Topology.Ring
        if current_cluster_type in (
            ttnn.cluster.ClusterType.T3K,
            ttnn.cluster.ClusterType.GALAXY,
            ttnn.cluster.ClusterType.TG,
            ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        ):
            return ttnn.Topology.Ring if num_devices >= 8 else ttnn.Topology.Linear
        return ttnn.Topology.Linear if num_devices > 1 else None

    use_fused_all_gather_matmul = (
        num_devices == 8
        and not is_galaxy_cluster
        and (dim // ttnn.TILE_SIZE // num_devices) % num_devices == 0
        and num_devices > 1
        and ccl_topology() == ttnn.Topology.Ring
    )

    dram_weight_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )

    def find_grid(n):
        max_rows = 8 if ttnn.device.is_wormhole_b0(mesh_device) else 10
        max_cols = 8 if ttnn.device.is_wormhole_b0(mesh_device) else 12
        possible_cores = [k for k in range(1, max_rows * max_cols + 1) if n % k == 0]
        possible_cores.sort(key=lambda x: abs(x - 32))
        for cores in possible_cores:
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols
        raise AssertionError(f"Cannot find grid for {n} tiles")

    def find_grid_k_n(k, n):
        possible_cores = [c for c in range(1, 65) if k % c == 0 and n % c == 0]
        possible_cores.sort(reverse=True)
        for cores in possible_cores:
            for rows in range(1, 9):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= 8:
                        return rows, cols
        raise AssertionError(f"Cannot find grid for K={k}, N={n}")

    def dram_shard_core_grid_for_k(k):
        rows, cols = find_grid(k // ttnn.TILE_SIZE)
        return ttnn.CoreGrid(x=cols, y=rows)

    def dram_shard_core_grid_for_k_and_n(k, n):
        rows, cols = find_grid_k_n(k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_largest_divisor(n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1

    def create_dram_sharded_mem_config(k, n, dram_grid=None):
        dram_cores = dram_grid_size.x
        padded_size = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * (ttnn.TILE_SIZE * dram_cores)
        grid = dram_grid or dram_weight_grid
        shard_spec = ttnn.ShardSpec(grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def dram_matmul_config(m, k, n, num_cores=None, fused_activation=None):
        if num_cores is None:
            num_cores = dram_shard_core_grid_for_k_and_n(k, n).num_cores
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=find_largest_divisor(k // (ttnn.TILE_SIZE * num_cores)),
            per_core_M=math.ceil(m / ttnn.TILE_SIZE),
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
            fused_activation=fused_activation,
        )

    def create_sharded_norm_config(grid):
        block_w = dim // grid.num_cores // ttnn.TILE_SIZE
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=tile_padded_batch_rows // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )

    def decode_all_gather_matmul_program_config():
        if not use_fused_all_gather_matmul:
            return None
        do_core_grid_size = (8, 1)
        do_per_core_n = dim // num_devices // ttnn.TILE_SIZE // (do_core_grid_size[0] * do_core_grid_size[1])
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=do_core_grid_size,
            in0_block_w=dim // ttnn.TILE_SIZE // (do_core_grid_size[0] * do_core_grid_size[1]),
            out_subblock_h=1,
            out_subblock_w=get_out_subblock_w(do_per_core_n, out_subblock_h=1),
            per_core_M=tile_padded_batch_rows // ttnn.TILE_SIZE,
            per_core_N=do_per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def decode_all_gather_matmul_output_mem_config():
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                num_to_core_range_set(num_devices),
                [tile_padded_batch_rows, dim // num_devices],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def decode_residual_mem_config():
        residual_grid = dram_shard_core_grid_for_k(dim // num_devices)
        return ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // residual_grid.num_cores // num_devices),
            residual_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    lm_head_num_rows = 8
    lm_head_cores_per_row = 8
    while dim % (ttnn.TILE_SIZE * lm_head_num_rows * lm_head_cores_per_row) != 0:
        lm_head_num_rows -= 1
        if lm_head_num_rows == 0:
            lm_head_cores_per_row -= 1
            if lm_head_cores_per_row == 0:
                raise ValueError("Could not find a valid LM head core grid")
            lm_head_num_rows = 8
    lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=lm_head_cores_per_row)
    max_columns_per_device_lm_head = 668 * lm_head_core_grid.num_cores
    attn_input_grid = dram_shard_core_grid_for_k(dim)
    mlp_core_grid = dram_shard_core_grid_for_k_and_n(dim, hidden_dim // num_devices)
    mlp2_core_grid = dram_shard_core_grid_for_k_and_n(hidden_dim // num_devices, dim)

    def get_decode_norm_config(norm_type):
        if norm_type == "attn":
            grid = attn_input_grid
            mem = ttnn.create_sharded_memory_config(
                (tile_padded_batch_rows, dim // grid.num_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        elif norm_type == "ff":
            grid = mlp_core_grid
            mem = ttnn.create_sharded_memory_config(
                (tile_padded_batch_rows, dim // grid.num_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        elif norm_type == "lm_head":
            grid = lm_head_core_grid
            mem = ttnn.create_sharded_memory_config(
                (tile_padded_batch_rows, nearest_32(dim // grid.num_cores)),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            raise ValueError(f"Invalid norm_type: {norm_type}")
        return {
            "sharded_program_config": create_sharded_norm_config(grid),
            "sharded_output_config": mem,
            "output_mem_config": None,
        }

    def get_decode_mlp_ff1_3_prg_config():
        return dram_matmul_config(tile_padded_batch_rows, dim, hidden_dim // cluster_shape[1], mlp_core_grid.num_cores)

    def get_decode_mlp_ff2_prg_config():
        return dram_matmul_config(tile_padded_batch_rows, hidden_dim // cluster_shape[1], dim, mlp2_core_grid.num_cores)

    def get_decode_mlp_binary_mult_mem_config():
        return ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, hidden_dim // cluster_shape[1] // mlp2_core_grid.num_cores),
            mlp2_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def get_tensor_dtype(layer_num, tensor):
        return decoder_precision.get_tensor_dtype(layer_num, tensor)

    def get_math_fidelity(layer_num, op):
        kernel_lookup = {
            "lofi": compute_kernel_config_lofi,
            "hifi2": compute_kernel_config_hifi2,
            "hifi2na": compute_kernel_config_hifi2_na,
            "hifi2fp16": compute_kernel_config_hifi2_fp16,
            "hifi2nol1acc": compute_kernel_config_hifi2_nol1acc,
            "hifi4": compute_kernel_config_hifi4,
            "hifi4fp32": compute_kernel_config_hifi4_fp32,
        }
        return kernel_lookup[decoder_precision._op_fidelity[layer_num][op]]

    def get_state_dict_prefix(module_name, layer_num):
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {"MLP": "feed_forward", "Attention": "attention", "TransformerBlock": "", "": ""}
        return layer_prefix + module_map[module_name]

    def cache_path(dtype):
        cache_path_root = Path(model_cache_path)
        if instruct:
            return (
                cache_path_root
                / {
                    ttnn.bfloat16: "tensor_cache_instruct_bf16",
                    ttnn.bfloat8_b: "tensor_cache_instruct_bfp8",
                }[dtype]
            )
        return cache_path_root / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]

    model_config = {
        "SDPA_DECODE_PROGCFG": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        ),
        "CREATE_QKV_DECODE_SHARD": (
            ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, head_dim),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            if ttnn.device.is_blackhole(mesh_device)
            else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        ),
        "ATTN_OUTPUT_PROGCFG": dram_matmul_config(
            m=tile_padded_batch_rows,
            k=(n_heads * head_dim) // num_devices,
            n=dim,
            num_cores=n_heads // num_devices,
        ),
        "ATTN_ALL_GATHER_MATMUL_PROGCFG": decode_all_gather_matmul_program_config(),
        "ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG": decode_all_gather_matmul_output_mem_config(),
        "MLP_RS_CONFIG": {
            "chunks_per_sync": 10,
            "num_workers_per_link": 2,
            "rs_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        },
    }
    model_config["DECODE_RESIDUAL_MEMCFG"] = decode_residual_mem_config()

    tt_ccl_inst = get_tt_ccl(mesh_device) if num_devices > 1 else None
    weight_cache_path = Path(weight_cache_path) if weight_cache_path else None
    embedding_cache_path = cache_path(dtype or ttnn.bfloat8_b)

    def mesh_shard(dim: int) -> ttnn.MeshMapperConfig:
        return ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(dim)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        )

    def cache_path_for(base, *parts):
        if base is None:
            return None
        return Path(base).joinpath(*parts)

    def make_embedding_config() -> Embedding1DConfig:
        base_name = get_state_dict_prefix("", None) + "tok_embeddings.weight"
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_dir = cache_path_for(embedding_cache_path, "embedding")
        return Embedding1DConfig(
            weights=LazyWeight(
                source=torch_weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_dir, "tok_embeddings") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            weights_dtype=ttnn.bfloat16,
            weights_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )

    def make_rope_config() -> Rope1DConfig:
        return Rope1DConfig(
            cos_matrix=LazyWeight(source=rope_cos, device=mesh_device),
            sin_matrix=LazyWeight(source=rope_sin, device=mesh_device),
            max_batch_size=max_batch_size,
            head_dim=head_dim,
            device=mesh_device,
            use_qk_fused=True,
        )

    def norm_weight_name(layer_num: int | None, weight_key: str, state_dict_prefix: str | None = None) -> str:
        if state_dict_prefix:
            return f"{state_dict_prefix}{weight_key}.weight"
        if layer_num is None:
            return f"{weight_key}.weight"
        return f"layers.{layer_num}.{weight_key}.weight"

    def make_norm_config(
        *,
        layer_num: int | None,
        weight_key: str,
        state_dict_prefix: str | None = None,
        sharded_program_config=None,
        sharded_output_config=None,
    ) -> RMSNorm1DConfig:
        weight_name = norm_weight_name(layer_num, weight_key, state_dict_prefix)
        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )
        return RMSNorm1DConfig(
            weight=LazyWeight(
                source=torch_weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(weight_cache_path, weight_name) if weight_cache_path else None,
                mesh_mapper_config=(
                    ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementReplicate()],
                        mesh_shape_override=ttnn.MeshShape([num_devices]),
                    )
                    if num_devices > 1
                    else None
                ),
            ),
            eps=norm_eps,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            max_batch_size=max_batch_size,
            prefill_distributed=num_devices > 1 and dim >= 4096,
            decode_program_config=sharded_program_config,
            decode_memory_config=sharded_output_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )

    def make_attention_config(layer_num: int, transformation_mats: dict[str, ttnn.Tensor]) -> Attention1DConfig:
        layer_name = get_state_dict_prefix("Attention", layer_num)
        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        wqkv_dtype = get_tensor_dtype(layer_num, "wqkv")
        wo_dtype = get_tensor_dtype(layer_num, "wo")
        kv_cache_dtype = get_tensor_dtype(layer_num, "kv_cache")
        activation_dtype = get_tensor_dtype(layer_num, "activation")

        qkv_list = []
        for device_idx in range(num_devices):
            wq = torch.transpose(torch.chunk(state_dict[f"{wq_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            wk = torch.transpose(torch.chunk(state_dict[f"{wk_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            wv = torch.transpose(torch.chunk(state_dict[f"{wv_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        wqkv = LazyWeight(
            source=qkv_cat,
            dtype=wqkv_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=create_dram_sharded_mem_config(dim, qkv_size // num_devices),
            mesh_mapper_config=mesh_shard(-1),
            cache_dir_weight_name=(weight_cache_path / layer_name, "wqkv_sharded") if weight_cache_path else None,
        )
        wo = LazyWeight(
            source=state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0),
            dtype=wo_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_fused_all_gather_matmul
                else create_dram_sharded_mem_config((n_heads * head_dim) // num_devices, dim)
            ),
            mesh_mapper_config=mesh_shard(-1 if use_fused_all_gather_matmul else -2),
            cache_dir_weight_name=(
                (weight_cache_path / layer_name, "wo_width_sharded" if use_fused_all_gather_matmul else "wo")
                if weight_cache_path
                else None
            ),
        )

        qk_norm_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def make_qk_norm_config(name: str) -> RMSNorm1DConfig | None:
            weight_name = f"{name}.weight"
            if weight_name not in state_dict:
                return None
            return RMSNorm1DConfig(
                weight=LazyWeight(
                    source=state_dict[weight_name].reshape(1, 1, -1, TILE_SIZE),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_dir_weight_name=(
                        (weight_cache_path / layer_name, name.rsplit(".", 1)[-1]) if weight_cache_path else None
                    ),
                ),
                mesh_device=mesh_device,
                eps=norm_eps,
                decode_in_sharded=False,
                decode_out_sharded=False,
                prefill_distributed=False,
                compute_kernel_config=qk_norm_compute_kernel,
            )

        wqkv_bias = None
        if f"{wq_str}.bias" in state_dict:
            wqkv_bias = LazyWeight(
                source=torch.concat(
                    [
                        torch.concat(
                            [
                                torch.chunk(state_dict[f"{wq_str}.bias"], num_devices)[device_idx],
                                torch.chunk(state_dict[f"{wk_str}.bias"], num_devices)[device_idx],
                                torch.chunk(state_dict[f"{wv_str}.bias"], num_devices)[device_idx],
                            ],
                            dim=-1,
                        )
                        for device_idx in range(num_devices)
                    ],
                    dim=-1,
                )
            )

        scale = head_dim**-0.5
        return Attention1DConfig(
            wqkv=wqkv,
            wo=wo,
            q_norm_config=make_qk_norm_config(q_norm_str),
            k_norm_config=make_qk_norm_config(k_norm_str),
            wqkv_bias=wqkv_bias,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            topology=ccl_topology(),
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qkv_size=qkv_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            scale=scale,
            use_qk_fused=True,
            use_vllm_paged_kv_cache=use_paged_kv_cache,
            paged_attention_config=paged_attention_config,
            kv_cache_dtype=kv_cache_dtype,
            min_kv_prefill_shard_seqlen=min_kv_prefill_shard_seqlen,
            wqkv_dtype=wqkv_dtype,
            wo_dtype=wo_dtype,
            activation_dtype=activation_dtype,
            decode_sdpa_prg_config=model_config.get("SDPA_DECODE_PROGCFG"),
            decode_attn_output_prg_config=model_config.get("ATTN_OUTPUT_PROGCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
            decode_create_qkv_head_memcfg=model_config.get("CREATE_QKV_DECODE_SHARD"),
            use_fused_all_gather_matmul=use_fused_all_gather_matmul,
            decode_all_gather_matmul_prg_config=model_config.get("ATTN_ALL_GATHER_MATMUL_PROGCFG"),
            decode_all_gather_matmul_memcfg=model_config.get("ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"),
            li_qkv_decode_compute_kernel_cfg=get_math_fidelity(layer_num, "li_qkv_decode"),
            sdpa_decode_compute_kernel_cfg=get_math_fidelity(layer_num, "sdpa_decode"),
            li_o_decode_compute_kernel_cfg=get_math_fidelity(layer_num, "li_o_decode"),
            li_qkv_prefill_compute_kernel_cfg=get_math_fidelity(layer_num, "li_qkv_prefill"),
            sdpa_prefill_compute_kernel_cfg=get_math_fidelity(layer_num, "sdpa_prefill"),
            li_o_prefill_compute_kernel_cfg=get_math_fidelity(layer_num, "li_o_prefill"),
            transformation_mat_decode=transformation_mats.get("decode"),
            transformation_mat_prefill=transformation_mats.get("prefill"),
        )

    def make_mlp_config(layer_num: int) -> MLP1DConfig:
        state_dict_prefix = get_state_dict_prefix("MLP", layer_num)
        ff1_3_dtype = get_tensor_dtype(layer_num, "ff1_ff3")
        ff2_dtype = get_tensor_dtype(layer_num, "ff2")
        activation_dtype = get_tensor_dtype(layer_num, "activation")
        mlp_rs_cfg = model_config.get("MLP_RS_CONFIG", {})

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        w1_w3_mem_config = _create_dram_sharded_mem_config(
            k=dim,
            n=hidden_dim // num_devices,
            dram_grid=dram_grid,
            tile_size=TILE_SIZE,
            dram_cores=dram_size.x,
        )
        w2_mem_config = _create_dram_sharded_mem_config(
            k=hidden_dim // num_devices,
            n=dim,
            dram_grid=dram_grid,
            tile_size=TILE_SIZE,
            dram_cores=dram_size.x,
        )
        cache_dir = cache_path_for(weight_cache_path, state_dict_prefix)

        def make_weight_source(name: str, shard_dim: int):
            tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
            return pad_dim_to_size(tensor, dim=shard_dim, size=hidden_dim)

        return MLP1DConfig(
            w1=LazyWeight(
                source=make_weight_source("w1", -1),
                dtype=ff1_3_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w1_w3_mem_config,
                cache_dir_weight_name=(cache_dir, "w1_sharded") if cache_dir else None,
            ),
            w2=LazyWeight(
                source=make_weight_source("w2", -2),
                dtype=ff2_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-2),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w2_mem_config,
                cache_dir_weight_name=(cache_dir, "w2_sharded") if cache_dir else None,
            ),
            w3=LazyWeight(
                source=make_weight_source("w3", -1),
                dtype=ff1_3_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w1_w3_mem_config,
                cache_dir_weight_name=(cache_dir, "w3_sharded") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            dim=dim,
            hidden_dim=hidden_dim,
            max_batch_size=max_batch_size,
            mlp_activation_type=ttnn.UnaryOpType.SILU,
            topology=ccl_topology(),
            decode_rs_memory_config=mlp_rs_cfg.get("rs_memory_config", ttnn.L1_MEMORY_CONFIG),
            decode_rs_chunks_per_sync=mlp_rs_cfg.get("chunks_per_sync", 1),
            decode_rs_num_workers_per_link=mlp_rs_cfg.get("num_workers_per_link", 1),
            decode_w1_w3_prg_config=get_decode_mlp_ff1_3_prg_config(),
            decode_w2_prg_config=get_decode_mlp_ff2_prg_config(),
            decode_mlp2_input_memcfg=get_decode_mlp_binary_mult_mem_config(),
            decode_residual_memcfg=decode_residual_mem_config(),
            w1_w3_dtype=ff1_3_dtype,
            w2_dtype=ff2_dtype,
            activation_dtype=activation_dtype,
            ff1_3_compute_kernel_cfg=get_math_fidelity(layer_num, "li_ff1_ff3"),
            ff2_compute_kernel_cfg=get_math_fidelity(layer_num, "li_ff2"),
            decode_ff1_3_compute_kernel_cfg=get_math_fidelity(layer_num, "li_ff1_ff3"),
            decode_ff2_compute_kernel_cfg=get_math_fidelity(layer_num, "li_ff2"),
            prefill_len_cutoff=prefill_len_cutoff,
        )

    def make_lm_head_config() -> LMHead1DConfig:
        lm_head_padded_vocab_size = math.ceil(vocab_size / TILE_SIZE) * TILE_SIZE
        size_per_device = lm_head_padded_vocab_size // num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device_lm_head)
        split_sizes = [min(size_per_device, max_columns_per_device_lm_head)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        state_dict_prefix = get_state_dict_prefix("", None)
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)
        if vocab_size < lm_head_padded_vocab_size:
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(
                        torch_output_weights.shape[0],
                        lm_head_padded_vocab_size - vocab_size,
                        dtype=torch_output_weights.dtype,
                    ),
                ],
                dim=-1,
            )

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        cache_dir = cache_path_for(weight_cache_path, "lm_head")
        output_weights = []
        weights_memcfgs = []
        for split_idx, split_size in enumerate(split_sizes):
            device_splits = []
            for device_idx in range(num_devices):
                start = device_idx * size_per_device + sum(split_sizes[:split_idx])
                end = start + split_size
                device_splits.append(torch_output_weights[:, start:end])
            combined_split = torch.cat(device_splits, dim=-1)
            mem_cfg = _create_dram_sharded_mem_config(
                k=dim,
                n=math.ceil(combined_split.shape[-1] / num_devices),
                dram_grid=dram_grid,
                tile_size=TILE_SIZE,
                dram_cores=dram_size.x,
            )
            weights_memcfgs.append(mem_cfg)
            output_weights.append(
                LazyWeight(
                    source=combined_split,
                    dtype=dtype if dtype is not None else ttnn.bfloat8_b,
                    device=mesh_device,
                    mesh_mapper_config=mesh_shard(-1),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem_cfg,
                    cache_dir_weight_name=(
                        (cache_dir, f"output_split_{split_idx}_{combined_split.shape[-1]}") if cache_dir else None
                    ),
                )
            )

        lm_head_tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
        input_memcfg = ttnn.create_sharded_memory_config(
            (
                lm_head_tile_padded_batch_rows,
                math.ceil((dim // lm_head_core_grid.num_cores) / TILE_SIZE) * TILE_SIZE,
            ),
            lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return LMHead1DConfig(
            output_weights=output_weights,
            mesh_device=mesh_device,
            dim=dim,
            max_batch_size=max_batch_size,
            program_configs=[
                dram_matmul_config(lm_head_tile_padded_batch_rows, dim, split_size, lm_head_core_grid.num_cores)
                for split_size in split_sizes
            ],
            compute_kernel_config=_compute_kernel_config_hifi2(),
            output_memcfg=ttnn.L1_MEMORY_CONFIG,
            input_memcfg=input_memcfg,
            weights_memcfgs=weights_memcfgs,
        )

    def make_sampling_config() -> Sampling1DConfig | None:
        sampling_splits = num_devices if list(mesh_device.shape) != [1, 1] else 2
        if vocab_size // sampling_splits > 64 * 1024:
            return None

        use_galaxy_force_argmax = is_galaxy_cluster and num_devices >= 8
        return Sampling1DConfig(
            vocab_size=padded_vocab_size,
            valid_vocab_size=vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            max_batch_size=max_batch_size,
            pad_to_power_of_2=pad_logits_to_power_of_2,
            # Decode uses force-argmax for greedy rows; prefill can still force
            # the top-k path at the executor call site when a platform needs it.
            allow_force_argmax=True,
            num_argmax_gather_links=4 if use_galaxy_force_argmax else 1,
            ag_topology=ttnn.Topology.Ring if use_galaxy_force_argmax else ttnn.Topology.Linear,
            argmax_num_workers_per_link=2,
        )

    rope_config = make_rope_config()
    trans_mats_dict = RotarySetup1D.from_config(rope_config).get_both_trans_mats()
    attn_norm_cfg = get_decode_norm_config("attn")
    ff_norm_cfg = get_decode_norm_config("ff")
    lm_head_norm_cfg = get_decode_norm_config("lm_head")
    activation_dtypes = [get_tensor_dtype(i, "activation") for i in range(n_layers)]

    return Llama3Transformer1DConfig(
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dim=dim,
        num_devices=num_devices,
        mesh_device=mesh_device,
        embedding_config=make_embedding_config(),
        rope_config=rope_config,
        block_configs=[
            TransformerBlock1DConfig(
                attention_norm_config=make_norm_config(
                    layer_num=i,
                    weight_key="attention_norm",
                    sharded_program_config=attn_norm_cfg.get("sharded_program_config"),
                    sharded_output_config=attn_norm_cfg.get("sharded_output_config"),
                ),
                attention_config=make_attention_config(i, trans_mats_dict),
                ff_norm_config=make_norm_config(
                    layer_num=i,
                    weight_key="ffn_norm",
                    sharded_program_config=ff_norm_cfg.get("sharded_program_config"),
                    sharded_output_config=ff_norm_cfg.get("sharded_output_config"),
                ),
                mlp_config=make_mlp_config(i),
                decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
                activation_dtype=activation_dtypes[i],
            )
            for i in range(n_layers)
        ],
        norm_config=make_norm_config(
            layer_num=None,
            weight_key="norm",
            state_dict_prefix=get_state_dict_prefix("", None),
            sharded_program_config=lm_head_norm_cfg.get("sharded_program_config"),
            sharded_output_config=lm_head_norm_cfg.get("sharded_output_config"),
        ),
        lm_head_config=make_lm_head_config(),
        sampling_config=make_sampling_config(),
        decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
        activation_dtypes=activation_dtypes,
        tt_ccl=tt_ccl_inst,
        cache_path=str(weight_cache_path) if weight_cache_path else None,
    )
