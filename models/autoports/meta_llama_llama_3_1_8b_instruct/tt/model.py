# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TTNN Llama model for the repo-local Llama 3.1 8B Instruct autoport."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch
import ttnn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    MODEL_ID,
    _require_llama31_8b_config,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TARGET_TOPOLOGY,
    MultiChipDecoder,
    MultiChipDecoderPolicy,
    _mesh_mapper_1d,
    _require_target_mesh,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    _compute_kernel_config_hifi2_fp16,
    _dram_matmul_config,
    _dram_sharded_weight_memcfg,
    _width_sharded_l1_memcfg,
)
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.tensor_utils import TILE_SIZE


FULL_MODEL_POLICY_NAME = "llama31_8b_full_t3k_1x8_tp8_split_sampling_v1"
MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_PRECISION_CONFIG_PATH = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"
PRECISION_CONFIG_ENV = "LLAMA31_8B_PRECISION_CONFIG"
_BASELINE_PRECISION_SENTINELS = {"baseline", "code-default", "none", "off", "disabled"}
_AUTO_PRECISION_SENTINELS = {"", "auto", "default", "selected"}
_DTYPE_BY_NAME = {
    "bfloat16": ttnn.bfloat16,
    "bf16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bf8": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "bf4": ttnn.bfloat4_b,
}
_MATH_FIDELITY_BY_NAME = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


@dataclass(frozen=True)
class AutoportFullModelConfig:
    hf_model_id: str = MODEL_ID
    max_batch_size: int = 1
    max_seq_len: int = 128 * 1024
    page_block_size: int = 64
    max_num_blocks: int | None = None
    num_layers: int | None = None
    cache_dir: Path | None = None
    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b
    precision_config_id: str = "code_default"
    precision_config_path: Path | None = None


def _dtype_from_name(value: str | None, *, default: ttnn.DataType) -> ttnn.DataType:
    if value is None:
        return default
    if value not in _DTYPE_BY_NAME:
        raise ValueError(f"Unsupported dtype {value!r}; expected one of {sorted(_DTYPE_BY_NAME)}")
    return _DTYPE_BY_NAME[value]


def _math_fidelity_from_name(value: str | None, *, default: ttnn.MathFidelity) -> ttnn.MathFidelity:
    if value is None:
        return default
    if value not in _MATH_FIDELITY_BY_NAME:
        raise ValueError(
            f"Unsupported math fidelity {value!r}; expected one of {sorted(_MATH_FIDELITY_BY_NAME)}"
        )
    return _MATH_FIDELITY_BY_NAME[value]


def _resolve_precision_config_path(precision_config_path: str | Path | None) -> Path | None:
    requested = os.environ.get(PRECISION_CONFIG_ENV) if precision_config_path is None else precision_config_path
    if requested is None:
        return DEFAULT_SELECTED_PRECISION_CONFIG_PATH if DEFAULT_SELECTED_PRECISION_CONFIG_PATH.exists() else None

    requested_str = str(requested)
    requested_key = requested_str.strip().lower()
    if requested_key in _BASELINE_PRECISION_SENTINELS:
        return None
    if requested_key in _AUTO_PRECISION_SENTINELS:
        return DEFAULT_SELECTED_PRECISION_CONFIG_PATH if DEFAULT_SELECTED_PRECISION_CONFIG_PATH.exists() else None

    path = Path(requested_str)
    if not path.is_absolute():
        path = MODEL_DIR / path
    return path


def _load_precision_config(precision_config_path: str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    path = _resolve_precision_config_path(precision_config_path)
    if path is None:
        return None, None
    if not path.exists():
        raise FileNotFoundError(f"Precision config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config, path


def _weight_group_dtype(
    config: dict[str, Any],
    group_name: str,
    *,
    default: ttnn.DataType,
    override: dict[str, Any] | None = None,
) -> ttnn.DataType:
    value = None
    for source in (override or {}, config):
        weight_groups = source.get("weight_groups", {})
        group = weight_groups.get(group_name)
        if isinstance(group, dict) and "dtype" in group:
            value = group["dtype"]
            break
    return _dtype_from_name(value, default=default)


def _runtime_dtype(
    config: dict[str, Any],
    key: str,
    *,
    default: ttnn.DataType,
    override: dict[str, Any] | None = None,
) -> ttnn.DataType:
    value = None
    for source in (override or {}, config):
        runtime = source.get("runtime", {})
        if key in runtime:
            value = runtime[key]
            break
        if key in source:
            value = source[key]
            break
    return _dtype_from_name(value, default=default)


def _compute_fidelity(
    config: dict[str, Any],
    key: str,
    *,
    default: ttnn.MathFidelity,
    override: dict[str, Any] | None = None,
) -> ttnn.MathFidelity:
    value = None
    for source in (override or {}, config):
        fidelities = source.get("compute_fidelities", {})
        if key in fidelities:
            value = fidelities[key]
            break
    return _math_fidelity_from_name(value, default=default)


def _layer_exception_for(config: dict[str, Any], layer_idx: int) -> dict[str, Any] | None:
    exceptions = config.get("layer_exceptions", [])
    if isinstance(exceptions, dict):
        override = exceptions.get(str(layer_idx))
        return override if isinstance(override, dict) else None
    if not isinstance(exceptions, list):
        return None
    for item in exceptions:
        if not isinstance(item, dict):
            continue
        layers = item.get("layers", [])
        if layers == "all":
            return item
        if isinstance(layers, int) and layers == layer_idx:
            return item
        if isinstance(layers, list) and layer_idx in {int(layer) for layer in layers}:
            return item
        if isinstance(layers, dict):
            start = int(layers.get("start", layer_idx + 1))
            end = int(layers.get("end", layer_idx - 1))
            if start <= layer_idx <= end:
                return item
    return None


def _layer_precision_policy(config: dict[str, Any] | None, layer_idx: int) -> MultiChipDecoderPolicy | None:
    if config is None:
        return None

    base = MultiChipDecoderPolicy()
    override = _layer_exception_for(config, layer_idx)
    attention_dtype = _weight_group_dtype(config, "attention", default=base.attention_weight_dtype, override=override)
    wqkv_dtype = _weight_group_dtype(config, "attention.wqkv", default=attention_dtype, override=override)
    wo_dtype = _weight_group_dtype(config, "attention.wo", default=attention_dtype, override=override)
    if wqkv_dtype != wo_dtype:
        raise ValueError("This full-model path requires attention.wqkv and attention.wo to use the same dtype")

    mlp_dtype = _weight_group_dtype(config, "mlp", default=base.mlp_gate_up_dtype, override=override)
    gate_dtype = _weight_group_dtype(config, "mlp.gate", default=mlp_dtype, override=override)
    up_dtype = _weight_group_dtype(config, "mlp.up", default=mlp_dtype, override=override)
    if gate_dtype != up_dtype:
        raise ValueError("This full-model path requires mlp.gate and mlp.up to use the same dtype")
    down_dtype = _weight_group_dtype(config, "mlp.down", default=mlp_dtype, override=override)

    config_id = str(config.get("config_id", "json_precision_config"))
    return MultiChipDecoderPolicy(
        name=config_id,
        activation_dtype=_runtime_dtype(config, "activation_dtype", default=base.activation_dtype, override=override),
        attention_weight_dtype=wqkv_dtype,
        mlp_gate_up_dtype=gate_dtype,
        mlp_down_dtype=down_dtype,
        kv_cache_dtype=_runtime_dtype(config, "kv_cache_dtype", default=base.kv_cache_dtype, override=override),
        mlp_mul_dtype=_runtime_dtype(config, "mlp_mul_dtype", default=base.mlp_mul_dtype, override=override),
        mlp_math_fidelity=_compute_fidelity(config, "mlp", default=base.mlp_math_fidelity, override=override),
    )


def _lm_head_dtype(config: dict[str, Any] | None) -> ttnn.DataType:
    if config is None:
        return ttnn.bfloat8_b
    return _weight_group_dtype(config, "lm_head", default=ttnn.bfloat8_b)


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(cos.shape) == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = torch.stack((cos[:, : cos.shape[1] // 2], cos[:, : cos.shape[1] // 2]), dim=-1).flatten(-2)
    sin = torch.stack((sin[:, : sin.shape[1] // 2], sin[:, : sin.shape[1] // 2]), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _build_rope_setup(mesh_device: ttnn.MeshDevice, hf_config: Any, max_seq_len: int, batch: int) -> RotarySetup1D:
    rotary = LlamaRotaryEmbedding(config=hf_config).eval()
    dummy = torch.zeros(1, 1, max_seq_len, hf_config.head_dim, dtype=torch.bfloat16)
    position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos_hf, sin_hf = rotary(dummy, position_ids)
    cos_meta, sin_meta = _permute_to_meta_format(cos_hf.float(), sin_hf.float())
    cos_lw = LazyWeight(
        source=cos_meta.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_lw = LazyWeight(
        source=sin_meta.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return RotarySetup1D.from_config(
        Rope1DConfig(
            cos_matrix=cos_lw,
            sin_matrix=sin_lw,
            max_batch_size=batch,
            head_dim=hf_config.head_dim,
            device=mesh_device,
            use_qk_fused=False,
            datatype=ttnn.bfloat16,
        )
    )


def _embedding_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    mesh_device: ttnn.MeshDevice,
    cache_dir: Path | None,
) -> Embedding1D:
    weight = state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0)
    return Embedding1D.from_config(
        Embedding1DConfig(
            weights=LazyWeight(
                source=weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_dir, "model_embed_tokens") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            weights_dtype=ttnn.bfloat16,
            weights_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )
    )


def _final_norm_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    hf_config: Any,
    mesh_device: ttnn.MeshDevice,
    max_batch_size: int,
    cache_dir: Path | None,
) -> RMSNorm1D:
    source = state_dict["model.norm.weight"].reshape(1, 1, hf_config.hidden_size // TILE_SIZE, TILE_SIZE)
    return RMSNorm1D.from_config(
        RMSNorm1DConfig(
            weight=LazyWeight(
                source=source,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_dir, "model_norm") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            eps=hf_config.rms_norm_eps,
            max_batch_size=max_batch_size,
            decode_in_sharded=True,
            decode_out_sharded=True,
            prefill_distributed=False,
        )
    )


def _lm_head_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    hf_config: Any,
    mesh_device: ttnn.MeshDevice,
    max_batch_size: int,
    dtype: ttnn.DataType,
    cache_dir: Path | None,
) -> tuple[LMHead1D, int, int]:
    vocab_size = int(hf_config.vocab_size)
    padded_vocab_size = math.ceil(vocab_size / TILE_SIZE) * TILE_SIZE
    num_devices = mesh_device.get_num_devices()
    if padded_vocab_size % num_devices != 0:
        padded_vocab_size = math.ceil(padded_vocab_size / (TILE_SIZE * num_devices)) * TILE_SIZE * num_devices
    size_per_device = padded_vocab_size // num_devices

    lm_weight = state_dict.get("lm_head.weight")
    if lm_weight is None:
        if not getattr(hf_config, "tie_word_embeddings", False):
            raise KeyError("Missing lm_head.weight and HF config does not tie word embeddings")
        lm_weight = state_dict["model.embed_tokens.weight"]

    torch_output_weights = lm_weight.transpose(0, 1).contiguous()
    if torch_output_weights.shape[1] < padded_vocab_size:
        torch_output_weights = torch.nn.functional.pad(
            torch_output_weights,
            (0, padded_vocab_size - torch_output_weights.shape[1]),
            mode="constant",
            value=0,
        )

    dim = int(hf_config.hidden_size)
    decode_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
    lm_head_grid = ttnn.CoreGrid(x=8, y=1)
    input_memcfg = _width_sharded_l1_memcfg(dim, lm_head_grid, rows=decode_rows)
    max_columns_per_device = 4096
    num_splits = math.ceil(size_per_device / max_columns_per_device)
    split_sizes = [max_columns_per_device] * (num_splits - 1)
    split_sizes.append(size_per_device - sum(split_sizes))

    output_weights = []
    weights_memcfgs = []
    program_configs = []
    split_start = 0
    for split_idx, split_size in enumerate(split_sizes):
        device_splits = []
        for dev in range(num_devices):
            start = dev * size_per_device + split_start
            end = start + split_size
            device_splits.append(torch_output_weights[:, start:end])
        combined_split = torch.cat(device_splits, dim=-1)
        weight_memcfg = _dram_sharded_weight_memcfg(dim, split_size, mesh_device)
        weights_memcfgs.append(weight_memcfg)
        program_configs.append(
            _dram_matmul_config(
                m=decode_rows,
                k=dim,
                n=split_size,
                num_cores=lm_head_grid.num_cores,
            )
        )
        output_weights.append(
            LazyWeight(
                source=combined_split,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper_config=_mesh_mapper_1d(num_devices, -1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memcfg,
                cache_dir_weight_name=(cache_dir, f"lm_head_vocab_{padded_vocab_size}_split_{split_idx}")
                if cache_dir
                else None,
            )
        )
        split_start += split_size

    lm_head = LMHead1D.from_config(
        LMHead1DConfig(
            output_weights=output_weights,
            mesh_device=mesh_device,
            dim=dim,
            max_batch_size=max_batch_size,
            program_configs=program_configs,
            compute_kernel_config=_compute_kernel_config_hifi2_fp16(),
            lm_head_dtype=dtype,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            input_memcfg=input_memcfg,
            weights_memcfgs=weights_memcfgs,
        )
    )
    return lm_head, vocab_size, padded_vocab_size


class Llama31_8B_InstructFullModel:
    """Full autoregressive model assembled from the optimized multichip decoder."""

    def __init__(
        self,
        *,
        hf_config: Any,
        mesh_device: ttnn.MeshDevice,
        embedding: Embedding1D,
        rope_setup: RotarySetup1D,
        layers: list[MultiChipDecoder],
        norm: RMSNorm1D,
        lm_head: LMHead1D,
        config: AutoportFullModelConfig,
        vocab_size: int,
        padded_vocab_size: int,
    ) -> None:
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.embedding = embedding
        self.rope_setup = rope_setup
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.full_model_config = config
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.n_layers = len(layers)
        self.num_devices = mesh_device.get_num_devices()
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.policy = layers[0].policy
        self.policy_name = config.precision_config_id
        self.decode_residual_memcfg = layers[0].decode_residual_memcfg
        self.prefill_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG
        self._embedding_gather_output_buffers: dict[str, ttnn.Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        model_id: str = MODEL_ID,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        num_layers: int | None = None,
        cache_dir: str | Path | None = None,
        precision_config_path: str | Path | None = None,
    ) -> "Llama31_8B_InstructFullModel":
        _require_target_mesh(mesh_device)
        hf_config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        hf_config._attn_implementation = "eager"
        _require_llama31_8b_config(hf_config)
        precision_config, resolved_precision_config_path = _load_precision_config(precision_config_path)

        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        if max_num_blocks is None:
            max_num_blocks = max(1, (max_batch_size * max_seq_len + page_block_size - 1) // page_block_size)
        num_layers = int(num_layers or hf_config.num_hidden_layers)
        if num_layers < 1 or num_layers > hf_config.num_hidden_layers:
            raise ValueError(f"num_layers must be in [1, {hf_config.num_hidden_layers}], got {num_layers}")

        root_cache_dir = Path(cache_dir) if cache_dir is not None else None
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        state_dict = {name: tensor.detach().cpu() for name, tensor in hf_model.state_dict().items()}
        del hf_model

        cfg = AutoportFullModelConfig(
            hf_model_id=model_id,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_block_size=page_block_size,
            max_num_blocks=max_num_blocks,
            num_layers=num_layers,
            cache_dir=root_cache_dir,
            lm_head_dtype=_lm_head_dtype(precision_config),
            precision_config_id=str(precision_config.get("config_id", "code_default")) if precision_config else "code_default",
            precision_config_path=resolved_precision_config_path,
        )
        embedding = _embedding_from_state_dict(state_dict, mesh_device=mesh_device, cache_dir=root_cache_dir)
        rope_setup = _build_rope_setup(mesh_device, hf_config, max_seq_len + 1, max_batch_size)
        layers = [
            MultiChipDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                page_block_size=page_block_size,
                max_num_blocks=max_num_blocks,
                policy=_layer_precision_policy(precision_config, layer_idx),
                cache_dir=(root_cache_dir / f"layers_{layer_idx}") if root_cache_dir else None,
            )
            for layer_idx in range(num_layers)
        ]
        norm = _final_norm_from_state_dict(
            state_dict,
            hf_config=hf_config,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            cache_dir=root_cache_dir,
        )
        lm_head, vocab_size, padded_vocab_size = _lm_head_from_state_dict(
            state_dict,
            hf_config=hf_config,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            dtype=cfg.lm_head_dtype,
            cache_dir=root_cache_dir,
        )
        return cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            embedding=embedding,
            rope_setup=rope_setup,
            layers=layers,
            norm=norm,
            lm_head=lm_head,
            config=cfg,
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size,
        )

    def make_sampling_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            vocab_size=self.vocab_size,
            padded_vocab_size=self.padded_vocab_size,
            max_batch_size=self.full_model_config.max_batch_size,
            max_top_k=32,
            cluster_shape=tuple(self.mesh_device.shape),
            sampling_all_gather_axis=1,
            num_devices=self.mesh_device.get_num_devices(),
            is_galaxy=False,
            sampling_dp=1,
            use_topk_logprobs=False,
            pad_logits_to_power_of_2=True,
            model_config={},
        )

    def owned_kv_cache(self) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        caches = []
        for layer in self.layers:
            layer.self_attn.load_device_weights()
            caches.append(layer.self_attn.kv_cache)
        return caches

    def _normalize_kv_cache(self, kv_cache: Any) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        if kv_cache is None:
            raise ValueError("kv_cache must not be None")
        if len(kv_cache) == 1 and isinstance(kv_cache[0], list) and len(kv_cache[0]) == self.n_layers:
            kv_cache = kv_cache[0]
        if len(kv_cache) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} KV-cache entries, got {len(kv_cache)}")
        normalized = []
        for layer_idx, entry in enumerate(kv_cache):
            if len(entry) != 2:
                raise ValueError(f"KV-cache entry {layer_idx} must be a (key, value) pair")
            normalized.append((entry[0], entry[1]))
        return normalized

    @contextmanager
    def bind_kv_cache(self, kv_cache: Any | None) -> Iterator[None]:
        """Temporarily bind caller-owned attention KV-cache tensors.

        Standalone readiness uses the model-owned cache. vLLM serving allocates
        the cache and passes it through every low-level call; this context keeps
        that ownership explicit without changing decoder layer logic.
        """
        if kv_cache is None:
            yield
            return

        normalized = self._normalize_kv_cache(kv_cache)
        previous = []
        for layer, cache_pair in zip(self.layers, normalized):
            layer.self_attn.load_device_weights()
            previous.append(layer.self_attn.kv_cache)
            layer.self_attn.kv_cache = cache_pair
        try:
            yield
        finally:
            for layer, cache_pair in zip(self.layers, previous):
                layer.self_attn.kv_cache = cache_pair

    def _all_gather_embedding(self, hidden_shard: ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        memory_config = self.decode_residual_memcfg if mode == "decode" else self.prefill_residual_memcfg
        persistent_output_buffer = self._embedding_gather_output_buffers.get(mode) if mode == "decode" else None
        if mode == "decode" and persistent_output_buffer is None:
            output_shape = list(hidden_shard.shape)
            output_shape[-1] *= self.num_devices
            persistent_output_buffer = ttnn.allocate_tensor_on_device(
                output_shape,
                hidden_shard.dtype,
                hidden_shard.layout,
                self.mesh_device,
                memory_config,
            )
            self._embedding_gather_output_buffers[mode] = persistent_output_buffer
        gathered = ttnn.experimental.all_gather_async(
            hidden_shard,
            persistent_output_buffer=persistent_output_buffer,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.tt_ccl.get_num_links(),
            topology=TARGET_TOPOLOGY,
            memory_config=memory_config,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        )
        hidden_shard.deallocate(True)
        return gathered

    def release_decode_persistent_buffers(self) -> None:
        tensor = self._embedding_gather_output_buffers.pop("decode", None)
        if tensor is not None and tensor.is_allocated():
            tensor.deallocate(True)
        for layer in self.layers:
            release_buffers = getattr(layer, "release_decode_persistent_buffers", None)
            if release_buffers is not None:
                release_buffers()

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return self._all_gather_embedding(x, mode="prefill")

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        x = self._all_gather_embedding(x, mode="decode")
        return ttnn.to_memory_config(x, self.decode_residual_memcfg)

    def _lm_head_logits(self, hidden_states: ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        if mode == "decode":
            hidden_states = self.norm.decode_forward(hidden_states)
            hidden_states = ttnn.to_memory_config(hidden_states, self.lm_head.config.input_memcfg)
        elif mode == "prefill":
            hidden_states = self.norm.prefill_forward(hidden_states)
            hidden_states = ttnn.to_memory_config(hidden_states, self.lm_head.config.input_memcfg)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        return self.lm_head.forward(hidden_states)

    def prefill_forward(
        self,
        tokens: ttnn.Tensor,
        *,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
        kv_cache: Any | None = None,
        return_all_logits: bool = False,
        last_token_idx: int | None = None,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        with self.bind_kv_cache(kv_cache):
            x = self.embed_prefill(tokens)
            for layer in self.layers:
                x = layer.prefill_forward(
                    x,
                    rot_mats=rot_mats,
                    page_table=page_table,
                    user_id=0,
                )

            if return_all_logits:
                logits = []
                for tile_start in range(0, x.shape[-2], TILE_SIZE):
                    tile = ttnn.slice(x, (0, 0, tile_start, 0), (1, 1, tile_start + TILE_SIZE, x.shape[-1]))
                    logits.append(self._lm_head_logits(tile, mode="prefill"))
                x.deallocate(True)
                return logits

            if last_token_idx is None:
                last_token_idx = x.shape[-2] - 1
            tile_start = (last_token_idx // TILE_SIZE) * TILE_SIZE
            tile = ttnn.slice(x, (0, 0, tile_start, 0), (1, 1, tile_start + TILE_SIZE, x.shape[-1]))
            x.deallocate(True)
            return self._lm_head_logits(tile, mode="prefill")

    def decode_forward_from_ttnn_inputs(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_idxs: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        increment_positions: bool,
        kv_cache: Any | None = None,
    ) -> tuple[ttnn.Tensor, None]:
        with self.bind_kv_cache(kv_cache):
            rot_mats = tuple(self.rope_setup.decode_forward(rot_idxs))
            x = self.embed_decode(tokens)
            for layer in self.layers:
                x = layer.decode_forward(
                    x,
                    current_pos=current_pos,
                    rot_mats=rot_mats,
                    page_table=page_table,
                )
            logits = self._lm_head_logits(x, mode="decode")
            if increment_positions:
                ttnn.plus_one(current_pos, skip_negative_entries=True)
                ttnn.plus_one(rot_idxs)
            return logits, None

    def decode_forward_from_prepared_rot_mats(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
        kv_cache: Any | None = None,
    ) -> tuple[ttnn.Tensor, None]:
        with self.bind_kv_cache(kv_cache):
            x = self.embed_decode(tokens)
            return self.decode_forward_from_hidden_states(
                x,
                current_pos=current_pos,
                rot_mats=rot_mats,
                page_table=page_table,
            )

    def decode_forward_from_hidden_states(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
        kv_cache: Any | None = None,
    ) -> tuple[ttnn.Tensor, None]:
        with self.bind_kv_cache(kv_cache):
            x = hidden_states
            for layer in self.layers:
                x = layer.decode_forward(
                    x,
                    current_pos=current_pos,
                    rot_mats=rot_mats,
                    page_table=page_table,
                )
            return self._lm_head_logits(x, mode="decode"), None


__all__ = [
    "AutoportFullModelConfig",
    "DEFAULT_SELECTED_PRECISION_CONFIG_PATH",
    "FULL_MODEL_POLICY_NAME",
    "Llama31_8B_InstructFullModel",
    "MODEL_ID",
    "PRECISION_CONFIG_ENV",
    "TARGET_MESH_SHAPE",
    "TARGET_TOPOLOGY",
]
