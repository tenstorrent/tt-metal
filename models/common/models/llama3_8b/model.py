# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.1-8B Transformer model.

Pure model — forward methods only. No input/output processing, no trace
management, no vLLM adaptation. Those belong in executor.py / generator.py.

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

Dependencies:
    - ttnn
    - models.common.modules.* (TTTv2 modules)
    - models.common.lightweightmodule (base class)
    No torch. No TTTv1 imports.
"""

from dataclasses import dataclass, field

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl

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

        attn_in = self.attention_norm.decode_forward(x)
        attn_out = self.attention.decode_forward(attn_in, current_pos, rot_mats, page_table=page_table)
        old = attn_out
        attn_out = ttnn.to_memory_config(attn_out, self.decode_residual_memcfg)
        if old is not attn_out:
            ttnn.deallocate(old)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.decode_residual_memcfg)
        residual = hidden_states

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
        self, x: ttnn.Tensor, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx
    ) -> ttnn.Tensor:
        residual = x

        attn_in = self.attention_norm.prefill_forward(x)
        attn_out = self.attention.prefill_forward(
            attn_in,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        old = attn_out
        attn_out = ttnn.to_memory_config(attn_out, self.prefill_residual_memcfg)
        if old is not attn_out:
            ttnn.deallocate(old)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.prefill_residual_memcfg)
        residual = hidden_states
        x.deallocate(True)

        hidden_states = self.ff_norm.prefill_forward(hidden_states)
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
        kv_cache=None,
    ):
        if kv_cache is not None:
            self.attention.kv_cache = tuple(kv_cache)
        if mode == "prefill":
            return self.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)
        return self.decode_forward(x, current_pos, rot_mats, page_table)


# =============================================================================
# Llama3Transformer1D
# =============================================================================


@dataclass
class Llama3Transformer1DConfig:
    """Full model config. Build via from_hf_config() or construct manually."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
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

    # Weight cache path (for from_hf_config)
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

        self.mesh_device = config.mesh_device
        self.tt_ccl = tt_ccl_inst
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers

    # =========================================================================
    # Forward methods — take pre-embedded tensors
    # =========================================================================

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
        kv_cache: list | None = None,
    ) -> ttnn.Tensor:
        """Decode forward. x_embed is already embedded, unsqueezed, and in decode_residual_memcfg."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            if kv_cache is not None:
                layer.attention.kv_cache = tuple(kv_cache[i])

            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None:
                old = x
                x = ttnn.to_memory_config(x, self.decode_residual_memcfg, activation_dtype)
                if old is not x:
                    ttnn.deallocate(old)

            x = layer.decode_forward(x, current_pos, rot_mats, page_table)

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
        kv_cache: list | None = None,
    ) -> ttnn.Tensor:
        """Prefill forward. x_embed is already embedded and unsqueezed to 4D."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            if kv_cache is not None:
                layer.attention.kv_cache = tuple(kv_cache[i])

            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None and x.dtype != activation_dtype:
                old = x
                x = ttnn.typecast(x, activation_dtype)
                ttnn.deallocate(old)

            x = layer.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)

        if get_last_token == -1:
            return x

        get_last_token_floor = (get_last_token // 32) * 32
        old = x
        x = ttnn.slice(x, (0, 0, get_last_token_floor, 0), (1, 1, get_last_token_floor + 32, x.shape[-1]))
        ttnn.deallocate(old)

        x = self.norm.prefill_forward(x)
        x = self.lm_head.forward(x)
        old = x
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if old is not x:
            ttnn.deallocate(old)
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
        kv_cache=None,
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
                kv_cache=kv_cache,
            )
        return self.decode_forward(
            x,
            current_pos,
            rot_mats,
            page_table=page_table,
            kv_cache=kv_cache,
        )

    # =========================================================================
    # Embedding + output processing helpers (called by executor)
    # =========================================================================

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
                topology=self.tt_ccl.topology,
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

    # =========================================================================
    # Factory: from_model_args (backward-compat bridge to TTTv1 ModelArgs)
    # =========================================================================

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        dtype=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        """Build Llama3Transformer1D from TTTv1 ModelArgs.

        This is the only place that imports from models.tt_transformers.
        Used by the generator and demo for backward compatibility.
        Bypasses __init__ to build from pre-constructed sub-modules.
        """
        from tqdm import tqdm

        from models.tt_transformers.tt.model_config import TensorGroup

        instance = object.__new__(cls)
        super(Llama3Transformer1D, instance).__init__()

        tt_ccl_inst = get_tt_ccl(mesh_device) if mesh_device.get_num_devices() > 1 else None
        model_config = args.get_model_config()

        instance.config = None
        instance.mesh_device = mesh_device
        instance.tt_ccl = tt_ccl_inst
        instance.vocab_size = args.vocab_size
        instance.n_layers = args.n_layers
        instance.num_devices = mesh_device.get_num_devices()
        instance.decode_residual_memcfg = model_config.get("DECODE_RESIDUAL_MEMCFG")
        instance.prefill_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG
        instance.activation_dtypes = [
            args.decoders_optimizations.get_tensor_dtype(decoder_id=i, tensor=TensorGroup.ACTIVATION)
            for i in range(args.n_layers)
        ]

        instance.embedding = Embedding1D.from_model_args(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype or ttnn.bfloat8_b),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        instance.rope_setup = RotarySetup1D.from_model_args(
            device=mesh_device,
            args=args,
            model_name=args.model_name,
        )
        trans_mats_dict = instance.rope_setup.get_both_trans_mats()

        layers = []
        for i in tqdm(range(args.n_layers), desc="Building TTTv2 layers"):
            attn_norm = RMSNorm1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                weight_key="attention_norm",
            )
            attention = Attention1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
            )
            ff_norm = RMSNorm1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                weight_key="ffn_norm",
            )
            mlp = MLP1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                model_config=model_config,
            )

            block = TransformerBlock1D(
                attention_norm=attn_norm,
                attention=attention,
                ff_norm=ff_norm,
                feed_forward=mlp,
                decode_residual_memcfg=instance.decode_residual_memcfg,
                prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                activation_dtype=instance.activation_dtypes[i],
            )
            layers.append(block)
        instance.layers = layers

        instance.norm = RMSNorm1D.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=None,
            weight_key="norm",
            state_dict_prefix=args.get_state_dict_prefix("", None),
        )

        state_dict_prefix = args.get_state_dict_prefix("", None)
        instance.lm_head = LMHead1D.from_model_args(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            max_columns_per_device=args.max_columns_per_device_lm_head,
            dtype=dtype,
            model_config=model_config,
            tt_ccl=tt_ccl_inst,
        )

        instance.sampling = None
        sampling_splits = mesh_device.get_num_devices() if list(mesh_device.shape) != [1, 1] else 2
        if args.vocab_size // sampling_splits <= 64 * 1024:
            instance.sampling = Sampling1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                model_config=model_config,
            )

        return instance
