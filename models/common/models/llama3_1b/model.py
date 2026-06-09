# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.2-1B Transformer model.

Pure model — forward methods only. No input/output processing, no trace
management, no vLLM adaptation. Those belong in executor.py / generator.py.

Architecture (mirrors HuggingFace LlamaForCausalLM structure):
    LlamaForCausalLM1D
    └── LlamaModel1D
        ├── Embedding1D             (embed_tokens)
        ├── RotarySetup1D           (rotary_emb)
        ├── LlamaDecoderLayer1D × n_layers
        │   ├── RMSNorm1D           (input_layernorm)
        │   ├── Attention1D         (self_attn)
        │   ├── RMSNorm1D           (post_attention_layernorm)
        │   └── MLP1D               (mlp)
        └── RMSNorm1D               (norm)
    └── LMHead1D                    (lm_head)
    └── Sampling1D (optional)

Llama 3.2-1B specifics:
    n_layers:           16
    hidden_size:        1024
    n_heads:            16
    n_kv_heads:         4  (GQA)
    intermediate_size:  8192
    vocab_size:         128,256
    max_position_embeddings: 128K

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
from models.common.modules.tt_ccl import TT_CCL, default_topology, get_tt_ccl

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


# =============================================================================
# LlamaDecoderLayer1D  (HF: LlamaDecoderLayer)
# =============================================================================


@dataclass
class LlamaDecoderLayer1DConfig:
    input_layernorm_config: RMSNorm1DConfig
    self_attn_config: Attention1DConfig
    post_attention_layernorm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtype: ttnn.DataType | None = None


class LlamaDecoderLayer1D(LightweightModule):
    """Single Llama decoder layer for 1D topologies (N150, N300, T3K).

    Mirrors HuggingFace LlamaDecoderLayer:
        hidden_states = input_layernorm(hidden_states)
        hidden_states, residual = self_attn(hidden_states) + residual
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states) + residual
    """

    def __init__(
        self,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: MLP1D,
        decode_residual_memcfg: ttnn.MemoryConfig | None = None,
        prefill_residual_memcfg: ttnn.MemoryConfig | None = None,
        activation_dtype: ttnn.DataType | None = None,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.decode_residual_memcfg = decode_residual_memcfg
        self.prefill_residual_memcfg = prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtype = activation_dtype

    @classmethod
    def from_config(cls, config: LlamaDecoderLayer1DConfig) -> "LlamaDecoderLayer1D":
        return cls(
            input_layernorm=RMSNorm1D.from_config(config.input_layernorm_config),
            self_attn=Attention1D.from_config(config.self_attn_config),
            post_attention_layernorm=RMSNorm1D.from_config(config.post_attention_layernorm_config),
            mlp=MLP1D.from_config(config.mlp_config),
            decode_residual_memcfg=config.decode_residual_memcfg,
            prefill_residual_memcfg=config.prefill_residual_memcfg,
            activation_dtype=config.activation_dtype,
        )

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        residual = hidden_states

        hidden_states = _all_gather_rmsnorm_tensor(
            self.input_layernorm,
            hidden_states,
            memory_config=self.input_layernorm.config.decode_memory_config,
        )
        hidden_states = self.input_layernorm.decode_forward(hidden_states)
        hidden_states = self.self_attn.decode_forward(hidden_states, current_pos, rot_mats, page_table=page_table)
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)

        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memcfg)
        residual = hidden_states

        hidden_states = _all_gather_rmsnorm_tensor(
            self.post_attention_layernorm,
            hidden_states,
            memory_config=self.post_attention_layernorm.config.decode_memory_config,
        )
        hidden_states = self.post_attention_layernorm.decode_forward(hidden_states)
        attn_out = hidden_states
        hidden_states = self.mlp.decode_forward(hidden_states)
        ttnn.deallocate(attn_out)

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
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        page_table: ttnn.Tensor | None,
        chunk_page_table: ttnn.Tensor | None,
        chunk_start_idx: int | None,
    ) -> ttnn.Tensor:
        residual = x

        hidden_states = self.input_layernorm.prefill_forward(x)
        hidden_states = _all_gather_rmsnorm_tensor(self.input_layernorm, hidden_states)
        hidden_states = self.self_attn.prefill_forward(
            hidden_states,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, self.prefill_residual_memcfg)

        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.prefill_residual_memcfg)
        residual = hidden_states
        x.deallocate(True)

        hidden_states = self.post_attention_layernorm.prefill_forward(residual)
        hidden_states = _all_gather_rmsnorm_tensor(self.post_attention_layernorm, hidden_states)
        attn_out = hidden_states
        hidden_states = self.mlp.prefill_forward(hidden_states)
        ttnn.deallocate(attn_out)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.prefill_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        current_pos: ttnn.Tensor | None = None,
        rot_mats: tuple | None = None,
        user_id: int = 0,
        mode: str = "decode",
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)
        return self.decode_forward(hidden_states, current_pos, rot_mats, page_table)


# =============================================================================
# LlamaModel1D  (HF: LlamaModel)
# =============================================================================


@dataclass
class LlamaModel1DConfig:
    """Config for the core transformer body (no LM head)."""

    n_layers: int
    mesh_device: ttnn.MeshDevice

    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    layer_configs: list[LlamaDecoderLayer1DConfig]
    norm_config: RMSNorm1DConfig

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)


class LlamaModel1D(LightweightModule):
    """Llama 3.2-1B transformer body (embedding + decoder layers + final norm).

    Mirrors HuggingFace LlamaModel. Does not include the LM head.

    Sub-modules (accessible by LlamaForCausalLM1D and executor):
        - embed_tokens: Embedding1D
        - rotary_emb:   RotarySetup1D
        - layers:       list[LlamaDecoderLayer1D]
        - norm:         RMSNorm1D (final)
    """

    def __init__(self, config: LlamaModel1DConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config

        self.embed_tokens = Embedding1D.from_config(config.embedding_config)
        self.rotary_emb = RotarySetup1D.from_config(config.rope_config)

        self.layers = [
            LlamaDecoderLayer1D.from_config(config.layer_configs[i])
            for i in tqdm(range(config.n_layers), desc="Building decoder layers")
        ]

        self.norm = RMSNorm1D.from_config(config.norm_config)

        self.mesh_device = config.mesh_device
        self.n_layers = config.n_layers
        self.decode_residual_memcfg = config.decode_residual_memcfg
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg, self.activation_dtypes[i])
            hidden_states = layer.decode_forward(hidden_states, current_pos, rot_mats, page_table)

        hidden_states = _all_gather_rmsnorm_tensor(
            self.norm, hidden_states, memory_config=self.norm.config.decode_memory_config
        )
        hidden_states = self.norm.decode_forward(hidden_states)
        return hidden_states

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        for i, layer in enumerate(self.layers):
            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None and hidden_states.dtype != activation_dtype:
                old = hidden_states
                hidden_states = ttnn.typecast(hidden_states, activation_dtype)
                ttnn.deallocate(old)

            hidden_states = layer.prefill_forward(
                hidden_states, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx
            )

        if get_last_token != -1:
            get_last_token_floor = (get_last_token // 32) * 32
            old = hidden_states
            hidden_states = ttnn.slice(
                hidden_states,
                (0, 0, get_last_token_floor, 0),
                (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
            )
            ttnn.deallocate(old)

        hidden_states = self.norm.prefill_forward(hidden_states)
        hidden_states = _all_gather_rmsnorm_tensor(self.norm, hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        current_pos: ttnn.Tensor | None = None,
        rot_mats: tuple | None = None,
        user_id: int = 0,
        mode: str = "decode",
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(
                hidden_states,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
            )
        return self.decode_forward(hidden_states, current_pos, rot_mats, page_table)


# =============================================================================
# LlamaForCausalLM1D  (HF: LlamaForCausalLM)
# =============================================================================


@dataclass
class LlamaForCausalLM1DConfig:
    """Full model config. Build via from_hf_config() or construct manually."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    num_devices: int
    mesh_device: ttnn.MeshDevice

    model_config: LlamaModel1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None

    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)

    tt_ccl: TT_CCL | None = None
    cache_path: "str | None" = None


class LlamaForCausalLM1D(LightweightModule):
    """TTTv2 Llama 3.2-1B causal language model.

    Mirrors HuggingFace LlamaForCausalLM:
        output = lm_head(model(input_ids))

    Constructor takes a config and builds everything internally:
        model = LlamaForCausalLM1D(config)

    Public sub-modules (accessible by executor for trace support):
        - model:    LlamaModel1D  (embed_tokens, rotary_emb, layers, norm)
        - lm_head:  LMHead1D
        - sampling: Sampling1D | None

    Forward methods take pre-embedded tensors. The executor handles
    embedding, input preparation, and output processing.
    """

    def __init__(self, config: LlamaForCausalLM1DConfig):
        super().__init__()
        self.config = config

        tt_ccl_inst = config.tt_ccl
        if tt_ccl_inst is None and config.num_devices > 1:
            tt_ccl_inst = get_tt_ccl(config.mesh_device)

        self.model = LlamaModel1D(config.model_config)
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

    # =========================================================================
    # KV cache binding
    # =========================================================================

    def set_kv_cache(self, kv_cache: list):
        """Bind static KV-cache pool via each attention layer's config.

        Must be called before the first forward (before load_device_weights runs).
        """
        assert len(kv_cache) == len(
            self.model.layers
        ), f"kv_cache has {len(kv_cache)} entries but model has {len(self.model.layers)} layers"
        for i, layer in enumerate(self.model.layers):
            layer.self_attn.config.kv_cache = tuple(kv_cache[i])

    # =========================================================================
    # Forward methods — take pre-embedded tensors
    # =========================================================================

    def decode_forward(
        self,
        inputs_embeds: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Decode forward. inputs_embeds is already embedded, unsqueezed, and in decode_residual_memcfg."""
        hidden_states = self.model.decode_forward(inputs_embeds, current_pos, rot_mats, page_table)
        logits = self.lm_head.forward(hidden_states)
        return logits

    def prefill_forward(
        self,
        inputs_embeds: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        """Prefill forward. inputs_embeds is already embedded and unsqueezed to 4D."""
        hidden_states = self.model.prefill_forward(
            inputs_embeds,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
        )

        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            hidden_states = ttnn.interleaved_to_sharded(hidden_states, lm_head_memcfg)

        logits = self.lm_head.forward(hidden_states)
        logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def forward(
        self,
        inputs_embeds: ttnn.Tensor,
        current_pos: ttnn.Tensor | None = None,
        rot_mats_global: tuple | None = None,
        rot_mats_local: tuple | None = None,
        user_id: int = 0,
        mode: str = "decode",
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        """Dispatcher. Llama 3.2-1B uses only global RoPE (no local rope)."""
        rot_mats = rot_mats_global
        if mode == "prefill":
            return self.prefill_forward(
                inputs_embeds,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
            )
        return self.decode_forward(inputs_embeds, current_pos, rot_mats, page_table)

    # =========================================================================
    # Embedding + output helpers (called by executor)
    # =========================================================================

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens and prepare for decode. Returns tensor in decode_residual_memcfg."""
        hidden_states = self.model.embed_tokens.forward(tokens)
        hidden_states = ttnn.unsqueeze_to_4D(hidden_states)
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        return hidden_states

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens for prefill. Returns tensor in DRAM interleaved."""
        hidden_states = self.model.embed_tokens.forward(tokens)
        hidden_states = ttnn.unsqueeze_to_4D(hidden_states)
        return hidden_states

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
        use_paged_kv_cache=None,
    ):
        """Build LlamaForCausalLM1D from TTTv1 ModelArgs.

        This is the only place that imports from models.tt_transformers.
        Used by the generator and demo for backward compatibility.
        Bypasses __init__ to build from pre-constructed sub-modules.
        """
        from tqdm import tqdm

        from models.tt_transformers.tt.common import Mode
        from models.tt_transformers.tt.model_config import TensorGroup

        instance = object.__new__(cls)
        super(LlamaForCausalLM1D, instance).__init__()

        if use_paged_kv_cache is None:
            use_paged_kv_cache = paged_attention_config is not None

        tt_ccl_inst = get_tt_ccl(mesh_device) if mesh_device.get_num_devices() > 1 else None
        model_config = args.get_model_config()
        model_config["DECODE_RESIDUAL_MEMCFG"] = args.get_residual_mem_config(Mode.DECODE)

        instance.config = None
        instance.mesh_device = mesh_device
        instance.tt_ccl = tt_ccl_inst
        instance.vocab_size = args.vocab_size
        instance.n_layers = args.n_layers
        instance.num_devices = mesh_device.get_num_devices()
        instance.decode_residual_memcfg = model_config["DECODE_RESIDUAL_MEMCFG"]
        instance.prefill_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

        # Build LlamaModel1D internals directly (bypassing LlamaModel1D.__init__)
        model_instance = object.__new__(LlamaModel1D)
        super(LlamaModel1D, model_instance).__init__()

        model_instance.config = None
        model_instance.mesh_device = mesh_device
        model_instance.n_layers = args.n_layers
        model_instance.decode_residual_memcfg = model_config["DECODE_RESIDUAL_MEMCFG"]
        model_instance.prefill_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG
        model_instance.activation_dtypes = [
            args.decoders_optimizations.get_tensor_dtype(decoder_id=i, tensor=TensorGroup.ACTIVATION)
            for i in range(args.n_layers)
        ]

        model_instance.embed_tokens = Embedding1D.from_model_args(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype or ttnn.bfloat8_b),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        model_instance.rotary_emb = RotarySetup1D.from_model_args(
            device=mesh_device,
            args=args,
            model_name=args.model_name,
        )
        trans_mats_dict = model_instance.rotary_emb.get_both_trans_mats()

        attn_norm_cfg = args.get_norm_config("attn", Mode.DECODE)
        ff_norm_cfg = args.get_norm_config("ff", Mode.DECODE)
        lm_head_norm_cfg = args.get_norm_config("lm_head", Mode.DECODE)

        layers = []
        for i in tqdm(range(args.n_layers), desc="Building TTTv2 layers"):
            input_layernorm = RMSNorm1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                weight_key="attention_norm",
                sharded_program_config=attn_norm_cfg.get("sharded_program_config"),
                sharded_output_config=attn_norm_cfg.get("sharded_output_config"),
            )
            self_attn = Attention1D.from_model_args(
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
            post_attention_layernorm = RMSNorm1D.from_model_args(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl_inst,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                weight_key="ffn_norm",
                sharded_program_config=ff_norm_cfg.get("sharded_program_config"),
                sharded_output_config=ff_norm_cfg.get("sharded_output_config"),
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

            layer = LlamaDecoderLayer1D(
                input_layernorm=input_layernorm,
                self_attn=self_attn,
                post_attention_layernorm=post_attention_layernorm,
                mlp=mlp,
                decode_residual_memcfg=instance.decode_residual_memcfg,
                prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                activation_dtype=model_instance.activation_dtypes[i],
            )
            layers.append(layer)

        model_instance.layers = layers

        model_instance.norm = RMSNorm1D.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=None,
            weight_key="norm",
            state_dict_prefix=args.get_state_dict_prefix("", None),
            sharded_program_config=lm_head_norm_cfg.get("sharded_program_config"),
            sharded_output_config=lm_head_norm_cfg.get("sharded_output_config"),
        )

        instance.model = model_instance

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
