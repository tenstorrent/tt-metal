# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Gemma-3 text encoder for LTX-2.

Forward-only (no KV cache): runs all tokens through the decoder stack and returns
the hidden states. Follows the T5Encoder pattern in tt_dit/encoders/t5/model_t5.py.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate
from ...utils.tracing import StateTensor


class GemmaConfig:
    """Configuration for Gemma-3 text encoder."""

    def __init__(
        self,
        vocab_size: int = 262208,
        hidden_size: int = 3840,
        intermediate_size: int = 15360,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        rope_local_base_freq: float = 10000.0,
        rope_linear_scaling_factor: float = 8.0,
        sliding_window_pattern: int = 6,
        max_position_embeddings: int = 8192,
        hidden_layer_index: int = -1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        # Gemma-3 uses dual RoPE: global (full-attention) layers use rope_theta with
        # linear position scaling; local (sliding-window) layers use rope_local_base_freq
        # with no scaling. A layer is global when (layer_idx + 1) % sliding_window_pattern == 0.
        self.rope_theta = rope_theta
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_linear_scaling_factor = rope_linear_scaling_factor
        self.sliding_window_pattern = sliding_window_pattern
        self.max_position_embeddings = max_position_embeddings
        self.hidden_layer_index = hidden_layer_index


class GemmaRMSNorm(RMSNorm):
    """Gemma RMSNorm.

    Gemma-3 stores RMSNorm weights centered at 0 and scales by ``(1 + weight)``
    (HF ``Gemma3RMSNorm``). The underlying ``dit_rms_norm_unary_fused`` op applies
    the raw ``weight`` instead, so we fold the ``+1`` into the weight at load time.
    """

    def __init__(self, config: GemmaConfig, mesh_device, dim: int | None = None):
        super().__init__(
            embedding_dim=dim if dim is not None else config.hidden_size,
            norm_eps=config.rms_norm_eps,
            bias=False,
            mesh_device=mesh_device,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            # Gemma's (1 + weight) convention. Keep fp32 precision for the offset;
            # the Parameter is cast to its target dtype on load.
            state["weight"] = state["weight"].float() + 1.0
        super()._prepare_torch_state(state)


class GemmaRotaryEmbedding(Module):
    """Precompute RoPE cos/sin tables on host, store on device.

    ``base`` is the rope frequency base; ``linear_scaling_factor`` divides inv_freq
    (HF "linear" rope scaling, used by Gemma-3 global layers). Local layers pass
    base=rope_local_base_freq with linear_scaling_factor=1.0 (no scaling).
    """

    def __init__(self, config: GemmaConfig, mesh_device, base: float, linear_scaling_factor: float = 1.0):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = base
        self.max_seq_len = config.max_position_embeddings
        self.mesh_device = mesh_device

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim))
        inv_freq = inv_freq / linear_scaling_factor  # HF "linear" rope scaling
        t = torch.arange(min(self.max_seq_len, 8192), dtype=torch.float64)
        freqs = torch.outer(t, inv_freq).float()
        self._cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, D/2)
        self._sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)
        # seq_len is fixed per pipeline, so the device cos/sin are constant: bind once and
        # keep resident, mirroring the LTX transformer's StateTensor rope buffers.
        self._tt_cos = StateTensor()
        self._tt_sin = StateTensor()

    def forward(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Device cos/sin for the (fixed) sequence length, bound once and reused."""
        if self._tt_cos.value is None:
            # HF-format cos/sin for ttnn.experimental.rotary_embedding_hf: duplicate the half-dim
            # freqs to full head_dim as [c0..c_{d/2-1}, c0..c_{d/2-1}] (concat, not interleave).
            cos = torch.cat([self._cos_cached, self._cos_cached], dim=-1)[:, :, :seq_len, :].bfloat16()
            sin = torch.cat([self._sin_cached, self._sin_cached], dim=-1)[:, :, :seq_len, :].bfloat16()
            self._tt_cos.update(
                ttnn.from_torch(cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), False
            )
            self._tt_sin.update(
                ttnn.from_torch(sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), False
            )
        return self._tt_cos.value, self._tt_sin.value


class GemmaAttention(Module):
    """Gemma GQA self-attention with RoPE. No KV cache."""

    def __init__(
        self,
        config: GemmaConfig,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        tp = parallel_config.tensor_parallel.factor
        self.num_local_heads = self.num_heads // tp
        self.num_local_kv_heads = self.num_kv_heads // tp

        # FSDP: shard weights on the sequence-parallel axis (gathered per-op).
        sp = parallel_config.sequence_parallel
        fsdp_mesh_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None

        col_kwargs = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
        }
        if fsdp_mesh_axis is not None:
            col_kwargs["fsdp_mesh_axis"] = fsdp_mesh_axis
            col_kwargs["ccl_manager"] = ccl_manager

        # Fused QKV: one matmul instead of three, laid out per-TP-shard [q|k|v] heads so the
        # fused output splits cleanly with nlp_create_qkv_heads (GQA: num_kv_heads < num_heads).
        qkv_out = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.wqkv = ColParallelLinear(self.hidden_size, qkv_out, **col_kwargs)
        # o_proj carries the ccl_manager so its head-input all_gather can fuse into the matmul on
        # Ring (all_gather_minimal_matmul_async); wqkv's input is replicated so it never needs it.
        o_proj_kwargs = dict(col_kwargs)
        o_proj_kwargs.setdefault("ccl_manager", ccl_manager)
        self.o_proj = ColParallelLinear(self.num_heads * self.head_dim, self.hidden_size, **o_proj_kwargs)
        # The fused CCL-matmul ops are a ring-write scheme (4x8 FABRIC_1D_RING); on Linear (2x4)
        # they trip a worker-partitioning assert, so gate them on Ring and fall back to explicit CCL.
        self._tp_ring = ccl_manager.topology == ttnn.Topology.Ring

        self.input_layernorm = GemmaRMSNorm(config, mesh_device)
        # Gemma-3: post-attn norm applied to attn output before residual add
        self.post_attention_layernorm = GemmaRMSNorm(config, mesh_device)

        # Gemma-3 QK normalization (RMSNorm per head, shape [head_dim]).
        # Uses the same (1 + weight) convention as the other Gemma norms.
        self.q_norm = GemmaRMSNorm(config, mesh_device, dim=self.head_dim)
        self.k_norm = GemmaRMSNorm(config, mesh_device, dim=self.head_dim)

        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Fuse q/k/v into one wqkv weight. HF weights are [out, in]; transpose to [in, out],
        # group heads per TP shard, cat q|k|v on the head axis, flatten back to [out, in] so
        # ColParallelLinear column-fracturing hands each shard its local [q|k|v] heads.
        q = pop_substate(state, "q_proj")
        k = pop_substate(state, "k_proj")
        v = pop_substate(state, "v_proj")
        n_dev = self.parallel_config.tensor_parallel.factor
        d = self.head_dim

        def _heads(w: torch.Tensor, num: int) -> torch.Tensor:
            return w.T.reshape(self.hidden_size, n_dev, num // n_dev, d)  # [in, n_dev, local_heads, d]

        qkv = torch.cat(
            [
                _heads(q["weight"], self.num_heads),
                _heads(k["weight"], self.num_kv_heads),
                _heads(v["weight"], self.num_kv_heads),
            ],
            dim=2,
        )
        state["wqkv.weight"] = qkv.reshape(self.hidden_size, -1).T.contiguous()  # [(H+2KV)*d, in]
        # o_proj / q_norm / k_norm keep their keys and load as child modules directly.

    def forward(self, hidden_states, cos, sin, attn_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Fused QKV projection + fused head split.
        qkv = self.wqkv(hidden_states, compute_kernel_config=self.compute_config)
        qkv = ttnn.reshape(qkv, (qkv.shape[0], 1, qkv.shape[1], qkv.shape[2]))  # (B,1,seq,(H+2KV)*D)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # QK normalization (Gemma-3): RMSNorm per head before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Fused HF RoPE; cos/sin are HF-format [1,1,seq,head_dim] (GemmaRotaryEmbedding.forward).
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        # GQA: expand KV heads to match Q heads. Must use repeat_interleave (each kv
        # head duplicated contiguously: [kv0,kv0,kv1,kv1,...]) to match HF repeat_kv —
        # ttnn.repeat does block-tile ([kv0..kv7,kv0..kv7]) which mispairs q/kv heads.
        if self.num_local_kv_heads < self.num_local_heads:
            repeats = self.num_local_heads // self.num_local_kv_heads
            k = ttnn.repeat_interleave(k, repeats, dim=1)
            v = ttnn.repeat_interleave(v, repeats, dim=1)

        # Ensure DRAM interleaved layout for SDPA compatibility
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        # SDPA — use is_causal when no mask, or explicit mask when padding present.
        # TTNN SDPA doesn't support both is_causal and attn_mask simultaneously.
        # Gemma-3 scales by query_pre_attn_scalar**-0.5 = head_dim**-0.5 here; pass it
        # explicitly rather than relying on the SDPA default.
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(attn_mask is None),
            attn_mask=attn_mask,
            scale=1.0 / math.sqrt(self.head_dim),
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_config,
        )

        # Concat heads: (B, H, seq, D) → (B, seq, H*D)
        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        # Output projection. o_proj input is head-sharded; on Ring the gather-to-full-heads fuses
        # into the matmul (parallel_config triggers all_gather_minimal_matmul_async), on Linear it
        # is an explicit all_gather first.
        attn_output = ttnn.unsqueeze(attn_output, 0)
        tp_gt1 = self.parallel_config.tensor_parallel.factor > 1
        if tp_gt1 and self._tp_ring:
            output = self.o_proj(
                attn_output, compute_kernel_config=self.compute_config, parallel_config=self.parallel_config
            )
        else:
            if tp_gt1:
                attn_output = self.ccl_manager.all_gather(
                    attn_output,
                    dim=3,
                    mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                    use_hyperparams=True,
                )
            output = self.o_proj(attn_output, compute_kernel_config=self.compute_config)
        if tp_gt1:
            output = self.ccl_manager.all_gather(
                output,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        output = ttnn.squeeze(output, 0)

        # Gemma-3: post-attn norm before residual add
        output = self.post_attention_layernorm(output)
        return output + residual


class GemmaFF(Module):
    """Gemma-3 gated MLP: down_proj(gelu_tanh(gate_proj(x)) * up_proj(x)).

    Gemma-3's hidden_activation is gelu_pytorch_tanh (NOT silu)."""

    def __init__(
        self,
        config: GemmaConfig,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        # FSDP: shard weights on the sequence-parallel axis (gathered per-op).
        sp = parallel_config.sequence_parallel
        fsdp_mesh_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None

        col_kwargs = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
        }
        if fsdp_mesh_axis is not None:
            col_kwargs["fsdp_mesh_axis"] = fsdp_mesh_axis
            col_kwargs["ccl_manager"] = ccl_manager

        self.gate_proj = ColParallelLinear(
            config.hidden_size, config.intermediate_size, activation_fn="gelu_tanh", **col_kwargs
        )
        self.up_proj = ColParallelLinear(config.hidden_size, config.intermediate_size, **col_kwargs)
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Gemma-3: two norms sandwich the FFN
        self.pre_feedforward_layernorm = GemmaRMSNorm(config, mesh_device)
        self.post_feedforward_layernorm = GemmaRMSNorm(config, mesh_device)

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "mlp.gate_proj", "gate_proj")
        rename_substate(state, "mlp.up_proj", "up_proj")
        rename_substate(state, "mlp.down_proj", "down_proj")

    def forward(self, x):
        residual = x
        x = self.pre_feedforward_layernorm(x)

        gate = self.gate_proj(x, compute_kernel_config=self.compute_config)
        up = self.up_proj(x, compute_kernel_config=self.compute_config)
        x = gate * up

        x = self.down_proj(x, compute_kernel_config=self.compute_config)
        x = ttnn.unsqueeze(x, 0)
        if self.parallel_config.tensor_parallel.factor > 1:
            x = self.ccl_manager.all_gather(
                x,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        x = ttnn.squeeze(x, 0)

        # Gemma-3: post-FFN norm before residual add
        x = self.post_feedforward_layernorm(x)
        return x + residual


class GemmaEncoderLayer(Module):
    """Single Gemma decoder layer used as encoder (no KV cache)."""

    def __init__(self, config, mesh_device, ccl_manager, parallel_config):
        super().__init__()
        self.self_attn = GemmaAttention(config, mesh_device, ccl_manager, parallel_config)
        self.ff = GemmaFF(config, mesh_device, ccl_manager, parallel_config)

    def _prepare_torch_state(self, state):
        # HF Gemma-3 per-layer keys and their destinations:
        #   input_layernorm              → self_attn.input_layernorm  (pre-attn)
        #   post_attention_layernorm     → self_attn.post_attention_layernorm  (post-attn, before residual)
        #   pre_feedforward_layernorm    → ff.pre_feedforward_layernorm  (before FFN)
        #   post_feedforward_layernorm   → ff.post_feedforward_layernorm  (after FFN, before residual)
        #   mlp.*                        → ff.*
        rename_substate(state, "input_layernorm", "self_attn.input_layernorm")
        rename_substate(state, "post_attention_layernorm", "self_attn.post_attention_layernorm")
        rename_substate(state, "pre_feedforward_layernorm", "ff.pre_feedforward_layernorm")
        rename_substate(state, "post_feedforward_layernorm", "ff.post_feedforward_layernorm")
        rename_substate(state, "mlp.gate_proj", "ff.gate_proj")
        rename_substate(state, "mlp.up_proj", "ff.up_proj")
        rename_substate(state, "mlp.down_proj", "ff.down_proj")

    def forward(self, hidden_states, cos, sin, attn_mask=None):
        hidden_states = self.self_attn(hidden_states, cos, sin, attn_mask=attn_mask)
        hidden_states = self.ff(hidden_states)
        return hidden_states


class GemmaEncoder(Module):
    """
    TTNN Gemma-3 text encoder.

    Runs the full decoder stack and returns the list of all hidden states
    (input embedding, each decoder layer, then the final norm). Layer selection
    is the caller's responsibility.

    Usage:
        encoder = GemmaEncoder(config, mesh_device, ccl_manager, parallel_config)
        encoder.load_torch_state_dict(state_dict)
        hidden_states = encoder(token_ids, attention_mask)
    """

    def __init__(
        self,
        config: GemmaConfig,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, device=mesh_device)
        # Gemma-3 dual RoPE: global layers use rope_theta + linear scaling; local
        # (sliding-window) layers use rope_local_base_freq with no scaling.
        self.rotary_emb_global = GemmaRotaryEmbedding(
            config, mesh_device, base=config.rope_theta, linear_scaling_factor=config.rope_linear_scaling_factor
        )
        self.rotary_emb_local = GemmaRotaryEmbedding(
            config, mesh_device, base=config.rope_local_base_freq, linear_scaling_factor=1.0
        )

        self.layers = ModuleList(
            GemmaEncoderLayer(config, mesh_device, ccl_manager, parallel_config)
            for _ in range(config.num_hidden_layers)
        )

        self.norm = GemmaRMSNorm(config, mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF Gemma3ForConditionalGeneration key prefix: language_model.model.*
        # Strip prefix to get model.embed_tokens, model.layers.N.*, model.norm
        prefix = "language_model.model."
        stripped = {}
        for k, v in list(state.items()):
            if k.startswith(prefix):
                stripped[k[len(prefix) :]] = v
                del state[k]
        state.update(stripped)

        # Drop text-decoder-unused keys (lm_head, vision tower, multimodal projector).
        pop_substate(state, "lm_head")
        pop_substate(state, "language_model")
        for prefix_to_remove in ["vision_tower", "multi_modal_projector", "model.vision"]:
            pop_substate(state, prefix_to_remove)

    def build_attn_mask(self, attention_mask, seq_len: int) -> ttnn.Tensor | None:
        """Host-build the combined causal+padding SDPA mask (B,1,seq,seq) as a device tensor.
        Kept out of the traced forward: per-prompt host work whose output the trace copies in.
        TTNN SDPA can't take is_causal + attn_mask together, so causal and padding are merged."""
        if attention_mask is None:
            return None
        mask_host = (
            attention_mask
            if isinstance(attention_mask, torch.Tensor)
            else ttnn.to_torch(ttnn.get_device_tensors(attention_mask)[0])
        )
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)[None, None, :, :]
        pad_mask = torch.where(mask_host[:, None, None, :].bool(), 0.0, float("-inf"))
        combined = causal + pad_mask  # broadcasts to (B, 1, seq, seq)
        return ttnn.from_torch(
            combined.bfloat16(),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, token_ids: ttnn.Tensor, *, tt_attn_mask: ttnn.Tensor | None = None) -> list[ttnn.Tensor]:
        """Embed → 48 decoder layers → final norm. Returns [embed, L0..L47, final_norm] on device.
        Pure device graph (mask pre-built by build_attn_mask, RoPE cos/sin resident) so the whole
        forward replays as one ttnn trace."""
        hidden_states = self.embed_tokens(token_ids)
        hidden_states = ttnn.multiply(hidden_states, math.sqrt(self.config.hidden_size))

        seq_len = token_ids.shape[-1]
        cos_g, sin_g = self.rotary_emb_global(seq_len)
        cos_l, sin_l = self.rotary_emb_local(seq_len)

        # Gemma-3: a layer is global (full attention) when (idx + 1) % sliding_window_pattern == 0.
        all_hidden_states = [hidden_states]
        for idx, layer in enumerate(self.layers):
            is_global = (idx + 1) % self.config.sliding_window_pattern == 0
            cos, sin = (cos_g, sin_g) if is_global else (cos_l, sin_l)
            hidden_states = layer(hidden_states, cos, sin, attn_mask=tt_attn_mask)
            all_hidden_states.append(hidden_states)

        output = self.norm(hidden_states)
        all_hidden_states.append(output)
        return all_hidden_states
