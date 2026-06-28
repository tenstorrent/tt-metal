# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# VALIDATED on Blackhole (2x4 mesh, FABRIC_1D, Linear). The single block passes
# its PCC gate (>=0.99; measured ~0.9997) across {B=1,2} x {1,2 segments} x
# {1088, 4224 seq} x {tp1sp1, tp2, tp4, sp2, sp2tp2, sp2tp4} — see the test file.
#
# Notable design points (see the class docstring for the parallelism scheme):
#   1. rotate-half MRoPE convention (cos/sin = cat(freqs, freqs); _rotate_half),
#      precomputed on host from the reference Ideogram4MRoPE.
#   2. block-diagonal segment-id attention mask fed as an additive bias to SDPA;
#      under SP, K/V are all-gathered and the mask is sharded on query rows.
#   3. tanh-gated 4-branch AdaLN (scale_msa, gate_msa, scale_mlp, gate_mlp),
#      no shift terms — differs from the 6-branch ports.
#   4. the double-RMSNorm sandwich residual structure (norm1 pre-scale on input,
#      norm2 post-norm on the attn/ff output before the gate).
#   5. head_dim=256 / 18 heads — TP pads heads up to a multiple of tp_factor.
# =============================================================================

from __future__ import annotations

import torch

import ttnn

from ...layers.feedforward import ParallelFeedForward
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...utils.padding import pad_weight_tensor
from ...utils.substate import pop_substate

# Per-token role indicators (mirrors reference src/ideogram4/constants.py).
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

MATH_FIDELITY_HIFI2 = ttnn.MathFidelity.HiFi2
MATH_FIDELITY_HIFI4 = ttnn.MathFidelity.HiFi4


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    """Half-split rotate, matching reference _rotate_half: [-x2, x1].

    Reference (modeling_ideogram4.py): x1=x[..., :half], x2=x[..., half:],
    returns cat((-x2, x1)). NOT the interleaved alt_complex_rotate90 convention.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """q_embed = q * cos + rotate_half(q) * sin (reference _apply_rotary_pos_emb)."""
    return x * cos + _rotate_half(x) * sin


class Ideogram4TransformerBlock(Module):
    """Single-stream Ideogram 4.0 transformer block.

    Faithful port of reference `Ideogram4TransformerBlock`. Text and image tokens
    are a SINGLE unified sequence sharing one set of projections (no separate text
    branch, no separate modulation). Forward mirrors the reference exactly:

        mod = adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4)
        gate_* = tanh(gate_*); scale_* = 1 + scale_*

        attn_out = attention(attention_norm1(x) * scale_msa, ...)
        x = x + gate_msa * attention_norm2(attn_out)
        x = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))

    Parallelism (Megatron-style TP + ring sequence parallelism):
      * The residual stream stays REPLICATED on hidden and SHARDED on sequence
        ([B, L/sp, hidden]). Norms / AdaLN run replicated (regular RMSNorm/Linear),
        so their numerics match the TP=1 validated path exactly.
      * Only the four big matmuls are sharded across the TP axis: qkv is
        ColParallel (heads fractured), o is RowParallel, and the SwiGLU FFN is a
        ParallelFeedForward (ff1 col, ff2 row). Each sub-block reduces back to a
        replicated-hidden activation via an all-gather, i.e. an all-reduce.
      * 18 heads are not mesh-friendly; head padding (PaddingConfig) rounds the head
        count up to a multiple of tp_factor with zero-initialized heads, so the
        padded heads/o-proj columns contribute nothing.
      * When sequence_parallel.factor > 1 the sequence is sharded on the SP axis and
        attention uses ring_joint_scaled_dot_product_attention, which all-gathers
        K/V across the SP axis. cos/sin are sharded on sequence to match.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adaln_dim: int,
        attention_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.adaln_dim = adaln_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.padding_config = padding_config

        self.tp_factor = parallel_config.tensor_parallel.factor if parallel_config is not None else 1
        self.tp_axis = parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else 0
        self.sp_factor = parallel_config.sequence_parallel.factor if parallel_config is not None else 1
        self.sp_axis = parallel_config.sequence_parallel.mesh_axis if parallel_config is not None else 0

        if self.tp_factor > 1 or self.sp_factor > 1:
            assert ccl_manager is not None, "TP/SP require a CCLManager"

        # Head padding for TP divisibility (18 heads is not mesh-friendly).
        self.padded_heads = padding_config.target_heads if padding_config is not None else num_heads
        assert self.padded_heads % self.tp_factor == 0, "padded heads must be divisible by tp_factor"
        self.n_local_heads = self.padded_heads // self.tp_factor
        padded_inner_dim = self.head_dim * self.padded_heads

        # --- attention projections (fused QKV, no bias; matches reference qkv/o) ---
        # qkv is column-parallel (output heads fractured across TP); o is row-parallel
        # (input heads fractured, partial sums reduce-scattered then all-gathered).
        self.qkv = ColParallelLinear(
            hidden_size,
            3 * padded_inner_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )
        self.o = RowParallelLinear(
            padded_inner_dim,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )

        # QK-RMSNorm over head_dim (reference norm_q / norm_k = Ideogram4RMSNorm, eps=1e-5).
        # head_dim is never sharded, so these stay replicated regular RMSNorms.
        self.norm_q = RMSNorm(embedding_dim=self.head_dim, norm_eps=attention_eps, bias=False, mesh_device=mesh_device)
        self.norm_k = RMSNorm(embedding_dim=self.head_dim, norm_eps=attention_eps, bias=False, mesh_device=mesh_device)

        # --- the four block RMSNorms (no affine bias; weight only) ---
        self.attention_norm1 = RMSNorm(
            embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device
        )
        self.attention_norm2 = RMSNorm(
            embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device
        )
        self.ffn_norm1 = RMSNorm(embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        self.ffn_norm2 = RMSNorm(embedding_dim=hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)

        # --- SwiGLU MLP: w2(silu(w1(x)) * w3(x)) ---
        # ParallelFeedForward fuses w1 (gate) and w3 (value) into ff1 (ColParallel)
        # via the "swiglu" activation; ff2 (RowParallel) reduce-scatters its output.
        self.feed_forward = ParallelFeedForward(
            dim=hidden_size,
            dim_out=hidden_size,
            inner_dim=intermediate_size,
            activation_fn="swiglu",
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )

        # --- AdaLN: project adaln_input -> 4 * hidden_size (with bias) ---
        self.adaln_modulation = Linear(adaln_dim, 4 * hidden_size, bias=True, mesh_device=mesh_device)

        # SDPA config (fidelity recipe §4 — HiFi2, fp32 acc off; flip on if attn PCC suffers).
        # head_dim=256 (2x the usual 128) doubles SDPA's per-core K/V/score CBs;
        # k_chunk_size=512 overflows Blackhole L1 (1.59MB > 1.5MB max). Halve to 256
        # so the buffers fit. Flash attention is exact regardless of chunk size.
        self.sdpa_q_chunk_size = 128
        self.sdpa_k_chunk_size = 256
        device_grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(device_grid.x, device_grid.y),
            q_chunk_size=self.sdpa_q_chunk_size,
            k_chunk_size=self.sdpa_k_chunk_size,
            exp_approx_mode=False,
        )
        # Ring SDPA (sequence parallel) reserves the last worker row for the CCL all-gather.
        self.sdpa_worker_grid = (device_grid.x, device_grid.y - 1)
        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=self.sdpa_q_chunk_size,
            k_chunk_size=self.sdpa_k_chunk_size,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=MATH_FIDELITY_HIFI2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # flip True if attention PCC is the culprit
        )
        # Higher fidelity for the small AdaLN projection (matmul-class but stability matters).
        self.adaln_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _merge_qkv_for_tp(self, qkv_weight: torch.Tensor) -> torch.Tensor:
        """Rearrange the fused reference QKV weight so column-fracturing shards heads.

        The reference qkv is a single nn.Linear [3*H*hd, hidden] laid out as
        view(.., 3, H, hd): the output is the block [q(all heads) | k | v]. We split
        it back into q/k/v, optionally pad the heads to padded_heads with zeros, then
        interleave per-device so device d's contiguous output slice is
        [q_local | k_local | v_local] for its n_local_heads — exactly what
        split_query_key_value_and_split_heads(num_heads=n_local_heads) consumes.
        """
        hidden = qkv_weight.shape[1]
        per = self.num_heads * self.head_dim
        q, k, v = qkv_weight[:per], qkv_weight[per : 2 * per], qkv_weight[2 * per :]  # each [H*hd, hidden]
        n_dev = self.tp_factor

        def _shape(w):  # nn.Linear [out=H*hd, in=hidden] -> [in, n_dev, n_local_heads, hd]
            w = w.T  # [hidden, H*hd]
            if self.padding_config is not None:
                w = pad_weight_tensor(w, self.padding_config, pad_output_dim=True)  # -> [hidden, padded_H*hd]
            return w.reshape(hidden, n_dev, self.n_local_heads, self.head_dim)

        qkv = torch.cat([_shape(q), _shape(k), _shape(v)], dim=2)  # [hidden, n_dev, 3*n_local_heads, hd]
        qkv = qkv.reshape(hidden, 3 * self.padded_heads * self.head_dim)
        return qkv.T  # nn.Linear layout [3*padded_inner, hidden]; ColParallel transposes + shards output

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Reference submodule names:
        #   attention.qkv / attention.o / attention.norm_q / attention.norm_k
        #   feed_forward.w1 / feed_forward.w2 / feed_forward.w3
        #   attention_norm1/2, ffn_norm1/2, adaln_modulation
        attn = pop_substate(state, "attention")
        if "qkv.weight" in attn:
            state["qkv.weight"] = self._merge_qkv_for_tp(attn["qkv.weight"])
        if "o.weight" in attn:
            # o is nn.Linear [hidden_out, inner_in]. Pad the input (head) dim at the end
            # with zeros to padded_inner so the padded heads contribute nothing; the
            # RowParallel _prepare then transposes [out, in] -> [in, out].
            o_weight = attn["o.weight"]
            if self.padding_config is not None:
                o_weight = pad_weight_tensor(o_weight, self.padding_config, pad_input_dim=False, pad_output_dim=True)
            state["o.weight"] = o_weight
        if "norm_q.weight" in attn:
            state["norm_q.weight"] = attn["norm_q.weight"]
        if "norm_k.weight" in attn:
            state["norm_k.weight"] = attn["norm_k.weight"]

        # SwiGLU: reference is w2(silu(w1(x)) * w3(x)).
        # FeedForward.ff1 with "swiglu" computes: chunk(ff1_out,2) -> a, gate; a * silu(gate).
        # So a must be w3 (value) and gate must be w1 (silu'd). ff1.weight = [w3; w1] stacked
        # on the output dim, then transposed by Linear._prepare_torch_state.
        ff = pop_substate(state, "feed_forward")
        if "w1.weight" in ff and "w3.weight" in ff:
            # nn.Linear weight is [out, in]; stack along out so first half=value, second half=gate.
            w_value = ff["w3.weight"]  # value branch -> "a"
            w_gate = ff["w1.weight"]  # gate branch -> silu(gate)
            state["feed_forward.ff1.weight"] = torch.cat([w_value, w_gate], dim=0)
        if "w2.weight" in ff:
            state["feed_forward.ff2.weight"] = ff["w2.weight"]

    def _all_gather_hidden(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather a TP-fractured-on-hidden activation back to replicated hidden.

        No-op when TP is disabled. Together with the RowParallel reduce-scatter inside
        o / ff2 this forms an all-reduce, restoring the replicated-hidden residual.
        """
        if self.tp_factor <= 1:
            return t
        return self.ccl_manager.all_gather_persistent_buffer(t, dim=2, mesh_axis=self.tp_axis, use_hyperparams=True)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        adaln_input: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None = None,
        spatial_sequence_length: int | None = None,
    ) -> ttnn.Tensor:
        """Single-stream block forward.

        Args:
            x: [B, L/sp, hidden_size] unified text+image sequence. Replicated on the TP
                axis, sharded on the SP axis (full L when sp_factor == 1).
            cos, sin: [B, 1, L/sp, head_dim] rotary tables (rotate-half convention),
                precomputed on host from Ideogram4MRoPE. Broadcast over heads; sharded
                on sequence to match x when sp_factor > 1.
            adaln_input: [B, 1, adaln_dim] (or [B, L, adaln_dim]) SiLU'd time cond.
            attn_mask: additive mask (0 = attend, -inf = block), built from segment ids.
                None => full attention. Without SP: [B, 1, L, L]. With SP: sharded on the
                query-row dim to [B, 1, L/sp, L] (full on keys); the masked-SP path
                all-gathers K/V instead of using ring attention.
            spatial_sequence_length: logical (unpadded) full sequence length. Required
                when sp_factor > 1; defaults to x.shape[1] otherwise.
        """
        batch_size, local_seq_len, _ = x.shape
        if spatial_sequence_length is None:
            spatial_sequence_length = local_seq_len

        # --- AdaLN: 4-branch, tanh gates, (1 + scale). Replicated on all TP devices. ---
        mod = self.adaln_modulation(adaln_input, compute_kernel_config=self.adaln_compute_kernel_config)
        scale_msa, gate_msa, scale_mlp, gate_mlp = ttnn.chunk(mod, 4, -1)
        gate_msa = ttnn.tanh(gate_msa, fast_and_approximate_mode=False)
        gate_mlp = ttnn.tanh(gate_mlp, fast_and_approximate_mode=False)
        scale_msa = scale_msa + 1.0
        scale_mlp = scale_mlp + 1.0

        # ----------------- attention sub-block -----------------
        attn_in = self.attention_norm1(x) * scale_msa
        attn_out = self._attention(
            attn_in, cos=cos, sin=sin, attn_mask=attn_mask, spatial_sequence_length=spatial_sequence_length
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        # ----------------- feed-forward sub-block -----------------
        ff_in = self.ffn_norm1(x) * scale_mlp
        ff_out = self._all_gather_hidden(self.feed_forward(ff_in))  # fractured -> replicated hidden
        x = x + gate_mlp * self.ffn_norm2(ff_out)

        return x

    def _attention(
        self,
        x: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None,
        spatial_sequence_length: int,
    ) -> ttnn.Tensor:
        # qkv is ColParallel: replicated-hidden input -> output fractured on heads, so
        # the per-device output slice is [q_local | k_local | v_local] for n_local_heads.
        qkv = self.qkv(x)  # [B, L/sp, 3 * n_local_heads * head_dim]
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.n_local_heads,
            transpose_key=False,
            memory_config=qkv.memory_config(),
        )  # each [B, n_local_heads, L/sp, head_dim]

        # QK-RMSNorm over head_dim (applied per-head, before RoPE — matches reference,
        # which norms q/k of shape [..., hd] then transposes then applies rope).
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.sp_factor > 1 and attn_mask is None:
            # Sequence parallel, unmasked (full attention): ring SDPA all-gathers K/V
            # across the SP axis. Empty "joint" => plain self-attention over the full seq.
            empty = ttnn.zeros(
                [q.shape[0], self.n_local_heads, 0, self.head_dim],
                device=self.mesh_device,
                layout=q.layout,
                dtype=q.dtype,
            )
            out, _prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                empty,
                empty,
                empty,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(k.shape, 2, self.sp_axis),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(v.shape, 2, self.sp_axis),
                joint_strategy="rear",
                logical_n=spatial_sequence_length,
                program_config=self.ring_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.sp_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
            )  # [B, n_local_heads, L/sp, head_dim]
        else:
            if self.sp_factor > 1:
                # Sequence parallel + segment mask: ring SDPA takes no additive mask, so
                # all-gather full K/V across the SP axis and run masked SDPA with Q kept
                # sequence-sharded. The output stays sharded on sequence, so the rest of
                # the block remains sequence-parallel; only K/V are replicated here.
                # attn_mask is [B, 1, L/sp, L] (sharded on the query rows, full on keys).
                k = self.ccl_manager.all_gather_persistent_buffer(
                    k, dim=2, mesh_axis=self.sp_axis, use_hyperparams=True
                )
                v = self.ccl_manager.all_gather_persistent_buffer(
                    v, dim=2, mesh_axis=self.sp_axis, use_hyperparams=True
                )
            out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )  # [B, n_local_heads, L/sp, head_dim]

        out = ttnn.transformer.concatenate_heads(out)  # [B, L/sp, n_local_heads * head_dim] (fractured on TP)
        # o is RowParallel: fractured-heads input -> reduce-scatter -> fractured hidden;
        # all-gather restores the replicated-hidden residual stream.
        return self._all_gather_hidden(self.o(out))


class Ideogram4Transformer(Module):
    """Full Ideogram 4.0 single-stream denoiser (34 blocks).

    Provided for completeness/structure; the bringup target validated here is the
    single block. The end-to-end model wires the input/LLM/time embeddings, MRoPE,
    the block stack, and the final layer. Several host-side preprocessing pieces
    (sinusoidal time embed, MRoPE table construction, masked token fusion) are
    expected to be precomputed on host and fed to the block stack, mirroring how
    the Mochi/Qwen ports precompute RoPE and timestep features on host.

    NOTE: only the block (Ideogram4TransformerBlock) is covered by the bringup
    test. This class is a thin scaffold and is itself UNVALIDATED.
    """

    def __init__(
        self,
        *,
        emb_dim: int = 4608,
        num_layers: int = 34,
        num_heads: int = 18,
        intermediate_size: int = 12288,
        adaln_dim: int = 512,
        in_channels: int = 128,
        norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.mesh_device = mesh_device

        self.layers = ModuleList(
            Ideogram4TransformerBlock(
                hidden_size=emb_dim,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                norm_eps=norm_eps,
                adaln_dim=adaln_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        h: ttnn.Tensor,
        *,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        adaln_input: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None = None,
        spatial_sequence_length: int | None = None,
    ) -> ttnn.Tensor:
        for layer in self.layers:
            h = layer(
                h,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
                attn_mask=attn_mask,
                spatial_sequence_length=spatial_sequence_length,
            )
        return h
