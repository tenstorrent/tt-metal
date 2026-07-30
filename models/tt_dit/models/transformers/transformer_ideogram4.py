# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedRMSNorm, LayerNorm, RMSNorm
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...reference.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from ...utils.mochi import get_rot_transformation_mat
from ...utils.padding import pad_weight_tensor
from ...utils.substate import pop_substate
from ...utils.tensor import bf16_tensor


def rope_halfsplit_to_interleaved_perm(head_dim: int) -> torch.Tensor:
    """Index permutation mapping the reference half-split RoPE layout to the interleaved
    (adjacent-pair) layout that ``ttnn.experimental.rotary_embedding_llama`` + the 32x32
    transformation matrix consume: ``out[2i] = in[i]``, ``out[2i+1] = in[i + head_dim/2]``.

    Applied identically to the Q/K projection output channels, the QK-norm affine weights,
    and the cos/sin tables. Because it is a SHARED permutation of head_dim on Q and K (and V/O
    are untouched), attention Q·Kᵀ is invariant -> the layout swap is numerically neutral.
    """
    h = head_dim // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], dim=1).flatten()


def rope_halfsplit_to_interleaved(cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """Permute half-split cos/sin tables (cat(freqs,freqs)) into the interleaved
    [f0,f0,f1,f1,...] layout expected by rotary_embedding_llama. Host-side helper for callers."""
    perm = rope_halfsplit_to_interleaved_perm(head_dim)
    return cos[..., perm], sin[..., perm]


MATH_FIDELITY_HIFI2 = ttnn.MathFidelity.HiFi2
MATH_FIDELITY_HIFI4 = ttnn.MathFidelity.HiFi4


# RoPE is applied with ttnn.experimental.rotary_embedding_llama (the INTERLEAVED
# adjacent-pair convention, rotate via the 32x32 transformation matrix). The reference
# MRoPE is half-split; we permute Q/K channels + cos/sin into the interleaved layout
# (rope_halfsplit_to_interleaved*) so this fused op applies -- a shared head_dim
# permutation on Q and K, so attention Q·Kᵀ is unchanged. This lets Ideogram reuse the
# standard rope fusion and the fused per-head norm+RoPE op.


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

    Parallelism (Wan-style TP-fractured residual + ring sequence parallelism):
      * The residual stream stays FRACTURED on hidden (TP axis) and SHARDED on
        sequence (SP axis): [B, L/sp, hidden/tp]. This is the Wan layout — the four
        block norms are DistributedRMSNorm (cross-device stats via one small stats
        all-gather), which removes the AllGather-to-replicated that the old
        replicated-hidden residual forced after every RowParallel reduce-scatter.
      * The four big matmuls are ColParallel: qkv/ff1 fracture their output on
        heads/inner; o/ff2 fracture their OUTPUT on hidden (Wan's `to_out` scheme)
        so the residual add lands on the fractured stream directly — no RowParallel
        reduce-scatter + separate all-gather. On Linear topology (the loudbox) the
        fractured-hidden input to each ColParallel matmul is all-gathered to
        replicated first (`use_nonfused_agmm`), matching Wan's Linear path. The
        Ring-only fused input-AG-matmul / MM+RS ops are NOT used here (SP axis size
        4 on the (4,2) loudbox runs Linear, exactly as Wan does for axes <= 4).
      * AdaLN (scale_msa, gate_msa, scale_mlp, gate_mlp) is projected fractured on
        hidden (ColParallel-style interleave) so the 4 branches align with the
        fractured norm outputs / residual.
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

        # The fused per-head QK-norm (dit_fused_distributed_rmsnorm) needs a CCLManager on
        # every config (it allocates a semaphore + stats buffer), including single device.
        assert ccl_manager is not None, "Ideogram4TransformerBlock requires a CCLManager"

        # Head padding for TP divisibility (18 heads is not mesh-friendly).
        self.padded_heads = padding_config.target_heads if padding_config is not None else num_heads
        assert self.padded_heads % self.tp_factor == 0, "padded heads must be divisible by tp_factor"
        self.n_local_heads = self.padded_heads // self.tp_factor
        padded_inner_dim = self.head_dim * self.padded_heads

        # --- attention projections (fused QKV, no bias; matches reference qkv/o) ---
        # qkv is column-parallel (output heads fractured across TP). o is ALSO
        # ColParallel (Wan `to_out` scheme): its input is the concatenated-heads
        # activation (fractured on heads -> all-gathered to replicated under Linear
        # topology), and its output is fractured on HIDDEN, matching the fractured
        # residual stream. No RowParallel reduce-scatter + separate all-gather.
        # chunks=3: the fused qkv matmul splits its per-device output
        # [q_local | k_local | v_local] into the three tensors on-device
        # (ttnn.experimental.minimal_matmul_split), avoiding the slice ops a post-hoc
        # ttnn.chunk would emit. _merge_qkv_for_tp already lays the weight out so each
        # device's contiguous output is exactly [q_local | k_local | v_local].
        self.qkv = ColParallelLinear(
            hidden_size,
            3 * padded_inner_dim,
            bias=False,
            chunks=3,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )
        self.o = ColParallelLinear(
            padded_inner_dim,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )

        # QK-RMSNorm over head_dim (reference norm_q / norm_k = Ideogram4RMSNorm, eps=1e-5).
        # Fuse the per-head QK-norm + head-split + interleaved RoPE into ONE op via
        # DistributedRMSNorm in per-head mode (num_heads_per_device). head_dim is fully
        # on-device, so the per-head RMS reduction needs no all-gather (works on a single
        # device too); embedding_dim = padded_inner_dim so the per-device weight slice covers
        # n_local_heads * head_dim.
        _qk_norm_kwargs = dict(
            embedding_dim=padded_inner_dim,
            norm_eps=attention_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_axis=self.tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.norm_q = DistributedRMSNorm(**_qk_norm_kwargs)
        self.norm_k = DistributedRMSNorm(**_qk_norm_kwargs)

        # --- the four block RMSNorms (weight only, no bias) ---
        # DistributedRMSNorm operates on the TP-fractured hidden dim, computing the RMS
        # statistic across devices via a small stats all-gather (Wan scheme); at tp_factor == 1
        # it degrades to a single-device norm (no-op AG). Always the fused op so the norm1
        # adaLN scale can be folded in via dynamic_weight (see forward).
        def _block_norm():
            return DistributedRMSNorm(
                embedding_dim=hidden_size,
                norm_eps=norm_eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_axis=self.tp_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )

        self.attention_norm1 = _block_norm()
        self.attention_norm2 = _block_norm()
        self.ffn_norm1 = _block_norm()
        self.ffn_norm2 = _block_norm()

        # --- SwiGLU MLP: w2(silu(w1(x)) * w3(x)) ---
        # ff1 (ColParallel, swiglu) fuses w1 (gate) + w3 (value) and fractures its
        # inner output; ff2 (ColParallel) fractures its OUTPUT on hidden to match the
        # fractured residual (Wan scheme). ff2's input (fractured on inner) is
        # all-gathered to replicated under Linear topology before the matmul.
        self.ff1 = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            activation_fn="swiglu",
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )
        self.ff2 = ColParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )

        # --- AdaLN: project adaln_input -> 4 * hidden_size (with bias) ---
        # ColParallel so the output is fractured on hidden; the weight is interleaved
        # (device-outer, branch-inner) so each device's contiguous 4*hidden/tp slice
        # is [s_msa | g_msa | s_mlp | g_mlp] for its hidden-shard, and chunk(4) yields
        # the local hidden-shard of each of the 4 branches — aligned with the
        # fractured norm outputs and residual. At tp==1 ColParallelLinear is a no-op shard
        # (identical to a plain Linear), so no tp branch is needed — matches qkv/o/ff1/ff2.
        self.adaln_modulation = ColParallelLinear(
            adaln_dim,
            4 * hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )

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
        self._ring_sdpa_pc_cache = {}
        # ------------------------------------------------------------------
        # MIXED math fidelity (mirrors Wan2.2): HiFi2 for the heavy per-token
        # compute (the four big block matmuls + SDPA), HiFi4 for the
        # precision-sensitive ops (RMSNorm/QK-norm run HiFi4 internally, AdaLN
        # modulation matmul, and the once-per-forward model-level projections).
        #
        # Rationale: a blanket HiFi4 (the previous state) doubled matmul/SDPA
        # device time; a blanket HiFi2 dropped whole-model real-weight PCC to
        # ~0.956. The split keeps HiFi4 where the dynamic range hurts and HiFi2
        # where the flops dominate, halving matmul/SDPA cost while holding PCC.
        # ------------------------------------------------------------------

        # HiFi2 + fp32 accumulate for the four big matmuls (QKV, O, FF1, FF2).
        self.mm_hifi2_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # HiFi2 for SDPA. head_dim=256 is precision-heavy, but the fp32_dest_acc
        # sweep showed fp32_dest_acc={True,False} give the SAME block PCC
        # (img1024 99.976% / img4096 99.980% either way) and the SAME device
        # time at 2048px, so we land fp32_dest_acc=False to match Wan. Flip to
        # True if a future config regresses SDPA correctness.
        self.sdpa_hifi2_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )
        # HiFi4 + fp32 accumulate for the precision-sensitive matmul-class ops
        # (currently the AdaLN modulation projection). RMSNorm / QK-RMSNorm run
        # HiFi4 internally (see layers/normalization.py). RoPE is pure
        # element-wise (mul/neg/concat) and takes no compute-kernel config, so
        # its precision is unaffected by these settings.
        self.hifi4_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Wiring: the four big matmuls + SDPA -> HiFi2; AdaLN -> HiFi4.
        self.matmul_compute_kernel_config = self.mm_hifi2_config
        self.sdpa_compute_kernel_config = self.sdpa_hifi2_config
        self.adaln_compute_kernel_config = self.hifi4_config

        # RoPE (interleaved): the 32x32 transformation matrix that rotates adjacent channel
        # pairs, replicated across the mesh. Fed to the fused per-head QK-norm+RoPE op.
        self.rope_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
        # Cached 0-element "joint" for ring SDPA (self-attention has no joint tokens). Allocated
        # once here (not per-forward) so trace capture never hits ttnn.zeros' host-write path.
        self.dummy_joint = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

    def _get_ring_sdpa_program_config(self, local_seq_len):
        """Ring-SDPA program config (fixed q_chunk=128).

        NOTE: a standalone *plain*-SDPA micro-sweep suggested q_chunk=256 was ~1.3x
        faster at long local seq (4352 @ 2048px). That does NOT hold for the production
        ring_joint SDPA: its extra joint/CCL circular buffers push q_chunk=256 past L1
        ("CBs grow to 1712656 B > 1572864 B max L1") and it fails at true 2048px. So we
        keep the proven q_chunk=128 at every resolution (k_chunk=256, head_dim cap).
        """
        qc = self.sdpa_q_chunk_size
        pc = self._ring_sdpa_pc_cache.get(qc)
        if pc is None:
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.sdpa_worker_grid,
                q_chunk_size=qc,
                k_chunk_size=self.sdpa_k_chunk_size,
                exp_approx_mode=False,
            )
            self._ring_sdpa_pc_cache[qc] = pc
        return pc

    def _merge_qkv_for_tp(self, qkv_weight: torch.Tensor) -> torch.Tensor:
        """Rearrange the fused reference QKV weight so column-fracturing shards heads.

        The reference qkv is a single nn.Linear [3*H*hd, hidden] laid out as
        view(.., 3, H, hd): the output is the block [q(all heads) | k | v]. We split
        it back into q/k/v, optionally pad the heads to padded_heads with zeros, then
        interleave per-device so device d's contiguous output slice is
        [q_local | k_local | v_local] for its n_local_heads — which the ColParallel
        chunks=3 matmul then splits on-device into the three q/k/v tensors.
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

        # Permute Q and K head_dim channels half-split -> interleaved (NOT V) so the projected
        # q/k match the interleaved RoPE convention; V/O are untouched (attention Q·Kᵀ invariant).
        rope_perm = rope_halfsplit_to_interleaved_perm(self.head_dim)
        qkv = torch.cat(
            [_shape(q)[..., rope_perm], _shape(k)[..., rope_perm], _shape(v)], dim=2
        )  # [hidden, n_dev, 3*n_local_heads, hd]
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
            # o is nn.Linear [hidden_out, inner_in], now a ColParallelLinear that
            # fractures its hidden OUTPUT. Pad the input (inner/head) column dim with
            # zeros to padded_inner so the padded heads (and the AG'd zero head slots)
            # contribute nothing; ColParallel._prepare then transposes [out, in] ->
            # [in, out] and shards the hidden output on TP.
            o_weight = attn["o.weight"]
            if self.padding_config is not None:
                o_weight = pad_weight_tensor(o_weight, self.padding_config, pad_input_dim=False, pad_output_dim=True)
            state["o.weight"] = o_weight
        # QK-norm affine weights are per-head_dim and applied before RoPE, so permute them by the
        # same half-split -> interleaved perm as the Q/K projection channels. For the fused
        # per-head QK-norm the DistributedRMSNorm weight spans padded_inner_dim
        # (= padded_heads * head_dim), so tile the single per-head_dim weight across all
        # padded heads; TP-sharding then hands each device its n_local_heads' worth.
        rope_perm = rope_halfsplit_to_interleaved_perm(self.head_dim)

        def _qk_norm_weight(w):  # w: [head_dim] -> [padded_inner_dim]
            return w[..., rope_perm].repeat(self.padded_heads)

        if "norm_q.weight" in attn:
            state["norm_q.weight"] = _qk_norm_weight(attn["norm_q.weight"])
        if "norm_k.weight" in attn:
            state["norm_k.weight"] = _qk_norm_weight(attn["norm_k.weight"])

        # SwiGLU: reference is w2(silu(w1(x)) * w3(x)).
        # ff1 (ColParallel, swiglu) computes: chunk(ff1_out,2) -> a, gate; a * silu(gate).
        # So a must be w3 (value) and gate must be w1 (silu'd). ff1.weight = [w3; w1]
        # stacked on the output dim; ColParallel._prepare transposes + swiglu-permutes.
        # ff2 (ColParallel) fractures its hidden output: keep nn.Linear [hidden, inner]
        # and ColParallel._prepare transposes [out, in] -> [in, out], shards hidden.
        ff = pop_substate(state, "feed_forward")
        if "w1.weight" in ff and "w3.weight" in ff:
            w_value = ff["w3.weight"]  # value branch -> "a"
            w_gate = ff["w1.weight"]  # gate branch -> silu(gate)
            state["ff1.weight"] = torch.cat([w_value, w_gate], dim=0)
        if "w2.weight" in ff:
            state["ff2.weight"] = ff["w2.weight"]

        # AdaLN modulation: at tp>1 the ColParallel output must be interleaved
        # device-outer / branch-inner so device d's contiguous 4*hidden/tp slice is
        # [s_msa_d | g_msa_d | s_mlp_d | g_mlp_d] (each hidden/tp wide). Then chunk(4)
        # on the local output yields the 4 branch shards in order.
        if self.tp_factor > 1 and "adaln_modulation.weight" in state:
            n_dev = self.tp_factor
            hs = self.hidden_size
            w = state["adaln_modulation.weight"]  # nn.Linear [4*hidden, adaln_dim]
            # [4, n_dev, hidden/tp, adaln_dim] -> [n_dev, 4, hidden/tp, adaln_dim]
            w = w.reshape(4, n_dev, hs // n_dev, w.shape[1]).permute(1, 0, 2, 3).reshape(4 * hs, w.shape[1])
            state["adaln_modulation.weight"] = w
            if "adaln_modulation.bias" in state:
                b = state["adaln_modulation.bias"]  # [4*hidden]
                b = b.reshape(4, n_dev, hs // n_dev).permute(1, 0, 2).reshape(4 * hs)
                state["adaln_modulation.bias"] = b

    def _all_gather_hidden(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather a TP-fractured-on-hidden activation to replicated hidden.

        No-op when TP is disabled. Used to feed the fractured residual/norm output
        into a ColParallel matmul (which expects replicated input) under Linear
        topology (`use_nonfused_agmm` — the Ring fused input-AG-matmul is not used
        on the size-4 SP axis of the (4,2) loudbox).
        """
        if self.tp_factor <= 1:
            return t
        return self.ccl_manager.all_gather_persistent_buffer(t, dim=2, mesh_axis=self.tp_axis, use_hyperparams=True)

    def _block_norm(self, norm, x: ttnn.Tensor, dynamic_weight: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """Apply a block RMSNorm. DistributedRMSNorm requires 4-D input, so unsqueeze
        [B, L, D/tp] -> [1, B, L, D/tp] around it and squeeze back. When dynamic_weight is
        given (an adaLN (1 + scale) factor), it is folded into the norm's affine in-op,
        replacing a separate elementwise scale multiply."""
        x = ttnn.unsqueeze(x, 0)
        x = norm(x, dynamic_weight=dynamic_weight)
        return ttnn.squeeze(x, 0)

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
            x: [B, L/sp, hidden/tp] unified text+image sequence. FRACTURED on the TP
                axis (hidden) and sharded on the SP axis (full L when sp_factor == 1).
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

        Returns [B, L/sp, hidden/tp] — fractured on TP (hidden), sharded on SP.
        """
        batch_size, local_seq_len, _ = x.shape
        # Single-batch invariant: the attention folds to a batch-1 layout (ttnn.unsqueeze(q_f, 0))
        # and the cos/sin MRoPE tables broadcast over dim 0, so genuinely-batched per-sample MRoPE
        # (distinct position_ids per sample) is NOT handled. The pipeline always runs B=1; assert it
        # loudly rather than silently mis-applying rope the day someone batches.
        assert batch_size == 1, f"Ideogram4 transformer supports batch_size == 1 only (got {batch_size})"
        # Under SP the local shard is L/sp, so the local length is NOT the logical n for ring
        # SDPA; require the caller to pass the true (unpadded) full length rather than silently
        # using a wrong logical_n. Without SP the local length IS the full length.
        if spatial_sequence_length is None:
            assert self.sp_factor == 1, "spatial_sequence_length is required when sequence-parallel (sp_factor > 1)"
            spatial_sequence_length = local_seq_len

        # --- AdaLN: 4-branch, tanh gates, (1 + scale). Fractured on hidden (TP). ---
        # ColParallel adaln_modulation output is interleaved so chunk(4) gives each
        # branch's local hidden-shard, aligned with the fractured norm outputs.
        mod = self.adaln_modulation(adaln_input, compute_kernel_config=self.adaln_compute_kernel_config)
        scale_msa, gate_msa, scale_mlp, gate_mlp = ttnn.chunk(mod, 4, -1)
        gate_msa = ttnn.tanh(gate_msa, fast_and_approximate_mode=False)
        gate_mlp = ttnn.tanh(gate_mlp, fast_and_approximate_mode=False)
        # The reference applies `norm(x) * (1 + scale)`. Rather than a separate elementwise
        # multiply, fold (1 + scale) into the norm1 affine via dynamic_weight so the fused
        # RMSNorm applies it in-op (fp32 internals). scale_* are per-sample [B, 1, hidden/tp],
        # so (1 + scale) is a small tensor the norm multiplies its static weight by.
        one_plus_scale_msa = ttnn.add(scale_msa, 1.0)
        one_plus_scale_mlp = ttnn.add(scale_mlp, 1.0)

        # ----------------- attention sub-block -----------------
        # norm1(x): DistributedRMSNorm on fractured hidden (cross-device RMS stats), with the
        # adaLN (1 + scale_msa) folded into its affine -> fractured out (no separate scale op).
        attn_in = self._block_norm(self.attention_norm1, x, dynamic_weight=one_plus_scale_msa)
        attn_out = self._attention(
            attn_in, cos=cos, sin=sin, attn_mask=attn_mask, spatial_sequence_length=spatial_sequence_length
        )  # fractured on hidden
        # x = x + gate_msa * norm2(attn_out); all fractured on hidden.
        x = ttnn.addcmul(x, gate_msa, self._block_norm(self.attention_norm2, attn_out), value=1.0)

        # ----------------- feed-forward sub-block -----------------
        # norm(x) * (1 + scale_mlp): (1 + scale_mlp) folded into the ffn_norm1 affine.
        ff_in = self._block_norm(self.ffn_norm1, x, dynamic_weight=one_plus_scale_mlp)
        ff_in = self._all_gather_hidden(ff_in)  # fractured -> replicated for ColParallel ff1
        ff_hidden = self.ff1(ff_in, compute_kernel_config=self.matmul_compute_kernel_config)  # fractured on inner
        ff_hidden = self._all_gather_hidden(ff_hidden)  # inner fractured -> replicated for ColParallel ff2
        ff_out = self.ff2(ff_hidden, compute_kernel_config=self.matmul_compute_kernel_config)  # fractured on hidden
        x = ttnn.addcmul(x, gate_mlp, self._block_norm(self.ffn_norm2, ff_out), value=1.0)

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
        # x arrives fractured on hidden; qkv is ColParallel and expects replicated
        # input, so all-gather to replicated first (Linear-topology `use_nonfused_agmm`).
        x = self._all_gather_hidden(x)
        # qkv is ColParallel with chunks=3: replicated-hidden input -> three tensors each
        # fractured on heads, [B, L/sp, n_local_heads*head_dim] = [q_local | k_local | v_local]
        # split on-device by the fused matmul. Unsqueeze to the [1, B, L/sp, F] layout the
        # fused QK-norm consumes. norm_q/k then norm each head over head_dim AND apply
        # interleaved RoPE in one op, emitting head-split [B, n_local_heads, L/sp, head_dim];
        # v is only head-split (no norm/RoPE).
        q_f, k_f, v_f = self.qkv(x, compute_kernel_config=self.matmul_compute_kernel_config)
        q_f = ttnn.unsqueeze(q_f, 0)
        k_f = ttnn.unsqueeze(k_f, 0)
        v_f = ttnn.unsqueeze(v_f, 0)
        # per_head_norm=True: QK-RMSNorm is independent over each head's head_dim (the op
        # default is whole-row; Ideogram4 must opt in).
        q = self.norm_q(
            q_f,
            num_heads_per_device=self.n_local_heads,
            rope_cos=cos,
            rope_sin=sin,
            trans_mat=self.rope_trans_mat,
            per_head_norm=True,
        )
        k = self.norm_k(
            k_f,
            num_heads_per_device=self.n_local_heads,
            rope_cos=cos,
            rope_sin=sin,
            trans_mat=self.rope_trans_mat,
            per_head_norm=True,
        )
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v_f, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )

        if self.sp_factor > 1 and attn_mask is None:
            # Sequence parallel, unmasked (full attention): ring SDPA all-gathers K/V
            # across the SP axis. Empty "joint" => plain self-attention over the full seq.
            empty = self.dummy_joint
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
                program_config=self._get_ring_sdpa_program_config(q.shape[2]),
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

        out = ttnn.transformer.concatenate_heads(out)  # [B, L/sp, n_local_heads * head_dim] (fractured on heads)
        # o is ColParallel: all-gather the concatenated-heads activation to the full
        # (padded) inner dim, then matmul -> output FRACTURED on hidden, matching the
        # fractured residual. No RowParallel reduce-scatter + separate all-gather.
        out = self._all_gather_hidden(out)  # heads fractured -> replicated padded_inner
        out = self.o(out, compute_kernel_config=self.matmul_compute_kernel_config)  # fractured on hidden
        return out


class _EmbedScalarMLP(Module):
    """The learnable tail of reference Ideogram4EmbedScalar: silu(mlp_in(x)) -> mlp_out.

    The parameter-free sinusoidal embedding is precomputed on host (see
    Ideogram4Transformer.sinusoidal_embedding) and fed in.
    """

    def __init__(self, dim: int, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.mlp_in = Linear(dim, dim, bias=True, mesh_device=mesh_device)
        self.mlp_out = Linear(dim, dim, bias=True, mesh_device=mesh_device)

    def forward(self, emb: ttnn.Tensor) -> ttnn.Tensor:
        return self.mlp_out(ttnn.silu(self.mlp_in(emb)))


class _FinalLayer(Module):
    """Reference Ideogram4FinalLayer: linear(norm_final(x) * (1 + adaln_modulation(silu(c))))."""

    def __init__(self, hidden_size: int, out_channels: int, adaln_dim: int, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, norm_eps=1e-6, norm_elementwise_affine=False, mesh_device=mesh_device)
        self.linear = Linear(hidden_size, out_channels, bias=True, mesh_device=mesh_device)
        self.adaln_modulation = Linear(adaln_dim, hidden_size, bias=True, mesh_device=mesh_device)
        # Once-per-forward final projection + scale/shift: keep HiFi4 for safety
        # (cheap, and it is the last layer before the velocity output).
        self.hifi4_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor, c: ttnn.Tensor) -> ttnn.Tensor:
        scale = self.adaln_modulation(ttnn.silu(c), compute_kernel_config=self.hifi4_config) + 1.0
        return self.linear(self.norm_final(x) * scale, compute_kernel_config=self.hifi4_config)


class Ideogram4Transformer(Module):
    """Full Ideogram 4.0 single-stream flow-matching denoiser (34 blocks).

    Faithful port of the reference Ideogram4Transformer. Learnable transforms run on
    device; the parameter-free scaffolding (MRoPE cos/sin tables, the sinusoidal time
    embedding, the indicator-derived masks and the image-indicator index) is
    precomputed on host and fed in as device tensors — mirroring the Mochi/Qwen/Wan
    ports. Forward mirrors the reference:

        x = input_proj(x * out_img_mask) * out_img_mask
        llm = llm_cond_proj(llm_cond_norm(llm * llm_mask)) * llm_mask
        h = x + llm + embed_image_indicator(is_image)
        adaln_input = silu(adaln_proj(t_embedding(t_sin)))
        for layer: h = layer(h, cos, sin, adaln_input, mask)
        out = final_layer(h, adaln_input)

    Parallelism is delegated to the blocks (Wan-style TP-fractured residual): the
    wrapper embeddings build h replicated on hidden, the model fractures it on the TP
    axis (a local mesh_partition slice) before the block loop, the block stack runs on
    the fractured stream, and the model all-gathers back to replicated hidden for the
    (replicated) final layer. SP shards the sequence (sequence-parallel wrapper handling
    is threaded through forward via pre-sharded inputs + spatial_sequence_length).
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
        llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS),
        norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.llm_features_dim = llm_features_dim
        self.head_dim = emb_dim // num_heads
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        # The residual stream is TP-fractured on hidden inside the blocks. The wrapper
        # embeddings build h replicated on hidden; fracture it before the block loop and
        # gather back to replicated for the (replicated) final layer.
        self.tp_factor = parallel_config.tensor_parallel.factor if parallel_config is not None else 1
        self.tp_axis = parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else 0

        # --- input / conditioning embeddings (replicated) ---
        self.input_proj = Linear(in_channels, emb_dim, bias=True, mesh_device=mesh_device)
        self.llm_cond_norm = RMSNorm(embedding_dim=llm_features_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.llm_cond_proj = Linear(llm_features_dim, emb_dim, bias=True, mesh_device=mesh_device)
        self.t_embedding = _EmbedScalarMLP(emb_dim, mesh_device=mesh_device)
        self.adaln_proj = Linear(emb_dim, adaln_dim, bias=True, mesh_device=mesh_device)
        self.embed_image_indicator = Embedding(2, emb_dim, device=mesh_device)

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

        self.final_layer = _FinalLayer(emb_dim, in_channels, adaln_dim, mesh_device=mesh_device)

        # Once-per-forward patchify / conditioning projections: keep HiFi4 for
        # safety (cheap; the block matmuls run HiFi2 — see the block class).
        self.hifi4_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY_HIFI4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # rotary_emb is a parameter-free buffer module in the reference; drop it (cos/sin
        # are precomputed on host). Everything else maps 1:1 by attribute name.
        for key in list(state):
            if key.startswith("rotary_emb."):
                state.pop(key)

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, *, input_range=(0.0, 1.0), scale: float = 1e4) -> torch.Tensor:
        """Host-side replica of reference Ideogram4EmbedScalar's sinusoid (no params).

        t: (...,) scalars. Returns (..., dim). Feed the result to t_embedding on device.
        """
        import math

        lo, hi = input_range
        x = scale * (t.to(torch.float32) - lo) / (hi - lo)
        half = dim // 2
        freq = math.log(scale) / (half - 1)
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * -freq)
        emb = x.unsqueeze(-1) * freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        *,
        x: ttnn.Tensor,
        llm_features: ttnn.Tensor,
        t_sin: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        image_indicator_index: ttnn.Tensor,
        llm_token_mask: ttnn.Tensor,
        output_image_mask: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None = None,
        spatial_sequence_length: int | None = None,
    ) -> ttnn.Tensor:
        """Velocity prediction.

        Host-precomputed device inputs (sharded on sequence for SP, replicated otherwise):
            x: [B, L/sp, in_channels] noise tokens.
            llm_features: [B, L/sp, llm_features_dim] Qwen3-VL features.
            t_sin: [B, 1, emb_dim] sinusoidal time embedding (per-sample).
            cos, sin: [B, 1, L/sp, head_dim] MRoPE tables.
            image_indicator_index: [B, L/sp] uint32 (1 = OUTPUT_IMAGE token, else 0).
            llm_token_mask, output_image_mask: [B, L/sp, 1] bf16 (0/1).
            attn_mask: segment mask (see Ideogram4TransformerBlock.forward).
            spatial_sequence_length: logical full sequence length (required for SP).

        Returns [B, L/sp, in_channels] velocity (only OUTPUT_IMAGE positions are meaningful).
        """
        x = self.input_proj(x * output_image_mask, compute_kernel_config=self.hifi4_config) * output_image_mask

        llm = self.llm_cond_norm(llm_features * llm_token_mask)
        llm = self.llm_cond_proj(llm, compute_kernel_config=self.hifi4_config) * llm_token_mask

        h = x + llm
        h = h + self.embed_image_indicator(image_indicator_index)

        adaln_input = ttnn.silu(self.adaln_proj(self.t_embedding(t_sin)))

        # Fracture the replicated-hidden residual on the TP axis (local slice, the
        # inverse of all-gather — no cross-device movement) so the block stack runs on
        # [B, L/sp, hidden/tp].
        if self.tp_factor > 1:
            h = ttnn.mesh_partition(h, dim=-1, cluster_axis=self.tp_axis)

        for layer in self.layers:
            h = layer(
                h,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
                attn_mask=attn_mask,
                spatial_sequence_length=spatial_sequence_length,
            )

        # Gather the fractured residual back to replicated hidden for the final layer
        # (norm_final / adaln / proj_out all run replicated).
        if self.tp_factor > 1:
            h = self.ccl_manager.all_gather_persistent_buffer(h, dim=2, mesh_axis=self.tp_axis, use_hyperparams=True)

        return self.final_layer(h, adaln_input)
