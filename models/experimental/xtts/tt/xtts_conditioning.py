# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 audio conditioning path.

Mirrors ``reference/xtts_conditioning.py``: ``ConditioningEncoder`` (init conv +
6 attention blocks) followed by ``PerceiverResampler`` (32 latents, depth 2),
producing the GPT conditioning latents ``[b, 1024, 32]`` from a mel ``[b, 80, s]``.

Everything runs in ``[batch, seq, channels]`` (tokens x channels) layout so the
per-timestep ``Conv1d(k=1)`` layers become plain ``ttnn.linear`` and attention is
``ttnn.transformer.scaled_dot_product_attention``. Key equivalences used:

  * ConditioningEncoder QKV scale ``1/sqrt(sqrt(ch))`` on both q and k == the
    standard ``1/sqrt(head_dim)`` SDPA scale (default, non-causal).
  * The perceiver's ``F.normalize(x, dim=-1) * sqrt(dim) * gamma`` == ``ttnn.rms_norm``.

GroupNorm(32, 1024) is computed manually (see ``_group_norm``): free of any reshape OR
transpose, using a block-diagonal group-averaging matmul.

PERF NOTES (device time per pass, mel s=269, blackhole). The head-split/merge plumbing used to
dominate: a ``reshape`` splitting the LAST dim (``[1,s,3072] -> [1,s,16,192]``) is not a view — it
untilizes + retilizes the whole activation (189 us/block) — and the ``permute`` + last-dim
``reshape`` merge cost another 101 us/block. Both are gone:

  * The per-head-interleaved checkpoint qkv layout ``[h0:q,k,v | h1:q,k,v | ...]`` is repermuted ONCE
    on host (``_perm_qkv_out``) into the standard ``[q_all | k_all | v_all]`` head-major blocks, so
    the fused ``ttnn.experimental.nlp_create_qkv_heads`` (one op: split + per-head reshape) and
    ``ttnn.transformer.concatenate_heads`` (one op: permute + merge) replace 1 reshape + 3 slices +
    4 permutes + 1 reshape per block. Only a leading-singleton ``reshape`` remains, which IS a view.
  * The perceiver cross-attention fuses ``to_q``/``to_kv`` into ONE ``[1024, 1536]`` weight applied
    to ``[latents ; context]``, so it too goes through ``nlp_create_qkv_heads``; the latents' Q is
    then a tile-aligned ROW slice (rows 0..32). The fused matmul reads the same weight bytes as the
    two it replaces, so it is not slower.
  * The perceiver GEGLU no longer slices ``[1, 32, 5460]`` in half — an offset-2730 slice is not
    tile-aligned, so it forced an untilize/retilize. The single ``ff.0`` weight is split on host into
    the value/gate halves (two matmuls, same total weight bytes), and the gate's GELU is fused into
    its matmul epilogue (``activation="gelu"`` == exact erf GELU, matching ``F.gelu``).
  * The group norm no longer permutes into a channels-first ``[1, 1024, s]`` layout to reduce over
    seq. ``ttnn.mean`` over ``dim=-2`` is a native H reduction (``reduce_impl``'s ``single_reduce_op``
    covers ``rank-1`` and ``rank-2``, so only OTHER dims get a transpose injected under the hood), so
    the reduction runs in place on ``[1, s, 1024]`` and the two ``permute``s per block are gone —
    12 of the pass's 14 ``TransposeDeviceOperation``s. Traced device time (mel s=269, blackhole):
    Transpose 32.5 -> 3.0 us, and FillPad 94.7 -> 59.1 us as a side effect, because the reduce's
    ``fill_implicit_tile_padding`` now writes 19 WHOLE rows of 1024 instead of a 19-column stripe
    down 1024 rows. Those two are worth -65 us but the row layout makes the group norm's own work
    slightly dearer (its expand matmuls +17 us, the H-broadcast eltwise +21 us), so the pass nets
    2228 -> 2201 us, -1.2%; traced wall 2.31 -> 2.26 ms. Eager is unchanged (12.6 ms either way) —
    12 fewer launches out of 183 does not move it. It also HELPS accuracy: over a mel-length sweep
    vs the fp32 reference, min PCC 0.9922 -> 0.9938, mean 0.9963 -> 0.9973.
  * The INIT matmul ([1,s,80] @ [80,1024]) has a PINNED program config, ``_mm_2d``: ttnn's auto
    choice was a 32-core 1D config at 13 us, the 2D 8x9 grid does it on 72 cores in 5 us, and the
    bias folds into the matmul epilogue instead of a separate ``BinaryNg`` (-3 us more): 2201 ->
    2190 us. See ``_mm_2d`` for the config sweep and ``INIT_KERNEL_CONFIG`` for why this one op
    wants fp32 dest accumulation once its program config is pinned.
  * The two group-norm EXPAND matmuls ([1,1024] x [1024,1024], 12 per pass) have a pinned config too,
    ``self._gn_mm``. This one is a gemv whose cost is purely the 2 MB DRAM read of E, and the lever is
    ``in0_block_w``, not cores: 16 -> 10 us each, 217 GB/s instead of 130 (42% of DRAM peak, up from
    25%). Biggest single win in the pass: 2190 -> 2103 us, -4.0%, and bit-identical numerically.
  * ACTIVATIONS in L1, WEIGHTS in DRAM — no exceptions. Activations stay on-chip end to end so the
    matmuls read input-0 from L1 instead of round-tripping to DRAM; every constant operand, trained
    or generated, stays in DRAM in the input-1 slot, where the matmul's own prefetch covers the
    read. Both L1 weight pins this file used to have were measured and dropped: see ``_mm_2d``
    (init weight, ~1 us) and ``_gn_expand`` (2 MB, no measurable difference at all).
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_conditioning import (
    NUM_ATTN_HEADS,
    NUM_LATENTS,
)

GROUP_NORM_GROUPS = 32
GROUP_NORM_EPS = 1e-5
ENC_HEAD_DIM = HIDDEN_SIZE // NUM_ATTN_HEADS  # 64
PERCEIVER_HEADS = 8
PERCEIVER_HEAD_DIM = 64
PERCEIVER_DEPTH = 2
PERCEIVER_INNER = PERCEIVER_HEADS * PERCEIVER_HEAD_DIM  # 512

L1 = ttnn.L1_MEMORY_CONFIG

# HiFi4 for every matmul in this path. All of them are BANDWIDTH-bound, not FLOP-bound (the perf
# report puts them at 22-74% of DRAM peak but only 3-14% of FLOP peak), which is exactly the case
# where the extra math passes are almost free: measured +0.3 ms/pass eager. It buys real accuracy —
# over a 10-point sweep of mel lengths, mean PCC vs the fp32 reference 0.9954 -> 0.9978 and the
# WORST case 0.9898 -> 0.9953. At HiFi2 the worst case sits below the 0.99 test gate, and a
# conditioning prompt that drifts changes the voice the GPT then generates, so this is not a
# free-accuracy-for-nothing tweak — it is load-bearing.
# NOTE: fp32_dest_acc_en is deliberately OFF here. It halves the tiles per math pass on these large
# matmuls (+0.4 ms) and measured WORSE than plain HiFi4 (mean 0.9973 vs 0.9978).
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# The group-norm STATISTICS additionally accumulate in fp32. The reference GroupNorm32 deliberately
# computes in fp32 (``super().forward(x.float())``), and these are long reductions — over seq for the
# per-channel stats, then over the 32 channels of a group — where bf16 accumulation was the accuracy
# floor of the whole path (it cost ~0.01 PCC on one sample). The tensors are [1, 1024, 1], so fp32
# dest accumulation is free here, unlike on the big matmuls above.
STATS_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# The INIT matmul gets its own config, and unlike the big matmuls above it DOES want fp32 dest
# accumulation. Pinning its program config (see ``_mm_2d``) changes the K-accumulation/subblock
# rounding, and with plain bf16 dest that cost real end-to-end accuracy: over an 8-point mel sweep
# the worst-case latents PCC fell 0.9938 -> 0.9929. fp32 dest acc on this one op buys it back and
# then some (worst case 0.9939, i.e. better than before the program config was pinned) for ~1 us,
# because K here is only 3 tiles so there is very little to re-run at half the tiles per pass.
# NOTE: fp32 dest acc is NOT a free win in general here — with the AUTO program config it measured
# worst case 0.9884, BELOW the 0.99 gate. It is specifically the pinned config that wants it.
INIT_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _lin(torch_tensor, device):
    """torch [out, in] (or conv [out, in, 1]) -> ttnn linear weight [in, out] on device (DRAM)."""
    w = torch_tensor
    if w.dim() == 3:  # conv1d kernel-1 -> [out, in]
        w = w.squeeze(-1)
    return ttnn.from_torch(
        w.t().contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )


def _vec(torch_tensor, device):
    """torch [n] -> ttnn tile [n] on device (bias / affine params)."""
    return ttnn.from_torch(torch_tensor.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _row(torch_tensor, device):
    """torch [n] -> ttnn ``[1, 1, n]`` on device: a per-CHANNEL parameter for the group-norm's
    ``[1, s, 1024]`` (channels-last) layout, broadcast over seq on dim 1."""
    return ttnn.from_torch(
        torch_tensor.reshape(1, 1, -1).to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def _perm_qkv_out(t):
    """Reorder the output channels of a ConditioningEncoder qkv weight/bias from the checkpoint's
    per-head-interleaved layout into the ``[q_all | k_all | v_all]`` head-major layout that
    ``nlp_create_qkv_heads`` expects.

    The reference ``QKVAttention`` reads channel ``h*192 + t*64 + d`` as (head h, t in {q,k,v},
    dim d); the fused op wants ``t*1024 + h*64 + d``. ``t`` is the leading axis after the permute,
    so indexing the first (output-channel) dim with it does the whole relabel on host, once."""
    idx = torch.arange(3 * HIDDEN_SIZE).reshape(NUM_ATTN_HEADS, 3, ENC_HEAD_DIM).permute(1, 0, 2).reshape(-1)
    return t[idx]


def _mm_2d(grid, mt, kt, nt, gx=8):
    """2D (block) multicast matmul program config for an ``[Mt, Kt] x [Kt, Nt]`` tile shape.

    Why pin one at all: for the tiny INIT matmul (Mt=9, Kt=3, Nt=32) ttnn's auto-selection picks a
    1D width-multicast config on 32 cores; in the model that is 13 us, and this 2D 8x9 grid (72
    cores, one 32-row stripe of M per core row, 4 N-tiles per core column) is 4 us. K is only 3
    tiles, so per-core cost is dominated by moving the [288, 1024] OUTPUT rather than by math —
    spreading the output over more cores is what buys the time, not extra FLOPs.

    Swept in isolation (traced repeat harness, so the absolute numbers run high vs the in-model
    profiler, but the ordering held): auto 19.5, 1D configs 17-56 (best at its own 32 cores), 2D
    4x9 9.6, 2D 11x9 8.1, 2D 13x9 7.8, 2D 8x9 7.6. ``gx=8`` wins because it DIVIDES Nt=32; 11 and
    13 leave their last core column partly idle. Also measured and NOT used: HiFi2 saves ~0.3 us
    (not worth any accuracy on a path where PCC is load-bearing), a DRAM output costs +3.6, in0
    L1-height-sharded reaches 6.1 but only via a single-COLUMN shard grid the upstream ``permute``
    does not produce, so it needs an extra reshard that costs more than the 0.7 us it saves, and
    in0 block-sharded (13.1) / width-sharded weight (7.6) both lost to plain L1 interleaved.
    Holding the WEIGHT in L1 measured 6.8 vs 7.6 and is bit-identical numerically, but weights
    belong in DRAM, so it is not used.

    ``gy`` is chosen per call from the actual mel length: the conditioning module is built once but
    sees several sequence lengths, and per_core_M must cover Mt."""
    gy = max(1, min(mt, grid.y))
    per_core_m, per_core_n = -(-mt // gy), -(-nt // gx)
    # out_subblock_h * out_subblock_w <= 8 (dest register budget), each dividing its per_core dim.
    sub_w = max(w for w in range(1, per_core_n + 1) if per_core_n % w == 0 and w <= 8)
    sub_h = max(h for h in range(1, per_core_m + 1) if per_core_m % h == 0 and h * sub_w <= 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=kt,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


class TtXttsConditioning(LightweightModule):
    def __init__(self, state_dict, device):
        super().__init__()
        self.device = device
        e = "gpt.conditioning_encoder."
        p = "gpt.conditioning_perceiver."

        # Block-diagonal group-averaging matrix E [1024, 1024] (E[c,c'] = 1/cpg iff channels c,c'
        # share a group) used by _group_norm to reduce per-group WITHOUT a reshape to [1,32,32s]
        # (that reshape needed ROW_MAJOR<->TILE conversions = Tilize/Untilize ops every block).
        # It lives in DRAM. It used to be pinned in L1, because back when the group norm ran in the
        # channels-first layout the product was ``E @ cmean``, i.e. E was input-0 (the activation
        # slot), and from DRAM those matmuls only reached ~25% of peak bandwidth. The transpose-free
        # group norm reversed the product to ``cmean @ E``, so E is now input-1 — the weight slot,
        # where the matmul's own prefetch covers the read. Measured either way, the 12 expand
        # matmuls are 16 us each and the pass is 2191 (L1) vs 2190 us (DRAM), so the 2 MB of L1 was
        # buying nothing and is given back to the activations.
        cpg = HIDDEN_SIZE // GROUP_NORM_GROUPS
        e_mat = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE)
        for gi in range(GROUP_NORM_GROUPS):
            e_mat[gi * cpg : (gi + 1) * cpg, gi * cpg : (gi + 1) * cpg] = 1.0 / cpg
        self._gn_expand = ttnn.from_torch(
            e_mat.reshape(1, HIDDEN_SIZE, HIDDEN_SIZE).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        # Program config for the two expand matmuls. Unlike the init matmul this shape never varies
        # with the mel length (always [1, 1024] x [1024, 1024]), so it is built once here.
        #
        # It is a gemv whose cost IS the 2 MB DRAM read of E, and the lever is ``in0_block_w`` — how
        # many K tiles are staged per pass — NOT the core count. Swept with E in DRAM: at ibw=1 every
        # core count lands on ~29 us, at ibw=4 ~10.7 us, at ibw=8 ~9.1 us, then it degrades again
        # (ibw=16 ~10.4, ibw=32 ~12-16). Core count barely registers by comparison: at ibw=8, 32 /
        # 64 / 130 cores are 9.6 / 9.5 / 9.0 us. ttnn's auto choice is ibw=1-equivalent at 16.6 us.
        # 32 cores is chosen because N is 32 tiles, so per_core_N=1 gives every core exactly one
        # output tile with nothing left over.
        self._gn_mm = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),  # 32 cores = 32 output tiles
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # --- ConditioningEncoder ---
        self.init_w = _lin(state_dict[e + "init.weight"], device)  # [80 -> 1024]
        self.init_b = _vec(state_dict[e + "init.bias"], device)
        self._grid = device.compute_with_storage_grid_size()

        self.blocks = []
        i = 0
        while (e + f"attn.{i}.qkv.weight") in state_dict:
            self.blocks.append(
                {
                    # group-norm affine as [1, 1, 1024] rows, broadcast over seq on dim 1.
                    "gn_w": _row(state_dict[e + f"attn.{i}.norm.weight"], device),
                    "gn_b": _row(state_dict[e + f"attn.{i}.norm.bias"], device),
                    # qkv output channels relabelled to [q|k|v] head-major for nlp_create_qkv_heads.
                    "qkv_w": _lin(_perm_qkv_out(state_dict[e + f"attn.{i}.qkv.weight"]), device),  # [1024 -> 3072]
                    "qkv_b": _vec(_perm_qkv_out(state_dict[e + f"attn.{i}.qkv.bias"]), device),
                    "proj_w": _lin(state_dict[e + f"attn.{i}.proj_out.weight"], device),  # [1024 -> 1024]
                    "proj_b": _vec(state_dict[e + f"attn.{i}.proj_out.bias"], device),
                }
            )
            i += 1

        # --- PerceiverResampler ---
        # Stored pre-shaped [1, 32, 1024] (host reshape) so the forward never reshapes a weight.
        self.latents = ttnn.from_torch(
            state_dict[p + "latents"].reshape(1, NUM_LATENTS, HIDDEN_SIZE).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.layers = []
        for j in range(PERCEIVER_DEPTH):
            # to_q [512, 1024] and to_kv [1024, 1024] fused into ONE [1024 -> 1536] weight whose
            # output blocks are [q | k | v] head-major (to_kv's own halves are already [k | v]) —
            # one matmul over [latents ; context] + one nlp_create_qkv_heads for all three.
            qkv = torch.cat([state_dict[p + f"layers.{j}.0.to_q.weight"], state_dict[p + f"layers.{j}.0.to_kv.weight"]])
            # GEGLU value/gate halves split on host: the fused [1, 32, 5460] output could only be
            # halved with an offset-2730 (non-tile-aligned) slice, which untilizes the tensor.
            ff0_w, ff0_b = state_dict[p + f"layers.{j}.1.0.weight"], state_dict[p + f"layers.{j}.1.0.bias"]
            inner = ff0_w.shape[0] // 2  # 2730
            self.layers.append(
                {
                    "qkv_w": _lin(qkv, device),  # [1024 -> 1536]
                    "to_out": _lin(state_dict[p + f"layers.{j}.0.to_out.weight"], device),  # [512 -> 1024]
                    "ff_val_w": _lin(ff0_w[:inner], device),  # [1024 -> 2730] (GEGLU value half)
                    "ff_val_b": _vec(ff0_b[:inner], device),
                    "ff_gate_w": _lin(ff0_w[inner:], device),  # [1024 -> 2730] (GEGLU gate half)
                    "ff_gate_b": _vec(ff0_b[inner:], device),
                    "ff2_w": _lin(state_dict[p + f"layers.{j}.1.2.weight"], device),  # [2730 -> 1024]
                    "ff2_b": _vec(state_dict[p + f"layers.{j}.1.2.bias"], device),
                }
            )
        self.perc_norm_gamma = _vec(state_dict[p + "norm.gamma"], device)

    # ------------------------------------------------------------------ #
    def _group_norm(self, x, gamma_row, beta_row):
        """GroupNorm(32, 1024) over (channels-in-group, seq). x: [1, s, 1024] -> [1, s, 1024]. Consumes x.

        Reshape-FREE: a full group mean/var is order-independent, so per-group stats == the group
        average of per-channel stats. Compute the per-channel mean over seq, expand to per-group via
        a matmul with the block-diagonal averaging matrix ``self._gn_expand``, and likewise for the
        (centered) variance. Everything stays TILE, so this avoids the old reshape-to-[1,32,32s]
        round trip and its four Tilize/Untilize ops.

        TRANSPOSE-FREE too: it runs in the module's native ``[1, s, 1024]`` layout. Reducing over seq
        there is ``mean(dim=-2)``, which ttnn lowers to a native ``ReduceOpDim::H`` kernel — H and W
        are the two dims ``reduce_impl`` handles without injecting a ``transpose`` — so there is no
        reason to permute to channels-first (``[1, 1024, s]``) just to make seq the last dim. Two
        ``permute``s per block, 12 per pass, dropped for free. The group-averaging matmul flips
        accordingly (``cmean @ E`` instead of ``E @ cmean``), which is exact: E is symmetric.

        gamma/beta are ``[1, 1, 1024]`` rows broadcast over seq on dim 1."""
        cmean = ttnn.mean(
            x, dim=-2, keepdim=True, compute_kernel_config=STATS_KERNEL_CONFIG
        )  # [1, 1, 1024] per-channel mean over seq
        mu = ttnn.matmul(
            cmean,
            self._gn_expand,
            memory_config=L1,
            compute_kernel_config=STATS_KERNEL_CONFIG,
            program_config=self._gn_mm,
        )  # group mean, expanded per channel
        ttnn.deallocate(cmean)
        xc = ttnn.subtract(x, mu, memory_config=L1)  # center by group mean (stable variance)
        ttnn.deallocate(mu)
        sq = ttnn.multiply(xc, xc, memory_config=L1)
        cvar = ttnn.mean(
            sq, dim=-2, keepdim=True, compute_kernel_config=STATS_KERNEL_CONFIG
        )  # [1, 1, 1024] per-channel var
        ttnn.deallocate(sq)
        var = ttnn.matmul(
            cvar,
            self._gn_expand,
            memory_config=L1,
            compute_kernel_config=STATS_KERNEL_CONFIG,
            program_config=self._gn_mm,
        )  # [1, 1, 1024] group variance
        ttnn.deallocate(cvar)
        # NOTE: gamma is applied to the ACTIVATION, not folded into this [1, 1, 1024] scale. Folding it
        # (scale = gamma * rsqrt(var+eps)) saves one full-size eltwise op (~48 us/pass) but rounds the
        # per-channel product to bf16 ONCE, so every position of that channel gets the SAME wrong
        # scale — a coherent distortion, where applying gamma elementwise leaves incoherent noise that
        # partly cancels. Measured over 7 real mels: folded min PCC 0.9886 (one input BELOW the 0.99
        # gate) vs 0.9906 unfolded. Not worth 2% of the pass.
        rs = ttnn.rsqrt(ttnn.add(var, GROUP_NORM_EPS), memory_config=L1)
        ttnn.deallocate(var)
        y = ttnn.multiply(xc, rs, memory_config=L1)
        ttnn.deallocate(xc)
        ttnn.deallocate(rs)
        ttnn.multiply(y, gamma_row, memory_config=L1, output_tensor=y)  # gamma/beta bcast over seq
        ttnn.add(y, beta_row, memory_config=L1, output_tensor=y)
        return y

    def _attn_block(self, x, blk):
        """One ConditioningEncoder AttentionBlock: y = gn(x); y + proj(attn(qkv(y))). Consumes x."""
        y = self._group_norm(x, blk["gn_w"], blk["gn_b"])  # consumes x
        qkv = ttnn.linear(
            y, blk["qkv_w"], bias=blk["qkv_b"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, s, 3072] = [q|k|v]
        b, s, _ = qkv.shape
        # Leading-singleton reshape only (a metadata view — nlp_create_qkv_heads wants rank 4);
        # the heads split itself is the fused op, no last-dim reshape / slices / permutes.
        qkv = ttnn.reshape(qkv, (b, 1, s, 3 * HIDDEN_SIZE))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NUM_ATTN_HEADS, transpose_k_heads=False, memory_config=L1
        )  # each [1, heads, s, head_dim]
        ttnn.deallocate(qkv)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, memory_config=L1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)  # [1, s, 1024] (fused permute + merge)
        ttnn.deallocate(attn)
        h = ttnn.linear(
            out, blk["proj_w"], bias=blk["proj_b"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )
        ttnn.deallocate(out)
        res = ttnn.add(y, h, memory_config=L1)  # residual is on the NORMED y (matches the reference)
        ttnn.deallocate(y)
        ttnn.deallocate(h)
        return res

    def _perceiver_attn(self, latents, context, layer):
        """Cross-attention: latents attend to [latents ; context].

        One fused matmul over the concatenated sequence gives [q|k|v] for every row, so the heads
        split is a single ``nlp_create_qkv_heads``; the latents' Q is rows 0..NUM_LATENTS of that
        result — a tile-aligned row slice (NUM_LATENTS is 32), not a data shuffle."""
        ctx = ttnn.concat([latents, context], dim=1, memory_config=L1)  # [1, 32+s, 1024]
        qkv = ttnn.linear(
            ctx, layer["qkv_w"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, 32+s, 1536]
        ttnn.deallocate(ctx)
        n = qkv.shape[1]
        qkv = ttnn.reshape(qkv, (1, 1, n, 3 * PERCEIVER_INNER))  # leading-singleton view
        q_all, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=PERCEIVER_HEADS, transpose_k_heads=False, memory_config=L1
        )  # each [1, 8, 32+s, 64]
        ttnn.deallocate(qkv)
        q = ttnn.slice(
            q_all, [0, 0, 0, 0], [1, PERCEIVER_HEADS, NUM_LATENTS, PERCEIVER_HEAD_DIM], memory_config=L1
        )  # the latents' rows
        ttnn.deallocate(q_all)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, memory_config=L1)  # [1,8,32,64]
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)  # [1, 32, 512]
        ttnn.deallocate(attn)
        proj = ttnn.linear(
            out, layer["to_out"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, 32, 1024]
        ttnn.deallocate(out)
        return proj

    def _perceiver_ff(self, x, layer):
        """GEGLU feed-forward. The value/gate halves are separate weights (see __init__), so there is
        no non-tile-aligned half-slice, and the gate's GELU rides the matmul epilogue."""
        val = ttnn.linear(
            x, layer["ff_val_w"], bias=layer["ff_val_b"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, 32, 2730]
        gate = ttnn.linear(
            x,
            layer["ff_gate_w"],
            bias=layer["ff_gate_b"],
            activation="gelu",
            memory_config=L1,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
        )  # exact erf GELU, fused
        h = ttnn.multiply(gate, val, memory_config=L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(val)
        out = ttnn.linear(
            h, layer["ff2_w"], bias=layer["ff2_b"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, 32, 1024]
        ttnn.deallocate(h)
        return out

    # ------------------------------------------------------------------ #
    def mel_to_device(self, mel):
        """Host log-mel ``[1, 80, s]`` -> device bf16 TILE tensor (the ``from_torch`` host->device
        write, kept OUTSIDE any trace capture — writes are fatal inside a trace)."""
        return ttnn.from_torch(
            mel.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16, memory_config=L1
        )

    def forward(self, mel):
        """mel: torch tensor ``[1, 80, s]`` -> conditioning latents ttnn ``[1, 1024, 32]``."""
        return self.forward_dev(self.mel_to_device(mel))

    def forward_dev(self, mel_tt):
        """Trace-compatible: ``mel_tt`` is an already-on-device ``[1, 80, s]`` bf16 tensor (no
        host->device write here), so this can run inside a captured trace. -> ttnn ``[1, 1024, 32]``.
        ``mel_tt`` is the caller's tensor and is left allocated."""
        x = ttnn.permute(mel_tt, (0, 2, 1), memory_config=L1)  # [1, s, 80]
        s = x.shape[1]
        h = ttnn.linear(
            x,
            self.init_w,
            bias=self.init_b,
            memory_config=L1,
            compute_kernel_config=INIT_KERNEL_CONFIG,
            program_config=_mm_2d(self._grid, -(-s // 32), -(-x.shape[2] // 32), HIDDEN_SIZE // 32),
        )  # [1, s, 1024]
        ttnn.deallocate(x)
        x = h

        for blk in self.blocks:
            x = self._attn_block(x, blk)  # consumes x; ConditioningEncoder output [1, s, 1024]

        # PerceiverResampler (self.latents is stored pre-shaped [1, 32, 1024] — never freed here)
        latents = self.latents
        for layer in self.layers:
            attn = self._perceiver_attn(latents, x, layer)
            latents = ttnn.add(attn, latents, memory_config=L1)
            ttnn.deallocate(attn)
            ff = self._perceiver_ff(latents, layer)
            nxt = ttnn.add(ff, latents, memory_config=L1)
            ttnn.deallocate(ff)
            ttnn.deallocate(latents)
            latents = nxt
        ttnn.deallocate(x)
        normed = ttnn.rms_norm(latents, weight=self.perc_norm_gamma, epsilon=1e-12, memory_config=L1)  # [1, 32, 1024]
        ttnn.deallocate(latents)
        out = ttnn.permute(normed, (0, 2, 1), memory_config=L1)  # [1, 1024, 32]
        ttnn.deallocate(normed)
        return out
