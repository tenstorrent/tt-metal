# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini attention, batch-1 decode on a single device.

Covers both layer types of the HF reference (`Inkling2SlidingAttention`,
`Inkling2FullAttention`): k/v short convolution (static for local layers,
static + per-token dynamic taps for global layers), q/k L2 clip, unweighted v
RMS norm, hidden-conditioned relative-position bias, per-token attention sinks
and the sigmoid-gated output projection.

Caches
- local: one ring of `WINDOW` slots, slot = pos % WINDOW.
- global: the same ring (it carries the biased near part) plus a full cache
  `[1, 8, max_ctx, 128]` read for keys at distance >= WINDOW (no bias).
Logits are `near ++ far ++ sink`, one softmax.

Every position-dependent value lives in `DecodePositions` (device tensors that
the host rewrites before each step), so one trace serves every step and every
layer of the model.

Weights come as host tensors in the weight-cache contract (design.md §4).
"""

import math
import os

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear

HIDDEN = 2304
N_HEADS = 24
N_KV = 8
HEAD_DIM = 128
GROUPS = N_HEADS // N_KV
D_REL = 16
WINDOW = 1024  # sliding window == rel_extent
TAPS = 4
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0
EPS = 1e-6
LOG_ALPHA = 0.1
LOG_FLOOR = 8192
NEG = -1.0e9

Q_COLS = (0, 3072)
GATE_COLS = (3072, 6144)
REL_COLS = (6144, 6528)
SINK_COLS = (6528, 6552)
KV0 = 6560  # first k column of the fused projection


# Conv window rows. single: q = k = v input (local layers outside the loop); triple: separate q/k/v inputs.
# "k"/"v": row of the input at lag 0 (lags 1..3 follow); "new": rows the step's inputs land in.
ROWS = {
    "single": {"k": 0, "v": 0, "new": (0,)},
    "triple": {"k": 1, "v": 5, "new": (0, 1, 5)},
}


def log_scale(pos):
    """tau(pos) of `Inkling2LogScale`."""
    return 1.0 + LOG_ALPHA * math.log(max((pos + 1) / LOG_FLOOR, 1.0))


class DecodePositions:
    """Per-step position inputs, shared by every attention layer.

    `update(pos)` rewrites the host-side copies into persistent device tensors
    (trace safe). `prepare()` derives the masks on device; call it once per step
    inside the trace, before the layers.
    """

    def __init__(self, device, max_ctx):
        self.device = device
        self.max_ctx = max_ctx
        host = self._host(0)
        self.inputs = {k: ttnn.to_device(v, device) for k, v in host.items()}
        kpos = torch.arange(max_ctx, dtype=torch.float32).view(1, 1, 1, max_ctx)
        self.kpos = ttnn.from_torch(kpos, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _host(self, pos):
        ring = pos % WINDOW
        slots = torch.arange(WINDOW)
        dist = (ring - slots) % WINDOW  # distance of the token held by each ring slot
        scal = torch.tensor([float(pos), log_scale(pos)], dtype=torch.float32)
        return {
            "pos": ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32),
            "ring": ttnn.from_torch(torch.tensor([ring], dtype=torch.int32), dtype=ttnn.int32),
            "dist_idx": ttnn.from_torch(dist.view(1, WINDOW).to(torch.int32), dtype=ttnn.uint32),
            "dist": ttnn.from_torch(dist.view(1, 1, 1, WINDOW).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
            "pos_f": ttnn.from_torch(scal[0].view(1, 1, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
            "tau": ttnn.from_torch(scal[1].view(1, 1, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        }

    def update(self, pos):
        if not 0 <= pos < self.max_ctx:
            raise ValueError(f"pos {pos} outside [0, {self.max_ctx})")
        for name, host in self._host(pos).items():
            ttnn.copy_host_to_device_tensor(host, self.inputs[name])

    def prepare(self):
        inp = self.inputs
        # ring slot is valid iff its distance <= pos (slot already written)
        self.near_mask = ttnn.multiply(ttnn.gt(inp["dist"], inp["pos_f"]), NEG)
        # far keys: positions <= pos - WINDOW
        far_limit = ttnn.subtract(inp["pos_f"], float(WINDOW))
        self.far_mask = ttnn.multiply(ttnn.gt(self.kpos, far_limit), NEG)
        self.tau = inp["tau"]
        self.dist_idx = inp["dist_idx"]
        self.pos = inp["pos"]
        self.ring = inp["ring"]


# Projections (w_all, w_o) as one bank-streamed program each, bit-equal to the DRAM-sharded matmul;
# INKLING_PROJ_BANK=0: the ttnn.linear path (A/B runs).
PROJ_BANK = os.environ.get("INKLING_PROJ_BANK", "1") != "0"


class _DramShardedLinear:
    """Decode linear [32, K] x [K, N] with the weight width-sharded over the DRAM banks (one row tile of
    activations). Two pipelined readers per bank: 2304x6560 70 -> 45 us, 3072x2304 70 -> 24 us (traced)."""

    CORES = 8  # activation shard grid; K must split into whole tiles over it
    READERS = 2

    def __init__(self, device, w, dtype, trim=False):
        """trim: slice the output back to N when N is padded to whole bank shards (callers that use all of N)."""
        k, n = w.shape
        banks = device.dram_grid_size().x
        bank_w = math.ceil(n / (32 * banks)) * 32
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, (k, bank_w), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.n = n if trim and bank_w * banks != n else None
        self.bank = None
        if PROJ_BANK:  # same output bit for bit, one bank-streamed program (tt/fused/bank_linear.py), DRAM out
            ckc = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True,
            )  # fmt: skip
            self.bank = BankLinear(device, w, dtype, bank_w * banks, kb=2, cores_per_bank=self.READERS,
                                   compute_kernel_config=ckc, out_cols=n if trim else bank_w * banks,
                                   out_dtype=ttnn.bfloat16)  # fmt: skip
            self.bank_ckc = ckc
        else:
            w = torch.nn.functional.pad(w, (0, bank_w * banks - n)).view(1, 1, k, -1)
            self.weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        assert k % (32 * self.CORES) == 0, k
        self.act_mem = ttnn.create_sharded_memory_config(
            (32, k // self.CORES),
            ttnn.CoreGrid(x=self.CORES, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=2,
            per_core_M=1,
            per_core_N=math.ceil(bank_w * banks / (32 * self.CORES)),
            fused_activation=None,
        )
        self.cfg.num_workers_per_dram_bank = self.READERS
        self.cfg.use_multicore_pipeline = True

    def __call__(self, x, dtype, ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        if self.bank is not None and dtype == ttnn.bfloat16 and memory_config == ttnn.DRAM_MEMORY_CONFIG:
            assert ckc.math_fidelity == self.bank_ckc.math_fidelity and ckc.fp32_dest_acc_en, ckc
            assert x.shape[-2] <= 32, x.shape  # one tile row (decode)
            return self.bank(x)
        assert self.bank is None, "bank-streamed projection: bf16 DRAM output only"
        y = ttnn.linear(
            x if x.is_sharded() else ttnn.to_memory_config(x, self.act_mem),  # sharded: already in act_mem
            self.weight,
            program_config=self.cfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=dtype,
            compute_kernel_config=ckc,
        )
        y = ttnn.to_memory_config(y, memory_config)
        if self.n is None:
            return y
        return ttnn.slice(y, [0, 0, 0, 0], [*list(y.shape)[:-1], self.n], memory_config=memory_config)


class AttentionState:
    """Device state of one (layer, pass): caches and conv histories."""

    def __init__(self, device, is_global, max_ctx, cache_dtype):
        def zeros(shape, dtype):
            return ttnn.from_torch(torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.k_ring = zeros([1, N_KV, WINDOW, HEAD_DIM], cache_dtype)
        self.v_ring = zeros([1, N_KV, WINDOW, HEAD_DIM], cache_dtype)
        self.k_full = self.v_full = None
        if is_global:
            self.k_full = zeros([1, N_KV, max_ctx, HEAD_DIM], cache_dtype)
            self.v_full = zeros([1, N_KV, max_ctx, HEAD_DIM], cache_dtype)
        # k/v conv window as projection INPUTS, one [32, 2304] bf16 tile row (layout: Inkling2Attention.ROWS).
        # The projection runs on the whole window, so the conv is one multiply + row sum.
        # k_hist / v_hist keep the names model.reset() zeroes.
        self.x_hist = zeros([1, 1, 32, HIDDEN], ttnn.bfloat16)
        self.k_hist = [self.x_hist]
        self.v_hist = []

    def tensors(self):
        out = [self.k_ring, self.v_ring, self.x_hist]
        if self.k_full is not None:
            out += [self.k_full, self.v_full]
        return out


class Inkling2Attention:
    """One attention layer. `weights` maps §4 names (without `attn.`) to host tensors."""

    def __init__(self, device, weights, is_global, prefetcher=None):
        """prefetcher: optional tt.prefetch.PrefetchedLinear; the fused q/k/v projection then streams through it."""
        self.device = device
        self.is_global = is_global
        dram = ttnn.DRAM_MEMORY_CONFIG

        def torch_of(name):
            return ttnn.to_torch(weights[name]).float()

        def put(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=dram)

        self.w_o = _DramShardedLinear(device, torch_of("w_o")[0, 0], weights["w_o"].dtype, trim=True)
        self.q_scale = put(torch_of("q_clip_scale").view(1, 1, N_HEADS, HEAD_DIM))
        self.rel_scale = put(torch_of("rel_clip_scale").view(1, 1, 1, N_HEADS * D_REL))
        # clip reductions (see _clip_factor): [128, 1] over head_dim; [384, 32] per 16-wide rel head (cols >= 24
        # stay 0 -> factor 1, killed by rel_bcast's zero rows)
        self.qk_red = put(torch.full((1, 1, HEAD_DIM, 1), QK_CLIP**-2), dtype=ttnn.bfloat16)
        self.q_red = self.qk_red  # q's clip reduction (a subclass may fold a q pre-scale into q_scale and this)
        grp = torch.zeros(N_HEADS * D_REL, 32)
        for h in range(N_HEADS):
            grp[h * D_REL : (h + 1) * D_REL, h] = 1.0
        self.rel_red = put((grp * REL_CLIP**-2).view(1, 1, -1, 32), dtype=ttnn.bfloat16)
        self.rel_bcast = put(grp.T.contiguous().view(1, 1, 32, -1), dtype=ttnn.bfloat16)
        # relative profiles as an embedding table: row d = proj[:, d]
        proj_t = torch_of("rel_proj").view(D_REL, WINDOW).T.contiguous()
        self.rel_table = put(proj_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # One projection [qgrs | k | v (| k_dyn | v_dyn)] on the whole conv window (rows: ROWS). Pieces start
        # on 32-column boundaries, so re-packing the BFP8 tiles is exact.
        width = N_KV * HEAD_DIM
        w_k, w_v = torch_of("w_k")[0, 0], torch_of("w_v")[0, 0]
        pieces = [torch_of("w_qgrs")[0, 0], w_k[:, :width], w_v[:, :width]]
        if is_global:
            pieces += [w_k[:, width:], w_v[:, width:]]
        w_all = torch.cat(pieces, dim=-1)
        self.n_kv = w_all.shape[-1] - KV0
        if prefetcher is not None:  # E8: -0.11 ms @4K for w_all; w_o through it was slower (+0.27), so not used
            self.w_all = prefetcher.linear(w_all)
        else:
            self.w_all = _DramShardedLinear(device, w_all, weights["w_qgrs"].dtype)

        # conv taps per (window row, channel), fp32. The residual and 1/sqrt(K+1) (local) or 1/sqrt(2K+1)
        # (global, static + dynamic taps) are folded in.
        self.conv_scale = 1.0 / math.sqrt((2 if is_global else 1) * TAPS + 1)
        taps = {}
        for kv in ("k", "v"):
            w = torch_of(f"{kv}_conv").view(TAPS, width).clone()
            w[0] += 1.0
            taps[kv] = w * self.conv_scale
        self.modes = {}
        for mode in ("triple",) if is_global else ("single", "triple"):
            rows = ROWS[mode]
            t = torch.zeros(32, self.n_kv)
            t[rows["k"] : rows["k"] + TAPS, :width] = taps["k"]
            t[rows["v"] : rows["v"] + TAPS, width : 2 * width] = taps["v"]
            shift = torch.zeros(32, 32)  # window <- shifted old window + one-hot rows of the new inputs
            for r0 in {rows["k"], rows["v"]}:
                for lag in range(TAPS - 1):
                    shift[r0 + lag + 1, r0 + lag] = 1.0
            onehots = []
            for r in rows["new"]:
                m = torch.zeros(1, 1, 32, HIDDEN)
                m[..., r, :] = 1.0
                onehots.append(put(m, dtype=ttnn.bfloat16))
            self.modes[mode] = {
                "taps": put(t.view(1, 1, 32, -1)),
                "shift": put(shift.view(1, 1, 32, 32), dtype=ttnn.bfloat16),
                "onehots": onehots,
            }
        if is_global:
            # dynamic taps: dyn[kv][h*TAPS + lag] (row ROWS k/v of the projection) scales head h at lag
            rows = ROWS["triple"]
            bcast = torch.zeros(32, 32)  # copy row k / v to its 4 lag rows
            lagsel = torch.zeros(32, 2 * N_KV * TAPS)
            expand = torch.zeros(2 * N_KV * TAPS, self.n_kv)
            for b, kv in enumerate(("k", "v")):
                for lag in range(TAPS):
                    bcast[rows[kv] + lag, rows[kv]] = 1.0
                    for h in range(N_KV):
                        j = b * N_KV * TAPS + h * TAPS + lag
                        lagsel[rows[kv] + lag, j] = self.conv_scale
                        c0 = b * width + h * HEAD_DIM
                        expand[j, c0 : c0 + HEAD_DIM] = 1.0
            self.dyn_bcast = put(bcast.view(1, 1, 32, 32), dtype=ttnn.bfloat16)
            self.dyn_lagsel = put(lagsel.view(1, 1, 32, -1))
            self.dyn_expand = put(expand.view(1, 1, 2 * N_KV * TAPS, -1), dtype=ttnn.bfloat16)
        # sink column block: col 0 = sink logit, others masked
        sink_pad = torch.full((1, N_KV, GROUPS, 32), NEG)
        sink_pad[..., 0] = 0.0
        self.sink_pad = put(sink_pad)

        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.hifi4 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, HEAD_DIM),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def new_state(self, max_ctx, cache_dtype=ttnn.bfloat16):
        return AttentionState(self.device, self.is_global, max_ctx, cache_dtype)

    # ---------------------------------------------------------------- pieces
    def _clip_factor(self, x, red, tau):
        """L2 clip factor min(t / |x|, 1) (* tau) as rsqrt(max(|x|^2 / t^2, 1)); `red` sums groups of x^2
        with 1/t^2 folded in (t^2 = 512 / 64, exact). A matmul instead of sum: no fill-pad + reduce.
        HF adds 1e-6 to |x|; that changes the factor by < 1e-6 / t relative."""
        s = ttnn.matmul(ttnn.square(x), red, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
        f = ttnn.rsqrt(ttnn.clamp(s, min=1.0))
        return f if tau is None else ttnn.multiply(f, tau)

    def _head_clip(self, x, tau=None, red=None):
        """QK_CLIP L2 clip over head_dim of [1,1,heads,128]; returns bf16."""
        return ttnn.multiply(x, self._clip_factor(x, self.qk_red if red is None else red, tau), dtype=ttnn.bfloat16)

    def _window(self, state, xq, xk, xv):
        """Shift the step's inputs into the conv window and project it. Returns (projection [1,1,32,N] bf16
        (row 0 = q input), mode)."""
        mode = "single" if (xk is xq and xv is xq) else "triple"
        m = self.modes[mode]
        new = [xq] if mode == "single" else [xq, xk, xv]
        # [1,W] inputs broadcast over rows times a one-hot row mask: no untilize / concat / tilize
        win = ttnn.matmul(m["shift"], state.x_hist, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)
        for i, (x, onehot) in enumerate(zip(new, m["onehots"])):
            last = i == len(new) - 1
            win = ttnn.addcmul(win, x, onehot, value=1.0, output_tensor=state.x_hist if last else None)
        return self.w_all(state.x_hist, ttnn.bfloat16, self.mm_cfg), mode

    def _window_project(self, state, xq, xk, xv):
        """Shift the step's inputs into the conv window, project it, apply the k/v conv.
        Returns (projection [1,1,32,N] bf16 (row 0 = q input), k conv [1,1,1,1024], v conv) fp32."""
        proj, mode = self._window(state, xq, xk, xv)
        m = self.modes[mode]
        p_kv = ttnn.slice(proj, [0, 0, 0, KV0], [1, 1, 32, KV0 + self.n_kv])
        taps = m["taps"]
        if self.is_global:
            width = 2 * N_KV * HEAD_DIM
            dyn = ttnn.slice(proj, [0, 0, 0, KV0 + width], [1, 1, 32, KV0 + self.n_kv])
            dyn = ttnn.matmul(self.dyn_bcast, dyn, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
            dyn = ttnn.multiply(dyn, self.dyn_lagsel)
            dyn = ttnn.matmul(dyn, self.dyn_expand, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
            taps = ttnn.add(taps, dyn)
        conv = ttnn.sum(ttnn.multiply(p_kv, taps), dim=-2, keepdim=True)
        width = N_KV * HEAD_DIM
        k = ttnn.slice(conv, [0, 0, 0, 0], [1, 1, 1, width])
        v = ttnn.slice(conv, [0, 0, 0, width], [1, 1, 1, 2 * width])
        return proj, k, v

    def _kv_norm(self, conv, kv):
        """k: L2 clip; v: unweighted RMS norm. [1,1,1,1024] -> [1, 1(batch), 8 heads, 128] bf16."""
        heads = ttnn.reshape(conv, [1, 1, N_KV, HEAD_DIM])
        if kv == "k":
            return self._head_clip(heads)
        return ttnn.rms_norm(heads, epsilon=EPS, compute_kernel_config=self.hifi4)

    def _write(self, cache, row, idx):
        ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(row, self.kv_shard), update_idxs_tensor=idx)

    def _project(self, state, positions, xq, xk, xv):
        """Everything before the attention core. Returns, all bf16 TILE:
        q [1,1,24,128] (clipped, tau-scaled), rel [1,1,24,16] (clipped), gate [1,1,1,3072] (pre-sigmoid),
        sink [1,8,3,1], k [1,1,8,128] (clipped), v [1,1,8,128] (RMS-normed). Advances state.x_hist."""
        tau = positions.tau if self.is_global else None
        qgrs, k, v = self._window_project(state, xq, xk, xv)

        def cols(c, heads=None, d=None):
            t = ttnn.slice(qgrs, [0, 0, 0, c[0]], [1, 1, 1, c[1]])
            return t if heads is None else ttnn.reshape(t, [1, 1, heads, d])

        # scale and clip in fp32, one bf16 rounding at the end (as v1's _scaled_clip)
        q = ttnn.multiply(cols(Q_COLS, N_HEADS, HEAD_DIM), self.q_scale, dtype=ttnn.float32)
        q = self._head_clip(q, tau, self.q_red)
        # rel clipped flat ([1, 384], 16-wide heads): per-head sums and their broadcast back are matmuls
        r = ttnn.multiply(cols(REL_COLS), self.rel_scale, dtype=ttnn.float32)
        f = ttnn.matmul(self._clip_factor(r, self.rel_red, tau), self.rel_bcast, compute_kernel_config=self.hifi4)
        rel = ttnn.reshape(ttnn.multiply(r, f, dtype=ttnn.bfloat16), [1, 1, N_HEADS, D_REL])
        gate = cols(GATE_COLS)
        sink = ttnn.reshape(cols(SINK_COLS), [1, N_KV, GROUPS, 1])
        return q, rel, gate, sink, self._kv_norm(k, "k"), self._kv_norm(v, "v")

    # ---------------------------------------------------------------- decode
    def decode(self, xq, state, positions, xk=None, xv=None):
        """xq/xk/xv: [1,1,1,2304] bf16 TILE (local layers use xq for all three)."""
        xk = xq if xk is None else xk
        xv = xq if xv is None else xv

        q, rel, gate, sink, k, v = self._project(state, positions, xq, xk, xv)
        self._write(state.k_ring, k, positions.ring)
        self._write(state.v_ring, v, positions.ring)
        if self.is_global:
            self._write(state.k_full, k, positions.pos)
            self._write(state.v_full, v, positions.pos)

        # relative bias per ring slot: rel_states @ proj[:, dist(slot)]
        prof = ttnn.embedding(positions.dist_idx, self.rel_table, layout=ttnn.TILE_LAYOUT)  # [1, 1024, 16]
        prof = ttnn.reshape(prof, [1, 1, WINDOW, D_REL])
        bias = ttnn.matmul(rel, prof, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)
        bias = ttnn.reshape(bias, [1, N_KV, GROUPS, WINDOW])

        qg = ttnn.reshape(q, [1, N_KV, GROUPS, HEAD_DIM])
        scaling = HEAD_DIM**-0.5
        near = ttnn.matmul(qg, state.k_ring, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
        near = ttnn.add(ttnn.add(ttnn.multiply(near, scaling), bias), positions.near_mask)
        parts = [near]
        if self.is_global:
            far = ttnn.matmul(qg, state.k_full, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
            parts.append(ttnn.add(ttnn.multiply(far, scaling), positions.far_mask))
        parts.append(ttnn.add(self.sink_pad, sink))
        logits = ttnn.concat(parts, dim=-1)
        probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=self.hifi4, numeric_stable=True)
        probs = ttnn.typecast(probs, ttnn.bfloat16)

        p_near = ttnn.slice(probs, [0, 0, 0, 0], [1, N_KV, GROUPS, WINDOW])
        attn = ttnn.matmul(p_near, state.v_ring, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
        if self.is_global:
            s = state.k_full.shape[-2]
            p_far = ttnn.slice(probs, [0, 0, 0, WINDOW], [1, N_KV, GROUPS, WINDOW + s])
            attn = ttnn.add(
                attn, ttnn.matmul(p_far, state.v_full, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
            )
        attn = ttnn.typecast(attn, ttnn.bfloat16)
        attn = ttnn.reshape(attn, [1, 1, 1, N_HEADS * HEAD_DIM])

        gated = ttnn.multiply(attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        return self.w_o(gated, ttnn.bfloat16, self.mm_cfg)


# -------------------------------------------------------------------- weights
def host_weights_from_hf(sd, is_global, dtype=ttnn.bfloat8_b):
    """§4 host tensors from an HF `Inkling2*Attention` state dict (torch). Used by tests."""

    def mm(t):  # [out, in] -> [1,1,in,out]
        return t.float().T.contiguous().view(1, 1, t.shape[1], t.shape[0])

    def host(t, dt, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t.float(), dtype=dt, layout=layout)

    rel_lin = sd["relative_position_bias.linear.weight"]
    qgrs = torch.cat(
        [sd["q_proj.linear.weight"], sd["o_proj.gate_proj.weight"], rel_lin, sd["sink_proj.weight"]], dim=0
    )
    qgrs = torch.cat([qgrs, torch.zeros(6560 - qgrs.shape[0], HIDDEN)], dim=0)
    out = {"w_qgrs": host(mm(qgrs), dtype), "w_o": host(mm(sd["o_proj.linear.weight"]), dtype)}
    for kv in ("k", "v"):
        w = sd[f"{kv}_proj.linear.weight"]
        if is_global:
            w = torch.cat([w, sd[f"{kv}_proj.sconv.weight_proj.weight"]], dim=0)
            taps = sd[f"{kv}_proj.sconv.static_weight"]  # [4, C], row = lag
        else:
            taps = sd[f"{kv}_proj.sconv.conv1d.weight"].squeeze(1).flip(-1).T  # [C,1,K] -> [K(lag), C]
        out[f"w_{kv}"] = host(mm(w), dtype)
        out[f"{kv}_conv"] = host(taps.contiguous().view(1, 1, TAPS, -1), ttnn.bfloat16)
    out["q_clip_scale"] = host(sd["q_proj.norm.scale"].view(1, 1, N_HEADS, HEAD_DIM), ttnn.bfloat16)
    out["rel_clip_scale"] = host(sd["relative_position_bias.norm.scale"].view(1, 1, N_HEADS, D_REL), ttnn.bfloat16)
    out["rel_proj"] = host(sd["relative_position_bias.proj"].view(1, 1, D_REL, WINDOW), ttnn.bfloat16)
    return out
