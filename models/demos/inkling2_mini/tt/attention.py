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

import torch

import ttnn

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
        # conv inputs at lag 1..3 (bf16-exact values, kept bf16)
        width = N_KV * HEAD_DIM
        self.k_hist = [zeros([1, 1, 1, width], ttnn.bfloat16) for _ in range(TAPS - 1)]
        self.v_hist = [zeros([1, 1, 1, width], ttnn.bfloat16) for _ in range(TAPS - 1)]

    def tensors(self):
        out = [self.k_ring, self.v_ring, *self.k_hist, *self.v_hist]
        if self.k_full is not None:
            out += [self.k_full, self.v_full]
        return out


class Inkling2Attention:
    """One attention layer. `weights` maps §4 names (without `attn.`) to host tensors."""

    def __init__(self, device, weights, is_global):
        self.device = device
        self.is_global = is_global
        dram = ttnn.DRAM_MEMORY_CONFIG

        def dev(name):
            return ttnn.to_device(weights[name], device, memory_config=dram)

        def torch_of(name):
            return ttnn.to_torch(weights[name]).float()

        def put(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=dram)

        self.w_qgrs = dev("w_qgrs")
        self.w_k = dev("w_k")
        self.w_v = dev("w_v")
        self.w_o = dev("w_o")
        self.q_scale = put(torch_of("q_clip_scale").view(1, 1, N_HEADS, HEAD_DIM))
        self.rel_scale = put(torch_of("rel_clip_scale").view(1, 1, N_HEADS, D_REL))
        # relative profiles as an embedding table: row d = proj[:, d]
        proj_t = torch_of("rel_proj").view(D_REL, WINDOW).T.contiguous()
        self.rel_table = put(proj_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # conv taps per lag, fp32. Local layers fold the residual and 1/sqrt(K+1) into the taps;
        # global layers keep static taps and scale after adding the dynamic part.
        self.conv_scale = 1.0 / math.sqrt((2 if is_global else 1) * TAPS + 1)
        self.taps = {}
        for kv in ("k", "v"):
            w = torch_of(f"{kv}_conv").view(TAPS, N_KV * HEAD_DIM)
            if not is_global:
                w = w.clone()
                w[0] += 1.0
                w *= self.conv_scale
            self.taps[kv] = [put(w[r].view(1, 1, 1, -1)) for r in range(TAPS)]
        if is_global:
            # dyn[h*TAPS + lag] -> lag block of [TAPS * 1024], broadcast over the 128 channels of head h
            expand = torch.zeros(N_KV * TAPS, TAPS * N_KV * HEAD_DIM)
            for h in range(N_KV):
                for lag in range(TAPS):
                    c0 = lag * N_KV * HEAD_DIM + h * HEAD_DIM
                    expand[h * TAPS + lag, c0 : c0 + HEAD_DIM] = 1.0
            self.expand = put(expand, dtype=ttnn.bfloat16)
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
    def _linear(self, x, w, dtype=ttnn.bfloat16):
        return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=self.mm_cfg)

    @staticmethod
    def _l2_clip(x32, threshold):
        """x * min(threshold / (|x| + 1e-6), 1) over the last dim, fp32."""
        norm = ttnn.sqrt(ttnn.sum(ttnn.square(x32), dim=-1, keepdim=True))
        factor = ttnn.clamp(ttnn.reciprocal(ttnn.add(norm, 1e-6)), max=1.0 / threshold)
        return ttnn.multiply(x32, ttnn.multiply(factor, threshold))

    def _scaled_clip(self, x, scale, threshold, tau):
        """`Inkling2ScaledL2Clip` / `Inkling2LogScaledL2Clip` on [1,1,heads,d] bf16."""
        y = ttnn.typecast(self._l2_clip(ttnn.multiply(ttnn.typecast(x, ttnn.float32), scale), threshold), ttnn.bfloat16)
        if tau is not None:
            y = ttnn.typecast(ttnn.multiply(ttnn.typecast(y, ttnn.float32), tau), ttnn.bfloat16)
        return y

    def _conv(self, kv, cur, hist, dyn):
        """Short causal conv of `cur` [1,1,1,1024] bf16 with its last 3 inputs; returns bf16."""
        taps = self.taps[kv]
        window = [ttnn.typecast(t, ttnn.float32) for t in (cur, *hist)]
        acc = None
        for lag, x in enumerate(window):
            w = taps[lag]
            if dyn is not None:
                w = ttnn.add(w, dyn[lag])
            term = ttnn.multiply(x, w)
            acc = term if acc is None else ttnn.add(acc, term)
        if dyn is not None:
            acc = ttnn.multiply(ttnn.add(acc, window[0]), self.conv_scale)
        # history shift: lag3 <- lag2 <- lag1 <- cur
        ttnn.copy(hist[1], hist[2])
        ttnn.copy(hist[0], hist[1])
        ttnn.copy(cur, hist[0])
        return ttnn.typecast(acc, ttnn.bfloat16)

    def _kv(self, x, w, kv, hist):
        width = N_KV * HEAD_DIM
        proj = self._linear(x, w)
        dyn = None
        if self.is_global:
            cur = ttnn.slice(proj, [0, 0, 0, 0], [1, 1, 1, width])
            dyn_w = ttnn.slice(proj, [0, 0, 0, width], [1, 1, 1, width + N_KV * TAPS])
            dyn_all = ttnn.matmul(dyn_w, self.expand, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
            dyn = [ttnn.slice(dyn_all, [0, 0, 0, lag * width], [1, 1, 1, (lag + 1) * width]) for lag in range(TAPS)]
        else:
            cur = proj
        conv = self._conv(kv, cur, hist, dyn)
        heads = ttnn.typecast(ttnn.reshape(conv, [1, 1, N_KV, HEAD_DIM]), ttnn.float32)
        if kv == "k":
            out = self._l2_clip(heads, QK_CLIP)
        else:
            out = ttnn.multiply(heads, ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.square(heads), dim=-1, keepdim=True), EPS)))
        return ttnn.typecast(out, ttnn.bfloat16)  # [1, 1(batch), 8 heads, 128]

    def _write(self, cache, row, idx):
        ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(row, self.kv_shard), update_idxs_tensor=idx)

    # ---------------------------------------------------------------- decode
    def decode(self, xq, state, positions, xk=None, xv=None):
        """xq/xk/xv: [1,1,1,2304] bf16 TILE (local layers use xq for all three)."""
        xk = xq if xk is None else xk
        xv = xq if xv is None else xv
        tau = positions.tau if self.is_global else None

        qgrs = self._linear(xq, self.w_qgrs)

        def cols(c, heads=None, d=None):
            t = ttnn.slice(qgrs, [0, 0, 0, c[0]], [1, 1, 1, c[1]])
            return t if heads is None else ttnn.reshape(t, [1, 1, heads, d])

        q = self._scaled_clip(cols(Q_COLS, N_HEADS, HEAD_DIM), self.q_scale, QK_CLIP, tau)
        rel = self._scaled_clip(cols(REL_COLS, N_HEADS, D_REL), self.rel_scale, REL_CLIP, tau)
        gate = cols(GATE_COLS)
        sink = ttnn.reshape(cols(SINK_COLS), [1, N_KV, GROUPS, 1])

        k = self._kv(xk, self.w_k, "k", state.k_hist)
        v = self._kv(xv, self.w_v, "v", state.v_hist)
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
        parts.append(ttnn.add(self.sink_pad, ttnn.typecast(sink, ttnn.float32)))
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

        gated = ttnn.multiply(ttnn.typecast(attn, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
        return self._linear(ttnn.typecast(gated, ttnn.bfloat16), self.w_o)


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
