# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextGatedDeltaNet`, tensor-parallel over TP chips.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextGatedDeltaNet`
(+ `torch_chunk_gated_delta_rule`, the path this checkpoint actually takes -- the fla/causal-conv1d
fast paths are not installed, so the torch implementations are the golden).

    in_proj_qkvz : hidden -> nk * (2*head_k + 2*nrep*head_v)   # per key head: [q | k | v | z]
    in_proj_ba   : hidden -> nv * 2                            # per value head: [b | a]
    depthwise causal conv1d (kernel 4) over q|k|v, then silu
    beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
    chunked gated delta rule (l2-normalised q/k) -> gated RMSNorm against z -> out_proj

WHY THE DELTA RULE CLOSES IN A HANDFUL OF MATMULS HERE
------------------------------------------------------
`torch_chunk_gated_delta_rule` runs at chunk_size=64 and its state recurrence walks chunk by chunk.
This port now walks it too: `__call__` loops over 64-token chunks carrying `state` (the reference's
`last_recurrent_state`, shape (1, heads, head_k, head_v)), so there is no sequence-length ceiling.
Pass `initial_state=` to continue from a cache and `output_final_state=True` to get the final state
back alongside the output.

With NO incoming state and a single chunk, `last_recurrent_state` is zero throughout: `v_prime` and
`attn_inter` vanish and the chunk output collapses to

    core_attn_out = ((q @ k^T) * decay) @ (T @ (v * beta))

The one piece that is NOT a matmul in the reference is `T`: a 64-step Python loop doing forward
substitution on a strictly-lower-triangular M. That loop computes exactly (I - M)^-1, and since M is
strictly lower triangular it is nilpotent (M^64 = 0), so

    (I - M)^-1 = I + M + M^2 + ... + M^63 = (I+M)(I+M^2)(I+M^4)(I+M^8)(I+M^16)(I+M^32)

-- six factors, ten 64x64 matmuls, no loop-carried dependency. Verified against the HF reference at
PCC 1.0 before this port was written.

Tensor-parallel scheme (Mamba/SSD principle: shard the HEAD axis, never the time axis):

  * in_proj_qkvz / in_proj_ba are COLUMN-parallel on the head axis. qkvz is laid out key-head-major
    with that head's q, k, v and z inside one block, so splitting the output feature axis puts a
    whole head -- and the value heads it feeds -- on one chip.
  * conv1d is DEPTHWISE, so each channel is independent and simply travels with its head. Its
    channel order is [all q | all k | all v] rather than head-major, so the per-chip channel set is
    three separated blocks; `build` re-orders the conv weight on the host into chip-major order and
    then shards it with a plain split.
  * A_log / dt_bias are per value head, so they shard WITH the heads. `norm` scales head_v_dim (not
    the sharded axis) and stays REPLICATED.
  * The scan stays sequential within each chip's heads and the time axis is never split.
  * out_proj reduces back to model dim, so it is ROW-parallel over the value-head axis: each chip
    produces a partial sum and one all_reduce reassembles the golden output.
"""
from __future__ import annotations

import math

import torch
import ttnn

TILE = 32


def num_devices(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    ids = getattr(device, "get_device_ids", None)
    if callable(ids):
        try:
            return len(ids()) or 1
        except Exception:
            pass
    return 1


def to_device(host_tensor, device, *, mesh_mapper=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    kwargs = dict(dtype=dtype, layout=layout, device=device)
    if mesh_mapper is not None:
        kwargs["mesh_mapper"] = mesh_mapper
    return ttnn.from_torch(host_tensor.contiguous(), **kwargs)


def replicate_mapper(device, n):
    if n <= 1:
        return None
    try:
        return ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        return None


def shard_mapper(device, n, dim):
    return None if n <= 1 else ttnn.ShardTensorToMesh(device, dim=dim)


def matmul_weight(w):
    """torch nn.Linear stores (out, in); ttnn matmul wants a 4-D (1, 1, in, out)."""
    return w.t().unsqueeze(0).unsqueeze(0).contiguous()


def cols_to_heads(x, seq, offsets, width):
    """Stack feature-axis blocks onto the head axis: (1,1,S,F) -> (1,len(offsets),S,width)."""
    heads = [ttnn.slice(x, [0, 0, 0, o], [1, 1, seq, o + width]) for o in offsets]
    return ttnn.concat(heads, dim=1) if len(heads) > 1 else heads[0]


def cols_gather(x, seq, offsets, width):
    """Concatenate feature-axis blocks back along the feature axis: (1,1,S,F) -> (1,1,S,n*width)."""
    parts = [ttnn.slice(x, [0, 0, 0, o], [1, 1, seq, o + width]) for o in offsets]
    return ttnn.concat(parts, dim=-1) if len(parts) > 1 else parts[0]


def heads_to_cols(x, seq, num_heads, width):
    """(1,H,S,W) -> (1,1,S,H*W), head-major (HF's `.reshape(..., -1)`)."""
    if num_heads == 1:
        return x
    parts = [ttnn.slice(x, [0, h, 0, 0], [1, h + 1, seq, width]) for h in range(num_heads)]
    return ttnn.concat(parts, dim=-1)


def repeat_heads(x, times):
    """Repeat each head `times` times, contiguously (HF `repeat_interleave` on the head axis)."""
    if times == 1:
        return x
    n = int(x.shape[1])
    seq, width = int(x.shape[-2]), int(x.shape[-1])
    if n == 1:
        return ttnn.concat([x] * times, dim=1)
    pieces = []
    for h in range(n):
        head = ttnn.slice(x, [0, h, 0, 0], [1, h + 1, seq, width])
        pieces.extend([head] * times)
    return ttnn.concat(pieces, dim=1)


CHUNK = 64  # the reference's chunk_size for torch_chunk_gated_delta_rule


def rows(x, start, stop):
    """Slice the TOKEN axis of a (1, H, S, W) tensor: x[:, :, start:stop, :]."""
    n, width = int(x.shape[1]), int(x.shape[-1])
    if start == 0 and stop == int(x.shape[-2]):
        return x
    return ttnn.slice(x, [0, 0, start, 0], [1, n, stop, width])


def fit_width(x, width):
    """Tile/trim a tensor whose columns are all identical to exactly `width` columns."""
    have = int(x.shape[-1])
    if have == width:
        return x
    if have < width:
        x = ttnn.repeat(x, ttnn.Shape([1, 1, 1, -(-width // have)]))
    n, seq = int(x.shape[1]), int(x.shape[-2])
    return ttnn.slice(x, [0, 0, 0, 0], [1, n, seq, width])


class TtQwen3NextGatedDeltaNet:
    """Native ttnn Qwen3-Next gated delta net, sharded head-wise over the TP mesh."""

    def __init__(self, device, cfg) -> None:
        self.device = device
        self.__dict__.update(cfg)
        self.num_devices = num_devices(device)
        self._replicate = replicate_mapper(device, self.num_devices)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._masks = {}

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("gated_delta_net stub needs the torch reference module for its weights")
        sd = torch_module.state_dict()

        hk = int(torch_module.head_k_dim)
        hv = int(torch_module.head_v_dim)
        nk = int(torch_module.num_k_heads)
        nv = int(torch_module.num_v_heads)
        nrep = nv // nk
        hidden = int(torch_module.hidden_size)
        kernel = int(torch_module.conv_kernel_size)
        eps = float(torch_module.layer_norm_epsilon)

        n = num_devices(device)
        tp = n if (nk % n == 0 and nv % n == 0) else 1
        nk_l, nv_l = nk // tp, nv // tp

        shard_out = shard_mapper(device, tp, -1)
        shard_in = shard_mapper(device, tp, -2)
        replicate = replicate_mapper(device, n)

        w_qkvz = sd["in_proj_qkvz.weight"].detach().float()
        w_ba = sd["in_proj_ba.weight"].detach().float()
        w_out = sd["out_proj.weight"].detach().float()
        w_conv = sd["conv1d.weight"].detach().float()  # (conv_dim, 1, K)
        conv_bias = sd.get("conv1d.bias")
        a_log = sd["A_log"].detach().float()
        dt_bias = sd["dt_bias"].detach().float()

        # in_proj_ba packs [b | a] two-per-key-head; b/a are per-value-head SCALARS, and a scalar is
        # a 2-column slice that no tile-aligned device slice can peel off cleanly. Instead fold the
        # broadcast into the projection: build one weight per stream whose output block for value
        # head vh is that head's row repeated head_v_dim times, so the projection emits b (and
        # a + dt_bias, via the bias term) already broadcast across the head width.
        def ba_expanded(select_a):
            out = torch.empty(nv * hv, hidden, dtype=w_ba.dtype)
            at = 0
            for h in range(nk):
                for r in range(nrep):
                    # one assignment per value head; the (hidden,) row broadcasts over head width
                    out[at : at + hv] = w_ba[h * 2 * nrep + (nrep if select_a else 0) + r]
                    at += hv
            return out  # (nv * hv, hidden)

        w_b = matmul_weight(ba_expanded(False))
        w_a = matmul_weight(ba_expanded(True))
        dt_bias_exp = dt_bias.repeat_interleave(hv).view(1, 1, 1, nv * hv)
        neg_exp_a = (-a_log.exp()).repeat_interleave(hv).view(1, nv, 1, hv)

        # conv1d channels run [all q | all k | all v]; a chip's channels are therefore three
        # separated blocks. Re-order to chip-major on the host so a plain split lands right.
        key_dim, value_dim = hk * nk, hv * nv
        kd_l, vd_l = key_dim // tp, value_dim // tp
        ordered = torch.empty(2 * key_dim + value_dim, *w_conv.shape[1:], dtype=w_conv.dtype)
        for c in range(tp):
            at = c * (2 * kd_l + vd_l)
            ordered[at : at + kd_l] = w_conv[c * kd_l : (c + 1) * kd_l]
            ordered[at + kd_l : at + 2 * kd_l] = w_conv[key_dim + c * kd_l : key_dim + (c + 1) * kd_l]
            ordered[at + 2 * kd_l : at + 2 * kd_l + vd_l] = w_conv[
                2 * key_dim + c * vd_l : 2 * key_dim + (c + 1) * vd_l
            ]
        w_conv = ordered.squeeze(1)  # (conv_dim, K), chip-major

        # The output norm IS the graduated `r_m_s_norm_gated` stub (the port of
        # `layers.*.linear_attn.norm`).  Imported here rather than at module scope because that
        # stub imports this module's device helpers.
        from models.demos.qwen3_coder_next._stubs.r_m_s_norm_gated import TtQwen3NextRMSNormGated

        cfg = dict(
            hidden_size=hidden,
            head_k_dim=hk,
            head_v_dim=hv,
            tp=tp,
            num_k_heads_local=nk_l,
            num_v_heads_local=nv_l,
            n_rep=nrep,
            conv_kernel_size=kernel,
            eps=eps,
            qkvz_block=2 * hk + 2 * nrep * hv,
            key_dim_local=hk * nk_l,
            value_dim_local=hv * nv_l,
            w_qkvz=to_device(matmul_weight(w_qkvz), device, mesh_mapper=shard_out),
            w_b=to_device(w_b, device, mesh_mapper=shard_out),
            w_a=to_device(w_a, device, mesh_mapper=shard_out),
            dt_bias=to_device(dt_bias_exp, device, mesh_mapper=shard_out),
            neg_exp_a=to_device(neg_exp_a, device, mesh_mapper=shard_mapper(device, tp, 1)),
            w_out=to_device(matmul_weight(w_out), device, mesh_mapper=shard_in),
            norm=TtQwen3NextRMSNormGated.build(device, torch_module.norm),
            # One (1,1,1,conv_dim) tensor per tap: a tap is a full-width per-channel scale, and
            # keeping them separate avoids ever slicing a TILE tensor on the row axis at offset 1..3.
            conv_taps=[
                to_device(
                    w_conv[:, j].view(1, 1, 1, -1), device, mesh_mapper=shard_mapper(device, tp, -1)
                )
                for j in range(kernel)
            ],
            conv_bias=(
                None
                if conv_bias is None
                else to_device(
                    conv_bias.detach().float().view(1, 1, 1, -1), device, mesh_mapper=shard_mapper(device, tp, -1)
                )
            ),
        )
        return cls(device, cfg)

    # -------------------------------------------------------------- helpers

    def _const(self, key, builder):
        if key not in self._masks:
            self._masks[key] = to_device(builder(), self.device, mesh_mapper=self._replicate)
        return self._masks[key]

    def _shift_matrix(self, seq, d):
        """(1,1,S,S) with ones on the d-th sub-diagonal: `S_d @ x` gives x shifted down by d rows."""

        def build():
            m = torch.zeros(seq, seq)
            if d < seq:
                m[torch.arange(d, seq), torch.arange(0, seq - d)] = 1.0
            return m.view(1, 1, seq, seq)

        return self._const(f"shift{seq}:{d}", build)

    def _tri(self, seq, diagonal):
        def build():
            return torch.tril(torch.ones(seq, seq), diagonal=diagonal).view(1, 1, seq, seq)

        return self._const(f"tril{seq}:{diagonal}", build)

    def _ones(self, r, c):
        """(1,1,r,c) of ones. `ones(r,n) @ g` puts the WHOLE-chunk sum of g in every row, which is
        how the last row of a cumsum is read without slicing at a non-tile-aligned offset."""
        return self._const(f"ones{r}:{c}", lambda: torch.ones(r, c).view(1, 1, r, c))

    def _eye(self, seq):
        return self._const(f"eye{seq}", lambda: torch.eye(seq).view(1, 1, seq, seq))

    def _causal_conv(self, x, seq):
        """Depthwise causal conv1d over the token axis, as a sum of shifted per-channel scales.

        out[t, c] = sum_j w[c, j] * x[t - (K-1-j), c]. The shift is a matmul with a sub-diagonal
        matrix, which keeps every tensor tile-aligned -- a row slice at offset 1..3 would not be.
        """
        kernel = self.conv_kernel_size
        acc = None
        for d in range(kernel):
            tap = self.conv_taps[kernel - 1 - d]
            shifted = (
                x
                if d == 0
                else ttnn.matmul(self._shift_matrix(seq, d), x, compute_kernel_config=self.compute_config)
            )
            term = ttnn.multiply(shifted, tap)
            acc = term if acc is None else ttnn.add(acc, term)
        if self.conv_bias is not None:
            acc = ttnn.add(acc, self.conv_bias)
        return ttnn.silu(acc)

    def _l2norm(self, x, scale):
        """`l2norm(x) * scale`, expressed as a weightless rms_norm.

        rms_norm(x, eps/D) = x * rsqrt(sum(x^2)/D + eps/D) = sqrt(D) * x * rsqrt(sum(x^2) + eps),
        so dividing by sqrt(D) recovers the fla-style l2norm exactly.
        """
        d = self.head_k_dim
        y = ttnn.rms_norm(x, epsilon=1e-6 / d, compute_kernel_config=self.compute_config)
        return ttnn.multiply(y, scale / math.sqrt(d))

    # -------------------------------------------------------------- forward

    def __call__(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        *,
        initial_state=None,
        output_final_state=False,
        **kwargs,
    ):
        hk, hv = self.head_k_dim, self.head_v_dim
        nk_l, nv_l = self.num_k_heads_local, self.num_v_heads_local
        blk = self.qkvz_block
        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))

        # --- column-parallel input projections -------------------------------------------------
        qkvz = ttnn.linear(x, self.w_qkvz, compute_kernel_config=self.compute_config)

        q_off = [h * blk for h in range(nk_l)]
        k_off = [h * blk + hk for h in range(nk_l)]
        v_off = [h * blk + 2 * hk + r * hv for h in range(nk_l) for r in range(self.n_rep)]
        z_off = [h * blk + 2 * hk + self.n_rep * hv + r * hv for h in range(nk_l) for r in range(self.n_rep)]

        # conv1d sees the flat [q | k | v] channel order the reference builds before transposing.
        mixed = ttnn.concat(
            [
                cols_gather(qkvz, seq, q_off, hk),
                cols_gather(qkvz, seq, k_off, hk),
                cols_gather(qkvz, seq, v_off, hv),
            ],
            dim=-1,
        )
        mixed = self._causal_conv(mixed, seq)

        kd, vd = self.key_dim_local, self.value_dim_local
        query = cols_to_heads(mixed, seq, [i * hk for i in range(nk_l)], hk)
        key = cols_to_heads(ttnn.slice(mixed, [0, 0, 0, kd], [1, 1, seq, 2 * kd]), seq, [i * hk for i in range(nk_l)], hk)
        value = cols_to_heads(
            ttnn.slice(mixed, [0, 0, 0, 2 * kd], [1, 1, seq, 2 * kd + vd]), seq, [i * hv for i in range(nv_l)], hv
        )
        z = cols_to_heads(qkvz, seq, z_off, hv)

        # --- beta / g, already broadcast across the head width by the folded projections ---------
        beta = ttnn.sigmoid(ttnn.linear(x, self.w_b, compute_kernel_config=self.compute_config))
        beta = cols_to_heads(beta, seq, [i * hv for i in range(nv_l)], hv)
        adt = ttnn.linear(x, self.w_a, bias=self.dt_bias, compute_kernel_config=self.compute_config)
        g = ttnn.multiply(
            cols_to_heads(ttnn.softplus(adt), seq, [i * hv for i in range(nv_l)], hv), self.neg_exp_a
        )

        # --- GQA-style expansion of the key heads onto the value heads ---------------------------
        query = repeat_heads(query, self.n_rep)
        key = repeat_heads(key, self.n_rep)

        # --- gated delta rule, chunked, carrying the recurrent state across chunks --------------
        # Reference: `torch_chunk_gated_delta_rule`. Inside a chunk the rule closes in matmuls; the
        # ONLY thing crossing a chunk boundary is `state` -- the reference's `last_recurrent_state`,
        # shape (1, heads, head_k, head_v). With state=None every cross-chunk term vanishes and this
        # collapses to exactly the previous single-chunk form, which is what the seq<=64 PCC tests
        # still check.
        q_n = self._l2norm(query, 1.0 / math.sqrt(hk))  # includes the 1/sqrt(head_k) attention scale
        k_n = self._l2norm(key, 1.0)
        beta_k = fit_width(beta, hk)
        k_beta = ttnn.multiply(k_n, beta_k)
        v_beta = ttnn.multiply(value, fit_width(beta, hv))

        state = initial_state
        pieces = []
        for start_t in range(0, seq, CHUNK):
            stop_t = min(start_t + CHUNK, seq)
            n = stop_t - start_t

            q_c = rows(q_n, start_t, stop_t)
            k_c = rows(k_n, start_t, stop_t)
            kb_c = rows(k_beta, start_t, stop_t)
            vb_c = rows(v_beta, start_t, stop_t)
            g_c = rows(g, start_t, stop_t)

            tril = self._tri(n, 0)
            gc = ttnn.matmul(tril, g_c, compute_kernel_config=self.compute_config)  # cumsum over time
            gc_sq = fit_width(gc, n)
            # decay[i, j] = exp(gc_i - gc_j) for i >= j. Masking BEFORE exp matters: above the
            # diagonal the difference is positive and would overflow to inf, and inf * 0 is NaN.
            diff = ttnn.multiply(ttnn.subtract(gc_sq, ttnn.transpose(gc_sq, -2, -1)), tril)
            decay = ttnn.multiply(ttnn.exp(diff), tril)

            strict = self._tri(n, -1)
            m = ttnn.multiply(
                ttnn.multiply(
                    ttnn.matmul(kb_c, ttnn.transpose(k_c, -2, -1), compute_kernel_config=self.compute_config),
                    decay,
                ),
                strict,
            )
            m = ttnn.neg(m)

            eye = self._eye(n)
            t_inv = ttnn.add(eye, m)  # (I - M)^-1 by Neumann doubling; M is nilpotent (strictly lower)
            power = m
            for _ in range(max(int(math.ceil(math.log2(max(n, 2)))) - 1, 0)):
                power = ttnn.matmul(power, power, compute_kernel_config=self.compute_config)
                t_inv = ttnn.matmul(t_inv, ttnn.add(eye, power), compute_kernel_config=self.compute_config)

            val = ttnn.matmul(t_inv, vb_c, compute_kernel_config=self.compute_config)
            attn = ttnn.multiply(
                ttnn.matmul(q_c, ttnn.transpose(k_c, -2, -1), compute_kernel_config=self.compute_config),
                decay,
            )
            if state is None:
                v_new = val
                core_c = ttnn.matmul(attn, v_new, compute_kernel_config=self.compute_config)
            else:
                exp_gc = fit_width(ttnn.exp(gc), hk)  # only the cross-chunk terms need it
                # v_prime = k_cumdecay @ S ;  v_new = v - v_prime      (reference lines 435-439)
                k_cumdecay = ttnn.matmul(
                    t_inv, ttnn.multiply(kb_c, exp_gc), compute_kernel_config=self.compute_config
                )
                v_new = ttnn.subtract(
                    val, ttnn.matmul(k_cumdecay, state, compute_kernel_config=self.compute_config)
                )
                # attn_inter = (q * exp(g_cum)) @ S                    (reference line 440)
                attn_inter = ttnn.matmul(
                    ttnn.multiply(q_c, exp_gc), state, compute_kernel_config=self.compute_config
                )
                core_c = ttnn.add(
                    attn_inter, ttnn.matmul(attn, v_new, compute_kernel_config=self.compute_config)
                )
            pieces.append(core_c)

            if stop_t < seq or output_final_state:
                # S <- S * exp(g_total) + (k * exp(g_total - g_cum))^T @ v_new   (reference line 442)
                # `ones(r, n) @ g_c` puts the chunk's TOTAL sum of g in every row, which reads the
                # last row of the cumsum without slicing at row n-1 (never tile-aligned).
                g_tot = ttnn.matmul(self._ones(n, n), g_c, compute_kernel_config=self.compute_config)
                w = fit_width(ttnn.exp(ttnn.subtract(g_tot, gc)), hk)
                upd = ttnn.matmul(
                    ttnn.transpose(ttnn.multiply(k_c, w), -2, -1),
                    v_new,
                    compute_kernel_config=self.compute_config,
                )
                if state is None:
                    state = upd  # S was 0, so the decayed term drops out
                else:
                    g_tot_k = ttnn.matmul(
                        self._ones(hk, n), g_c, compute_kernel_config=self.compute_config
                    )
                    state = ttnn.add(ttnn.multiply(state, ttnn.exp(g_tot_k)), upd)

        core = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-2)

        # --- gated RMSNorm against z, then the row-parallel output projection --------------------
        # `core` is (1, heads_local, S, head_v_dim); the gated norm operates per row over the last
        # axis, so the head axis rides along -- that is the stub's own 4-D branch.
        core = self.norm(core, z)
        core = heads_to_cols(core, seq, nv_l, hv)

        partial = ttnn.linear(core, self.w_out, compute_kernel_config=self.compute_config)
        # Gate on the degree the weights were ACTUALLY sharded at, not the mesh width: when the head
        # counts do not divide the mesh, `build` falls back to tp=1 (replicated) and a reduce then
        # sums N identical COMPLETE results. Same bug already fixed in `_stubs/attention.py`.
        if self.tp > 1:
            partial = ttnn.all_reduce(partial)
        out = ttnn.reshape(partial, (1, seq, self.hidden_size))
        return (out, state) if output_final_state else out


def build(device, torch_module=None):
    return TtQwen3NextGatedDeltaNet.build(device, torch_module)


def gated_delta_net(device, torch_module=None):
    return TtQwen3NextGatedDeltaNet.build(device, torch_module)
