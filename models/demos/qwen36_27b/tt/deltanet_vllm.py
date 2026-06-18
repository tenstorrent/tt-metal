# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
TT-NN Gated DeltaNet for Qwen3.6-27B — vLLM + tensor-parallel (TP=8) path.

Adapted from the coder_next template. Differences vs the single-device
``deltanet.py``:
  * ``TtDeltaNetState`` gains a ``batch=`` param and continuous-batching slot
    management (write_recurrent_slot / write_conv_slot / _scatter_row /
    reset_slots) so one persistent [max_batch, ...] state can hold many requests.
  * ``TtGatedDeltaNet`` shards the big projection weights across the mesh:
      - in_proj_qkv / in_proj_z : column(output)-parallel (dim=3) -> all_gather
        the linear output back to full before the (replicated) fused kernel.
      - out_proj                : row(input)-parallel (dim=2) -> mesh_partition
        the input + all_reduce the partials.
    Per-head tensors (b/a/A_log/dt_bias/norm/conv) stay replicated so the fused
    kernel still reads them by global head_idx (matches tp_model.py's replicated
    per-head approach for the non-fused composition path).

Qwen3.6-27B DeltaNet dims: 16 k-heads, 48 v-heads (ratio 3), k/v head_dim 128,
conv_kernel 4. conv_dim = 2048*2 + 6144 = 10240.
"""

import torch
import ttnn
from models.demos.qwen36_27b.tt.mesh_utils import to_torch as mesh_to_torch

from models.common.lightweightmodule import LightweightModule


class TtDeltaNetState:
    """Manages DeltaNet recurrent + conv1d state for all layers.

    ``batch`` rows are independent sequences. batch=1 reproduces the legacy
    single-sequence shapes exactly.
    """

    def __init__(self, num_layers, layer_types, device, config, batch=1):
        self.device = device
        self.batch = batch
        self.conv_states = {}
        self.recurrent_states = {}
        # On-device decode conv history: per layer, the 3 prior conv inputs as
        # [1,1,1,conv_dim] device tensors (h0=oldest..h2=newest-1). Lazily seeded
        # from the prefill conv_state on the first decode step, then updated on-device.
        self.conv_hist = {}
        # set by a new request's prefill (in-place path): re-seed conv_hist buffers
        # in-place from the new conv_state on the next decode step (keeps trace addrs).
        self._reseed_conv_hist = False
        # When True, state updates write IN-PLACE into the fixed buffers (ttnn.copy)
        # instead of swapping the tensor handle — required for trace (static buffers).
        self.trace_mode = False

        for i in range(num_layers):
            if layer_types[i] == "linear_attention":
                num_v_heads = config.linear_num_value_heads
                k_dim = config.linear_key_head_dim
                v_dim = config.linear_value_head_dim
                conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 + v_dim * num_v_heads

                self.recurrent_states[i] = ttnn.zeros(
                    [batch, num_v_heads, k_dim, v_dim],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                self.conv_states[i] = ttnn.zeros(
                    [batch, 1, conv_dim, 32],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

    def get_recurrent_state(self, layer_idx):
        return self.recurrent_states.get(layer_idx)

    def _set(self, store, layer_idx, state):
        old = store.get(layer_idx)
        if (
            self.trace_mode
            and old is not None
            and old is not state
            and list(old.shape) == list(state.shape)
        ):
            ttnn.copy(state, old)  # new -> fixed buffer (in place); keep handle
            try:
                ttnn.deallocate(state)
            except Exception:
                pass
            return
        if old is not None and old is not state:
            try:
                ttnn.deallocate(old)
            except Exception:
                pass
        store[layer_idx] = state

    def set_recurrent_state(self, layer_idx, state):
        self._set(self.recurrent_states, layer_idx, state)

    def get_conv_state(self, layer_idx):
        return self.conv_states.get(layer_idx)

    def set_conv_state(self, layer_idx, state):
        self._set(self.conv_states, layer_idx, state)

    def get_conv_state_cpu(self, layer_idx, conv_k=4):
        cs = self.conv_states.get(layer_idx)
        if cs is None:
            return None
        cs_cpu = mesh_to_torch(cs)  # [B, 1, conv_dim, 32]
        return cs_cpu[0, :, :, :conv_k]  # [1, conv_dim, conv_k]

    def set_conv_state_from_cpu(self, layer_idx, cpu_state):
        conv_dim = cpu_state.shape[1]
        conv_k = cpu_state.shape[2]
        padded = torch.nn.functional.pad(cpu_state, (0, 32 - conv_k))  # [1, conv_dim, 32]
        padded = padded.unsqueeze(0).to(torch.bfloat16)  # [1, 1, conv_dim, 32]
        self.conv_states[layer_idx] = ttnn.from_torch(
            padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    # ----- decode trace: snapshot/restore the running (decode-mutated) state -----
    def snapshot_decode_state(self):
        """Host snapshot of the state mutated during decode: recurrent_states (rs)
        and conv_hist. The trace compile+capture runs advance this running state
        twice; restoring this snapshot afterward resets it to the post-prefill value
        so the first replay starts correctly. (conv_states/cs is NOT mutated in the
        decode path, so it is not snapshotted.)"""
        return {
            "recur": {li: mesh_to_torch(rs).clone() for li, rs in self.recurrent_states.items()},
            "hist": {li: [mesh_to_torch(t).clone() for t in h] for li, h in self.conv_hist.items()},
        }

    def restore_decode_state(self, snap):
        """Restore the snapshot IN-PLACE into the existing fixed buffers (so captured
        trace buffer addresses stay valid)."""
        rep = ttnn.ReplicateTensorToMesh(self.device)
        for li, rs_cpu in snap["recur"].items():
            tmp = ttnn.from_torch(rs_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=self.device, mesh_mapper=rep)
            ttnn.copy(tmp, self.recurrent_states[li])
            ttnn.deallocate(tmp)
        for li, hs in snap["hist"].items():
            for k, t_cpu in enumerate(hs):
                tmp = ttnn.from_torch(t_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=self.device, mesh_mapper=rep)
                ttnn.copy(tmp, self.conv_hist[li][k])
                ttnn.deallocate(tmp)

    # ----- vLLM continuous batching: per-request slot management -----
    @staticmethod
    def _scatter_row(full, row, row_tensor):
        """Return a copy of `full` ([B,...]) with dim-0 index `row` replaced by
        `row_tensor` ([1,...]). Old `full` is deallocated."""
        B = full.shape[0]
        nd = len(full.shape)

        def _slice(lo, hi):
            start = [0] * nd
            end = list(full.shape)
            start[0] = lo
            end[0] = hi
            return ttnn.slice(full, start, end)

        parts = []
        if row > 0:
            parts.append(_slice(0, row))
        parts.append(row_tensor)
        if row < B - 1:
            parts.append(_slice(row + 1, B))
        new = row_tensor if len(parts) == 1 else ttnn.concat(parts, dim=0)
        if new is not full:
            try:
                ttnn.deallocate(full)
            except Exception:
                pass
        return new

    def write_recurrent_slot(self, layer_idx, row, state_1):
        """Write a single-request recurrent state [1,H,Dk,Dv] into row `row`."""
        self.recurrent_states[layer_idx] = self._scatter_row(self.recurrent_states[layer_idx], row, state_1)

    def write_conv_slot(self, layer_idx, row, conv_1):
        """Write a single-request conv state [1,1,conv_dim,32] into row `row`."""
        self.conv_states[layer_idx] = self._scatter_row(self.conv_states[layer_idx], row, conv_1)

    def reset_slots(self, rows):
        """Zero the recurrent + conv state for the given batch rows (new requests)."""
        for layer_idx in list(self.recurrent_states.keys()):
            rshape = [1] + list(self.recurrent_states[layer_idx].shape)[1:]
            cshape = [1] + list(self.conv_states[layer_idx].shape)[1:]
            for row in rows:
                zr = ttnn.zeros(rshape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                zc = ttnn.zeros(cshape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                self.recurrent_states[layer_idx] = self._scatter_row(self.recurrent_states[layer_idx], row, zr)
                self.conv_states[layer_idx] = self._scatter_row(self.conv_states[layer_idx], row, zc)


def tt_rms_norm_per_head(x, weight, dv, eps=1e-6):
    """Per-head RMSNorm on device. x: [1, H, 1, Dv], weight: [1, 1, 1, Dv]."""
    x_sq = ttnn.mul(x, x)
    x_sq_sum = ttnn.sum(x_sq, dim=-1, keepdim=True)
    mean = ttnn.mul(x_sq_sum, 1.0 / dv)
    inv_rms = ttnn.rsqrt(ttnn.add(mean, eps))
    normed = ttnn.mul(x, inv_rms)
    return ttnn.mul(normed, weight)


try:
    _deltanet_decode_full_op = ttnn.experimental.deltanet_decode_full
    USE_FULL_FUSED_KERNEL = True
except AttributeError:
    USE_FULL_FUSED_KERNEL = False

try:
    _deltanet_prefill_full_op = ttnn.experimental.deltanet_prefill_full
    USE_PREFILL_FUSED_KERNEL = True
except AttributeError:
    USE_PREFILL_FUSED_KERNEL = False

import os

if os.environ.get("DISABLE_FUSED_KERNEL") or os.environ.get("DISABLE_FULL_FUSED_KERNEL"):
    USE_FULL_FUSED_KERNEL = False
    USE_PREFILL_FUSED_KERNEL = False
    _deltanet_decode_full_op = None
    _deltanet_prefill_full_op = None


class TtGatedDeltaNet(LightweightModule):
    """Gated DeltaNet layer with optional tensor-parallel sharding (TP)."""

    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.config = config
        proj_dtype = config.get_dense_dtype(getattr(config, "weights_dtype", dtype))
        self.dense_tp = getattr(config, "dense_tp", False)
        self.tp_size = getattr(config, "tp_size", 8)

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.head_expand_ratio = self.num_v_heads // self.num_k_heads
        self.scale = self.head_k_dim**-0.5

        prefix = f"model.layers.{layer_idx}"

        # TP sharding: in_proj_qkv/z are column(output)-parallel (dim=3); out_proj
        # is row(input)-parallel (dim=2). Per-head tensors stay replicated.
        qkv_sd = 3 if self.dense_tp else None
        z_sd = 3 if self.dense_tp else None
        out_sd = 2 if self.dense_tp else None
        self.in_proj_qkv_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_qkv.weight", proj_dtype, transpose=True, shard_dim=qkv_sd
        )
        self.in_proj_z_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_z.weight", proj_dtype, transpose=True, shard_dim=z_sd
        )
        self.in_proj_b_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_b.weight", proj_dtype, transpose=True
        )
        self.in_proj_a_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_a.weight", proj_dtype, transpose=True
        )
        self.out_proj_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.out_proj.weight", proj_dtype, transpose=True, shard_dim=out_sd
        )
        # Qwen3NextRMSNormGated uses the weight DIRECTLY (no +1).
        self.norm_weight = self._load_weight(state_dict, f"{prefix}.linear_attn.norm.weight", dtype)

        self.A_log_bf16 = self._load_weight(state_dict, f"{prefix}.linear_attn.A_log", dtype)
        self.dt_bias_bf16 = self._load_weight(state_dict, f"{prefix}.linear_attn.dt_bias", dtype)

        a_log_key = f"{prefix}.linear_attn.A_log"
        dt_bias_key = f"{prefix}.linear_attn.dt_bias"
        self.A_log_cpu = state_dict[a_log_key][: self.num_v_heads].float() if a_log_key in state_dict else None
        self.dt_bias_cpu = state_dict[dt_bias_key][: self.num_v_heads].float() if dt_bias_key in state_dict else None
        norm_key = f"{prefix}.linear_attn.norm.weight"
        self.norm_weight_cpu = state_dict[norm_key][: self.head_v_dim].float() if norm_key in state_dict else None

        conv_w_key = f"{prefix}.linear_attn.conv1d.weight"
        if conv_w_key in state_dict:
            self.conv1d_weight = state_dict[conv_w_key].squeeze(1)  # [conv_dim, kernel_size]
            w_padded = torch.nn.functional.pad(self.conv1d_weight, (0, 28))  # [conv_dim, 32]
            self.conv1d_weight_tt = ttnn.from_torch(
                w_padded.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            # Per-tap conv weight columns [1,1,1,conv_dim] (replicated across the mesh,
            # matching the gathered full-width qkv) for the on-device decode conv1d
            # (dot = w0*h0 + w1*h1 + w2*h2 + w3*x). Avoids the per-layer CPU round-trip.
            _rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
            self.conv_w_cols = [
                ttnn.from_torch(
                    self.conv1d_weight[:, k].contiguous().reshape(1, 1, 1, -1).to(torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=_rep,
                )
                for k in range(self.conv_kernel_size)
            ]
        else:
            self.conv1d_weight = None
            self.conv1d_weight_tt = None
            self.conv_w_cols = None

        # Head-parallel SHARDED decode weights for num_seq>1 (batched decode). Built only
        # when config.batched_decode (set by the generator when max_batch_size>1) so the
        # num_seq=1 path pays no extra weight memory. The gathered weights above stay for
        # prefill (+ num_seq=1 decode).
        self.batched_decode = bool(getattr(config, "batched_decode", False))
        if self.dense_tp and self.batched_decode and self.conv1d_weight is not None:
            self._build_sharded_decode_weights(state_dict, prefix, proj_dtype, dtype)

    def _build_sharded_decode_weights(self, state_dict, prefix, proj_dtype, dtype):
        """Head-parallel SHARDED weights for batched (num_seq>1) decode: each chip owns
        nkp=nk/TP k-heads and nvp=nv/TP v-heads. Additive — the gathered prefill/decode
        weights are kept (num_seq=1 path unchanged). in_proj_qkv and the conv1d weight
        columns mix q|k|v so they need PER-CHIP-INTERLEAVED layout [q_c|k_c|v_c] (like
        Wqkv/MLP); z/b/a/A_log/dt_bias/out_proj are head-aligned so plain dim-3 shard
        works; norm is replicated."""
        TP = self.tp_size
        Dk, Dv, Hk, Hv = self.head_k_dim, self.head_v_dim, self.num_k_heads, self.num_v_heads
        nkp, nvp = Hk // TP, Hv // TP
        self.nkp, self.nvp = nkp, nvp
        SH3 = ttnn.ShardTensorToMesh(self.device, dim=3)

        def interleave_qkv(flat):  # flat: [H, conv_dim] = [q(key_dim)|k(key_dim)|v(value_dim)]
            q = flat[:, :self.key_dim].reshape(-1, Hk, Dk)
            k = flat[:, self.key_dim:2 * self.key_dim].reshape(-1, Hk, Dk)
            v = flat[:, 2 * self.key_dim:].reshape(-1, Hv, Dv)
            blocks = []
            for c in range(TP):
                blocks.append(torch.cat([
                    q[:, c * nkp:(c + 1) * nkp, :].reshape(q.shape[0], -1),
                    k[:, c * nkp:(c + 1) * nkp, :].reshape(k.shape[0], -1),
                    v[:, c * nvp:(c + 1) * nvp, :].reshape(v.shape[0], -1),
                ], dim=1))
            return torch.cat(blocks, dim=1)  # [H_or_rows, conv_dim] per-chip interleaved

        # in_proj_qkv: [conv_dim, H] -> T -> [H, conv_dim] -> interleaved -> shard dim3
        qkv_w = state_dict[f"{prefix}.linear_attn.in_proj_qkv.weight"].T.contiguous()
        self.in_proj_qkv_sh = ttnn.from_torch(
            interleave_qkv(qkv_w).unsqueeze(0).unsqueeze(0), dtype=proj_dtype,
            layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=SH3)
        # conv1d weight columns: [conv_dim, K] -> per-tap [1,1,1,conv_dim] interleaved -> shard dim3
        self.conv_w_cols_sh = [
            ttnn.from_torch(
                interleave_qkv(self.conv1d_weight[:, k].reshape(1, -1)).reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=SH3)
            for k in range(self.conv_kernel_size)
        ]
        # z / b / a / A_log / dt_bias: head-aligned -> plain dim3 shard
        self.in_proj_z_sh = self._load_weight(state_dict, f"{prefix}.linear_attn.in_proj_z.weight", proj_dtype, transpose=True, shard_dim=3)
        self.in_proj_b_sh = self._load_weight(state_dict, f"{prefix}.linear_attn.in_proj_b.weight", proj_dtype, transpose=True, shard_dim=3)
        self.in_proj_a_sh = self._load_weight(state_dict, f"{prefix}.linear_attn.in_proj_a.weight", proj_dtype, transpose=True, shard_dim=3)
        self.A_log_sh = self._load_weight(state_dict, f"{prefix}.linear_attn.A_log", dtype, shard_dim=3)
        self.dt_bias_sh = self._load_weight(state_dict, f"{prefix}.linear_attn.dt_bias", dtype, shard_dim=3)
        # out_proj already row-parallel (dim2, head-aligned); norm_weight replicated — reuse both.

    def _load_weight(self, state_dict, key, dtype, transpose=False, shard_dim=None):
        if key not in state_dict:
            return None
        w = state_dict[key]
        if isinstance(w, torch.Tensor):
            if w.dim() == 1:
                w = w.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif w.dim() == 2:
                if transpose:
                    w = w.T.contiguous()
                w = w.unsqueeze(0).unsqueeze(0)
            mapper = ttnn.ShardTensorToMesh(self.device, dim=shard_dim) if shard_dim is not None else None
            return ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=mapper)
        return w

    # ---- TP collectives ----
    def _tp_gather(self, x):
        """Reconstruct a full activation from a column-sharded linear output.
        Pads the token dim to a tile for the collective, then slices back."""
        if self.dense_tp:
            S = int(x.shape[-2])
            Sp = ((S + 31) // 32) * 32
            if Sp != S:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, Sp - S), (0, 0)], value=0.0)
            g = ttnn.all_gather(x, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
            if Sp != S:
                g = ttnn.slice(g, [0, 0, 0, 0], [int(g.shape[0]), int(g.shape[1]), S, int(g.shape[-1])])
            return g
        return x

    def _out_proj(self, x):
        """out_proj with row(input)-parallel weight: mesh_partition the (full)
        input, matmul local rows, all_reduce the partials."""
        if self.dense_tp:
            S = int(x.shape[-2])
            Sp = ((S + 31) // 32) * 32
            if Sp != S:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, Sp - S), (0, 0)], value=0.0)
            xs = ttnn.mesh_partition(x, 3)
            p = ttnn.linear(xs, self.out_proj_w)
            ttnn.deallocate(xs)
            out = ttnn.all_reduce(p, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
            if Sp != S:
                out = ttnn.slice(out, [0, 0, 0, 0], [int(out.shape[0]), int(out.shape[1]), S, int(out.shape[-1])])
            return out
        return ttnn.linear(x, self.out_proj_w)

    @staticmethod
    def _to_batch_major(x, B):
        if B == 1:
            return x
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        W = rm.shape[-1]
        rm = ttnn.reshape(rm, [B, 1, 1, W])
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    @staticmethod
    def _to_seq_major(x, B):
        if B == 1:
            return x
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        W = rm.shape[-1]
        rm = ttnn.reshape(rm, [1, 1, B, W])
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def _precompute_beta_decay(self, b_proj, a_proj):
        """Compute the DeltaNet gate scalars host-side with vectorized ttnn ops
        so the reader kernel does NOT run expf/logf on the NCRISC dataflow core
        (its local data region is tiny on Wormhole and the libm lookup tables
        overflow it). The fused readers now read these precomputed values from
        the b_proj / a_proj op slots instead of A_log/dt_bias.

          beta  = sigmoid(b_proj)
          decay = exp(-exp(A_log) * softplus(a_proj + dt_bias))

        b_proj/a_proj are [1,1,*,H]; A_log/dt_bias are [1,1,1,H] (broadcast over
        the token/batch dim). All elementwise + per-head → TP-replication safe.
        Consumes b_proj/a_proj; returns (beta, decay) with the same shapes."""
        beta = ttnn.sigmoid(b_proj)
        ttnn.deallocate(b_proj)

        a_dt = ttnn.add(a_proj, self.dt_bias_bf16)
        ttnn.deallocate(a_proj)
        sp = ttnn.softplus(a_dt, beta=1.0, threshold=20.0)
        ttnn.deallocate(a_dt)
        exp_A = ttnn.exp(self.A_log_bf16)
        scaled = ttnn.mul(exp_A, sp)  # [1,1,1,H] * [1,1,*,H] -> [1,1,*,H]
        ttnn.deallocate(exp_A)
        ttnn.deallocate(sp)
        decay = ttnn.exp(ttnn.neg(scaled))
        ttnn.deallocate(scaled)
        return beta, decay

    def _l2norm_per_head_dev(self, x_flat, H, D, scale=None):
        """On-device per-head L2norm of [1,1,1,H*D] over each D-vector; optional scale.
        Returns [1,1,1,H*D]. The [1,1,1,H*D] <-> [1,1,H,D] reshape is done in
        ROW_MAJOR (a pure contiguous view); doing it in TILE layout mis-orders data
        across tile boundaries (heads get scrambled)."""
        xr = ttnn.to_layout(x_flat, ttnn.ROW_MAJOR_LAYOUT)
        xh = ttnn.reshape(xr, [1, 1, H, D])
        ttnn.deallocate(xr)
        xh = ttnn.to_layout(xh, ttnn.TILE_LAYOUT)
        sq = ttnn.mul(xh, xh)
        s = ttnn.sum(sq, dim=-1, keepdim=True)  # [1,1,H,1]
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(ttnn.add(s, 1e-6))
        ttnn.deallocate(s)
        out = ttnn.mul(xh, inv)  # broadcast over D
        ttnn.deallocate(xh)
        ttnn.deallocate(inv)
        if scale is not None:
            out2 = ttnn.mul(out, scale)
            ttnn.deallocate(out)
            out = out2
        outr = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(out)
        flat = ttnn.reshape(outr, [1, 1, 1, H * D])
        ttnn.deallocate(outr)
        return ttnn.to_layout(flat, ttnn.TILE_LAYOUT)

    def _seed_conv_hist(self, deltanet_state):
        """Create the FIXED conv-history buffers (h0,h1,h2) for this layer from the
        prefill conv_state (cols 1,2,3 = the 3 prior conv inputs); zeros if none.
        Stored in deltanet_state.conv_hist[layer_idx]. For trace, call this for all
        DeltaNet layers BEFORE capture so the buffers exist to snapshot/restore."""
        li = self.layer_idx
        conv_dim = self.key_dim * 2 + self.value_dim
        rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None

        def mk(vec):
            return ttnn.from_torch(vec.contiguous().reshape(1, 1, 1, -1).to(torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        cs = deltanet_state.get_conv_state_cpu(li, self.conv_kernel_size)  # [1,conv_dim,ck] or None
        if cs is not None:
            csq = cs.squeeze(0).float()
            cols = [csq[:, 1], csq[:, 2], csq[:, 3]]
        else:
            z = torch.zeros(conv_dim)
            cols = [z, z, z]
        existing = deltanet_state.conv_hist.get(li)
        if existing is not None:
            # in-place re-seed: keep the same buffers so a captured trace stays valid
            for buf, col in zip(existing, cols):
                tmp = mk(col)
                ttnn.copy(tmp, buf)
                ttnn.deallocate(tmp)
            return existing
        hist = [mk(c) for c in cols]
        deltanet_state.conv_hist[li] = hist
        return hist

    def _decode_conv_prep(self, qkv, deltanet_state, B):
        """On-device causal conv1d + SiLU + per-head L2norm (+ q scale) for decode
        (the deltanet_decode_full reader is PASS-THROUGH for qkv). Eliminates the
        per-layer CPU round-trip: conv via persistent history tensors (h0,h1,h2) and
        precomputed weight columns (dot = w0*h0+w1*h1+w2*h2+w3*x), all ttnn ops.
        Consumes `qkv` (raw projection becomes next step's history — caller must NOT
        deallocate it). Returns conv'd+normed [1,1,1,conv_dim]. B>1 falls back to CPU."""
        if B != 1 or self.conv_w_cols is None:
            return self._decode_conv_prep_cpu(qkv, deltanet_state, B)
        li = self.layer_idx
        conv_dim = self.key_dim * 2 + self.value_dim
        kd, Dk, Hk = self.key_dim, self.head_k_dim, self.num_k_heads
        rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None

        hist = deltanet_state.conv_hist.get(li)
        if hist is None:
            hist = self._seed_conv_hist(deltanet_state)
        h0, h1, h2 = hist  # fixed buffers
        w0, w1, w2, w3 = self.conv_w_cols
        dot = ttnn.add(ttnn.add(ttnn.mul(w0, h0), ttnn.mul(w1, h1)),
                       ttnn.add(ttnn.mul(w2, h2), ttnn.mul(w3, qkv)))
        conv = ttnn.silu(dot)
        ttnn.deallocate(dot)
        # shift history IN-PLACE into the same buffers: h0<-h1, h1<-h2, h2<-x (raw qkv).
        # Order matters (read-before-overwrite). qkv's value now lives in h2 -> free qkv.
        ttnn.copy(h1, h0)
        ttnn.copy(h2, h1)
        ttnn.copy(qkv, h2)
        ttnn.deallocate(qkv)
        # split q|k|v and per-head L2norm (q also scaled by 1/sqrt(Dk)); v unchanged
        q = ttnn.slice(conv, [0, 0, 0, 0], [1, 1, 1, kd])
        k = ttnn.slice(conv, [0, 0, 0, kd], [1, 1, 1, 2 * kd])
        v = ttnn.slice(conv, [0, 0, 0, 2 * kd], [1, 1, 1, conv_dim])
        ttnn.deallocate(conv)
        q = self._l2norm_per_head_dev(q, Hk, Dk, scale=self.scale)
        k = self._l2norm_per_head_dev(k, Hk, Dk)
        out = ttnn.concat([q, k, v], dim=3)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return out

    def _decode_conv_prep_cpu(self, qkv, deltanet_state, B):
        """CPU fallback (B>1): conv1d+SiLU+L2norm+q-scale via host round-trip.
        qkv: gathered [1,1,B,conv_dim] (replicated). Returns replicated [1,1,B,conv_dim]."""
        conv_dim = self.key_dim * 2 + self.value_dim
        ck = self.conv_kernel_size
        qkv_cpu = mesh_to_torch(qkv).reshape(-1, conv_dim)  # [B, conv_dim]
        cs = deltanet_state.get_conv_state_cpu(self.layer_idx, ck)  # [1, conv_dim, ck] or None
        outs = []
        for b in range(B):
            qkv_b = qkv_cpu[b].float()  # [conv_dim]
            if cs is not None:
                cstate = torch.roll(cs.squeeze(0).clone(), shifts=-1, dims=-1)  # [conv_dim, ck]
                cstate[:, -1] = qkv_b
                qkv_conv = (cstate * self.conv1d_weight).sum(dim=-1)  # [conv_dim]
                new_cs = cstate
            else:
                new_cs = qkv_b.unsqueeze(-1).expand(-1, ck).clone()
                qkv_conv = qkv_b
            qkv_conv = torch.nn.functional.silu(qkv_conv)
            q, k, v = torch.split(qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = self._l2norm_cpu(q.reshape(self.num_k_heads, self.head_k_dim), dim=-1) * self.scale
            k = self._l2norm_cpu(k.reshape(self.num_k_heads, self.head_k_dim), dim=-1)
            outs.append(torch.cat([q.reshape(-1), k.reshape(-1), v], dim=-1))  # [conv_dim]
            deltanet_state.set_conv_state_from_cpu(self.layer_idx, new_cs.unsqueeze(0))
        out = torch.stack(outs, dim=0).reshape(1, 1, B, conv_dim).to(torch.bfloat16)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        ttnn.deallocate(qkv)
        return ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=self.device, mesh_mapper=mapper)

    def _decode_step_full_fused(self, hidden_states, deltanet_state):
        """Single-token (batched) decode via the fully fused DeltaNet kernel.
        hidden_states: [1,1,B,H]."""
        B = hidden_states.shape[2]

        qkv = self._tp_gather(ttnn.linear(hidden_states, self.in_proj_qkv_w))
        z = self._tp_gather(ttnn.linear(hidden_states, self.in_proj_z_w))
        b_proj = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_proj = ttnn.linear(hidden_states, self.in_proj_a_w)
        # Reader kernel expects precomputed gate scalars in the b/a slots.
        beta_proj, decay_proj = self._precompute_beta_decay(b_proj, a_proj)

        H = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        # The deltanet_decode_full reader is PASS-THROUGH for qkv: it expects qkv
        # to ALREADY have causal conv1d + SiLU + per-head L2norm (+ q scale) applied
        # host-side (see reader_deltanet_full.cpp). The prefill reader does conv
        # on-core, but the decode reader does NOT — so we apply it here, mirroring
        # the single-device deltanet.py::_decode_step_fused. Without this, decode
        # feeds raw projections to the recurrence (the decode-gibberish bug).
        # q/k stay in num_k_heads (16) layout; the kernel expands to 48 v-heads.
        # NOTE: _decode_conv_prep owns `qkv`'s lifetime (on-device path keeps the raw
        # projection as next step's conv history; CPU fallback frees it). Don't free here.
        qkv = self._decode_conv_prep(qkv, deltanet_state, B)

        qkv_k = self._to_batch_major(qkv, B)
        z_k = self._to_batch_major(z, B)
        b_k = self._to_batch_major(beta_proj, B)
        a_k = self._to_batch_major(decay_proj, B)

        conv_state = deltanet_state.get_conv_state(self.layer_idx)
        state = deltanet_state.get_recurrent_state(self.layer_idx)

        output_tt, new_state, new_conv_state = _deltanet_decode_full_op(
            qkv_k,
            z_k,
            b_k,
            a_k,
            conv_state,
            state,
            self.conv1d_weight_tt,
            self.A_log_bf16,
            self.dt_bias_bf16,
            self.norm_weight,
            num_heads=H,
            num_k_heads=self.num_k_heads,
            k_head_dim=Dk,
            v_head_dim=Dv,
            conv_dim=self.key_dim * 2 + self.value_dim,
            conv_kernel_size=self.conv_kernel_size,
            head_expand_ratio=self.head_expand_ratio,
            # NOTE: the compiled deltanet_decode_full op has no `batch` kwarg;
            # batch is derived from tensor shapes (B is encoded in the
            # batch-major qkv/state shapes). Passing batch=B raises TypeError.
        )

        deltanet_state.set_recurrent_state(self.layer_idx, new_state)
        # NOTE: do NOT set_conv_state(new_conv_state) — the decode reader is
        # pass-through and returns the conv_state unchanged; _decode_conv_prep
        # already rolled+updated the conv state host-side this step. Overwriting
        # with the stale op output would revert that update.

        output_tt = self._to_seq_major(output_tt, B)  # [B,1,1,H*Dv] -> [1,1,B,H*Dv]
        return self._out_proj(output_tt)

    def _prefill_fused(self, hidden_states, deltanet_state):
        """Multi-token (B=1) prefill via the fused device kernel."""
        S = hidden_states.shape[2]
        H = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        qkv_all = self._tp_gather(ttnn.linear(hidden_states, self.in_proj_qkv_w))
        z_all = self._tp_gather(ttnn.linear(hidden_states, self.in_proj_z_w))
        b_all = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_all = ttnn.linear(hidden_states, self.in_proj_a_w)
        # Reader kernel expects precomputed gate scalars in the b/a slots.
        beta_all, decay_all = self._precompute_beta_decay(b_all, a_all)

        conv_state = deltanet_state.get_conv_state(self.layer_idx)
        state = deltanet_state.get_recurrent_state(self.layer_idx)

        results = _deltanet_prefill_full_op(
            qkv_all,
            z_all,
            beta_all,
            decay_all,
            conv_state,
            state,
            self.conv1d_weight_tt,
            self.A_log_bf16,
            self.dt_bias_bf16,
            self.norm_weight,
            num_heads=H,
            num_k_heads=self.num_k_heads,
            k_head_dim=Dk,
            v_head_dim=Dv,
            conv_dim=self.key_dim * 2 + self.value_dim,
            conv_kernel_size=self.conv_kernel_size,
            head_expand_ratio=self.head_expand_ratio,
            seq_len=S,
        )
        output_tt, new_state, new_conv_state = results[0], results[1], results[2]

        ttnn.deallocate(qkv_all)
        ttnn.deallocate(z_all)
        ttnn.deallocate(beta_all)
        ttnn.deallocate(decay_all)

        deltanet_state.set_recurrent_state(self.layer_idx, new_state)
        deltanet_state.set_conv_state(self.layer_idx, new_conv_state)

        out_rm = ttnn.to_layout(output_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(output_tt)
        out_reshaped = ttnn.reshape(out_rm, [1, 1, S, H * Dv])
        ttnn.deallocate(out_rm)
        out_tile = ttnn.to_layout(out_reshaped, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out_reshaped)

        return self._out_proj(out_tile)

    def forward(self, hidden_states, deltanet_state, mode="decode"):
        if mode == "decode":
            if USE_FULL_FUSED_KERNEL:
                return self._decode_step_full_fused(hidden_states, deltanet_state)
            raise RuntimeError(
                "qwen36 vLLM DeltaNet decode requires the fused deltanet_decode_full kernel "
                "(set DISABLE_* envs unset). The CPU fallback is not wired for the TP path."
            )
        else:
            if USE_PREFILL_FUSED_KERNEL:
                return self._prefill_fused(hidden_states, deltanet_state)
            raise RuntimeError(
                "qwen36 vLLM DeltaNet prefill requires the fused deltanet_prefill_full kernel."
            )

    @staticmethod
    def _l2norm_cpu(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
