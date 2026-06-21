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

    def __init__(self, num_layers, layer_types, device, config, batch=1, batched=None):
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
        # batched (continuous batching): batch ROWS whose request just (re)prefilled and
        # whose sharded conv_hist row must be reseeded (preserving other rows' accumulated
        # history) before the next decode step.
        self._reseed_rows = set()
        # When True, state updates write IN-PLACE into the fixed buffers (ttnn.copy)
        # instead of swapping the tensor handle — required for trace (static buffers).
        self.trace_mode = False
        # num_seq>1: recurrent state is head-parallel SHARDED (dim1: num_v_heads -> nvp
        # per chip) so the batched decode op reads each chip's nvp heads. conv_states
        # stay gathered (conv_hist derives from them, re-laid per-chip at seed time).
        # `batched` overrides config (the prefill TEMP state must stay GATHERED even when
        # the persistent decode state is sharded — prefill uses the gathered op).
        self.batched = bool(getattr(config, "batched_decode", False)) if batched is None else bool(batched)
        tp = getattr(config, "tp_size", 8)
        rs_mapper = ttnn.ShardTensorToMesh(device, dim=1) if (self.batched and getattr(config, "dense_tp", False)) else None

        for i in range(num_layers):
            if layer_types[i] == "linear_attention":
                num_v_heads = config.linear_num_value_heads
                k_dim = config.linear_key_head_dim
                v_dim = config.linear_value_head_dim
                conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 + v_dim * num_v_heads

                self.recurrent_states[i] = ttnn.from_torch(
                    torch.zeros(batch, num_v_heads, k_dim, v_dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rs_mapper,
                ) if rs_mapper is not None else ttnn.zeros(
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
        # batched: rs is sharded dim1, conv_hist sharded dim3 -> gather those dims to host.
        rdim = 1 if self.batched else None
        hdim = 3 if self.batched else None
        return {
            "recur": {li: mesh_to_torch(rs, dim=rdim).clone() for li, rs in self.recurrent_states.items()},
            "hist": {li: [mesh_to_torch(t, dim=hdim).clone() for t in h] for li, h in self.conv_hist.items()},
        }

    def restore_decode_state(self, snap):
        """Restore the snapshot IN-PLACE into the existing fixed buffers (so captured
        trace buffer addresses stay valid)."""
        rep = ttnn.ReplicateTensorToMesh(self.device)
        rs_map = ttnn.ShardTensorToMesh(self.device, dim=1) if self.batched else rep
        h_map = ttnn.ShardTensorToMesh(self.device, dim=3) if self.batched else rep
        for li, rs_cpu in snap["recur"].items():
            tmp = ttnn.from_torch(rs_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=self.device, mesh_mapper=rs_map)
            ttnn.copy(tmp, self.recurrent_states[li])
            ttnn.deallocate(tmp)
        for li, hs in snap["hist"].items():
            for k, t_cpu in enumerate(hs):
                tmp = ttnn.from_torch(t_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=self.device, mesh_mapper=h_map)
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

    # ================= batched (num_seq>1) head-parallel SHARDED decode =================
    @staticmethod
    def _flatten_B(x):
        """[1,1,B,W] -> [1,1,1,B*W] (flatten batch into the channel dim, ROW_MAJOR)."""
        B, W = int(x.shape[2]), int(x.shape[3])
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        xr = ttnn.reshape(xr, [1, 1, 1, B * W])
        out = ttnn.to_layout(xr, ttnn.TILE_LAYOUT)
        ttnn.deallocate(xr)
        return out

    def _seed_conv_hist_sharded(self, deltanet_state, B, rows=None, inplace=False):
        """Build/reseed per-chip per-batch conv history h0,h1,h2 = [1,1,B,cdp] from the
        (gathered) prefill conv_states [B,1,conv_dim,32]. conv_dim channels are re-laid
        PER-CHIP-INTERLEAVED [q_c|k_c|v_c] and sharded (matching in_proj_qkv_sh); cols 1,2,3
        are the 3 prior conv inputs. rows=None -> build ALL B rows fresh. rows=set ->
        reseed ONLY those batch rows in the existing conv_hist (preserve the others'
        accumulated decode history) — for continuous batching when a request joins."""
        li = self.layer_idx
        Dk, Dv, Hk, Hv, TP = self.head_k_dim, self.head_v_dim, self.num_k_heads, self.num_v_heads, self.tp_size
        nkp, nvp = self.nkp, self.nvp
        cs = mesh_to_torch(deltanet_state.conv_states.get(li)).float()  # [B,1,conv_dim,32]

        def interleave_rows(flat):  # [rows, conv_dim] -> per-chip-interleaved [rows, conv_dim]
            q = flat[:, :self.key_dim].reshape(-1, Hk, Dk)
            k = flat[:, self.key_dim:2 * self.key_dim].reshape(-1, Hk, Dk)
            v = flat[:, 2 * self.key_dim:].reshape(-1, Hv, Dv)
            blk = [torch.cat([q[:, c * nkp:(c + 1) * nkp].reshape(q.shape[0], -1),
                              k[:, c * nkp:(c + 1) * nkp].reshape(k.shape[0], -1),
                              v[:, c * nvp:(c + 1) * nvp].reshape(v.shape[0], -1)], dim=1) for c in range(TP)]
            return torch.cat(blk, dim=1)
        cols = [interleave_rows(cs[:, 0, :, j]) for j in (1, 2, 3)]  # each [B, conv_dim] interleaved
        SH3 = ttnn.ShardTensorToMesh(self.device, dim=3)
        existing = deltanet_state.conv_hist.get(li)
        if rows is None or existing is None:  # build all B rows fresh
            hist = [ttnn.from_torch(c.reshape(1, 1, B, -1).to(torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=SH3)
                    for c in cols]
            deltanet_state.conv_hist[li] = hist
            return hist
        # reseed only `rows`: rebuild each h_k per-row (new for rows, old slice otherwise)
        cdp_full = cols[0].shape[1]
        new_hist = []
        for k in range(3):
            parts = []
            for b in range(B):
                if b in rows:
                    parts.append(ttnn.from_torch(cols[k][b:b + 1].reshape(1, 1, 1, -1).to(torch.bfloat16),
                                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=SH3))
                else:
                    parts.append(ttnn.slice(existing[k], [0, 0, b, 0], [1, 1, b + 1, int(existing[k].shape[3])]))
            nh = ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]
            if inplace:
                # copy into the EXISTING fixed buffer (preserves trace-captured address)
                ttnn.copy(nh, existing[k])
                ttnn.deallocate(nh)
            else:
                ttnn.deallocate(existing[k])
                new_hist.append(nh)
        if inplace:
            return existing
        deltanet_state.conv_hist[li] = new_hist
        return new_hist

    def _conv_prep_sharded(self, qkv, deltanet_state, B):
        """Per-chip per-batch conv1d+SiLU+L2norm. qkv [1,1,B,cdp] per chip (per-chip-
        interleaved channels). Returns op-layout qkvF [1,1,1,B*cdp] (B flattened into
        the head dim)."""
        li = self.layer_idx
        Dk, Dv, nkp, nvp = self.head_k_dim, self.head_v_dim, self.nkp, self.nvp
        kdp, vdp = nkp * Dk, nvp * Dv
        cdp = 2 * kdp + vdp
        hist = deltanet_state.conv_hist.get(li)
        if hist is None:
            hist = self._seed_conv_hist_sharded(deltanet_state, B)  # all rows (first decode)
        elif deltanet_state._reseed_rows:
            hist = self._seed_conv_hist_sharded(deltanet_state, B, rows=deltanet_state._reseed_rows)
        h0, h1, h2 = hist
        w0, w1, w2, w3 = self.conv_w_cols_sh  # [1,1,1,cdp] broadcast over B
        dot = ttnn.add(ttnn.add(ttnn.mul(w0, h0), ttnn.mul(w1, h1)),
                       ttnn.add(ttnn.mul(w2, h2), ttnn.mul(w3, qkv)))
        conv = ttnn.silu(dot)
        ttnn.deallocate(dot)
        ttnn.copy(h1, h0)
        ttnn.copy(h2, h1)
        ttnn.copy(qkv, h2)
        ttnn.deallocate(qkv)
        q = ttnn.slice(conv, [0, 0, 0, 0], [1, 1, B, kdp])
        k = ttnn.slice(conv, [0, 0, 0, kdp], [1, 1, B, 2 * kdp])
        v = ttnn.slice(conv, [0, 0, 0, 2 * kdp], [1, 1, B, cdp])
        ttnn.deallocate(conv)
        qf = self._l2norm_per_head_dev(self._flatten_B(q), B * nkp, Dk, scale=self.scale)
        kf = self._l2norm_per_head_dev(self._flatten_B(k), B * nkp, Dk)
        vf = self._flatten_B(v)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.concat([qf, kf, vf], dim=3)  # [1,1,1,B*cdp]
        ttnn.deallocate(qf)
        ttnn.deallocate(kf)
        ttnn.deallocate(vf)
        return out

    def _out_proj_sharded(self, x):
        """out_proj for sharded decode: x already per-chip [1,1,B,vdp] (the row-parallel
        input shard) -> local matmul -> all_reduce. (No mesh_partition.)"""
        out = ttnn.linear(x, self.out_proj_w, compute_kernel_config=self.config.matmul_kcfg())
        return ttnn.all_reduce(out, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)

    def _decode_step_sharded(self, hidden_states, deltanet_state):
        """Batched (num_seq>1) head-parallel sharded decode. Each chip owns nkp k-heads
        / nvp v-heads; the op runs num_heads=B*nvp (B flattened into the head dim).
        No all_gather; out_proj all_reduces."""
        B = hidden_states.shape[2]
        Dk, Dv, nkp, nvp = self.head_k_dim, self.head_v_dim, self.nkp, self.nvp
        kdp, vdp = nkp * Dk, nvp * Dv
        cdp = 2 * kdp + vdp
        qkv = ttnn.linear(hidden_states, self.in_proj_qkv_sh)   # [1,1,B,cdp]
        z = ttnn.linear(hidden_states, self.in_proj_z_sh)       # [1,1,B,vdp]
        bproj = ttnn.linear(hidden_states, self.in_proj_b_sh)   # [1,1,B,nvp]
        aproj = ttnn.linear(hidden_states, self.in_proj_a_sh)
        beta = ttnn.sigmoid(bproj)
        ttnn.deallocate(bproj)
        a_dt = ttnn.add(aproj, self.dt_bias_sh)
        ttnn.deallocate(aproj)
        sp = ttnn.softplus(a_dt, beta=1.0, threshold=20.0)
        ttnn.deallocate(a_dt)
        expA = ttnn.exp(self.A_log_sh)
        scaled = ttnn.mul(expA, sp)
        ttnn.deallocate(expA)
        ttnn.deallocate(sp)
        decay = ttnn.exp(ttnn.neg(scaled))
        ttnn.deallocate(scaled)
        qkvF = self._conv_prep_sharded(qkv, deltanet_state, B)  # [1,1,1,B*cdp]; consumes qkv
        zf = self._flatten_B(z)
        betaf = self._flatten_B(beta)
        decayf = self._flatten_B(decay)
        ttnn.deallocate(z)
        ttnn.deallocate(beta)
        ttnn.deallocate(decay)
        state = deltanet_state.get_recurrent_state(self.layer_idx)  # [B,nvp,Dk,Dv] per chip
        # vestigial conv_state + conv1d_weight (decode reader is pass-through); sized [.,.,B*cdp,32] per chip
        conv_dummy = self._sharded_conv_dummy(B)
        o, new_state, _ = _deltanet_decode_full_op(
            qkvF, zf, betaf, decayf, conv_dummy, state,
            conv_dummy, self.A_log_sh, self.dt_bias_sh, self.norm_weight,
            num_heads=B * nvp, num_k_heads=B * nkp, k_head_dim=Dk, v_head_dim=Dv,
            conv_dim=B * cdp, conv_kernel_size=self.conv_kernel_size, head_expand_ratio=self.head_expand_ratio,
        )
        deltanet_state.set_recurrent_state(self.layer_idx, new_state)
        for t in (qkvF, zf, betaf, decayf):
            ttnn.deallocate(t)
        o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.reshape(o, [1, 1, B, vdp])  # un-flatten B
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
        return self._out_proj_sharded(o)

    def _sharded_conv_dummy(self, B):
        """Vestigial sharded conv tensor [1,1,B*cdp_full,32] (the decode reader is
        pass-through, so conv_state/conv1d_weight content is unused) — built once."""
        if getattr(self, "_conv_dummy", None) is None or self._conv_dummy_B != B:
            conv_dim = self.key_dim * 2 + self.value_dim
            SH2 = ttnn.ShardTensorToMesh(self.device, dim=2)
            self._conv_dummy = ttnn.from_torch(
                torch.zeros(1, 1, B * conv_dim, 32, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=SH2)
            self._conv_dummy_B = B
        return self._conv_dummy

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

    def _prefill_chunked(self, hidden_states, deltanet_state, C=32):
        """Chunked-parallel gated delta-rule prefill via batched ttnn ops (no
        custom kernel). Conv1d+SiLU+L2norm + factored-form scalings done host-side
        (cheap, O(S)); the recurrence — the 91% prefill bottleneck — runs on device
        as matmuls batched over all 48 v-heads, S/C chunk steps. Math verified in
        chunked_prefill_pipeline_test.py (fp32 PCC=1.0, bf16 0.99999)."""
        Hk, Hv, Dk, Dv = self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim
        kd, vd = self.key_dim, self.value_dim
        ck = self.conv_kernel_size
        conv_dim = kd * 2 + vd
        S = hidden_states.shape[2]

        # ---- projections (reuse the fused path's gather) ----
        qkv_all = self._tp_gather(ttnn.linear(hidden_states, self.in_proj_qkv_w))
        z_all = self._tp_gather(ttnn.linear(hidden_states, self.in_proj_z_w))
        b_all = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_all = ttnn.linear(hidden_states, self.in_proj_a_w)
        beta_all, decay_all = self._precompute_beta_decay(b_all, a_all)

        # ---- host: conv1d(causal) + silu + l2norm(+q scale) + head-expand ----
        qkv = mesh_to_torch(qkv_all).float().reshape(S, conv_dim)
        ttnn.deallocate(qkv_all)
        w = self.conv1d_weight.float()  # [conv_dim, ck]
        xpad = torch.cat([torch.zeros(ck - 1, conv_dim), qkv], 0)
        conv = sum(w[:, j] * xpad[j:j + S] for j in range(ck))
        conv = torch.nn.functional.silu(conv)
        q, k, v = torch.split(conv, [kd, kd, vd], dim=-1)
        q = self._l2norm_cpu(q.reshape(S, Hk, Dk)) * self.scale
        k = self._l2norm_cpu(k.reshape(S, Hk, Dk))
        v = v.reshape(S, Hv, Dv)
        q = q.repeat_interleave(Hv // Hk, dim=1).permute(1, 0, 2).contiguous()  # [Hv,S,Dk]
        k = k.repeat_interleave(Hv // Hk, dim=1).permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()                                     # [Hv,S,Dv]
        z = mesh_to_torch(z_all).float().reshape(S, Hv, Dv).permute(1, 0, 2).contiguous()
        ttnn.deallocate(z_all)
        beta = mesh_to_torch(beta_all).float().reshape(S, Hv).T.contiguous()    # [Hv,S]
        decay = mesh_to_torch(decay_all).float().reshape(S, Hv).T.contiguous()

        # ---- host: pad S to multiple of C; factored per-token scalings ----
        Sp = ((S + C - 1) // C) * C
        nC = Sp // C
        def pad(t):  # [Hv,S,*] -> [Hv,Sp,*]
            return torch.nn.functional.pad(t, (0, 0, 0, Sp - S)) if t.dim() == 3 else \
                   torch.nn.functional.pad(t, (0, Sp - S))
        q, k, v, z = pad(q), pad(k), pad(v), pad(z)
        beta = pad(beta); decay = torch.nn.functional.pad(decay, (0, Sp - S), value=1.0)
        d = torch.cumprod(decay.reshape(Hv, nC, C), dim=2).reshape(Hv, Sp)      # chunk-reset
        dinv = 1.0 / d
        Kdec = (beta * d).unsqueeze(-1) * k     # [Hv,Sp,Dk]
        Kinv = dinv.unsqueeze(-1) * k
        Qd = d.unsqueeze(-1) * q
        rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        # HiFi4 + fp32 accumulation: the factored form amplifies by 1/d, so the
        # chained Neumann/recurrence matmuls need fp32 accumulation (device bf16
        # LoFi accumulation degrades later tokens; CPU ref accumulates in fp32).
        mmcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        mm = lambda a, b: ttnn.matmul(a, b, compute_kernel_config=mmcfg)
        # custom fused kernel path: one op runs the whole per-head chunk loop on-device
        # (no per-chunk host dispatch). Pre-scaled inputs match the kernel contract.
        if os.environ.get("QWEN36_CHUNKED_KERNEL"):
            Sp2 = Sp
            r3 = lambda t: t.reshape(Hv * Sp2, -1)                       # [Hv,Sp,D]->[Hv*Sp,D]
            d_exp = d.unsqueeze(-1).expand(-1, -1, Dv)
            beta_exp = beta.unsqueeze(-1).expand(-1, -1, Dv)
            dlast_per = d.reshape(Hv, nC, C)[:, :, -1]                   # [Hv,nC]
            dlast_full = dlast_per[:, :, None, None].expand(Hv, nC, C, Dv).reshape(Hv, Sp2, Dv)
            KiT_h = Kinv.transpose(-2, -1).reshape(Hv * Dk, Sp2)         # [Hv*Dk, Sp]
            ttup = lambda t: ttnn.from_torch(t.to(torch.bfloat16).contiguous(), dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
            S0k = deltanet_state.get_recurrent_state(self.layer_idx)     # [1,Hv,Dk,Dv]
            res = ttnn.experimental.deltanet_prefill_chunked(
                ttup(r3(k)), ttup(r3(q)), ttup(r3(v)), ttup(r3(z)),
                ttup(r3(Kdec)), ttup(KiT_h), ttup(r3(Qd)),
                ttup(r3(d_exp)), ttup(r3(beta_exp)), ttup(r3(dlast_full)),
                S0k, self.norm_weight,
                num_heads=Hv, k_head_dim=Dk, v_head_dim=Dv, chunk=C, n_chunks=nC, seq_len=S)
            okernel, new_state = res[0], res[1]
            deltanet_state.set_recurrent_state(self.layer_idx, new_state)
            # output [Hv*Sp,Dv] -> [Hv,Sp,Dv] -> [:S] -> [S,Hv,Dv] -> [1,1,S,Hv*Dv]
            o = ttnn.reshape(okernel, [Hv, Sp2, Dv])
            o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
            o = ttnn.slice(o, [0, 0, 0], [Hv, S, Dv]) if Sp2 != S else o
            o = ttnn.permute(o, [1, 0, 2])
            o = ttnn.reshape(o, [1, 1, S, Hv * Dv])
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            return self._out_proj(o)

        # const masks uploaded once; per-chunk inputs streamed inside the loop so the
        # full-S [Hv,S,D] tensors (~19GB at 256K) never materialize on-device.
        ones = torch.ones(C, C)
        tident = up(torch.eye(C).reshape(1, C, C).expand(Hv, C, C).contiguous())
        ttrils = up(torch.tril(ones, -1).reshape(1, C, C).expand(Hv, C, C).contiguous())
        ttrili = up(torch.tril(ones, 0).reshape(1, C, C).expand(Hv, C, C).contiguous())
        d_exp = d.unsqueeze(-1).expand(-1, -1, Dv)                              # host views
        beta_exp = beta.unsqueeze(-1).expand(-1, -1, Dv)

        S0 = deltanet_state.get_recurrent_state(self.layer_idx)  # [1,Hv,Dk,Dv] (zeros for temp)
        S0 = ttnn.reshape(S0, [Hv, Dk, Dv])

        # ---- device: chunked recurrence, batched over Hv, per-chunk streamed ----
        O_chunks = []
        for c in range(nC):
            sl = slice(c * C, (c + 1) * C)
            kc, qc, vc = up(k[:, sl]), up(q[:, sl]), up(v[:, sl])
            Kdc, Kic, Qdc = up(Kdec[:, sl]), up(Kinv[:, sl]), up(Qd[:, sl])
            dc, bc = up(d_exp[:, sl].contiguous()), up(beta_exp[:, sl].contiguous())  # [Hv,C,Dv]
            KiT = ttnn.transpose(Kic, -2, -1)                                  # [Hv,Dk,C]
            kS0 = mm(kc, S0)                                                    # [Hv,C,Dv]
            qS0 = mm(qc, S0)
            A = ttnn.mul(mm(Kdc, KiT), ttrils)                                 # [Hv,C,C]
            # Neumann inverse: inv=(I-A); P=A; 4x: P=P@P; inv=inv@(I+P)
            inv = ttnn.sub(tident, A)
            P = A
            for _ in range(4):
                P = mm(P, ttnn.clone(P))   # avoid self-aliased matmul operands
                inv = mm(inv, ttnn.add(tident, P))
            rhs = ttnn.mul(bc, ttnn.sub(vc, ttnn.mul(dc, kS0)))                # [Hv,C,Dv]
            U = mm(inv, rhs)
            M = ttnn.mul(mm(Qdc, KiT), ttrili)
            Oc = ttnn.add(ttnn.mul(dc, qS0), mm(M, U))                         # [Hv,C,Dv]
            O_chunks.append(Oc)
            if os.environ.get("QWEN36_CHUNKED_DEBUG") and self.layer_idx == 0 and c == 0:
                self._dbg = {n: mesh_to_torch(t).float() for n, t in
                             [("A", A), ("KiT", KiT), ("inv", inv), ("U", U), ("kS0", kS0)]}
            dlast = dc[:, C - 1:C]  # [Hv,1,Dv] last token's chunk-cumprod decay
            S0 = ttnn.mul(dlast, ttnn.add(S0, mm(KiT, U)))
            for t in (kc, qc, vc, Kdc, Kic, Qdc, dc, bc, KiT, kS0, qS0, A, inv, rhs, U, M):
                ttnn.deallocate(t)
        deltanet_state.set_recurrent_state(self.layer_idx, ttnn.reshape(S0, [1, Hv, Dk, Dv]))

        O = ttnn.concat(O_chunks, dim=1)[:, :S]                                # [Hv,S,Dv]
        # ---- device: gated RMSNorm ----
        var = ttnn.mean(ttnn.mul(O, O), dim=-1, keepdim=True)
        On = ttnn.mul(O, ttnn.rsqrt(ttnn.add(var, 1e-6)))
        On = ttnn.mul(On, self.norm_weight)  # [Dv] broadcast
        z_dev = ttnn.from_torch(z[:, :S].to(torch.bfloat16).contiguous(), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        out = ttnn.mul(On, ttnn.silu(z_dev))                                   # [Hv,S,Dv] (maybe rank-4)
        # ---- reshape [Hv,S,Dv] -> [1,1,S,Hv*Dv] (token-major, head-concat) ----
        out = ttnn.reshape(out, [Hv, S, Dv])                                   # ensure rank-3
        # permute+merge(Hv,Dv) in ROW_MAJOR: TILE-layout reshape scrambles heads
        # across tile boundaries (same gotcha as _l2norm_per_head_dev).
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.permute(out, [1, 0, 2])                                     # [S,Hv,Dv]
        out = ttnn.reshape(out, [1, 1, S, Hv * Dv])
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        if os.environ.get("QWEN36_CHUNKED_DEBUG") and self.layer_idx == 0:
            self._chunked_debug_pcc(out, q, k, v, z, beta, decay, S, Sp, C, Hv, Dk, Dv)
        return self._out_proj(out)

    def _chunked_debug_pcc(self, out, q, k, v, z, beta, decay, S, Sp, C, Hv, Dk, Dv):
        import torch as _t
        Oref = _t.zeros(Hv, Sp, Dv)
        for h in range(Hv):
            Sm = _t.zeros(Dk, Dv)
            for c0 in range(0, Sp, C):
                qc, kc, vc = q[h, c0:c0 + C], k[h, c0:c0 + C], v[h, c0:c0 + C]
                bcv, gcv = beta[h, c0:c0 + C], decay[h, c0:c0 + C]
                dd = _t.cumprod(gcv, 0); di = 1.0 / dd
                A = _t.tril(((bcv * dd)[:, None] * kc) @ (di[:, None] * kc).T, -1)
                kS = kc @ Sm
                U = _t.linalg.solve(_t.eye(C) + A, bcv[:, None] * (vc - dd[:, None] * kS))
                M = _t.tril((dd[:, None] * qc) @ (di[:, None] * kc).T, 0)
                Oref[h, c0:c0 + C] = dd[:, None] * (qc @ Sm) + M @ U
                Sm = dd[-1] * Sm + dd[-1] * ((di[:, None] * kc).T @ U)
        # localize: compare device A/KiT/inv/U (chunk0) vs host for head 0
        if hasattr(self, "_dbg"):
            h = 0
            qc, kc, vc = q[h, :C], k[h, :C], v[h, :C]
            bcv, gcv = beta[h, :C], decay[h, :C]
            dd = _t.cumprod(gcv, 0); di = 1.0 / dd
            KiT_h = (di[:, None] * kc).T                       # [Dk,C]
            A_h = _t.tril(((bcv * dd)[:, None] * kc) @ (di[:, None] * kc).T, -1)
            invm = _t.eye(C) - A_h; P = A_h
            for _ in range(4):
                P = P @ P; invm = invm @ (_t.eye(C) + P)
            U_h = invm @ (bcv[:, None] * (vc - dd[:, None] * (kc @ _t.zeros(self.head_k_dim, Dv))))
            def _p(a, b):
                return _t.corrcoef(_t.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()
            print(f"[chunkdbg] head0 A PCC={_p(A_h, self._dbg['A'][h]):.5f} "
                  f"KiT PCC={_p(KiT_h, self._dbg['KiT'][h]):.5f} inv PCC={_p(invm, self._dbg['inv'][h]):.5f} "
                  f"U PCC={_p(U_h, self._dbg['U'][h]):.5f}", flush=True)
            print(f"[chunkdbg]  A_h[2,:3]={A_h[2,:3].tolist()} dev={self._dbg['A'][h][2,:3].tolist()}", flush=True)
            print(f"[chunkdbg]  inv_h[3,:4]={invm[3,:4].tolist()} dev={self._dbg['inv'][h][3,:4].tolist()}", flush=True)
            del self._dbg
        nw = self.norm_weight_cpu.float()
        On = Oref * _t.rsqrt((Oref * Oref).mean(-1, keepdim=True) + 1e-6) * nw
        Oref = (On * _t.nn.functional.silu(z))[:, :S].permute(1, 0, 2).reshape(S, Hv * Dv)
        dev = mesh_to_torch(out).float().reshape(S, Hv * Dv)
        def _pcc(a, b):
            return _t.corrcoef(_t.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()
        refh = Oref.reshape(S, Hv, Dv); devh = dev.reshape(S, Hv, Dv)
        print(f"[chunkdbg] overall PCC={_pcc(Oref, dev):.5f}", flush=True)
        for h in [0, 1, 2, 24, 47]:
            print(f"[chunkdbg]  head {h:2d} PCC={_pcc(refh[:, h], devh[:, h]):.5f}", flush=True)
        for tk in range(min(S, 6)):
            print(f"[chunkdbg]  token {tk} PCC={_pcc(refh[tk], devh[tk]):.5f} "
                  f"dev[{tk},h0,:3]={devh[tk,0,:3].tolist()} ref={refh[tk,0,:3].tolist()}", flush=True)

    def _chunked_decay_vectors(self, decay_all, beta_all, C=32):
        """Host precompute for chunked prefill (factored form): from the per-token
        gate scalars decay/beta [1,1,S,H] produce d = chunk-reset cumprod(decay)
        and dinv = 1/d as [1,1,S,H] tensors. d resets to 1 at each chunk boundary
        (every C tokens), matching the kernel's per-chunk entering state S0.
        bd = beta*d is recomputed in the reader (cheap multiply). Validated:
        chunked_prefill_pipeline_test.py (fp32 PCC=1.0, bf16 0.99999)."""
        dec = mesh_to_torch(decay_all).float().reshape(-1, self.num_v_heads)  # [S,H]
        S = dec.shape[0]
        d = torch.empty_like(dec)
        for c0 in range(0, S, C):
            c1 = min(c0 + C, S)
            d[c0:c1] = torch.cumprod(dec[c0:c1], dim=0)
        dinv = 1.0 / d
        rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        mk = lambda t: ttnn.from_torch(t.reshape(1, 1, S, self.num_v_heads).to(torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=self.device, mesh_mapper=rep)
        return mk(d), mk(dinv)

    def forward(self, hidden_states, deltanet_state, mode="decode"):
        if mode == "decode":
            if USE_FULL_FUSED_KERNEL:
                if self.batched_decode:  # num_seq>1: head-parallel sharded batched decode
                    return self._decode_step_sharded(hidden_states, deltanet_state)
                return self._decode_step_full_fused(hidden_states, deltanet_state)
            raise RuntimeError(
                "qwen36 vLLM DeltaNet decode requires the fused deltanet_decode_full kernel "
                "(set DISABLE_* envs unset). The CPU fallback is not wired for the TP path."
            )
        else:
            if os.environ.get("QWEN36_CHUNKED_PREFILL"):
                return self._prefill_chunked(hidden_states, deltanet_state)
            if USE_PREFILL_FUSED_KERNEL:
                return self._prefill_fused(hidden_states, deltanet_state)
            raise RuntimeError(
                "qwen36 vLLM DeltaNet prefill requires the fused deltanet_prefill_full kernel."
            )

    @staticmethod
    def _l2norm_cpu(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
