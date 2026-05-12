# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B GatedDeltaNet block for BH GLX 8×4 mesh.

Mesh sharding strategy
-----------------------
Mesh is 8 rows × 4 cols (32 chips).
- 48 V-heads sharded across 8 rows → 6 V-heads/row  (n_v_per_row = 6)
- 16 K-heads sharded across 8 rows → 2 K-heads/row  (n_k_per_row = 2)
- Cols are replicated (all 4 cols on a row do identical work)
- Hidden dim H=5120 is replicated (matches residual-stream convention)

Weight layout per row
---------------------
QKV projections are split into per-row sub-weights (Bug4 fix):
  - Q row-weight : [H=5120, 2*128=256]   → head_k_dim * n_k_per_row
  - K row-weight : [H=5120, 256]
  - V row-weight : [H=5120, 6*128=768]   → head_v_dim * n_v_per_row
  - z row-weight : [H=5120, 768]         (gate, same sharding as V)
  - a row-weight : [H=5120, 6]           (n_v_per_row scalars)
  - b row-weight : [H=5120, 6]           (n_v_per_row scalars)
  - out_proj row : [768, 5120]           V-head chunk → hidden

Conv1d weight pre-interleaved (Bug1 fix):
  - Each row gets [Q_conv_i(256)|K_conv_i(256)|V_conv_i(768)] = 1280 channels

A_log and dt_bias are 1-D [48]; per-row slice [6] stored as [1, 1, 6] (Bug2 fix).

norm.weight [128] is replicated (per-head, same for all heads).

Reduction (Bug3 fix):
  out_proj partial [B, T, 5120] per row → ttnn.all_gather + fast_reduce_nc across rows.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    chunk_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_ttnn,
)


def _causal_conv1d_fir_mesh(
    x,
    w_per_tap,
    kernel_size,
    mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    conv_state=None,
):
    """Mesh-compatible depthwise causal conv1d + SiLU via FIR decomposition.

    FIR decomposition: output[t] = sum_k(w[k] * x[t-k])  (causal)

    This avoids ttnn.conv1d which calls device.arch() and fails on MeshDevice.
    Works with ROW_MAJOR for padding/slicing (avoids tile-alignment issues on T-dim).

    Args:
        x:           [B, T, D] TILE_LAYOUT on mesh device
        w_per_tap:   list of K TTNN tensors each [1, 1, D] bfloat16, sharded per row.
                     w_per_tap[0] = most-recent tap weight (delay-0 coefficient)
        kernel_size: K (e.g., 4)
        mesh_device: ttnn.MeshDevice (for zeros allocation)
        memory_config: memory config for outputs
        conv_state:  [B, K-1, D] TILE_LAYOUT or None. The last K-1 tokens from the
                     previous forward pass (for decode continuity). If None, uses zeros.

    Returns:
        (output [B, T, D] bfloat16 TILE_LAYOUT, new_conv_state [B, K-1, D] bfloat16 TILE_LAYOUT)
        new_conv_state contains the last K-1 tokens of x (to be used as conv_state
        for the next forward call).
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    # Convert x to ROW_MAJOR to allow non-tile-aligned slicing
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)

    # Build padded input in ROW_MAJOR: [B, T + K-1, D]
    # Use conv_state (last K-1 tokens) if provided, else zeros.
    if conv_state is not None:
        # conv_state: [B, K-1, D] ROW_MAJOR (exact shape, no tile padding).
        # Always make a new tensor so we can safely call pad.deallocate(True) below.
        if conv_state.layout == ttnn.TILE_LAYOUT:
            cs_rm = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
        else:
            cs_rm = conv_state
        pad = ttnn.slice(cs_rm, [0, 0, 0], [B, kernel_size - 1, D], memory_config=memory_config)
        if conv_state.layout == ttnn.TILE_LAYOUT:
            cs_rm.deallocate(True)
        # Note: cs_rm == conv_state in the ROW_MAJOR case, so don't deallocate cs_rm
        # (the caller still holds a reference to conv_state).
    else:
        pad_torch = torch.zeros(B, kernel_size - 1, D, dtype=torch.bfloat16)
        pad = ttnn.from_torch(
            pad_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    x_padded = ttnn.concat([pad, x_rm], dim=1, memory_config=memory_config)
    pad.deallocate(True)

    # Compute new conv state: last K-1 tokens of x
    if T >= kernel_size - 1:
        new_conv_state_rm = ttnn.slice(x_rm, [0, T - (kernel_size - 1), 0], [B, T, D], memory_config=memory_config)
    else:
        # T < K-1: take all of x, prepend (K-1-T) tokens from old pad
        # For simplicity: take last (K-1) tokens from x_padded before new x
        # x_padded[0:K-1] = old_pad; x_padded[K-1:K-1+T] = new x
        # new pad = x_padded[T : T + K-1]
        new_conv_state_rm = ttnn.slice(x_padded, [0, T, 0], [B, T + kernel_size - 1, D], memory_config=memory_config)
    # Keep new_conv_state in ROW_MAJOR to avoid tile-padding of the K-1 time dimension.
    new_conv_state = new_conv_state_rm  # [B, K-1, D] ROW_MAJOR (exact, no tile padding)
    x_rm.deallocate(True)
    # x_padded: [B, T + K-1, D] ROW_MAJOR

    out = None
    for k in range(kernel_size):
        # Match F.conv1d semantics (padding=0, pre-padded input):
        # out[t] = sum_k weight[k] * x_padded[t + k]
        # k=0 → x_padded[t+0] = start at 0, end = T
        # k=K-1 → x_padded[t+K-1] = start at K-1, end = T + K-1
        start = k
        end = k + T
        x_slice = ttnn.slice(x_padded, [0, start, 0], [B, end, D], memory_config=memory_config)
        # Convert to TILE_LAYOUT for multiply
        x_slice_tl = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT, memory_config=memory_config)
        x_slice.deallocate(True)
        term = ttnn.multiply(x_slice_tl, w_per_tap[k], memory_config=memory_config)
        x_slice_tl.deallocate(True)
        if out is None:
            out = term
        else:
            prev = out
            out = ttnn.add(prev, term, memory_config=memory_config)
            prev.deallocate(True)
            term.deallocate(True)

    x_padded.deallocate(True)
    return ttnn.silu(out, memory_config=memory_config), new_conv_state


class TtQwen36DeltaNet(LightweightModule):
    """Mesh-aware GatedDeltaNet block for Qwen3.6-27B on BH GLX 8×4 mesh.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        Full 8×4 mesh.
    args : TtQwen36ModelArgs
        Model configuration (cluster_shape, norm_eps, etc.).
    state_dict : dict
        Per-layer weights with keys:
          in_proj_qkv.weight [10240, 5120]
          in_proj_z.weight   [6144, 5120]
          in_proj_a.weight   [48, 5120]
          in_proj_b.weight   [48, 5120]
          conv1d.weight      [10240, 1, 4]
          A_log              [48]
          dt_bias            [48]
          norm.weight        [128]
          out_proj.weight    [5120, 6144]
    layer_num : int
        Layer index (for logging only).
    dtype : ttnn.DataType
        Activation dtype (default bfloat16).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args,
        state_dict: dict,
        layer_num: int = 0,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        # Mesh topology
        self.cluster_shape = list(mesh_device.shape)  # [8, 4]
        self.mesh_rows = self.cluster_shape[0]  # 8
        self.mesh_cols = self.cluster_shape[1]  # 4

        # Model dimensions
        self.hidden_size = args.dim  # 5120
        self.n_k_heads = args.linear_num_key_heads  # 16
        self.n_v_heads = args.linear_num_value_heads  # 48
        self.head_dim = args.linear_head_dim  # 128
        self.conv_kernel = args.linear_conv_kernel  # 4
        self.eps = args.norm_eps  # 1e-6

        # Per-row head counts
        assert (
            self.n_v_heads % self.mesh_rows == 0
        ), f"n_v_heads={self.n_v_heads} must be divisible by mesh_rows={self.mesh_rows}"
        assert (
            self.n_k_heads % self.mesh_rows == 0
        ), f"n_k_heads={self.n_k_heads} must be divisible by mesh_rows={self.mesh_rows}"
        self.n_k_per_row = self.n_k_heads // self.mesh_rows  # 2
        self.n_v_per_row = self.n_v_heads // self.mesh_rows  # 6
        self.q_per_row = self.n_k_per_row * self.head_dim  # 256
        self.v_per_row = self.n_v_per_row * self.head_dim  # 768
        self.conv_per_row = self.q_per_row + self.q_per_row + self.v_per_row  # 1280

        # Compute kernel config (reuse from llama_attention)
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Build and upload all weights
        self._build_weights(state_dict)

        # Store raw conv weights for diagnostic test (host-side)
        self._conv_w_host = self._build_conv_host_blocks(state_dict)

    # ------------------------------------------------------------------
    # Weight construction helpers
    # ------------------------------------------------------------------

    def _to_device(
        self,
        t: torch.Tensor,
        mapper,
        dtype=None,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """Upload a torch tensor to the mesh device."""
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=dtype or self.dtype,
            layout=layout,
            memory_config=memory_config,
            mesh_mapper=mapper,
        )

    def _build_conv_host_blocks(self, sd: dict) -> list:
        """Build list of per-row conv weights [1280, 4] on host (for diagnostic test).

        Each element is a [1280, 4] tensor representing the conv1d channels
        that belong to row i: [Q_conv_row_i | K_conv_row_i | V_conv_row_i].
        """
        conv_w = sd["conv1d.weight"].squeeze(1)  # [10240, 4]
        conv_Q = conv_w[: self.n_k_heads * self.head_dim]  # [2048, 4]
        conv_K = conv_w[self.n_k_heads * self.head_dim : 2 * self.n_k_heads * self.head_dim]  # [2048, 4]
        conv_V = conv_w[2 * self.n_k_heads * self.head_dim :]  # [6144, 4]

        blocks = []
        for i in range(self.mesh_rows):
            qc = conv_Q[i * self.q_per_row : (i + 1) * self.q_per_row]
            kc = conv_K[i * self.q_per_row : (i + 1) * self.q_per_row]
            vc = conv_V[i * self.v_per_row : (i + 1) * self.v_per_row]
            blocks.append(torch.cat([qc, kc, vc], dim=0))  # [1280, 4]
        return blocks

    def _build_weights(self, sd: dict):
        """Prepare all weights and upload to the mesh device.

        Key design decisions (per hard-won bug fixes):
        - Bug1: Conv1d channels pre-interleaved by row before ShardTensor2dMesh
        - Bug2: A_log/dt_bias reshaped to [1, 1, n_v_per_row] (3-D)
        - Bug4: QKV weight split by head-group before sharding, not uniform split
        """
        H = self.hidden_size  # 5120
        mesh_rows = self.mesh_rows
        hd_k = self.head_dim  # 128
        hd_v = self.head_dim  # 128
        n_k = self.n_k_heads  # 16
        n_v = self.n_v_heads  # 48
        n_k_per_row = self.n_k_per_row  # 2
        n_v_per_row = self.n_v_per_row  # 6
        q_per_row = self.q_per_row  # 256
        v_per_row = self.v_per_row  # 768

        row_shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.cluster_shape)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)

        # ------------------------------------------------------------------
        # 1. QKV projection weights (Bug4 fix: block-wise split, not uniform)
        # ------------------------------------------------------------------
        # in_proj_qkv: [10240, 5120] = [Q(2048)|K(2048)|V(6144), H]
        qkv_w = sd["in_proj_qkv.weight"]  # [10240, 5120]
        Q_w = qkv_w[: n_k * hd_k]  # [2048, 5120]
        K_w = qkv_w[n_k * hd_k : 2 * n_k * hd_k]  # [2048, 5120]
        V_w = qkv_w[2 * n_k * hd_k :]  # [6144, 5120]

        # For linear: x [B, T, H] @ W.T → W stored as [H, out] for ttnn.linear
        # Build per-row interleaved weight: [H, rows*(q_per_row+q_per_row+v_per_row)]
        # Then shard dim=1 across rows.
        # Actually: build [rows*q_per_row, H] for Q, [rows*q_per_row, H] for K, etc.
        # and shard dim=0 → each row gets its slice.
        # Shape after transpose: [H, out_dim]; shard dim=1 gives each row out_dim/rows cols.
        # We shard Q, K, V separately and broadcast through separate linears per row.

        # Approach: stack per-row slices along the shard dim so ShardTensor2dMesh works.
        # Q and K are in K-head space; shard uniformly (they're contiguous already).
        # V is in V-head space; shard uniformly.

        # Q: [2048, 5120] → shard dim=0 across 8 rows → each row gets [256, 5120]
        # Transpose to [5120, 2048] then shard dim=1 → each row gets [5120, 256]
        Q_w_T = Q_w.T.contiguous()  # [5120, 2048]
        K_w_T = K_w.T.contiguous()  # [5120, 2048]
        V_w_T = V_w.T.contiguous()  # [5120, 6144]
        Z_w_T = sd["in_proj_z.weight"].T.contiguous()  # [5120, 6144]

        # Shard along output dim (dim=1 of transposed weight [H, out_dim]) across rows.
        # dims=(1, None): shard tensor dim=1 across mesh ROWS (8), replicate across cols (4).
        # Bug1 fix: NOT dims=(None, 1) which would shard across cols (4).
        row_shard_out = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(1, None), mesh_shape=self.cluster_shape)

        self.w_q = self._to_device(Q_w_T, row_shard_out)  # per-row: [5120, 256]
        self.w_k = self._to_device(K_w_T, row_shard_out)  # per-row: [5120, 256]
        self.w_v = self._to_device(V_w_T, row_shard_out)  # per-row: [5120, 768]
        self.w_z = self._to_device(Z_w_T, row_shard_out)  # per-row: [5120, 768]

        # in_proj_a, in_proj_b: [48, 5120] → shard dim=0 across 8 rows → each row [6, 5120]
        a_w_T = sd["in_proj_a.weight"].T.contiguous()  # [5120, 48]
        b_w_T = sd["in_proj_b.weight"].T.contiguous()  # [5120, 48]
        self.w_a = self._to_device(a_w_T, row_shard_out)  # per-row: [5120, 6]
        self.w_b = self._to_device(b_w_T, row_shard_out)  # per-row: [5120, 6]

        # ------------------------------------------------------------------
        # 2. Conv1d weight (Bug1 fix: pre-interleave by row)
        # ------------------------------------------------------------------
        conv_w = sd["conv1d.weight"].squeeze(1)  # [10240, 4]
        conv_Q_w = conv_w[: n_k * hd_k]  # [2048, 4]
        conv_K_w = conv_w[n_k * hd_k : 2 * n_k * hd_k]  # [2048, 4]
        conv_V_w = conv_w[2 * n_k * hd_k :]  # [6144, 4]

        # Pre-interleave: for each row i, cat(Q_i, K_i, V_i) along channels
        chunks = []
        for i in range(mesh_rows):
            qc = conv_Q_w[i * q_per_row : (i + 1) * q_per_row]  # [256, 4]
            kc = conv_K_w[i * q_per_row : (i + 1) * q_per_row]  # [256, 4]
            vc = conv_V_w[i * v_per_row : (i + 1) * v_per_row]  # [768, 4]
            chunks.append(torch.cat([qc, kc, vc], dim=0))  # [1280, 4]
        conv_w_interleaved = torch.cat(chunks, dim=0)  # [8*1280=10240, 4]

        # Store per-tap conv weights for FIR decomposition:
        # conv_w_interleaved: [8*1280, 4] → per row [1280, 4]
        # We need K=4 tensors each [rows*1280, 1, 1] → shard to [1280, 1, 1] per row
        # Shape for FIR: [1, 1, D] per device for broadcasting with [B, T, D]
        row_shard_dim0 = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.cluster_shape)

        # conv_w_interleaved: [8*1280, 4], taps on dim=1
        self.conv_weight_taps = []
        for tap in range(self.conv_kernel):
            # Extract tap k from all rows: [8*1280] → reshape to [1, 1, 8*1280]
            # After sharding dim=2: each row gets [1, 1, 1280]
            tap_vec = conv_w_interleaved[:, tap]  # [8*1280]
            # Reshape to [1, 1, 8*1280] and shard dim=2 across rows
            tap_3d = tap_vec.reshape(1, 1, mesh_rows * self.conv_per_row)  # [1, 1, 10240]
            # Bug1 fix: shard dim=2 across ROWS (8), not across cols.
            # dims=(2, None): shard tensor dim=2 across mesh rows (8), replicate across cols.
            row_shard_3d_chan = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, None), mesh_shape=self.cluster_shape)
            tap_tt = ttnn.from_torch(
                tap_3d,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=row_shard_3d_chan,
            )  # per-row: [1, 1, 1280]
            self.conv_weight_taps.append(tap_tt)

        # ------------------------------------------------------------------
        # 3. A_log, dt_bias (Bug2 fix: 3-D [1, 1, n_v_per_row])
        # ------------------------------------------------------------------
        A_log = sd["A_log"]  # [48]
        dt_bias = sd["dt_bias"]  # [48]

        # Shard [48] across 8 rows → each row [6], stored as [1, 1, 6]
        A_log_3d = A_log.reshape(1, 1, n_v)  # [1, 1, 48]
        dt_bias_3d = dt_bias.reshape(1, 1, n_v)  # [1, 1, 48]

        # Bug3 fix: dims=(2, None) shards tensor dim=2 across mesh ROWS (8).
        # NOT dims=(None, 2) which would shard across cols (4).
        row_shard_3d = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, None), mesh_shape=self.cluster_shape)

        self.A_log = self._to_device(A_log_3d, row_shard_3d, layout=ttnn.TILE_LAYOUT)  # per-row: [1, 1, 6]
        self.dt_bias = self._to_device(dt_bias_3d, row_shard_3d, layout=ttnn.TILE_LAYOUT)  # per-row: [1, 1, 6]

        # ------------------------------------------------------------------
        # 4. norm.weight [128] — standard RMSNorm, replicated
        # ------------------------------------------------------------------
        norm_w = sd["norm.weight"]  # [128] — STANDARD (no +1 shift, Bug5 fix)
        norm_w_4d = norm_w.reshape(1, 1, self.head_dim // 32, 32)
        self.norm_weight = self._to_device(
            norm_w_4d,
            replicate,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # ------------------------------------------------------------------
        # 5. out_proj [5120, 6144] — shard input dim (6144) across rows
        # ------------------------------------------------------------------
        # out_proj.weight: [5120, 6144] = [H, n_v * hd_v]
        # For partial reduce: each row computes [B, T, 5120] partial.
        # We shard the input dimension: each row handles its V-head chunk.
        # ttnn.linear(x[B,T,768], W[768, 5120]) = [B, T, 5120] partial.
        out_proj_w = sd["out_proj.weight"]  # [5120, 6144]
        # Transpose to [6144, 5120]; shard dim=0 across rows → each row [768, 5120]
        out_proj_w_T = out_proj_w.T.contiguous()  # [6144, 5120]

        row_shard_out0 = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.cluster_shape)
        self.w_out = self._to_device(out_proj_w_T, row_shard_out0)  # per-row: [768, 5120]

    # ------------------------------------------------------------------
    # Diagnostic accessor (for test_deltanet_sharding_correctness)
    # ------------------------------------------------------------------

    def get_conv_weight_row(self, row_i: int) -> torch.Tensor:
        """Return host-side conv weight block for row_i [1280, 4]."""
        return self._conv_w_host[row_i]

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        mode: str = "prefill",
        recurrent_state: ttnn.Tensor | None = None,
        conv_state: ttnn.Tensor | None = None,
        return_state: bool = False,
    ):
        """DeltaNet forward pass.

        Parameters
        ----------
        hidden_states : ttnn.Tensor
            [B, T, H=5120] replicated across all mesh devices.
        mode : str
            "prefill" (T>1, chunked kernel) or "decode" (T=1, recurrent kernel).
        recurrent_state : ttnn.Tensor or None
            [B, n_v_per_row, K=128, V=128] sharded per row.
            If None, initialised to zeros on-device.
        conv_state : ttnn.Tensor or None
            [B, K-1, conv_per_row] per row — last K-1 tokens from the previous
            forward pass, for causal conv continuity between prefill and decode.
            If None, zero history is assumed (correct for fresh sequences).
        return_state : bool
            If True, return (output, new_recurrent_state, new_conv_state).

        Returns
        -------
        output : ttnn.Tensor
            [B, T, H=5120] replicated across mesh (after all_reduce over rows).
        new_recurrent_state : ttnn.Tensor (only if return_state=True)
        new_conv_state : ttnn.Tensor (only if return_state=True)
            [B, K-1, conv_per_row] — last K-1 tokens of the conv input, to be
            passed as conv_state on the next forward call.
        """
        if mode == "prefill":
            return self._forward_prefill(hidden_states, recurrent_state, conv_state, return_state)
        else:
            return self._forward_decode(hidden_states, recurrent_state, conv_state, return_state)

    def _project_inputs(self, x: ttnn.Tensor, T: int, B: int):
        """Run all linear projections and return per-row activations.

        Returns:
          q   [B, T, 256]  — Q per row
          k   [B, T, 256]  — K per row
          v   [B, T, 768]  — V per row
          z   [B, T, 768]  — gate per row
          a   [B, T, 6]    — a per row
          b   [B, T, 6]    — b per row
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        ck = self.compute_kernel

        q = ttnn.linear(x, self.w_q, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        k = ttnn.linear(x, self.w_k, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        v = ttnn.linear(x, self.w_v, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        z = ttnn.linear(x, self.w_z, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        a = ttnn.linear(x, self.w_a, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        b = ttnn.linear(x, self.w_b, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)

        return q, k, v, z, a, b

    def _get_single_device(self, chip_row: int = 0) -> "ttnn.Device":
        """Get the device object for the first chip on the given row."""
        # For single-device kernel calls within the mesh context,
        # we operate on the multi-device tensor directly — the device
        # parameter is used for buffer allocation in the kernels.
        # Use mesh_device as a stand-in (the kernels will broadcast).
        return self.mesh_device

    def _apply_conv_and_split(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        B: int,
        T: int,
        conv_state=None,
    ):
        """Concatenate q|k|v → apply depthwise causal conv1d → split back.

        Each row processes its own [B, T, 1280] = [Q_row|K_row|V_row].
        Uses FIR decomposition for mesh-compatibility (Bug from prior attempt:
        ttnn.conv1d uses device.arch() which fails on MeshDevice).

        Args:
            conv_state: [B, K-1, 1280] TILE_LAYOUT or None. The last K-1 tokens from
                        the previous forward pass, for causal context continuity.

        Returns:
            (q_conv, k_conv, v_conv, new_conv_state)
            new_conv_state: [B, K-1, 1280] TILE_LAYOUT (last K-1 tokens of mixed).
        """
        mem = ttnn.DRAM_MEMORY_CONFIG

        # Concat q, k, v along channel dim → [B, T, 1280] per row
        mixed = ttnn.concat([q, k, v], dim=-1, memory_config=mem)

        # Apply mesh-compatible FIR causal conv1d + SiLU
        mixed_conv, new_conv_state = _causal_conv1d_fir_mesh(
            mixed,
            self.conv_weight_taps,
            self.conv_kernel,
            self.mesh_device,
            memory_config=mem,
            conv_state=conv_state,
        )  # [B, T, 1280]
        mixed.deallocate(True)

        # Split back to q, k, v with per-row sizes [256, 256, 768]
        q_conv = ttnn.slice(mixed_conv, [0, 0, 0], [B, T, self.q_per_row], memory_config=mem)
        k_conv = ttnn.slice(mixed_conv, [0, 0, self.q_per_row], [B, T, 2 * self.q_per_row], memory_config=mem)
        v_conv = ttnn.slice(mixed_conv, [0, 0, 2 * self.q_per_row], [B, T, self.conv_per_row], memory_config=mem)
        mixed_conv.deallocate(True)

        return q_conv, k_conv, v_conv, new_conv_state

    def _compute_beta_g(
        self,
        b: ttnn.Tensor,
        a: ttnn.Tensor,
        B: int,
        T: int,
    ):
        """Compute beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias).

        Returns:
          beta [B, T, n_v_per_row]
          g    [B, T, n_v_per_row]
        """
        mem = ttnn.DRAM_MEMORY_CONFIG

        # beta = sigmoid(b)
        beta = ttnn.sigmoid(b, memory_config=mem)  # [B, T, 6]

        # g = -exp(A_log) * softplus(a + dt_bias)
        # dt_bias: [1, 1, 6] (Bug2: 3-D shape for correct broadcast with 3-D a)
        a_biased = ttnn.add(a, self.dt_bias, memory_config=mem)  # [B, T, 6]
        sp = ttnn.softplus(a_biased, memory_config=mem)  # [B, T, 6]
        A_exp = ttnn.exp(self.A_log, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, 1, 6]
        g = ttnn.multiply(ttnn.neg(A_exp, memory_config=mem), sp, memory_config=mem)  # [B, T, 6]

        return beta, g

    def _gqa_expand_q_k(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        B: int,
        T: int,
    ):
        """Expand Q and K from n_k_per_row heads to n_v_per_row heads via repeat_interleave.

        q: [B, T, n_k_per_row, head_dim] → [B, T, n_v_per_row, head_dim]
        k: same
        """
        ratio = self.n_v_per_row // self.n_k_per_row  # 3
        mem = ttnn.DRAM_MEMORY_CONFIG

        q_e = ttnn.repeat_interleave(q, ratio, dim=2, memory_config=mem)
        k_e = ttnn.repeat_interleave(k, ratio, dim=2, memory_config=mem)
        return q_e, k_e

    def _apply_norm_gated(
        self,
        core_out: ttnn.Tensor,
        z: ttnn.Tensor,
        B: int,
        T: int,
    ) -> ttnn.Tensor:
        """GroupRMSNormGated: standard rms_norm per head_v_dim, then * silu(z).

        core_out: [B, T, n_v_per_row, head_v_dim]  (output from delta rule kernel)
        z:        [B, T, n_v_per_row, head_v_dim]
        Returns:  [B, T, v_per_row=768]

        Bug5 fix: STANDARD norm (w*x, no +1 shift); norm.weight is replicated as-is.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG

        n_v = self.n_v_per_row  # 6
        hd = self.head_dim  # 128

        # Flatten to [..., head_v_dim] for per-head norm
        # core_out: [B, T, n_v, hd] → reshape to [B*T*n_v, hd] implicitly via rms_norm
        # ttnn.rms_norm operates on the last dim; input needs to be [..., hd]
        out = ttnn.rms_norm(
            core_out,
            weight=self.norm_weight,
            epsilon=self.eps,
            memory_config=mem,
        )  # [B, T, n_v, hd]

        # Apply SiLU gate: z [B, T, n_v, hd]
        z_silu = ttnn.silu(z, memory_config=mem)
        out = ttnn.multiply(out, z_silu, memory_config=mem)
        z_silu.deallocate(True)

        # Flatten: [B, T, n_v, hd] → [B, T, v_per_row]
        out = ttnn.reshape(out, [B, T, self.v_per_row])
        return out

    def _output_proj_and_reduce(
        self,
        out_flat: ttnn.Tensor,
        B: int,
        T: int,
    ) -> ttnn.Tensor:
        """Apply output projection and sum-reduce across mesh rows.

        out_flat: [B, T, v_per_row=768] per row (partial output)
        Returns:  [B, T, H=5120] replicated across mesh
        """
        mem = ttnn.DRAM_MEMORY_CONFIG

        # out_proj: [B, T, 768] @ [768, 5120] = [B, T, 5120] partial per row
        partial = ttnn.linear(
            out_flat,
            self.w_out,
            dtype=self.dtype,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel,
        )  # [B, T, 5120]

        # All-reduce across rows (cluster_axis=0): gather partials + sum
        # Bug3 fix: use ttnn.all_gather + fast_reduce_nc (the validated approach)
        gathered = ttnn.all_gather(
            partial,
            dim=0,
            num_links=1,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            memory_config=mem,
        )  # [8*B, T, 5120] (each row gets all 8 row copies stacked on dim=0)

        reduced = ttnn.experimental.fast_reduce_nc(
            gathered,
            dims=[0],
            output=None,
            compute_kernel_config=None,
            memory_config=mem,
        )  # [B, T, 5120]
        gathered.deallocate(True)
        partial.deallocate(True)

        return reduced  # [B, T, H] replicated

    def _forward_prefill(
        self,
        x: ttnn.Tensor,
        recurrent_state=None,
        conv_state=None,
        return_state: bool = False,
    ):
        """Prefill (T>1) using chunked delta rule kernel."""
        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape

        mem = ttnn.DRAM_MEMORY_CONFIG

        # 1. Projections
        q, k, v, z, a, b = self._project_inputs(x, T, B)

        # 2. Conv1d + split (conv_state provides causal context for the conv)
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(q, k, v, B, T, conv_state)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

        # 3. Reshape to per-head layout
        q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_row, self.head_dim])  # [B,T,2,128]
        k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_row, self.head_dim])  # [B,T,2,128]
        v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_row, self.head_dim])  # [B,T,6,128]
        z_h = ttnn.reshape(z, [B, T, self.n_v_per_row, self.head_dim])  # [B,T,6,128]
        q_conv.deallocate(True)
        k_conv.deallocate(True)
        v_conv.deallocate(True)

        # 4. Beta and g
        beta, g = self._compute_beta_g(b, a, B, T)
        # Reshape to [B, T, n_v_per_row] for kernel
        # beta and g are already [B, T, 6]
        b.deallocate(True)
        a.deallocate(True)

        # 5. GQA expand q, k: n_k_per_row → n_v_per_row via repeat_interleave
        q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, B, T)
        q_h.deallocate(True)
        k_h.deallocate(True)
        # q_exp, k_exp: [B, T, n_v_per_row=6, head_dim=128]

        # 6. Chunked delta rule kernel (prefill)
        # IMPORTANT arg order for chunk: (q, k, v, beta, g) for ttnn — same as recurrent
        # Note: torch reference chunk has (q, k, v, g, beta) but TTNN has (q, k, v, beta, g)
        core_out, new_state = chunk_gated_delta_rule_ttnn(
            q=q_exp,
            k=k_exp,
            v=v_h,
            beta=beta,
            g=g,
            chunk_size=32,
            initial_state=recurrent_state,
            device=self.mesh_device,
        )
        # core_out: [B, T, n_v_per_row, head_v_dim]
        q_exp.deallocate(True)
        k_exp.deallocate(True)
        v_h.deallocate(True)
        beta.deallocate(True)
        g.deallocate(True)

        # 7. GroupRMSNormGated
        out = self._apply_norm_gated(core_out, z_h, B, T)
        core_out.deallocate(True)
        z_h.deallocate(True)

        # 8. Output projection + all-reduce across rows
        output = self._output_proj_and_reduce(out, B, T)
        out.deallocate(True)

        if return_state:
            return output, new_state, new_conv_state
        return output

    def _forward_decode(
        self,
        x: ttnn.Tensor,
        recurrent_state=None,
        conv_state=None,
        return_state: bool = False,
    ):
        """Decode (T=1) using recurrent delta rule kernel."""
        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape

        assert T == 1, f"Decode expects T=1, got T={T}"
        mem = ttnn.DRAM_MEMORY_CONFIG

        # 1. Projections
        q, k, v, z, a, b = self._project_inputs(x, T, B)

        # 2. Conv1d + split (conv_state provides causal context from prefill)
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(q, k, v, B, T, conv_state)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

        # 3. Reshape to per-head layout
        q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_row, self.head_dim])
        k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_row, self.head_dim])
        v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_row, self.head_dim])
        z_h = ttnn.reshape(z, [B, T, self.n_v_per_row, self.head_dim])
        q_conv.deallocate(True)
        k_conv.deallocate(True)
        v_conv.deallocate(True)

        # 4. Beta and g
        beta, g = self._compute_beta_g(b, a, B, T)
        b.deallocate(True)
        a.deallocate(True)

        # 5. GQA expand q, k
        q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, B, T)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # 6. Recurrent delta rule kernel (decode)
        # Arg order for ttnn recurrent: (q, k, v, beta, g)
        core_out, new_state = recurrent_gated_delta_rule_ttnn(
            q=q_exp,
            k=k_exp,
            v=v_h,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            device=self.mesh_device,
        )
        q_exp.deallocate(True)
        k_exp.deallocate(True)
        v_h.deallocate(True)
        beta.deallocate(True)
        g.deallocate(True)

        # 7. GroupRMSNormGated
        out = self._apply_norm_gated(core_out, z_h, B, T)
        core_out.deallocate(True)
        z_h.deallocate(True)

        # 8. Output projection + all-reduce across rows
        output = self._output_proj_and_reduce(out, B, T)
        out.deallocate(True)

        if return_state:
            return output, new_state, new_conv_state
        return output
