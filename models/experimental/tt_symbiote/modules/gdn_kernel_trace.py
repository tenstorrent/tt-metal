# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Trace-compatible GDN decode-step glue for Qwen3.5/3.6 linear attention.

This is the trace-compatible successor to gdn_kernel.py's `gdn_recurrent_step`.
The existing `gdn_recurrent_step` is correct but does host packing (`_pack_step`)
plus 5x ttnn.from_torch and 1x ttnn.to_torch per call, which breaks Metal
Trace capture. This module replaces those host crossings with TTNN-only ops
plus one new tt-lang pack kernel.

DESIGN (trace-safe pipeline):

  Replicated Q/K/V/g/beta (post-l2norm, post-repeat_interleave)
        |
        |  ttnn.gather(input, dim=2, index=per_device_head_idx)
        |    where per_device_head_idx is sharded along dim 2 with values
        |    [d*H_PER_DEV + 0, d*H_PER_DEV + 1, ...] on device d.
        v
  Head-sharded Q/K/V/g/beta  ([1, 1, H_PER_DEV, D] / [1, 1, H_PER_DEV])
        |
        |  ttnn.exp(g)   -> alpha
        v
  Head-sharded alpha
        |
        |  pack_natural_to_gdn_layout (THIS MODULE - tt-lang kernel, TODO)
        |    rewrites [1, 1, H_PER_DEV, D] -> [H_PER_DEV * D, TILE] with col 0 only
        v
  Pre-allocated kernel-input buffers (q_pack, k_pack, v_pack, alpha_pack, beta_pack)
        |
        |  gdn_step_4head / gdn_step_8head (existing, in gdn_kernel.py)
        v
  out_buf [H_PER_DEV * D_V, TILE], state_out_buf [H_PER_DEV * D_K, D_V]
        |
        |  unpack_gdn_to_natural_layout (THIS MODULE - tt-lang kernel, TODO)
        |    rewrites [H_PER_DEV * D_V, TILE] -> [1, 1, H_PER_DEV, D_V]
        v
  Head-sharded core_attn_out [1, 1, H_PER_DEV, D_V]   (TTNN, fed to ttnn_gated_rms_norm)

WHY HEAD-SHARDED, NOT REPLICATED:

  tt-lang kernels run independently on each device with no device-id intrinsic
  (`ttl.node()` is per-core, not per-device). Producing per-device-different
  output therefore requires per-device-different input. Going from
  replicated to head-sharded would require either:

    (a) A CCL op (e.g., `ttnn.experimental.all_to_all_async`), which assumes
        the input is already sharded along some dim,
    (b) Restructured projections so that Q/K/V emerge head-sharded,
    (c) A per-device-sharded index tensor + `ttnn.gather` -- THIS PATH.

  (c) is the cheapest: the index tensor is uploaded once at warm-up via
  `ShardTensorToMesh(dim=2)`, encoding the per-device head offset implicitly.
  Each `ttnn.gather` call extracts this device's 4 heads from the replicated
  input. No CCL, no projection refactor.

PER-MODULE STATE TO PRE-ALLOCATE (in TTNNQwen3LinearAttention.move_weights_to_device_impl):

  - _trace_head_idx_d128: [1, 1, H_PER_DEV, D_K=128] sharded dim=2, UINT32, TILE
                          Used as gather index for Q/K and (broadcast) for V.
  - _trace_head_idx_scalar: [1, 1, H_PER_DEV] sharded dim=2, UINT32, TILE
                            Used as gather index for g/beta.
  - q_pack, k_pack, v_pack: [H_PER_DEV * D, TILE] sharded dim=0, BF16, TILE
                            Pre-allocated kernel input buffers.
  - alpha_pack, beta_pack: [H_PER_DEV * TILE, TILE] sharded dim=0, BF16, TILE
  - state_a, state_b: [H_PER_DEV * D_K, D_V] sharded dim=0, BF16, TILE
                      Two state buffers for ping-pong (so steady-state decode
                      doesn't allocate).
  - out_a, out_b: [H_PER_DEV * D_V, TILE] sharded dim=0, BF16, TILE.

KERNEL TODO (not yet implemented):

  pack_natural_to_gdn_layout(q_local, k_local, v_local, alpha_local, beta_local,
                             q_pack, k_pack, v_pack, alpha_pack, beta_pack):
      For each output tile (t, 0) in q_pack (t = 0..H_PER_DEV*K_TILES-1):
          h = t // K_TILES
          kt = t % K_TILES
          q_pack[t, 0, c=0..31] = q_local[h, kt*32 + 0..31]
          q_pack[t, 0, c=1..31] = 0
      Same for k_pack, v_pack with their respective layouts.
      For alpha_pack (and beta_pack):
          alpha_pack[h*TILE..(h+1)*TILE, :] = alpha_local[h]   (scalar broadcast)

  unpack_gdn_to_natural_layout(out_buf, core_out):
      core_out[1, 1, h, d] = out_buf[h*D_V + d, 0]   (read col 0 only)

  Both kernels follow the existing gdn_step_*head pattern -- tt-lang
  @ttl.operation with grid=(H_PER_DEV, ?) and per-tile data movement.

CALLER (gdn_recurrent_step_traced - skeleton below):

  Drop-in replacement for gdn_recurrent_step. Takes TTNN tensors throughout,
  returns TTNN core_attn_out. Inside a Metal Trace capture region, the only
  ops are TTNN gathers, TTNN.exp, the tt-lang pack/unpack kernels, and the
  existing gdn_step_*head dispatch -- all device-resident, no host crossings.
"""

import torch

import ttnn
import ttl

from .gdn_kernel import (
    NUM_V_HEADS,
    D_K,
    D_V_DIM,
    TILE,
    HEADS_PER_DEVICE_BY_MESH,
    _KERNEL_BY_MESH,
)


# Per-head K-segment count: D_K / TILE = 4 for Qwen3.5-35B-A3B (D_K = 128).
K_TILES_PER_HEAD = D_K // TILE
# Per-head V-segment count: D_V_DIM / TILE = 4.
V_TILES_PER_HEAD = D_V_DIM // TILE


# ============================================================
# Per-device head-index tensors (warm-up only; pre-allocated)
# ============================================================


def alloc_head_idx_d(mesh_device, head_dim):
    """Pre-allocate the per-device gather-index tensor for a head_dim slice.

    Shape: [1, 1, H_PER_DEV, head_dim] sharded along dim=2, UINT32, TILE.
    Device d's slice contains rows = [d*H_PER_DEV, ..., d*H_PER_DEV + H_PER_DEV - 1],
    each row broadcast across head_dim columns (so gather pulls the whole
    head's vector).

    Used as the `index` argument to ttnn.gather(replicated_input, dim=2, index=...).
    """
    n = mesh_device.get_num_devices()
    if n not in HEADS_PER_DEVICE_BY_MESH:
        raise ValueError(f"unsupported mesh size {n}")
    h_per_dev = HEADS_PER_DEVICE_BY_MESH[n]
    # Build the global aggregate: [1, 1, NUM_V_HEADS, head_dim] with
    # idx[0, 0, h, k] = h. ShardTensorToMesh(dim=2) slices into n shards,
    # so device d gets idx[0, 0, d*h_per_dev:(d+1)*h_per_dev, :], whose values
    # are the head ids for that device.
    idx = torch.arange(NUM_V_HEADS, dtype=torch.int32).view(1, 1, NUM_V_HEADS, 1)
    idx = idx.expand(1, 1, NUM_V_HEADS, head_dim).contiguous()
    return ttnn.from_torch(
        idx,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )


def alloc_head_idx_scalar(mesh_device):
    """Pre-allocate the per-device gather-index for scalar-per-head tensors (g, beta).

    Shape: [1, 1, H_PER_DEV] sharded dim=2, UINT32, TILE.
    """
    n = mesh_device.get_num_devices()
    if n not in HEADS_PER_DEVICE_BY_MESH:
        raise ValueError(f"unsupported mesh size {n}")
    idx = torch.arange(NUM_V_HEADS, dtype=torch.int32).view(1, 1, NUM_V_HEADS)
    return ttnn.from_torch(
        idx,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )


# ============================================================
# Shift matrix (constant; pre-allocated once per layer at warm-up)
# ============================================================
#
# The packing transformation is:
#   q_pack[h*D_K + k, c] = q_local[h, k]   if c == 0 else 0
#
# We express this per output tile (h*K_TILES + kt, 0) as a single matmul:
#   output_tile = transpose(input_tile_kt) @ shift_h
# where input_tile_kt is the kt-th col-tile of q_local viewed as [HEADS, D_K]
# (rows=heads, cols=k-indices in segment kt) and shift_h is a TILE x TILE tile
# with a single 1 at (h, 0) and zeros elsewhere.
#
# Verification:
#   transpose(input)[r, k] = input[k, r]
#   (transpose(input) @ shift_h)[r, c]
#     = sum_k input[k, r] * shift_h[k, c]
#     = input[h, r] * (1 if c == 0 else 0)
#   So output[r, 0] = input[h, r] = q_local[h, kt*32 + r], output[r, c>0] = 0.
#
# The shift tensor stacks H_PER_DEV shift tiles vertically:
#   shift[h*TILE + h, 0] = 1   for h in [0, H_PER_DEV)
# so shift_h = shift[h*TILE : (h+1)*TILE, 0:TILE] reads tile h via index `h, 0`
# (tile-coord) in the kernel's dm_read.


def alloc_shift_matrix(mesh_device, dtype=ttnn.bfloat16):
    """Build the per-head shift matrix used by pack_qkv_kernel.

    Shape: [H_PER_DEV * TILE, TILE] = e.g. [128, 32] for 4 heads/dev.
    Tile h has shift[h*TILE + h, 0] = 1, all else 0.
    Replicated across mesh devices.
    """
    n = mesh_device.get_num_devices()
    if n not in HEADS_PER_DEVICE_BY_MESH:
        raise ValueError(f"unsupported mesh size {n}")
    h_per_dev = HEADS_PER_DEVICE_BY_MESH[n]
    shift = torch.zeros(h_per_dev * TILE, TILE, dtype=torch.bfloat16)
    for h in range(h_per_dev):
        shift[h * TILE + h, 0] = 1.0
    return ttnn.from_torch(
        shift,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=(ttnn.ReplicateTensorToMesh(mesh_device) if n > 1 else None),
    )


# ============================================================
# Pack kernel for Q/K/V (one of the three at a time)
# ============================================================
#
# Single tt-lang kernel; called 3x per decode step (Q, K, V). The kernel is
# parameterised over HEADS_PER_DEV at decoration time, so we generate one
# binding for the 8-device case and one for the 4-device case via a factory.


def _make_pack_qkv_kernel(heads_per_dev):
    """Factory: build a tt-lang pack kernel for the given heads-per-device."""
    K_TILES = K_TILES_PER_HEAD  # = 4 for Qwen3.5-35B-A3B

    @ttl.operation(
        grid=(heads_per_dev, K_TILES),
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32 --ttl-maximize-dst",
    )
    def pack_qkv(in_t, shift_t, out_t):
        """Pack [HEADS_PER_DEV, D] head-sharded -> [HEADS_PER_DEV*D, TILE] col-0 layout.

        Per-device tensor layouts (in tile coords):
          in_t    : 1 row-tile  x K_TILES col-tiles   = [HEADS_PER_DEV(<=32), D]
          shift_t : HEADS_PER_DEV row-tiles x 1 col-tile  (replicated across mesh)
          out_t   : HEADS_PER_DEV*K_TILES row-tiles x 1 col-tile

        Per core (h, kt): output tile (h*K_TILES+kt, 0) col 0 = input row h
        of col-tile kt; other cols zero.
        """
        qi = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
        si = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
        qo = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with qi.wait() as q_t, si.wait() as s_t, qo.reserve() as out_b:
                # transpose(input_tile_kt) @ shift_h: see header derivation.
                out_b.store(ttl.transpose(q_t) @ s_t)

        @ttl.datamovement()
        def dm_read():
            h, kt = ttl.node(dims=2)
            with qi.reserve() as blk:
                ttl.copy(in_t[0, kt], blk).wait()
            with si.reserve() as blk:
                ttl.copy(shift_t[h, 0], blk).wait()

        @ttl.datamovement()
        def dm_write():
            h, kt = ttl.node(dims=2)
            with qo.wait() as blk:
                ttl.copy(blk, out_t[h * K_TILES + kt, 0]).wait()

    return pack_qkv


# Per-mesh kernel cache.
_PACK_KERNEL_BY_MESH = {
    8: _make_pack_qkv_kernel(HEADS_PER_DEVICE_BY_MESH[8]),  # 4 heads/dev (T3K)
    4: _make_pack_qkv_kernel(HEADS_PER_DEVICE_BY_MESH[4]),  # 8 heads/dev (QB2)
}


def pack_qkv_one(in_t, shift_t, out_t, mesh_device):
    """Run pack_qkv for a single tensor (Q, K, or V). Writes into pre-allocated `out_t`."""
    n = mesh_device.get_num_devices()
    kernel_fn = _PACK_KERNEL_BY_MESH[n]
    kernel_fn(in_t, shift_t, out_t)


# ============================================================
# Pack for alpha/beta (scalar-per-head -> tile-fill)
# ============================================================
#
# Target layout:    alpha_pack[h*TILE:(h+1)*TILE, :] = alpha[h]  (32x32 tile of scalar)
# Input layout:     alpha_local in TILE_LAYOUT, shape [1, TILE] logically [4 or 8]
#                   (one tile, with values at (0, 0..H_PER_DEV-1)).
#
# Per-output-tile compute is two matmuls with constants:
#   step 1:   row_bcast = K_const @ alpha_tile
#             where K_const[r, 0] = 1, K_const[r, c>0] = 0 (col 0 = ones).
#             K_const @ alpha_tile produces a tile where every row equals
#             alpha_tile's row 0:  row_bcast[r, c] = alpha[c] (for c < H_PER_DEV).
#   step 2:   output_tile = row_bcast @ L_h
#             where L_h[k, c] = 1 if k == h else 0 (row h = ones, others = 0).
#             output[r, c] = sum_k row_bcast[r, k] * L_h[k, c]
#                          = alpha[h] * (1 for any c)  (only k=h contributes)
#                          = alpha[h]   (for all r, c)
#
# Constants (pre-allocated once, replicated across mesh):
#   K_const:    [TILE, TILE]                col 0 = 1, others 0
#   L_per_head: [H_PER_DEV * TILE, TILE]    tile h has row h = 1
#
# Grid: (H_PER_DEV, 1) -- one core per output tile.


def alloc_K_const(mesh_device, dtype=ttnn.bfloat16):
    """[TILE, TILE] constant: col 0 all 1s, others 0. Replicated across mesh."""
    n = mesh_device.get_num_devices()
    K = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    K[:, 0] = 1.0
    return ttnn.from_torch(
        K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if n > 1 else None,
    )


def alloc_L_per_head(mesh_device, dtype=ttnn.bfloat16):
    """[H_PER_DEV * TILE, TILE] per-head row-selector. Tile h has row h = 1.
    Replicated across mesh.
    """
    n = mesh_device.get_num_devices()
    if n not in HEADS_PER_DEVICE_BY_MESH:
        raise ValueError(f"unsupported mesh size {n}")
    h_per_dev = HEADS_PER_DEVICE_BY_MESH[n]
    L = torch.zeros(h_per_dev * TILE, TILE, dtype=torch.bfloat16)
    for h in range(h_per_dev):
        L[h * TILE + h, :] = 1.0  # tile h, row h, all cols
    return ttnn.from_torch(
        L,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if n > 1 else None,
    )


def _make_pack_alpha_kernel(heads_per_dev):
    """Factory: build a tt-lang pack-alpha/beta kernel for the given heads-per-device."""

    @ttl.operation(
        grid=(heads_per_dev, 1),
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32 --ttl-maximize-dst",
    )
    def pack_alpha(in_t, K_const, L_per_head, out_t):
        """Pack [1, TILE_padded] scalar-per-head -> [HEADS_PER_DEV*TILE, TILE].

        Per core (h, _): output tile (h, 0) is filled with in_t[0, h].
        """
        ai = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
        ki = ttl.make_dataflow_buffer_like(K_const, shape=(1, 1), block_count=2)
        li = ttl.make_dataflow_buffer_like(L_per_head, shape=(1, 1), block_count=2)
        inter = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
        out_b = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            # Step 1: K @ alpha -> tile with every row = alpha's row 0
            with ki.wait() as k_t, ai.wait() as a_t, inter.reserve() as ib:
                ib.store(k_t @ a_t)
            # Step 2: (K @ alpha) @ L_h -> tile filled with alpha[h]
            with inter.wait() as i_t, li.wait() as l_t, out_b.reserve() as ob:
                ob.store(i_t @ l_t)

        @ttl.datamovement()
        def dm_read():
            h, _ = ttl.node(dims=2)
            with ai.reserve() as blk:
                ttl.copy(in_t[0, 0], blk).wait()  # alpha_local: 1 tile total
            with ki.reserve() as blk:
                ttl.copy(K_const[0, 0], blk).wait()
            with li.reserve() as blk:
                ttl.copy(L_per_head[h, 0], blk).wait()

        @ttl.datamovement()
        def dm_write():
            h, _ = ttl.node(dims=2)
            with out_b.wait() as blk:
                ttl.copy(blk, out_t[h, 0]).wait()

    return pack_alpha


_PACK_ALPHA_KERNEL_BY_MESH = {
    8: _make_pack_alpha_kernel(HEADS_PER_DEVICE_BY_MESH[8]),
    4: _make_pack_alpha_kernel(HEADS_PER_DEVICE_BY_MESH[4]),
}


def pack_alpha_beta(alpha_local, beta_local, alpha_pack, beta_pack, mesh_device, K_const, L_per_head):
    """Pack alpha and beta scalar-per-head into [HEADS_PER_DEV*TILE, TILE] tile-fill layout."""
    n = mesh_device.get_num_devices()
    kernel_fn = _PACK_ALPHA_KERNEL_BY_MESH[n]
    kernel_fn(alpha_local, K_const, L_per_head, alpha_pack)
    kernel_fn(beta_local, K_const, L_per_head, beta_pack)


# ============================================================
# Unpack [HEADS*D_V, TILE] -> [HEADS, D_V]
# ============================================================
#
# Mirror of pack_qkv with grid=(1, V_TILES_PER_HEAD) per device. Per output
# tile (0, ct) of core_out [HEADS_PER_DEV, D_V_DIM]: aggregate from
# HEADS_PER_DEV input tiles (one per head h) of out_buf at row h*V_TILES+ct.
#
# Math (reuses the SAME shift_matrix from the pack direction):
#   out_tile = sum_{h=0..HEADS_PER_DEV-1} shift_h @ transpose(in_tile_h_ct)
#
# Verification: shift_h has shift_h[h, 0] = 1, else 0.
#   transpose(in_tile)[k, c] = in_tile[c, k]; in_tile is col-0-only so
#     transpose(in_tile)[0, c] = in_tile[c, 0]; transpose(in_tile)[k>0, c] = 0.
#   (shift_h @ transpose(in_tile))[r, c]
#     = sum_k shift_h[r, k] * transpose(in_tile)[k, c]
#     = (1 if r == h, k == 0 else 0) * transpose(in_tile)[0, c]
#     = in_tile[c, 0] if r == h else 0.
# Summed over h: out_tile[r, c] = in_tile_r[c, 0] for r in [0, HEADS_PER_DEV), else 0.
#
# Accumulation uses ping-pong between two intermediate DFBs (acc1, acc2) so
# we never read+write the same DFB in one compute step.

# Note: tt-lang does not allow Python list captures inside @ttl.compute()
# closures, so we define separate kernel bodies for HEADS_PER_DEV=4 and =8
# rather than parameterising via a list.


@ttl.operation(
    grid=(1, V_TILES_PER_HEAD),
    fp32_dest_acc_en=True,
    options="--ttl-matmul-full-fp32 --ttl-maximize-dst",
)
def _unpack_kernel_h4(in_t, shift_t, out_t):
    """Unpack with HEADS_PER_DEV=4 (T3K layout).

    Per core (0, ct): output tile (0, ct) = sum over h=0..3 of
    shift_h @ transpose(in_tile[h*V_TILES+ct, 0]).
    """
    V_TILES = V_TILES_PER_HEAD
    b0_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b0_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b1_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b1_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b2_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b2_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b3_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b3_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    acc1 = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
    acc2 = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
    out_b = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # h=0: initialize acc1
        with b0_s.wait() as s, b0_in.wait() as i, acc1.reserve() as ab:
            ab.store(s @ ttl.transpose(i))
        # h=1: acc1 -> acc2
        with acc1.wait() as a, b1_s.wait() as s, b1_in.wait() as i, acc2.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        # h=2: acc2 -> acc1
        with acc2.wait() as a, b2_s.wait() as s, b2_in.wait() as i, acc1.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        # h=3: acc1 -> acc2
        with acc1.wait() as a, b3_s.wait() as s, b3_in.wait() as i, acc2.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        # Final: acc2 -> out
        with acc2.wait() as a, out_b.reserve() as ob:
            ob.store(a)

    @ttl.datamovement()
    def dm_read():
        _, ct = ttl.node(dims=2)
        with b0_in.reserve() as blk:
            ttl.copy(in_t[0 * V_TILES + ct, 0], blk).wait()
        with b0_s.reserve() as blk:
            ttl.copy(shift_t[0, 0], blk).wait()
        with b1_in.reserve() as blk:
            ttl.copy(in_t[1 * V_TILES + ct, 0], blk).wait()
        with b1_s.reserve() as blk:
            ttl.copy(shift_t[1, 0], blk).wait()
        with b2_in.reserve() as blk:
            ttl.copy(in_t[2 * V_TILES + ct, 0], blk).wait()
        with b2_s.reserve() as blk:
            ttl.copy(shift_t[2, 0], blk).wait()
        with b3_in.reserve() as blk:
            ttl.copy(in_t[3 * V_TILES + ct, 0], blk).wait()
        with b3_s.reserve() as blk:
            ttl.copy(shift_t[3, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        _, ct = ttl.node(dims=2)
        with out_b.wait() as blk:
            ttl.copy(blk, out_t[0, ct]).wait()


@ttl.operation(
    grid=(1, V_TILES_PER_HEAD),
    fp32_dest_acc_en=True,
    options="--ttl-matmul-full-fp32 --ttl-maximize-dst",
)
def _unpack_kernel_h8(in_t, shift_t, out_t):
    """Unpack with HEADS_PER_DEV=8 (QB2 layout). Same logic as h4, more heads."""
    V_TILES = V_TILES_PER_HEAD
    b0_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b0_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b1_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b1_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b2_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b2_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b3_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b3_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b4_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b4_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b5_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b5_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b6_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b6_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    b7_in = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    b7_s = ttl.make_dataflow_buffer_like(shift_t, shape=(1, 1), block_count=2)
    acc1 = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
    acc2 = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
    out_b = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # Ping-pong acc1 <-> acc2 across 8 heads (final result lands in acc2 if H is even)
        with b0_s.wait() as s, b0_in.wait() as i, acc1.reserve() as ab:
            ab.store(s @ ttl.transpose(i))
        with acc1.wait() as a, b1_s.wait() as s, b1_in.wait() as i, acc2.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc2.wait() as a, b2_s.wait() as s, b2_in.wait() as i, acc1.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc1.wait() as a, b3_s.wait() as s, b3_in.wait() as i, acc2.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc2.wait() as a, b4_s.wait() as s, b4_in.wait() as i, acc1.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc1.wait() as a, b5_s.wait() as s, b5_in.wait() as i, acc2.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc2.wait() as a, b6_s.wait() as s, b6_in.wait() as i, acc1.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc1.wait() as a, b7_s.wait() as s, b7_in.wait() as i, acc2.reserve() as ab:
            ab.store(a + s @ ttl.transpose(i))
        with acc2.wait() as a, out_b.reserve() as ob:
            ob.store(a)

    @ttl.datamovement()
    def dm_read():
        _, ct = ttl.node(dims=2)
        with b0_in.reserve() as blk:
            ttl.copy(in_t[0 * V_TILES + ct, 0], blk).wait()
        with b0_s.reserve() as blk:
            ttl.copy(shift_t[0, 0], blk).wait()
        with b1_in.reserve() as blk:
            ttl.copy(in_t[1 * V_TILES + ct, 0], blk).wait()
        with b1_s.reserve() as blk:
            ttl.copy(shift_t[1, 0], blk).wait()
        with b2_in.reserve() as blk:
            ttl.copy(in_t[2 * V_TILES + ct, 0], blk).wait()
        with b2_s.reserve() as blk:
            ttl.copy(shift_t[2, 0], blk).wait()
        with b3_in.reserve() as blk:
            ttl.copy(in_t[3 * V_TILES + ct, 0], blk).wait()
        with b3_s.reserve() as blk:
            ttl.copy(shift_t[3, 0], blk).wait()
        with b4_in.reserve() as blk:
            ttl.copy(in_t[4 * V_TILES + ct, 0], blk).wait()
        with b4_s.reserve() as blk:
            ttl.copy(shift_t[4, 0], blk).wait()
        with b5_in.reserve() as blk:
            ttl.copy(in_t[5 * V_TILES + ct, 0], blk).wait()
        with b5_s.reserve() as blk:
            ttl.copy(shift_t[5, 0], blk).wait()
        with b6_in.reserve() as blk:
            ttl.copy(in_t[6 * V_TILES + ct, 0], blk).wait()
        with b6_s.reserve() as blk:
            ttl.copy(shift_t[6, 0], blk).wait()
        with b7_in.reserve() as blk:
            ttl.copy(in_t[7 * V_TILES + ct, 0], blk).wait()
        with b7_s.reserve() as blk:
            ttl.copy(shift_t[7, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        _, ct = ttl.node(dims=2)
        with out_b.wait() as blk:
            ttl.copy(blk, out_t[0, ct]).wait()


_UNPACK_KERNEL_BY_MESH = {
    8: _unpack_kernel_h4,  # mesh-size 8 -> 4 heads/dev
    4: _unpack_kernel_h8,  # mesh-size 4 -> 8 heads/dev
}


def unpack_gdn_to_natural_layout(out_buf, core_out, mesh_device, shift_matrix):
    """Unpack GDN kernel output to head-sharded [HEADS_PER_DEV, D_V] layout."""
    n = mesh_device.get_num_devices()
    kernel_fn = _UNPACK_KERNEL_BY_MESH[n]
    kernel_fn(out_buf, shift_matrix, core_out)


# ============================================================
# Trace-compatible decode step (orchestration; drops in for gdn_recurrent_step)
# ============================================================


def gdn_recurrent_step_traced(
    mesh_device,
    q_replicated,  # [1, 1, NUM_V_HEADS, D_K]  bf16, replicated, l2-normalised
    k_replicated,  # [1, 1, NUM_V_HEADS, D_K]
    v_replicated,  # [1, 1, NUM_V_HEADS, D_V_DIM]
    g_replicated,  # [1, 1, NUM_V_HEADS]
    beta_replicated,  # [1, 1, NUM_V_HEADS]
    state_in,  # [H_PER_DEV*D_K, D_V_DIM] sharded dim=0, ttnn.Tensor
    state_out,  # [H_PER_DEV*D_K, D_V_DIM] sharded dim=0, ttnn.Tensor (pre-alloc)
    out_buf,  # [H_PER_DEV*D_V_MESH, TILE] sharded dim=0 (pre-alloc)
    core_out,  # [1, 1, H_PER_DEV, D_V_DIM] sharded dim=2 (pre-alloc)
    head_idx_d_qk,  # [1, 1, H_PER_DEV, D_K] sharded dim=2, uint32 (pre-alloc, alloc_head_idx_d)
    head_idx_d_v,  # [1, 1, H_PER_DEV, D_V_DIM] sharded dim=2, uint32 (pre-alloc)
    head_idx_scalar,  # [1, 1, H_PER_DEV] sharded dim=2, uint32 (pre-alloc)
    shift_matrix,  # [H_PER_DEV*TILE, TILE] replicated (pre-alloc, alloc_shift_matrix)
    K_const,  # [TILE, TILE] replicated (pre-alloc, alloc_K_const) -- alpha/beta pack
    L_per_head,  # [H_PER_DEV*TILE, TILE] replicated (pre-alloc, alloc_L_per_head)
    q_pack,
    k_pack,
    v_pack,
    alpha_pack,
    beta_pack,  # pre-alloc kernel inputs
):
    """Run one GDN decode step, fully on device.

    Drop-in replacement for gdn_recurrent_step that takes TTNN tensors and
    returns a TTNN tensor. All ops are TTNN or tt-lang dispatches; no host
    crossings, no allocations. Suitable for capture inside Metal Trace.

    Pipeline:
      1. Per-device gather: replicated -> head-sharded (Q/K/V/g/beta)
      2. alpha = ttnn.exp(g)
      3. pack_natural_to_gdn_layout: head-sharded -> kernel-input buffers
      4. gdn_step_4head/_8head: dispatch GDN kernel
      5. unpack_gdn_to_natural_layout: kernel output -> head-sharded core_out

    Returns:
      core_out (head-sharded ttnn.Tensor [1, 1, H_PER_DEV, D_V_DIM]).
      The state has been written in-place to state_out.
    """
    # Step 1: replicated -> head-sharded via ttnn.gather.
    q_local = ttnn.gather(q_replicated, dim=2, index=head_idx_d_qk)
    k_local = ttnn.gather(k_replicated, dim=2, index=head_idx_d_qk)
    v_local = ttnn.gather(v_replicated, dim=2, index=head_idx_d_v)
    g_local = ttnn.gather(g_replicated, dim=2, index=head_idx_scalar)
    beta_local = ttnn.gather(beta_replicated, dim=2, index=head_idx_scalar)

    # Step 2: alpha = exp(g) on device.
    alpha_local = ttnn.exp(g_local)

    # Step 3: pack into kernel input layout.
    # Q/K/V via the tt-lang pack kernel above (3 dispatches, same kernel binary).
    pack_qkv_one(q_local, shift_matrix, q_pack, mesh_device)
    pack_qkv_one(k_local, shift_matrix, k_pack, mesh_device)
    pack_qkv_one(v_local, shift_matrix, v_pack, mesh_device)
    pack_alpha_beta(
        alpha_local,
        beta_local,
        alpha_pack,
        beta_pack,
        mesh_device,
        K_const,
        L_per_head,
    )

    # Step 4: existing GDN kernel dispatch.
    n = mesh_device.get_num_devices()
    kernel_fn, _ = _KERNEL_BY_MESH[n]
    kernel_fn(state_in, q_pack, k_pack, v_pack, alpha_pack, beta_pack, state_out, out_buf)

    # Step 5: unpack kernel output to natural [1,1,H,D_V] layout.
    unpack_gdn_to_natural_layout(out_buf, core_out, mesh_device, shift_matrix)

    return core_out
