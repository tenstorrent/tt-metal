# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GDN (Gated Delta Network) decode kernel for Qwen3.5 linear attention.

Wraps the tt-lang `gdn_step_4head` kernel for use within tt-metal. Decode-only:
replaces a single timestep of `recurrent_gated_delta_rule` on an 8-device mesh.
Each device runs 4 of the 32 model heads (head-parallel sharding).

Imports `ttl` (tt-lang). Only import this module when TTNN_GDN_KERNEL=1 so the
tt-lang dependency is opt-in.

The kernel computes one step of:
    S_t = alpha * S + beta * k * (v - (alpha*S)^T @ k)
    o_t = S_t^T @ q
"""

import torch

import ttnn
import ttl


# ---- Dimensions (Qwen3.5-35B-A3B) ----
TILE: int = 32
D_K: int = 128
D_V_DIM: int = 128
NUM_V_HEADS: int = 32
D_V_MESH: int = D_V_DIM
K_TILES: int = D_K // TILE  # 4
V_TILES_MESH: int = D_V_MESH // TILE  # 4
BF16 = torch.bfloat16

# Supported mesh sizes -> heads per device. The kernel grid is
# (NUM_HEADS_PER_DEVICE, V_TILES_MESH); we have one variant per
# heads-per-device value. Total heads must equal NUM_V_HEADS.
HEADS_PER_DEVICE_BY_MESH = {
    8: 4,  # T3K / 8x BH: 4 heads * 8 devices = 32
    4: 8,  # QB2 / 4x BH: 8 heads * 4 devices = 32
}

_GDN_COMPILER_OPTS = "--ttl-matmul-full-fp32 " "--ttl-reduce-full-fp32 " "--ttl-maximize-dst"


# ============================================================
# Kernel: 4 heads per device, full D_V=128 (8-device mesh slice)
# Verbatim from tt-lang/example.py:gdn_step_4head except the comment block.
# ============================================================
@ttl.operation(
    grid=(4, 4),
    fp32_dest_acc_en=True,
    options=_GDN_COMPILER_OPTS,
)
def gdn_step_4head(state_in, q_in, k_in, v_in, alpha_in, beta_in, state_out, out):
    """GDN decode step for 4 heads on one device (1/8 of 8-device mesh).

    Tensor layouts (per-device, in elements):
      state_in/out : [NUM_HEADS_PER_DEVICE*D_K, D_V_MESH]  = [512, 128]
      q_in, k_in   : [NUM_HEADS_PER_DEVICE*D_K, TILE]      = [512, 32]
      v_in, out    : [NUM_HEADS_PER_DEVICE*D_V_MESH, TILE] = [512, 32]
      alpha_in     : [NUM_HEADS_PER_DEVICE*TILE, TILE]     = [128, 32]
      beta_in      : [NUM_HEADS_PER_DEVICE*TILE, TILE]     = [128, 32]
    """
    si = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    ki_e = ttl.make_dataflow_buffer_like(k_in, shape=(K_TILES, 1), block_count=2)
    ki_o = ttl.make_dataflow_buffer_like(k_in, shape=(K_TILES, 1), block_count=2)
    qi = ttl.make_dataflow_buffer_like(q_in, shape=(K_TILES, 1), block_count=2)
    vi = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    ai = ttl.make_dataflow_buffer_like(alpha_in, shape=(1, 1), block_count=2)
    bi = ttl.make_dataflow_buffer_like(beta_in, shape=(1, 1), block_count=2)

    so = ttl.make_dataflow_buffer_like(state_out, shape=(K_TILES, 1), block_count=3)
    oo = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    alpha_bcast = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    beta_bcast = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    ss_e = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    ss_sn = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    e_buf = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    dt_buf = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    outer_buf = ttl.make_dataflow_buffer_like(k_in, shape=(K_TILES, 1), block_count=2)
    sn_local = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)

    @ttl.compute()
    def compute():
        with ai.wait() as a_scalar:
            with alpha_bcast.reserve() as ab:
                ab.store(ttl.math.broadcast(a_scalar, ab, dims=[0]))
        with bi.wait() as b_scalar:
            with beta_bcast.reserve() as bb:
                bb.store(ttl.math.broadcast(b_scalar, bb, dims=[0]))

        with si.wait() as s, alpha_bcast.wait() as a_blk:
            s_scaled = a_blk * s
            with ss_e.reserve() as se:
                se.store(s_scaled)
            with ss_sn.reserve() as ssn:
                ssn.store(s_scaled)

        with ss_e.wait() as ss, ki_e.wait() as ke:
            e = ttl.transpose(ss) @ ke
            with e_buf.reserve() as eb:
                eb.store(e)

        with vi.wait() as v, e_buf.wait() as e_val:
            delta = v - e_val
            dt = ttl.transpose(delta)
            with dt_buf.reserve() as db:
                db.store(dt)

        with ki_o.wait() as ko, dt_buf.wait() as dt_val:
            outer = ko @ dt_val
            with outer_buf.reserve() as ob:
                ob.store(outer)

        with ss_sn.wait() as ss_val, beta_bcast.wait() as b_blk, outer_buf.wait() as o_val:
            s_new = ss_val + b_blk * o_val
            with so.reserve() as sb:
                sb.store(s_new)

        with so.wait() as sn_src:
            with sn_local.reserve() as snl:
                snl.store(sn_src)

        with sn_local.wait() as sn, qi.wait() as q:
            output = ttl.transpose(sn) @ q
            with oo.reserve() as oo_blk:
                oo_blk.store(output)

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = nx
        vt = ny

        k_slice = k_in[h * K_TILES : h * K_TILES + K_TILES, 0:1]

        with si.reserve() as blk:
            ttl.copy(
                state_in[h * K_TILES : h * K_TILES + K_TILES, vt : vt + 1],
                blk,
            ).wait()
        with ki_e.reserve() as blk:
            ttl.copy(k_slice, blk).wait()
        with ki_o.reserve() as blk:
            ttl.copy(k_slice, blk).wait()
        with qi.reserve() as blk:
            ttl.copy(q_in[h * K_TILES : h * K_TILES + K_TILES, 0:1], blk).wait()
        with vi.reserve() as blk:
            ttl.copy(v_in[h * V_TILES_MESH + vt : h * V_TILES_MESH + vt + 1, 0:1], blk).wait()
        with ai.reserve() as blk:
            ttl.copy(alpha_in[h, 0], blk).wait()
        with bi.reserve() as blk:
            ttl.copy(beta_in[h, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = nx
        vt = ny

        with so.wait() as blk:
            ttl.copy(blk, state_out[h * K_TILES : h * K_TILES + K_TILES, vt : vt + 1]).wait()
        with oo.wait() as blk:
            ttl.copy(blk, out[h * V_TILES_MESH + vt : h * V_TILES_MESH + vt + 1, 0:1]).wait()


# ============================================================
# Kernel: 8 heads per device, full D_V=128 (4-device QB2 mesh slice)
# Same compute phases as gdn_step_4head; grid=(8,4)=32 cores.
# ============================================================
@ttl.operation(
    grid=(8, 4),
    fp32_dest_acc_en=True,
    options=_GDN_COMPILER_OPTS,
)
def gdn_step_8head(state_in, q_in, k_in, v_in, alpha_in, beta_in, state_out, out):
    """GDN decode step for 8 heads on one device (1/4 of 4-device QB2 mesh).

    Tensor layouts (per-device, in elements):
      state_in/out : [8*D_K, D_V_MESH]      = [1024, 128]
      q_in, k_in   : [8*D_K, TILE]           = [1024, 32]
      v_in, out    : [8*D_V_MESH, TILE]      = [1024, 32]
      alpha_in     : [8*TILE, TILE]          = [256, 32]
      beta_in      : [8*TILE, TILE]          = [256, 32]
    """
    si = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    ki_e = ttl.make_dataflow_buffer_like(k_in, shape=(K_TILES, 1), block_count=2)
    ki_o = ttl.make_dataflow_buffer_like(k_in, shape=(K_TILES, 1), block_count=2)
    qi = ttl.make_dataflow_buffer_like(q_in, shape=(K_TILES, 1), block_count=2)
    vi = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    ai = ttl.make_dataflow_buffer_like(alpha_in, shape=(1, 1), block_count=2)
    bi = ttl.make_dataflow_buffer_like(beta_in, shape=(1, 1), block_count=2)

    so = ttl.make_dataflow_buffer_like(state_out, shape=(K_TILES, 1), block_count=3)
    oo = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    alpha_bcast = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    beta_bcast = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    ss_e = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    ss_sn = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)
    e_buf = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    dt_buf = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    outer_buf = ttl.make_dataflow_buffer_like(k_in, shape=(K_TILES, 1), block_count=2)
    sn_local = ttl.make_dataflow_buffer_like(state_in, shape=(K_TILES, 1), block_count=2)

    @ttl.compute()
    def compute():
        with ai.wait() as a_scalar:
            with alpha_bcast.reserve() as ab:
                ab.store(ttl.math.broadcast(a_scalar, ab, dims=[0]))
        with bi.wait() as b_scalar:
            with beta_bcast.reserve() as bb:
                bb.store(ttl.math.broadcast(b_scalar, bb, dims=[0]))

        with si.wait() as s, alpha_bcast.wait() as a_blk:
            s_scaled = a_blk * s
            with ss_e.reserve() as se:
                se.store(s_scaled)
            with ss_sn.reserve() as ssn:
                ssn.store(s_scaled)

        with ss_e.wait() as ss, ki_e.wait() as ke:
            e = ttl.transpose(ss) @ ke
            with e_buf.reserve() as eb:
                eb.store(e)

        with vi.wait() as v, e_buf.wait() as e_val:
            delta = v - e_val
            dt = ttl.transpose(delta)
            with dt_buf.reserve() as db:
                db.store(dt)

        with ki_o.wait() as ko, dt_buf.wait() as dt_val:
            outer = ko @ dt_val
            with outer_buf.reserve() as ob:
                ob.store(outer)

        with ss_sn.wait() as ss_val, beta_bcast.wait() as b_blk, outer_buf.wait() as o_val:
            s_new = ss_val + b_blk * o_val
            with so.reserve() as sb:
                sb.store(s_new)

        with so.wait() as sn_src:
            with sn_local.reserve() as snl:
                snl.store(sn_src)

        with sn_local.wait() as sn, qi.wait() as q:
            output = ttl.transpose(sn) @ q
            with oo.reserve() as oo_blk:
                oo_blk.store(output)

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = nx  # head index 0..7
        vt = ny  # V-tile column 0..3

        k_slice = k_in[h * K_TILES : h * K_TILES + K_TILES, 0:1]

        with si.reserve() as blk:
            ttl.copy(
                state_in[h * K_TILES : h * K_TILES + K_TILES, vt : vt + 1],
                blk,
            ).wait()
        with ki_e.reserve() as blk:
            ttl.copy(k_slice, blk).wait()
        with ki_o.reserve() as blk:
            ttl.copy(k_slice, blk).wait()
        with qi.reserve() as blk:
            ttl.copy(q_in[h * K_TILES : h * K_TILES + K_TILES, 0:1], blk).wait()
        with vi.reserve() as blk:
            ttl.copy(v_in[h * V_TILES_MESH + vt : h * V_TILES_MESH + vt + 1, 0:1], blk).wait()
        with ai.reserve() as blk:
            ttl.copy(alpha_in[h, 0], blk).wait()
        with bi.reserve() as blk:
            ttl.copy(beta_in[h, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = nx
        vt = ny

        with so.wait() as blk:
            ttl.copy(blk, state_out[h * K_TILES : h * K_TILES + K_TILES, vt : vt + 1]).wait()
        with oo.wait() as blk:
            ttl.copy(blk, out[h * V_TILES_MESH + vt : h * V_TILES_MESH + vt + 1, 0:1]).wait()


# Map mesh size -> (kernel function, heads per device).
_KERNEL_BY_MESH = {
    8: (gdn_step_4head, 4),
    4: (gdn_step_8head, 8),
}


# ============================================================
# Host-side packing / mesh upload helpers
# ============================================================


def _pack_step(q, k, v, alpha, beta, n_heads):
    """Pack one decode step's per-head data into kernel-stacked tensors.

    Inputs (CPU, bf16-castable):
      q     : [n_heads, D_K]      L2-normalised
      k     : [n_heads, D_K]      L2-normalised
      v     : [n_heads, D_V_DIM]
      alpha : [n_heads]            exp(g)
      beta  : [n_heads]
    """
    H = n_heads
    q_all = torch.zeros(H * D_K, TILE, dtype=BF16)
    k_all = torch.zeros(H * D_K, TILE, dtype=BF16)
    v_all = torch.zeros(H * D_V_MESH, TILE, dtype=BF16)
    a_all = torch.zeros(H * TILE, TILE, dtype=BF16)
    b_all = torch.zeros(H * TILE, TILE, dtype=BF16)

    q_bf, k_bf, v_bf = q.to(BF16), k.to(BF16), v.to(BF16)
    a_bf, b_bf = alpha.to(BF16), beta.to(BF16)

    for h in range(H):
        q_all[h * D_K : (h + 1) * D_K, 0] = q_bf[h]
        k_all[h * D_K : (h + 1) * D_K, 0] = k_bf[h]
        v_all[h * D_V_MESH : (h + 1) * D_V_MESH, 0] = v_bf[h]
        a_all[h * TILE : (h + 1) * TILE, :] = a_bf[h].item()
        b_all[h * TILE : (h + 1) * TILE, :] = b_bf[h].item()

    return q_all, k_all, v_all, a_all, b_all


def _to_mesh(tensor, mesh_device):
    """Upload a CPU bf16 tensor to mesh, sharding along dim=0 (head-parallel)."""
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _from_mesh(tensor, mesh_device):
    """Download a sharded mesh tensor back to CPU, concatenating along dim=0."""
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


# ============================================================
# Decode-step entry point
# ============================================================


def alloc_state_buffer(mesh_device):
    """Allocate one DRAM-resident state buffer on the mesh, zero-initialised.

    Caller can keep this around to avoid per-step state allocation+upload.
    Layout: [NUM_V_HEADS*D_K, D_V_DIM] BF16 sharded along dim 0 across the mesh.
    """
    return _to_mesh(torch.zeros(NUM_V_HEADS * D_K, D_V_DIM, dtype=BF16), mesh_device)


def alloc_out_buffer(mesh_device):
    """Allocate one DRAM-resident output buffer on the mesh, zero-initialised.

    Layout: [NUM_V_HEADS*D_V_MESH, TILE] BF16 sharded along dim 0 across the mesh.
    """
    return _to_mesh(torch.zeros(NUM_V_HEADS * D_V_MESH, TILE, dtype=BF16), mesh_device)


def _is_ttnn_tensor(x):
    return isinstance(x, ttnn.Tensor)


def gdn_recurrent_step(
    mesh_device,
    query,
    key,
    value,
    g,
    beta,
    recurrent_state,
    state_out_buf=None,
    out_buf=None,
    return_state_as_ttnn=False,
):
    """Run one GDN decode step on an 8-device or 4-device mesh.

    Drop-in replacement for `recurrent_gated_delta_rule(..., seq_len=1)`.
    Caller is responsible for L2-normalising query/key (the kernel does NOT
    apply the qk-l2norm step internally).

    Mesh size selects the kernel:
      - 8 devices -> gdn_step_4head (4 heads/device, T3K layout)
      - 4 devices -> gdn_step_8head (8 heads/device, QB2 layout)

    Args:
      mesh_device : ttnn mesh device with 8 or 4 devices
      query       : [batch=1, seq=1, NUM_V_HEADS, D_K]      bf16, L2-normalised
      key         : [batch=1, seq=1, NUM_V_HEADS, D_K]      bf16, L2-normalised
      value       : [batch=1, seq=1, NUM_V_HEADS, D_V_DIM]  bf16
      g           : [batch=1, seq=1, NUM_V_HEADS]            log-decay (any float dtype)
      beta        : [batch=1, seq=1, NUM_V_HEADS]            bf16
      recurrent_state : one of:
                          - None                                          (zeros)
                          - torch [batch=1, NUM_V_HEADS, D_K, D_V_DIM]    (will be uploaded)
                          - ttnn.Tensor                                    (used directly, no upload)
      state_out_buf : optional pre-allocated ttnn.Tensor to receive new state;
                      if None, allocated per call (slow path).
      out_buf       : optional pre-allocated ttnn.Tensor to receive output;
                      if None, allocated per call.
      return_state_as_ttnn : if True, the returned `last_recurrent_state` is the
                             ttnn.Tensor that received the new state (no download).
                             If False (default), state is downloaded to a torch tensor
                             matching the original PyTorch layout.

    Returns:
      core_attn_out        : torch [batch=1, seq=1, NUM_V_HEADS, D_V_DIM] bf16
      last_recurrent_state : ttnn.Tensor when return_state_as_ttnn=True, else
                             torch [batch=1, NUM_V_HEADS, D_K, D_V_DIM] bf16
    """
    if query.shape[0] != 1 or query.shape[1] != 1:
        raise ValueError(f"gdn_recurrent_step requires batch=1 seq=1, got {query.shape[:2]}")
    if query.shape[2] != NUM_V_HEADS:
        raise ValueError(f"gdn_recurrent_step requires {NUM_V_HEADS} heads, got {query.shape[2]}")
    n_devices = mesh_device.get_num_devices()
    if n_devices not in _KERNEL_BY_MESH:
        raise ValueError(f"gdn_recurrent_step requires a {sorted(_KERNEL_BY_MESH)}-device mesh, " f"got {n_devices}")
    kernel_fn, _ = _KERNEL_BY_MESH[n_devices]

    # Strip leading (batch, seq) dims; alpha = exp(g) on CPU before packing.
    q_h = query[0, 0]  # [H, D_K]
    k_h = key[0, 0]
    v_h = value[0, 0]  # [H, D_V_DIM]
    alpha_h = g[0, 0].float().exp()  # [H]
    beta_h = beta[0, 0]

    q_p, k_p, v_p, a_p, b_p = _pack_step(q_h, k_h, v_h, alpha_h, beta_h, NUM_V_HEADS)

    # Resolve state input: TTNN -> use as-is; torch/None -> upload.
    if _is_ttnn_tensor(recurrent_state):
        state_in_tt = recurrent_state
    else:
        if recurrent_state is None:
            state_packed = torch.zeros(NUM_V_HEADS * D_K, D_V_DIM, dtype=BF16)
        else:
            state_packed = recurrent_state[0].to(BF16).reshape(NUM_V_HEADS * D_K, D_V_DIM).contiguous()
        state_in_tt = _to_mesh(state_packed, mesh_device)

    q_tt = _to_mesh(q_p, mesh_device)
    k_tt = _to_mesh(k_p, mesh_device)
    v_tt = _to_mesh(v_p, mesh_device)
    a_tt = _to_mesh(a_p, mesh_device)
    b_tt = _to_mesh(b_p, mesh_device)

    if state_out_buf is None:
        state_out_buf = alloc_state_buffer(mesh_device)
    if out_buf is None:
        out_buf = alloc_out_buffer(mesh_device)

    kernel_fn(state_in_tt, q_tt, k_tt, v_tt, a_tt, b_tt, state_out_buf, out_buf)

    # Output always returns to host (the gated RMS-norm is on PyTorch).
    out_packed = _from_mesh(out_buf, mesh_device)  # [H*D_V_MESH, TILE]
    core_attn_out = out_packed.reshape(NUM_V_HEADS, D_V_MESH, TILE)[:, :, 0].contiguous()
    core_attn_out = core_attn_out.reshape(1, 1, NUM_V_HEADS, D_V_DIM)

    if return_state_as_ttnn:
        return core_attn_out, state_out_buf

    state_packed_out = _from_mesh(state_out_buf, mesh_device)  # [H*D_K, D_V_DIM]
    new_state = state_packed_out.reshape(1, NUM_V_HEADS, D_K, D_V_DIM)
    return core_attn_out, new_state
