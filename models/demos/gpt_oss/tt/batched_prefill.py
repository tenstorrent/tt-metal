# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Helpers of the batched prefill: B users of S tokens each go through one forward as [1, 1, B*S, H] rows stacked by
user (user b holds rows b*S .. b*S+S-1), attention per user ([B, heads, S, head_dim]).

Every helper keeps each user's rows bit-identical to the batch-1 prefill of that user alone:
- The cross-device reductions (o_proj and MoE partial sums) run on a [B, 1, S, H] view. The ring reduce-scatter sums
  each (batch, channel) unit in an order fixed by the tile's position inside the unit, so a user's rows reduce as in a
  batch-1 [1, 1, S, H] call. A [1, 1, B*S, H] call would scatter the rows over the devices differently and sum them in
  another order.
- A batch-1 input of at most DIRECT_REDUCE_SCATTER_MAX_BYTES per device takes the one-shot direct reduce-scatter
  (ttnn.reduce_scatter / ttnn.all_reduce), which sums in another order than the ring. The batched input is B times
  larger and would take the ring, so per_user_reduce calls the direct op itself where the user's own input would.
- Sequence-parallel rows: device d holds rows d*S/N .. of every user ([B, 1, S/N, H] stacked by user), as a batch-1
  sequence-parallel prefill of each user would; gathers and partitions run on the [B, 1, *, W] view.
"""

import torch

import ttnn
from models.tt_transformers.tt.ccl import get_num_links

# Per-device input size up to which ttnn.reduce_scatter (and the reduce-scatter half of ttnn.all_reduce) takes the
# one-shot direct reduce-scatter: k_direct_rs_max_input_bytes in ttnn/cpp/ttnn/operations/ccl/reduce_scatter/
# reduce_scatter.cpp. tests/unit/test_batched_prefill.py checks the batched reduction against the batch-1 one on
# both sides of it.
DIRECT_REDUCE_SCATTER_MAX_BYTES = 512 << 10

# Fabric links of the batched reductions, where the mesh axis has them (batched_reduce_links; one worker core per link
# on the direct op; the ring ops split each slice's pages over the links). Every tile is still summed in the same
# order as the 1-link batch-1 call (tests/unit/test_batched_prefill.py). Measured on BH 1x4, bf8, ms per call,
# 1 -> 2 links:
#   direct reduce-scatter + all_gather [32, 1, 128, 2880]: 1.158 -> 0.664
#   ring all_reduce [32, 1, 512, 2880]: 1.871 -> 1.287; ring reduce_scatter [16, 1, 1024, 2880]: 1.278 -> 0.695
# More links than the fabric has: TT_FATAL (3 on BH 1x4).
BATCHED_REDUCE_LINKS = 2

_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048, ttnn.float32: 4096}

# chunks_per_sync of the prefill ring reduce-scatters (ring_reduce_scatter). The ring result does not depend on it
# (bit-identical to the default in every case measured). Measured on BH 1x4, bf8, 2 links, ms per call, default -> 128
# (l14/mb_ccl14.py, l14/logs/mb_ccl14.log):
#   [32, 1, 2048, 2880] 2.816 -> 2.503; [16, 1, 4096, 2880] 2.807 -> 2.491; [8, 1, 8192, 2880] 2.773 -> 2.456;
#   [32, 1, 512, 2880] 0.703 -> 0.647; [1, 1, 16384, 2880] 0.711 -> 0.647; [1, 1, 4096, 2880] 0.182 -> 0.184.
# 64 / 256 are within 0.5% of 128; num_workers_per_link 6 and num_buffers_per_channel 8 on top: within 1-4%, mixed.
RING_REDUCE_SCATTER_CHUNKS_PER_SYNC = 128


def _user_view(x, batch_size):
    """[1, 1, B*R, W] rows stacked by user -> [B, 1, R, W] (a view when R is a multiple of the tile height)."""
    rows = x.shape[-2]
    assert rows % batch_size == 0, f"{rows} rows do not split over {batch_size} users"
    return ttnn.reshape(x, (batch_size, 1, rows // batch_size, x.shape[-1]))


def _flat_view(x):
    """[B, 1, R, W] -> [1, 1, B*R, W]."""
    return ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[-2], x.shape[-1]))


def takes_direct_reduce_scatter(user_rows, width, dtype):
    """Whether a batch-1 [1, 1, user_rows, width] input of dtype takes the direct (one-shot) reduce-scatter."""
    tile_bytes = _TILE_BYTES.get(dtype)
    if tile_bytes is None:
        return False
    tiles = -(-user_rows // ttnn.TILE_SIZE) * -(-width // ttnn.TILE_SIZE)
    return tiles * tile_bytes <= DIRECT_REDUCE_SCATTER_MAX_BYTES


def _input_bytes(x):
    """Per-device bytes of a TILE tensor of a dtype in _TILE_BYTES (None otherwise)."""
    tile_bytes = _TILE_BYTES.get(x.dtype)
    if tile_bytes is None or x.layout != ttnn.TILE_LAYOUT:
        return None
    shape = x.padded_shape
    lead = 1
    for d in range(len(shape) - 2):
        lead *= shape[d]
    return lead * (shape[-2] // ttnn.TILE_SIZE) * (shape[-1] // ttnn.TILE_SIZE) * tile_bytes


def ring_reduce_scatter(x, cluster_axis, num_links, topology=ttnn.Topology.Ring, memory_config=None):
    """ttnn.reduce_scatter (dim 2) of a prefill partial sum, with RING_REDUCE_SCATTER_CHUNKS_PER_SYNC where the call
    takes the ring anyway. Any hyperparameter turns off the direct (one-shot) op in ttnn.reduce_scatter, which sums in
    another order, so an input of at most DIRECT_REDUCE_SCATTER_MAX_BYTES per device (or of unknown size, or another
    topology) keeps the plain call: same op, same rows as without this helper."""
    kw = {}
    nbytes = _input_bytes(x)
    if topology == ttnn.Topology.Ring and nbytes is not None and nbytes > DIRECT_REDUCE_SCATTER_MAX_BYTES:
        kw["chunks_per_sync"] = RING_REDUCE_SCATTER_CHUNKS_PER_SYNC
    if memory_config is not None:
        kw["memory_config"] = memory_config
    return ttnn.reduce_scatter(x, 2, cluster_axis=cluster_axis, num_links=num_links, topology=topology, **kw)


def _model_cache(ccl_manager):
    """Per-model state of the batched prefill, kept on the model's CCL manager: it lives as long as the model and its
    mesh, and a new mesh gets a new manager (no module-level cache keyed by a device id that a reopened mesh can
    reuse)."""
    return ccl_manager.__dict__.setdefault("_batched_prefill_cache", {})


def batched_reduce_links(ccl_manager, cluster_axis):
    """Links of per_user_reduce on cluster_axis: BATCHED_REDUCE_LINKS, capped at the links the mesh axis has
    (tt_transformers get_num_links: 2 on BH 1x4), and never fewer than the model's ccl_manager.num_links."""
    cache = _model_cache(ccl_manager)
    key = ("links", cluster_axis)
    if key not in cache:
        try:
            available = get_num_links(ccl_manager.mesh_device, cluster_axis)
        except (ValueError, KeyError):  # mesh not in the link table
            available = 0
        cache[key] = max(ccl_manager.num_links, min(BATCHED_REDUCE_LINKS, available))
    return cache[key]


def per_user_reduce(x, batch_size, cluster_axis, num_links, scatter=False):
    """Sum the [1, 1, B*S, H] partial sums of B users (stacked by user) over the mesh axis on num_links links
    (the model passes batched_reduce_links).

    scatter=False: the all-reduce, [1, 1, B*S, H] on every device. scatter=True: the reduce-scatter half,
    [1, 1, B*S/N, H] holding rows d*S/N .. of every user on device d. Each user's rows are bit-identical to
    ttnn.all_reduce / ttnn.reduce_scatter (dim 2) of that user's [1, 1, S, H] alone."""
    xb = _user_view(x, batch_size)
    user_rows = xb.shape[-2]
    direct = takes_direct_reduce_scatter(user_rows, xb.shape[-1], x.dtype)
    num_dev = xb.device().shape[cluster_axis]
    if not scatter and not direct and (user_rows // ttnn.TILE_SIZE) % num_dev:
        out = ttnn.all_reduce(
            xb,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return _flat_view(out)
    # ttnn.all_reduce of a [B, 1, S, H] whose S tiles divide by the mesh axis (H = 2880 is 90 tiles, which do not) is
    # this ring ttnn.reduce_scatter over dim 2 plus ttnn.all_gather; run them here to set chunks_per_sync on the ring.
    # Bit-identical, measured on BH 1x4, bf8, 2 links, ms per call (l14/logs/mb_ccl14.log): [32, 1, 512, 2880]
    # all_reduce 1.301 -> 1.244, [32, 1, 256, 2880] 0.658 -> 0.638 (all_gather_async in place of all_gather: slower).
    if direct:
        rs = ttnn.experimental.reduce_scatter_minimal_direct(
            xb,
            2,
            cluster_axis=cluster_axis,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        rs = ring_reduce_scatter(xb, cluster_axis, num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if scatter:
        return _flat_view(rs)
    out = ttnn.all_gather(rs, 2, cluster_axis=cluster_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rs.deallocate(True)
    return _flat_view(out)


def _all_gather_rows(x, cluster_axis, ccl_manager):
    """Row all-gather (dim 2) over cluster_axis. With ccl_manager: ttnn.experimental.all_gather_async on
    batched_reduce_links links (2 on BH 1x4; ttnn.all_gather picks its own links and workers). Pure data movement, so
    bit-identical. Measured on BH 1x4, bf8, ms per call, all_gather -> all_gather_async 2 links:
    [32, 1, 512, 2880] 2.394 -> 2.056; [16, 1, 1024, 2880] 2.388 -> 2.008; [8, 1, 2048, 2880] 2.389 -> 1.999."""
    if ccl_manager is None:
        return ttnn.all_gather(x, 2, cluster_axis=cluster_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.experimental.all_gather_async(
        x,
        dim=2,
        cluster_axis=cluster_axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Ring,
        multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
        num_links=batched_reduce_links(ccl_manager, cluster_axis),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
    )


def gather_users(x_rows, batch_size, cluster_axis, ccl_manager=None):
    """All-gather sequence-parallel rows: [1, 1, B*S/N, W] (device d: rows d*S/N .. of every user) ->
    [1, 1, B*S, W] stacked by user. batch_size 1 is the plain row all-gather. ccl_manager: gather with
    all_gather_async on 2 links (_all_gather_rows; the callers pass it for the [*, H] hidden rows)."""
    if batch_size == 1:
        return _all_gather_rows(x_rows, cluster_axis, ccl_manager)
    full = _all_gather_rows(_user_view(x_rows, batch_size), cluster_axis, ccl_manager)
    return _flat_view(full)


def partition_users(x, batch_size, cluster_axis):
    """Inverse of gather_users on a replicated [1, 1, B*S, W]: device d keeps rows d*S/N .. of every user."""
    if batch_size == 1:
        return ttnn.mesh_partition(x, 2, cluster_axis=cluster_axis)
    return _flat_view(ttnn.mesh_partition(_user_view(x, batch_size), 2, cluster_axis=cluster_axis))


def fill_batch_idx(ccl_manager, batch_size):
    """[B] uint32 0..B-1 (DRAM, row major, replicated on the model's mesh): paged_fill_cache batch_idx_tensor that
    fills input batch b through page-table row b. One per model and batch size (cached on the CCL manager)."""
    cache = _model_cache(ccl_manager)
    key = ("fill_batch_idx", batch_size)
    t = cache.get(key)
    if t is None:
        mesh_device = ccl_manager.mesh_device
        t = ttnn.from_torch(
            torch.arange(batch_size, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        cache[key] = t
    return t


def last_token_rows(hidden, last_token_idx, seq_len, mesh_device, cluster_axis=None):
    """Row last_token_idx[b] of every user b: [1, 1, B*S, H] stacked by user (replicated) or, with cluster_axis,
    sequence-parallel rows [1, 1, B*S/N, H] -> [1, 1, B, H] (B padded to the tile height with zero rows), same
    dtype as hidden.

    One-hot selection matmul: each output row is 1.0 * one input row plus zeros, exact at HiFi4 (the selected
    row keeps its bfloat8_b blocks, which hold whole 16-value row segments). With sequence-parallel rows each
    device selects the users whose last token it holds and an all-reduce adds the zero rows of the other
    devices."""
    batch = len(last_token_idx)
    out_rows = -(-batch // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    num_dev = 1 if cluster_axis is None else mesh_device.shape[cluster_axis]
    local = seq_len // num_dev
    rows = hidden.shape[-2]
    sel = torch.zeros(num_dev, 1, out_rows, rows, dtype=torch.float32)
    for b, t in enumerate(last_token_idx):
        t = int(t)
        assert 0 <= t < seq_len, f"user {b}: last token {t} outside [0, {seq_len})"
        d, i = divmod(t, local)
        sel[d, 0, b, b * local + i] = 1.0
    sel_tt = ttnn.from_torch(
        sel if num_dev > 1 else sel[0:1],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=(
            ttnn.ShardTensorToMesh(mesh_device, dim=0) if num_dev > 1 else ttnn.ReplicateTensorToMesh(mesh_device)
        ),
    )
    picked = ttnn.matmul(
        sel_tt,
        hidden,
        dtype=hidden.dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    sel_tt.deallocate(True)
    if num_dev > 1:
        summed = ttnn.all_reduce(
            picked,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        picked.deallocate(True)
        picked = summed
    return picked
