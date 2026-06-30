# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device validation for the zero_padded_kv_cache experimental op.

Seeds a block-cyclic cache with all-ones, runs the op, reconstructs natural order, asserts:
  * [0, valid_global)              real (==1)
  * [valid_global, ceil_128(v))    zero  (pad window)
  * [ceil_128(v), :)               real (==1)  -- nothing past the window
"""
import math
import struct

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

# Production chunk_size_global only (chunk_local = 5120/8 = 640). At this csg a 128-pad window never
# crosses a chip border, so every case below lands its window on a single chip; the op's multi-chip
# path (window crossing) stays exercised by the on-device per-chip logic but is not csg-varied here.
# (chunk_size_global, seq_len_cache, valid_global, expected_boundary_chip)  expected_chip=None => skip
_CASES = [
    (5120, 5120, 740, 1),  # chip1 slab0, single-tile partial
    (5120, 5120, 2600, 4),  # chip4 slab0, 3-tile window
    (5120, 5120, 4512, 7),  # chip7 slab0, row_start=0 -> 3 full tiles
    (5120, 5120, 20, 0),  # chip0 slab0, 4-tile window
    (5120, 5120, 4480, None),  # 128-aligned -> skip
    (5120, 10240, 6668, 2),  # chip2 slab1, 4-tile window
]
_IDS = [f"v{v}_chip{ch}" if ch is not None else f"v{v}_aligned" for (_c, _s, v, ch) in _CASES]


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize("chunk_size_global,seq_len_cache,valid_global,expected_chip", _CASES, ids=_IDS)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache(mesh_device, chunk_size_global, seq_len_cache, valid_global, expected_chip):
    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[1]
    kvpe = 64
    seq_len_local = seq_len_cache // sp

    cache = init_kvpe_cache(kvpe, mesh_device, seq_len_cache, mesh_shape, sp_axis, num_kvpe_cache_layers=1)

    # seed all-ones across the whole cache
    ones = torch.ones(1, 1, seq_len_local, kvpe, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.fill_cache(cache, tt_ones, 0, update_idx=0)
    ttnn.synchronize_device(mesh_device)

    if expected_chip is not None:
        tile_start = (valid_global // 32) * 32
        chip = (tile_start % chunk_size_global) // (chunk_size_global // sp)  # block-cyclic boundary chip
        assert chip == expected_chip, f"v={valid_global} -> chip {chip}, expected {expected_chip}"
        logger.info(f"valid_global={valid_global} boundary chip={chip}")

    # ---- the device op ----
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, 0, 0, 1, valid_global, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)

    # ---- reconstruct natural order ----
    positions = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    natural = torch.zeros(seq_len_cache)
    for di, dt in enumerate(ttnn.get_device_tensors(cache)):
        if di % tp != 0:
            continue
        sp_coord = di // tp
        rows = ttnn.to_torch(dt).float()[0, 0, :, :].mean(dim=-1)
        for lr in range(seq_len_local):
            natural[int(positions[sp_coord * seq_len_local + lr].item())] = rows[lr].item()

    ceil_v = math.ceil(valid_global / 128) * 128
    real, pad, rest = natural[:valid_global], natural[valid_global:ceil_v], natural[ceil_v:]
    logger.info(
        f"v={valid_global} ceil128={ceil_v}: real.min={real.min():.2f} "
        f"pad.max={pad.max() if len(pad) else 0:.2f} rest.min={rest.min() if len(rest) else 1:.2f}"
    )
    assert real.min() > 0.9, f"real [0,{valid_global}) clobbered (min={real.min()})"
    if len(pad):
        assert pad.max() < 0.1, f"pad [{valid_global},{ceil_v}) not zeroed (max={pad.max()})"
    if len(rest):
        assert rest.min() > 0.9, f"rows past window [{ceil_v},:) touched (min={rest.min()})"
    logger.success(f"zero_padded_kv_cache v={valid_global} PASSED")


# (num_layers, num_users, slot_idx, layer_idx, valid_global, expected_chip) -- exercises the cache
# batch-slot linearization batch_idx = slot_idx*num_layers + layer_idx (layer_idx>0, slot_idx>0) and
# that the op touches ONLY the target slot. Full 61-layer model + multi-user kv cache.
_MULTI_CASES = [
    (61, 1, 0, 0, 2600, 4),  # full 61-layer cache, user0, layer0
    (61, 1, 0, 60, 2600, 4),  # full 61-layer cache, user0, LAST layer (batch=60)
    (61, 2, 1, 60, 2600, 4),  # full 61-layer cache, user1, LAST layer (batch=121)
    (8, 2, 1, 5, 4512, 7),  # multi-user, user1 layer5 (batch=13), full-tile window
]
_MULTI_IDS = [f"L{nl}_U{nu}_slot{s}_layer{ly}_v{v}" for (nl, nu, s, ly, v, _ch) in _MULTI_CASES]


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "num_layers,num_users,slot_idx,layer_idx,valid_global,expected_chip", _MULTI_CASES, ids=_MULTI_IDS
)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_layers_users(
    mesh_device, num_layers, num_users, slot_idx, layer_idx, valid_global, expected_chip
):
    """Seed every (user, layer) cache slot to all-ones, zero ONE slot's pad window, and assert the
    target slot's window is zeroed AND every other slot is untouched -- validating the
    batch_idx = slot_idx*num_layers + layer_idx addressing across a full 61-layer / multi-user cache."""
    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[1]
    kvpe = 64
    chunk_size_global = 5120
    seq_len_cache = 5120
    seq_len_local = seq_len_cache // sp
    num_batches = num_users * num_layers
    target_batch = slot_idx * num_layers + layer_idx

    cache = init_kvpe_cache(
        kvpe,
        mesh_device,
        seq_len_cache,
        mesh_shape,
        sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )

    # seed EVERY batch slot to all-ones so a wrongly-touched slot shows up as a non-1 reading
    ones = torch.ones(1, 1, seq_len_local, kvpe, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for b in range(num_batches):
        ttnn.fill_cache(cache, tt_ones, b, update_idx=0)
    ttnn.synchronize_device(mesh_device)

    logger.info(
        f"layers={num_layers} users={num_users} -> {num_batches} slots; "
        f"zeroing slot={slot_idx} layer={layer_idx} -> target_batch={target_batch}"
    )

    # ---- the device op (only the target (slot, layer)) ----
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, slot_idx, layer_idx, num_layers, valid_global, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)

    # ---- verify: target slot window zeroed, all other slots untouched ----
    positions = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    natural = torch.zeros(seq_len_cache)
    other_min = 1.0
    for di, dt in enumerate(ttnn.get_device_tensors(cache)):
        if di % tp != 0:
            continue
        sp_coord = di // tp
        t = ttnn.to_torch(dt).float()  # [num_batches, 1, seq_len_local, kvpe]
        for b in range(num_batches):
            rows = t[b, 0, :, :].mean(dim=-1)
            if b == target_batch:
                for lr in range(seq_len_local):
                    natural[int(positions[sp_coord * seq_len_local + lr].item())] = rows[lr].item()
            else:
                other_min = min(other_min, rows.min().item())

    ceil_v = math.ceil(valid_global / 128) * 128
    real, pad, rest = natural[:valid_global], natural[valid_global:ceil_v], natural[ceil_v:]
    logger.info(
        f"target_batch={target_batch}: real.min={real.min():.2f} pad.max={pad.max():.2f} "
        f"rest.min={rest.min():.2f} | other_slots.min={other_min:.2f}"
    )
    assert other_min > 0.9, f"a non-target slot was corrupted (min={other_min})"
    assert real.min() > 0.9, f"target real [0,{valid_global}) clobbered (min={real.min()})"
    assert pad.max() < 0.1, f"target pad [{valid_global},{ceil_v}) not zeroed (max={pad.max()})"
    assert rest.min() > 0.9, f"target rows past window touched (min={rest.min()})"
    logger.success(f"layers={num_layers} users={num_users} slot={slot_idx} layer={layer_idx} PASSED")


# 3 x uint32: [slot_id, actual_start, actual_end] -- the runner's canonical h2d_socket_sync payload.
# zero_padded_kv_cache reads slot_idx (index 0) and valid_global (= actual_end, index 2).
_H2D_METADATA_SIZE_BYTES = 12

# (slot_idx, valid_global): a single-tile partial (740), a 3-tile window (2600), a full-tile window
# with row_start=0 (4512), and a non-zero slot. valid_global = actual_end.
_EQUIV_CASES = [(0, 740), (0, 2600), (0, 4512), (1, 2600)]
_EQUIV_IDS = [f"slot{s}_v{v}" for (s, v) in _EQUIV_CASES]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("slot_idx,valid_global", _EQUIV_CASES, ids=_EQUIV_IDS)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_metadata_matches_scalar(mesh_device, slot_idx, valid_global):
    """The metadata path and the scalar path must produce bit-identical caches.

    Drives the metadata path from a REAL H2D service (not a hand-built tensor): push the runner's
    canonical [slot_id, actual_start, actual_end] payload through ttnn.H2DStreamService +
    inbound_socket_service_sync, hand the resulting device metadata tensor to the op (it reads slot_idx
    = index 0 and valid_global = actual_end = index 2 on-device), and compare the zeroed cache against
    the same call done via the original scalar signature, per device, bit-exact."""
    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    kvpe = 64
    chunk_size_global = 5120
    seq_len_cache = 5120
    seq_len_local = seq_len_cache // sp
    num_users, num_layers = 2, 1
    isl_per_chip = chunk_size_global // sp
    per_chip_bytes = isl_per_chip * 4  # uint32

    def _make_seeded_cache():
        c = init_kvpe_cache(
            kvpe, mesh_device, seq_len_cache, mesh_shape, sp_axis, num_kvpe_cache_layers=num_layers, num_users=num_users
        )
        ones = torch.ones(1, 1, seq_len_local, kvpe, dtype=torch.bfloat16)
        tt_ones = ttnn.from_torch(
            ones,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for b in range(num_users * num_layers):
            ttnn.fill_cache(c, tt_ones, b, update_idx=0)
        return c

    # Real H2D service (same construction as the runner / test_h2d_socket_sync).
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([sp, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    mapper = ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
    )
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,
        scratch_cb_size_bytes=per_chip_bytes,
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=_H2D_METADATA_SIZE_BYTES,
    )

    mesh_device.enable_program_cache()
    try:
        cache_scalar = _make_seeded_cache()
        cache_meta = _make_seeded_cache()
        ttnn.synchronize_device(mesh_device)

        # Metadata from the real H2D service: [slot_id, actual_start(unused here), actual_end=valid_global].
        dummy_tokens = torch.zeros(sp, 1, isl_per_chip, dtype=torch.int32).contiguous().numpy()
        meta = struct.pack("<III", slot_idx, 0, valid_global)
        service.forward_to_tensor_bytes(dummy_tokens, metadata=meta)
        tt_tokens, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_H2D_METADATA_SIZE_BYTES
        )
        ttnn.deallocate(tt_tokens)

        # Scalar path and metadata path on identical seeded caches.
        ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
            cache_scalar, slot_idx, 0, num_layers, valid_global, chunk_size_global, sp_axis, 128
        )
        ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
            cache_meta, tt_meta, 0, num_layers, chunk_size_global, sp_axis, 128
        )
        ttnn.synchronize_device(mesh_device)

        scalar_devs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(cache_scalar)]
        meta_devs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(cache_meta)]
        for di, (a, b) in enumerate(zip(meta_devs, scalar_devs)):
            assert torch.equal(a, b), (
                f"slot {slot_idx} v {valid_global} device {di}: metadata-path cache differs from scalar-path "
                f"(max abs diff {(a - b).abs().max().item()})"
            )
        logger.success(f"slot={slot_idx} valid_global={valid_global}: metadata path == scalar path (bit-exact)")
        ttnn.deallocate(tt_meta)
    finally:
        service.barrier()
        del service
