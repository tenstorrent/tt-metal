# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device validation for the zero_padded_kv_cache experimental op.

Seeds a block-cyclic cache with all-ones, runs the op, reconstructs natural order, asserts:
  * [0, valid_global)              real (==1)
  * [valid_global, ceil_128(v))    zero  (pad window)
  * [ceil_128(v), :)               real (==1)  -- nothing past the window
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
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

_FORMATS = [
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT),
]
_FORMAT_IDS = ["bfp8_tile", "bf16_rm", "fp8_rm"]

# The block-cyclic cases require SP=8. Preserve both existing execution environments: the LoudBox
# proxy uses TorusY and the production Galaxy uses TorusXY.
_MESHES = [
    pytest.param(
        (8, 1),
        torus_y_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
        id="torus-y-8x1",
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


def _init_cache_filled_with_ones(
    mesh_device,
    *,
    head_dim,
    seq_len_cache,
    chunk_size_global,
    sp_axis,
    dtype,
    layout,
    num_layers=1,
    num_users=1,
):
    """Initialize a block-cyclic cache and overwrite every physical row with one."""
    mesh_shape = list(mesh_device.shape)
    cache = init_kvpe_cache(
        head_dim,
        mesh_device,
        seq_len_cache,
        mesh_shape,
        sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
        dtype=dtype,
        layout=layout,
    )
    assert cache.dtype == dtype
    assert cache.layout == layout

    # FP8 cannot enter through this mesh-mapper path directly, so create BF16 row-major and typecast
    # on device, matching the production MLA path.
    ones = torch.ones(1, 1, chunk_size_global, head_dim, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        device=mesh_device,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(2, None)),
    )
    if dtype == ttnn.fp8_e4m3:
        tt_ones = ttnn.typecast(tt_ones, ttnn.fp8_e4m3)

    assert seq_len_cache % chunk_size_global == 0
    for slot_idx in range(num_users):
        for layer_idx in range(num_layers):
            for kv_actual_global in range(0, seq_len_cache, chunk_size_global):
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    tt_ones,
                    slot_idx=slot_idx,
                    layer_idx=layer_idx,
                    num_layers=num_layers,
                    kv_actual_global=kv_actual_global,
                    cluster_axis=sp_axis,
                )
    ttnn.synchronize_device(mesh_device)
    return cache


def _cache_in_natural_order(cache, mesh_device, *, chunk_size_global, seq_len_cache, sp_axis):
    """Gather a block-cyclic mesh cache and restore natural token order."""
    mesh_shape = list(mesh_device.shape)
    cache_shard_order = ttnn.to_torch(
        cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:, :1]
    positions = blockcyclic_positions(mesh_shape[sp_axis], chunk_size_global, seq_len_cache)
    natural = torch.empty(cache_shard_order.shape[0], seq_len_cache, cache_shard_order.shape[-1], dtype=torch.bfloat16)
    for batch_idx in range(cache_shard_order.shape[0]):
        natural[batch_idx, positions] = cache_shard_order[batch_idx, 0]
    return natural


def _assert_cache_windows(
    cache,
    mesh_device,
    *,
    chunk_size_global,
    seq_len_cache,
    sp_axis,
    num_layers,
    windows,
):
    """Check exact zero windows while requiring every other cache element to remain one."""
    natural = _cache_in_natural_order(
        cache,
        mesh_device,
        chunk_size_global=chunk_size_global,
        seq_len_cache=seq_len_cache,
        sp_axis=sp_axis,
    )
    expected = torch.ones_like(natural)
    for (slot_idx, layer_idx), (start, end) in windows.items():
        expected[slot_idx * num_layers + layer_idx, start:end] = 0
    assert torch.equal(natural, expected), (
        f"cache mismatch: {torch.count_nonzero(natural != expected).item()} elements differ; "
        f"actual range=[{natural.min().item()}, {natural.max().item()}]"
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("dtype,layout", _FORMATS, ids=_FORMAT_IDS)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_program_cache_cross_chip(mesh_device, dtype, layout):
    """Cover one cross-chip window and a program-cache hit with new runtime args."""
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    sp_axis = 0
    chunk_size_global = 384  # local=192: [180,256) crosses chip0 -> chip1
    seq_len_cache = 384
    head_dim = 64
    num_users = 2
    num_layers = 1

    cache = _init_cache_filled_with_ones(
        mesh_device,
        head_dim=head_dim,
        seq_len_cache=seq_len_cache,
        chunk_size_global=chunk_size_global,
        sp_axis=sp_axis,
        dtype=dtype,
        layout=layout,
        num_layers=num_layers,
        num_users=num_users,
    )

    mesh_device.enable_program_cache()
    cache_entries_before = mesh_device.num_program_cache_entries()
    windows = {
        (0, 0): (180, 256),  # crosses chip0 -> chip1
        (1, 0): (350, 384),  # same program, different slot and valid_global
    }
    for (slot_idx, layer_idx), (valid_global, _pad_end) in windows.items():
        ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
            cache,
            slot_idx,
            layer_idx,
            num_layers,
            valid_global,
            chunk_size_global,
            sp_axis,
            128,
        )
    ttnn.synchronize_device(mesh_device)

    _assert_cache_windows(
        cache,
        mesh_device,
        chunk_size_global=chunk_size_global,
        seq_len_cache=seq_len_cache,
        sp_axis=sp_axis,
        num_layers=num_layers,
        windows=windows,
    )

    # Both calls share structural args, so slot/valid changes must reuse one program.
    assert mesh_device.num_program_cache_entries() == cache_entries_before + 1


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
@pytest.mark.parametrize("dtype,layout", _FORMATS, ids=_FORMAT_IDS)
@pytest.mark.parametrize("chunk_size_global,seq_len_cache,valid_global,expected_chip", _CASES, ids=_IDS)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache(
    mesh_device, dtype, layout, chunk_size_global, seq_len_cache, valid_global, expected_chip
):
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    head_dim = 64

    cache = _init_cache_filled_with_ones(
        mesh_device,
        head_dim=head_dim,
        seq_len_cache=seq_len_cache,
        chunk_size_global=chunk_size_global,
        sp_axis=sp_axis,
        dtype=dtype,
        layout=layout,
    )

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

    ceil_v = math.ceil(valid_global / 128) * 128
    _assert_cache_windows(
        cache,
        mesh_device,
        chunk_size_global=chunk_size_global,
        seq_len_cache=seq_len_cache,
        sp_axis=sp_axis,
        num_layers=1,
        windows={(0, 0): (valid_global, ceil_v)},
    )
    logger.success(f"zero_padded_kv_cache {layout} {dtype} v={valid_global} PASSED")


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


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
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
    chunk_size_global = PREFILL_CHUNK_TOKENS
    seq_len_cache = chunk_size_global
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


# (slot_idx, valid_global): a single-tile partial (740), a 3-tile window (2600), a full-tile window
# with row_start=0 (4512), and a non-zero slot.
_EQUIV_CASES = [(0, 740), (0, 2600), (0, 4512), (1, 2600)]
_EQUIV_IDS = [f"slot{s}_v{v}" for (s, v) in _EQUIV_CASES]


def _make_scalar_tensor(mesh_device, value):
    """A 1-element uint32 tensor [1,1,1,1], ROW_MAJOR, DRAM, replicated across the mesh -- the
    per-element view the tensor-path overload reads element 0 from."""
    t = torch.tensor([[[[value]]]], dtype=torch.int32)  # ttnn maps int32->uint32 storage
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
@pytest.mark.parametrize("slot_idx,valid_global", _EQUIV_CASES, ids=_EQUIV_IDS)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_tensor_matches_scalar(mesh_device, slot_idx, valid_global):
    """The per-element-tensor path and the scalar path must produce bit-identical caches.

    Builds two 1-element uint32 replicated-DRAM tensors (slot_idx, valid_global), hands them to the
    tensor-path overload (the reader/writer read element 0 of each on-device), and compares the zeroed
    cache against the same call done via the original scalar signature, per device, bit-exact."""
    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    kvpe = 64
    chunk_size_global = PREFILL_CHUNK_TOKENS
    seq_len_cache = chunk_size_global
    seq_len_local = seq_len_cache // sp
    num_users, num_layers = 2, 1

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

    mesh_device.enable_program_cache()
    cache_scalar = _make_seeded_cache()
    cache_tensor = _make_seeded_cache()
    ttnn.synchronize_device(mesh_device)

    # 1-element uint32 tensors: slot_idx and valid_global, each read element 0 on-device.
    tt_slot_idx = _make_scalar_tensor(mesh_device, slot_idx)
    tt_valid_global = _make_scalar_tensor(mesh_device, valid_global)

    # Scalar path and per-element-tensor path on identical seeded caches.
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache_scalar, slot_idx, 0, num_layers, valid_global, chunk_size_global, sp_axis, 128
    )
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache_tensor, tt_slot_idx, tt_valid_global, 0, num_layers, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)

    scalar_devs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(cache_scalar)]
    tensor_devs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(cache_tensor)]
    for di, (a, b) in enumerate(zip(tensor_devs, scalar_devs)):
        assert torch.equal(a, b), (
            f"slot {slot_idx} v {valid_global} device {di}: tensor-path cache differs from scalar-path "
            f"(max abs diff {(a - b).abs().max().item()})"
        )
    logger.success(f"slot={slot_idx} valid_global={valid_global}: tensor path == scalar path (bit-exact)")
    ttnn.deallocate(tt_slot_idx)
    ttnn.deallocate(tt_valid_global)


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_tensor_program_reuse(mesh_device):
    """Cache-HIT coverage of the tensor path: two successive tensor-overload calls with DIFFERENT
    1-element metadata tensors (distinct DRAM addresses) must reuse ONE cached program, and the second
    call must zero ITS OWN window -- proving override_runtime_arguments patches the metadata tensor
    addresses (common args 10/11) on the cache hit rather than reusing the first call's tensors. (The
    tensor path is TILE-only, so this uses a TILE cache.)"""
    sp_axis = 0
    kvpe = 64
    chunk_size_global = PREFILL_CHUNK_TOKENS
    seq_len_cache = chunk_size_global
    num_users, num_layers = 2, 1

    cache = _init_cache_filled_with_ones(
        mesh_device,
        head_dim=kvpe,
        seq_len_cache=seq_len_cache,
        chunk_size_global=chunk_size_global,
        sp_axis=sp_axis,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        num_layers=num_layers,
        num_users=num_users,
    )
    mesh_device.enable_program_cache()

    # Two DISTINCT (slot, valid_global) pairs, each via its OWN metadata tensors (distinct DRAM addresses).
    slot_a, va = 0, 740
    slot_b, vb = 1, 2600
    ceil128 = lambda v: math.ceil(v / 128) * 128  # noqa: E731

    slot_ta, valid_ta = _make_scalar_tensor(mesh_device, slot_a), _make_scalar_tensor(mesh_device, va)
    # First tensor-overload call: program-cache MISS (compiles the tensor program).
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, slot_ta, valid_ta, 0, num_layers, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)
    entries_after_first = mesh_device.num_program_cache_entries()

    slot_tb, valid_tb = _make_scalar_tensor(mesh_device, slot_b), _make_scalar_tensor(mesh_device, vb)
    # Second call with DIFFERENT tensors -> program-cache HIT; the override must patch the new DRAM
    # addresses (common args 10/11) so the kernel reads slot_b/vb, not the first call's slot_a/va.
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, slot_tb, valid_tb, 0, num_layers, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)

    assert mesh_device.num_program_cache_entries() == entries_after_first, (
        f"tensor-path program was recompiled on the second call instead of reused "
        f"({entries_after_first} -> {mesh_device.num_program_cache_entries()}) -- successive chunks must "
        f"reuse one cached program via the patched metadata addresses"
    )
    # Both windows zeroed, every other element still one: proves the SECOND (cache-hit) call read the NEW
    # tensors' values (slot_b/vb) via the patched addresses. If patching had failed it would have re-zeroed
    # slot_a/va and slot_b's window would still be all ones -> this assertion would fail.
    _assert_cache_windows(
        cache,
        mesh_device,
        chunk_size_global=chunk_size_global,
        seq_len_cache=seq_len_cache,
        sp_axis=sp_axis,
        num_layers=num_layers,
        windows={(slot_a, 0): (va, ceil128(va)), (slot_b, 0): (vb, ceil128(vb))},
    )
    logger.success("tensor-path program reused across chunks; second (cache-hit) call zeroed its own window")
    for t in (slot_ta, valid_ta, slot_tb, valid_tb):
        ttnn.deallocate(t)
