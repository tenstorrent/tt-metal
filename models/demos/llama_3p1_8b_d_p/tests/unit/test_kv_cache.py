# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for Llama-3.1 packed K/V caches."""

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.device_utils import addresses as _addresses
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics as _metrics
from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt import kv_cache as cache_module
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import (
    LlamaKVCache,
    allocate_kv_cache,
    max_user_slots,
    slot_bytes_per_chip,
    write_kv_chunk,
)
from models.demos.llama_3p1_8b_d_p.tt.qkv import QKVProjection
from models.demos.llama_3p1_8b_d_p.tt.rope import apply_indexed_rope, build_indexed_rope, build_transformation_mat

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
SP_AXIS = 0
GLOBAL_CHUNK = 1024
LOCAL_CHUNK = GLOBAL_CHUNK // SP
MAX_SEQ_LEN = 2048
LOCAL_CACHE_SEQUENCE = MAX_SEQ_LEN // SP
NUM_USERS = 2
NUM_LAYERS = 32
HEAD_DIM = Llama31_8BConfig.HEAD_DIM
HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
NUM_Q_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS
NUM_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
CACHE_SHAPE = (NUM_USERS * NUM_LAYERS, 1, LOCAL_CACHE_SEQUENCE, HEAD_DIM)
BF8_PAGE_BYTES = 4 * 1088
QKV_WEIGHT_NAMES = {
    "q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
    "v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
}


def _load_layer_zero_qkv_weights():
    return read_raw_weights(HF_MODEL, QKV_WEIGHT_NAMES)


def _half_split_to_adjacent_independent(tensor):
    half = tensor.shape[-1] // 2
    return torch.stack((tensor[..., :half], tensor[..., half:]), dim=-1).reshape(tensor.shape)


def _owned_positions(start):
    owned = [[] for _ in range(SP)]
    for position in range(start, start + GLOBAL_CHUNK):
        owned[(position % GLOBAL_CHUNK) // LOCAL_CHUNK].append(position)
    assert all(len(group) == LOCAL_CHUNK for group in owned)
    return owned


def _device_major_positions(start):
    return [position for group in _owned_positions(start) for position in group]


def _cache_rows_by_sp():
    rows = [[] for _ in range(SP)]
    for position in range(MAX_SEQ_LEN):
        rows[(position % GLOBAL_CHUNK) // LOCAL_CHUNK].append(position)
    assert all(len(group) == LOCAL_CACHE_SEQUENCE for group in rows)
    return rows


def _fixture_values(kind, positions):
    assert min(positions) >= 0 and max(positions) < 4096
    pos = torch.tensor(positions, dtype=torch.int64)
    heads = torch.arange(NUM_KV_HEADS, dtype=torch.int64)
    dims = torch.arange(HEAD_DIM, dtype=torch.int64)
    # Each 128-wide vector encodes the complete head/position identity as a repeated bit pattern.
    # Values 32 and 64 are exactly representable in BF16 and every BF8_B block, so a mismatch is
    # storage/placement corruption rather than source rounding.
    # Fifteen identity bits cover eight heads and every physical source position (up to 3039).
    tag = heads[:, None] * 4096 + pos[None, :]
    bit = (tag[:, :, None] >> (dims[None, None, :] % 15)) & 1
    magnitude = (32 + bit * 32).float()
    return magnitude if kind == "k" else -magnitude


def _chunk_fixture(start):
    positions = _device_major_positions(start)
    k_values = _fixture_values("k", positions).to(torch.bfloat16).float()
    v_values = _fixture_values("v", positions).to(torch.bfloat16).float()
    return positions, k_values, v_values


def _cache_memory_config(mesh_device):
    grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0))
            for bank in range(mesh_device.dram_grid_size().x)
        ]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, 32, HEAD_DIM],
            grid=grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _to_cache(mesh_device, value, dtype):
    host = torch.full(CACHE_SHAPE, value, dtype=torch.bfloat16)
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_cache_memory_config(mesh_device),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sentinel_cache(mesh_device, dtype):
    return LlamaKVCache(
        k=_to_cache(mesh_device, 7.0, dtype),
        v=_to_cache(mesh_device, -7.0, dtype),
        num_users=NUM_USERS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        sp=SP,
    )


def _to_chunk(mesh_device, values):
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _record(slot, layer, start, end, positions, k_values, v_values):
    return {
        "slot": slot,
        "layer": layer,
        "start": start,
        "end": end,
        "positions": positions,
        "k": k_values,
        "v": v_values,
    }


def _run_write(mesh_device, cache, *, slot, layer, start, end, positions=None, k_values=None, v_values=None):
    if positions is None:
        positions, k_values, v_values = _chunk_fixture(start)
    tt_k = _to_chunk(mesh_device, k_values)
    tt_v = _to_chunk(mesh_device, v_values)
    inputs = (tt_k, tt_v)
    before = [[ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)] for tensor in inputs]
    input_addresses = _addresses(tt_k) + _addresses(tt_v)
    cache_addresses = _addresses(cache.k) + _addresses(cache.v)
    write_kv_chunk(
        cache,
        tt_k,
        tt_v,
        slot_idx=slot,
        layer_idx=layer,
        actual_start=start,
        actual_end=end,
    )
    ttnn.synchronize_device(mesh_device)
    assert cache_addresses == _addresses(cache.k) + _addresses(cache.v)
    for tensor, snapshots in zip(inputs, before):
        for shard, snapshot in zip(ttnn.get_device_tensors(tensor), snapshots):
            assert torch.equal(ttnn.to_torch(shard), snapshot)
    return _record(slot, layer, start, end, positions, k_values, v_values), inputs, input_addresses


def _expected_plane(sp_coord, tp_coord, records, *, kind, sentinel):
    expected = torch.full(CACHE_SHAPE, sentinel, dtype=torch.float32)
    status = torch.zeros(CACHE_SHAPE[0], CACHE_SHAPE[2], dtype=torch.int8)
    positions_for_sp = _cache_rows_by_sp()[sp_coord]
    row_for_position = {position: row for row, position in enumerate(positions_for_sp)}
    for record in records:
        batch = record["slot"] * NUM_LAYERS + record["layer"]
        input_row = {position: index for index, position in enumerate(record["positions"])}
        for position in range(record["start"], record["end"]):
            if position >= MAX_SEQ_LEN or position not in row_for_position:
                continue
            row = row_for_position[position]
            expected[batch, 0, row] = record[kind][tp_coord, input_row[position]]
            status[batch, row] = 1
        padded_end = min(MAX_SEQ_LEN, math.ceil(record["end"] / 32) * 32)
        for position in range(record["end"], padded_end):
            if position in row_for_position:
                row = row_for_position[position]
                expected[batch, 0, row] = 0
                status[batch, row] = 2
    return expected, status


def _verify_cache(
    cache,
    records,
    dtype,
    *,
    label,
    sentinels=(7.0, -7.0),
    exact_payload=False,
):
    pcc_limit, nl2_limit = (0.9999, 0.01) if dtype == ttnn.bfloat16 else (0.999, 0.02)
    errors = []
    for kind, tensor, sentinel in (("k", cache.k, sentinels[0]), ("v", cache.v, sentinels[1])):
        shards = ttnn.get_device_tensors(tensor)
        assert len(shards) == SP * TP
        for sp_coord in range(SP):
            for tp_coord in range(TP):
                device_idx = sp_coord * TP + tp_coord
                actual = ttnn.to_torch(shards[device_idx]).float()[
                    : CACHE_SHAPE[0], :1, :LOCAL_CACHE_SEQUENCE, :HEAD_DIM
                ]
                expected, status = _expected_plane(sp_coord, tp_coord, records, kind=kind, sentinel=sentinel)
                untouched = status == 0
                padding = status == 2
                valid = status == 1
                assert torch.equal(actual[:, 0][untouched], expected[:, 0][untouched])
                assert torch.count_nonzero(actual[:, 0][padding]) == 0
                if torch.count_nonzero(valid):
                    expected_valid = expected[:, 0][valid]
                    actual_valid = actual[:, 0][valid]
                    if exact_payload:
                        assert torch.equal(
                            actual_valid, expected_valid
                        ), f"{label} {kind} chip={device_idx} changed an exactly representable payload"
                    pcc, nl2 = _metrics(expected_valid, actual_valid)
                    errors.append((pcc, nl2))
                    assert pcc >= pcc_limit, f"{label} {kind} chip={device_idx} PCC={pcc:.7f} NL2={nl2:.7f}"
                    assert nl2 <= nl2_limit, f"{label} {kind} chip={device_idx} PCC={pcc:.7f} NL2={nl2:.7f}"
    logger.info(
        f"{label}: dtype={dtype}, valid_written_min_PCC={min(x[0] for x in errors):.7f}, "
        f"valid_written_max_NL2={max(x[1] for x in errors):.7f}, "
        "untouched_exact=True, padding_exact_zero=True"
    )


# Model continuation must use a contiguous prefix shared by all executed layers. Replace only
# device validation/enqueue here to cover a gap, overlap, restart, empty write, and K/V enqueue
# failure without hardware. The placement test below exercises the same tracking on device.
def test_cache_prefix_tracks_complete_writes_and_invalidates_failed_suffix(monkeypatch, expect_error):
    cache = LlamaKVCache(object(), object(), NUM_USERS, NUM_LAYERS, MAX_SEQ_LEN, SP)
    calls = []
    failing_tensor = None

    def enqueue(tensor, source, **metadata):
        calls.append(tensor)
        if tensor is failing_tensor:
            raise RuntimeError("injected enqueue failure")

    monkeypatch.setattr(cache_module, "_validate_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(cache_module, "_write_one", enqueue)

    def write(slot, layer, start, end):
        write_kv_chunk(cache, None, None, slot_idx=slot, layer_idx=layer, actual_start=start, actual_end=end)

    assert cache.populated_end(0, 1) == 0
    write(0, 0, 0, 1024)
    write(0, 0, 1024, 1500)
    assert cache.populated_end(0, 1) == 1500
    assert cache.populated_end(0, 2) == 0
    write(0, 0, 1504, 1600)
    assert cache.populated_end(0, 1) == 1500
    write(0, 1, 0, 1024)
    write(0, 1, 1024, 1536)
    write(1, 0, 0, 512)
    assert cache.populated_end(0, 2) == 1500
    assert cache.populated_end(1, 1) == 512

    before_empty = len(calls)
    write(0, 0, 0, 0)
    assert len(calls) == before_empty
    assert cache.populated_end(0, 1) == 1500
    write(0, 0, 1472, 1600)
    assert cache.populated_end(0, 1) == 1600
    assert cache.populated_end(0, 2) == 1536
    write(0, 0, 0, 64)
    assert cache.populated_end(0, 1) == 64

    for tensor in (cache.k, cache.v):
        failing_tensor = tensor
        with expect_error(RuntimeError, "injected enqueue failure"):
            write(0, 0, 32, 100)
        assert cache.populated_end(0, 1) == 32
        assert cache.populated_end(1, 1) == 512
        failing_tensor = None
        write(0, 0, 0, 1024)
    write(0, 0, 1024, 1600)
    assert cache.populated_end(0, 2) == 1536

    # An old full-model cache must not survive a partial restart, including downstream
    # layers which a reduced model never reaches. Invalidate before the first layer runs.
    for layer in range(NUM_LAYERS):
        write(0, layer, 0, 1024)
    assert cache.populated_end(0, NUM_LAYERS) == 1024
    cache.truncate_prefix(0, 512)
    assert cache.populated_end(0, NUM_LAYERS) == 512
    cache.truncate_prefix(0, 0)
    write(0, 0, 0, 1024)
    assert cache.populated_end(0, 1) == 1024
    assert cache.populated_end(0, NUM_LAYERS) == 0
    assert cache.populated_end(1, 1) == 512
    for layer in range(1, NUM_LAYERS - 1):
        write(0, layer, 0, 1024)
    assert cache.populated_end(0, NUM_LAYERS - 1) == 1024
    assert cache.populated_end(0, NUM_LAYERS) == 0


# Allocate the exact 2-user/32-layer cache on the actual DRAM bank grid and read every chip; this
# catches wrong batch packing, local sequence size, dtype, NdShard page geometry, or nonzero startup.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8-b"])
def test_allocate_kv_cache_is_zero_and_has_packed_page_geometry(mesh_device, cache_dtype, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
    assert isinstance(cache, LlamaKVCache)
    assert all(cache.populated_end(slot, NUM_LAYERS) == 0 for slot in range(NUM_USERS))
    assert (cache.num_users, cache.num_layers, cache.max_seq_len, cache.sp) == (2, 32, 2048, 4)
    expected_grid = _cache_memory_config(mesh_device).nd_shard_spec.grid
    for tensor in (cache.k, cache.v):
        assert tuple(tensor.shape) == CACHE_SHAPE
        assert tensor.dtype == cache_dtype
        assert tensor.layout == ttnn.TILE_LAYOUT
        memory = tensor.memory_config()
        assert memory.buffer_type == ttnn.BufferType.DRAM
        assert tuple(memory.nd_shard_spec.shard_shape) == (1, 1, 32, 128)
        assert memory.nd_shard_spec.grid == expected_grid
        assert memory.nd_shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
        assert memory.nd_shard_spec.shard_distribution_strategy == ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
        for device_idx, shard in enumerate(ttnn.get_device_tensors(tensor)):
            assert torch.count_nonzero(ttnn.to_torch(shard)) == 0, f"nonzero cache on chip {device_idx}"
    assert BF8_PAGE_BYTES == 4352
    cache.k.deallocate(True)
    cache.v.deallocate(True)

    with expect_error(ValueError, "num_users must be a positive int"):
        allocate_kv_cache(mesh_device, mesh_config, num_users=0)
    with expect_error(TypeError, "num_users must be an eager Python int"):
        allocate_kv_cache(mesh_device, mesh_config, num_users=2.0)
    with expect_error(ValueError, "num_layers=32"):
        allocate_kv_cache(mesh_device, mesh_config, num_layers=31)
    with expect_error(ValueError, "positive multiple of 1024"):
        allocate_kv_cache(mesh_device, mesh_config, max_seq_len=2049)
    wrong_mesh = SimpleNamespace(mesh_shape=(8, 4), sp=8, tp=4, sp_axis=0, tp_axis=1)
    with expect_error(ValueError, "requires mesh_shape"):
        allocate_kv_cache(mesh_device, wrong_mesh)


# More than two concurrent users: allocate four slots and check the packing that makes them
# independent. Every slot is its own 32-plane K/V region at batch = slot * 32 + layer, so this
# catches a slot stride that collides, a batch extent that silently truncates the extra slots, and
# an out-of-range slot that would alias slot 0 instead of failing.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8-b"])
def test_allocate_kv_cache_keeps_extra_user_slots_independent(mesh_device, cache_dtype, expect_error):
    slots, layer = 4, 5
    cache = allocate_kv_cache(
        mesh_device,
        MeshConfig(MESH_SHAPE, TP),
        num_users=slots,
        max_seq_len=MAX_SEQ_LEN,
        cache_dtype=cache_dtype,
    )
    try:
        assert (cache.num_users, cache.num_layers, cache.max_seq_len, cache.sp) == (slots, NUM_LAYERS, MAX_SEQ_LEN, SP)
        for tensor in (cache.k, cache.v):
            assert tuple(tensor.shape) == (slots * NUM_LAYERS, 1, LOCAL_CACHE_SEQUENCE, HEAD_DIM)

        # One distinct power of two per slot: exact in BF16 and in every BF8_B block, so any
        # difference below is placement, not rounding.
        values = {slot: float(2**slot) for slot in range(slots)}
        for slot, value in values.items():
            chunk = torch.full((NUM_KV_HEADS, GLOBAL_CHUNK, HEAD_DIM), value)
            tt_k, tt_v = _to_chunk(mesh_device, chunk), _to_chunk(mesh_device, -chunk)
            write_kv_chunk(cache, tt_k, tt_v, slot_idx=slot, layer_idx=layer, actual_start=0, actual_end=GLOBAL_CHUNK)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
        ttnn.synchronize_device(mesh_device)

        # A GLOBAL_CHUNK write fills each SP rank's first LOCAL_CHUNK rows of the addressed plane.
        for kind, tensor, sign in (("k", cache.k, 1.0), ("v", cache.v, -1.0)):
            for device_idx, shard in enumerate(ttnn.get_device_tensors(tensor)):
                plane = ttnn.to_torch(shard).float()
                for batch in range(slots * NUM_LAYERS):
                    slot, written = divmod(batch, NUM_LAYERS)
                    want = sign * values[slot] if written == layer else 0.0
                    rows = plane[batch, 0, :LOCAL_CHUNK]
                    assert torch.equal(
                        rows, torch.full_like(rows, want)
                    ), f"{kind} chip={device_idx} batch={batch} (slot {slot}, layer {written}) expected {want}"
                    tail = plane[batch, 0, LOCAL_CHUNK:LOCAL_CACHE_SEQUENCE]
                    assert torch.count_nonzero(tail) == 0, f"{kind} chip={device_idx} batch={batch} tail not zero"

        chunk = torch.full((NUM_KV_HEADS, GLOBAL_CHUNK, HEAD_DIM), 1.0)
        tt_k, tt_v = _to_chunk(mesh_device, chunk), _to_chunk(mesh_device, chunk)
        with expect_error(ValueError, f"slot_idx {slots} out of range"):
            write_kv_chunk(cache, tt_k, tt_v, slot_idx=slots, layer_idx=0, actual_start=0, actual_end=GLOBAL_CHUNK)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        logger.info(f"{slots}-slot cache: per-slot planes independent for K and V on all {SP * TP} chips")
    finally:
        cache.k.deallocate(True)
        cache.v.deallocate(True)


# "As many slots as the device allows" is a division, and this pins the numerator and the divisor.
# The divisor is pure arithmetic, so assert the closed form against the layout it comes from rather
# than against itself: a slot is 32 layer planes of max_seq_len/4 rows, in both caches, replicated on
# every chip. A regression in local_cache_sequence or in the block-float element size would silently
# resize every auto-sized deployment, and nothing else in the suite would notice.
def test_slot_bytes_per_chip_is_exact_and_linear_in_capacity(expect_error):
    for max_seq_len in (1024, 2048, 8192, 32768, 131072):
        planes = NUM_LAYERS * (max_seq_len // SP) * HEAD_DIM
        assert slot_bytes_per_chip(max_seq_len, ttnn.bfloat16) == 2 * planes * 2
        assert slot_bytes_per_chip(max_seq_len, ttnn.bfloat8_b) == int(2 * planes * 1.0625)
        # Both caches, per chip, per token of capacity. Measured on a 4x8 Blackhole galaxy as
        # exactly 17.0 / 68.0 / 272.0 MiB at 8K / 32K / 128K, which these products reproduce.
        assert slot_bytes_per_chip(max_seq_len, ttnn.bfloat8_b) == 2176 * max_seq_len
        assert slot_bytes_per_chip(max_seq_len, ttnn.bfloat16) == 4096 * max_seq_len
    with expect_error(ValueError, "bfloat16 or bfloat8_b"):
        slot_bytes_per_chip(2048, ttnn.float32)


# The numerator: free DRAM, read at call time. Sizing off a hardware constant would be wrong the
# moment weights change, so this checks the count tracks what is actually free, shrinks as capacity
# grows, and that "max" really allocates what it promised. The reserve is inflated to leave room for
# a chosen handful of slots, which exercises the whole resolution path without filling DRAM -- and
# makes the assertion exact, since the count is then a number this test computed rather than
# whatever the device happened to have free.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_max_user_slots_tracks_free_dram_and_allocates_what_it_promises(mesh_device, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    banks = mesh_device.dram_grid_size().x
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    usable = min(view.total_bytes_free_per_bank, view.largest_contiguous_bytes_free_per_bank) * banks

    # Capacity and slot count trade off exactly: 4x the context, a quarter of the slots.
    small = max_user_slots(mesh_device, max_seq_len=8192, reserve_bytes=0)
    large = max_user_slots(mesh_device, max_seq_len=32768, reserve_bytes=0)
    assert small == usable // slot_bytes_per_chip(8192)
    assert abs(small - 4 * large) <= 4

    # The reserve is withheld, not ignored, and a reserve past the end is reported instead of
    # producing a zero-slot cache that would fail later at an unrelated call site.
    assert max_user_slots(mesh_device, max_seq_len=MAX_SEQ_LEN, reserve_bytes=usable) == 0
    with expect_error(RuntimeError, "no KV slot fits"):
        allocate_kv_cache(mesh_device, mesh_config, num_users="max", reserve_bytes=usable)

    wanted = 3
    slot_bytes = slot_bytes_per_chip(MAX_SEQ_LEN)
    reserve = usable - wanted * slot_bytes
    free_slots = max_user_slots(mesh_device, max_seq_len=MAX_SEQ_LEN, reserve_bytes=0)
    assert max_user_slots(mesh_device, max_seq_len=MAX_SEQ_LEN, reserve_bytes=reserve) == wanted
    before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank * banks
    cache = allocate_kv_cache(mesh_device, mesh_config, num_users="max", reserve_bytes=reserve)
    try:
        assert cache.num_users == wanted
        assert tuple(cache.k.shape) == (wanted * NUM_LAYERS, 1, LOCAL_CACHE_SEQUENCE, HEAD_DIM)
        taken = before - ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank * banks
        assert taken == wanted * slot_bytes, f"took {taken} B for {wanted} slots, predicted {wanted * slot_bytes} B"

        # The count is a live reading rather than a constant: with this cache resident the device
        # has exactly these slots fewer to offer than it did before allocating it. One slot of
        # tolerance, because a large allocation can also cost a little contiguity.
        now = max_user_slots(mesh_device, max_seq_len=MAX_SEQ_LEN, reserve_bytes=0)
        assert 0 <= (free_slots - wanted) - now <= 1, f"{free_slots} slots free, took {wanted}, now offers {now}"

        # Allocated is not the same as addressable: write the top slot, which carries the largest
        # batch index the packing produces, and read it back off the device.
        chunk = torch.full((NUM_KV_HEADS, GLOBAL_CHUNK, HEAD_DIM), 3.0)
        tt_k, tt_v = _to_chunk(mesh_device, chunk), _to_chunk(mesh_device, -chunk)
        write_kv_chunk(cache, tt_k, tt_v, slot_idx=wanted - 1, layer_idx=0, actual_start=0, actual_end=GLOBAL_CHUNK)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        top_plane = (wanted - 1) * NUM_LAYERS
        for device_idx, shard in enumerate(ttnn.get_device_tensors(cache.k)):
            rows = ttnn.to_torch(shard).float()[top_plane, 0, :LOCAL_CHUNK]
            assert torch.equal(rows, torch.full_like(rows, 3.0)), f"top slot unwritten on chip {device_idx}"
        logger.info(
            f"auto-sized cache: {wanted} slots took {taken / 2**20:.1f} MiB/chip, "
            f"top slot (batch {top_plane}) writable; a full-DRAM ask at max_seq_len={MAX_SEQ_LEN} "
            f"would have given {max_user_slots(mesh_device, max_seq_len=MAX_SEQ_LEN)} slots"
        )
    finally:
        cache.k.deallocate(True)
        cache.v.deallocate(True)


# Write boundary, partial-tile, full-chunk, continuation, and physical-tail cases into sentinel
# caches for BF16 and BF8_B; this catches wrong SP ownership, TP deduplication, slot/layer flattening,
# stale scalar metadata or addresses, tail truncation, padding damage, and mutations on rejected calls.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8-b"])
def test_write_kv_chunk_places_bounded_rows_and_reuses_programs(mesh_device, cache_dtype, expect_error):
    warm_cache = _sentinel_cache(mesh_device, cache_dtype)
    test_cache = _sentinel_cache(mesh_device, cache_dtype)
    assert set(_addresses(warm_cache.k) + _addresses(warm_cache.v)).isdisjoint(
        _addresses(test_cache.k) + _addresses(test_cache.v)
    )
    mesh_device.enable_program_cache()
    retained_inputs = []
    warm_cases = [(0, 0, 0, 1), (0, 13, 768, 1057), (1, 31, 1024, 2048)]
    for case in warm_cases:
        _, inputs, _ = _run_write(
            mesh_device,
            warm_cache,
            slot=case[0],
            layer=case[1],
            start=case[2],
            end=case[3],
        )
        retained_inputs.extend(inputs)
    warm_entries = mesh_device.num_program_cache_entries()
    assert warm_entries > 0
    warm_input_addresses = {address for tensor in retained_inputs for address in _addresses(tensor)}

    records = []
    scenarios = [
        (0, 0, 0, 1),
        (1, 13, 32, 33),
        (0, 31, 224, 257),
        (1, 0, 256, 1023),
        (0, 13, 768, 1057),
        (1, 31, 0, 1024),
        (1, 31, 1024, 2048),
        (0, 31, 2016, 2048),
    ]
    new_inputs = []
    for slot, layer, start, end in scenarios:
        record, inputs, addresses = _run_write(mesh_device, test_cache, slot=slot, layer=layer, start=start, end=end)
        assert warm_input_addresses.isdisjoint(addresses)
        records.append(record)
        new_inputs.extend(inputs)
        assert mesh_device.num_program_cache_entries() == warm_entries

    # Layer zero in slot zero contains only token zero. Gapped writes in other planes do
    # not advertise data before them, and neither slots nor layers share prefix metadata.
    assert test_cache.populated_end(0, 1) == 1
    assert test_cache.populated_end(0, NUM_LAYERS) == 0
    assert test_cache.populated_end(1, 1) == 0
    assert warm_cache.populated_end(0, 1) == 1
    before_noop = _addresses(test_cache.k) + _addresses(test_cache.v)
    noop_record, noop_inputs, _ = _run_write(mesh_device, test_cache, slot=1, layer=13, start=1024, end=1024)
    assert noop_record["start"] == noop_record["end"]
    assert before_noop == _addresses(test_cache.k) + _addresses(test_cache.v)
    assert mesh_device.num_program_cache_entries() == warm_entries
    new_inputs.extend(noop_inputs)

    valid_positions, valid_k, valid_v = _chunk_fixture(0)
    invalid_k = _to_chunk(mesh_device, valid_k)
    invalid_v = _to_chunk(mesh_device, valid_v)
    invalid_cases = [
        ({"slot_idx": -1, "layer_idx": 0, "actual_start": 0, "actual_end": 1}, "slot_idx"),
        ({"slot_idx": 2, "layer_idx": 0, "actual_start": 0, "actual_end": 1}, "slot_idx"),
        ({"slot_idx": 0, "layer_idx": -1, "actual_start": 0, "actual_end": 1}, "layer_idx"),
        ({"slot_idx": 0, "layer_idx": 32, "actual_start": 0, "actual_end": 1}, "layer_idx"),
        ({"slot_idx": 0, "layer_idx": 0, "actual_start": -32, "actual_end": 0}, "actual_start"),
        ({"slot_idx": 0, "layer_idx": 0, "actual_start": 1, "actual_end": 1}, "actual_start"),
        ({"slot_idx": 0, "layer_idx": 0, "actual_start": 32, "actual_end": 31}, "actual range"),
        ({"slot_idx": 0, "layer_idx": 0, "actual_start": 1024, "actual_end": 2049}, "actual range"),
        (
            {"slot_idx": 0, "layer_idx": 0, "actual_start": 0, "actual_end": 1025},
            "at most 1024",
        ),
    ]
    for metadata, message in invalid_cases:
        with expect_error((TypeError, ValueError), message):
            write_kv_chunk(test_cache, invalid_k, invalid_v, **metadata)
    with expect_error(TypeError, "slot_idx must be an eager Python int"):
        write_kv_chunk(test_cache, invalid_k, invalid_v, slot_idx=0.0, layer_idx=0, actual_start=0, actual_end=1)
    wrong_dtype = ttnn.typecast(invalid_k, ttnn.bfloat8_b)
    with expect_error(ValueError, "input must be bfloat16"):
        write_kv_chunk(test_cache, wrong_dtype, invalid_v, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
    wrong_dtype.deallocate(True)
    invalid_k.deallocate(True)
    invalid_v.deallocate(True)

    _verify_cache(
        test_cache,
        records,
        cache_dtype,
        label="cache-placement",
        exact_payload=True,
    )
    for tensor in retained_inputs + new_inputs:
        tensor.deallocate(True)
    for cache in (warm_cache, test_cache):
        cache.k.deallocate(True)
        cache.v.deallocate(True)


def _composition_input(start):
    positions = torch.tensor(_device_major_positions(start), dtype=torch.float32)
    columns = torch.arange(HIDDEN_SIZE, dtype=torch.float32)
    values = torch.sin(positions[:, None] / 23 + columns[None, :] / 127) * 0.125
    return values.reshape(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE)


def _composition_reference(host_input, weights, start, hf_rotary):
    x = host_input.to(torch.bfloat16).float()
    q_hf = F.linear(x, weights["q_proj.weight"].to(torch.bfloat16).float())
    k_hf = F.linear(x, weights["k_proj.weight"].to(torch.bfloat16).float())
    v = F.linear(x, weights["v_proj.weight"].to(torch.bfloat16).float())
    q_hf = q_hf.reshape(1, GLOBAL_CHUNK, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k_hf = k_hf.reshape(1, GLOBAL_CHUNK, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.reshape(1, GLOBAL_CHUNK, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    position_ids = torch.tensor([_device_major_positions(start)], dtype=torch.long)
    cos, sin = hf_rotary(q_hf, position_ids)
    _, k_hf = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    return _half_split_to_adjacent_independent(k_hf), v


# Compose real layer-0 projection, indexed Llama3 RoPE, and cache writes at a rotated boundary and
# physical tail; this catches a double/missing QK frame conversion and tail access that isolated
# projection, RoPE, or cache tests could each miss while still passing their own local oracle.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8-b"])
def test_real_qkv_rope_cache_composition_matches_hf(mesh_device, cache_dtype):
    weights = _load_layer_zero_qkv_weights()
    projection = QKVProjection(mesh_device, MeshConfig(MESH_SHAPE, TP), weights)
    rope_tables = build_indexed_rope(mesh_device, max_seq_len=MAX_SEQ_LEN, chunk_size=GLOBAL_CHUNK, sp_axis=SP_AXIS)
    transformation = build_transformation_mat(mesh_device)
    hf_rotary = LlamaRotaryEmbedding(AutoConfig.from_pretrained(HF_MODEL))
    cache = allocate_kv_cache(mesh_device, MeshConfig(MESH_SHAPE, TP), cache_dtype=cache_dtype)
    records = []
    for start, end in ((224, 257), (2016, 2048)):
        host_input = _composition_input(start)
        expected_k, expected_v = _composition_reference(host_input, weights, start, hf_rotary)
        tt_input = ttnn.from_torch(
            host_input.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None)),
        )
        input_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tt_input)]
        tt_q, tt_k, tt_v = projection(tt_input)
        tt_k_rot = apply_indexed_rope(tt_k, rope_tables, transformation, kv_actual_global=start, sp_axis=SP_AXIS)
        write_kv_chunk(
            cache,
            tt_k_rot,
            tt_v,
            slot_idx=0,
            layer_idx=0,
            actual_start=start,
            actual_end=end,
        )
        ttnn.synchronize_device(mesh_device)
        for shard, before in zip(ttnn.get_device_tensors(tt_input), input_before):
            assert torch.equal(ttnn.to_torch(shard), before)
        records.append(
            _record(
                0,
                0,
                start,
                end,
                _device_major_positions(start),
                expected_k.squeeze(0),
                expected_v.squeeze(0),
            )
        )
        for tensor in (tt_q, tt_k, tt_v, tt_k_rot, tt_input):
            tensor.deallocate(True)
    _verify_cache(cache, records, cache_dtype, label="real-qkv-rope-cache", sentinels=(0.0, 0.0))
    cache.k.deallocate(True)
    cache.v.deallocate(True)
    for tensor in (*rope_tables, transformation):
        tensor.deallocate(True)
