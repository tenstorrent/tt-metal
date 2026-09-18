# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""KV-cache tests for Llama-3.1-8B prefill (tt-blaze#4141).

Split deliberately into host math and device writes, because they fail in different ways.

The **host math** (slot packing, the block-cyclic position inverse, the multi-turn resume
alignment, and the two divisibility rules) is what the migration side's address table also has to
agree with. It needs no device, so it is tested exhaustively and cheaply here rather than inferred
from a PCC number. The block-cyclic inverse in particular is asserted to be a true permutation of
the cache: a subtly wrong walk still covers *most* positions, so spot-checking a few rows would
pass while some tokens silently shared a row.

The **device writes** check that ``update_padded_kv_cache`` lands a chunk where the host math says
it should, including the tail-pad chunk (where only ``[actual_start, actual_end)`` may be written
and the padded remainder must not be) and the multi-chunk case (where chunk 1 must not disturb
chunk 0).

Cache dtype is ``bfloat8_b``, so the reference is round-tripped through bf8 before comparing. A
full-precision reference leaves a spurious ~0.94-0.96 gap that reads as a real bug.

Run (host math only, no device):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_kv_cache_vs_ref.py -k "not device"
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK as BANK_TOKENS
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import (
    aligned_resume_length,
    allocate_kv_cache,
    rotated_chip_positions,
    slot_index,
    validate_chunk_layout,
    write_kv_chunk,
)

HEAD_DIM = Llama31_8BConfig.HEAD_DIM  # 128
NUM_LAYERS = Llama31_8BConfig.NUM_LAYERS  # 32


# =====================================================================================
# Host math — no device
# =====================================================================================
def test_slot_packing_is_user_major():
    """``slot = user * num_layers + layer``, and the packing is a bijection.

    User-major keeps a user's 32 layers contiguous, which is what lets the migration address table
    describe a user with one base offset. The bijection check is the point: layer-major packing
    (``layer * num_users + user``) produces identical slots when ``num_users == 1``, so bring-up on a
    single user cannot distinguish them and the mismatch would first appear as cross-talk between
    users.
    """
    num_users = 4
    seen = {}
    for user in range(num_users):
        for layer in range(NUM_LAYERS):
            slot = slot_index(user, layer, NUM_LAYERS)
            assert slot not in seen, f"slot {slot} collides: {seen[slot]} and {(user, layer)}"
            seen[slot] = (user, layer)

    assert len(seen) == num_users * NUM_LAYERS
    assert max(seen) == num_users * NUM_LAYERS - 1, "packing must be dense — no unused slots"
    # A user's layers are contiguous and ascending.
    for user in range(num_users):
        slots = [slot_index(user, layer, NUM_LAYERS) for layer in range(NUM_LAYERS)]
        assert slots == list(range(slots[0], slots[0] + NUM_LAYERS))
    # And distinguishable from layer-major with more than one user.
    assert slot_index(1, 0, NUM_LAYERS) != 0 * num_users + 1


@pytest.mark.parametrize("sp", [1, 2, 4, 8])
def test_chunk_layout_rules_accept_the_served_configuration(sp):
    """The real chunk sizes pass for every plausible SP."""
    for chunk_size in (256, 512, 1024, 2048):
        if chunk_size % (sp * BANK_TOKENS):
            continue
        validate_chunk_layout(max_seq_len=8192, chunk_size=chunk_size, sp=sp)


def test_chunk_layout_rules_are_enforced(expect_error):
    """``max_seq_len % chunk_size`` and ``chunk_size % (sp * 32)`` are checked, not assumed.

    Neither violation crashes at write time — they place tokens in the wrong DRAM bank or on the
    wrong chip, which only shows up as a KV mismatch at some interior position.
    """
    with expect_error(ValueError, "multiple of chunk_size"):
        validate_chunk_layout(max_seq_len=5000, chunk_size=512, sp=4)
    # 128 tokens over sp=8 is 16 per chip — half a 32-token DRAM shard.
    with expect_error(ValueError, "multiple of sp"):
        validate_chunk_layout(max_seq_len=8192, chunk_size=128, sp=8)
    # Divides max_seq_len evenly, so only the bank-granularity rule can catch it.
    with expect_error(ValueError, "multiple of sp"):
        validate_chunk_layout(max_seq_len=8192, chunk_size=16, sp=1)


def test_aligned_resume_rounds_down_and_never_leaves_a_hole():
    """A continuation resumes at the previous length aligned DOWN, replaying the remainder.

    Rounding up would satisfy ``update_padded_kv_cache``'s ``% 32 == 0`` assertion just as well,
    which is why this needs pinning: the up-rounded offset skips the partial bank, leaving up to 31
    positions mid-sequence that nothing ever writes. Attention would read stale DRAM there,
    permanently, and only for continued conversations.
    """
    for prev in (0, 1, 31, 32, 33, 63, 64, 8191):
        resumed = aligned_resume_length(prev)
        assert resumed % BANK_TOKENS == 0, f"{resumed} is not bank-aligned"
        assert resumed <= prev, f"resume point {resumed} is past the written prefix {prev}"
        assert prev - resumed < BANK_TOKENS, "dropped more than one bank's worth of tokens"

    assert aligned_resume_length(32) == 32, "an already-aligned length must not move"
    assert aligned_resume_length(63) == 32
    # The replayed span is what keeps the sequence hole-free; rounding up would skip it.
    prev = 70
    assert aligned_resume_length(prev) == 64
    rounded_up = -(-prev // BANK_TOKENS) * BANK_TOKENS
    assert rounded_up == 96 and rounded_up > prev, "round-up would start past the written prefix"


@pytest.mark.parametrize("sp", [2, 4])
@pytest.mark.parametrize("chunk_local", [32, 64, 256])
def test_rotated_chip_positions_matches_the_deepseek_original(sp, chunk_local):
    """The local copy tracks ``deepseek_v3_d_p``'s, which is the one graded against the kernel.

    Llama restates it instead of importing it, because that module drags ``safetensors`` and
    ``transformers`` onto the import path of an import-light runtime. Restating is only safe while
    something notices a drift, which is this test — the same arrangement ``block_cyclic_reorder``
    has in ``test_rope_vs_ref.py``.

    Swept over offsets that cover all three branches of the staircase: slab-aligned (the identity
    case), inside a chip's slab (the boundary chip rotates too) and past a whole slab.
    """
    from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions as original

    chunk_global = sp * chunk_local
    offsets = [0, 32, chunk_local, chunk_local + 32, chunk_global, chunk_global + chunk_local + 32]
    for kv_actual in offsets:
        assert rotated_chip_positions(kv_actual, sp, chunk_local) == original(
            kv_actual, sp, chunk_local
        ), f"local copy diverged at kv_actual={kv_actual}, sp={sp}, chunk_local={chunk_local}"


@pytest.mark.parametrize("sp", [2, 4])
@pytest.mark.parametrize("chunk_local", [32, 64])
def test_rotated_chip_positions_tiles_the_chunk_exactly(sp, chunk_local):
    """Every position in the chunk appears once, and each chip's rows increase.

    Both properties are load-bearing and neither is implied by matching the reference. The cover
    means no position is dropped or written twice; the monotonicity per chip is what keeps a
    request's real tokens a prefix and its pad a suffix on every chip, which is the reason the pad
    tail can be left inert under causality with no extra plumbing.
    """
    chunk_global = sp * chunk_local
    for kv_actual in (0, 32, chunk_local + 32, chunk_global + 32):
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = sorted(p for chip in positions for p in chip)
        assert flat == list(
            range(kv_actual, kv_actual + chunk_global)
        ), f"chunk at {kv_actual} does not tile [{kv_actual}, {kv_actual + chunk_global}) exactly"
        for chip, rows in enumerate(positions):
            assert rows == sorted(rows), f"chip {chip} positions are not increasing at kv_actual={kv_actual}"


def test_rotated_chip_positions_is_contiguous_only_when_chunk_aligned():
    """The identity case, stated as a test so the trap in #4148 is written down somewhere.

    At a chunk-aligned offset the map is exactly the contiguous split, which is why prompt order
    works for every single-turn request; one tile further along it is not, which is why the first
    continuation breaks a runtime that assumes prompt order.
    """
    sp, chunk_local = 4, 64
    chunk_global = sp * chunk_local

    for aligned in (0, chunk_global, 3 * chunk_global):
        positions = rotated_chip_positions(aligned, sp, chunk_local)
        expected = [[aligned + c * chunk_local + r for r in range(chunk_local)] for c in range(sp)]
        assert positions == expected, f"offset {aligned} should be the contiguous split"

    off_by_one_tile = rotated_chip_positions(chunk_global + 32, sp, chunk_local)
    contiguous = [[chunk_global + 32 + c * chunk_local + r for r in range(chunk_local)] for c in range(sp)]
    assert off_by_one_tile != contiguous, "a 32-aligned non-chunk-aligned offset must not be contiguous"


@pytest.mark.parametrize("sp", [1, 2, 4])
@pytest.mark.parametrize("chunk_size, max_seq_len", [(128, 512), (256, 1024), (128, 1024)])
def test_blockcyclic_positions_is_a_permutation(sp, chunk_size, max_seq_len):
    """The block-cyclic inverse maps cache rows to global positions bijectively.

    ``blockcyclic_positions`` is the host-side inverse of the writer kernel's walk, and every
    golden comparison and every migration address depends on it. Asserting it is a permutation of
    ``[0, max_seq_len)`` is much stronger than checking a few rows: a walk that is wrong for one
    chip still covers most positions, so two cache rows would claim the same token and a spot check
    would miss it.
    """
    if chunk_size % sp or max_seq_len % chunk_size:
        pytest.skip("configuration not expressible for this sp")

    positions = blockcyclic_positions(sp, chunk_size, max_seq_len)
    assert positions.shape == (max_seq_len,)
    assert torch.equal(positions.sort().values, torch.arange(max_seq_len)), "not a permutation"

    # Each chip owns a contiguous slice of the cache; within a slab its rows are consecutive global
    # positions, which is what makes the per-chip write a plain contiguous copy.
    seq_local = max_seq_len // sp
    chunk_local = chunk_size // sp
    for chip in range(sp):
        chip_rows = positions[chip * seq_local : (chip + 1) * seq_local]
        first_slab = chip_rows[:chunk_local]
        expected = torch.arange(chip * chunk_local, chip * chunk_local + chunk_local)
        assert torch.equal(first_slab, expected), f"chip {chip} slab 0 is not contiguous: {first_slab[:8]}"


def test_blockcyclic_first_chunk_covers_exactly_the_first_chunk():
    """Chunk 0's rows hold global positions ``[0, chunk_size)`` and nothing else.

    Pins the property the multi-chunk device test relies on: writing chunk 1 must not touch any row
    that chunk 0 owns.
    """
    sp, chunk_size, max_seq_len = 4, 256, 1024
    positions = blockcyclic_positions(sp, chunk_size, max_seq_len)
    seq_local, chunk_local = max_seq_len // sp, chunk_size // sp

    owned = torch.cat([positions[c * seq_local : c * seq_local + chunk_local] for c in range(sp)])
    assert torch.equal(owned.sort().values, torch.arange(chunk_size))


# =====================================================================================
# Device writes
# =====================================================================================
def _to_bf8_roundtrip(t: torch.Tensor, mesh_device) -> torch.Tensor:
    """Round-trip a torch tensor through the device's bfloat8_b so the reference carries the same
    quantisation as the cache. Comparing against full precision leaves a ~0.94-0.96 PCC gap."""
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt)[0]).float()
    ttnn.deallocate(tt)
    return out


def _read_slot(cache, device_index: int, slot: int) -> torch.Tensor:
    """Host copy of one device's ``[1, seq_local, head_dim]`` slot."""
    return ttnn.to_torch(ttnn.get_device_tensors(cache)[device_index]).float()[slot, 0]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card-sp1"),
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        # The production SP=4 x TP=8, the only shape a Galaxy opens. Same block-cyclic walk as the
        # 4x2 arm (SP is 4 in both), now with the production number of TP replicas.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("chunk_size, max_seq_len", [(256, 1024)], ids=["c256-s1024"])
def test_device_write_matches_blockcyclic_reference(mesh_device, device_params, chunk_size, max_seq_len, reset_seeds):
    """A written chunk lands exactly where the block-cyclic host math says.

    ``sp=1`` isolates the write itself (block-cyclic degenerates to a contiguous copy); the ``4x2``
    case is the one that actually exercises the block-cyclic distribution across SP rows, which is
    where a wrong walk would put tokens on the wrong chip.
    """
    torch.manual_seed(0)
    sp_axis = 0
    sp = mesh_device.shape[sp_axis]
    num_layers, layer_idx, user_id = 2, 1, 0

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        chunk_size=chunk_size,
    )

    # One KV head per chip is the production layout, so the chunk is [1, 1, chunk, head_dim].
    k_chunk = torch.randn(1, 1, chunk_size, HEAD_DIM)
    v_chunk = torch.randn(1, 1, chunk_size, HEAD_DIM)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2  # SP-shard the sequence; replicate across TP
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims))

    def to_dev(t):
        return ttnn.from_torch(t, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

    tt_k, tt_v = to_dev(k_chunk), to_dev(v_chunk)
    write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=user_id, layer_idx=layer_idx, kv_actual=0, sp_axis=sp_axis)
    ttnn.synchronize_device(mesh_device)

    slot = slot_index(user_id, layer_idx, num_layers)
    positions = blockcyclic_positions(sp, chunk_size, max_seq_len)
    seq_local, chunk_local = max_seq_len // sp, chunk_size // sp
    rows, cols = mesh_device.shape

    k_ref = _to_bf8_roundtrip(k_chunk, mesh_device)[0, 0]
    v_ref = _to_bf8_roundtrip(v_chunk, mesh_device)[0, 0]

    pccs = []
    for chip_row in range(sp):
        device_index = chip_row * cols  # column 0 of this SP row; TP replicates
        chip_positions = positions[chip_row * seq_local : chip_row * seq_local + chunk_local]
        for cache, ref in ((kv_cache.k, k_ref), (kv_cache.v, v_ref)):
            got = _read_slot(cache, device_index, slot)[:chunk_local]
            want = ref[chip_positions]
            _, pcc = comp_pcc(want, got, 0.999)
            pccs.append(pcc)

    assert min(pccs) >= 0.999, f"written KV does not match the block-cyclic reference: min pcc {min(pccs)}"
    logger.info(f"sp={sp} block-cyclic write min PCC: {min(pccs)}")

    # Rows past the written chunk must still be zero: the cache is allocated zeroed and only
    # [0, chunk_size) was written.
    tail = _read_slot(kv_cache.k, 0, slot)[chunk_local:]
    assert torch.count_nonzero(tail) == 0, "write spilled past the chunk it was given"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("chunk_size, max_seq_len", [(256, 1024)], ids=["c256-s1024"])
def test_device_second_chunk_preserves_the_first(mesh_device, device_params, chunk_size, max_seq_len, reset_seeds):
    """Writing chunk 1 at ``kv_actual=chunk_size`` leaves chunk 0 intact and lands after it.

    The acceptance criterion is "chunk 1 reads chunk 0's KV correctly"; the failure this guards
    against is chunk 1 writing at offset 0 (ignoring ``kv_actual_global``), which a single-chunk
    test cannot see at all.
    """
    torch.manual_seed(0)
    sp_axis = 0
    num_layers, layer_idx, user_id = 1, 0, 0

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        chunk_size=chunk_size,
    )

    chunk0 = torch.randn(1, 1, chunk_size, HEAD_DIM)
    chunk1 = torch.randn(1, 1, chunk_size, HEAD_DIM)

    def to_dev(t):
        return ttnn.from_torch(t, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    for index, chunk in enumerate((chunk0, chunk1)):
        tt = to_dev(chunk)
        write_kv_chunk(
            kv_cache,
            tt,
            tt,
            slot_idx=user_id,
            layer_idx=layer_idx,
            kv_actual=index * chunk_size,
            sp_axis=sp_axis,
        )
    ttnn.synchronize_device(mesh_device)

    slot = slot_index(user_id, layer_idx, num_layers)
    got = _read_slot(kv_cache.k, 0, slot)
    ref0 = _to_bf8_roundtrip(chunk0, mesh_device)[0, 0]
    ref1 = _to_bf8_roundtrip(chunk1, mesh_device)[0, 0]

    _, pcc0 = comp_pcc(ref0, got[:chunk_size], 0.999)
    _, pcc1 = comp_pcc(ref1, got[chunk_size : 2 * chunk_size], 0.999)
    logger.info(f"chunk0 pcc={pcc0}, chunk1 pcc={pcc1}")
    assert pcc0 >= 0.999, f"chunk 1's write corrupted chunk 0: pcc {pcc0}"
    assert pcc1 >= 0.999, f"chunk 1 did not land at kv_actual={chunk_size}: pcc {pcc1}"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card")],
    indirect=["mesh_device", "device_params"],
)
def test_device_tail_pad_chunk_writes_only_the_valid_prefix(mesh_device, device_params, reset_seeds):
    """The final chunk is full-width with a padded tail; only ``[actual_start, actual_end)`` is written.

    The injected chunk is always the full ``chunk_size`` and the tail past ``actual_end`` is PAD, so
    there is no size cue other than ``actual_end``. Writing the pad would poison positions that a
    later turn legitimately resumes into.
    """
    torch.manual_seed(0)
    sp_axis, chunk_size, max_seq_len = 0, 256, 1024
    valid = 160  # bank-aligned prefix; the remaining 96 rows are pad
    assert valid % BANK_TOKENS == 0

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        chunk_size=chunk_size,
    )

    padded = torch.randn(1, 1, chunk_size, HEAD_DIM)
    valid_only = torch.zeros_like(padded)
    valid_only[:, :, :valid] = padded[:, :, :valid]

    tt = ttnn.from_torch(valid_only, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    write_kv_chunk(kv_cache, tt, tt, slot_idx=0, layer_idx=0, kv_actual=0, sp_axis=sp_axis)
    ttnn.synchronize_device(mesh_device)

    got = _read_slot(kv_cache.k, 0, slot_index(0, 0, 1))
    ref = _to_bf8_roundtrip(valid_only, mesh_device)[0, 0]
    _, pcc = comp_pcc(ref[:valid], got[:valid], 0.999)
    logger.info(f"tail-pad chunk valid-prefix pcc={pcc}")
    assert pcc >= 0.999, f"valid prefix of the tail chunk is wrong: {pcc}"
    assert torch.count_nonzero(got[chunk_size:]) == 0, "wrote past the chunk width"
