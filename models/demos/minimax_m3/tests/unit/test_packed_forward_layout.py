# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the per-chip MoE padding counts and per-row real masks of a packed forward (tt/packed_rows.py). A
forward of F chunks puts each chunk's chunk_local rows back to back on every chip."""

import pytest
import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions, rotated_chunk_real_counts
from models.demos.common.prefill.prefill_chunk import PrefillChunk, validate_chunks
from models.demos.minimax_m3.tt.packed_rows import packed_real_row_mask, packed_real_rows, packed_routed_pad_rows

C, SP = 1024, 8
S = C // SP
EMPTY = PrefillChunk(0, 0, 0)


def brute_flags(chunks, sp, chunk_local):
    """Per chip, each row's real flag from the position the KV writer gives it (rotated_chunk_positions)."""
    out = []
    for c in range(sp):
        flags = []
        for e in chunks:
            flags += [
                not e.is_empty and e.actual_start <= p < e.actual_end
                for p in rotated_chunk_positions(e.actual_start, sp, chunk_local)[c]
            ]
        out.append(flags)
    return out


def brute_real_rows(chunks, sp, chunk_local):
    """Rows up to and including each chip's last real row, from the rows' real flags."""
    flags_per_chip = brute_flags(chunks, sp, chunk_local)
    return tuple(max((i + 1 for i, real in enumerate(flags) if real), default=0) for flags in flags_per_chip)


@pytest.mark.parametrize("sp,chunk_local", [(2, 512), (4, 256), (8, 128), (8, 640)])
def test_packed_real_rows_single_entry_matches_rotated_counts(sp, chunk_local):
    chunk = sp * chunk_local
    for start in range(0, 2 * chunk + 1, 32):
        for n in (1, 31, chunk_local, chunk // 2 + 7, chunk - 1, chunk):
            e = PrefillChunk(0, start, start + n)
            assert packed_real_rows([e], sp, chunk_local) == tuple(rotated_chunk_real_counts(start, n, sp, chunk_local))


CASES = {
    "all_full": [
        PrefillChunk(0, 0, C),
        PrefillChunk(0, C, 2 * C),
        PrefillChunk(1, 5 * C, 6 * C),
        PrefillChunk(2, 0, C),
    ],
    "partial_last": [
        PrefillChunk(0, 0, C),
        PrefillChunk(1, 0, C),
        PrefillChunk(2, 2 * C, 3 * C),
        PrefillChunk(3, 0, 300),
    ],
    "interleaved_partials": [
        PrefillChunk(0, 0, C),
        PrefillChunk(0, C, C + 700),
        PrefillChunk(1, 4 * C, 5 * C),
        PrefillChunk(2, 0, 300),
    ],
    "empty_tail": [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, 0, 600), EMPTY, EMPTY],
    "partial_then_full_then_empty": [PrefillChunk(0, 0, C), PrefillChunk(1, 0, 600), PrefillChunk(2, 0, C), EMPTY],
    "mid_slab": [
        PrefillChunk(0, 2 * C + 96, 3 * C + 96),
        PrefillChunk(0, 3 * C + 96, 3 * C + 900),
        PrefillChunk(1, C + 512, 2 * C + 512),
        EMPTY,
    ],
}


@pytest.mark.parametrize("name", CASES)
def test_packed_real_rows(name):
    chunks = CASES[name]
    validate_chunks(chunks, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)
    rows = packed_real_rows(chunks, SP, S)
    assert rows == brute_real_rows(chunks, SP, S)
    if name == "all_full":
        assert rows == (4 * S,) * SP
    if name == "partial_last":
        assert rows == tuple(3 * S + r for r in rotated_chunk_real_counts(0, 300, SP, S))
    if name == "empty_tail":
        assert rows == tuple(S + r for r in rotated_chunk_real_counts(0, 600, SP, S))  # empties are pad: no route-all
    if name == "interleaved_partials":
        # chips whose pad rows precede a later real row route them, up to their last real row (chunk 3's)
        assert rows == tuple(3 * S + r for r in rotated_chunk_real_counts(0, 300, SP, S))
    if name == "partial_then_full_then_empty":
        # the empty chunk's rows stay pad even though chunk 0's pad rows are routed
        assert rows == (3 * S,) * SP


def test_packed_real_rows_random():
    gen = torch.Generator().manual_seed(0)
    lengths = [0, 1, 33, S, C // 2, C - 1, C]

    def randint(lo, hi):  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    for _ in range(300):
        chunks = []
        for _ in range(randint(1, 4)):
            start = 32 * randint(0, 4 * C // 32 - 1)
            chunks.append(PrefillChunk(len(chunks), start, start + lengths[randint(0, len(lengths) - 1)]))
        chunks.sort(key=lambda e: e.is_empty)
        assert packed_real_rows(chunks, SP, S) == brute_real_rows(chunks, SP, S)


@pytest.mark.parametrize("sp,chunk_local", [(8, 640), (4, 640), (8, 128)])
def test_packed_routed_pad_rows(sp, chunk_local):
    """A partial chunk ahead of a full one routes every one of its pad rows; behind it, none."""
    chunk = sp * chunk_local
    for start in (0, chunk, 3 * chunk):
        for n in (32, chunk // 2 + 32, chunk - 32):
            partial, full = PrefillChunk(0, start, start + n), PrefillChunk(1, 0, chunk)
            assert packed_routed_pad_rows([full, partial], sp, chunk_local) == 0
            assert packed_routed_pad_rows([partial, full], sp, chunk_local) == chunk - n
            assert packed_routed_pad_rows([partial, EMPTY], sp, chunk_local) == 0
            assert packed_routed_pad_rows([partial, full, EMPTY], sp, chunk_local) == chunk - n


@pytest.mark.parametrize("name", CASES)
def test_packed_real_row_mask(name):
    """The mask flags exactly the real rows, so with it the MoE skips every pad row: those packed_real_rows leaves
    inside its count (a partial chunk ahead of a non-empty one, also at a rotated start) and those after it."""
    chunks = CASES[name]
    mask = packed_real_row_mask(chunks, SP, S)
    assert [list(map(bool, chip)) for chip in mask] == brute_flags(chunks, SP, S)
    rows = packed_real_rows(chunks, SP, S)
    routed = sum(rows[c] - sum(mask[c]) for c in range(SP))
    assert routed == packed_routed_pad_rows(chunks, SP, S)
    if name in ("interleaved_partials", "partial_then_full_then_empty"):
        assert routed > 0  # the cases the single count cannot mask


def test_packed_real_row_mask_rotated_partial_ahead_of_full():
    """A rotated partial chunk (mid-slab start) ahead of a full one: its pad rows sit inside the leading count on
    every chip, and the mask zeroes exactly them."""
    for start in (C + 96, 2 * C + 512, 3 * C + 32 * 7):
        for n in (32, 300, C // 2 + 32, C - 32):
            chunks = [PrefillChunk(0, start, start + n), PrefillChunk(1, 0, C)]
            mask = packed_real_row_mask(chunks, SP, S)
            counts = rotated_chunk_real_counts(start, n, SP, S)
            for chip in range(SP):
                assert mask[chip] == (1,) * counts[chip] + (0,) * (S - counts[chip]) + (1,) * S
            assert packed_routed_pad_rows(chunks, SP, S) == C - n


def test_packed_padding_masks_a_partial_chunk_ahead_of_a_full_one(monkeypatch):
    """Every chip's last row real (a partial chunk ahead of a full one) still yields the row mask: the eager forward
    must mask those pad rows as the traced one does. Every row real yields no padding at all."""
    from models.demos.deepseek_v3_d_p.tt.moe import tt_moe_gate_prefill

    monkeypatch.setattr(tt_moe_gate_prefill, "padding_config_tensor", lambda _, config, sp_axis: config)
    chunks = [PrefillChunk(0, 0, 896), PrefillChunk(1, 0, C)]
    padding = tt_moe_gate_prefill.packed_padding(None, chunks, SP, S, sp_axis=0)
    assert padding is not None and padding.row_mask == packed_real_row_mask(chunks, SP, S)
    assert padding.config == [[r, 0] for r in packed_real_rows(chunks, SP, S)] == [[2 * S, 0]] * SP
    assert tt_moe_gate_prefill.packed_padding(None, [PrefillChunk(0, 0, C)] * 2, SP, S, sp_axis=0) is None
    trailing = tt_moe_gate_prefill.packed_padding(
        None, [PrefillChunk(0, 0, C), PrefillChunk(1, 0, 896)], SP, S, sp_axis=0
    )
    assert trailing.row_mask is None and trailing.config == [
        [r, 0] for r in packed_real_rows([PrefillChunk(0, 0, C), PrefillChunk(1, 0, 896)], SP, S)
    ]


def test_packed_real_row_mask_random():
    gen = torch.Generator().manual_seed(1)
    lengths = [0, 1, 33, S, C // 2, C - 1, C]

    def randint(lo, hi):  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    for _ in range(300):
        chunks = []
        for _ in range(randint(1, 4)):
            start = 32 * randint(0, 4 * C // 32 - 1)
            chunks.append(PrefillChunk(len(chunks), start, start + lengths[randint(0, len(lengths) - 1)]))
        mask = packed_real_row_mask(chunks, SP, S)
        assert [list(map(bool, chip)) for chip in mask] == brute_flags(chunks, SP, S)


# --- Kimi-K3.1: packed forwards with a DSA indexer per layer (no indexer indices between ranks) ---


def _k31_params(forward_chunks, use_trace, num_users=4):
    from models.demos.common.prefill.adapter import PrefillRunParams

    return PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=24,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=False,
        max_seq_len=8 * 5120,
        chunk_size=5120,
        num_users=num_users,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
        tp_shard_kv=True,
        use_trace=use_trace,
        forward_chunks=forward_chunks,
    )


def test_k31_adapter_packs_on_pp4():
    """K3.1 packs with K3's capabilities. Its PP4 split passes no indexer indices (every DSA layer owns its indexer),
    so the runner's multi-rank check (indexer_indices = num_ranks > 1 and passes_indexer_indices) refuses nothing."""
    from models.demos.common.prefill.prefill_chunk import packed_forward_capabilities, packed_forward_refusals
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter

    assert KimiK31Adapter.supports_packed_forward
    assert not KimiK31Adapter.passes_indexer_indices
    caps = packed_forward_capabilities(KimiK31Adapter)
    assert caps == packed_forward_capabilities(KimiK3Adapter)
    assert packed_forward_refusals(2, indexer_indices=KimiK31Adapter.passes_indexer_indices, **caps) == []
    # Trace and D2H acks stay refused at the runner, as for K3, until a runner-level run has driven them.
    assert packed_forward_refusals(2, use_trace=True, **caps) == packed_forward_refusals(
        2, use_trace=True, **packed_forward_capabilities(KimiK3Adapter)
    )


@pytest.mark.parametrize("forward_chunks,use_trace,extra", [(1, True, 0), (2, False, 0), (2, True, 1)])
def test_k31_cache_slots_hold_the_traced_scratch_slot(forward_chunks, use_trace, extra):
    """Every K3.1 KV cache, the DSA index cache included (allocate_kv_cache sizes both from kv_cache_slots), holds the
    traced packed forward's scratch slot, which an empty chunk position writes."""
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter

    params = _k31_params(forward_chunks, use_trace)
    assert KimiK31Adapter().kv_cache_slots(params) == params.num_users + extra


class _GuardsPassed(Exception):
    pass


def _k31_stub(forward_chunks=2, has_indexer=True, num_mla_layers=1):
    from types import SimpleNamespace

    class Stub(SimpleNamespace):
        @property
        def is_first_rank(self):  # first read after the argument checks
            raise _GuardsPassed

    return Stub(
        forward_chunks=forward_chunks,
        has_indexer=has_indexer,
        schedule=SimpleNamespace(num_mla_layers=num_mla_layers),
        first_layer_idx=0,
    )


def test_k31_packed_forward_takes_the_index_cache():
    """TtKimiK3Transformer.forward_packed: K3.1 needs its index cache, K3 refuses one, and neither takes seed indices
    (no layer reuses another's top-k)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer

    chunks = [PrefillChunk(0, 0, C), EMPTY]
    fwd = TtKimiK3Transformer.forward_packed
    with pytest.raises(_GuardsPassed):
        fwd(_k31_stub(), None, None, chunks, index_kv_cache=object())
    with pytest.raises(ValueError, match="pass index_kv_cache"):
        fwd(_k31_stub(), None, None, chunks)
    with pytest.raises(_GuardsPassed):  # a rank with no DSA layer has no index cache
        fwd(_k31_stub(num_mla_layers=0), None, None, chunks)
    with pytest.raises(ValueError, match="seed_indexer_indices"):
        fwd(_k31_stub(), None, None, chunks, index_kv_cache=object(), seed_indexer_indices=object())
    with pytest.raises(ValueError, match="no DSA indexer"):
        fwd(_k31_stub(has_indexer=False), None, None, chunks, index_kv_cache=object())
    with pytest.raises(_GuardsPassed):
        fwd(_k31_stub(has_indexer=False), None, None, chunks)
