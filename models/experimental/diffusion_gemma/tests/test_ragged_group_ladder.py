# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ragged group ladder must collapse the geometry without moving a single real row.

Why the ladder exists: each group goes to ``ttnn.sparse_matmul`` as
[1, group_size, m_blocks*TILE, H] with nnz=group_size, and ``group_size`` is the number of
expert-segments that happen to have that m_blocks -- routing-dependent, so a new prompt is a new
program built ON THE HOST. Measured on QB2 2026-07-31 with DG_PREFILL_CPU_PROBE: 4.0-7.98 s per
prefill at cache_len 128 and 5.0-17.4 s at 2048, thread_cpu_frac 0.947-0.982, with py-spy landing
in ``ragged_sparse_prefill_forward``'s sparse_matmul.

Host-only. ``_ragged_metadata_host`` consumes a ttnn tensor, so these build the same unpadded
group layout it produces and exercise ``_quantize_ragged_groups`` directly. What must hold:
  * every (token, k) still resolves to a row holding that token, marked valid, on the same expert;
  * padded rows are inert (slot_token 0, slot_valid 0);
  * sparsity keeps exactly one non-zero per group row -- nnz != count_nonzero(sparsity) HANGS the
    kernel on device (matmul_nanobind.cpp:1053), and nnz is passed as the padded group_size;
  * group sizes land on the ladder and the shape space collapses across different routings;
  * packed_rows equals the padded total, since it sizes the concat the gather indexes into.
"""

from __future__ import annotations

import torch

from models.experimental.diffusion_gemma.tt.sparse_moe import (
    RAGGED_MAX_M_BLOCKS,
    TILE,
    _GROUP_LADDER,
    _ladder_group_size,
    _quantize_ragged_groups,
)

E = 128


def _build_groups(layout, seed=0):
    """Build an unpadded (groups, token_slot, packed_rows) triple.

    ``layout`` is [(m_blocks, [rows_used_per_segment, ...]), ...] in ascending m_blocks, matching
    the real builder: segments are expert-homogeneous, padded to a tile multiple inside the
    segment, and groups are concatenated in m_blocks order.
    """
    generator = torch.Generator().manual_seed(seed)
    groups, assignments = [], {}
    offset, token = 0, 0
    for m_blocks, segment_rows in layout:
        group_size = len(segment_rows)
        rows_per_segment = m_blocks * TILE
        total_rows = group_size * rows_per_segment
        slot_token = torch.zeros(total_rows, dtype=torch.int32)
        slot_valid = torch.zeros((total_rows, 1), dtype=torch.bfloat16)
        experts = torch.randperm(E, generator=generator)[:group_size]
        sparsity = torch.zeros((1, 1, group_size, E), dtype=torch.bfloat16)
        sparsity[0, 0, torch.arange(group_size), experts] = 1
        for segment, used in enumerate(segment_rows):
            assert used <= rows_per_segment
            for row in range(used):
                absolute = offset + segment * rows_per_segment + row
                slot_token[segment * rows_per_segment + row] = token
                slot_valid[segment * rows_per_segment + row] = 1
                assignments[token] = (absolute, int(experts[segment]))
                token += 1
        groups.append((m_blocks, group_size, slot_token, slot_valid, sparsity))
        offset += total_rows
    token_slot = torch.zeros((token, 1), dtype=torch.int32)
    for tok, (absolute, _) in assignments.items():
        token_slot[tok, 0] = absolute
    return groups, token_slot, offset, assignments


def _resolve(groups, token_slot, num_tokens):
    """Reconstruct {token -> expert} through the packed layout, asserting row identity."""
    slot_token = torch.cat([g[2].reshape(-1) for g in groups])
    slot_valid = torch.cat([g[3].reshape(-1) for g in groups])
    expert_of_row = torch.cat([g[4][0, 0].argmax(dim=-1).repeat_interleave(g[0] * TILE) for g in groups])
    out = {}
    for tok in range(num_tokens):
        row = int(token_slot[tok, 0])
        assert int(slot_token[row]) == tok, f"row {row} holds {int(slot_token[row])}, not token {tok}"
        assert float(slot_valid[row]) == 1.0, f"row {row} for token {tok} is not valid"
        out[tok] = int(expert_of_row[row])
    return out


_LAYOUTS = [
    [(1, [32, 17, 3])],
    [(1, [32] * 5), (2, [64, 40])],
    [(1, [1]), (2, [33, 64, 12]), (3, [96] * 9), (4, [128, 5])],
]


def test_ladder_rounds_up_and_is_monotone():
    for n in range(1, 600):
        step = _ladder_group_size(n)
        assert step >= n
        assert step in _GROUP_LADDER or step > _GROUP_LADDER[-1]
    assert _ladder_group_size(1) == 1
    assert _ladder_group_size(5) == 8
    assert _ladder_group_size(9) == 12
    assert _ladder_group_size(33) == 48
    # beyond the table it keeps doubling rather than raising
    assert _ladder_group_size(700) == 1024


def test_padding_preserves_every_real_assignment():
    for seed, layout in enumerate(_LAYOUTS):
        groups, token_slot, packed_rows, assignments = _build_groups(layout, seed=seed)
        num_tokens = len(assignments)
        before = _resolve(groups, token_slot, num_tokens)
        padded, padded_slot, padded_rows = _quantize_ragged_groups(groups, token_slot, packed_rows)
        assert _resolve(padded, padded_slot, num_tokens) == before
        assert padded_rows >= packed_rows


def test_quantizing_twice_is_a_fixed_point():
    groups, token_slot, packed_rows, assignments = _build_groups(_LAYOUTS[-1], seed=3)
    once = _quantize_ragged_groups(groups, token_slot, packed_rows)
    twice = _quantize_ragged_groups(*once)
    assert twice[2] == once[2]
    assert _resolve(twice[0], twice[1], len(assignments)) == _resolve(once[0], once[1], len(assignments))


def test_padded_groups_are_on_ladder_inert_and_one_hot():
    for seed, layout in enumerate(_LAYOUTS):
        groups, token_slot, packed_rows, _ = _build_groups(layout, seed=seed)
        padded, _, padded_rows = _quantize_ragged_groups(groups, token_slot, packed_rows)
        total = 0
        for m_blocks, group_size, slot_token, slot_valid, sparsity in padded:
            assert 1 <= m_blocks <= RAGGED_MAX_M_BLOCKS
            assert _ladder_group_size(group_size) == group_size, f"off-ladder group_size {group_size}"
            rows = group_size * m_blocks * TILE
            assert slot_token.numel() == rows
            assert slot_valid.numel() == rows
            assert tuple(sparsity.shape) == (1, 1, group_size, E)
            # nnz is passed as group_size; a mismatch deadlocks the kernel on device.
            assert int((sparsity != 0).sum()) == group_size
            assert torch.equal((sparsity != 0).sum(dim=-1)[0, 0], torch.ones(group_size, dtype=torch.int64))
            invalid = slot_valid.reshape(-1) == 0
            assert torch.all(slot_token.reshape(-1)[invalid] == 0), "padded rows must gather row 0"
            total += rows
        assert padded_rows == total


def test_geometry_collapses_across_routings():
    """The point of the change: different routings must reuse one small set of shapes."""
    raw_shapes, padded_shapes = set(), set()
    for seed in range(12):
        generator = torch.Generator().manual_seed(seed)
        layout = [
            (
                m_blocks,
                [
                    int(x)
                    for x in torch.randint(
                        1, m_blocks * TILE, (int(torch.randint(1, 20, (1,), generator=generator)),), generator=generator
                    )
                ],
            )
            for m_blocks in range(1, RAGGED_MAX_M_BLOCKS + 1)
        ]
        groups, token_slot, packed_rows, _ = _build_groups(layout, seed=seed)
        raw_shapes.update((g[0], g[1]) for g in groups)
        padded, _, _ = _quantize_ragged_groups(groups, token_slot, packed_rows)
        padded_shapes.update((g[0], g[1]) for g in padded)
    assert len(padded_shapes) < len(raw_shapes), f"no collapse: {len(raw_shapes)} -> {len(padded_shapes)}"
    assert len(padded_shapes) <= RAGGED_MAX_M_BLOCKS * len(_GROUP_LADDER)
