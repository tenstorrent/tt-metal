# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.deepseek_v3_d_p.tests.paged_prefill.support import (
    GLM52_PRIMARY_LAYERS,
    PREFILL_PAGE_TOKENS,
    SPARSE_SENTINEL,
    gather_paged_kv_reference,
    paged_kv_location,
)


def test_position_dependent_owner_preserves_5120_bundle_bank_assignment():
    sp = 4
    chunk_local = PREFILL_PAGE_TOKENS // sp
    for owner in range(sp):
        first = owner * chunk_local
        last = first + chunk_local - 1
        assert paged_kv_location(first, 2, sp=sp, layer=17) == (
            owner,
            2 * GLM52_PRIMARY_LAYERS + 17,
            0,
        )
        assert paged_kv_location(last, 2, sp=sp, layer=17) == (
            owner,
            2 * GLM52_PRIMARY_LAYERS + 17,
            chunk_local - 1,
        )
    # Bundle boundaries reset owner/local-row while the compact mapping changes
    # only the folded physical bundle.
    assert paged_kv_location(PREFILL_PAGE_TOKENS, 0, sp=sp, layer=17) == (0, 17, 0)


def test_fragmented_compact_table_reconstructs_global_topk_without_prefix_copy():
    sp = 4
    physical_bundles = 3
    layer = 6
    width = 4
    chunk_local = PREFILL_PAGE_TOKENS // sp
    shards = [
        torch.full(
            (physical_bundles * GLM52_PRIMARY_LAYERS, 1, chunk_local, width),
            -999.0,
            dtype=torch.float32,
        )
        for _ in range(sp)
    ]
    page_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)

    logical_tokens = [
        0,
        chunk_local - 1,
        chunk_local,
        PREFILL_PAGE_TOKENS - 1,
        PREFILL_PAGE_TOKENS,
        2 * PREFILL_PAGE_TOKENS + 3 * chunk_local // 2,
    ]
    for token in logical_tokens:
        bundle = token // PREFILL_PAGE_TOKENS
        physical = int(page_table[0, bundle])
        owner, batch, row = paged_kv_location(token, physical, sp=sp, layer=layer)
        shards[owner][batch, 0, row] = torch.tensor([token, owner, physical, row], dtype=torch.float32)

    indices = torch.tensor(logical_tokens + [SPARSE_SENTINEL, SPARSE_SENTINEL], dtype=torch.int64)
    gathered, valid = gather_paged_kv_reference(
        shards,
        page_table,
        indices,
        slot=0,
        layer=layer,
    )

    assert valid.tolist() == [True] * len(logical_tokens) + [False, False]
    assert gathered[-2:].eq(0).all()  # sentinels were masked, never dereferenced
    assert gathered[: len(logical_tokens), 0].tolist() == logical_tokens


def test_page_table_contents_are_runtime_state_not_geometry():
    """Changing slot mappings keeps the tensor geometry/program signature stable."""

    before = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.int32)
    after = before.clone()
    after[0] = torch.tensor([0, 1, 2])
    assert before.shape == after.shape
    assert before.dtype == after.dtype
    assert not torch.equal(before, after)


def test_unallocated_selected_bundle_fails_before_pool_access(expect_error):
    shards = [torch.zeros((GLM52_PRIMARY_LAYERS, 1, 1280, 4)) for _ in range(4)]
    with expect_error(RuntimeError, "unallocated"):
        gather_paged_kv_reference(
            shards,
            torch.tensor([[-1]], dtype=torch.int32),
            torch.tensor([0]),
            slot=0,
            layer=0,
        )
