# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only invariants for the SP8 affine-prefix device barrier."""

from models.experimental.kimi_delta_attention.tt.sp_affine_prefix import _SP_SIZE, _fabric_tree_edges


def test_sp8_fabric_tree_orders_global_gather_and_release() -> None:
    """All ranks arrive at rank zero, then rank zero releases all ranks.

    The device implementation queues gather levels in ascending distance and
    release levels in descending distance.  This test models those two ordered
    dependency chains, rather than merely comparing their edge lists.
    """
    gather, release = _fabric_tree_edges()
    subtrees = {rank: {rank} for rank in range(_SP_SIZE)}
    for level in gather:
        for child, parent in level:
            assert child in subtrees
            assert parent in subtrees
            # At the next level a prior receiver becomes a sender, modelling
            # the command-queue dependency that gives its parent an entire
            # completed child subtree rather than a single rank token.
            subtrees[parent].update(subtrees[child])
    # Every non-root rank is a gather sender exactly once.  Its parent may
    # itself send only after its own child edge's receiver has completed.
    assert {child for level in gather for child, _ in level} == set(range(1, _SP_SIZE))
    assert all(parent < child for level in gather for child, parent in level)
    assert subtrees[0] == set(range(_SP_SIZE))

    released = {0}
    for level in reversed(release):
        for parent, child in level:
            assert parent in released
            released.add(child)
    assert released == set(range(_SP_SIZE))
