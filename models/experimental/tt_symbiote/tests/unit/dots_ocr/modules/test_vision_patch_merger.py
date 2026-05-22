# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: :class:`TTNNDotsPatchMerger`.

Per Phase 0 finding §11.6 the patch merger does not appear in the
captured module matrix. Inspection (``modules/dots_ocr_vision.py:1375``)
shows it IS a ``TTNNModule`` subclass with ``from_torch`` — but its
``forward`` requires the post-trunk-norm tensor as input, and the HF
``PatchMerger`` reference also requires the same logical input.

We mark this test as ``pytest.skip`` per the implementer guidance: the
patch merger is exercised transitively by ``e2e/test_vision_tower.py``
and its standalone weight layout (``col-sharded w2``) requires
production-matched sharding setup that isn't reproducible with
``replicate``-mapped inputs alone. Phase 4 should revisit with a proper
sharded-mem-config helper.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "TTNNDotsPatchMerger has no isolatable from_torch path that survives the"
        " replicated-input + replicate-gather pattern used by Phase 3 module tests:"
        " its move_weights_to_device_impl col-shards w2 across the TP mesh axis,"
        " and its forward returns a tensor with the same TP-sharded layout. The"
        " test would need a production-matched ShardTensor2dMesh input wrapper"
        " (Phase 4 follow-up). Covered transitively by e2e/test_vision_tower.py."
    )
)
def test_vision_patch_merger():
    """See module docstring."""
