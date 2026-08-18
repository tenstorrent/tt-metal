# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for ``ttml.ops.sample.sample_op``.

This lives in the ttml package (rather than in an example) so that every caller of the
sampler -- not just the GRPO completers -- can build a validated positions tensor.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import ttnn

import ttml


def positions_to_tensor(positions: Sequence[int], B: int, tokens: int, dp_mapper: Any) -> ttml.autograd.Tensor:
    """Per-row sample positions for ``sample_op`` as [B, 1, 1, 1] UINT32.

    ``dp_mapper`` must be the SAME mapper the batch was sharded with (``None`` when the batch is
    replicated). That is what makes the shard landing on each device BE that device's rows -- true by
    construction, rather than by two separately-written mapper configs happening to agree.

    Validate here, while the values are still on the host: once they land in device memory the op
    cannot range-check them on the dispatch path (reading them back would be a blocking sync). The
    device kernels clamp out-of-range positions to the last real token (and assert under watcher),
    so a bad value can no longer silently sample a padding row -- but the clamp is a containment
    measure, not a diagnosis. This assert is the loud, early check that names the offending rows.
    """
    positions = [int(p) for p in positions]
    assert len(positions) == B, f"expected {B} positions, got {len(positions)}"
    bad = [(b, p) for b, p in enumerate(positions) if not 0 <= p < tokens]
    assert not bad, f"positions outside [0, {tokens}): {bad[:8]}"
    return ttml.autograd.Tensor.from_numpy(
        np.asarray(positions, dtype=np.uint32).reshape(B, 1, 1, 1),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
        dp_mapper,
    )
