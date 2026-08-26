# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for ``ttml.ops.sample.sample_op``.

This lives in the ttml package (rather than in an example) so that every caller of the
sampler -- not just the GRPO completers -- can build a validated positions tensor.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import ttnn

import ttml


def positions_to_tensor(
    positions: Sequence[int], B: int, tokens: int, dp_mapper: Any, num_shards: Optional[int] = None
) -> ttml.autograd.Tensor:
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
    if dp_mapper is not None:
        # A 1D shard mapper that receives fewer rows than it has shard slots SHRINKS the tensor's
        # distribution shape to the actual chunk count (distributed_tensor.cpp, "If the distribution
        # shape is 1D and we have less shards than devices"). The op then rejects the tensor for not
        # matching the logits' topology -- a true but unhelpful message. Catch the real cause here:
        # an uneven batch, typically the tail of a dataset whose prompt count does not divide the
        # rollout batch size.
        #
        # The divisor is the number of batch shards THE MAPPER produces, which is a property of the
        # mapper and not of axis semantics: shard_tensor_to_mesh_mapper(device, dim) without a
        # cluster_axis -- the only batch mapper ttml builds today -- flattens the WHOLE mesh and
        # shards across every device, so the default is the full device count of the AutoContext
        # mesh (the only mesh a ttml mapper can target, and where this tensor is about to land).
        # A mapper that shards across a subset of the mesh (e.g. a cluster_axis batch mapper
        # that replicates across tp) produces fewer shards; the mapper object is opaque from Python,
        # so such callers must pass their shard count via ``num_shards``.
        if num_shards is None:
            num_shards = ttml.autograd.AutoContext.get_instance().get_device().get_num_devices()
        assert B % num_shards == 0, (
            f"batch of {B} rows does not divide across {num_shards} batch shards; a sharded "
            f"positions tensor requires a divisible batch (pad or drop the tail batch before "
            f"sampling)"
        )
    return ttml.autograd.Tensor.from_numpy(
        np.asarray(positions, dtype=np.uint32).reshape(B, 1, 1, 1),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
        dp_mapper,
    )
