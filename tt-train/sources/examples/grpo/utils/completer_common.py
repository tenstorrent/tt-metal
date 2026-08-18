# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the GRPO completers (Llama and Qwen3).

Only byte-identical, completer-agnostic utilities live here. Per-completer
knobs (e.g. the ``CHUNK`` decode-readback cadence) intentionally stay in each
completer module so they can diverge independently.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import ttnn

import ttml


def positions_to_tensor(positions: Sequence[int], B: int, tokens: int, dp_mapper: Any) -> ttml.autograd.Tensor:
    """Per-row sample positions for ``sample_op`` as [B, 1, 1, 1] UINT32.

    ``dp_mapper`` must be the SAME mapper the batch was sharded with (``None`` when the batch is
    replicated). That is what makes the shard landing on each device BE that device's rows -- true by
    construction, rather than by two separately-written mapper configs happening to agree.

    The op cannot range-check these: they live in device memory, and reading them back would mean a
    blocking sync on the dispatch path. So the loud check asserts live here, once, against the global list
    while the values are still on the host.
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


def deallocate_tensors(tensors: Any) -> None:
    if tensors is None:
        return
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, ttml.autograd.Tensor):
            ttnn.deallocate(t.get_value(), force=True)
        elif isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t, force=True)


def async_read_to_host(tensors: List[Any], mesh_device: Any) -> Tuple[List[Any], Any]:
    """Issue non-blocking d2h reads for ``tensors`` on the single command queue.

    Returns ``(host_tensors, event)``. The caller must call
    ``event_synchronize(event)`` before consuming ``host_tensors``; deallocating
    the source ``tensors`` before then races with the in-flight DMA.
    """
    hosts = [t.cpu(blocking=False) for t in tensors]
    done = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
    return hosts, done
