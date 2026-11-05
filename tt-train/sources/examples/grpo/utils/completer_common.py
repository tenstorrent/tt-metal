# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the GRPO completers (Llama and Qwen3).

Only byte-identical, completer-agnostic utilities live here. Per-completer
knobs (e.g. the ``CHUNK`` decode-readback cadence) intentionally stay in each
completer module so they can diverge independently.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import ttnn

import ttml


def apply_precision(model: Any, precision: Optional[dict]) -> None:
    """Apply the YAML's ``training_config.precision`` block (a dict; ``None`` applies nothing) to *model*'s parameters
    with ``ttml.core.apply_precision_config``. A completer calls it right after building the model and before loading
    any weights: loading keeps each parameter's stored dtype, so fp32 master weights get the loaded values unrounded."""
    if precision is not None:
        ttml.core.apply_precision_config({"precision": precision}, model.parameters())


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
