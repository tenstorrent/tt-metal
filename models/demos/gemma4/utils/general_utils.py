# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the Gemma4 demo.
"""

from __future__ import annotations

import glob
import os


def get_cache_file_name(tensor_cache_path, name):
    """Build a tensor-cache path prefix for ``ttnn.as_tensor``.

    CI mounts ``/mnt/MLPerf/huggingface/tt_cache`` read-only. Existing
    ``.tensorbin`` files are still readable; newly introduced names (e.g.
    DRAM-sharded ``*.ws``) must not attempt to create files on that mount —
    return ``None`` so the weight is tilized in-memory instead.
    """
    if not tensor_cache_path:
        return None
    path = f"{tensor_cache_path}/{name}"
    # as_tensor appends ``_dtype_*_layout_*.tensorbin``; any match means a hit.
    if glob.glob(path + "*"):
        return path
    directory = os.path.dirname(path) or "."
    probe = directory
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if probe and os.path.isdir(probe) and not os.access(probe, os.W_OK):
        return None
    return path


def cast_host_for_ttnn(torch_tensor, ttnn_dtype):
    """Cast a host torch tensor to the torch dtype matching ``ttnn_dtype``.

    ``ttnn.as_tensor``/``from_torch`` run a C++ ``to_dtype`` conversion whenever the
    source dtype differs from the requested one. That conversion queries tile metadata
    on a ROW_MAJOR host intermediate and emits the #18536 "extract tile information out
    of a ROW MAJOR layout" warning. Matching the host dtype up front lets ``to_dtype``
    short-circuit, so no warning is emitted.

    Only float targets representable in torch are handled; block formats (bfloat8_b /
    bfloat4_b) have no torch equivalent and are returned unchanged.
    """
    import ttnn

    try:
        import torch
    except ImportError:
        return torch_tensor

    mapping = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
    target = mapping.get(ttnn_dtype)
    if target is not None and torch_tensor.dtype != target:
        return torch_tensor.to(target)
    return torch_tensor
