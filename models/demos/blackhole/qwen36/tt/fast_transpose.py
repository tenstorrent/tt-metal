# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Fast host transpose for weight preprocessing on a tensor-cache miss.

``w.T.contiguous()`` of a large bf16 checkpoint weight runs at ~0.5 GB/s in torch (single-threaded strided copy) and
was ~60% of a cold Qwen3.x weight-cache build. ``fast_t`` returns the same tensor using ttnn's blocked multithreaded
transpose; it falls back to torch for anything that is not a large contiguous 2-D bf16/fp32 CPU tensor.
"""

import torch

import ttnn

_MIN_ELEMENTS = 1 << 16


def fast_t(t: torch.Tensor) -> torch.Tensor:
    """Equivalent to ``t.T.contiguous()``."""
    transpose = getattr(getattr(ttnn._ttnn, "fast_bfp", None), "transpose", None)
    if (
        transpose is not None
        and t.dim() == 2
        and t.device.type == "cpu"
        and t.dtype in (torch.bfloat16, torch.float32)
        and t.is_contiguous()
        and t.numel() >= _MIN_ELEMENTS
    ):
        return torch.from_dlpack(transpose(t))
    return t.T.contiguous()
