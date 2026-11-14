# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Descriptor-cache key helper shared by the custom ops."""


def tensor_key(t):
    """What a cached program bakes in about tensor ``t``: its address (runtime args) and its shape, dtype, layout and
    memory config (CB data formats, page sizes, TensorAccessor args). A freed bf16 tensor and a new bf8 tensor can get
    the same address, so the address alone would hand back a program built for the wrong page size."""
    if t is None:
        return None
    return (t.buffer_address(), tuple(t.shape), str(t.dtype), str(t.layout), str(t.memory_config()))
