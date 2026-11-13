# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One way to turn a weight into a device tensor through a ttnn weight cache.

``ttnn.as_tensor`` writes ``<cache_file>_dtype_<DTYPE>_layout_<LAYOUT>.tensorbin`` on a miss and loads it on a
hit, but a hit still wants a torch tensor to be passed (it ignores it), which is how a placeholder ends up
dumped as the weights when a cache is not what the caller thought it was. ``from_torch_cached`` takes the weight
OR ``None``: ``None`` means "cache-only", which loads the file and raises if there is none, so that form can
neither convert nor write anything.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn


def tensorbin_path(cache_file: str, dtype, layout) -> Path:
    """The file ``as_tensor`` reads and writes for ``cache_file`` at this dtype and layout."""
    return Path(f"{cache_file}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def from_torch_cached(
    x: torch.Tensor | None,
    *,
    dtype,
    layout,
    device,
    memory_config,
    mesh_mapper,
    cache_file: str,
):
    """``x`` through the cache at ``cache_file``: a miss converts and writes it, a hit loads it.

    ``x=None`` is the cache-only form: load, or raise ``FileNotFoundError``. ``device=None`` keeps the tensor on
    the host (the cache build, which never needs the device memory)."""
    if x is None:
        file = tensorbin_path(cache_file, dtype, layout)
        if not file.is_file():
            raise FileNotFoundError(f"weight cache miss with no torch weight to convert: {file}")
        return ttnn.load_tensor(file, device=device)
    return ttnn.as_tensor(
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config if device is not None else None,
        mesh_mapper=mesh_mapper,
        cache_file_name=cache_file,
    )
