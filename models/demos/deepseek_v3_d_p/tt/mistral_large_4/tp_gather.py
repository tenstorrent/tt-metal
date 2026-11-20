# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The ffn_norm output gathered over TP once per block (FFN_GATHER=shared).

The router, the routed experts and the shared expert each all_gather the same [1, 1, S/8, H/4] ffn_norm output to
[1, 1, S/8, H] over TP. With FFN_GATHER=shared the first of them gathers it and the others reuse it; the gathered
tensor lives until the next block's gather replaces it (a reader never deallocates it). FFN_GATHER=each keeps one
gather per module.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

_last = [None, None]  # [the tensor gathered, its gather]


def shared() -> bool:
    return settings.get("FFN_GATHER") == "shared"


def gather(x: ttnn.Tensor, topology, hidden: int | None = None) -> ttnn.Tensor:
    """x gathered over TP along its last dim; with FFN_GATHER=shared the same tensor for every call on the same x.
    x already ``hidden`` wide (NORM_GATHER=input: the norm hands on the full row) is returned as is."""
    if hidden is not None and x.shape[-1] == hidden:
        return x
    if not shared():
        return ttnn.all_gather(x, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=topology)
    if _last[0] is x and _last[1] is not None and _last[1].is_allocated():
        return _last[1]
    old = _last[1]
    _last[0] = x
    _last[1] = ttnn.all_gather(x, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=topology)
    if old is not None and old.is_allocated():
        ttnn.deallocate(old)
    return _last[1]


def done(x_full: ttnn.Tensor, x: ttnn.Tensor | None = None) -> None:
    """A reader is done with its gather: deallocated unless it is the shared one or the reader's input x itself."""
    if x_full is not _last[1] and x_full is not x:
        ttnn.deallocate(x_full)
