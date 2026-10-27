# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host checks of the vae_flux2_klein.py decoder routing for the tt-media-server API canvases (PR #5466):
(1) ``Flux2KleinVaeDecode.partitioned`` sends a canvas to the H8 decoder only when every decoder level is tile-aligned
per device (``h_partition_tile_aligned``; the fused distributed group norm TT_FATALs otherwise); (2) the replicated
fallback decoder registers ``klein_replicated_decoder_slices`` (auto = 0 by default) over the fixed 1024 counts, and
nothing else changes (H8 canvases, encoder registrations, shipped ``slice_default`` keys). Run from tt-metal, env.sh
sourced:  python -m pytest models/tt_dit/tests/models/flux2_klein/test_vae_routing_flux2_klein.py -q
"""

from __future__ import annotations

import pytest

from models.tt_dit.layers.conv2d import Conv2d
from models.tt_dit.pipelines.flux2_klein import vae_flux2_klein as V

API = {  # (width, height) -> decoder after the fix
    (1024, 1024): "H8",
    (2048, 2048): "H8",
    (1376, 768): "rep",
    (2752, 1536): "H8",
    (768, 1376): "rep",
    (1536, 2752): "rep",
    (1184, 896): "rep",
    (2368, 1792): "H8",
    (896, 1184): "rep",
    (1792, 2368): "rep",
    (1248, 832): "rep",
    (2496, 1664): "rep",
    (832, 1248): "rep",
    (1664, 2496): "rep",
    (1344, 768): "H8",
    (896, 1120): "rep",  # the tree's SUBGRID_CANVASES (width, height): unchanged
}


def decode_helper(h_factor: int = 8) -> V.Flux2KleinVaeDecode:
    helper = object.__new__(V.Flux2KleinVaeDecode)
    helper._h_factor = h_factor
    return helper


@pytest.mark.parametrize("wh", list(API), ids=[f"{w}x{h}" for w, h in API])
def test_partition_route(wh) -> None:
    w, h = wh
    got = "H8" if decode_helper().partitioned(h, w) else "rep"
    assert got == API[wh]
    if got == "H8":  # every level per-device tile-aligned
        lh, lw = h // 8, w // 8
        for _ in range(4):
            assert (lh // 8) * lw % 32 == 0
            lh, lw = 2 * lh, 2 * lw


def test_single_device_unchanged() -> None:
    assert decode_helper(1).partitioned(768, 1376) and decode_helper(1).partitioned(1376, 768)


def test_replicated_registration_scope() -> None:
    mesh = (99, 98)  # a private table
    Conv2d.slice_params.pop(mesh, None)
    V.register_klein_conv_slices(mesh, 1024, 1024)  # encoder-style registration
    V.register_klein_conv_slices(mesh, 2048, 2048, h_factor=8, tp_factor=4)  # H8 decode
    before = dict(Conv2d.slice_params[mesh])
    V.register_klein_conv_slices(mesh, 832, 1248, tp_factor=4, replicated_fallback=True)
    table = Conv2d.slice_params[mesh]
    rep = V.klein_replicated_decoder_slices(832, 1248, tp_factor=4)
    assert rep and all(table[k] == 0 for k in rep)
    changed = {k for k in before if before[k] != table[k]}
    assert not changed, changed  # nothing registered before is touched (different keys)
    for k, v in Conv2d.slice_default.items():
        assert table[k] == v
    Conv2d.slice_params.pop(mesh, None)
