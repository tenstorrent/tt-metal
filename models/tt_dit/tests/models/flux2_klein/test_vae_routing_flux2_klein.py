# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host checks of the vae_flux2_klein.py decoder routing for the tt-media-server API canvases (PR #5466):
(1) ``Flux2KleinVaeDecode.padded_height`` / ``partitioned``: a canvas whose decoder levels are all tile-aligned per
device (``h_partition_tile_aligned``; the fused distributed group norm TT_FATALs otherwise) decodes on the H8 decoder
as it is; every other API canvas decodes on it bottom-padded to ``h_partition_padded_height`` (the smallest 128-px
multiple with every level aligned; ``Flux2VaeDecoder.forward(logical_h=...)`` keeps the padding rows out of the image),
and only ``pad_rows = False`` (``FLUX2_KLEIN_VAE_PAD_ROWS=0``) sends them to the replicated fallback; (2) the
replicated fallback decoder registers ``klein_replicated_decoder_slices`` (auto = 0 by default) over the fixed 1024
counts, and nothing else changes (H8 canvases, encoder registrations, shipped ``slice_default`` keys). Run from
tt-metal, env.sh sourced:  python -m pytest models/tt_dit/tests/models/flux2_klein/test_vae_routing_flux2_klein.py -q
"""

from __future__ import annotations

import pytest

from models.tt_dit.layers.conv2d import Conv2d
from models.tt_dit.pipelines.flux2_klein import vae_flux2_klein as V

API = {  # (width, height) -> (decoder without row padding, decode height with it)
    (1024, 1024): ("H8", 1024),
    (2048, 2048): ("H8", 2048),
    (1376, 768): ("rep", 1024),
    (2752, 1536): ("H8", 1536),
    (768, 1376): ("rep", 1408),
    (1536, 2752): ("rep", 2816),
    (1184, 896): ("rep", 1024),
    (2368, 1792): ("H8", 1792),
    (896, 1184): ("rep", 1280),
    (1792, 2368): ("rep", 2432),
    (1248, 832): ("rep", 1024),
    (2496, 1664): ("rep", 1792),
    (832, 1248): ("rep", 1280),
    (1664, 2496): ("rep", 2560),
    (1344, 768): ("H8", 768),
    (896, 1120): ("rep", 1152),  # the tree's SUBGRID_CANVASES (width, height)
}


def decode_helper(h_factor: int = 8, *, pad_rows: bool = True) -> V.Flux2KleinVaeDecode:
    helper = object.__new__(V.Flux2KleinVaeDecode)
    helper._h_factor = h_factor
    helper.pad_rows = pad_rows
    return helper


def aligned_per_device(height: int, width: int, h_factor: int = 8) -> bool:
    """Every decoder level (/8, /4, /2, x1) holds whole 32-row tiles per device, and whole token rows split evenly."""
    lh, lw = height // 8, width // 8
    if (height // 16) % h_factor:
        return False
    for _ in range(4):
        if lh % h_factor or (lh // h_factor) * lw % 32:
            return False
        lh, lw = 2 * lh, 2 * lw
    return True


@pytest.mark.parametrize("wh", list(API), ids=[f"{w}x{h}" for w, h in API])
def test_partition_route(wh) -> None:
    w, h = wh
    unpadded, decode_height = API[wh]
    assert ("H8" if decode_helper(pad_rows=False).partitioned(h, w) else "rep") == unpadded
    helper = decode_helper()
    assert helper.partitioned(h, w) and helper.padded_height(h, w) == decode_height
    assert helper.route(h, w) == ("H8" if decode_height == h else "H8+pad")
    assert aligned_per_device(decode_height, w)
    # the smallest such height: no 128-px step below it is aligned (128 = 16 px per token row x 8 devices)
    assert decode_height == h or not any(aligned_per_device(c, w) for c in range(h, decode_height, 128))
    # the attention's keys / values are the image's own latent rows: a whole number of tiles
    assert (h // 8) * (w // 8) % 32 == 0


def test_padded_height_none_without_whole_key_tiles() -> None:
    # 256(w) x 240(h): latent level 30 x 32 = 30 tiles, h 15 % 8 != 0 -> padded to 256; 272 x 272 -> 34 x 34 = 1156
    # latent-level rows, not a whole number of tiles: no padding helps
    assert V.h_partition_padded_height(240, 256, 8) == 256
    assert V.h_partition_padded_height(272, 272, 8) is None
    assert decode_helper().padded_height(272, 272) is None and not decode_helper().partitioned(272, 272)


def test_single_device_unchanged() -> None:
    assert decode_helper(1).partitioned(768, 1376) and decode_helper(1).partitioned(1376, 768)
    assert decode_helper(1).padded_height(768, 1376) == 768


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
