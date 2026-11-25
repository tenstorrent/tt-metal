# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""tt/pf_tiles.py: the expert tensors rebuilt on the card from a PF-layout cache are the TILE cache's, byte for
byte. Needs the card and both caches of the run mode (skips without them)."""

import numpy as np
import pytest

import ttnn
from models.demos.inkling2_mini import modes
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.convert import stick
from models.demos.inkling2_mini.tt import pf_tiles
from models.demos.inkling2_mini.tt.device import open_device
from models.demos.inkling2_mini.tt.fused.moe_bank import pf_memory_config

PF, TILE = modes.roots()


def _tiles(t, path):
    ttnn.dump_tensor(str(path), t.cpu())
    dtype = "BFP4" if t.dtype == ttnn.bfloat4_b else "BFP8"
    return dtype, np.array(stick._read(path, dtype)[1])


@pytest.mark.skipif(
    not ((PF / "manifest.json").is_file() and (TILE / "manifest.json").is_file()), reason="needs the PF and TILE caches"
)
@pytest.mark.parametrize("layer", [0, 12])  # a BFP4 and a BFP8 gate_up in the default map
def test_rebuilt_experts_are_the_tile_cache(layer, tmp_path):
    assert wcache.layouts(PF, layer).get("experts.w_gate_up") == "PF", "the decode cache is not PF-layout"
    w = wcache.load_layer(PF, layer, None)
    dev = open_device()
    try:
        for name, fn in (("experts.w_gate_up", pf_tiles.gate_up), ("experts.w_down", pf_tiles.down)):
            main = ttnn.to_device(w[name], dev, pf_memory_config(w[name].shape[1], w[name].shape[2]))
            extra = ttnn.to_device(w[name + ".x"], dev, ttnn.DRAM_MEMORY_CONFIG)
            out = fn(main, extra)
            dtype, got = _tiles(out, tmp_path / "t.tensorbin")
            ref = stick._read(TILE / "layers" / f"L{layer:02d}" / f"{name}.tensorbin", dtype)[1]
            assert got.shape == ref.shape and np.array_equal(got, ref), f"L{layer} {name}"
            for t in (out, main, extra):
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)
