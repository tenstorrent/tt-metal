"""Round-trip checks for engine/weight_layouts.py (host only; the quantized check needs ttnn, no device).

python engine/test_weight_layouts.py            -> lossless fp round trip + BFP8 / BFP4 quantized round trip
Bank map: the BH P150 DRAM bank -> preferred logical worker map (dev.get_optimal_dram_bank_to_logical_worker_assignment
on NOC0) is passed in by the caller in the engine; here a fixed 8-bank map is enough (placement is data-independent).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weight_layouts as WL  # noqa: E402

TILE = 32
BANK_LOC = [(0, 0), (0, 3), (0, 7), (5, 9), (6, 0), (11, 3), (11, 7), (11, 9)]   # any 8 distinct points
BANKS = 8


def _bfp_like(shape, seed):
    g = torch.Generator().manual_seed(seed)
    # tile-varying magnitudes so the BFP shared exponents differ between tiles
    w = torch.randn(*shape, generator=g)
    r, c = shape[0] // TILE, shape[1] // TILE
    s = torch.exp2(torch.randint(-6, 6, (r, 1, c, 1), generator=g).float())
    return (w.reshape(r, TILE, c, TILE) * s).reshape(shape)


def main(quantize=True):
    owners = [(100 + i, 10 + i % 2, i // 2) for i in range(16)]                    # cols 10-11, rows 0-7
    gdn = [(200 + i, i % 12, 8 + i // 12) for i in range(24)]                       # rows 8-9, row-major
    cases = [
        ("qkv", WL.QkvSplitKLayout(BANK_LOC, BANKS), (5120, 112 * TILE)),
        ("wo", WL.wo_unit_layout(owners, BANK_LOC, BANKS), (48 * TILE, 160 * TILE)),
        ("gdn_out", WL.gdn_out_layout(gdn, BANK_LOC, BANKS), (24 * TILE, 160 * TILE)),
    ]
    ok = True
    for i, (name, lay, shape) in enumerate(cases):
        w = _bfp_like(shape, 11 + i)
        fp, q = WL.roundtrip(lay, w, quantize=quantize)
        pg = lay.pack(w)
        cover = int((pg.abs().flatten(1).sum(1) > 0).sum())
        n_tiles = (shape[0] // TILE) * (shape[1] // TILE)
        print(f"[wl] {name}: dtype {lay.dtype} pages/layer {lay.pages_per_layer} filled {cover} = tiles {n_tiles} "
              f"fp-roundtrip {fp} quantized-roundtrip {q}", flush=True)
        ok &= fp and cover == n_tiles and (q is None or q)
        # multi-layer packer: layer L at L * pages_per_layer
        ws = [w, 2 * w]
        all_ = WL.pack_layers(ws, lay)
        ok &= torch.equal(lay.unpack(all_.reshape(2, lay.pages_per_layer, TILE, TILE))[1], 2 * w)
    print(f"[wl] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(quantize=os.environ.get("WL_QUANT", "1") == "1") else 1)
