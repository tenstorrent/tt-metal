# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is `ttnn.concat` along a *tiled* axis correct when the first operand is not tile-aligned?

The streaming bisect named `hift_mel`, and the one thing that cache does is this:

    joined = ttnn.concat([state.hift_mel, mel], dim=1)     # [1, 20, 80] ++ [1, 110, 80]

`dim=1` is the second-to-last axis, so under `TILE_LAYOUT` it is **tiled**: rows live in
32-row tiles. A 20-row first operand means the second operand's row 0 has to land at row 20
of the result -- 12 rows into a tile that the first operand only partly fills. That is a
re-tiling shuffle, not a memcpy, and it is exactly the kind of thing that can be right on
one architecture and wrong on another.

`probe_streaming_amplitude.py` cannot see this. It records the mel RMS **inside**
`flow_chunk`, before any streaming machinery runs, so the concat's output was never
measured -- only its input and, 40 convolutions later, its consequence.

Two controls make the answer unambiguous:

  - **row-index values.** Every element of row `i` is `i`, so a wrong result names the rows
    it actually returned instead of merely failing a tolerance. Values stay under 256 so
    bfloat16 represents each row index exactly (above 256 the spacing is 2 and a "mismatch"
    would be rounding, not corruption -- a trap this probe's predecessor fell into).
  - **the sweep covers aligned and unaligned splits alike.** If 32/64/96 pass and 20 fails,
    the axis alignment is the variable. If everything passes, concat is exonerated and the
    failure is downstream -- see `probe_hift_isolate.py` for that half.

    python3 models/demos/cosyvoice/scripts/probe_concat_rows.py
"""
from __future__ import annotations

import torch

import ttnn

# (first operand rows, second operand rows). 20 ++ 110 is the shipped streaming case;
# 20 ++ 152 is chunk 0's would-be successor; the rest bracket it with aligned and
# unaligned splits so alignment can be read off rather than assumed.
CASES = [
    (20, 110),  # <- the streaming case
    (20, 152),
    (20, 76),
    (1, 110),
    (12, 110),
    (31, 110),
    (32, 110),  # aligned
    (33, 110),
    (34, 76),  # the mel_overlap fade's split, which the bisect exonerated
    (64, 110),  # aligned
    (96, 150),  # aligned. Total stays under 256 -- see the bfloat16 note above; an
    # earlier version ran 96 ++ 186 and reported row 257 as corrupt, when 257 simply
    # is not representable and 256 is the correct rounding.
]
WIDTHS = (80, 1)  # 80 = mel channels; 1 = the waveform/source axis the fades use


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  ttnn.concat(dim=1) on TILE_LAYOUT, values = global row index\n")
        print(f"  {'width':>7}{'a rows':>8}{'b rows':>8}{'a%32':>7}{'max|err|':>11}{'  first bad row'}")
        print("  " + "-" * 62)
        bad = 0
        for w in WIDTHS:
            for n, m in CASES:
                total = n + m
                ref = torch.arange(total, dtype=torch.float32).reshape(1, total, 1).expand(1, total, w).contiguous()
                a = ttnn.from_torch(
                    ref[:, :n, :].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                b = ttnn.from_torch(
                    ref[:, n:, :].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                got = ttnn.to_torch(ttnn.concat([a, b], dim=1)).float()
                err = (got - ref).abs().amax(dim=2)[0]
                worst = float(err.max())
                first = int(err.argmax()) if worst > 0.5 else -1
                note = "" if worst <= 0.5 else f"   row {first}: got {float(got[0, first, 0]):.0f} want {first}"
                if worst > 0.5:
                    bad += 1
                print(f"  {w:>7}{n:>8}{m:>8}{n % 32:>7}{worst:>11.3f}{note}")
                ttnn.deallocate(a)
                ttnn.deallocate(b)
        print(f"\n  {bad} of {len(CASES) * len(WIDTHS)} cases wrong.")
        print("  A failure confined to rows where a%32 != 0 makes this a tiled-axis concat defect.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
