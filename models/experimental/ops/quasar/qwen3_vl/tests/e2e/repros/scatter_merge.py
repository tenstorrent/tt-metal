# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: the vision-token merge as the model does it on device (tt/common.py merge_vision_tokens_single_user_ttnn):
ttnn.scatter of image rows into the text embeddings along dim 0, and of a deepstack tensor into zeros. Scatter is a
copy, so the result must be bit-exact.

usage: scatter_merge.py [tiny|demo] [tile|rm]   tiny: 78 text rows, 64 image rows (the e2e preset); demo: 2766 / 2752.
tile (default) is the model's layout; rm uploads the embeddings row-major."""
import sys

import torch
import ttnn

SIZES = {"tiny": (78, 64, 5), "demo": (2766, 2752, 7)}  # text rows, image rows, first image row
H = 2560


def main():
    s, n, first = SIZES[sys.argv[1] if len(sys.argv) > 1 else "tiny"]
    layout = ttnn.ROW_MAJOR_LAYOUT if sys.argv[2:] == ["rm"] else ttnn.TILE_LAYOUT
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    text = torch.randn(s, H).bfloat16()
    image = torch.randn(n, H).bfloat16()
    deep = torch.randn(n, H).bfloat16()
    rows = torch.arange(first, first + n)
    index = rows.view(n, 1).expand(n, H).contiguous()
    up = lambda t: ttnn.from_torch(t, device=dev, dtype=ttnn.bfloat16, layout=layout)
    t_text, t_image, t_deep = up(text), up(image), up(deep)
    t_index = ttnn.from_torch(index, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    zeros = ttnn.zeros_like(t_text)

    merged = ttnn.to_torch(ttnn.scatter(t_text, 0, t_index, t_image))
    ref = text.clone()
    ref[rows] = image
    bad = int((merged != ref).sum())
    print(f"text<-image   [{s},{H}] <- [{n},{H}] exact={bad == 0} mismatches={bad}/{ref.numel()}", flush=True)
    if bad:
        row_bad = (merged != ref).sum(1)
        kept = [r for r in range(s) if r < first or r >= first + n]
        print(
            f"  untouched rows bad: {int((row_bad[kept] > 0).sum())}/{len(kept)}; image rows bad: "
            f"{int((row_bad[first:first + n] > 0).sum())}/{n}; out absmax={merged.float().abs().max().item():.3g} "
            f"zeros={int((merged == 0).sum())}; row {first} got {merged[first, :4].tolist()} want {ref[first, :4].tolist()}",
            flush=True,
        )

    deep_out = ttnn.to_torch(ttnn.scatter(zeros, 0, t_index, t_deep))
    ref_d = torch.zeros_like(text)
    ref_d[rows] = deep
    bad_d = int((deep_out != ref_d).sum())
    print(f"zeros<-deep   [{s},{H}] <- [{n},{H}] exact={bad_d == 0} mismatches={bad_d}/{ref_d.numel()}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
