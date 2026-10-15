# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only study of serving SCA's fused-MSDA corner gather from local L1 (#60688).

The idea under study: block-shard each (camera, level) feature map over the core grid by its
normalized (u, v), and send each (camera, query, z) reference point to the core that owns its
(u, v), so the four bilinear corners it samples are in that core's L1. This script measures, on
the real SCA geometry, what that would deliver before any kernel is written:

* how often the reference point's core is the sample point's core;
* the share of in-bounds corner reads that are local, per level and halo width;
* how many cores one query's pillar z-points land on (the (q, h) reduction would span them);
* per-core load under that dispatch (max/mean), and its L1 footprint.

Offsets follow trained SCA: a per-head square grid of radius 1..P pixels of each level (the
VADv2 checkpoint's biases keep that init), plus a Gaussian spread for the query-dependent part.

Run from the repo root; no device is needed::

    python models/experimental/bevformer/tests/perf/msda_locality_sim.py --preset nuscenes_base
    python models/experimental/bevformer/tests/perf/msda_locality_sim.py --band 0.3333,0.6667
"""

import argparse
import math

import torch

from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    generate_reference_points,
    point_sampling_3d_to_2d,
)
from models.experimental.bevformer.tests.camera_rig import lidar2img_for_dataset


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="nuscenes_base")
    ap.add_argument("--bev", type=int, default=200, help="BEV grid side")
    ap.add_argument("--grid", default="11x10", help="core grid, XxY")
    ap.add_argument("--sigma", type=float, default=0.6, help="std of the query-dependent offset part, px")
    ap.add_argument("--band", default="0.0,1.0", help="v-range sharded to L1; reads outside it stay in DRAM")
    return ap.parse_args()


def _tile_bounds(core, extent, g):
    """[lo, hi) pixel range owned by `core` when `extent` pixels are split over `g` cores."""
    lo = (core * extent + g - 1) // g
    hi = ((core + 1) * extent + g - 1) // g
    return lo, hi


def main():
    args = _parse_args()
    cfg = get_preset_config(args.preset)
    ds, mc = cfg.dataset_config, cfg.model_config
    img_w, img_h = ds.input_size
    shapes = ds.spatial_shapes[: mc.num_levels]  # [W_l, H_l]
    H, L, P, Z = mc.num_heads, mc.num_levels, mc.num_points, mc.num_points_in_pillar
    D = mc.embed_dims // H
    GX, GY = map(int, args.grid.split("x"))
    NC = GX * GY
    v0, v1 = map(float, args.band.split(","))
    torch.manual_seed(0)

    # Reference points, compacted per camera the way SCA rebatches them.
    ref3d = generate_reference_points(
        args.bev, args.bev, {"num_points": Z, "start": ds.pc_range[2], "end": ds.pc_range[5]}
    )
    ref_cam, mask = point_sampling_3d_to_2d(
        ref3d, ds.pc_range, lidar2img_for_dataset(ds).unsqueeze(0), img_shape=(img_h, img_w)
    )
    ref_cam, mask = ref_cam[:, 0], mask[:, 0]  # [C, Q, Z, 2], [C, Q, Z]
    C = ref_cam.shape[0]
    valid_cq = mask.any(-1)  # SCA samples every z of a query the camera sees
    cam_idx, q_idx = valid_cq.nonzero(as_tuple=True)
    refs = ref_cam[cam_idx, q_idx].clamp(-10, 10)  # [N, Z, 2]
    N = refs.shape[0]
    print(f"{args.preset}: image {img_w}x{img_h}, levels {shapes}, H={H} D={D} L={L} P={P} Z={Z}, grid {GX}x{GY}")
    print(f"valid (camera, query) pairs {N}, per camera {valid_cq.sum(-1).tolist()}")
    print(f"z-points inside an image: {mask.float().mean().item() * 100:.1f}% of all (camera, query, z)")

    # Offsets in pixels of each level.
    theta = torch.arange(H) * (2 * math.pi / H)
    grid = torch.stack([theta.cos(), theta.sin()], -1)
    grid = grid / grid.abs().max(-1, keepdim=True).values  # [H, 2]
    radius = torch.arange(P) + 1.0
    off = (grid[:, None, None, :] * radius[None, None, :, None]).expand(H, L, P, 2)
    off = off[None] + args.sigma * torch.randn(N, H, L, P, 2)  # [N, H, L, P, 2]
    z_of_p = torch.arange(P) % Z  # pillar mode: point p samples around z-point p % Z

    wl = torch.tensor([s[0] for s in shapes], dtype=torch.float32)
    hl = torch.tensor([s[1] for s in shapes], dtype=torch.float32)
    ref_p = refs[:, None, None, :, :][:, :, :, z_of_p].expand(N, H, L, P, 2)
    px = ref_p[..., 0] * wl[None, None, :, None] - 0.5 + off[..., 0]
    py = ref_p[..., 1] * hl[None, None, :, None] - 0.5 + off[..., 1]
    x0, y0 = px.floor().long(), py.floor().long()

    # Dispatch core from the reference point. u, v are normalized, so it is the same for every level.
    def owner_norm(u, g):
        return (u * g).floor().long().clamp(0, g - 1)

    ref_core_x = owner_norm(refs[..., 0], GX)  # [N, Z]
    ref_core_y = owner_norm((refs[..., 1] - v0) / (v1 - v0), GY)
    in_band_ref = (refs[..., 1] >= v0) & (refs[..., 1] < v1)

    Wl = wl.long()[None, None, :, None]
    Hl = hl.long()[None, None, :, None]
    cx_s = ref_core_x[:, None, None, z_of_p].expand(N, H, L, P)
    cy_s = ref_core_y[:, None, None, z_of_p].expand(N, H, L, P)
    band_s = in_band_ref[:, None, None, z_of_p].expand(N, H, L, P)

    band_y0 = (v0 * hl).floor().long()[None, None, :, None]
    band_y1 = (v1 * hl).ceil().long()[None, None, :, None]
    band_h = band_y1 - band_y0

    xlo, xhi = _tile_bounds(cx_s, Wl, GX)
    ylo, yhi = _tile_bounds(cy_s, band_h, GY)
    ylo, yhi = ylo + band_y0, yhi + band_y0

    corners = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
    inb = [(cx >= 0) & (cx < Wl) & (cy >= 0) & (cy < Hl) for cx, cy in corners]
    n_reads = sum(m.sum() for m in inb).item()
    print(f"in-bounds corner reads per SCA call: {n_reads / 1e6:.2f} M ({n_reads * D * 2 / 1e6:.0f} MB at {D * 2} B)")

    print("\nLocal share of in-bounds corner reads, by halo width (px) and level")
    print(
        "  halo  "
        + "  ".join(f"L{l}({shapes[l][0]}x{shapes[l][1]})" for l in range(L))
        + "    all  [right/bottom halo only]"
    )
    for halo in [0, 1, 2, 3, 5, 8]:
        row, tot_hit, tot, rb_hit = [], 0, 0, 0
        for l in range(L):
            hit = cnt = rbh = 0
            for (cx, cy), m in zip(corners, inb):
                mm = m[:, :, l] & band_s[:, :, l]
                lx, hx, ly, hy = xlo[:, :, l], xhi[:, :, l], ylo[:, :, l], yhi[:, :, l]
                cxl, cyl = cx[:, :, l], cy[:, :, l]
                local = (cxl >= lx - halo) & (cxl < hx + halo) & (cyl >= ly - halo) & (cyl < hy + halo)
                right_bottom = (cxl >= lx) & (cxl < hx + halo) & (cyl >= ly) & (cyl < hy + halo)
                hit += (local & mm).sum().item()
                rbh += (right_bottom & mm).sum().item()
                cnt += mm.sum().item()
            row.append(hit / max(cnt, 1))
            tot_hit += hit
            rb_hit += rbh
            tot += cnt
        print(
            f"  {halo:4d}  "
            + "  ".join(f"{r * 100:10.1f}%" for r in row)
            + f"  {tot_hit / max(tot, 1) * 100:5.1f}%  [{rb_hit / max(tot, 1) * 100:5.1f}%]"
        )

    # The hypothesis as stated: does the sample point fall on its reference point's core?
    sx = (px.floor().long().clamp(min=0) * GX // Wl).clamp(0, GX - 1)
    sy = ((py.floor().long() - band_y0).clamp(min=0) * GY // band_h).clamp(0, GY - 1)
    agree = ((sx == cx_s) & (sy == cy_s) & band_s).sum().item() / band_s.sum().item()
    print(f"\nreference-point core == sample-point core (no halo, inside the band): {agree * 100:.1f}%")

    # Pillar split: distinct cores one (camera, query) touches over its in-image z-points.
    in_image = (refs[..., 0] >= 0) & (refs[..., 0] <= 1) & (refs[..., 1] >= 0) & (refs[..., 1] <= 1)
    core_id = ref_core_y * GX + ref_core_x
    distinct = torch.tensor([len(set(core_id[i][in_image[i]].tolist())) for i in range(N)])
    print(
        "distinct cores per (camera, query):",
        {k: f"{(distinct == k).float().mean() * 100:.1f}%" for k in range(1, Z + 1)},
    )

    # Load: in-bounds corner reads attributed to the core each reference z-point is sent to.
    load = torch.zeros(NC)
    dram = 0
    core_s = cy_s * GX + cx_s
    for m in inb:
        w = m & band_s
        load.index_add_(0, core_s[w], torch.ones(int(w.sum())))
        dram += (m & ~band_s).sum().item()
    print(
        f"\nper-core load (local reads): max/mean {(load.max() / load.mean()).item():.2f}, idle cores {(load == 0).sum().item()}"
    )
    print(f"reads outside the band (left in DRAM): {dram / n_reads * 100:.1f}%")
    print("share of reads per core row:", [f"{r / load.sum() * 100:.0f}%" for r in load.view(GY, GX).sum(1).tolist()])

    # L1 per core: every (camera, level) tile of the band plus the halo, H*D bf16 per pixel.
    stick = H * D * 2
    for halo in [0, 1, 2, 5]:
        worst = 0
        for gy in range(GY):
            for gx in range(GX):
                nbytes = 0
                for w_, h_ in shapes:
                    bh = math.ceil(v1 * h_) - math.floor(v0 * h_)
                    tw = math.ceil((gx + 1) * w_ / GX) - math.ceil(gx * w_ / GX)
                    th = math.ceil((gy + 1) * bh / GY) - math.ceil(gy * bh / GY)
                    nbytes += min(tw + 2 * halo, w_) * min(th + 2 * halo, bh) * stick * C
                worst = max(worst, nbytes)
        print(f"L1 per core, halo {halo}: worst {worst / 1024:.0f} KB")
    total = sum(w_ * h_ for w_, h_ in shapes) * stick * C
    print(
        f"whole value tensor {total / 1e6:.1f} MB; core tile L0 {shapes[0][0] / GX:.1f}x{shapes[0][1] * (v1 - v0) / GY:.1f} px, "
        f"L{L - 1} {shapes[-1][0] / GX:.2f}x{shapes[-1][1] * (v1 - v0) / GY:.2f} px"
    )


if __name__ == "__main__":
    main()
