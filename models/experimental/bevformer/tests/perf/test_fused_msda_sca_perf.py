# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for fused_msda_from_offsets on the spatial-cross-attention call BEVFormer makes.

``test_fused_msda_perf.py`` draws reference points uniformly, which spreads the gather over the
whole feature map. SCA reference points come from projecting BEV pillars into the camera rig,
so they cluster around the horizon, and SCA pads every camera to the longest one. This builds
that call: nuscenes_base, 200x200 BEV, the nuScenes rig, the rebatch compaction with its
padding, and offsets shaped like trained ones (the init grid of radius 1..P pixels per head,
which the trained VADv2 SCA biases keep, plus a small query-dependent spread).

Variants:

* ``refs``: ``rig`` projects the BEV pillars; ``uniform`` keeps the shapes but draws the
  reference points uniformly, to separate the cost of their distribution.
* ``value_mem``: ``value`` in DRAM or in L1 interleaved.
* ``all_rows`` / ``valid_rows``: without and with ``num_valid_queries``, i.e. with and without
  the rebatch padding rows.

Read the signposted region only. Used as the baseline for #60688 (gathering corners from local
L1).
"""

import math

import pytest
import torch
from tracy import signpost

import ttnn
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    generate_reference_points,
    point_sampling_3d_to_2d,
)
from models.experimental.bevformer.tests.camera_rig import lidar2img_for_dataset

BEV = 200
OFFSET_SPREAD_PX = 0.6


def _rig_reference_points(ds, num_depth):
    """Per-camera reference points padded the way build_rebatch_plan pads them: each camera's
    visible queries first, then the last query's point up to rebatch_len."""
    width, height = ds.input_size
    ref3d = generate_reference_points(
        BEV, BEV, {"num_points": num_depth, "start": ds.pc_range[2], "end": ds.pc_range[5]}
    )
    ref_cam, mask = point_sampling_3d_to_2d(
        ref3d, ds.pc_range, lidar2img_for_dataset(ds).unsqueeze(0), img_shape=(height, width)
    )
    ref_cam, mask = ref_cam[:, 0].clamp(-10, 10), mask[:, 0]
    valid = mask.any(-1)
    max_len = int(valid.sum(-1).max())
    rebatch_len = (max_len // 32 + 1) * 32
    num_cams, num_queries = valid.shape
    ids = torch.full((num_cams, rebatch_len), num_queries - 1, dtype=torch.long)
    for c in range(num_cams):
        v = valid[c].nonzero().squeeze(-1)
        ids[c, : v.numel()] = v
    return torch.stack([ref_cam[c, ids[c]] for c in range(num_cams)]), rebatch_len, valid.sum(-1).tolist()


def _trained_like_offsets(num_cams, rebatch_len, heads, levels, points):
    theta = torch.arange(heads) * (2 * math.pi / heads)
    grid = torch.stack([theta.cos(), theta.sin()], -1)
    grid = grid / grid.abs().max(-1, keepdim=True).values
    radius = torch.arange(points) + 1.0
    bias = (grid[:, None, None, :] * radius[None, None, :, None]).expand(heads, levels, points, 2)
    offsets = bias + OFFSET_SPREAD_PX * torch.randn(num_cams, rebatch_len, heads, levels, points, 2)
    return offsets.reshape(num_cams, rebatch_len, heads, levels * points * 2)


@pytest.mark.parametrize("skip_padding", [False, True], ids=["all_rows", "valid_rows"])
@pytest.mark.parametrize("value_mem", ["l1", "dram"])
@pytest.mark.parametrize("refs", ["rig", "uniform"])
def test_fused_msda_sca_perf(device, refs, value_mem, skip_padding):
    torch.manual_seed(0)
    cfg = get_preset_config("nuscenes_base")
    ds, mc = cfg.dataset_config, cfg.model_config
    heads, levels, points, depth = mc.num_heads, mc.num_levels, mc.num_points, mc.num_points_in_pillar
    # Dataset shapes are [W, H]; the op takes (H, W).
    spatial_shapes = [(h, w) for w, h in ds.spatial_shapes[:levels]]
    num_keys = sum(h * w for h, w in spatial_shapes)

    ref, rebatch_len, num_valid_queries = _rig_reference_points(ds, depth)
    num_cams = ref.shape[0]
    if refs == "uniform":
        ref = torch.rand_like(ref)

    offsets = _trained_like_offsets(num_cams, rebatch_len, heads, levels, points)
    attn = torch.softmax(torch.randn(num_cams, rebatch_len, heads, levels * points), -1)
    value = torch.randn(num_cams, num_keys, mc.embed_dims)

    def to_dev(t):
        return ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_value, tt_ref, tt_off, tt_attn = map(to_dev, (value, ref, offsets, attn))
    if value_mem == "l1":
        tt_value = ttnn.to_memory_config(tt_value, ttnn.L1_MEMORY_CONFIG)

    def run(skip=skip_padding):
        return ttnn.experimental.fused_msda_from_offsets(
            tt_value,
            tt_ref,
            tt_off,
            tt_attn,
            spatial_shapes,
            reference_mode="pillar",
            num_valid_queries=num_valid_queries if skip else None,
        )

    run()
    ttnn.synchronize_device(device)
    signpost(header=f"sca_{refs}_{value_mem}_{skip_padding}_start")
    out = run()
    ttnn.synchronize_device(device)
    signpost(header=f"sca_{refs}_{value_mem}_{skip_padding}_end")
    assert tuple(out.shape) == (num_cams, rebatch_len, mc.embed_dims)
    if skip_padding:
        # The output inherits value's L1 placement; two of them next to value do not fit.
        skipped = ttnn.to_torch(out)
        ttnn.deallocate(out)
        full = ttnn.to_torch(run(skip=False))
        for c, n in enumerate(num_valid_queries):
            assert torch.equal(skipped[c, :n], full[c, :n]), f"camera {c}: valid rows differ when padding is skipped"
