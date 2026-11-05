# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared parts of the vision encoder unit tests: for the device tests (test_vision_*_vs_ref.py) the image grids, the
host to device copies and the comparison with the reference, and for the host tests a stand-in for the mesh device
(grid_device). The reference model and its activations are fixtures in tests/conftest.py.
hf_vision_modules builds Hugging Face's vision tower and projector holding given weights.
tests/perf/test_vision_encoder_timing.py also uses VISION_GRIDS and to_host."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Mapping

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3.tt.vision import common
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from tests.ttnn.utils_for_testing import comp_relative_frobenius

# Patch grid (t, h, w) of one image: its size in pixels divided by the 14 pixel patch. The ids name the grid in
# patches, never as AxB: the Galaxy leg selects -k "8x4", which the pixel size "448x448" would contain.
VISION_GRIDS = {
    "h16w16": (1, 16, 16),  # 224x224: 256 patches, 64 vision tokens
    "h32w32": (1, 32, 32),  # 448x448: 1024 patches
    "h48w48": (1, 48, 48),  # 672x672, the largest the processor makes: 2304 patches, 576 tokens
    "h34w46": (1, 34, 46),  # 640x480 like models/sample_data/huggingface_cat_image.jpg: 1564 patches, not whole tiles
    "h40w54": (1, 40, 54),  # 4:3 photos of 800x600 and larger, resized to 756x560: 2160 patches, not whole tiles
    "h36w36": (1, 36, 36),  # 504x504: 1296 patches, not whole tiles; 1 tile of rows on a transposed split's last core
    "h18w18": (1, 18, 18),  # 252x252: 324 patches, not whole tiles, where QKV and fc1 keep ttnn's own config
    # 168x168: 144 patches, where only the patch embed, the out projection and fc2 get a program config, and SDPA
    # masks 112 of its 256 keys
    "h12w12": (1, 12, 12),
    # 56x56, the smallest the processor makes: 16 patches and 4 tokens, every linear on ttnn's own config, and SDPA
    # with one 128-key chunk, 112 of its keys masked
    "h4w4": (1, 4, 4),
}
# The CI entry "MiniMax-M3 vision unit tests (1-card)", which has a 5 minute timeout, runs h16w16 and h40w54. h40w54
# runs what h16w16 does not: the RoPE op's reload kernels, partial row tiles and, on 12x10, the transposed matmul split
# with a partial last block and program configs for the merge linears. The P150 runners in main's CI logs have 11x10
# grids, where h40w54 takes neither: there the transposed split comes at h36w36 and h48w48, and the merge linears keep
# ttnn's own config at every grid here. M3_VISION_ALL_GRIDS=1 adds the other grids. They are left out rather than
# skipped, so the leg shows no skipped vision tests.
CI_GRIDS = ("h16w16", "h40w54")
# The attention test also runs h18w18 in CI. SDPA pads the keys to a multiple of its 128-key chunk and masks the padded
# ones: 16 of 2176 at 2160 patches, too few for a broken mask to move the output past its bound, and 60 of 384 at 324.
ATTENTION_CI_GRIDS = CI_GRIDS + ("h18w18",)
# Twice the largest relative L2 error per test, measured on one P150 with a 12x10 compute grid at the grids of
# VISION_GRIDS from 256 patches up, each logged 4-place value taken at the top of its rounding (0.0059 as 0.00595).
MAX_REL_L2 = {
    "patch_embed": 0.0035,
    "layernorm": 0.0039,
    "attention": 0.0119,
    "mlp": 0.0061,
    "block": 0.0083,
    "projector": 0.0081,
    "tower": 0.0125,
    "final": 0.0149,
}


def parametrize_vision_grids(ci_grids=CI_GRIDS):
    """Parametrize grid_thw over ci_grids, or over all of VISION_GRIDS when M3_VISION_ALL_GRIDS=1. Raises ValueError
    for an id of ci_grids that VISION_GRIDS lacks, whose tests would otherwise drop out of the leg without a sign."""
    unknown = sorted(set(ci_grids) - VISION_GRIDS.keys())
    if unknown:
        raise ValueError(f"grid ids {unknown} of ci_grids are not in VISION_GRIDS")
    run_all = os.getenv("M3_VISION_ALL_GRIDS") == "1"
    params = [
        pytest.param(grid, id=grid_id) for grid_id, grid in VISION_GRIDS.items() if run_all or grid_id in ci_grids
    ]
    return pytest.mark.parametrize("grid_thw", params)


def grid_name(grid_thw) -> str:
    t, h, w = grid_thw
    return f"h{h}w{w}" if t == 1 else f"t{t}h{h}w{w}"


# A check's bound at a grid where its relative L2 error passes MAX_REL_L2. At h4w4 SDPA attends over 16 keys (112 of
# its one 128-key chunk masked), so each output row averages few bf16 products: on a P150 the attention's relL2 was
# 0.0135 there (PCC 0.999915), against 0.0048 to 0.0070 from 144 to 2304 patches.
MAX_REL_L2_AT_GRID = {("attention", "h4w4"): 0.027}


def max_rel_l2(check: str, grid_thw) -> float:
    """The relative L2 bound of check (a key of MAX_REL_L2) at grid_thw: its MAX_REL_L2_AT_GRID bound, if any."""
    return MAX_REL_L2_AT_GRID.get((check, grid_name(grid_thw)), MAX_REL_L2[check])


def rows_to_device(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """(L, D) host rows -> (1, 1, L, D) bf16 tile tensor in DRAM, replicated over the mesh (common.to_device)."""
    return common.to_device(x.reshape(1, 1, *x.shape), mesh_device)


def grid_device(x: int, y: int):
    """Stands in for a mesh device in host tests: the program configs only ask it for its compute grid, x by y cores."""
    return SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(x, y))


def hf_vision_modules(config: MiniMaxM3VisionConfig, weights: Mapping[str, torch.Tensor]) -> tuple:
    """Hugging Face's MiniMaxM3VLVisionModel and MiniMaxM3VLMultiModalProjector of config, fp32 and in eval mode,
    holding the vision_tower.* and multi_modal_projector.* weights of weights, which are under transformers 5.12's
    names (vision_checkpoint.random_vision_weights) with the patch embedding in its Conv3d layout. The tower runs eager
    attention, the plain softmax(q k^T / sqrt(head_dim)) v that the device tests' bounds were measured against.

    The modules are built on the meta device and take the weights without a copy (fp32 ones share their storage). The
    weights of the layers past config.num_hidden_layers are left out and the load is strict otherwise, so a missing
    weight, one that the modules have no place for and one of another shape raise. Needs transformers with
    minimax_m3_vl (tt-metal pins 5.12.1)."""
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as hf

    c = config
    hf_config = hf.MiniMaxM3VLConfig(
        vision_config=dict(
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_hidden_layers=c.num_hidden_layers,
            num_attention_heads=c.num_attention_heads,
            num_channels=c.num_channels,
            patch_size=c.patch_size,
            temporal_patch_size=c.temporal_patch_size,
            spatial_merge_size=c.spatial_merge_size,
            layer_norm_eps=c.layer_norm_eps,
            rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta},
        ),
        text_config=dict(hidden_size=c.text_hidden_size),
        projector_hidden_size=c.projector_hidden_size,
    )
    read_back = MiniMaxM3VisionConfig.from_hf_config(hf_config.to_dict())
    assert read_back == config, f"transformers' config reads back as {read_back}, not {config}"

    def own(prefix: str) -> dict:
        """The weights under prefix, without it and in fp32, but those of the layers past the config's."""
        out = {name[len(prefix) :]: value.float() for name, value in weights.items() if name.startswith(prefix)}
        past = [name for name in out if name.startswith("layers.") and int(name.split(".")[1]) >= c.num_hidden_layers]
        for name in past:
            del out[name]
        return out

    with torch.device("meta"):
        tower = hf.MiniMaxM3VLVisionModel(hf_config.vision_config)
        projector = hf.MiniMaxM3VLMultiModalProjector(hf_config)
    tower.set_attn_implementation("eager")
    tower.load_state_dict(own("vision_tower."), strict=True, assign=True)
    projector.load_state_dict(own("multi_modal_projector."), strict=True, assign=True)
    return tower.eval(), projector.eval()


def to_host(x: ttnn.Tensor, mesh_device, shape) -> torch.Tensor:
    """The first device's copy of x, as an fp32 host tensor of the given shape."""
    out = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[:1].reshape(shape).float()


def check_close(name: str, ref: torch.Tensor, out: torch.Tensor, min_pcc: float, max_rel_l2: float):
    """Compares out with the reference ref. Returns the line to log, "[name] <comp_allclose> <PCC> relL2=<relative L2
    error>", and whether out passes: no NaN or Inf, a PCC of at least min_pcc and a relative L2 error of at most
    max_rel_l2. PCC alone passes NaN (comp_pcc zeroes it) and any scale. The line of a failure adds the bounds, the
    relative L2 error to 6 places, the NaN and Inf counts and the least-squares gain of out over ref, which shows a
    scale error.

    With the bounds of MAX_REL_L2 the relative L2 error decides. A NaN or Inf value makes it NaN or Inf, which fails,
    and for the tests' outputs 1 - PCC is about relL2**2 / 2, so the PCC bars of 0.9999 and 0.9998 bind only at errors
    past every bound."""
    pcc_ok, pcc = comp_pcc(ref, out, min_pcc)
    rel_l2, _ = comp_relative_frobenius(ref, out)
    line = f"[{name}] {comp_allclose(ref, out)} {pcc} relL2={rel_l2:.4f}"
    num_nan, num_inf = int(out.isnan().sum()), int(out.isinf().sum())
    passed = pcc_ok and rel_l2 <= max_rel_l2 and not num_nan and not num_inf
    if not passed:
        gain = ((out * ref).sum() / (ref * ref).sum()).item()
        line += (
            f", needs PCC >= {min_pcc} and relL2 <= {max_rel_l2}: relL2 {rel_l2:.6f}, gain {gain:.4f}, {num_nan} NaN"
            f" and {num_inf} Inf in {out.numel()} values"
        )
    return line, passed


def assert_close(name: str, ref: torch.Tensor, out: torch.Tensor, min_pcc: float, max_rel_l2: float):
    """Logs check_close's line and asserts that out passes it."""
    line, passed = check_close(name, ref, out, min_pcc, max_rel_l2)
    logger.info(line)
    assert passed, line
