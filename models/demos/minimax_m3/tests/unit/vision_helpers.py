# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared parts of the vision encoder unit tests (test_vision_*_vs_ref.py): the image grids, the host to
device copies and the comparison with the reference. The reference model and its activations are fixtures in
conftest.py."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3.tt.vision import common
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
}
# The 1-card CI leg, which has a 5 minute timeout, runs h16w16 and h40w54. h40w54 runs what h16w16 does not: the RoPE
# op's reload kernels, partial row tiles and, on 12x10, the transposed matmul split with a partial last block and
# program configs for the merge linears. M3_VISION_ALL_GRIDS=1 adds the others. They are left out rather than skipped,
# so the leg shows no skipped vision tests.
CI_GRIDS = ("h16w16", "h40w54")
# The attention test also runs h18w18 in CI. SDPA pads the keys to a multiple of its 128-key chunk and masks the padded
# ones: 16 of 2176 at 2160 patches, too few for a broken mask to move the output past its bound, and 60 of 384 at 324.
ATTENTION_CI_GRIDS = CI_GRIDS + ("h18w18",)
# Twice the largest relative L2 error measured on one P150 at h16w16, h32w32, h48w48 and h34w46, per test.
MAX_REL_L2 = {
    "patch_embed": 0.0058,
    "layernorm": 0.0136,
    "attention": 0.0128,
    "mlp": 0.0082,
    "block": 0.0122,
    "projector": 0.0108,
    "tower": 0.0188,
    "final": 0.0208,
}


def parametrize_vision_grids(ci_grids=CI_GRIDS):
    """Parametrize grid_thw over ci_grids, or over all of VISION_GRIDS when M3_VISION_ALL_GRIDS=1."""
    run_all = os.getenv("M3_VISION_ALL_GRIDS") == "1"
    params = [
        pytest.param(grid, id=grid_id) for grid_id, grid in VISION_GRIDS.items() if run_all or grid_id in ci_grids
    ]
    return pytest.mark.parametrize("grid_thw", params)


def grid_name(grid_thw) -> str:
    t, h, w = grid_thw
    return f"h{h}w{w}" if t == 1 else f"t{t}h{h}w{w}"


def to_device(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """(L, D) host tensor -> (1, 1, L, D) bf16 tile tensor in DRAM, replicated over the mesh (common.to_device)."""
    return common.to_device(x.reshape(1, 1, *x.shape), mesh_device)


def to_host(x: ttnn.Tensor, mesh_device, shape) -> torch.Tensor:
    """The first device's copy of x, as an fp32 host tensor of the given shape."""
    out = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[:1].reshape(shape).float()


def check_close(name: str, ref: torch.Tensor, out: torch.Tensor, min_pcc: float, max_rel_l2: float):
    """Compares out with the reference ref. Returns the line to log, "[name] <comp_allclose> <PCC> relL2=<relative L2
    error>", and whether out passes: no NaN or Inf, a PCC of at least min_pcc and a relative L2 error of at most
    max_rel_l2. PCC alone passes NaN (comp_pcc zeroes it) and any scale. The line of a failure adds the bounds, the
    relative L2 error to 6 places, the NaN and Inf counts and the least-squares gain of out over ref, which shows a
    scale error."""
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


def assert_pcc(name: str, ref: torch.Tensor, out: torch.Tensor, min_pcc: float, max_rel_l2: float):
    """Logs check_close's line and asserts that out passes it."""
    line, passed = check_close(name, ref, out, min_pcc, max_rel_l2)
    logger.info(line)
    assert passed, line
