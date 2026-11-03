# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared parts of the vision encoder's unit tests.

Image grids, error bounds, copies to and from the chip, the comparison with the reference (check_close), Hugging
Face's modules (hf_vision_modules) and a stand-in device for host tests. Also the weight matmul's checks and circular
buffers, for the tests of its program configs and of the L1 budget.
"""

from __future__ import annotations

import math
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
# The grids that the CI entry "MiniMax-M3 vision unit tests (1-card)" runs. h40w54 covers partial row tiles and the RoPE
# op's reload kernels, which h16w16 does not. M3_VISION_ALL_GRIDS=1 adds the other grids, left out rather than skipped.
CI_GRIDS = ("h16w16", "h40w54")
# The attention test also runs h18w18 in CI. SDPA pads the keys to a multiple of its 128-key chunk and masks the padded
# ones: 16 of 2176 at 2160 patches, too few for a broken mask to move the output past its bound, and 60 of 384 at 324.
ATTENTION_CI_GRIDS = CI_GRIDS + ("h18w18",)
# The tower test of activations in L1 against DRAM runs h16w16 and h36w36 in CI: both keep the activations in L1 on
# 11x10, 12x10 and 13x10. h36w36 adds a partial tile row, one tile row on the last core of every block linear's split,
# and on 12x10 and 13x10 splits that leave cores out.
L1_VS_DRAM_CI_GRIDS = ("h16w16", "h36w36")
# The grids of common.L1_GRIDS by test id, without "AxB": the 1-card leg selects "1x1", which "11x10" would contain.
L1_GRID_IDS = {f"x{x}y{y}": (x, y) for x, y in common.L1_GRIDS}
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
    """Parametrize grid_thw over ci_grids, or over all of VISION_GRIDS when M3_VISION_ALL_GRIDS=1.

    Raises ValueError for a ci_grids id that VISION_GRIDS lacks, which would silently drop its tests.
    """
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


# Looser bounds for a check at one grid, where its error exceeds MAX_REL_L2. At h4w4 SDPA attends over only 16 keys,
# so each output row averages few bf16 products and the attention's error is larger.
MAX_REL_L2_AT_GRID = {("attention", "h4w4"): 0.027}


def max_rel_l2(check: str, grid_thw) -> float:
    """The relative L2 bound of check (a key of MAX_REL_L2) at grid_thw: its MAX_REL_L2_AT_GRID bound, if any."""
    return MAX_REL_L2_AT_GRID.get((check, grid_name(grid_thw)), MAX_REL_L2[check])


def rows_to_device(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Upload (L, D) host rows as a (1, 1, L, D) bf16 tile tensor in DRAM (common.to_device)."""
    return common.to_device(x.reshape(1, 1, *x.shape), mesh_device)


def grid_device(x: int, y: int):
    """A stand-in for a mesh device in host tests: it only has a compute grid, of x by y cores."""
    return SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(x, y))


def hf_vision_modules(config: MiniMaxM3VisionConfig, weights: Mapping[str, torch.Tensor]) -> tuple:
    """Hugging Face's vision tower and projector of config, in fp32 and eval mode, holding weights.

    weights are under transformers 5.12's names, as random_vision_weights gives them, with the patch embedding in its
    Conv3d layout. fp32 weights are shared, not copied. Layers past config.num_hidden_layers are left out. Otherwise
    the load is strict: a missing or unexpected weight, or one of the wrong shape, raises. The tower runs eager
    attention, which the device tests' bounds were measured against. Needs transformers with minimax_m3_vl.
    """
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
        """The weights under prefix, renamed without it, in fp32, minus the layers past the config's."""
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
    """Compare out with the reference ref, and return the line to log and whether out passes.

    out passes with no NaN or Inf, a PCC of at least min_pcc and a relative L2 error of at most max_rel_l2. PCC alone
    would pass NaN and any scale. The line is "[name] <comp_allclose> <PCC> relL2=<error>". A failure's line adds the
    bounds, the NaN and Inf counts, and the gain of out over ref, which shows a scale error.
    """
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


def matmul_cb_bytes(config, k_tiles: int, fp32_dest_acc: bool, bias: bool = True) -> int:
    """The circular buffers per core of the 2D multicast matmul factory for a linear_program_config config, bf16 tiles
    of 2 KB, with the input and output interleaved (matmul_multicore_reuse_mcast_2d_program_factory.cpp): the in0 and
    in1 K blocks, double-buffered when K takes more than one, the output block, a bias row, and the output block's
    partial sums, which with fp32 accumulation are fp32 tiles in a buffer of their own."""
    depth = 2 if k_tiles > config.in0_block_w else 1
    block = config.out_block_h * config.out_block_w
    tiles = depth * config.in0_block_w * (config.out_block_h + config.out_block_w) + block
    if bias:
        tiles += config.out_block_w
    return 2048 * tiles + (4096 * block if fp32_dest_acc else 0)


def assert_matmul_config_valid(config, rows, in_features, out_features, grid, fp32_dest_acc):
    """The checks the matmul device op and its 2D multicast factory make on this config, for the input and output
    interleaved in DRAM or L1, which those checks do not tell apart (matmul_device_operation.cpp,
    matmul_multicore_reuse_mcast_2d_program_factory.cpp), and that its circular buffers fit in LINEAR_L1_BYTES."""
    m_tiles, k_tiles, n_tiles = (math.ceil(d / ttnn.TILE_SIZE) for d in (rows, in_features, out_features))
    assert (config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y) == grid
    assert k_tiles % config.in0_block_w == 0
    assert config.out_block_h % config.out_subblock_h == 0 and config.out_block_w % config.out_subblock_w == 0
    assert config.per_core_M % config.out_block_h == 0 and config.per_core_N % config.out_block_w == 0
    assert config.out_subblock_h * config.out_subblock_w <= (4 if fp32_dest_acc else 8)  # half the dest register
    assert config.per_core_M <= m_tiles
    blocks_x, blocks_y = math.ceil(n_tiles / config.per_core_N), math.ceil(m_tiles / config.per_core_M)
    if config.transpose_mcast:
        blocks_x, blocks_y = blocks_y, blocks_x
    assert blocks_x <= grid[0] and blocks_y <= grid[1]
    assert blocks_x >= 2 and blocks_y >= 2  # a 2D split, as ttnn's own 2D configs for DRAM operands
    assert not config.fuse_batch and config.fused_activation is None
    assert matmul_cb_bytes(config, k_tiles, fp32_dest_acc) <= common.LINEAR_L1_BYTES
