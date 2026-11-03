# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only checks of the vision encoder's weight-matmul program configs (tt/vision/common.py, linear_program_config)
and of linear(), which runs every weight matmul with them. No device: the configs are built with the real ttnn classes
and checked on the host.

- At 672x672 (2304 patches) on a 12x10 grid every linear gets the config pinned in CONFIGS_672.
- At the smaller sizes of CONFIGS_MID on 12x10, a linear whose transposed split would have fewer than
  LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core gets the untransposed split or keeps ttnn's own config, and one
  with exactly that many gets the transposed split.
- At 256 patches (h16w16, a CI grid), on 11x10, 12x10 and 13x10, the tower's linears get a config and the merge
  linears keep ttnn's own, so the 1-card device tests run both paths; linear_1 and linear_2 get a config on 13x10 only.
  At 16 patches, the smallest image, every linear keeps ttnn's own.
- The helper takes its three limits as arguments: with other limits, linear_2 at 1024 patches gets the transposed split
  and fc1 at 2304 patches other K blocks.
- For every patch count the image processor gives by default (multiples of 4 from 16 to 2304), on the 11x10, 12x10
  and 13x10 core grids that Blackhole chips have depending on harvesting, every config passes the checks the matmul
  device op and its 2D multicast factory make, the factory's circular buffers fit in LINEAR_L1_BYTES, and no config
  uses transpose_mcast with fewer than LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core.
- linear() gives ttnn.linear the helper's config for the tiles of its operands' padded shapes and runs the activation,
  the accurate GELU in the MLPs, as an op of its own on the matmul's output.
- M3VLMLP.fuse_gelu, off as built, runs the accurate GELU as that op; on, it puts the GELU into fc1's config where
  fc1 gets one (256, 2160 and 2304 patches on each grid) and runs it as that op where fc1 keeps ttnn's own (324).
  linear() takes the fused form only with the activation.

The 1-card leg selects these tests by name (test_vision_), as they have no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_linear_program_config.py
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision import mlp
from models.demos.minimax_m3.tt.vision.common import (
    LINEAR_L1_BYTES,
    LINEAR_MIN_TRANSPOSED_PER_CORE_M,
    _linear_blocking,
    linear,
    linear_compute_config,
    linear_program_config,
)
from models.demos.minimax_m3.tt.vision.mlp import accurate_gelu

from .vision_helpers import grid_device

CONFIG = MiniMaxM3VisionConfig()
HEADS_SIZE = CONFIG.num_attention_heads * nearest_32(CONFIG.head_dim)  # q, k and v each, with the heads padded (1536)
MERGE = CONFIG.spatial_merge_size**2
# The encoder's weight matmuls: (in_features, out_features, patches per row).
LINEARS = {
    "patch_embed": (CONFIG.patch_dim, CONFIG.hidden_size, 1),
    "qkv": (CONFIG.hidden_size, 3 * HEADS_SIZE, 1),
    "out_proj": (HEADS_SIZE, CONFIG.hidden_size, 1),
    "fc1": (CONFIG.hidden_size, CONFIG.intermediate_size, 1),
    "fc2": (CONFIG.intermediate_size, CONFIG.hidden_size, 1),
    "linear_1": (CONFIG.hidden_size, CONFIG.projector_hidden_size, 1),
    "linear_2": (CONFIG.projector_hidden_size, CONFIG.text_hidden_size, 1),
    "merge_linear_1": (CONFIG.merged_hidden_size, CONFIG.projector_hidden_size, MERGE),
    "merge_linear_2": (CONFIG.projector_hidden_size, CONFIG.text_hidden_size, MERGE),
}
FIELDS = ("transpose_mcast", "per_core_M", "per_core_N", "in0_block_w", "out_subblock_h", "out_subblock_w")
# At 2304 patches on a 12x10 grid.
CONFIGS_672 = {
    "patch_embed": (True, 6, 4, 1, 2, 2),
    "qkv": (True, 6, 15, 4, 3, 1),
    "out_proj": (True, 6, 4, 8, 2, 2),
    "fc1": (True, 6, 16, 4, 2, 2),
    "fc2": (True, 6, 4, 8, 2, 2),
    "linear_1": (True, 6, 20, 2, 2, 2),
    "linear_2": (True, 6, 20, 2, 2, 2),  # K blocks of 4 ran faster in a matmul sweep but overflow LINEAR_L1_BYTES
    "merge_linear_1": (False, 2, 16, 8, 2, 2),
    "merge_linear_2": (False, 2, 16, 8, 2, 2),
}
# (linear, patches): the config on a 12x10 grid, None for ttnn's own. The comments give the image and the per-core
# block (M x N tiles) of the transposed split.
CONFIGS_MID = {
    ("qkv", 324): None,  # 252x252; 1x15
    ("fc2", 324): (False, 2, 4, 8, 2, 2),  # 1x4
    ("linear_2", 768): None,  # 336x448; 2x20
    ("linear_2", 1024): None,  # 448x448; 3x20
    ("merge_linear_2", 1296): None,  # 504x504; 1x20
    ("fc1", 1296): (True, 4, 16, 5, 2, 2),  # 4x16
}
# Ids without "AxB": the 1-card leg selects "1x1", which "11x10" would contain.
GRIDS = {"x11y10": (11, 10), "x12y10": (12, 10), "x13y10": (13, 10)}


def _config(name, patches, grid=(12, 10), compute_kernel_config=None):
    """linear_program_config for one of LINEARS in the forward of an image of `patches` patches."""
    in_features, out_features, patches_per_row = LINEARS[name]
    if compute_kernel_config is None:
        compute_kernel_config = linear_compute_config()
    return linear_program_config(
        grid_device(*grid), patches // patches_per_row, in_features, out_features, compute_kernel_config
    )


def _assert_valid(config, rows, in_features, out_features, grid, fp32_dest_acc):
    """The checks the matmul device op and its 2D multicast factory make on this config, for operands and output in
    DRAM (matmul_device_operation.cpp, matmul_multicore_reuse_mcast_2d_program_factory.cpp)."""
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
    # The factory's circular buffers, bf16 tiles of 2 KB: the in0 and in1 K blocks (double-buffered when K takes
    # more than one), the output block, a bias row, and the output block's partial sums, which with fp32
    # accumulation are fp32 tiles in a buffer of their own.
    depth = 2 if k_tiles > config.in0_block_w else 1
    block = config.out_block_h * config.out_block_w
    cb_bytes = 2048 * (depth * config.in0_block_w * (config.out_block_h + config.out_block_w) + block)
    cb_bytes += 2048 * config.out_block_w + (4096 * block if fp32_dest_acc else 0)
    assert cb_bytes <= LINEAR_L1_BYTES


def test_vision_linear_program_config_pinned():
    """The configs at 672x672 and at the sizes of CONFIGS_MID on 12x10; which linears keep ttnn's own config at 256
    patches on each grid and at 16 patches; the helper's choice with other limits."""
    for name, want in CONFIGS_672.items():
        config = _config(name, 2304)
        assert config is not None, name
        assert tuple(getattr(config, f) for f in FIELDS) == want, f"{name}: {config}"
        assert (config.out_block_h, config.out_block_w) == (config.per_core_M, config.per_core_N), name
        in_features, out_features, patches_per_row = LINEARS[name]
        _assert_valid(config, 2304 // patches_per_row, in_features, out_features, (12, 10), fp32_dest_acc=True)
    for (name, patches), want in CONFIGS_MID.items():
        config = _config(name, patches)
        got = None if config is None else tuple(getattr(config, f) for f in FIELDS)
        assert got == want, f"{name} at {patches} patches: {config}"
    tower = {"patch_embed", "qkv", "out_proj", "fc1", "fc2"}
    # On 13x10 linear_1 and linear_2 get 1x15 tiles per core at 256 patches, fewer than a 1D multicast's 16.
    for grid, with_config in (((11, 10), tower), ((12, 10), tower), ((13, 10), tower | {"linear_1", "linear_2"})):
        assert {name for name in LINEARS if _config(name, 256, grid) is not None} == with_config, grid
    assert all(_config(name, 16) is None for name in LINEARS)
    # The limits are arguments of the cached helper. In tiles (M, K, N): linear_2 at 1024 patches with a limit of 3
    # tiles of M per core, and fc1 at 2304 with 1.4 MB of circular buffers and with K blocks of at most 2.
    assert _linear_blocking(32, 192, 192, 12, 10, True, min_transposed_per_core_m=3) == (3, 20, 6, 1, 4, True)
    assert _linear_blocking(72, 40, 160, 12, 10, True, l1_bytes=1_400_000) == (6, 16, 8, 2, 2, True)
    assert _linear_blocking(72, 40, 160, 12, 10, True, max_in0_block_w=2) == (6, 16, 2, 2, 2, True)


@pytest.mark.parametrize("fp32_dest_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("grid", list(GRIDS.values()), ids=list(GRIDS))
def test_vision_linear_program_config_valid(grid, fp32_dest_acc):
    """Every config for every image size, as the device op and the 2D factory check it, and transpose_mcast only with
    at least LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core. The encoder accumulates in fp32; the bf16 case is
    linear_compute_config with fp32 accumulation turned off, as a measurement may set it."""
    compute_kernel_config = linear_compute_config()
    compute_kernel_config.fp32_dest_acc_en = fp32_dest_acc
    configs = transposed = 0
    for name, (in_features, out_features, patches_per_row) in LINEARS.items():
        for patches in range(16, 2304 + 1, 4):
            config = _config(name, patches, grid, compute_kernel_config)
            if config is not None:
                _assert_valid(config, patches // patches_per_row, in_features, out_features, grid, fp32_dest_acc)
                if config.transpose_mcast:
                    assert config.per_core_M >= LINEAR_MIN_TRANSPOSED_PER_CORE_M, f"{name} at {patches} patches"
                    transposed += 1
                configs += 1
    assert configs > 0 and transposed > 0


def test_vision_linear(monkeypatch):
    """linear() gives ttnn.linear the bias, the compute config, DRAM for the output and the helper's config for the
    tiles of x's and w's padded shapes: fc1 of a 2160-patch image (2176 rows padded) gets the config of 2160 rows, and
    of a 16-patch one ttnn's own. It runs the activation, the accurate GELU here, as an op of its own on ttnn.linear's
    output, not as ttnn.linear's activation argument, and without one returns ttnn.linear's output."""
    calls = []

    def record(op):
        def run(x, *args, **kwargs):
            calls.append((op, x, args, kwargs))
            return f"{op} output"

        return run

    monkeypatch.setattr(ttnn, "linear", record("linear"))
    monkeypatch.setattr(ttnn, "gelu", record("gelu"))
    compute_kernel_config = linear_compute_config()
    in_features, out_features, _ = LINEARS["fc1"]
    device = grid_device(12, 10)
    w = SimpleNamespace(padded_shape=(in_features, out_features))
    for patches, padded_rows in ((2160, 2176), (16, 32)):
        x = SimpleNamespace(padded_shape=(1, 1, padded_rows, in_features), device=lambda: device)
        calls.clear()
        assert linear(x, w, "bias", compute_kernel_config, activation=accurate_gelu) == "gelu output"
        [(op, x_in, args, kwargs), gelu] = calls
        assert op == "linear" and x_in is x and args == (w,), calls
        config, want = kwargs.pop("program_config"), _config("fc1", patches)
        assert (config is None) == (want is None), (patches, config)
        if want is not None:
            assert tuple(getattr(config, f) for f in FIELDS) == tuple(getattr(want, f) for f in FIELDS), config
        assert kwargs == dict(
            bias="bias", compute_kernel_config=compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # fast_and_approximate_mode=True would select ttnn's fast lookup-table GELU.
        assert gelu == ("gelu", "linear output", (), {"fast_and_approximate_mode": False}), gelu
        calls.clear()
        assert linear(x, w, None, compute_kernel_config) == "linear output" and len(calls) == 1


def test_vision_mlp_fuse_gelu(monkeypatch, expect_error):
    """M3VLMLP.fuse_gelu, False as built, on each grid. Off, ttnn.gelu runs once, on fc1's output, and no config
    carries an activation. On, where fc1 gets a config (256, 2160 and 2304 patches), that config, with the same
    blocking, carries the accurate GELU and ttnn.gelu does not run; where fc1 keeps ttnn's own (324 patches), ttnn.gelu
    runs as off. fc2 fuses nothing, and a forward with the switch off after one with it on fuses nothing. linear()
    refuses the fused GELU without the separate one, which would leave fc1 without a GELU where it has no config."""
    calls = []

    # Each recorder returns its call, with the padded shape and device that linear() reads, so an op's x is the call
    # whose output it reads.
    def record_linear(x, w, program_config, compute_kernel_config, **kwargs):
        shape = (*x.padded_shape[:-1], w.padded_shape[-1])
        calls.append(
            SimpleNamespace(
                op="linear",
                x=x,
                config=program_config,
                compute_kernel_config=compute_kernel_config,
                padded_shape=shape,
                device=x.device,
            )
        )
        return calls[-1]

    def record_gelu(x, **kwargs):
        calls.append(SimpleNamespace(op="gelu", x=x, kwargs=kwargs, padded_shape=x.padded_shape, device=x.device))
        return calls[-1]

    monkeypatch.setattr(mlp, "as_linear_weight", lambda mesh_device, w, dtype: SimpleNamespace(padded_shape=w.T.shape))
    monkeypatch.setattr(mlp, "as_linear_bias", lambda mesh_device, b, dtype: b)
    monkeypatch.setattr(ttnn, "linear", record_linear)
    monkeypatch.setattr(ttnn, "gelu", record_gelu)
    monkeypatch.setattr(ttnn, "deallocate", lambda x: None)
    hidden, intermediate, _ = LINEARS["fc1"]
    weights = dict(
        fc1_weight=torch.empty(intermediate, hidden),
        fc1_bias=torch.empty(intermediate),
        fc2_weight=torch.empty(hidden, intermediate),
        fc2_bias=torch.empty(hidden),
    )
    for grid in GRIDS.values():
        device = grid_device(*grid)
        model = mlp.M3VLMLP(device, **weights)
        assert model.fuse_gelu is False
        fused = set()
        for patches in (256, 324, 2160, 2304):
            x = SimpleNamespace(padded_shape=(1, 1, nearest_32(patches), hidden), device=lambda: device)
            want = _config("fc1", patches, grid)
            for fuse_gelu in (True, False):
                model.fuse_gelu = fuse_gelu
                calls.clear()
                out = model(x)
                fc1, *between, fc2 = calls
                assert fc1.op == fc2.op == "linear" and fc1.x is x and out is fc2, (grid, patches, calls)
                assert fc1.compute_kernel_config is fc2.compute_kernel_config is model.compute_kernel_config
                assert fc2.config is None or fc2.config.fused_activation is None, (grid, patches)
                if fuse_gelu and want is not None:
                    assert not between and fc2.x is fc1, (grid, patches, calls)
                    assert tuple(getattr(fc1.config, f) for f in FIELDS) == tuple(getattr(want, f) for f in FIELDS)
                    # A param of 1.0 would select the fast approximation.
                    assert fc1.config.fused_activation.op_type == ttnn.UnaryOpType.GELU, (grid, patches)
                    assert list(fc1.config.fused_activation.params) == [0.0], (grid, patches)
                    fused.add(patches)
                else:
                    assert fc1.config is None or fc1.config.fused_activation is None, (grid, patches, fuse_gelu)
                    assert [c.op for c in between] == ["gelu"] and between[0].x is fc1 and fc2.x is between[0], calls
                    assert between[0].kwargs == {"fast_and_approximate_mode": False}, (grid, patches)
        assert fused == {256, 2160, 2304}, (grid, fused)
    with expect_error(AssertionError, "fused_activation needs activation"):
        linear(None, None, None, linear_compute_config(), fused_activation=mlp.ACCURATE_GELU_FUSED)
