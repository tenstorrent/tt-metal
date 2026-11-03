# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host tests of the weight matmuls' program configs (linear_program_config) and of linear(), which runs those matmuls.

They pin the configs at 672x672 and a few smaller sizes. For every image size on the 11x10, 12x10 and 13x10 grids of
Blackhole chips, every config must pass the matmul op's own checks and fit LINEAR_L1_BYTES. They also check how
linear() and M3VLMLP.fuse_gelu run the GELU, and where the MLP writes its outputs. The configs are built with the real
ttnn classes; no device.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision import mlp
from models.demos.minimax_m3.tt.vision.common import (
    DRAM_PLACEMENT,
    L1_INTERLEAVED_PLACEMENT,
    LINEAR_MIN_TRANSPOSED_PER_CORE_M,
    _linear_blocking,
    linear,
    linear_compute_config,
    linear_program_config,
)
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.mlp import accurate_gelu

from .vision_helpers import L1_GRID_IDS, assert_matmul_config_valid, grid_device

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


def _config(name, patches, grid=(12, 10), compute_kernel_config=None):
    """linear_program_config for one of LINEARS in the forward of an image of `patches` patches."""
    in_features, out_features, patches_per_row = LINEARS[name]
    if compute_kernel_config is None:
        compute_kernel_config = linear_compute_config()
    return linear_program_config(
        grid_device(*grid), patches // patches_per_row, in_features, out_features, compute_kernel_config
    )


def test_vision_linear_program_config_pinned():
    """The configs match CONFIGS_672 and CONFIGS_MID, and other limits change the helper's choice.

    At 256 patches only the expected linears get a config on each grid. At 16 patches none does, on 12x10.
    """
    for name, want in CONFIGS_672.items():
        config = _config(name, 2304)
        assert config is not None, name
        assert tuple(getattr(config, f) for f in FIELDS) == want, f"{name}: {config}"
        assert (config.out_block_h, config.out_block_w) == (config.per_core_M, config.per_core_N), name
        in_features, out_features, patches_per_row = LINEARS[name]
        assert_matmul_config_valid(config, 2304 // patches_per_row, in_features, out_features, (12, 10), True)
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
@pytest.mark.parametrize("grid", list(L1_GRID_IDS.values()), ids=list(L1_GRID_IDS))
def test_vision_linear_program_config_valid(grid, fp32_dest_acc):
    """Every config for every image size passes the matmul op's checks, with fp32 or bf16 accumulation.

    A config uses transpose_mcast only with at least LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core.
    """
    compute_kernel_config = linear_compute_config()
    compute_kernel_config.fp32_dest_acc_en = fp32_dest_acc
    configs = transposed = 0
    for name, (in_features, out_features, patches_per_row) in LINEARS.items():
        for patches in range(16, 2304 + 1, 4):
            config = _config(name, patches, grid, compute_kernel_config)
            if config is not None:
                rows = patches // patches_per_row
                assert_matmul_config_valid(config, rows, in_features, out_features, grid, fp32_dest_acc)
                if config.transpose_mcast:
                    assert config.per_core_M >= LINEAR_MIN_TRANSPOSED_PER_CORE_M, f"{name} at {patches} patches"
                    transposed += 1
                configs += 1
    assert configs > 0 and transposed > 0


def test_vision_linear(monkeypatch):
    """linear() gives ttnn.linear the helper's config for the padded shapes, then runs the activation as its own op.

    ttnn.linear also gets the bias, the compute config and its memory_config, DRAM by default. The activation writes
    activation_memory_config, and ttnn.gelu gets no memory_config when that is None.
    """
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
    # (linear's memory keywords, ttnn.linear's memory_config, ttnn.gelu's keywords)
    placements = (
        ({}, ttnn.DRAM_MEMORY_CONFIG, {"fast_and_approximate_mode": False}),
        (
            dict(memory_config=ttnn.L1_MEMORY_CONFIG, activation_memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.L1_MEMORY_CONFIG,
            {"fast_and_approximate_mode": False, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ),
    )
    for patches, padded_rows in ((2160, 2176), (16, 32)):
        x = SimpleNamespace(padded_shape=(1, 1, padded_rows, in_features), device=lambda: device)
        for memory, linear_memory_config, gelu_kwargs in placements:
            calls.clear()
            assert linear(x, w, "bias", compute_kernel_config, activation=accurate_gelu, **memory) == "gelu output"
            [(op, x_in, args, kwargs), gelu] = calls
            assert op == "linear" and x_in is x and args == (w,), calls
            config, want = kwargs.pop("program_config"), _config("fc1", patches)
            assert (config is None) == (want is None), (patches, config)
            if want is not None:
                assert tuple(getattr(config, f) for f in FIELDS) == tuple(getattr(want, f) for f in FIELDS), config
            assert kwargs == dict(
                bias="bias", compute_kernel_config=compute_kernel_config, memory_config=linear_memory_config
            )
            # fast_and_approximate_mode=True would select ttnn's fast lookup-table GELU.
            assert gelu == ("gelu", "linear output", (), gelu_kwargs), gelu
        calls.clear()
        assert linear(x, w, None, compute_kernel_config) == "linear output" and len(calls) == 1


def test_vision_mlp_fuse_gelu(monkeypatch, expect_error):
    """With M3VLMLP.fuse_gelu on, fc1's config carries the GELU where fc1 gets a config.

    Elsewhere, and with fuse_gelu off as built, ttnn.gelu runs on fc1's output. fc2 never fuses. linear() refuses the
    fused GELU without the separate one. In DRAM_PLACEMENT and L1_INTERLEAVED_PLACEMENT, fc1, GELU and fc2 write the
    placement's fc1_out, gelu_out and residual, and a fused fc1 writes gelu_out.
    """
    calls = []

    # Each recorder returns its call, with the padded shape and device that linear() reads, so an op's x is the call
    # whose output it reads.
    def record_linear(x, w, program_config, compute_kernel_config, memory_config, **kwargs):
        shape = (*x.padded_shape[:-1], w.padded_shape[-1])
        calls.append(
            SimpleNamespace(
                op="linear",
                x=x,
                config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=memory_config,
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
    for grid in L1_GRID_IDS.values():
        device = grid_device(*grid)
        model = mlp.M3VLMLP(device, **weights)
        assert model.fuse_gelu is False
        fused = set()
        for patches in (256, 324, 2160, 2304):
            x = SimpleNamespace(padded_shape=(1, 1, nearest_32(patches), hidden), device=lambda: device)
            want = _config("fc1", patches, grid)
            for placement, fuse_gelu in itertools.product((DRAM_PLACEMENT, L1_INTERLEAVED_PLACEMENT), (True, False)):
                case = (grid, patches, placement.name, fuse_gelu)
                model.fuse_gelu = fuse_gelu
                calls.clear()
                out = model(x) if placement is DRAM_PLACEMENT else model(x, placement)
                fc1, *between, fc2 = calls
                assert fc1.op == fc2.op == "linear" and fc1.x is x and out is fc2, (case, calls)
                assert fc1.compute_kernel_config is fc2.compute_kernel_config is model.compute_kernel_config, case
                assert fc2.config is None or fc2.config.fused_activation is None, case
                assert fc2.memory_config == placement.residual, case
                if fuse_gelu and want is not None:
                    assert not between and fc2.x is fc1, (case, calls)
                    assert tuple(getattr(fc1.config, f) for f in FIELDS) == tuple(getattr(want, f) for f in FIELDS)
                    # A param of 1.0 would select the fast approximation.
                    assert fc1.config.fused_activation.op_type == ttnn.UnaryOpType.GELU, case
                    assert list(fc1.config.fused_activation.params) == [0.0], case
                    gelu_out = placement.fc1_out if placement.gelu_out is None else placement.gelu_out
                    assert fc1.memory_config == gelu_out, case
                    fused.add(patches)
                else:
                    assert fc1.config is None or fc1.config.fused_activation is None, case
                    assert [c.op for c in between] == ["gelu"] and between[0].x is fc1 and fc2.x is between[0], calls
                    assert fc1.memory_config == placement.fc1_out, case
                    output = {} if placement.gelu_out is None else {"memory_config": placement.gelu_out}
                    assert between[0].kwargs == {"fast_and_approximate_mode": False, **output}, case
        assert fused == {256, 2160, 2304}, (grid, fused)
    with expect_error(AssertionError, "fused_activation needs activation"):
        linear(None, None, None, linear_compute_config(), fused_activation=mlp.ACCURATE_GELU_FUSED)
