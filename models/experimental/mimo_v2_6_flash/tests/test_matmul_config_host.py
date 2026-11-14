# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of tt/common.auto_dram_matmul_config, the fixed-budget copy of ttnn's matmul auto-config that the prefill
linears take (dram_linear; no device). The expected configs are what auto-config picked on a Blackhole 1x8 row (12 x 10
compute grid) for these shapes: the fixed config hit auto-config's program (MIMO_MM_CONFIG_CHECK=1, serving warm-up)."""

from types import SimpleNamespace

import pytest

import ttnn
from models.experimental.mimo_v2_6_flash.tt import common
from models.experimental.mimo_v2_6_flash.tt.common import auto_dram_matmul_config, compute_config

GRID = SimpleNamespace(x=12, y=10)
DRAM = ttnn.DRAM_MEMORY_CONFIG


class Tensor:
    """What auto_dram_matmul_config reads of a tensor."""

    def __init__(self, shape, dtype, memory_config=DRAM, layout=ttnn.TILE_LAYOUT):
        self.padded_shape, self.dtype, self.layout = shape, dtype, layout
        self._mem = memory_config

    def memory_config(self):
        return self._mem

    def device(self):
        return SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(GRID.x, GRID.y))


def cfg(m, k, n, budget=None):
    x, w = Tensor((1, 1, m, k), ttnn.bfloat16), Tensor((k, n), ttnn.bfloat8_b)
    return auto_dram_matmul_config(x, w, compute_config(ttnn.MathFidelity.HiFi2), ttnn.bfloat16, budget)


def fields(c, *names):
    return tuple(getattr(c, n) for n in names)


BLOCKS = ("in0_block_w", "out_subblock_h", "out_subblock_w", "out_block_h", "out_block_w", "per_core_M", "per_core_N")


def test_narrow_prefill_linears_take_the_1d_config():
    """A 32-row pass (and any output 8x wider than tall) is 1D, in0 multicast, num_global_cb_receivers 0 (C++'s
    aggregate default; the Python default 1 is another program)."""
    c = cfg(32, 4096, 1920)
    assert isinstance(c, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig)
    assert fields(c, *BLOCKS) == (2, 1, 1, 1, 1, 1, 1) and c.mcast_in0 and c.num_global_cb_receivers == 0
    c = cfg(128, 4096, 1920)
    assert fields(c, *BLOCKS) == (2, 4, 1, 4, 1, 4, 1)
    assert fields(cfg(32, 4096, 19072), *BLOCKS) == (2, 1, 1, 1, 5, 1, 5)


def test_square_prefill_linears_take_the_2d_config():
    c = cfg(512, 4096, 1920)
    assert isinstance(c, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)
    assert fields(c, *BLOCKS) == (1, 2, 1, 2, 5, 2, 5)
    assert fields(cfg(512, 1536, 4096), *BLOCKS) == (4, 2, 1, 2, 11, 2, 11)


@pytest.mark.parametrize("m", [3584, 3840, 4096])
def test_the_fixed_budgets(m):
    """The SWA o_proj of a 3584-4096-row pass: in0_block_w 4 at the text replica's budget (what auto-config picked in
    its warm-up); with encoders on the device (MM_ENCODER_L1_BUDGET) 4 for 3584 / 3840 rows and 2 for 4096, whose
    in0_block_w-4 circular buffers clash with the L1 the encoders keep; auto-config itself moved between 4, 2 and 1 with
    the L1 free (an encoder's warm-up between the warm-up and the capture of a pass compiled a new program then)."""
    if "MIMO_MM_L1_BUDGET" in __import__("os").environ:
        pytest.skip("MIMO_MM_L1_BUDGET is set")
    assert common.MM_L1_BUDGET == 1228800 and common.MM_ENCODER_L1_BUDGET == 1150000
    assert cfg(m, 1536, 4096).in0_block_w == 4
    assert cfg(m, 1536, 4096, common.MM_ENCODER_L1_BUDGET).in0_block_w == (2 if m == 4096 else 4)
    assert cfg(m, 1536, 4096, 1081344).in0_block_w == 2
    assert cfg(4096, 1536, 4096, 1025024).in0_block_w == 1


def test_set_mm_l1_budget(monkeypatch):
    monkeypatch.delenv("MIMO_MM_L1_BUDGET", raising=False)
    monkeypatch.setattr(common, "MM_L1_BUDGET", 1228800)
    common.set_mm_l1_budget(common.MM_ENCODER_L1_BUDGET)
    assert common.MM_L1_BUDGET == common.MM_ENCODER_L1_BUDGET and cfg(4096, 1536, 4096).in0_block_w == 2
    monkeypatch.setenv("MIMO_MM_L1_BUDGET", "1228800")
    monkeypatch.setattr(common, "MM_L1_BUDGET", 1228800)
    common.set_mm_l1_budget(1000000)
    assert common.MM_L1_BUDGET == 1228800, "an explicit MIMO_MM_L1_BUDGET wins"


def test_other_operands_keep_auto_config():
    """Batched activations, L1 or sharded operands, row-major tensors: no fixed config (ttnn's auto-config runs)."""
    assert cfg(32, 4096, 1920) is not None
    x = Tensor((2, 1, 32, 4096), ttnn.bfloat16)
    w = Tensor((4096, 1920), ttnn.bfloat8_b)
    ckc = compute_config(ttnn.MathFidelity.HiFi2)
    assert auto_dram_matmul_config(x, w, ckc, ttnn.bfloat16) is None
    x = Tensor((1, 1, 32, 4096), ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert auto_dram_matmul_config(x, w, ckc, ttnn.bfloat16) is None
    x = Tensor((1, 1, 32, 4096), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    assert auto_dram_matmul_config(x, w, ckc, ttnn.bfloat16) is None


def test_a_shape_never_assumes_more_l1_than_is_free(monkeypatch):
    """The fixed budget was checked on the text replica with MTP and on the multimodal one; a configuration that
    keeps more L1 gets the config of the L1 actually free at the shape's first call (as ttnn's auto-config would
    size it), decided once per shape. With the budget free, nothing changes."""
    monkeypatch.delenv("MIMO_MM_L1_BUDGET", raising=False)
    monkeypatch.setattr(common, "MM_L1_BUDGET", 1228800)
    monkeypatch.setattr(common, "_shape_budgets", {})
    used, free = [], {"now": 1468992}
    monkeypatch.setattr(common, "l1_program_budget", lambda device: free["now"])
    monkeypatch.setattr(ttnn, "linear", lambda x, w, program_config=None, **kw: used.append(program_config))
    ckc = compute_config(ttnn.MathFidelity.HiFi2)
    kw = dict(dtype=ttnn.bfloat16, memory_config=DRAM, compute_kernel_config=ckc)

    def run(m):
        common.dram_linear(Tensor((1, 1, m, 1536), ttnn.bfloat16), Tensor((1536, 4096), ttnn.bfloat8_b), **kw)
        return used[-1].in0_block_w

    assert run(4096) == 4  # the text replica's config
    free["now"] = 1081344  # something else now holds L1
    assert run(4096) == 4  # this shape's program was decided at its first call
    assert run(3840) == 2  # a shape first seen now is sized to what is free, as auto-config would
    free["now"] = 1468992
    assert run(3840) == 2 and run(3584) == 4
    assert common._shape_budgets[((1, 1, 3840, 1536), (1536, 4096), 1228800)] == 1081344

    monkeypatch.setattr(common, "MM_L1_GUARD", False)
    monkeypatch.setattr(common, "_shape_budgets", {})
    free["now"] = 1025024
    assert run(4096) == 4  # MIMO_MM_L1_GUARD=0: the fixed budget whatever is free
    monkeypatch.setattr(common, "MM_L1_GUARD", True)
    monkeypatch.setattr(common, "_shape_budgets", {})
    monkeypatch.setattr(common, "l1_program_budget", lambda device: 1 / 0)  # no memory view: the fixed budget
    assert run(4096) == 4
