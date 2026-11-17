# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 7: regime R2 `height_sharded` (A L1 height-sharded, each core's resident shard holds all of
K for its rows and is consumed in place; B from DRAM; output in place in the requested L1 height shard)
and batched weights (R1d on interleaved A; R2 / R3 shards mapped onto batches).

Pins what the golden suite does not see:
  - PDL-shaped height shards on a non-rectangular core set (103 cores) with a ragged last shard;
  - the N-split streamed-output fallback (B's whole-K block too large for one pass) on a single core;
  - COL_MAJOR shard orientation; the op's default output placement when memory_config is None;
  - batched B on a height shard holding WHOLE images (block_batches > 1) with a ragged last shard,
    and on a block shard that lies inside one image;
  - batched B with every epilogue on interleaved A (grid_b > 1, flat placement).
Accuracy gate: helpers.TOLERANCES / LOW_FIDELITY_TOLERANCES via run_genify_matmul.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

_BF16, _BFP8, _FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
_LOFI = ttnn.MathFidelity.LoFi
_HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _run(
    a_shape,
    b_shape,
    *,
    placement="height_sharded",
    epilogue="none",
    dtype=_BF16,
    weight_dtype=_BF16,
    acc=True,
    act_shard=None,
    **extras,
):
    def go(device):
        ex = dict(extras)
        if act_shard is not None:
            ex["act_shard"] = act_shard
        run_genify_matmul(
            (a_shape, b_shape),
            dtype=dtype,
            weight_dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=acc,
            placement=placement,
            epilogue=epilogue,
            device=device,
            extras=ex,
        )

    return go


def _grid(x, y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))])


CASES = {
    # Cartesian-shaped (auto EVEN height shards).
    "tall_m_swiglu_bf8": _run((1, 1, 2048, 256), (256, 512), epilogue="swiglu", dtype=_BFP8, weight_dtype=_BFP8),
    "m12_bias_relu_dest16": _run((1, 384, 1536), (1536, 1024), epilogue="bias_relu", acc=False),
    "decode_m1_addcmul_fp32": _run((1, 2048), (2048, 1536), epilogue="addcmul", dtype=_FP32),
    # PDL res2 reduce: 103 cores (9 full grid rows + a partial one), ragged last shard, bf8, LoFi.
    "pdl_res2_reduce_103_cores": _run(
        (1, 1, 32768, 128),
        (128, 64),
        epilogue="bias_relu",
        dtype=_BFP8,
        weight_dtype=_BFP8,
        act_shard=([320, 128], 103),
        math_fidelity=_LOFI,
        bias_dtype=_BFP8,
    ),
    # PDL res3 shortcut: bf16 A -> bf8 output, 86 cores.
    "pdl_res3_shortcut_bf16_in": _run(
        (1, 1, 8192, 256),
        (256, 512),
        epilogue="bias",
        weight_dtype=_BFP8,
        act_shard=([96, 256], 86),
        math_fidelity=_LOFI,
        bias_dtype=_BFP8,
        output_dtype=_BFP8,
    ),
    # Batched B, one image per shard (the cartesian batched cell) and a shard holding WHOLE images
    # (3 images of 4 tiles over 8-tile shards: block_batches = 2, the last shard half padding).
    "batched_image_per_shard": _run((1, 4, 128, 128), (1, 4, 128, 256), epilogue="bias"),
    "batched_whole_images_ragged": _run((3, 128, 128), (3, 128, 256), act_shard=([256, 128], 2)),
    "batched_whole_images_addcmul": _run((3, 128, 128), (3, 128, 256), epilogue="addcmul", act_shard=([256, 128], 2)),
    # Batched B on block shards inside one image.
    "batched_block_sharded": _run((1, 4, 128, 128), (1, 4, 128, 256), placement="block_sharded", epilogue="swiglu"),
    # Batched B on interleaved A (R1d), every epilogue family.
    "batched_interleaved_none": _run((1, 4, 128, 128), (1, 4, 128, 256), placement="interleaved"),
    "batched_interleaved_swiglu": _run((2, 3, 96, 256), (2, 3, 256, 512), placement="interleaved", epilogue="swiglu"),
    "batched_interleaved_addcmul_bf8": _run(
        (1, 8, 64, 128),
        (1, 8, 128, 512),
        placement="interleaved",
        epilogue="addcmul",
        dtype=_BFP8,
        weight_dtype=_BFP8,
        acc=False,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_height_sharded_and_batched(name, device):
    CASES[name](device)


def _hs_inputs(device, m, k, n, shard_h, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    torch.manual_seed(0)
    a = torch.randn(1, 1, m, k)
    b = torch.randn(k, n) / k**0.5
    a_mc = ttnn.MemoryConfig(_HS, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [shard_h, k], orientation))
    ta = create_ttnn_input_tensor(a, device, dtype=_BF16, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(b, device, dtype=_BF16, layout=ttnn.TILE_LAYOUT)
    return a, b, ta, tb


def _pcc(got, ref):
    return torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1]


def test_default_output_placement_col_major(device):
    """memory_config=None: the op places the output L1 height-sharded on A's cores (A's shard height, the
    whole width, A's orientation). COL_MAJOR orientation maps shard r to the r-th core column-wise."""
    a, b, ta, tb = _hs_inputs(device, 512, 256, 320, 128, _grid(2, 2), ttnn.ShardOrientation.COL_MAJOR)
    out = genify_matmul(ta, tb)
    mc = out.memory_config()
    assert mc.memory_layout == _HS and mc.buffer_type == ttnn.BufferType.L1
    assert mc.shard_spec.grid == ta.memory_config().shard_spec.grid and list(mc.shard_spec.shape) == [128, 320]
    assert mc.shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR
    assert _pcc(ttnn.to_torch(out).float(), a.float() @ b.float()) > 0.9999


def test_height_shard_n_split(device, monkeypatch):
    """A budget one byte short of the whole-shard block (B's whole-K x whole-N block, single-buffered since
    the walk is one K-block) forces the N-split streamed-output fallback: the resident shard is re-pushed
    per N-block and the ragged last N-block is narrower. No cb_act_tiles is priced (A is used in place)."""
    a, b, ta, tb = _hs_inputs(device, 256, 512, 256, 128, _grid(2, 1))
    geom = pd._geometry(ta, tb, ta, True)
    topo = pd.BlockShardedTopology(None, 2, 1, 4, 16, 8, (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)))
    full = pd._footprint(geom, 4, 8, 16, partials=False, out_in_place=True, depth_weight=1, act_cb=False)
    assert pd.choose_block_sharded_blocking(geom, topo, full) == (pd.Blocking(1, 4, 8, 16, 1, 16, 1, 4), True)
    blk, in_place = pd.choose_block_sharded_blocking(geom, topo, full - 1)
    assert not in_place and blk.block_m_tiles == 4 and blk.block_n_tiles < 8
    monkeypatch.setattr(pd, "l1_cb_budget", lambda *_: full - 1)
    got = ttnn.to_torch(genify_matmul(ta, tb)).float()
    assert _pcc(got, a.float() @ b.float()) > 0.9999


def test_batched_shard_mapping(expect_error):
    """Shard -> (batch, rows) mapping: inside one image, whole images, or refused."""
    assert pd.shard_rows(4, 4, 1) == pd.ShardRows(1, 1)
    assert pd.shard_rows(3, 4, 8) == pd.ShardRows(2, 4)
    assert pd.shard_rows(1, 12, 5) == pd.ShardRows(1, 5)  # single B: no constraint
    with expect_error(pd.BlockShardedLayoutError, "straddles"):
        pd.shard_rows(3, 4, 6)
