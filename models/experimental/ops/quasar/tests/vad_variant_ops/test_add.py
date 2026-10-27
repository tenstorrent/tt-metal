# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.add`` — every distinct add the VAD-variant model ran.

Shape source (kernel cache): binary_ng compute ``eltwise_binary_no_bcast`` /
``eltwise_binary_row_bcast`` / ``eltwise_binary_sfpu_no_bcast`` with readers
``reader_interleaved_{no,row}_bcast`` and writers ``writer_interleaved_no_bcast``. Ct args
map via ``binary_ng_program_factory.cpp`` (fork revision ``6b9b6a0e0e2~1``):
reader = ``TensorAccessorArgs(a) ++ TensorAccessorArgs(b) ++ [has_sharding]``. A sharded
accessor carries page size (2048 = bf16 tile, 1088 = bfp8, 4096 = fp32), shard shape in
pages, and the 20-bank core list => dtype + shard spec are cache-determined.

Groups:
  * ResNet residual add + ReLU (``PACK_RELU``), sharded, one per stage           [H shard/dtype, M shape]
  * BEV-query adds on [1, 10000, 256] (height-sharded 20 cores, shard 512x256)   [H/M]
  * mixed-dtype adds (bf16 + bfp8 / bf16 + fp32), incl. row-broadcast bias adds  [M/L]
  * plain bf16 / fp32 adds (shapes not recoverable — interleaved, runtime args)   [L]

Hash prefixes in ids are the compute kernel variant.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED
HS_Q = ("hs", (512, E))  # [1, 10000, 256] height-sharded 20 cores -> 512-row shards

# fmt: off
# (a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem, relu)
SHARDED_CASES = [
    # ResNet residual add+ReLU, one camera at 384x640: layer1..layer4 outputs.
    pytest.param((1, 1, 15360, 256), (1, 1, 15360, 256), "bf16", "bfp8", ("hs", (768, 256)), ("hs", (768, 256)), ("hs", (768, 256)), True, id="B1-layer1-153070"),
    pytest.param((1, 1, 3840, 512), (1, 1, 3840, 512), "bfp8", "bfp8", ("hs", (192, 512)), ("hs", (192, 512)), ("hs", (192, 512)), True, id="B2-layer2-298647"),
    pytest.param((1, 1, 960, 1024), (1, 1, 960, 1024), "bfp8", "bfp8", ("bs", (192, 256)), ("bs", (192, 256)), ("bs", (192, 256)), True, id="B3-layer3-298647"),
    pytest.param((1, 1, 240, 2048), (1, 1, 240, 2048), "bf16", "bf16", ("bs", (96, 512), 3), ("bs", (96, 512), 3), ("bs", (96, 512), 3), True, id="B4-layer4-307369"),
    pytest.param((1, 1, 3840, 512), (1, 1, 3840, 512), "bfp8", "bfp8", "l1", ("hs", (192, 512)), "l1", True, id="B5-layer2-mixed-887859"),
    # BEV queries + positional embedding (both height-sharded).
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", HS_Q, HS_Q, HS_Q, False, id="B6-bev-query-505549"),
    # Row-broadcast add on a height-sharded BEV query (LLK row bcast).
    pytest.param((1, Q, E), (1, 1, E), "bf16", "bf16", HS_Q, "l1", HS_Q, False, id="B11-bev-rowbcast-hs-182662"),
]

INTERLEAVED_CASES = [
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bfp8", "l1", "dram", "l1", id="B7-bf16+bfp8-910086"),
    pytest.param((1, Q, E), (1, Q, E), "bf16", "fp32", "dram", "dram", "dram", id="B8-bf16+fp32-136460"),
    # can_bus MLP output broadcast onto every BEV query.
    pytest.param((1, Q, E), (1, 1, E), "bf16", "fp32", "dram", "dram", "dram", id="B9-canbus-rowbcast-175736"),
    # Un-fused linear bias (bfp8 weights' bias) broadcast over rows.
    pytest.param((1, Q, E), (1, 1, E), "bf16", "bfp8", "l1", "dram", "l1", id="B10-bias-rowbcast-l1-133548"),
    pytest.param((1, Q, E), (1, 1, E), "bf16", "bfp8", "dram", "dram", "dram", id="B10-bias-rowbcast-dram-133548"),
    pytest.param((1, Q, E), (1, 1, E), "bf16", "bf16", "l1", "dram", "l1", id="B11-rowbcast-l1-182662"),
    pytest.param((1, Q, E), (1, 1, E), "bf16", "bf16", "dram", "dram", "dram", id="B11-rowbcast-dram-182662"),
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", "l1", "l1", "l1", id="B14-bf16-l1-506018"),
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", "dram", "dram", "dram", id="B14-bf16-dram-506018"),
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", "l1", "dram", "l1", id="B14-bf16-l1+dram-506018"),
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", "dram", "l1", "dram", id="B14-bf16-dram+l1-506018"),
    # fp32 SFPU add (shape not recoverable; small decoder-side tensor assumed).
    pytest.param((1, 300, 32), (1, 300, 32), "fp32", "fp32", "l1", "l1", "l1", id="B19-fp32-531668"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem, relu", SHARDED_CASES)
def test_add_sharded(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem, relu):
    pcc = 0.99 if "bfp8" in (a_dt, b_dt) else 0.999
    U.run_binary(
        ttnn_mesh_device,
        ttnn.add,
        lambda x, y: x + y,
        a_shape,
        b_shape,
        a_dt,
        b_dt,
        a_mem,
        b_mem,
        out_mem,
        relu=relu,
        pcc=pcc,
    )


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem", INTERLEAVED_CASES)
def test_add(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    pcc = 0.99 if "bfp8" in (a_dt, b_dt) else 0.999
    U.run_binary(
        ttnn_mesh_device, ttnn.add, lambda x, y: x + y, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem, pcc=pcc
    )


@U.with_default_mesh()
@pytest.mark.parametrize(
    "a_shape",
    # fp32 ROW_MAJOR add on a DRAM tensor with last dim <= 16 (64-byte sticks:
    # reader_interleaved_rm_no_bcast/971191). Shape not recoverable; 2-wide coordinates assumed.
    [pytest.param((1, 300, 2), id="B19-fp32-rowmajor-971191")],
)
def test_add_fp32_row_major(ttnn_mesh_device, reset_seeds, a_shape):
    U.run_binary(
        ttnn_mesh_device,
        ttnn.add,
        lambda x, y: x + y,
        a_shape,
        a_shape,
        "fp32",
        "fp32",
        "dram",
        "dram",
        "dram",
        layout=ttnn.ROW_MAJOR_LAYOUT,
        pcc=0.9999,
    )
