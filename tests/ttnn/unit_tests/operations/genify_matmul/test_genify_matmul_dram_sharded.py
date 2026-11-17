# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4: regime R4 `dram_sharded` (B streamed from its DRAM bank shards, A broadcast from its
own L1 width-shard cores by rotating senders, output in place in an op-built L1 width shard).

Pins what the golden suite does not see:
  - the op-built output placement: one width shard per bank worker, shard w on bank w's worker
    (get_optimal_dram_bank_to_logical_worker_assignment), all padded rows;
  - A shard cores that are / are not bank workers, inside / outside the receiver rectangle;
  - a single A shard (fixed sender) and many 1-tile shards (64 rounds);
  - the K split of an A shard into several K-blocks (L1 / 16-bit-DEST cap) and the M split;
  - a ragged last bank (padding columns in the last B shard and output shard);
  - swiglu on natural (gate, up) bank columns, bias, addcmul.
Accuracy gate: helpers.TOLERANCES / LOW_FIDELITY_TOLERANCES via run_genify_matmul.
"""

import math

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

_BF16, _BFP8, _BFP4, _FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32
_LOFI = ttnn.MathFidelity.LoFi


def _run(a_shape, b_shape, *, epilogue="none", dtype=_BF16, weight_dtype=_BF16, acc=True, act_shard=None, **extras):
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
            placement="dram_sharded",
            epilogue=epilogue,
            device=device,
            extras=ex,
        )

    return go


CASES = {
    # Qwen decode-shaped: M = 1, bf16 x bf4b, LoFi, A on 8 / 10 cores (rotating senders).
    "decode_down_8cores": _run(
        (1, 1, 32, 1536), (1536, 5120), weight_dtype=_BFP4, act_shard=([32, 192], 8), math_fidelity=_LOFI
    ),
    "decode_swiglu_10cores": _run(
        (1, 1, 1, 5120),
        (5120, 9216),
        epilogue="swiglu",
        weight_dtype=_BFP4,
        act_shard=([32, 512], 10),
        math_fidelity=_LOFI,
    ),
    # Cartesian-shaped: A auto width-sharded EVEN (1-tile shards on 64 cores), M = 1 (m_non_aligned).
    "m1_64_shards": _run((1, 2048), (2048, 1536)),
    "bias_32_shards": _run((1, 1, 32, 1024), (1024, 2048), epilogue="bias", dtype=_BFP8, weight_dtype=_BFP8),
    "addcmul_32_shards": _run((1, 1, 32, 1024), (1024, 2048), epilogue="addcmul"),
    # Multi-row A (12 and 64 tile rows per shard): the block spans the rows; fp32 A forces an M split.
    "rows12_bias_relu": _run((1, 384, 1536), (1536, 1024), epilogue="bias_relu", acc=False),
    "rows64_fp32": _run((1, 1, 2048, 256), (256, 512), dtype=_FP32),
    "rows64_swiglu": _run((1, 1, 2048, 256), (256, 512), epilogue="swiglu", weight_dtype=_BFP4),
    # One A shard (explicit fixed sender) of 32 K tiles; 16-bit DEST splits it into 4 K-blocks.
    "single_shard_dest16": _run((1, 1, 32, 1024), (1024, 512), acc=False, act_shard=([32, 1024], 1)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_dram_sharded(name, device):
    CASES[name](device)


def _ds_tensors(device, k, n, w_dtype, a_cores, a_shard_w_tiles, m=32):
    torch.manual_seed(0)
    a = torch.randn(1, 1, m, k)
    w = torch.randn(1, 1, k, n) / math.sqrt(k)
    grid = ttnn.num_cores_to_corerangeset(a_cores, device.compute_with_storage_grid_size(), True)
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, a_shard_w_tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    banks = device.dram_grid_size().x
    padded_n = math.ceil(n / (32 * banks)) * 32 * banks
    w_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (k, padded_n // banks),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ta = create_ttnn_input_tensor(a, device, dtype=_BF16, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tw = create_ttnn_input_tensor(w, device, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, memory_config=w_mc)
    return a, w, ta, tw


def test_ragged_last_bank_and_output_placement(device):
    """N = 704 (22 tiles) over 8 banks of 3: the last bank holds one valid column. The output is the
    op-built L1 width shard: shard w on bank w's optimal worker."""
    k, n = 2048, 704
    a, w, ta, tw = _ds_tensors(device, k, n, _BFP8, a_cores=32, a_shard_w_tiles=2)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    out = genify_matmul(ta, tw, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=cfg)
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    workers = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, pd.DS_READ_NOC)
    shard_cores = ttnn.corerange_to_cores(mc.shard_spec.grid, None, True)
    assert [(c.x, c.y) for c in shard_cores] == [(c.x, c.y) for c in workers[: len(shard_cores)]]
    assert list(mc.shard_spec.shape) == [32, 96]
    ref = a.float() @ ttnn.to_torch(tw).float()
    got = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc > 0.9999, pcc
