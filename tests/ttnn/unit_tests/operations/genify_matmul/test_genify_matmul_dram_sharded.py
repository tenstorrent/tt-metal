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


# (cores_per_bank, A rounds on the writer, B K-blocks in flight, weight CB depth)
# Refinement 17: cores_per_bank = 2 does not divide the 3-column bank: slices of 2 and 1 columns (ragged).
_R6_KNOBS = [(3, True, 1, 2), (1, True, 1, 2), (3, False, 3, 4), (1, False, 2, 3), (2, True, 1, 2)]


@pytest.mark.parametrize(
    "knobs", _R6_KNOBS, ids=[f"cpb{c}-{'wact' if w else 'ract'}-t{t}-d{d}" for c, w, t, d in _R6_KNOBS]
)
def test_cores_per_bank_split(device, monkeypatch, knobs):
    """Refinement 6's knobs forced on N = 704 (8 banks of 3 columns, the last bank holds one valid column).
    cores_per_bank = 3: two slices of the last bank lie wholly in padding (no B reads, no output writes)
    and every slice lands in the bank's output shard on its primary worker; 1 = the Refinement 4 in-place
    topology. The A rounds run on the writer (default) or the reader; the B ring keeps 1-3 K-blocks in
    flight on transaction ids (32 one-K-block rounds each). bias exercises the per-slice epilogue row."""
    cores_per_bank, act_on_writer, b_ahead, depth = knobs
    monkeypatch.setattr(pd, "DS_CORES_PER_BANK", cores_per_bank)
    monkeypatch.setattr(pd, "DS_ACT_ON_WRITER", act_on_writer)
    monkeypatch.setattr(pd, "DS_B_AHEAD", b_ahead)
    monkeypatch.setattr(pd, "DS_DEPTH_WEIGHT", depth)
    k, n = 2048, 704
    a, w, ta, tw = _ds_tensors(device, k, n, _BFP8, a_cores=32, a_shard_w_tiles=2)
    bias = torch.randn(1, n)
    tbias = create_ttnn_input_tensor(bias, device, dtype=_BF16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    out = genify_matmul(
        ta, tw, bias=tbias, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=cfg
    )
    workers = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, pd.DS_READ_NOC)
    shard_cores = ttnn.corerange_to_cores(out.memory_config().shard_spec.grid, None, True)
    assert [(c.x, c.y) for c in shard_cores] == [(c.x, c.y) for c in workers[: len(shard_cores)]]
    ref = a.float() @ ttnn.to_torch(tw).float() + bias.float()
    got = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc > 0.9999, pcc


def _plan(device, a_shape, b_shape, a_cores, a_dtype=_BF16, w_dtype=_BFP4, acc=True, swiglu=False, fidelity=None):
    """R4 blocking for an allocated (uninitialized) A width-sharded over `a_cores` and B bank-sharded."""
    m_pad = math.prod(a_shape[:-2]) * (-(-a_shape[-2] // 32) * 32)
    k, n = a_shape[-1], b_shape[-1]
    grid = ttnn.num_cores_to_corerangeset(a_cores, device.compute_with_storage_grid_size(), True)
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [m_pad, k // a_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    banks = device.dram_grid_size().x
    granule = 64 if swiglu else 32
    b_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [k, -(-n // (granule * banks)) * granule],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape)), a_dtype, ttnn.TILE_LAYOUT, device, a_mc)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), w_dtype, ttnn.TILE_LAYOUT, device, b_mc)
    out_mc = pd.dram_sharded_output_memory_config(a, b, swiglu)
    n_out = n // 2 if swiglu else n
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(a_shape[:-1]) + [n_out]), a_dtype, ttnn.TILE_LAYOUT, device, out_mc
    )
    kw = {"math_fidelity": fidelity} if fidelity is not None else {}
    geom, topo, blk = pd.plan_dram_sharded(a, b, o, acc, epilogue="swiglu" if swiglu else "none", **kw)
    for t in (a, b, o):
        ttnn.deallocate(t)
    return geom, topo, blk


def test_plan_blocking(device):
    """The R4 extent knobs: one A shard per round unless L1 or the 16-bit-DEST chain cap splits it."""
    # PERF FOCUS 2 at its LoFi config: 8 shards of 17 K tiles, 20 bank columns split over 4 cores per bank
    # (Refinement 17: the smallest split whose compute hides under the bank stream; 5-column slices), one
    # block row, one K-block per shard. Every bank's slice readers sit in its home-row segment.
    lofi = ttnn.MathFidelity.LoFi
    geom, topo, blk = _plan(device, (1, 1, 1, 4352), (4352, 5120), 8, fidelity=lofi)
    assert (topo.num_shards, topo.shard_k_tiles, topo.bank_n_tiles) == (8, 17, 20)
    assert (topo.cores_per_bank, topo.core_n_tiles, len(topo.workers)) == (4, 5, 32)
    assert len({(c.x, c.y) for c in topo.workers}) == 32 and topo.workers[::4] == topo.primaries
    assert all(c.y == topo.workers[(i // 4) * 4].y for i, c in enumerate(topo.workers))
    assert (blk.block_m_tiles, blk.block_n_tiles, blk.block_k_tiles, blk.num_k_blocks) == (1, 5, 17, 8)
    # Qwen3.8 decode qkvg (PERF FOCUS 3): 14 bank columns over 4 cores -- slices of 4, 4, 4 and 2 columns.
    geom, topo, blk = _plan(device, (1, 1, 1, 5120), (5120, 3584), 10, fidelity=lofi)
    assert (topo.bank_n_tiles, topo.cores_per_bank, topo.core_n_tiles) == (14, 4, 4)
    # 16-bit DEST caps the in-DEST K chain at DEST16_BLOCK_K_MAX: a 32-tile shard is split in 4.
    geom, topo, blk = _plan(device, (1, 1, 32, 1024), (1024, 512), 1, w_dtype=_BF16, acc=False)
    assert topo.num_shards == 1 and blk.block_k_tiles == pd.DEST16_BLOCK_K_MAX and blk.num_k_blocks == 4
    # fp32 A, 64 tile rows: the whole-row block does not fit L1 next to the resident shards -> M split.
    geom, topo, blk = _plan(device, (1, 1, 2048, 256), (256, 512), 8, a_dtype=_FP32, w_dtype=_BF16)
    assert blk.block_m_tiles < geom.tensor_m_tiles and geom.tensor_m_tiles % blk.block_m_tiles == 0
