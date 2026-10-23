# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar nlp_concat_heads_decode NUM_HEADS-core requirement (llama32_1b decode STAGE 9).

The llama decode path merges the per-head SDPA output back into the hidden dim with
``ttnn.experimental.nlp_concat_heads_decode(attn_output_sharded, num_heads=32)`` (attention_1d.py:732).
Its output is HEIGHT_SHARDED with **one head-column per core**, so
``compute_output_specs`` builds the output grid via
``num_cores_to_corerangeset(num_heads, device->compute_with_storage_grid_size())``
(nlp_concat_heads_decode_device_operation.cpp:27) and even the sub-core path requires
``sub_core_grids.num_cores() >= num_heads`` (line 65). So the op needs **num_heads (=32) cores** regardless of
batch.

This is the crucial contrast with ``nlp_create_qkv_heads_decode``, which needs only ``num_users`` (= batch = 1)
cores -- which is why the llama decode gets *past* create_qkv (STAGE 4) but FATALs at concat (STAGE 9) on the
2-compute-node emulator:

    TT_FATAL work_split.cpp:99: Target number of cores 32 is greater than total number of available cores 2

The op / factory / work_split assert are byte-identical between the pr57616 side branch and merged main -- this
is NOT a merge regression. A run where it "worked" had >= 32 cores (full sim grid, or WH silicon = 8x4/8x8).
On the emulator's ``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"`` grid it cannot fit.

PORT/FIX TARGET: make the decode concat fit <= num_available cores (a grid-agnostic device head-merge -- the
output is just [1,batch,nq,hd] -> [1,1,batch,nq*hd]), NOT a host fallback. This test documents the requirement:
it SKIPS on a device with < num_heads cores today (the emulator), and PASSES on a >= num_heads device.

Run (Quasar sim, FULL grid -- do NOT clamp with TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_nlp_concat_heads_decode.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# llama-3.2-1B decode dims
N_HEADS = 32
HEAD_DIM = 64
Q_DIM = N_HEADS * HEAD_DIM  # 2048


def _batch_height_sharded(attn_torch, mesh_device, batch):
    """Per-head decode SDPA output -> HEIGHT_SHARDED over `batch` cores, shard (N_HEADS, HEAD_DIM).

    Mirrors decode_scores_memcfg (attention_1d.py:1727-1734) and the op's input contract
    (nlp_concat_heads_decode_device_operation.cpp: HEIGHT_SHARDED, shard[0]==padded_heads, num_cores==num_users).
    Built via RM upload + quasar.tilize (from_torch(TILE) hangs on the sim), then interleaved_to_sharded."""
    rm = ttnn.from_torch(
        attn_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qtil = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    t = (qtil or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    core_rs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, max(batch - 1, 0)))})
    shard = ttnn.ShardSpec(core_rs, (N_HEADS, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR)
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    return (qi2s or ttnn.interleaved_to_sharded)(t, memcfg)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("batch", [1], ids=["decode-batch1"])
def test_nlp_concat_heads_decode(mesh_device, batch):
    """Stock nlp_concat_heads_decode. Needs num_heads (=32) cores for its output grid, so it fits ONLY on a
    device exposing >= 32 compute cores.

    NOTE the dispatch-mode dependency of the grid: with TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2",
      - fast dispatch (TTNN_CONFIG_OVERRIDES enable_fast_runtime_mode=false) -> compute grid >= 32 cores -> PASSES;
      - slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1, the true 2-compute-node emulator) -> 2x1 = 2 cores -> SKIPS.
    The llama e2e runs under SLOW dispatch, so it sees 2 cores and FATALs here (work_split.cpp:99) -- the op
    cannot fit the 2-node emulator without a fix (pack heads/core, or a grid-agnostic device head-merge; NOT a
    host fallback). This test SKIPS on <32-core grids (documenting that) and, where it does run, validates shape,
    finiteness, AND value (PCC vs a torch head-merge reference) -- shape/finite alone would miss a wrong head
    ordering or per-head offset."""
    grid = mesh_device.compute_with_storage_grid_size()
    ncores = int(grid.x) * int(grid.y)
    if ncores < N_HEADS:
        pytest.skip(
            f"nlp_concat_heads_decode needs num_heads={N_HEADS} cores for its output grid; device has "
            f"{grid.x}x{grid.y}={ncores}. This is the emulator FATAL (work_split.cpp:99). Fix target: a "
            f"grid-agnostic device head-merge that fits <= {ncores} cores (not a host fallback)."
        )

    torch.manual_seed(0)
    attn = torch.randn(1, batch, N_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    attn_sharded = _batch_height_sharded(attn, mesh_device, batch)

    out = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=N_HEADS)
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    padded_batch = ((batch + 31) // 32) * 32
    assert tuple(o.shape) == (1, 1, padded_batch, Q_DIM), f"unexpected shape {tuple(o.shape)}"
    assert torch.isfinite(o).all(), "concat_heads_decode produced non-finite output"

    # Value check: concat_heads_decode maps input[0,b,h,d] -> output[0,0,b, h*head_dim + d], i.e. a per-user
    # flatten of the (n_heads, head_dim) block. The op pads the user/batch dim to a full tile, so compare only
    # the first `batch` rows against the torch reference.
    ref = attn.reshape(1, 1, batch, N_HEADS * HEAD_DIM)
    got = o[:, :, :batch, :]
    pcc = _pcc(got, ref)
    logger.info(
        f"[concat-heads-repro] out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}"
    )
    assert pcc > 0.99, f"concat_heads_decode value mismatch vs torch head-merge reference: PCC={pcc}"


def _grid_agnostic_head_merge(attn_torch, mesh_device, batch):
    """Grid-agnostic device head-merge: [1, batch, n_heads, head_dim] -> [1, 1, batch, n_heads*head_dim] TILE DRAM.

    The concat is a per-user flatten of the (n_heads, head_dim) block: input[0,b,h,d] -> output[0,0,b, h*hd+d].
    In ROW_MAJOR that is a contiguous reshape (same flat order), so upload row-major DRAM-interleaved, reshape to
    merge the (n_heads, head_dim) axes into the hidden dim, then quasar.tilize to the WO-matmul input layout.
    None of these steps shard by head, so this fits ANY grid -- including the 2-compute-node emulator where the
    stock nlp_concat_heads_decode (num_heads cores) FATALs. This is the model-side fix candidate for Quasar."""
    rm = ttnn.from_torch(
        attn_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    merged = ttnn.reshape(rm, (1, 1, batch, N_HEADS * HEAD_DIM))  # [1,1,batch,2048], row-major
    # tilize needs the height dim (batch) tile-aligned; the stock op pads batch to 32, so mirror that here
    # (pad the user/batch axis up to a full tile before tilize). Grid-agnostic: pad + tilize both fit any grid.
    pad_to = ((batch + 31) // 32) * 32
    if pad_to != batch:
        merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, pad_to - batch), (0, 0)], value=0.0)  # [1,1,pad_to,2048]
    qtil = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qtil or ttnn.tilize)(merged, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


@pytest.mark.parametrize("batch", [1], ids=["decode-batch1"])
def test_nlp_concat_heads_decode_grid_agnostic(mesh_device, batch):
    """Grid-agnostic device equivalent of nlp_concat_heads_decode (fits <= 2 cores, NO num_heads-core
    requirement, NO host fallback). PASSES on the 2-compute-node emulator (unlike the stock op, which SKIPS
    there). Validates it matches the torch head-merge reference -- this is the model-side fix candidate: route
    the decode concat through this device path on Quasar small grids."""
    torch.manual_seed(0)
    attn = torch.randn(1, batch, N_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    out = _grid_agnostic_head_merge(attn, mesh_device, batch)
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)

    ref = attn.reshape(1, 1, batch, N_HEADS * HEAD_DIM)
    got = o.reshape(1, 1, -1, Q_DIM)[:, :, :batch, :]  # slice off any tile-padding on the batch axis
    pcc = _pcc(got, ref)
    logger.info(
        f"[concat-heads-grid-agnostic] out {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}"
    )
    assert torch.isfinite(o).all(), "grid-agnostic head-merge produced non-finite output"
    assert got.shape[-1] == Q_DIM, f"unexpected hidden dim {got.shape}"
    assert pcc > 0.99, f"grid-agnostic head-merge mismatch vs torch reference: PCC={pcc}"
