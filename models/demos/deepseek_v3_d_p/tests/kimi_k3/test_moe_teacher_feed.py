# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 MoE layer teacher-fed from a vLLM trace on one 4x4 stage (SP on axis 1), chunk by chunk.

The device half of ``moe_teacher_feed.py`` (read its docstring for the three steps; its CPU half is tested in
``test_teacher_feed_contract.py``). The production MoE (the MoE block, ``TtMoEBlockFfn`` as ``TtKimiK3Block`` builds
it for an SP-on-axis-1 stage, its routed experts from the stage's TTNN weight cache) runs every 2560-token chunk of
the prefix in order, eager, with the chunk's real-token span as the runtime passes it. Its input is the trace-rebuilt
``post_attention_layernorm`` output, so every chunk is teacher-fed. Each chunk's MoE output and the device's routing
(read from the block's routing rows: per token the top-16 expert ids in the block's expert order, mapped back to
checkpoint ids, and their bf16 weights) are saved to ``$TT_MOE_TF_OUT``; nothing is scored while the device is held
(``moe_teacher_feed.py score``). Skips unless ``$TT_MOE_TF_INPUT`` and ``$TT_MOE_TF_OUT`` are set;
``$TT_MOE_TF_REVERSE=1`` runs the chunks in reverse order (``moe_teacher_feed.py same`` then compares two runs).

    TT_MOE_TF_INPUT=/mnt/.../k31-l49-moe-input-128k-v2 TT_MOE_TF_OUT=/mnt/.../k31-l49-moe-dev-128k \\
    models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- \\
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_moe_teacher_feed.py::test_k31_moe_layer_teacher_fed[blackhole-4x4-sp1]"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, provenance
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed import NUM_EXPERTS, TOP_K, owner_positions
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import chunk_path, finish_run, load_input, start_run

PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]
SOURCE, OUT = os.environ.get("TT_MOE_TF_INPUT"), os.environ.get("TT_MOE_TF_OUT")
# TT_MOE_TF_REVERSE=1 runs the chunks last to first: the MoE carries no state from one chunk to the next, so its
# per-row outputs must equal a forward run's bit for bit, and any trend over position is the inputs'.
REVERSE = os.environ.get("TT_MOE_TF_REVERSE") == "1"


def device_routing(ffn, block, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing of the block's last call as (checkpoint expert ids ``[rows, 16]`` int64, weights ``[rows, 16]``
    bf16) in the chunk's token order.

    ``block.last_head.routing_rows`` holds, on each chip, its owner rows: per token 16 expert ids (block order) then
    16 bf16 weights as uint16 words (``router_common.hpp routing_rows::build``). The block's chip n = (SP row r, TP
    column p) is mesh chip (p, r) with SP on axis 1 (``moe_block.to_mesh_order``); ``owner_positions`` places its
    owner rows in the chunk (SP row r holds the chunk's rows [r * chunk / SP, (r + 1) * chunk / SP), the input's
    ShardTensor2dMesh). Block expert b is checkpoint expert ``block_order_perm[b]``."""
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import block_order_perm

    mesh = ffn.mesh
    R, C = tuple(mesh.shape)
    sp_axis = ffn.options.sp_axis
    sp, tp = (R, C) if sp_axis == 0 else (C, R)
    chunk = block.resources.config.chunk_tokens
    assert block.splits == 1, f"the routing rows hold the last of {block.splits} splits only"
    words = torch.full((chunk, 2 * TOP_K), -1, dtype=torch.int64)
    for i, d in enumerate(ttnn.get_device_tensors(block.last_head.routing_rows)):
        mr, mc = divmod(i, C)
        r, p = (mr, mc) if sp_axis == 0 else (mc, mr)
        owned, pos = owner_positions(
            r, p, rows_per_sp_row=chunk // sp, pad_rows=block.pad_rows, row_blocks=block.row_blocks, tp=tp
        )
        words[pos] = (ttnn.to_torch(d).to(torch.int64).reshape(-1, 2 * TOP_K) & 0xFFFF)[owned]
    assert (words[:rows] >= 0).all(), "a chunk row no chip's routing rows hold"
    perm = block_order_perm(mesh, NUM_EXPERTS // (R * C), sp_axis)
    ids = perm[words[:rows, :TOP_K]]
    weights = words[:rows, TOP_K:].to(torch.int16).view(torch.bfloat16)
    return ids, weights


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.skipif(
    not (SOURCE and OUT),
    reason="set TT_MOE_TF_INPUT (a moe_teacher_feed input) and TT_MOE_TF_OUT (the run's output dir)",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
def test_k31_moe_layer_teacher_fed(mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import (
        MoEBlockFfnOptions,
        TtMoEBlockFfn,
        release_blocks,
    )
    from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_impl_of

    source, out = Path(SOURCE), Path(OUT)
    built = json.loads((source / MANIFEST).read_text())
    layer = built["layer"]
    count = int(os.environ.get("TT_MOE_TF_COUNT", built["segments"][-1]["end"]))
    mesh_shape = tuple(mesh_device.shape)
    layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
    chunk = layout.chunk
    weight_cache = k31_stage.weight_cache_path(mesh_shape)
    adapter = k31_stage.ADAPTER
    assert moe_impl_of(adapter) == "block", "the production K3.1 MoE is the MoE block"
    manifest = {
        "layer": layer,
        "input": str(source),
        "input_manifest": {k: built[k] for k in ("trace", "trace_commit", "input", "checkpoint")},
        "count": count,
        "chunk": chunk,
        "mesh": list(mesh_shape),
        "sp_axis": layout.sp_axis,
        "moe_impl": "block",
        "order": "reverse" if REVERSE else "forward",
        "weight_cache": str(weight_cache),
        **provenance(),
    }
    start_run(out, manifest)

    ffn = TtMoEBlockFfn(
        mesh_device, KimiK31Config, layer, cache_path=weight_cache, options=MoEBlockFfnOptions(sp_axis=layout.sp_axis)
    )
    dims = [None, None]
    dims[layout.sp_axis], dims[layout.tp_axis] = -2, -1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=tuple(dims))
    num_chunks = -(-count // chunk)
    began = time.perf_counter()
    try:
        for c in reversed(range(num_chunks)) if REVERSE else range(num_chunks):
            lo, hi = c * chunk, min((c + 1) * chunk, count)
            rows = torch.zeros(chunk, KimiK31Config.EMB_SIZE, dtype=torch.bfloat16)
            rows[: hi - lo] = load_input(source, lo, hi)
            x = ttnn.from_torch(
                rows.reshape(1, 1, chunk, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            y = ffn(x, actual_isl=hi - lo, actual_start=lo)  # (the runtime passes every chunk's real span)
            ttnn.deallocate(x)
            block = ffn._block(chunk)  # (the call's block: one per (mesh, chunk), cached - no new one is built)
            ids, weights = device_routing(ffn, block, hi - lo)
            moe_out = compose(mesh_device, y, hi - lo).to(torch.bfloat16)
            ttnn.deallocate(y)
            assert torch.isfinite(moe_out).all(), f"non-finite MoE output rows in chunk {c}"
            distinct = torch.sort(ids, dim=1).values.diff(dim=1) > 0
            assert distinct.all() and torch.isfinite(weights.float()).all(), f"broken device routing in chunk {c}"
            torch.save(
                {
                    "start": lo,
                    "end": hi,
                    "moe_out": moe_out,
                    "expert_ids": ids.to(torch.int16),
                    "expert_weights": weights,
                },
                chunk_path(out, c),
            )
            logger.info(
                f"MOE_TF layer {layer} chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) done, "
                f"{time.perf_counter() - began:.1f} s"
            )
    finally:
        release_blocks(mesh_device)
    seconds = time.perf_counter() - began
    finish_run(out, manifest, seconds)
    logger.info(f"MOE_TF done: layer {layer}, {num_chunks} chunks, {count} rows, {seconds:.1f} s -> {out}")
