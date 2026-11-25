# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context accuracy of the production Kimi-K3.1 MoE (the MoE block) on layers 1 and 2, teacher-fed from the FP64
chain, every row scored against it.

The golden is the registered FP64 CPU chain of K3.1 layers 0-3 over the vLLM K3.1-Pro 1M prompt
(``k31_chain_golden.K31_CHAIN_1M``); a missing or incomplete golden fails the test. Each 2560-row chunk of the prompt,
in order, feeds the chain's MoE input ``moe_input_L`` (after ``post_attention_layernorm``, rounded once to bf16) to
``TtMoEBlockFfn`` as one 4x4 stage (SP on axis 1) builds it, with the chunk's real span as the runtime passes it, and
its routed experts from the stage's TTNN weight cache. The device's MoE output and routing are scored against the
chain's ``moe_output_L`` and FP64 top-16 (``moe_long_context``): absolute and relative error per window and over the
run, routing recall, dropped FP64 weight and miss ranks, gated per window by ``moe_long_context.BARS``. The MoE has no
state across chunks, so there is no context-growth bar; the rungs show that the bars hold at every context length.

Run on one tray half of a Blackhole Galaxy (skips without the 4x4 binding or the weight cache):

    models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- \\
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_moe_long_context.py::test_k31_moe_long_context[blackhole-5k-layer1-4x4-sp1]"
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import RowSeries, check, trend
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.k31_chain_golden import K31_CHAIN_1M, open_k31_chain, read_rows
from models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_long_context import (
    BARS,
    TOP_K,
    append_chunk,
    device_routing,
    measure,
    row_floor,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import compose

WINDOW = 16384  # rows per scored window, aligned to row 0 (a shorter rung is one partial window)
# Context rungs in rows, the last the golden's whole prompt (408 chunks); 256k ends in a partial chunk.
RUNGS = {"5k": 5120, "20k": 20480, "256k": 262144, "1M": 1044480}
PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]

# Measured on one tray half: 1M rows ~4 min a layer (~50 s weight load, then ~0.45 s a chunk with the golden read
# overlapped); 256k ~100 s, 5k-20k ~15 s warm. pytest.ini's 300 s default leaves 1M too little margin.
pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only"),
    pytest.mark.timeout(900),
]


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
@pytest.mark.parametrize("layer", [1, 2], ids=["layer1", "layer2"])
@pytest.mark.parametrize("rows", list(RUNGS.values()), ids=list(RUNGS))
def test_k31_moe_long_context(mesh_device, device_params, layer: int, rows: int) -> None:
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import (
        MoEBlockFfnOptions,
        TtMoEBlockFfn,
        release_blocks,
    )
    from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_impl_of

    began = time.perf_counter()
    golden = open_k31_chain(K31_CHAIN_1M)  # (before any device work: a missing golden fails here)
    assert rows <= golden.rows, f"golden {golden.golden_id} has {golden.rows} rows, fewer than {rows}"
    mesh_shape = tuple(mesh_device.shape)
    layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
    chunk = layout.chunk
    assert moe_impl_of(k31_stage.ADAPTER) == "block", "the production K3.1 MoE is the MoE block"
    names = [f"moe_input_{layer}", f"moe_output_{layer}", f"routing_ids_{layer}", f"routing_weights_{layer}"]

    ffn = TtMoEBlockFfn(
        mesh_device,
        KimiK31Config,
        layer,
        cache_path=k31_stage.weight_cache_path(mesh_shape),
        options=MoEBlockFfnOptions(sp_axis=layout.sp_axis),
    )
    dims = [None, None]
    dims[layout.sp_axis], dims[layout.tp_axis] = -2, -1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=tuple(dims))
    series = RowSeries()
    num_chunks = -(-rows // chunk)  # (a partial last chunk is zero-padded, its real span passed, as in production)

    def span(c: int) -> tuple[int, int]:
        return c * chunk, min((c + 1) * chunk, rows)

    first_done = None  # (the first call also loads the layer's weights from the cache)
    with ThreadPoolExecutor(max_workers=1) as reader:  # (the next chunk's golden is read while the device runs)
        pending = reader.submit(read_rows, golden, names, *span(0))
        try:
            for c in range(num_chunks):
                lo, hi = span(c)
                want = pending.result()
                if c + 1 < num_chunks:
                    pending = reader.submit(read_rows, golden, names, *span(c + 1))
                x_rows = torch.zeros(chunk, KimiK31Config.EMB_SIZE, dtype=torch.bfloat16)
                x_rows[: hi - lo] = want[names[0]]
                x = ttnn.from_torch(
                    x_rows.reshape(1, 1, chunk, -1),
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                y = ffn(x, actual_isl=hi - lo, actual_start=lo)  # (the runtime passes every chunk's real span)
                ttnn.deallocate(x)
                ids, weights = device_routing(ffn, ffn._block(chunk), hi - lo)  # (the call's cached block)
                output = compose(mesh_device, y, hi - lo)
                ttnn.deallocate(y)
                assert torch.isfinite(output).all(), f"non-finite MoE output in rows [{lo}, {hi})"
                distinct = torch.sort(ids, dim=1).values.diff(dim=1) > 0
                assert distinct.all(), f"a row routed to one expert twice in rows [{lo}, {hi})"
                append_chunk(
                    series,
                    lo,
                    output=output,
                    ids=ids,
                    weights=weights,
                    want_output=want[names[1]],
                    want_ids=want[names[2]],
                    want_weights=want[names[3]],
                )
                first_done = first_done or time.perf_counter()
                if (c + 1) % 40 == 0 or c + 1 == num_chunks:
                    logger.info(
                        f"K31_MOE_LC layer {layer}: chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) scored, "
                        f"{time.perf_counter() - began:.1f} s"
                    )
        finally:
            pending.cancel()
            release_blocks(mesh_device)
    loop_done = time.perf_counter()

    of = measure(row_floor(series))
    windows = series.windows(WINDOW, of)
    whole = series.whole(
        lambda w: {**of(w), "miss_rank_histogram": torch.bincount(w["misses"]["rank"], minlength=TOP_K).tolist()}
    )
    result = {
        "golden_id": golden.golden_id,
        "layer": layer,
        "rows": rows,
        "chunk": chunk,
        "window": WINDOW,
        "seconds": {
            "to_first_chunk": round(first_done - began, 1),  # (golden open, weight load, first chunk)
            "other_chunks": round(loop_done - first_done, 1),
            "total": round(time.perf_counter() - began, 1),
        },
        "whole": whole,
        "trend": {
            metric: trend(windows, metric, worse)
            for metric, worse in (
                ("moe_rms_err", "max"),
                ("moe_rms_ref", "max"),
                ("moe_nrmse", "max"),
                ("moe_row_nrmse_median", "max"),
                ("recall_mean", "min"),
                ("dropped_weight_mean", "max"),
            )
        },
    }
    for w in windows:
        logger.info(
            "K31_MOE_LC_WINDOW " + json.dumps({k: round(v, 7) if isinstance(v, float) else v for k, v in w.items()})
        )
    failures = check(windows, BARS)
    result["bar_failures"] = failures
    print("K31_MOE_LONG_CONTEXT=" + json.dumps(result, sort_keys=True))
    assert not failures, failures
