# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How fast is Kimi-K3 prefill at 1, 5, 12 and 24 layers?

Two numbers per depth, because on this mesh they answer different questions.

**Device kernel time** is the sum over programs of that program's critical path across the 32 chips.
The real-time profiler reports one record per program per chip, and the chips run a program
concurrently, so the program costs what its slowest chip costs; programs then execute in sequence,
so the sum is the time the device spends inside kernels. It excludes the dispatch gaps between
programs, which is exactly what makes it the right number to compare against another model's kernels
and the wrong number to quote as throughput.

**Eager wall-clock** is a synchronized multi-iteration measurement of `forward` as the tests
actually call it, host dispatch included. The gap between the two is the host overhead that trace
capture removes, so reporting both says how much there is to win from tracing rather than leaving it
implicit.

Both are reported per token as well, since a depth comparison in milliseconds says little when the
layer counts differ by 24x.

The weight cache matters here more than anywhere: without it each depth spends over an hour reading
routed experts before the first token moves, and a perf sweep re-reads them at every rung.
"""

import time
from collections import defaultdict
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import (
    PLACEMENTS,
    SEQ_LEN,
    SP_AXIS,
    TP_AXIS,
    _model_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

DEPTHS = [1, 5, 12, 24]
# 1 chunk is 5120 tokens; 11 chunks is 56320 — the "55k" leg. The second is not just a longer run:
# MLA attends over every token cached so far, so chunk N costs more than chunk N-1, and the
# per-chunk curve is the only thing that shows how prefill scales with context.
CHUNK_COUNTS = [1, 11]
ITERATIONS = 5


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
@pytest.mark.parametrize("num_chunks", CHUNK_COUNTS, ids=[f"{n}chunk" for n in CHUNK_COUNTS])
def test_prefill_cost(mesh_device, device_params, num_layers, num_chunks):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    total_len = SEQ_LEN * num_chunks
    config = kimi_k3_hf_config(max_seq=total_len)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        # Single-rank test: nothing is inherited, so the second argument is always None.
        return TtAttnResResidual(
            TtAttnResWalk(
                attn_res,
                hidden,
                list(attn_res.weights.pre),
                list(attn_res.weights.post),
                attn_res.weights.output,
                num_layers,
            )
        )

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        _model_state_dict(checkpoint, num_layers, root, cache),
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=total_len,
        # `actual_start` is only accepted by a model built for chunking, and ttMLA asserts the pair
        # match. A single-chunk run passes neither.
        is_chunked=num_chunks > 1,
        weight_cache_path=cache,
    )

    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=total_len,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
        )
    chunks = [
        prepare_prefill_input_tensor(
            trace.token_ids(SEQ_LEN, SEQ_LEN * c)[0].tolist(),
            mesh_device,
            tuple(mesh_device.shape)[SP_AXIS],
            False,
            tuple(mesh_device.shape),
            SP_AXIS,
        )
        for c in range(num_chunks)
    ]

    def once():
        """One whole prefill: every chunk, in order, as a request would arrive."""
        model.reset_streams()
        for index, tokens in enumerate(chunks):
            start = index * SEQ_LEN if num_chunks > 1 else None
            out = model.forward(tokens, kvpe_cache=kvpe, actual_start=start)
            if out is not None:
                ttnn.deallocate(out)

    try:
        for _ in range(3):  # warm the program cache; the first pass compiles
            once()
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        for _ in range(ITERATIONS):
            once()
        ttnn.synchronize_device(mesh_device)
        eager_ms = (time.perf_counter() - start) / ITERATIONS * 1e3

        device_ms = None
        try:
            _, records = profile_realtime_program(mesh_device, once, collect_all=True)
            # One record per (program, chip). The chips run a program concurrently, so the program
            # costs its slowest chip; programs then run in sequence, so the sum is device kernel time.
            critical_path = defaultdict(float)
            for record in records:
                critical_path[record["runtime_id"]] = max(critical_path[record["runtime_id"]], record["duration_ns"])
            device_ms = sum(critical_path.values()) / 1e6
            programs = len(critical_path)
        except RuntimeError as error:
            # Dropping records makes the set partial, and a partial sum under-reports. Say so rather
            # than quote a number that looks like a measurement.
            logger.warning(f"  L{num_layers}: device kernel time unavailable — {error}")
            programs = 0
        # Per-chunk device time, when there is more than one chunk. The total divided by the count is an
        # average, and the average hides the thing worth knowing: MLA attends over every token cached so
        # far, so chunk N should cost more than chunk N-1 while the KDA layers stay flat. Profiling each
        # chunk separately is the only way to see that curve.
        if num_chunks > 1:
            model.reset_streams()
            per_chunk = []
            for index, tokens in enumerate(chunks):

                def one(_tokens=tokens, _index=index):
                    out = model.forward(_tokens, kvpe_cache=kvpe, actual_start=_index * SEQ_LEN)
                    if out is not None:
                        ttnn.deallocate(out)

                try:
                    _, records = profile_realtime_program(mesh_device, one, collect_all=True)
                    path = defaultdict(float)
                    for record in records:
                        path[record["runtime_id"]] = max(path[record["runtime_id"]], record["duration_ns"])
                    per_chunk.append(sum(path.values()) / 1e6)
                except RuntimeError:
                    per_chunk.append(float("nan"))
            logger.info(
                f"PERF L{num_layers:2d} per-chunk device ms: "
                + " ".join(f"{c:.1f}" for c in per_chunk)
                + f"  (first {per_chunk[0]:.1f}, last {per_chunk[-1]:.1f}, "
                + f"growth {per_chunk[-1] - per_chunk[0]:+.1f} ms over {num_chunks} chunks)"
            )

    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    logger.info(
        f"PERF L{num_layers:2d} x {num_chunks:2d}chunk ({total_len:6d} tok): eager {eager_ms:9.2f} ms "
        f"({total_len / eager_ms * 1e3:8.0f} tok/s, {eager_ms / num_layers:7.2f} ms/layer)"
    )
    if device_ms is not None:
        logger.info(
            f"PERF L{num_layers:2d} x {num_chunks:2d}chunk ({total_len:6d} tok): device {device_ms:9.2f} ms "
            f"over {programs} programs ({total_len / device_ms * 1e3:8.0f} tok/s, "
            f"{device_ms / num_layers:7.2f} ms/layer) -- host {eager_ms - device_ms:7.2f} ms "
            f"({(1 - device_ms / eager_ms) * 100:3.0f}%)"
        )
