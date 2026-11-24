# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Token feeder for a timed stage probe: the prefill producer's feed and layer-ack drain, without PCC.

A stage probe runs one pipeline stage alone on one rank (the runner's PREFILL_PP_LAYER_OFFSET). That path builds
no KV chunk table, so the stock producer can run it only with PREFILL_PRODUCER_CHECK_PCC=0, and then it does not
consume the layer-completion channel: no CHUNK_RETIRE lines, no stage times. This feeder pushes the same request
(warmup chunks, then the configured request, as the stock producer schedules them), drains the acks into the
stock _ChunkRetireClock, which logs one CHUNK_RETIRE line per chunk with per-layer ack times, and shuts the runner
down. Configuration is the stock producer's: PREFILL_PRODUCER_MANIFEST and PREFILL_* env vars.

CI's perf-only legs (ci/run_multirank_pcc.sh, e.g. GLM-5.4 on SC4) run it the same way against a multi-rank runner: a
looped model has no KV table either, and the master rank's completion ring carries every rank's acks.
"""

import os
import sys
import time

from loguru import logger

import models.demos.common.prefill.runners.prefill_producer as P
from models.demos.common.prefill.runners.layer_completion_ring_reader import (
    connect_layer_completion_channel,
    start_layer_completion_drain,
    wait_for_layer_completions,
)


def main() -> int:
    manifest = os.environ.get("PREFILL_PRODUCER_MANIFEST")
    if manifest:
        P._apply_manifest_env(manifest)
    P._load_env_config()
    cfg = P._config_from_env()
    if cfg.verify:
        logger.error("[probe-producer] PREFILL_PRODUCER_CHECK_PCC=1: a stage probe has no KV table to verify")
        return 2
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    service = P.ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    payload_bytes = service.payload_size_bytes()
    channel = connect_layer_completion_channel(timeout_s)
    if channel is None:
        logger.error("[probe-producer] no layer-completion channel: the runner is not acking layers")
        return 1
    chunk_layers = P._completion_layers_per_chunk(None)
    slot_traces, slot_lengths, pools_by_trace = P._resolve_slot_prompts(cfg)
    cfg.slot_lengths = slot_lengths
    logger.info(
        f"[probe-producer] attached service={service_id!r} payload={payload_bytes}B chunk={P.CHUNK_SIZE} "
        f"chunks={cfg.chunks_max} layers_per_chunk={chunk_layers}"
    )

    retire_clock = P._ChunkRetireClock()

    def push_forward(pushes: list) -> float:
        for p in pushes:
            retire_clock.on_push((p.slot_id, p.actual_start, p.actual_end))
        t0 = time.perf_counter()
        P._push(service, payload_bytes, *P._forward_payload(pushes, lambda s: pools_by_trace[slot_traces[s]]))
        return (time.perf_counter() - t0) * 1000.0

    drain = start_layer_completion_drain(
        channel,
        num_layers=P._completion_layer_bound(chunk_layers),
        chunk_layers=chunk_layers,
        on_completion=retire_clock.on_ack,
        on_chunk_complete=lambda key, cov: retire_clock.on_retire(key, time.perf_counter()),
    )
    try:
        warmup_chunks = int(os.environ.get("PREFILL_PRODUCER_WARMUP_CHUNKS", "0"))
        if warmup_chunks > 0:
            retire_clock.warmup = True
            warmup = [
                P.ChunkPush(0, c, c * P.CHUNK_SIZE, (c + 1) * P.CHUNK_SIZE, warmup_chunks * P.CHUNK_SIZE)
                for c in range(warmup_chunks)
            ]
            for i in range(0, warmup_chunks, P.FORWARD_CHUNKS):
                push_forward(warmup[i : i + P.FORWARD_CHUNKS])
            service.barrier()
            wait_for_layer_completions(drain, chunk_layers * warmup_chunks)
            retire_clock.warmup = False
            logger.info(f"[probe-producer] warmup done ({warmup_chunks} chunks); starting the measured request")
        stats = P.run_schedule(cfg, push_fn=push_forward)
        service.barrier()
        expected = chunk_layers * (max(warmup_chunks, 0) + stats.total_chunks)
        received = wait_for_layer_completions(drain, expected)
    finally:
        drain.stop()
    logger.info(
        f"[probe-producer] DONE wall={stats.wall_s:.1f}s chunks={stats.total_chunks} "
        f"completions={received}/{expected}"
    )
    rows, mtp_rows, sentinel = P._shutdown_payload()
    P._push(service, payload_bytes, rows, mtp_rows, sentinel)
    service.barrier()
    logger.info("[probe-producer] SHUTDOWN sentinel sent")
    return 0 if received == expected else 1


if __name__ == "__main__":
    sys.exit(main())
