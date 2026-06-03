#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""External producer for the H2D-streamed prefill_runner.

Pairs with `prefill_runner.py` running in `PREFILL_H2D_EXTERNAL_PRODUCER=1`
mode. The runner constructs an `H2DStreamService`, calls
`export_descriptor(service_id)` to drop a flatbuffer at
`/dev/shm/tt_h2d_stream_service_<service_id>.bin`, then loops on
`h2d_socket_sync` (blocks on `data_ready_sem`).

This producer:
  * Attaches to the same service via `ttnn.H2DStreamService.connect(service_id)`
    — no `MeshDevice` handle needed.
  * Reads `standalone_input.json` for the token IDs.
  * Replays the same host-side reorder + reshape the runner uses (`is_balanced`
    chunk order). Kept inline so this script does NOT import the runner module
    (which would pull in `_migration` via `migration_setup`).
  * Pushes the byte buffer `PREFILL_STANDALONE_ITERS` times via
    `forward_to_tensor_bytes`. The runner's per-iter `h2d_socket_sync`
    unblocks on each push.

Env vars:
  PREFILL_H2D_SERVICE_ID       — must match runner's value (default: "ds_prefill")
  PREFILL_STANDALONE_INPUT     — path to standalone JSON (default: next to this file)
  PREFILL_STANDALONE_ITERS     — # of pushes to send (default: 1)
  PREFILL_H2D_CONNECT_TIMEOUT  — seconds to wait for the descriptor file (default: 60)
  PREFILL_SP / PREFILL_TP / PREFILL_MAX_SEQ_LEN / PREFILL_IS_BALANCED
    — must match the runner so token packing produces the same byte layout.

Usage (two terminals):
    # terminal A — runner (creates service + exports descriptor + waits):
    PREFILL_STANDALONE=1 PREFILL_H2D_EXTERNAL_PRODUCER=1 \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner

    # terminal B — producer (pushes tokens):
    PREFILL_STANDALONE_ITERS=1 \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer
"""

import json
import os
import struct
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks

# Per-iter control metadata payload — matches the prefill scheduler's
# PrefillMetadata wire struct (12 bytes, 3 × uint32). Must equal the
# runner's H2D_METADATA_SIZE_BYTES. Source-of-truth definition lives in
# include/tt_llm_engine/scheduler/prefill/prefill_metadata.hpp.
_METADATA_SIZE_BYTES = 12


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
    """3 little-endian uint32s — [slot_id, actual_start, actual_end].

    Matches the scheduler's PrefillMetadata layout: slot_id selects the
    per-slot KV-cache buffer; [actual_start, actual_end) is the absolute
    KV-position range of the real (non-pad) tokens in the chunk.
    """
    return struct.pack("<III", slot_id, actual_start, actual_end)


_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 3200 * _sp))
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "1") == "1"

# Must match the runner's H2D_MAPPER_CONFIG so per-shard slices line up with
# the service's `get_per_shard_spec()`.
_H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
)


def _make_global_spec() -> ttnn.TensorSpec:
    """Per-iter global tensor spec — same shape/dtype/layout the runner uses."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    isl_per_chip = MAX_SEQ_LEN // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _tokens_to_host_tensor(token_ids: list[int], mapper) -> ttnn.Tensor:
    """Build the pre-distributed host tensor for `service.forward_to_tensor`.

    Identical reorder logic to the runner — replicated inline so this producer
    doesn't import the runner module (which pulls in `_migration`).
    """
    sp_factor = GLOBAL_MESH_SHAPE[0]
    assert len(token_ids) == MAX_SEQ_LEN, f"token_ids must be padded to MAX_SEQ_LEN={MAX_SEQ_LEN}, got {len(token_ids)}"
    isl_per_chip = MAX_SEQ_LEN // sp_factor

    if IS_BALANCED:
        chunk_order = create_balanced_chunk_order(sp_factor)
        t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)

    return ttnn.from_torch(
        token_ids_sharded.to(torch.int32),
        spec=_make_global_spec(),
        mesh_mapper=mapper,
    )


def main() -> None:
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))

    default_input = Path(__file__).parent / "standalone_input.json"
    input_path = Path(os.environ.get("PREFILL_STANDALONE_INPUT", default_input))

    logger.info(
        f"[producer] service_id={service_id!r} timeout={timeout_s}s "
        f"mesh={GLOBAL_MESH_SHAPE} max_seq_len={MAX_SEQ_LEN} iters={num_iterations}"
    )

    logger.info(f"[producer] connecting to /dev/shm/tt_h2d_stream_service_{service_id}.bin")
    t0 = time.perf_counter()
    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    logger.info(f"[producer] attached in {(time.perf_counter() - t0):.2f}s")

    logger.info(f"[producer] reading input from {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    task_id = data["task_id"]
    token_ids = list(data["token_ids"])

    if len(token_ids) > MAX_SEQ_LEN:
        raise ValueError(
            f"task_id={task_id} prompt has {len(token_ids)} tokens but MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            f"Set PREFILL_MAX_SEQ_LEN to match the runner."
        )
    actual_isl = len(token_ids)  # captured BEFORE padding — runner derives this from actual_end - actual_start
    if len(token_ids) < MAX_SEQ_LEN:
        token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))
    # Single-chunk-per-prefill mode: pack the whole prompt as one chunk
    # starting at KV position 0 on slot 0. Matches the scheduler's
    # PrefillMetadata wire format. Multi-chunk / multi-slot is what the
    # real prefill scheduler will drive; this demo producer just emits
    # the minimal single-chunk envelope.
    metadata = _pack_metadata(slot_id=0, actual_start=0, actual_end=actual_isl)
    assert len(metadata) == _METADATA_SIZE_BYTES, f"metadata pack expected {_METADATA_SIZE_BYTES}B, got {len(metadata)}"
    logger.info(f"[producer] metadata: slot_id=0 actual_start=0 actual_end={actual_isl} ({len(metadata)}B)")

    # Shape-only mapper: this process has no MeshDevice. The mapper sits on
    # host only — used by ttnn.from_torch to produce the per-shard host
    # tensor that forward_to_tensor streams directly to each coord.
    mapper = ttnn.create_mesh_mapper(ttnn.MeshShape(*GLOBAL_MESH_SHAPE), _H2D_MAPPER_CONFIG)
    host_tokens = _tokens_to_host_tensor(token_ids, mapper)
    logger.info(f"[producer] host tensor ready: shape={host_tokens.shape} dtype={host_tokens.dtype}")

    push_times_ms: list[float] = []
    for i in range(num_iterations):
        # Log BEFORE the push so a blocked call (backpressure when the FIFO
        # is full) is visible from the log alone — pre-line lands, post-line
        # delays until forward_to_tensor returns.
        logger.info(f"[producer] starting push iter={i}/{num_iterations} task_id={task_id}")
        t = time.perf_counter()
        service.forward_to_tensor(host_tokens, metadata=metadata)
        dt_ms = (time.perf_counter() - t) * 1000.0
        push_times_ms.append(dt_ms)
        logger.info(f"[producer]   push iter={i} returned in {dt_ms:.2f} ms")

    # Final barrier so the descriptor isn't released before the last push has
    # been drained by the service core. (`connect`-side `barrier` is supported.)
    service.barrier()
    logger.info(
        f"[producer] done. per-push ms = {[round(t, 2) for t in push_times_ms]}; "
        "exiting (the runner keeps its sync-op loop running)."
    )


if __name__ == "__main__":
    main()
