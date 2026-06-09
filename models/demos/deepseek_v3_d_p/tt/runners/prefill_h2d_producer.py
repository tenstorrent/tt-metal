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

# Per-iter control metadata payload: 3 × uint32 = 12 bytes. Must match the
# runner's H2D_METADATA_SIZE_BYTES and field order in prefill_runner.py, and
# the scheduler's PrefillMetadata wire struct.
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
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 60 * 1024))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
# Chunked / indexed-RoPE prefill is non-balanced (block-cyclic).
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "0") == "1"


def _chunk_to_host_array(chunk_token_ids: list[int]):
    """Build the un-sharded per-chunk token buffer for `service.forward_to_tensor_bytes`.

    Returns a contiguous CPU uint32 ndarray of shape [sp_factor, 1, chunk_local]
    (chunk_local = CHUNK_SIZE // sp_factor) in the runner's global ROW_MAJOR layout. The
    connected service splits it across SP coords via its own mapper (rebuilt device-lessly
    from the descriptor on `connect()`), so this process needs neither a mapper nor a
    MeshDevice. Identical reorder logic to the runner's prepare_prefill_input_tensor —
    replicated inline so this producer doesn't import the runner module.
    """
    sp_factor = GLOBAL_MESH_SHAPE[0]
    assert len(chunk_token_ids) == CHUNK_SIZE, f"chunk must be CHUNK_SIZE={CHUNK_SIZE}, got {len(chunk_token_ids)}"
    chunk_local = CHUNK_SIZE // sp_factor

    if IS_BALANCED:
        chunk_order = create_balanced_chunk_order(sp_factor)
        t = torch.tensor(chunk_token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, chunk_local)
    else:
        token_ids_sharded = torch.tensor(chunk_token_ids, dtype=torch.int64).reshape(sp_factor, 1, chunk_local)

    return token_ids_sharded.to(torch.uint32).contiguous().numpy()


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
    actual_isl = len(token_ids)  # captured BEFORE padding
    slot_id = int(data.get("slot_id", 0)) % NUM_USERS
    # Chunked prefill streams ONE chunk per push. Pad the prompt up to a chunk boundary, then
    # push each CHUNK_SIZE-wide chunk with per-chunk PrefillMetadata: actual_start is the chunk's
    # absolute KV start (= c * CHUNK_SIZE for chunk-aligned demo), actual_end clamps to actual_isl
    # so trailing pad positions in the last chunk are reported as such. When wired into the
    # inference server these come from the request envelope / scheduler.
    pad_to = ((actual_isl + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
    if len(token_ids) < pad_to:
        token_ids = token_ids + [1] * (pad_to - len(token_ids))
    n_chunks = pad_to // CHUNK_SIZE
    expected = service.payload_size_bytes()
    logger.info(
        f"[producer] task_id={task_id} slot_id={slot_id} actual_isl={actual_isl} "
        f"n_chunks={n_chunks} chunk_size={CHUNK_SIZE} iters={num_iterations}"
    )

    push_times_ms: list[float] = []
    for i in range(num_iterations):
        for c in range(n_chunks):
            chunk_start = c * CHUNK_SIZE
            actual_start = chunk_start
            actual_end = min(chunk_start + CHUNK_SIZE, actual_isl)
            metadata = _pack_metadata(slot_id=slot_id, actual_start=actual_start, actual_end=actual_end)
            assert len(metadata) == _METADATA_SIZE_BYTES, f"metadata {len(metadata)}B != {_METADATA_SIZE_BYTES}B"

            # Bytes path: no MeshDevice here. The service distributes the un-sharded per-chunk
            # buffer across SP coords via its descriptor-rebuilt mapper.
            host_tokens = _chunk_to_host_array(token_ids[chunk_start : chunk_start + CHUNK_SIZE])
            assert host_tokens.nbytes == expected, f"payload {host_tokens.nbytes}B != service-expected {expected}B"

            # Log BEFORE the push so backpressure (full FIFO) is visible from the log alone.
            logger.info(
                f"[producer] push iter={i} chunk={c}/{n_chunks} task_id={task_id} slot={slot_id} "
                f"actual_start={actual_start} actual_end={actual_end}"
            )
            t = time.perf_counter()
            service.forward_to_tensor_bytes(host_tokens, metadata=metadata)
            dt_ms = (time.perf_counter() - t) * 1000.0
            push_times_ms.append(dt_ms)
            logger.info(f"[producer]   push iter={i} chunk={c} returned in {dt_ms:.2f} ms")

    # Final barrier so the descriptor isn't released before the last push has
    # been drained by the service core. (`connect`-side `barrier` is supported.)
    service.barrier()
    logger.info(
        f"[producer] done. {len(push_times_ms)} pushes, per-push ms = {[round(t, 2) for t in push_times_ms]}; "
        "exiting (the runner keeps its sync-op loop running)."
    )


if __name__ == "__main__":
    main()
