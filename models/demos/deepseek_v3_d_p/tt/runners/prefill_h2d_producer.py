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
  * Reads the input token IDs from this variant's golden trace metadata.json (same source as the
    runner's standalone loop), pushing the first PREFILL_STANDALONE_NCHUNKS chunks.
  * Replays the same host-side block-cyclic reshape the runner uses. Kept inline so this
    script does NOT import the runner module (which would pull in `_migration`).
  * Pushes the byte buffer once per chunk via `forward_to_tensor_bytes`. The runner's per-iter
    `h2d_socket_sync` unblocks on each push.

Env vars:
  PREFILL_H2D_SERVICE_ID       — must match runner's value (default: "ds_prefill")
  PREFILL_MODEL_VARIANT        — selects the variant's prefill_trace_default (default: deepseek_v3_d_p)
  PREFILL_TRACE_DIR   — golden trace dir (overrides the variant default; must match the runner)
  PREFILL_STANDALONE_NCHUNKS   — chunks to push (default 11 -> 56320 tokens)
  PREFILL_NUM_USERS            — cache user slots (default 2); the requested slot wraps mod this
  PREFILL_STANDALONE_SLOT      — cache user slot (default 0, wrapped mod PREFILL_NUM_USERS)
  PREFILL_STANDALONE_ITERS     — times to replay the full chunk stream (default 1)
  PREFILL_H2D_CONNECT_TIMEOUT  — seconds to wait for the descriptor file (default: 60)
  PREFILL_SP / PREFILL_TP / PREFILL_MAX_SEQ_LEN / PREFILL_CHUNK_SIZE
    — must match the runner so token packing produces the same byte layout.

Usage (two terminals):
    # terminal A — runner (creates service + exports descriptor + waits):
    PREFILL_STANDALONE=1 PREFILL_H2D_EXTERNAL_PRODUCER=1 \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner

    # terminal B — producer (pushes tokens):
    PREFILL_STANDALONE_ITERS=1 \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer
"""

import os
import struct
import time

import torch
from loguru import logger

import ttnn

# runner_utils is migration-free (no _migration pull), so importing the variant + trace helpers here
# does not violate this producer's "don't import the runner module" constraint.
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import get_variant, load_trace_token_ids, resolve_trace_dir

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

VARIANT = get_variant(os.environ.get("PREFILL_MODEL_VARIANT", "deepseek_v3_d_p"))


def _load_tokens():
    """Return (task_id, slot_id, actual_isl, token_ids) — the input token_ids from this variant's
    golden trace metadata.json (PREFILL_TRACE_DIR overrides the variant default). Loads the
    first NCHUNKS*CHUNK_SIZE tokens (chunk-aligned, all real). Pairs with the runner's request-loop
    PCC check against the same trace's golden KV."""
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
    n_chunks = int(os.environ.get("PREFILL_STANDALONE_NCHUNKS", "11"))
    slot_id = int(os.environ.get("PREFILL_STANDALONE_SLOT", "0")) % NUM_USERS
    total_len = n_chunks * CHUNK_SIZE
    logger.info(f"[producer] reading {total_len} tokens ({n_chunks} chunks) from trace {trace_dir}")
    token_ids = load_trace_token_ids(trace_dir, total_len)
    assert (
        len(token_ids) == total_len
    ), f"trace has {len(token_ids)} tokens but need {total_len}; lower PREFILL_STANDALONE_NCHUNKS"
    return 0, slot_id, total_len, token_ids


def _chunk_to_host_array(chunk_token_ids: list[int]):
    """Build the un-sharded per-chunk token buffer for `service.forward_to_tensor_bytes`.

    Returns a contiguous CPU uint32 ndarray of shape [sp_factor, 1, chunk_local]
    (chunk_local = CHUNK_SIZE // sp_factor) in the runner's global ROW_MAJOR layout. The
    connected service splits it across SP coords via its own mapper (rebuilt device-lessly
    from the descriptor on `connect()`), so this process needs neither a mapper nor a
    MeshDevice. Chunked prefill is block-cyclic, so the chip-major reshape matches the runner's
    prepare_prefill_input_tensor (is_balanced=False) — replicated inline so this producer doesn't
    import the runner module.
    """
    sp_factor = GLOBAL_MESH_SHAPE[0]
    assert len(chunk_token_ids) == CHUNK_SIZE, f"chunk must be CHUNK_SIZE={CHUNK_SIZE}, got {len(chunk_token_ids)}"
    chunk_local = CHUNK_SIZE // sp_factor

    token_ids_sharded = torch.tensor(chunk_token_ids, dtype=torch.int64).reshape(sp_factor, 1, chunk_local)
    return token_ids_sharded.to(torch.uint32).contiguous().numpy()


def main() -> None:
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))

    logger.info(
        f"[producer] service_id={service_id!r} timeout={timeout_s}s "
        f"mesh={GLOBAL_MESH_SHAPE} max_seq_len={MAX_SEQ_LEN} iters={num_iterations}"
    )

    logger.info(f"[producer] connecting to /dev/shm/tt_h2d_stream_service_{service_id}.bin")
    t0 = time.perf_counter()
    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    logger.info(f"[producer] attached in {(time.perf_counter() - t0):.2f}s")

    task_id, slot_id, actual_isl, token_ids = _load_tokens()  # actual_isl captured BEFORE padding

    if len(token_ids) > MAX_SEQ_LEN:
        raise ValueError(
            f"task_id={task_id} prompt has {len(token_ids)} tokens but MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            f"Set PREFILL_MAX_SEQ_LEN to match the runner."
        )
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
