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
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 3200 * _sp))
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "1") == "1"


def _tokens_to_host_array(token_ids: list[int]):
    """Build the un-sharded global token buffer for `service.forward_to_tensor_bytes`.

    Returns a contiguous CPU uint32 ndarray of shape [sp_factor, 1, isl_per_chip]
    in the runner's global ROW_MAJOR layout. The connected service splits it
    across SP coords via its own mapper (rebuilt device-lessly from the
    descriptor on `connect()`), so this process needs neither a mapper nor a
    MeshDevice. Identical reorder logic to the runner — replicated inline so
    this producer doesn't import the runner module (which pulls in `_migration`).
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
    actual_isl = len(token_ids)  # captured BEFORE padding — runner uses this for MLA
    if len(token_ids) < MAX_SEQ_LEN:
        token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))
    # Pack per-iter control bytes. Single-chunk demo: whole prompt on slot 0
    # starting at KV pos 0, so [actual_start, actual_end) = [0, actual_isl).
    # When wired into the inference server these come from the request envelope.
    metadata = _pack_metadata(slot_id=0, actual_start=0, actual_end=actual_isl)
    assert len(metadata) == _METADATA_SIZE_BYTES, f"metadata pack expected {_METADATA_SIZE_BYTES}B, got {len(metadata)}"
    logger.info(f"[producer] metadata: slot_id=0 actual_start=0 actual_end={actual_isl} ({len(metadata)}B)")

    # Bytes path: this process has no MeshDevice. We hand the un-sharded global
    # buffer to the service, which distributes it across SP coords via its own
    # mapper (rebuilt from the descriptor's mapper_config on connect()). No
    # host-side mapper / ttnn.from_torch needed.
    host_tokens = _tokens_to_host_array(token_ids)
    expected = service.payload_size_bytes()
    assert host_tokens.nbytes == expected, f"payload {host_tokens.nbytes}B != service-expected {expected}B"
    logger.info(
        f"[producer] host buffer ready: shape={host_tokens.shape} dtype={host_tokens.dtype} ({host_tokens.nbytes}B)"
    )

    push_times_ms: list[float] = []
    for i in range(num_iterations):
        # Log BEFORE the push so a blocked call (backpressure when the FIFO
        # is full) is visible from the log alone — pre-line lands, post-line
        # delays until forward_to_tensor returns.
        logger.info(f"[producer] starting push iter={i}/{num_iterations} task_id={task_id}")
        t = time.perf_counter()
        service.forward_to_tensor_bytes(host_tokens, metadata=metadata)
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
