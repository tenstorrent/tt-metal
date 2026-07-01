#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""External producer for the H2D-streamed prefill_runner.

Pairs with `prefill_runner.py` running in `PREFILL_H2D_EXTERNAL_PRODUCER=1`
mode. The runner constructs an `H2DStreamService`, calls
`export_descriptor(service_id)` to drop a flatbuffer at
`/dev/shm/tt_h2d_stream_service_<service_id>.bin`, then loops on
`inbound_socket_service_sync` (blocks on `data_ready_sem`).

This producer:
  * Attaches to the same service via `ttnn.H2DStreamService.connect(service_id)`
    — no `MeshDevice` handle needed.
  * Reads the input token IDs from this variant's golden trace metadata.json (same source as the
    runner's standalone loop), pushing the first PREFILL_STANDALONE_NCHUNKS chunks.
  * Replays the same host-side block-cyclic reshape the runner uses. Kept inline so this
    script does NOT import the runner module (which would pull in `_migration`).
  * Pushes the byte buffer once per chunk via `forward_to_tensor_bytes`. The runner's per-iter
    `inbound_socket_service_sync` unblocks on each push.

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
# Per-layer LayerAck cadence is NUM_LAYERS acks per chunk; must match the runner's PREFILL_NUM_LAYERS.
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))

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


# ---------------------------------------------------------------------------
# KV chunk table + LayerAck + PCC (mock integration, single galaxy / 1 rank)
# ---------------------------------------------------------------------------


def _read_kv_chunk_table(timeout_s: int):
    """Read (deserialize) the KV chunk address table the runner serialized for this galaxy.

    The runner publishes it (PREFILL_MOCK_MIGRATION=1 → build_and_serialize_kv_chunk_table, or the
    full migration path) to PREFILL_MIGRATION_TABLE_PATH. This is fully device-less:
    import_from_protobuf_file rebuilds the KvChunkAddressTable from the protobuf alone (no device /
    ControlPlane). One galaxy => one complete table spanning all layers/slots. Polls for the file
    since the runner writes it during setup, possibly just after exporting the H2D descriptor.

    Returns the table, or None (so the producer can still push input even if the table isn't there).
    """
    table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
    t0 = time.perf_counter()
    while not os.path.exists(table_path):
        if time.perf_counter() - t0 > timeout_s:
            logger.warning(
                f"[producer] KV chunk table {table_path} not found after {timeout_s}s; "
                f"run the runner with PREFILL_MOCK_MIGRATION=1 to publish it. Skipping table read."
            )
            return None
        time.sleep(0.1)
    import_fn = getattr(ttnn.experimental.disaggregation, "import_from_protobuf_file", None)
    if import_fn is None:
        logger.error(
            "[producer] ttnn.experimental.disaggregation.import_from_protobuf_file is missing — "
            "rebuild ttnn after adding the import binding (disaggregation.cpp). Skipping table read."
        )
        return None
    table = import_fn(table_path)
    cfg = table.config()
    logger.info(
        f"[producer] read KV chunk table {table_path}: entries={table.total_entries()} "
        f"num_layers={cfg.num_layers} num_slots={cfg.num_slots} max_seq_len={cfg.max_sequence_length} "
        f"chunk_n_tokens={cfg.chunk_n_tokens} chunk_size_bytes={cfg.chunk_size_bytes}"
    )
    return table


def _connect_layer_ack_channel(timeout_s: int):
    """Attach (consumer side) to the runner's per-layer LayerAck channel. The single-rank runner
    creates `/tt_prefill_layer_acks_<service_id>` and injects 1 per layer (NUM_LAYERS per chunk);
    this connects and drains the deltas. Returns the channel, or None if it isn't available."""
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm = f"/tt_prefill_layer_acks_{service_id}"
    try:
        ch = ttnn.InterProcessCounterChannel.connect(shm, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(
            f"[producer] could not connect LayerAck channel {shm}: {e} "
            f"(only the single-rank runner creates it). Skipping ack wait."
        )
        return None
    logger.info(f"[producer] connected LayerAck channel {shm}")
    return ch


def _drain_layer_acks(ack_channel, expected: int, timeout_s: float = 600.0) -> int:
    """Block until `expected` (= NUM_LAYERS * n_chunks) per-layer acks have been drained, or timeout.
    Non-blocking poll of try_consume_all(); returns the count actually drained."""
    if ack_channel is None:
        return 0
    got = 0
    last_logged = -1
    t0 = time.perf_counter()
    while got < expected:
        got += ack_channel.try_consume_all()
        if got != last_logged:
            logger.info(f"[producer] layer acks {got}/{expected}")
            last_logged = got
        if got >= expected:
            break
        if time.perf_counter() - t0 > timeout_s:
            logger.warning(f"[producer] timed out at {got}/{expected} acks after {timeout_s}s")
            break
        time.sleep(0.01)
    logger.info(f"[producer] drained {got}/{expected} layer acks in {(time.perf_counter() - t0):.2f}s")
    return got


def _attach_mesh_for_read():
    """Open the galaxy in THIS producer process so `read_device_chunk` resolves.

    read_device_chunk → resolve_device → GetActiveDevice requires the chips to be ACTIVE in this
    process (device_manager.is_device_active() == device.is_initialized()); a device-less H2D
    connect is not enough. CAVEAT: a plain open_mesh_device fully initializes (resets) the chips —
    on one galaxy a live runner already owns them, so the correct mechanism is the non-destructive
    `minimal` attach (device.cpp: "attaching to a hung chip"), which is not yet exposed to Python
    (MeshDevice::create hardcodes minimal=false). Until that lands, this opt-in path is for setups
    where the producer can legitimately open the galaxy. Returns the mesh device."""
    from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import open_mesh_device

    logger.warning(
        "[producer] PREFILL_PRODUCER_CHECK_PCC=1: opening mesh to read KV — needs a non-destructive "
        "'minimal' attach to coexist with a live runner (see _attach_mesh_for_read)."
    )
    return open_mesh_device(GLOBAL_MESH_SHAPE, VARIANT.model_config)


def _read_kv_and_check_pcc(table, slot_id: int, n_chunks: int) -> float:
    """Read each KV chunk back from the device via the table and PCC-check against the golden trace.

    Mirrors test_kimi_kv_cache_mock (per (layer, position, slot) read_device_chunk →
    tensor_from_bfp8_bytes) and kv_cache_pcc_check's golden handling (nope direct; pe re-interleaved
    HF→Meta). The kimi table maps natural positions → block-cyclic storage, so iterating natural
    positions yields natural order with no un-rotation. Requires the mesh open in this process."""
    from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import _load_golden_kv_post, resolve_trace_dir
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from tests.ttnn.utils_for_testing import comp_pcc

    KVPE_HEAD_DIM = 576  # qk_rope_head_dim(64) + kv_lora_rank(512); same for DeepSeek and Kimi
    KV_LORA = 512
    chunk_tok = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    chunk_shape = [1, 1, chunk_tok, KVPE_HEAD_DIM]
    total_len = n_chunks * CHUNK_SIZE
    threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88"))
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))

    logger.info(f"[producer] reading KV cache via table: slot={slot_id} n_chunks={n_chunks} total_len={total_len}")
    min_pcc = 1.0
    failures = []
    for layer in range(NUM_LAYERS):
        rows = []
        for pos in range(0, total_len, chunk_tok):
            raw = table.read_device_chunk(layer=layer, position=pos, slot=slot_id)
            chunk = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw, chunk_shape)
            rows.append(ttnn.to_torch(chunk).to(torch.float32).reshape(chunk_tok, KVPE_HEAD_DIM))
        dev = torch.cat(rows, dim=0)[:total_len]  # natural order (table un-rotates block-cyclic)
        g = _load_golden_kv_post(trace_dir, layer, total_len)
        _, pcc_nope = comp_pcc(g[:, :KV_LORA], dev[:, :KV_LORA])
        ref_pe = g[:, KV_LORA:]
        d = ref_pe.shape[-1]
        ref_pe = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)  # HF -> Meta
        _, pcc_pe = comp_pcc(ref_pe, dev[:, KV_LORA:])
        layer_pcc = min(pcc_nope, pcc_pe)
        min_pcc = min(min_pcc, layer_pcc)
        logger.info(f"[producer] layer {layer} KV PCC: nope={pcc_nope:.6f} pe={pcc_pe:.6f} -> {layer_pcc:.6f}")
        if layer_pcc < threshold:
            failures.append((layer, layer_pcc))
    print(
        f"[producer] kv_cache_pcc_complete slot={slot_id} n_chunks={n_chunks} total_len={total_len} min_pcc={min_pcc:.6f}"
    )
    if failures:
        logger.error(f"[producer] KV cache PCC below {threshold} for layers: {failures}")
    else:
        logger.success(f"[producer] KV cache PCC PASSED (min {min_pcc:.6f} >= {threshold})")
    return min_pcc


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

    # Read the KV chunk table the runner published (device-less) and attach to the per-layer LayerAck
    # channel BEFORE pushing input — the runner serializes the table + creates the ack channel during
    # setup, before it enters the request loop.
    kv_table = _read_kv_chunk_table(timeout_s)
    ack_channel = _connect_layer_ack_channel(timeout_s)

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
        f"[producer] done pushing. {len(push_times_ms)} pushes, per-push ms = {[round(t, 2) for t in push_times_ms]}."
    )

    # Wait for the runner's per-layer LayerAcks: NUM_LAYERS per chunk, for every chunk pushed. When
    # they are all in, the KV cache for [0, n_chunks*CHUNK_SIZE) is fully populated across all layers.
    expected_acks = NUM_LAYERS * n_chunks * num_iterations
    _drain_layer_acks(ack_channel, expected_acks)

    # Optionally read the generated KV cache back via the table and PCC-check vs the golden trace.
    # read_device_chunk needs the galaxy open in THIS process, so this is opt-in (see _attach_mesh_for_read).
    if os.environ.get("PREFILL_PRODUCER_CHECK_PCC", "0") == "1" and kv_table is not None:
        mesh_device = None
        try:
            mesh_device = _attach_mesh_for_read()
            _read_kv_and_check_pcc(kv_table, slot_id=slot_id, n_chunks=n_chunks)
        except Exception as e:
            logger.error(f"[producer] KV read/PCC failed: {type(e).__name__}: {e}")
        finally:
            if mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)
    elif os.environ.get("PREFILL_PRODUCER_CHECK_PCC", "0") == "1":
        logger.error("[producer] PREFILL_PRODUCER_CHECK_PCC=1 but no KV chunk table available; skipping PCC.")

    logger.info("[producer] exiting (the runner keeps its sync-op loop running).")


if __name__ == "__main__":
    main()
