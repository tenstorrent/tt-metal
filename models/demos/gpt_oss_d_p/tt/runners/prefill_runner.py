# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS prefill runner — service entry point.

Two modes:
  * standalone: JSON token_ids in, first_token out (no C++ server) — for bring-up/bench
  * request:    H2D chunk loop from prefill_scheduler_driver / prefill_producer

Reference: models/demos/common/prefill/runners/prefill_runner.py
"""

import json
import os
import signal
import time
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import publish_table_and_wait_ready
from models.demos.common.prefill.runners.runner_utils import build_h2d_service

PREFILL_SP = int(os.getenv("PREFILL_SP", "4"))
PREFILL_TP = int(os.getenv("PREFILL_TP", "8"))
PREFILL_NUM_LAYERS = int(os.getenv("PREFILL_NUM_LAYERS", "36"))
PREFILL_MAX_SEQ_LEN = int(os.getenv("PREFILL_MAX_SEQ_LEN", "131072"))
PREFILL_CHUNK_SIZE = int(os.getenv("PREFILL_CHUNK_SIZE", "5120"))
PREFILL_NUM_USERS = int(os.getenv("PREFILL_NUM_USERS", "2"))
PREFILL_ENABLE_MIGRATION = int(os.getenv("PREFILL_ENABLE_MIGRATION", "0"))
PREFILL_STANDALONE = int(os.getenv("PREFILL_STANDALONE", "0"))
PREFILL_STANDALONE_ITERS = int(os.getenv("PREFILL_STANDALONE_ITERS", "1"))

GLOBAL_MESH_SHAPE = (PREFILL_SP, PREFILL_TP)
SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12
SHUTDOWN_METADATA_WORD = -1
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def build_pipeline(mesh_device: ttnn.MeshDevice):
    """Open mesh, create GPT-OSS model, wrap in GptOssPrefillPipeline, compile."""
    from models.demos.gpt_oss_d_p.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss_d_p.tt.common import create_tt_model
    from models.demos.gpt_oss_d_p.tt.tt_gpt_oss_prefill_pipeline import GptOssPrefillPipeline

    hf_model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not hf_model_path:
        raise ValueError(
            "DEEPSEEK_V3_HF_MODEL env var must be set to the GPT-OSS HF weights directory. "
            "Example: export DEEPSEEK_V3_HF_MODEL=/path/to/gpt-oss-120b"
        )

    use_prefill_kv = PREFILL_ENABLE_MIGRATION == 1 or PREFILL_NUM_USERS > 1
    mesh_config = MeshConfig(
        mesh_shape=mesh_device.shape,
        decode=ModeConfig(tp=PREFILL_TP, ep=mesh_device.shape[0]),
        prefill=ModeConfig(tp=PREFILL_TP, sp=PREFILL_SP, ep=1),
    )

    logger.info(
        f"build_pipeline: num_layers={PREFILL_NUM_LAYERS} max_seq_len={PREFILL_MAX_SEQ_LEN} "
        f"chunk_size={PREFILL_CHUNK_SIZE} num_users={PREFILL_NUM_USERS} "
        f"prefill_kv_cache={use_prefill_kv} mesh={mesh_device.shape}"
    )
    model_args, model, kv_cache, _ = create_tt_model(
        mesh_device,
        max_batch_size=PREFILL_NUM_USERS,
        max_seq_len=PREFILL_MAX_SEQ_LEN,
        num_layers=PREFILL_NUM_LAYERS,
        mesh_config=mesh_config,
        create_kv_cache=True,
        use_prefill_kv_cache=use_prefill_kv,
        sp_axis=0,
        tp_axis=1,
    )

    pipeline = GptOssPrefillPipeline(
        mesh_device=mesh_device,
        hf_config=model_args.hf_config,
        model=model,
        kv_cache=kv_cache,
        mesh_config=mesh_config,
        sp_factor=PREFILL_SP,
        max_seq_len=PREFILL_MAX_SEQ_LEN,
        chunk_size=PREFILL_CHUNK_SIZE,
        num_users=PREFILL_NUM_USERS,
    )

    if PREFILL_ENABLE_MIGRATION:
        logger.info(
            "Gate-2b slot-copy migration: external migration_worker + scheduler "
            "(not per-layer migrate_layer callback)"
        )
    else:
        logger.info("Migration disabled (PREFILL_ENABLE_MIGRATION=0)")

    pipeline.compile()
    return pipeline


def run_standalone_loop(pipeline) -> None:
    """Read token IDs from a JSON file, run prefill, print the first token."""
    default_path = Path(__file__).parent / "standalone_input.json"
    input_path = Path(os.environ.get("PREFILL_STANDALONE_INPUT", str(default_path)))
    logger.info(f"[standalone] reading input from {input_path}")

    with open(input_path) as f:
        data = json.load(f)
    task_id = data["task_id"]
    token_ids = list(data["token_ids"])

    if len(token_ids) > PREFILL_MAX_SEQ_LEN:
        raise ValueError(
            f"task_id={task_id} has {len(token_ids)} tokens but PREFILL_MAX_SEQ_LEN={PREFILL_MAX_SEQ_LEN}."
        )

    actual_isl = len(token_ids)
    logger.info(f"[standalone] task_id={task_id} actual_isl={actual_isl} iters={PREFILL_STANDALONE_ITERS}")

    iter_times_ms = []
    first_token = None
    for i in range(PREFILL_STANDALONE_ITERS):
        t0 = time.perf_counter()
        first_token = pipeline.prefill(token_ids, slot_id=0, actual_isl=actual_isl, dst_slot=0)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        iter_times_ms.append(dt_ms)
        logger.info(
            f"[prefill timing] task_id={task_id} iter={i} actual_isl={actual_isl} "
            f"pipeline.prefill()={dt_ms:.2f} ms first_token={first_token}"
        )

    logger.info(f"[iter timing summary] per-iter ms = {[round(t, 2) for t in iter_times_ms]}")
    print(f"[standalone] task_id={task_id} first_token={first_token}")


def _is_shutdown_sentinel(meta: dict) -> bool:
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _socket_next(h2d_service) -> tuple:
    """Block on the next H2D push: (tt_tokens, {slot_id, actual_start, actual_end})."""
    import torch

    tt_tokens, tt_metadata = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
    return tt_tokens, {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}


def _migration_done_path() -> str:
    return os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")


def _expected_h2d_chunks() -> int:
    """Chunks the scheduler driver will push before writing MIGRATION_DONE_FILE (0 = unknown)."""
    return int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "0"))


def _wait_migration_done_sentinel(
    done_file: str,
    *,
    chunk_count: int,
    poll_s: float,
    wait_s: float,
) -> None:
    """After the last expected H2D chunk, poll until the driver creates done_file.

    The scheduler sends a finite number of chunks then writes the DONE sentinel when
    migration finishes. Do not call _socket_next here — the driver will not send more
    chunks and blocking on H2D can hang after the driver shuts down its socket.
    """
    logger.info(
        f"[request] processed {chunk_count} H2D chunk(s); waiting for migration DONE sentinel "
        f"{done_file!r} (poll every {poll_s}s, timeout {wait_s}s). "
        f"No further H2D chunks are expected from the scheduler."
    )
    deadline = time.time() + wait_s
    while not _shutdown:
        if os.path.exists(done_file):
            logger.info(
                f"[request] migration DONE sentinel at {done_file}; " f"exiting H2D loop after {chunk_count} chunk(s)"
            )
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"[request] migration DONE sentinel {done_file} never appeared after {wait_s}s "
                "(did Terminal C finish prefill + migration?)"
            )
        time.sleep(poll_s)
    logger.info(f"[request] SIGTERM during DONE sentinel wait; exiting after {chunk_count} chunk(s)")


def _all_h2d_chunks_received(chunk_count: int, expected_chunks: int) -> bool:
    if expected_chunks > 0:
        return chunk_count >= expected_chunks
    # Migration loopback with validate: one scheduler SUBMIT => one chunk unless caller
    # sets PREFILL_STANDALONE_CHUNKED_NCHUNKS explicitly for multi-chunk prompts.
    return chunk_count > 0


def run_request_loop(mesh_device: ttnn.MeshDevice, pipeline) -> None:
    """Production H2D request loop for migration tests (single-rank).

    Blocks on inbound_socket_service_sync until the scheduler closes the stream, the
    migration DONE sentinel appears, or SIGTERM. When PREFILL_ENABLE_MIGRATION=1,
    publishes KV table + LayerAck before the loop and validates slot-copy after the
    DONE sentinel.
    """
    from models.demos.gpt_oss_d_p.tt.runners.prefill_kv_validation import validate_after_prefill

    mesh_device.clear_loaded_sub_device_manager()
    h2d_service = build_h2d_service(
        mesh_device,
        mesh_shape=GLOBAL_MESH_SHAPE,
        chunk_size=PREFILL_CHUNK_SIZE,
        mapper_config=H2D_MAPPER_CONFIG,
        worker_cores=SYNC_WORKER_CORES,
        metadata_size_bytes=METADATA_SIZE_BYTES,
    )
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "gpt_prefill")
    descriptor_path = h2d_service.export_descriptor(service_id)
    logger.info(
        f"[h2d] descriptor service_id={service_id!r} -> {descriptor_path}; "
        f"drive with prefill_scheduler_driver or prefill_producer"
    )

    ack_channel = None
    done_file = _migration_done_path()
    validate_migration = PREFILL_ENABLE_MIGRATION == 1 and os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") == "1"
    done_poll_s = float(os.environ.get("PREFILL_MIGRATION_DONE_POLL_S", "0.5"))
    done_wait_s = int(os.environ.get("PREFILL_MIGRATE_WAIT_S", "1200"))
    expected_chunks = _expected_h2d_chunks()

    if PREFILL_ENABLE_MIGRATION:
        if os.path.exists(done_file):
            logger.warning(f"[migration] removing stale DONE sentinel {done_file}")
            try:
                os.remove(done_file)
            except OSError as e:
                raise RuntimeError(
                    f"Cannot remove stale DONE sentinel {done_file}: {e}. "
                    "Another user's file is blocking migration validation. "
                    "Use a writable path in all three terminals, e.g. "
                    "export MIGRATION_DONE_FILE=$HOME/migration_done.sentinel"
                ) from e

        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/gpt_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))
        pipeline.build_kv_chunk_table(table_path)
        publish_table_and_wait_ready(
            mesh_device=mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            table_path=table_path,
            wait_ready_timeout_ms=wait_ready_ms,
        )

        ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
        stale = f"/dev/shm/{ack_shm_name.lstrip('/')}"
        if os.path.exists(stale):
            logger.warning(f"[migration] removing stale LayerAck shm {stale}")
            os.remove(stale)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        pipeline.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}")

    if validate_migration:
        logger.info(
            f"[request] entering H2D loop (after H2D chunk(s), wait for DONE file {done_file!r}; "
            f"expected_chunks={expected_chunks or 'auto-after-first-chunk'})"
        )
    else:
        logger.info("[request] entering H2D loop (unbounded until shutdown sentinel or SIGTERM)")
    t0 = time.perf_counter()
    chunk_count = 0
    last_real_end = 0

    try:
        while not _shutdown:
            if validate_migration and _all_h2d_chunks_received(chunk_count, expected_chunks):
                _wait_migration_done_sentinel(
                    done_file,
                    chunk_count=chunk_count,
                    poll_s=done_poll_s,
                    wait_s=done_wait_s,
                )
                break

            tt_tokens, meta = _socket_next(h2d_service)
            if _is_shutdown_sentinel(meta):
                logger.info(f"[request] SHUTDOWN sentinel after {chunk_count} chunks")
                ttnn.deallocate(tt_tokens)
                break

            pipeline.prefill_chunk(
                tt_tokens,
                slot_id=meta["slot_id"],
                actual_start=meta["actual_start"],
                actual_end=meta["actual_end"],
            )
            last_real_end = max(last_real_end, meta["actual_end"])
            chunk_count += 1

            if validate_migration and os.path.exists(done_file):
                logger.info(
                    f"[request] migration DONE sentinel at {done_file}; "
                    f"exiting H2D loop after {chunk_count} chunk(s)"
                )
                break

        ttnn.synchronize_device(mesh_device)
        logger.info(f"[request] processed {chunk_count} chunks in {(time.perf_counter() - t0) * 1000.0:.2f} ms")

        if chunk_count == 0:
            logger.warning(
                "[request] H2D loop exited with 0 chunks — scheduler may not have pushed tokens "
                "(start Terminal C after WORKER_READY) or a stale shutdown sentinel was received. "
                "Clear /dev/shm/tt_h2d_* between runs if unsure."
            )

        validate_after_prefill(
            pipeline,
            pipeline.kv_cache,
            num_users=PREFILL_NUM_USERS,
            real_end=last_real_end or None,
        )
    finally:
        h2d_service = None
        import gc

        gc.collect()
        if ack_channel is not None:
            ack_channel.shutdown()
            ack_channel = None

    logger.info(
        "[request] H2D loop finished — runner will exit after mesh teardown (expected after scheduler shutdown)"
    )


def _print_config() -> None:
    sep = "=" * 70
    rows = [
        ("DEEPSEEK_V3_HF_MODEL", os.environ.get("DEEPSEEK_V3_HF_MODEL", "<NOT SET>")),
        ("PREFILL_SP", str(PREFILL_SP)),
        ("PREFILL_TP", str(PREFILL_TP)),
        ("PREFILL_NUM_LAYERS", str(PREFILL_NUM_LAYERS)),
        ("PREFILL_MAX_SEQ_LEN", str(PREFILL_MAX_SEQ_LEN)),
        ("PREFILL_CHUNK_SIZE", str(PREFILL_CHUNK_SIZE)),
        ("PREFILL_NUM_USERS", str(PREFILL_NUM_USERS)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "gpt_prefill")),
        ("PREFILL_ENABLE_MIGRATION", str(PREFILL_ENABLE_MIGRATION)),
        ("PREFILL_VALIDATE_MIGRATION", os.environ.get("PREFILL_VALIDATE_MIGRATION", "0")),
        ("PREFILL_MIGRATE_PAIRWISE", os.environ.get("PREFILL_MIGRATE_PAIRWISE", "1")),
        ("PREFILL_STANDALONE_CHUNKED_NCHUNKS", os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "0")),
        ("PREFILL_STANDALONE", str(PREFILL_STANDALONE)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<default>")),
        ("PREFILL_STANDALONE_ITERS", str(PREFILL_STANDALONE_ITERS)),
        ("PREFILL_MIGRATION_TABLE_PATH", os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/gpt_kv_chunk_table.pb")),
        ("MIGRATION_DONE_FILE", os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")),
    ]
    config_lines = [sep, "prefill_runner configuration", sep]
    for label, val in rows:
        config_lines.append(f"  {label:<35} = {val}")
    config_lines.append(sep)
    logger.info("\n" + "\n".join(config_lines))


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _print_config()

    assert (
        PREFILL_CHUNK_SIZE % PREFILL_SP == 0
    ), f"PREFILL_CHUNK_SIZE={PREFILL_CHUNK_SIZE} must be divisible by PREFILL_SP={PREFILL_SP}"

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(PREFILL_SP, PREFILL_TP))
    try:
        pipeline = build_pipeline(mesh_device)

        if PREFILL_STANDALONE:
            logger.info("Setup complete, running standalone loop (file input)")
            run_standalone_loop(pipeline)
        else:
            logger.info("Setup complete, entering request loop (H2D)")
            run_request_loop(mesh_device, pipeline)
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
