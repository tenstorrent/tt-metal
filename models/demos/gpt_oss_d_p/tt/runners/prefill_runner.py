# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS prefill runner — service entry point.

Two modes:
  * standalone: JSON token_ids in, first_token out (no C++ server) — for bring-up/bench
  * SHM:        request loop over shared memory from the C++ inference server

JSON input format (for standalone mode):
  {"task_id": <int>, "token_ids": [<int>, ...]}

File path controlled by PREFILL_STANDALONE_INPUT (default: standalone_input.json
next to this script).

Key differences from DeepSeek / MiniMax-M2 runners:
  * PREFILL_NUM_LAYERS=36 (GPT-OSS 120B has 36 decoder layers, not 61/62)
  * PREFILL_MAX_SEQ_LEN=131072 (max_position_embeddings from config.json)
  * Model created via models.demos.gpt_oss.tt.common.create_tt_model
  * env var for HF model path reuses DEEPSEEK_V3_HF_MODEL (per PREFILL_PROPOSAL.md §7)

Reference: models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
"""

import json
import os
import signal
import time
from pathlib import Path

from loguru import logger

import ttnn

PREFILL_SP = int(os.getenv("PREFILL_SP", "4"))
PREFILL_TP = int(os.getenv("PREFILL_TP", "8"))
PREFILL_NUM_LAYERS = int(os.getenv("PREFILL_NUM_LAYERS", "36"))
PREFILL_MAX_SEQ_LEN = int(os.getenv("PREFILL_MAX_SEQ_LEN", "131072"))
PREFILL_ENABLE_MIGRATION = int(os.getenv("PREFILL_ENABLE_MIGRATION", "0"))
PREFILL_STANDALONE = int(os.getenv("PREFILL_STANDALONE", "0"))
PREFILL_STANDALONE_ITERS = int(os.getenv("PREFILL_STANDALONE_ITERS", "1"))

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def build_pipeline(mesh_device: ttnn.MeshDevice):
    """Open mesh, create GPT-OSS model, wrap in GptOssPrefillPipeline, compile.

    Env vars consumed (beyond the module-level ones above):
      DEEPSEEK_V3_HF_MODEL — path to HF weights dir (reused per PREFILL_PROPOSAL.md §7)

    Returns a compiled GptOssPrefillPipeline ready for prefill() calls.
    """
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.common import create_tt_model
    from models.demos.gpt_oss_d_p.tt.tt_gpt_oss_prefill_pipeline import GptOssPrefillPipeline

    hf_model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not hf_model_path:
        raise ValueError(
            "DEEPSEEK_V3_HF_MODEL env var must be set to the GPT-OSS HF weights directory. "
            "Example: export DEEPSEEK_V3_HF_MODEL=/path/to/gpt-oss-120b"
        )

    # Prefill MeshConfig: SP=rows (sequence parallel), TP=cols, EP=1
    mesh_config = MeshConfig(
        mesh_shape=mesh_device.shape,
        decode=ModeConfig(tp=PREFILL_TP, ep=mesh_device.shape[0]),
    )

    logger.info(
        f"build_pipeline: creating model num_layers={PREFILL_NUM_LAYERS} "
        f"max_seq_len={PREFILL_MAX_SEQ_LEN} mesh={mesh_device.shape} config={mesh_config}"
    )
    model_args, model, kv_cache, _ = create_tt_model(
        mesh_device,
        max_batch_size=1,
        max_seq_len=PREFILL_MAX_SEQ_LEN,
        num_layers=PREFILL_NUM_LAYERS,
        mesh_config=mesh_config,
        create_kv_cache=True,
    )

    pipeline = GptOssPrefillPipeline(
        mesh_device=mesh_device,
        hf_config=model_args.hf_config,
        model=model,
        kv_cache=kv_cache,
        mesh_config=mesh_config,
        sp_factor=PREFILL_SP,
        max_seq_len=PREFILL_MAX_SEQ_LEN,
    )

    if PREFILL_ENABLE_MIGRATION:
        from models.demos.gpt_oss_d_p.tt.runners.migration_setup import setup_prefill_migration

        endpoint = setup_prefill_migration()  # raises NotImplementedError until migration team delivers
        pipeline.setup_migration(endpoint)
        logger.info("Migration endpoint: REAL (PREFILL_ENABLE_MIGRATION=1)")
    else:
        logger.info("Migration endpoint: NoOp (PREFILL_ENABLE_MIGRATION=0)")

    pipeline.compile()
    return pipeline


def run_standalone_loop(pipeline) -> None:
    """Read token IDs from a JSON file, run prefill, print the first token.

    Reads PREFILL_STANDALONE_INPUT (default: standalone_input.json next to this
    script). Runs PREFILL_STANDALONE_ITERS iterations for latency benchmarking.

    JSON format: {"task_id": <int>, "token_ids": [<int>, ...]}
    """
    default_path = Path(__file__).parent / "standalone_input.json"
    input_path = Path(os.environ.get("PREFILL_STANDALONE_INPUT", str(default_path)))
    logger.info(f"[standalone] reading input from {input_path}")

    with open(input_path) as f:
        data = json.load(f)
    task_id = data["task_id"]
    token_ids = list(data["token_ids"])

    if len(token_ids) > PREFILL_MAX_SEQ_LEN:
        raise ValueError(
            f"task_id={task_id} has {len(token_ids)} tokens but PREFILL_MAX_SEQ_LEN={PREFILL_MAX_SEQ_LEN}. "
            f"Bump PREFILL_MAX_SEQ_LEN or shorten the prompt."
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
            f"[prefill timing] task_id={task_id} iter={i} num_tokens={PREFILL_MAX_SEQ_LEN} "
            f"pipeline.prefill()={dt_ms:.2f} ms first_token={first_token}"
        )

    logger.info(f"[iter timing summary] per-iter ms = {[round(t, 2) for t in iter_times_ms]}")
    # stdout: callers (tests / orchestrators) parse this line
    print(f"[standalone] task_id={task_id} first_token={first_token}")


def run_request_loop(pipeline) -> None:
    """SHM request loop from the C++ server. OWNER: serving/runner team (Tier 2)."""
    raise NotImplementedError("run_request_loop: owner=serving team; blocked on SHM protocol + C++ server.")


def _print_config() -> None:
    sep = "=" * 70
    rows = [
        ("DEEPSEEK_V3_HF_MODEL", os.environ.get("DEEPSEEK_V3_HF_MODEL", "<NOT SET>")),
        ("PREFILL_SP", str(PREFILL_SP)),
        ("PREFILL_TP", str(PREFILL_TP)),
        ("PREFILL_NUM_LAYERS", str(PREFILL_NUM_LAYERS)),
        ("PREFILL_MAX_SEQ_LEN", str(PREFILL_MAX_SEQ_LEN)),
        ("PREFILL_ENABLE_MIGRATION", str(PREFILL_ENABLE_MIGRATION)),
        ("PREFILL_STANDALONE", str(PREFILL_STANDALONE)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<default>")),
        ("PREFILL_STANDALONE_ITERS", str(PREFILL_STANDALONE_ITERS)),
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

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(PREFILL_SP, PREFILL_TP))
    try:
        pipeline = build_pipeline(mesh_device)

        if PREFILL_STANDALONE:
            logger.info("Setup complete, running standalone loop (file input)")
            run_standalone_loop(pipeline)
        else:
            logger.info("Setup complete, entering request loop")
            run_request_loop(pipeline)
    finally:
        ttnn.close_mesh_device(mesh_device)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
