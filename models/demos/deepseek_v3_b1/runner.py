#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""DeepSeek V3 B1 inference bridge.

Stage-0 (mesh_id == 0) bridges the C++ inference server via shared memory.
On receiving a prefill message (prompt token_ids + max_tokens), it runs the
full inference loop (prefill + decode) and streams each generated token back.

Launch with:
    tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml <path_to_this_file>
"""

import argparse
import os
import signal
import sys
from pathlib import Path

from shared_memory import DECODE_MAX_TOKEN_IDS, PREFILL_MAX_TOKEN_IDS, SharedMemory

import ttnn
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _rank() -> int:
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("tt-cpp-server runner")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("/mnt/models/deepseek-ai/cache-2026-03-22"),
        help="Path to the weight cache directory (required for --weights real)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=("synthetic", "real"),
        default="real",
        help="Use synthetic or real (cached) weights (default: real)",
    )
    parser.add_argument(
        "--fabric-max-payload-bytes",
        type=int,
        default=15232,
        help="Fabric router max packet payload bytes (must match DeepSeek V3 B1 pipeline config)",
    )
    parser.add_argument(
        "--fabric-router-sync-timeout-ms",
        type=int,
        default=30000,
        help="Set TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS (ms).",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def _shm_is_configured() -> bool:
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    return bool(c2p_name and p2c_name)


def _run_shm_bridge(model_pipeline: ModelPipeline) -> None:
    """Open shared-memory buffers and run the inference loop."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        raise RuntimeError("Shared memory bridge not configured. Set TT_IPC_SHM_C2P and TT_IPC_SHM_P2C.")

    def is_shutdown() -> bool:
        return _shutdown

    with SharedMemory(c2p_name, max_token_ids=PREFILL_MAX_TOKEN_IDS, is_shutdown=is_shutdown) as c2p, SharedMemory(
        p2c_name, max_token_ids=DECODE_MAX_TOKEN_IDS, is_shutdown=is_shutdown
    ) as p2c:
        print("Starting inference loop")
        while not _shutdown:
            msg = c2p.read()
            if msg is None:
                break
            print(f"Received message: {msg}")
            print(
                f"Running inference for task {msg.task_id} with "
                f"{len(msg.token_ids)} token_ids and max_tokens {msg.max_tokens}"
            )
            model_pipeline.run_inference(
                prompt_token_ids=msg.token_ids,
                max_new_tokens=msg.max_tokens,
                on_token=lambda tid: p2c.write_token(msg.task_id, tid),
                eos_token_id=1,
            )
            print("Inference completed")


def _fabric_config_for_num_procs(num_procs: int):
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs in (16, 64):
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, or 64)")


def _open_mesh_device(
    fabric_max_payload_bytes: int,
    fabric_router_sync_timeout_ms: int,
):
    os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = str(fabric_router_sync_timeout_ms)

    num_procs = int(ttnn.distributed_context_get_size())
    fabric_router_config = create_fabric_router_config(fabric_max_payload_bytes)
    ttnn.set_fabric_config(
        _fabric_config_for_num_procs(num_procs),
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    try:
        return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2), worker_l1_size=1431568)
    except TypeError:
        return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    rank = _rank()
    args = _parse_args(sys.argv[1:])

    if rank == 0:
        shm_enabled = _shm_is_configured()
        if not shm_enabled:
            raise RuntimeError(
                "Shared memory bridge not configured. Set TT_IPC_SHM_C2P and TT_IPC_SHM_P2C "
                "and ensure both exist under /dev/shm/."
            )

    try:
        try:
            ttnn.init_distributed_context()
        except Exception as e:
            # Some launchers may initialize distributed context before invoking this script.
            if "already" not in str(e).lower():
                raise
        mesh_device = _open_mesh_device(
            fabric_max_payload_bytes=args.fabric_max_payload_bytes,
            fabric_router_sync_timeout_ms=args.fabric_router_sync_timeout_ms,
        )
    except Exception as e:
        print(f"Rank {rank}: failed to open mesh device: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Rank {rank}: Opening model pipeline")
        model_pipeline = ModelPipeline(
            weights_mode=args.weights,
            cache_path=args.cache_path if args.weights == "real" else None,
            mesh_device=mesh_device,
        )

        if rank == 0:
            _run_shm_bridge(model_pipeline)
        else:
            print(f"Rank {rank}: Waiting (non-host)")
            while not _shutdown:
                signal.pause()

        model_pipeline.terminate()
    finally:
        if ttnn is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
