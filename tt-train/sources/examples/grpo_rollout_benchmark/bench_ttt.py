#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttt backend: GRPO BoolQ with REMOTE generation on a tt-transformers worker.

Two MPI ranks under tt-run: rank 0 (TTML) runs training + nlog-prob + the remote
completer; rank 1 (TTT) runs the TttGenerationWorker. runner.sh launches this via
ttrun.py and exports GRPO_BENCH_{TTML_DEVICES,TTT_DEVICES,STEPS,RUN}.
"""

from __future__ import annotations

import gc
import logging
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import ttnn  # noqa: E402

# Pin FABRIC_2D on both ranks BEFORE any device opens; otherwise TTT's
# open_mesh_device auto-escalates to FABRIC_1D and the mismatch deadlocks the
# cross-rank fabric init. Sole fabric set: do NOT re-set later (a repeat
# SetFabricConfig forces a peer-less control-plane reinit that deadlocks).
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

import benchmark_common as bc  # noqa: E402
from utils.ttt_remote.weight_bridge import TTML_RANK, TTT_RANK  # noqa: E402

TTT_MAX_SEQ_LEN = 2048

TTML_DEVICES = int(os.environ.get("GRPO_BENCH_TTML_DEVICES", "2"))
TTT_DEVICES = int(os.environ.get("GRPO_BENCH_TTT_DEVICES", "2"))
STEPS = int(os.environ.get("GRPO_BENCH_STEPS", "20"))
RUN = int(os.environ.get("GRPO_BENCH_RUN", "1"))

TTML_CONFIG = os.path.join(_THIS_DIR, "configs", f"ttml_{TTML_DEVICES}dev.yaml")


def _open_ttml_device(device_config):
    import ttml

    # Do NOT call enable_fabric() here: fabric is already pinned FABRIC_2D at
    # import. A repeat SetFabricConfig re-runs a peer-less control-plane reinit
    # collective (TTT never re-sets) and deadlocks device bring-up.
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return ctx.get_device()


def _ttml_main() -> None:
    import ttml
    from transformers import AutoTokenizer
    from ttml.trainers import GRPOTrainer
    from utils.ttt_remote.llama_grpo_completer import LlamaCompletionCtx, LlamaCompleterRemoteRollout
    from utils.ttt_remote.mpi_rollout import MPIRolloutClient
    from utils.ttt_remote.weight_bridge import HostWeightBridge

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, grpo_config, transformer_config, optimizer_dict, ttml_devices = bc.load_balanced_config(
        TTML_CONFIG, steps=STEPS
    )
    mesh_device = _open_ttml_device(device_config)

    completer = None
    client = None
    try:
        # Constructing the client blocks on the handshake until the ttt rank has
        # also constructed its MPIRolloutServer.
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)
        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

        tokenizer = AutoTokenizer.from_pretrained(bc.MODEL_ID)
        dataset = bc.build_boolq_dataset(tokenizer)

        completer = LlamaCompleterRemoteRollout(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            mesh_device=mesh_device,
            model_source=bc.MODEL_ID,
            inference_client=client,
            enable_ddp=device_config.enable_ddp,
        )
        # Replace the worker's dummy boot weights with real instruct weights.
        completer.push_weights()

        monitor = bc.BenchmarkMonitor(
            bc.benchmark_csv_path("ttt", ttml_devices, TTT_DEVICES),
            backend="ttt",
            run_index=RUN,
            ttml_devices=ttml_devices,
            ttt_devices=TTT_DEVICES,
        )
        logging.info(
            "ttt backend rank0 | ttml_devices=%d ttt_devices=%d completions/step=%d steps=%d run=%d",
            ttml_devices,
            TTT_DEVICES,
            bc.COMPLETIONS_PER_STEP,
            STEPS,
            RUN,
        )
        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_func=bc.boolq_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[bc.WeightSyncCallback(completer, every=1), monitor],
            model_source=bc.MODEL_ID,
        )
        trainer.train()
    finally:
        # Shut the server down BEFORE closing the mesh: the worker is blocked in
        # serve_forever() and MPI won't tear down cleanly otherwise.
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001
                pass
        completer = None
        gc.collect()
        ttml.autograd.AutoContext.get_instance().close_device()


def _ttt_main() -> None:
    from utils.ttt_remote.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad
    from utils.ttt_remote.mpi_rollout import MPIRolloutServer
    from utils.ttt_remote.ttt_generation_worker import TttGenerationWorker
    from utils.ttt_remote.weight_bridge import HostWeightBridge

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, TTT_DEVICES),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    # Per-submesh batch so the worker's global batch == COMPLETIONS_PER_STEP: the
    # generation batch exactly fills the mesh with no padding-only slots.
    max_batch_per_submesh = bc.COMPLETIONS_PER_STEP // TTT_DEVICES

    worker = None
    server = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(bc.MODEL_ID)
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=bc.MODEL_ID,
            max_batch_size=max_batch_per_submesh,
            max_seq_len=TTT_MAX_SEQ_LEN,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            temperature=bc.TEMPERATURE,
            top_k=0,
            top_p=1.0,
            seed=None,
        )
        bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=worker.submeshes)
        server = MPIRolloutServer(
            peer_rank=TTML_RANK,
            bridge=bridge,
            generate_fn=worker.generate,
            on_weights_received=worker.update_weights,
        )
        server.serve_forever()
    finally:
        worker = None
        server = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("GRPO_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
    if world_size != 2:
        raise RuntimeError(
            f"bench_ttt must run under tt-run with world_size == 2 (got {world_size}). " "Use runner.sh --backend ttt."
        )
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        _ttt_main()
    else:
        raise RuntimeError(f"Unexpected MPI rank {rank}; expected TTML={TTML_RANK} or TTT={TTT_RANK}.")
