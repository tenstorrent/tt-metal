#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO training of Llama-3.2-1B-Instruct on BoolQ across two tt-run ranks
(rank 0 TTML policy/training, rank 1 TTT generation). Requires HF_TOKEN."""

from __future__ import annotations

import csv
import gc
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")
logging.getLogger().setLevel(logging.ERROR)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Make ``utils.*`` importable when run directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttnn  # noqa: E402

# Pin FABRIC_2D on both ranks before any device opens; otherwise TTT's
# open_mesh_device auto-escalates to FABRIC_1D and the mismatch deadlocks the
# cross-rank fabric init. This is the SOLE fabric set: do NOT re-set fabric
# afterward (e.g. enable_fabric() at device-open) -- a repeat SetFabricConfig
# forces a peer-less control-plane reinit collective that deadlocks.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

from utils.weight_bridge import TTML_RANK, TTT_RANK  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Topology from GRPO_BOOLQ_TOPOLOGY (set by runner.sh). Defaults to 2x2;
# 4x4 currently hangs in the cross-rank handshake/transport on this host.
_TOPOLOGIES = {
    "2x2": {
        "ttml_device_config_rel": "tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_2dev.yaml",
        "ttt_parent_mesh_shape": (1, 2),
    },
    "4x4": {
        "ttml_device_config_rel": "tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml",
        "ttt_parent_mesh_shape": (1, 4),
    },
}

TOPOLOGY = os.environ.get("GRPO_BOOLQ_TOPOLOGY", "2x2")
if TOPOLOGY not in _TOPOLOGIES:
    raise RuntimeError(
        f"Unknown GRPO_BOOLQ_TOPOLOGY={TOPOLOGY!r}; expected one of {sorted(_TOPOLOGIES)}. "
        "Select it via boolq/runner.sh --topology."
    )
_TOPO = _TOPOLOGIES[TOPOLOGY]

TTML_DEVICE_CONFIG_REL = _TOPO["ttml_device_config_rel"]
TTT_PARENT_MESH_SHAPE = _TOPO["ttt_parent_mesh_shape"]

TTT_MAX_BATCH_SIZE = 32
TTT_MAX_SEQ_LEN = 2048
WEIGHT_SYNC_EVERY = 1

REPO_ROOT = Path(__file__).resolve().parents[5]


def _boolq_reward(completions, answer, **kwargs):
    rewards = []
    for text, ground_truth in zip(completions, answer):
        clean = text.strip().lower()
        accuracy = 2.0 if clean.startswith(ground_truth.lower()) else -1.0
        brevity = -0.1 * (len(text) / 20) ** 2
        rewards.append(accuracy + brevity)
    return rewards


def _run_output_dir() -> str:
    return os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


class WeightSyncCallback:
    """Push fresh policy weights to the TTT generation worker every ``every``
    steps. The caller does the initial push before ``trainer.train()``."""

    def __init__(self, completer: Any, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"WeightSyncCallback: 'every' must be >= 1 (got {every})")
        self.completer = completer
        self.every = every

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        if (step + 1) % self.every == 0:
            self.completer.push_weights()

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


class GRPOMonitor:
    """on_step_end CSV/stdout monitor."""

    def __init__(self, output_dir: str) -> None:
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step", "reward", "avg_length", "step_time_s", "step_time_with_weight_updates_s", "generation_time_s"]
            )

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        reward = kwargs["reward_mean"]
        length = kwargs["mean_completion_len"]
        min_length = kwargs["min_completion_len"]
        max_length = kwargs["max_completion_len"]
        step_time_s = kwargs.get("step_time_s", float("nan"))
        step_time_and_previous_callbacks_s = kwargs.get("step_time_and_previous_callbacks_s", float("nan"))
        generation_time_s = kwargs.get("generation_time_s", float("nan"))
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Step {step} | Reward: {reward:.4f} "
            f"| Len: {length:.2f} (min {min_length}, max {max_length}) tokens "
            f"| Step: {step_time_s:.2f}s (with updates: {step_time_and_previous_callbacks_s:.2f}s) | Gen: {generation_time_s:.2f}s"
        )
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, length, step_time_s, step_time_and_previous_callbacks_s, generation_time_s])

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        print("Training complete.")


def _load_device_config(device_config_rel: str = TTML_DEVICE_CONFIG_REL):
    from ttml.common.config import DeviceConfig, load_config

    raw = load_config(os.path.join(str(REPO_ROOT), device_config_rel))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    import ttml

    # Do NOT call enable_fabric() here: fabric is already pinned FABRIC_2D at
    # import. A repeat SetFabricConfig re-runs a control-plane reinit collective
    # with no peer (TTT never re-sets) and deadlocks device bring-up.
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    import ttml

    ttml.autograd.AutoContext.get_instance().close_device()


def _ttml_main() -> None:
    import ttml
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config
    from ttml.trainers import GRPOTrainer, get_grpo_config
    from utils.llama_grpo_completer import (
        LlamaCompletionCtx,
        LlamaCompleterRemoteRollout,
    )
    from utils.mpi_rollout import MPIRolloutClient
    from utils.weight_bridge import HostWeightBridge

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = _open_ttml_device(device_config)

    completer: Any = None
    client: Any = None
    try:
        # Constructing the client blocks on the handshake until the ttt rank
        # has also constructed its MPIRolloutServer.
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)
        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        system_prompt = "Answer the question. Your answer should begin with either a Yes or a No. Then, explain why you answered Yes or No."

        def format_boolq(example):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
            ]
            return {
                "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "answer": "yes" if example["answer"] else "no",
            }

        dataset = load_dataset("google/boolq", split="train").shuffle(seed=42).map(format_boolq)

        output_dir = _run_output_dir()
        grpo_config = get_grpo_config(raw, output_dir=output_dir)
        optimizer_dict = raw["training_config"]["optimizer"]
        transformer_config = get_model_config(raw["training_config"]["model_config"])

        completer = LlamaCompleterRemoteRollout(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            mesh_device=mesh_device,
            model_source=MODEL_ID,
            inference_client=client,
            enable_ddp=device_config.enable_ddp,
        )

        # Replace the worker's dummy boot weights with real instruct weights
        # before training starts.
        completer.push_weights()

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_func=_boolq_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[
                WeightSyncCallback(completer, every=WEIGHT_SYNC_EVERY),
                GRPOMonitor(output_dir),
            ],
            model_source=MODEL_ID,
        )
        trainer.train()
    finally:
        # Shut the server down BEFORE closing the mesh: the worker is blocked
        # in serve_forever() and MPI won't tear down cleanly otherwise.
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001
                pass
        completer = None
        gc.collect()
        _close_ttml_device()


def _ttt_main() -> None:
    from ttml.common.config import load_config
    from utils.llama_ttt_presets import (
        bf16_attn_bfp8_mlp_optimizations,
        llama_stop_and_pad,
    )
    from utils.mpi_rollout import MPIRolloutServer
    from utils.ttt_generation_worker import TttGenerationWorker
    from utils.weight_bridge import HostWeightBridge

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_PARENT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    # Read the same yaml as the ttml rank so both use the same GRPO sampling
    # temperature (the worker bakes it into the captured decode trace).
    raw = load_config(os.path.join(str(REPO_ROOT), TTML_DEVICE_CONFIG_REL))
    grpo_temperature = float(raw["training_config"]["grpo_config"]["temperature"])

    worker: Any = None
    server: Any = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)

        # One worker owns the whole parent mesh: it splits it into [1,1] submeshes
        # and runs generation data-parallel across them.
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=MODEL_ID,
            max_batch_size=TTT_MAX_BATCH_SIZE,
            max_seq_len=TTT_MAX_SEQ_LEN,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            temperature=grpo_temperature,
            top_k=0,
            top_p=1.0,
            seed=None,
        )

        # The bridge replicates each transferred policy onto every submesh; the
        # worker applies one dict per submesh in update_weights().
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
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
    if world_size != 2:
        raise RuntimeError(
            f"boolq_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use boolq/runner.sh."
        )

    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        _ttt_main()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {rank} (world_size={world_size}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
