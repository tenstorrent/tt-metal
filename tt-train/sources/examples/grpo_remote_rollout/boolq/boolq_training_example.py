#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO training of Llama-3.2-1B-Instruct on BoolQ across two tt-run ranks
(rank 0 TTML policy/training, rank 1 TTT generation). Requires HF_TOKEN."""

from __future__ import annotations

import gc
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make ``utils.*`` importable when the file is run directly (needed before
# any ``from utils.* import ...`` at module scope).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttml
import ttnn
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, get_model_config, load_config
from ttml.trainers import GRPOTrainer, get_grpo_config
from utils.llama_grpo_completer import LlamaCompleterRemoteRollout, LlamaCompletionCtx
from utils.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad
from utils.mpi_rollout import MPIRolloutClient, MPIRolloutServer
from utils.ttt_generation_worker import TttGenerationWorker
from utils.weight_bridge import HostWeightBridge, TTML_RANK, TTT_RANK

CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml"

REPO_ROOT = Path(__file__).resolve().parents[5]

# Pin fabric config before either _ttml_main or _ttt_main opens a device.
# Otherwise TTT's open_mesh_device auto-escalates to FABRIC_1D and the mismatch deadlocks the
# cross-rank fabric init.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)


def accuracy_reward(completions, answer, **kwargs):
    """+2 if the completion begins with the correct Yes/No token, -1 otherwise."""
    return [2.0 if text.strip().lower().startswith(gt.lower()) else -1.0 for text, gt in zip(completions, answer)]


def brevity_reward(completions, **kwargs):
    """Quadratic length penalty in characters, discouraging runaway completions."""
    return [-0.1 * (len(text) / 20) ** 2 for text in completions]


def get_output_dir() -> str:
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
        if step % self.every == 0:
            self.completer.push_weights()

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


def _load_device_config():
    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    # Do NOT call enable_fabric() here: fabric is already pinned FABRIC_2D at
    # import. A repeat SetFabricConfig re-runs a control-plane reinit collective
    # with no peer (TTT never re-sets) and deadlocks device bring-up.
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    ttml.autograd.AutoContext.get_instance().close_device()


def _ttml_main() -> None:
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config()
    mesh_device = _open_ttml_device(device_config)

    model_id = raw["training_config"]["model_id"]
    weight_sync_every = int(raw["training_config"]["weight_sync_every"])

    completer: Any = None
    client: Any = None
    try:
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        system_prompt = "You are a wordy professor. Explain in 3 long sentences before saying Yes or No."

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

        output_dir = get_output_dir()
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
            model_source=model_id,
            enable_ddp=device_config.enable_ddp,
        )

        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)
        completer._client = client

        # Replace the worker's dummy boot weights with real instruct weights
        # before training starts.
        completer.push_weights()

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_funcs=[accuracy_reward, brevity_reward],
            optimizer_dict=optimizer_dict,
            callbacks=[
                WeightSyncCallback(completer, every=weight_sync_every),
            ],
            model_source=model_id,
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
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    # Read the same yaml as the ttml rank so both use the same GRPO sampling
    # temperature (the worker bakes it into the captured decode trace) and so
    # the rollout mesh / batch / seq-len come from ``remote_rollout_config``.
    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    grpo_temperature = float(raw["training_config"]["grpo_config"]["temperature"])
    model_id = raw["training_config"]["model_id"]
    rr = raw["remote_rollout_config"]

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*rr["mesh_shape"]),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    worker: Any = None
    server: Any = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(model_id)

        # One worker owns the whole parent mesh: it splits it into [1,1] submeshes
        # and runs generation data-parallel across them.
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=model_id,
            max_batch_size=rr["max_batch_size"],
            max_seq_len=rr["max_seq_len"],
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
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"boolq_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use boolq/runner.sh."
        )

    rank = int(ttnn.distributed_context_get_rank())
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        _ttt_main()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {rank} (world_size={world_size}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
