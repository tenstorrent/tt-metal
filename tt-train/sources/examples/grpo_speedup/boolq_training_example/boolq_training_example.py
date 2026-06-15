#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO training of Llama-3.2-1B-Instruct on BoolQ across two ranks.

Launched under ``tt-run`` with world_size == 2 (see :file:`runner.sh`).

Topology
========

* Rank 0 (TTML) opens the full ``[1, 2]`` mesh declared by
  ``configurations/local2/mgd.textproto`` (one N300 board, two chips).
  It owns the policy ttml ``Llama`` model and drives training via
  :class:`ttml.trainers.GRPOTrainer`.
* Rank 1 (TTT) opens a ``[1, 1]`` mesh on its own board. It hosts a
  ``tt-transformers`` ``Transformer`` via
  :class:`utils.ttt_generation_worker.TttGenerationWorker` and serves
  generate / weight-transfer RPCs to the ttml rank.

The :class:`utils.inference_bridge.TttInferenceClient` constructor on
the ttml side and the :class:`utils.inference_bridge.TttInferenceServer`
constructor on the ttt side block until both have run -- the
``WeightBridge`` handshake inside their initialisers pins the two
ranks together before any RPC happens.

Lifecycle (TTML rank)
=====================

1. Build :class:`TttInferenceClient` -- handshake completes once the
   worker rank also constructs its server.
2. Build :class:`LlamaGRPOCompleter`. Loads the real instruct tokenizer
   and ttml model into device memory.
3. Call ``completer.push_weights()`` once. This overwrites the worker's
   dummy boot weights with real instruct weights so the very first
   ``trainer.train()`` generate request returns coherent completions.
4. Construct :class:`GRPOTrainer` with
   :class:`utils.llama_grpo_completer.WeightSyncCallback`
   (``every=1``) so every gradient step also pushes the freshly
   updated policy weights to the worker.
5. ``trainer.train()`` runs the loop.
6. ``client.shutdown()`` releases the worker. **Must** run before the
   ttml device is closed -- the worker is blocked inside
   ``serve_forever()`` until it sees ``OP_SHUTDOWN``.

Lifecycle (TTT rank)
====================

1. Initialise ttnn distributed context, open a ``[1, 1]`` submesh.
2. Resolve stop / pad token IDs by briefly loading the HF tokenizer
   for ``MODEL_ID`` (see
   :func:`utils.llama_ttt_presets.llama_stop_and_pad`). The tokenizer
   is dropped immediately; the worker never holds one.
3. Build :class:`TttGenerationWorker` -- ``dummy_weights=True`` plus
   ``disable_disk_cache=True`` so boot is fast and the asymmetric
   ``[1, 2] -> [1, 1]`` mesh handshake does not trip the disk-cache
   collective.
4. Build :class:`TttInferenceServer` with the worker's ``generate`` and
   ``update_weights`` as the two callbacks.
5. ``server.serve_forever()`` until ``OP_SHUTDOWN``.

Self-skips with a clear error if launched outside ``tt-run`` (world
size != 2). Requires ``HF_TOKEN`` set in the environment for the
initial HuggingFace download of the instruct model weights on the
ttml rank.
"""

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

# Silence Python-side log noise that would otherwise drown out the per-step
# GRPOMonitor line. tt-metal C++ logs are silenced separately via
# TT_LOGGER_LEVEL=Error in runner.sh.
logger.remove()
logger.add(sys.stderr, level="ERROR")
logging.getLogger().setLevel(logging.ERROR)

# Line-buffer stdout/stderr so any remaining print() (notably GRPOMonitor's
# per-step line) flushes on each '\n' without needing flush=True at every
# call. Under mpirun's --tag-output, stdout is a pipe and Python's default
# would otherwise be block-buffered (4 KB), making real-time output appear
# in long delayed bursts.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Make ``utils.*`` modules importable when this script is run directly
# (not under pytest's rootdir machinery): the launcher lives one level
# deeper than ``utils/`` so we insert the example root explicitly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttnn  # noqa: E402

# Pin fabric to FABRIC_2D on both ranks before any device opens. Without
# this, the TTT rank's open_mesh_device falls into DeviceManager's
# legacy auto-escalation to FABRIC_1D (see tt_metal/impl/device/device_manager.cpp),
# while TTML's ttml.core.distributed.enable_fabric(2) picks FABRIC_2D --
# the mismatch deadlocks the cross-rank fabric init collective. Mirrors
# tests/conftest.py's autouse _set_fabric_2d fixture.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

from utils.inference_bridge import TTML_RANK, TTT_RANK  # noqa: E402


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_2dev_ddp_gas_4.yaml"
TTT_MESH_SHAPE = (1, 1)

# Worker memory budget. 32 is comfortably below the [1, 1] N300 chip
# capacity for a 1B Llama instruct at bf16/bfp8 with 2K context, and
# divides cleanly into typical batch_size * num_generations products.
TTT_MAX_BATCH_SIZE = 32
TTT_MAX_SEQ_LEN = 2048

# Push weights every step. GRPO updates the policy each gradient step,
# so without this the worker would generate from stale weights.
WEIGHT_SYNC_EVERY = 1

REPO_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# Reward / monitor helpers (mirrors examples/grpo/boolq_training_example.py)
# ---------------------------------------------------------------------------


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
        "generated/tt-train/grpo_speedup_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


class GRPOMonitor:
    """on_step_end CSV/stdout monitor. Mirrors the equivalent class in
    ``examples/grpo/boolq_training_example.py``; kept as a plain class
    because ``TrainerCallback`` only exposes a no-op default that we
    don't otherwise need here."""

    def __init__(self, output_dir: str) -> None:
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "avg_length", "step_time_s", "generation_time_s"])

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        reward = kwargs["reward_mean"]
        length = kwargs["mean_completion_len"]
        min_length = kwargs["min_completion_len"]
        max_length = kwargs["max_completion_len"]
        step_time_s = kwargs.get("step_time_s", float("nan"))
        generation_time_s = kwargs.get("generation_time_s", float("nan"))
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Step {step} | Reward: {reward:.4f} "
            f"| Len: {length:.2f} (min {min_length}, max {max_length}) tokens "
            f"| Step: {step_time_s:.2f}s | Gen: {generation_time_s:.2f}s"
        )
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, length, step_time_s, generation_time_s])

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        print("Training complete.")


# ---------------------------------------------------------------------------
# Device-opening helpers (inlined from tests/_completer_utils.py so this
# script is self-contained -- avoids adding tests/ to sys.path).
# ---------------------------------------------------------------------------


def _load_device_config(device_config_rel: str = TTML_DEVICE_CONFIG_REL):
    from ttml.common.config import DeviceConfig, load_config

    raw = load_config(os.path.join(str(REPO_ROOT), device_config_rel))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    import ttml

    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    import ttml

    ttml.autograd.AutoContext.get_instance().close_device()


# ---------------------------------------------------------------------------
# TTML rank entrypoint
# ---------------------------------------------------------------------------


def _ttml_main() -> None:
    import ttml
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config
    from ttml.trainers.grpo_trainer_logged import GRPOTrainer
    from ttml.trainers import get_grpo_config

    from utils.inference_bridge import TttInferenceClient
    from utils.llama_grpo_completer import (
        LlamaCompletionCtx,
        LlamaGRPOCompleter,
        WeightSyncCallback,
    )

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = _open_ttml_device(device_config)

    completer: Any = None
    client: Any = None
    try:
        # Constructor blocks on the WeightBridge handshake -- returns
        # once the ttt rank has also constructed its TttInferenceServer.
        client = TttInferenceClient(peer_rank=TTT_RANK, device=mesh_device)

        # ------------------------------------------------------------ #
        # Dataset + GRPO config                                         #
        # ------------------------------------------------------------ #
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

        output_dir = _run_output_dir()
        grpo_config = get_grpo_config(raw, output_dir=output_dir)
        optimizer_dict = raw["training_config"]["optimizer"]
        transformer_config = get_model_config(raw["training_config"]["model_config"])

        # ------------------------------------------------------------ #
        # Completer                                                     #
        # ------------------------------------------------------------ #
        completer = LlamaGRPOCompleter(
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

        # Initial weight push: replace the worker's dummy boot weights
        # with real instruct weights BEFORE training kicks off.
        completer.push_weights()

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_func=_boolq_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[
                GRPOMonitor(output_dir),
                WeightSyncCallback(completer, every=WEIGHT_SYNC_EVERY),
            ],
            model_source=MODEL_ID,
        )
        trainer.train()
    finally:
        # Shutdown ordering: tell the server to exit BEFORE we drop the
        # completer or close the mesh. The worker is otherwise still
        # blocked in serve_forever() and MPI would never tear down cleanly.
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001 -- best-effort during teardown
                pass
        completer = None
        gc.collect()
        _close_ttml_device()


# ---------------------------------------------------------------------------
# TTT rank entrypoint
# ---------------------------------------------------------------------------


def _ttt_main() -> None:
    from ttml.common.config import load_config

    from utils.inference_bridge import TttInferenceServer
    from utils.llama_ttt_presets import (
        bf16_attn_bfp8_mlp_optimizations,
        llama_stop_and_pad,
    )
    from utils.ttt_generation_worker import TttGenerationWorker

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    # Open the [1, 1] submesh of the declared [1, 2] mesh -- matches the
    # weight-transfer test exactly. The mgd.textproto entries for rank 1
    # use a separate mesh_id, so the offset is (0, 0) within that mesh.
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    # Read the same yaml as the ttml rank to pick up the GRPO sampling
    # temperature. The worker bakes (temperature, top_k, top_p, seed) into
    # the captured decode trace at construction; the ttml-side completer
    # forwards the same temperature value via remote_generate(), so the
    # two stay consistent as long as both ranks read it from this file.
    raw = load_config(os.path.join(str(REPO_ROOT), TTML_DEVICE_CONFIG_REL))
    grpo_temperature = float(raw["training_config"]["grpo_config"]["temperature"])

    worker: Any = None
    server: Any = None
    try:
        # Tokenizer load is launcher-local: we extract stop/pad IDs and
        # drop the reference before the worker is built.
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)

        worker = TttGenerationWorker(
            mesh_device=mesh_device,
            model_source=MODEL_ID,
            max_batch_size=TTT_MAX_BATCH_SIZE,
            max_seq_len=TTT_MAX_SEQ_LEN,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            # On-device sampling. top_k=0 -> "no top-k restriction" (clamped
            # to the TT max of 32 inside format_sampling_params). seed=None
            # -> SeedManager picks a random 64-bit seed per slot per
            # generate() call, which is what GRPO wants for per-prompt
            # replica diversity (num_generations > 1).
            temperature=grpo_temperature,
            top_k=0,
            top_p=1.0,
            seed=None,
        )

        server = TttInferenceServer(
            peer_rank=TTML_RANK,
            device=mesh_device,
            generate_fn=worker.generate,
            on_weights_received=worker.update_weights,
        )
        server.serve_forever()
    finally:
        worker = None
        server = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
    if world_size != 2:
        raise RuntimeError(
            f"boolq_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use boolq_training_example/runner.sh."
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
