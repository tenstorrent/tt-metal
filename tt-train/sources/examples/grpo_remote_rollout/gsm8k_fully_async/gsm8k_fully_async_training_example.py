#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO on GSM8K, FULLY-ASYNC rollout variant.

Two-rank tt-run entrypoint:

  * Rank 0 (ttml): builds a real Qwen3 policy + optimizer and drives
    :class:`~ttml.trainers.FullyAsyncGRPOTrainer`. The trainer pops rollout
    batches from a :class:`RolloutQueue` (consumer), uses the batch's own
    sampled log-probs as the ratio denominator (proper off-policy PPO),
    runs one optimizer step per batch, and fires the updated weights
    through a :class:`ThreadedWeightBridge` (sender).
  * Rank 1 (TTT): :func:`run_inference_loop` from
    ``utils/fully_async_inference.py`` -- opens the parent mesh, builds a
    :class:`TttGenerationWorker`, installs ``theta_0`` from the bridge,
    then loops generation + push (queue producer) + poll_weights until
    ``TRAINING_STOPPED``.

Staleness handling follows the aReal formulation: at rank-0 step ``s``
(1-indexed), a rollout tagged ``vgen`` is dropped when
``(s - 1) - vgen > max_staleness``.

Diffs vs the sync sibling ``gsm8k_onestep_training_example.py``:

  * ``OneStepAsyncGRPOTrainer`` -> ``FullyAsyncGRPOTrainer``.
  * ``MPIRolloutClient`` / ``MPIRolloutServer`` request-response pair is
    replaced by :class:`RolloutQueue` (unidirectional producer -> consumer).
  * ``HostWeightBridge`` / ``MeshSocketWeightBridge`` is replaced by
    :class:`ThreadedWeightBridge` (sender / receiver thread per rank, dict
    payload, lazy on-device pads, cross-CQ event synchronization).
  * No ``Qwen3FullyAsyncCompleter`` -- reuses ``Qwen3CompleterRemoteRollout``
    with ``inference_client=None``. The generate / push_weights paths are
    dead code for the fully-async trainer.
  * The rollout loop lives in ``utils/fully_async_inference.py`` and is
    not visible in this file. Users of this example see only rank-0
    training code plus a one-liner dispatch.

Run:
    tt-train/sources/examples/grpo_remote_rollout/gsm8k_fully_async/runner.sh
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttml
import ttnn
from datasets import load_dataset
from ttml.common.config import DeviceConfig, get_model_config, load_config
from ttml.trainers import FullyAsyncGRPOTrainer, get_grpo_config

from utils.fully_async_inference import run_inference_loop
from utils.qwen3_grpo_completer import Qwen3CompleterRemoteRollout, Qwen3CompletionCtx
from utils.qwen3_overrides import qwen3_weights_ref_hf_dict
from utils.rollout_queue import RolloutQueue
from utils.threaded_weight_bridge import ThreadedWeightBridge
from utils.weight_bridge import TTML_RANK, TTT_RANK
from utils.async_training_event_channel import AsyncTrainingEvent, AsyncTrainingEventChannel


CONFIG_REL = "tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_fully_async.yaml"
REPO_ROOT = Path(__file__).resolve().parents[5]

DATASET = "openai/gsm8k"
DATASET_SPLIT = "train"

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "Respond in the following format:\n" f"{THINK_OPEN}\n...\n{THINK_CLOSE}\n" f"{ANSWER_OPEN}\n...\n{ANSWER_CLOSE}\n"
)

# NOTE: no ttnn.set_fabric_config here. Fully-async uses ThreadedWeightBridge
# (pure MPI, host-only transport) + RolloutQueue (pure MPI), not the
# fabric-based mesh_socket bridge. Skipping fabric init saves ~1s of boot
# on each rank AND avoids the two-rank barrier that would otherwise gate
# rank 1's ``ttnn.open_mesh_device`` on rank 0 also opening a mesh.

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

FORMAT_RE = re.compile(
    re.escape(THINK_OPEN)
    + r".*?"
    + re.escape(THINK_CLOSE)
    + r"\s*"
    + re.escape(ANSWER_OPEN)
    + r".*?"
    + re.escape(ANSWER_CLOSE),
    re.DOTALL,
)

STRICT_FORMAT_RE = re.compile(
    r"^\s*"
    + re.escape(THINK_OPEN)
    + r"\n.*?\n"
    + re.escape(THINK_CLOSE)
    + r"\n"
    + re.escape(ANSWER_OPEN)
    + r"\n.*?\n"
    + re.escape(ANSWER_CLOSE)
    + r"\s*$",
    re.DOTALL,
)


# ---- gsm8k dataset formatting + rewards (lifted from gsm8k_onestep) ------


def normalize_number(s: str) -> str:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return s
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def extract_hash_answer(gold: str) -> str:
    return normalize_number(gold.split("####")[-1])


def extract_tag_answer(text: str) -> Optional[str]:
    if ANSWER_OPEN not in text:
        return None
    body = text.split(ANSWER_OPEN)[-1].split(ANSWER_CLOSE)[0]
    nums = _NUM_RE.findall(body)
    return normalize_number(nums[-1]) if nums else None


def xmlcount_reward(completions, **kwargs) -> List[float]:
    def score(text: str) -> float:
        s = 0.0
        for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN):
            if text.count(tag) == 1:
                s += 0.125
        if text.count(ANSWER_CLOSE) == 1:
            s += 0.125
            s -= len(text.split(ANSWER_CLOSE)[-1].strip()) * 0.001
        return s

    return [score(c) for c in completions]


def soft_format_reward(completions, **kwargs) -> List[float]:
    return [0.5 if FORMAT_RE.search(c) else 0.0 for c in completions]


def strict_format_reward(completions, **kwargs) -> List[float]:
    return [0.5 if STRICT_FORMAT_RE.match(c) else 0.0 for c in completions]


def int_reward(completions, **kwargs) -> List[float]:
    def score(text: str) -> float:
        p = extract_tag_answer(text)
        return 0.5 if p is not None and p.lstrip("-").isdigit() else 0.0

    return [score(c) for c in completions]


def correctness_reward(completions, answer, **kwargs) -> List[float]:
    def score(text: str, gold: str) -> float:
        p = extract_tag_answer(text)
        return 2.0 if p is not None and p == gold else 0.0

    return [score(c, g) for c, g in zip(completions, answer)]


REWARD_FUNCS = [
    correctness_reward,
    xmlcount_reward,
    soft_format_reward,
    strict_format_reward,
    int_reward,
]


def _build_dataset_rows(seed: int):
    ds = load_dataset(DATASET, "main", split=DATASET_SPLIT)

    def to_example(row):
        return {
            "prompt": f"{SYSTEM_PROMPT}\nQuestion: {row['question']}\nAnswer:",
            "answer": extract_hash_answer(row["answer"]),
        }

    return ds.shuffle(seed=seed).map(to_example, remove_columns=ds.column_names)


def build_gsm8k_prompts_and_answers(tokenizer: Any, seed: int):
    """DatasetFactory used by rank 1's ``run_inference_loop``.

    Returns ``(tokenized_prompts, gold_answers)`` aligned 1:1. Rank 1 pushes
    ``{"answer": gold_answers_expanded}`` inside ``RolloutBatch.extra``,
    which the trainer's ``correctness_reward`` reads back on rank 0.
    """
    rows = _build_dataset_rows(seed)
    prompts: List[List[int]] = [tokenizer.encode(row["prompt"]) for row in rows]
    answers: List[str] = [str(row["answer"]) for row in rows]
    return prompts, answers


# ---- rank 0 (ttml) helpers -----------------------------------------------


def get_output_dir() -> str:
    return os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_gsm8k_fully_async_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


def _load_device_config():
    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    # ``num_command_queues=2`` is required by ``ThreadedWeightBridge``: the
    # sender thread issues ``ttnn.to_torch(pad, cq_id=1)`` in parallel with the
    # main thread's ``ttnn.copy`` / ``record_event`` on CQ0. Opening with the
    # default 1 CQ trips ``TT_FATAL: cq_id 1 is out of range`` as soon as the
    # first weight publish reaches the sender thread.
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids, num_command_queues=2)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    ttml.autograd.AutoContext.get_instance().close_device()


def _assert_fully_async_config(raw: dict) -> None:
    """Fail fast before rank 1 boots the TTT worker if the yaml disagrees
    with the FullyAsyncGRPOTrainer contract."""
    num_iterations = int(raw["training_config"]["grpo_config"].get("num_iterations", 1))
    if num_iterations != 1:
        raise ValueError(f"gsm8k_fully_async requires grpo_config.num_iterations == 1 (got {num_iterations}).")
    fa = raw["training_config"].get("fully_async_config", {})
    for key in ("max_staleness",):
        if key not in fa:
            raise ValueError(f"training_config.fully_async_config must contain '{key}'")


def _num_devices_from_config(device_config) -> int:
    n = 1
    for d in device_config.mesh_shape:
        n *= int(d)
    return n


def _ttml_main() -> None:
    """Real training loop.

    Builds a real Qwen3 policy, an optimizer, a
    :class:`ThreadedWeightBridge` sender, and a :class:`RolloutQueue`
    consumer. Publishes ``theta_0`` before the trainer's first pop; every
    subsequent optimizer step publishes ``theta_s`` for whichever rollout
    rank 1 starts next.

    On loop exit (or exception): sends ``TRAINING_STOPPED`` on the event
    channel, closes the bridge (drains the sender thread), closes the
    queue, and releases the ttml device.
    """
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config()
    _assert_fully_async_config(raw)
    mesh_device = _open_ttml_device(device_config)

    model_id = raw["training_config"]["model_id"]
    fa = raw["training_config"]["fully_async_config"]
    max_staleness: int = int(fa["max_staleness"])
    rollout_queue_capacity: int = int(fa.get("rollout_queue_capacity", 2))

    completer: Any = None
    bridge: Optional[ThreadedWeightBridge] = None
    queue: Optional[RolloutQueue] = None
    channel: Optional[AsyncTrainingEventChannel] = None
    try:
        output_dir = get_output_dir()
        grpo_config = get_grpo_config(raw, output_dir=output_dir)
        optimizer_dict = raw["training_config"]["optimizer"]
        transformer_config = get_model_config(raw["training_config"]["model_config"])

        # Build the completer. Fully-async never calls
        # ``generate`` / ``submit_generate`` / ``push_weights`` (rollouts
        # come from the queue, weights go over the bridge), so passing
        # ``inference_client=None`` is safe.
        print(f"[training] building Qwen3CompleterRemoteRollout(model_id={model_id!r})...", flush=True)
        completer = Qwen3CompleterRemoteRollout(
            ctx=Qwen3CompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            mesh_device=mesh_device,
            model_source=model_id,
            inference_client=None,
            enable_ddp=device_config.enable_ddp,
        )
        print("[training] completer built", flush=True)

        # Bridge FIRST, queue SECOND -- both do MPI_Comm_dup collectively;
        # the two ranks must call in the same order.
        print(f"[training] connecting ThreadedWeightBridge sender to rank {TTT_RANK}...", flush=True)
        bridge = ThreadedWeightBridge.sender(peer_rank=TTT_RANK, mesh_device=mesh_device)
        bridge.connect()
        print("[training] bridge connected + sender thread started", flush=True)

        print(
            f"[training] connecting RolloutQueue consumer to rank {TTT_RANK} "
            f"(capacity={rollout_queue_capacity})...",
            flush=True,
        )
        queue = RolloutQueue.consumer(peer_rank=TTT_RANK, capacity=rollout_queue_capacity)
        queue.connect()
        print("[training] queue connected + consumer thread started", flush=True)

        # Ship completions-per-batch to rank 1 -- exactly the count the
        # trainer's ``_setup`` derives from
        # ``per_device_train_batch_size * num_devices * grad_accum``.
        num_devices = _num_devices_from_config(device_config)
        completions_per_batch = (
            int(grpo_config.per_device_train_batch_size) * num_devices * int(grpo_config.gradient_accumulation_steps)
        )
        channel = AsyncTrainingEventChannel(peer_rank=TTT_RANK)
        channel.send(AsyncTrainingEvent.TRAINING_BATCH_SIZE, payload=completions_per_batch)
        print(f"[training] sent TRAINING_BATCH_SIZE({completions_per_batch}) to rank {TTT_RANK}", flush=True)

        # Build the trainer AND publish theta_0 BEFORE waiting for
        # INFERENCE_READY. Rank 1 blocks on the first weight dict before
        # sending INFERENCE_READY, so if we did it the other way around
        # both ranks would deadlock.
        trainer = FullyAsyncGRPOTrainer(
            completer=completer,
            config=grpo_config,
            rollout_queue=queue,
            weight_bridge=bridge,
            weights_export_fn=lambda: qwen3_weights_ref_hf_dict(
                completer.model, tie_word_embeddings=completer._tie_word_embeddings
            ),
            max_staleness=max_staleness,
            reward_funcs=REWARD_FUNCS,
            optimizer_dict=optimizer_dict,
            model_source=model_id,
        )
        print("[training] publishing theta_0 to the weight bridge...", flush=True)
        trainer.publish_current_weights()
        print("[training] theta_0 handed off to bridge sender thread", flush=True)

        # Wait for rank 1 to finish its heavy init AND install theta_0.
        print("[training] waiting for INFERENCE_READY from rank 1...", flush=True)
        ev, _ = channel.wait_for_next_event()
        if ev != AsyncTrainingEvent.INFERENCE_READY:
            raise RuntimeError(f"expected INFERENCE_READY, got {ev.name}")
        print("[training] got INFERENCE_READY; entering training loop", flush=True)

        trainer.train()
        print(f"[training] trainer.train() returned; ran {trainer.metrics.get('step', 0)} steps", flush=True)
    finally:
        # Signal rank 1 to exit its generation loop.
        if channel is not None:
            try:
                channel.send(AsyncTrainingEvent.TRAINING_STOPPED)
                print("[training] sent TRAINING_STOPPED", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[training] sending TRAINING_STOPPED failed: {type(e).__name__}: {e}", flush=True)
        if bridge is not None:
            try:
                bridge.close()
            except Exception as e:  # noqa: BLE001
                print(f"[training] bridge.close() raised {type(e).__name__}: {e}", flush=True)
        if queue is not None:
            try:
                queue.close()
            except Exception as e:  # noqa: BLE001
                print(f"[training] queue.close() raised {type(e).__name__}: {e}", flush=True)
        completer = None
        gc.collect()
        _close_ttml_device()


# ---- entrypoint dispatch ---------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"gsm8k_fully_async_training_example must run under tt-run with world_size == 2 "
            f"(got {world_size}). Use gsm8k_fully_async/runner.sh."
        )

    rank = int(ttnn.distributed_context_get_rank())
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        run_inference_loop(
            config_path=os.path.join(str(REPO_ROOT), CONFIG_REL),
            dataset_factory=build_gsm8k_prompts_and_answers,
        )
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {rank} (world_size={world_size}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
