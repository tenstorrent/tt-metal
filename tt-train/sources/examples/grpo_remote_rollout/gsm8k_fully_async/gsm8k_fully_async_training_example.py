#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO on GSM8K, FULLY-ASYNC rollout variant -- FIRST CUT.

Two-rank tt-run entrypoint (rank 0 = mock training loop that only drives the
event protocol; rank 1 = real TttGenerationWorker running a generation-with-
log-probs loop). Neither rank does actual GRPO training in this cut; the
purpose is to validate the event protocol, log-prob plumbing, and dataset
loader end-to-end before we swap in a real ``FullyAsyncGRPOTrainer``.

Diffs vs the one-step sibling ``gsm8k_onestep_training_example.py``:

  * No trainer, no completer, no weight bridge -- rank 0 is a sleep-based
    mock that fires typed :class:`AsyncTrainingEvent` s to rank 1.
  * Rank 1 owns its own tokenizer (via ``AutoTokenizer.from_pretrained``) so
    ``DatasetLoader`` can tokenise gsm8k prompts on the rollout side; the
    training rank doesn't touch the dataset in this cut.
  * All messaging goes through :class:`AsyncTrainingEventChannel`, backed
    by the new ``distributed_context_iprobe_bytes`` non-blocking probe.

Run:
    tt-train/sources/examples/grpo_remote_rollout/gsm8k_fully_async/runner.sh
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterator, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttml
import ttnn
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, load_config

from utils.async_training_event_channel import AsyncTrainingEvent, AsyncTrainingEventChannel
from utils.qwen3_ttt_presets import bf16_attn_bfp8_mlp_optimizations, qwen3_stop_and_pad
from utils.ttt_generation_worker import TttGenerationWorker
from utils.weight_bridge import TTML_RANK, TTT_RANK

CONFIG_REL = "tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_fully_async.yaml"
REPO_ROOT = Path(__file__).resolve().parents[5]

DATASET = "openai/gsm8k"
DATASET_SPLIT = "train"

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "Respond in the following format:\n" f"{THINK_OPEN}\n...\n{THINK_CLOSE}\n" f"{ANSWER_OPEN}\n...\n{ANSWER_CLOSE}\n"
)

# Matches gsm8k_onestep -- FABRIC_2D is required by MeshSocketWeightBridge in the
# real (follow-up) version and cheap to set unconditionally here.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


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


def build_dataset(seed: int):
    ds = load_dataset(DATASET, "main", split=DATASET_SPLIT)

    def to_example(row):
        return {
            "prompt": f"{SYSTEM_PROMPT}\nQuestion: {row['question']}\nAnswer:",
            "answer": extract_hash_answer(row["answer"]),
        }

    return ds.shuffle(seed=seed).map(to_example, remove_columns=ds.column_names)


class DatasetLoader:
    """Tokenises gsm8k prompts upfront, yields batches of ``prompts_per_batch``
    prompt token-id lists in a wrap-around loop.

    Expansion by ``num_generations`` is deliberately kept OUTSIDE the loader --
    the caller does ``[p for p in prompts for _ in range(G)]`` to align with the
    worker's output slots. Keeps this class single-responsibility.
    """

    def __init__(self, dataset: Any, tokenizer: Any, prompts_per_batch: int) -> None:
        assert prompts_per_batch > 0, f"prompts_per_batch must be > 0 (got {prompts_per_batch})"
        self._prompts: List[List[int]] = [tokenizer.encode(row["prompt"]) for row in dataset]
        if len(self._prompts) < prompts_per_batch:
            raise ValueError(
                f"dataset has {len(self._prompts)} prompts, need at least " f"{prompts_per_batch} for one batch"
            )
        self._prompts_per_batch: int = int(prompts_per_batch)

    def iter(self) -> Iterator[List[List[int]]]:
        i, n = 0, len(self._prompts)
        while True:
            batch = [self._prompts[(i + j) % n] for j in range(self._prompts_per_batch)]
            i = (i + self._prompts_per_batch) % n
            yield batch


def _load_device_config():
    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    ttml.autograd.AutoContext.get_instance().close_device()


def _num_devices_from_config(device_config) -> int:
    """Total ttml device count = product of mesh_shape dims. Used by the mock
    training loop to size ``completions_per_batch`` without opening a device."""
    shape = device_config.mesh_shape
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _ttml_main() -> None:
    """Mock training loop.

    Does NOT build a model or an optimizer -- just drives the event protocol so
    the two ranks can validate the handshake end-to-end.

    We DO open a ttml mesh on this rank even though the mock loop never touches
    device tensors: ttnn.set_fabric_config(FABRIC_2D) plus the intermesh
    connection declared in mgd.textproto (graph G0) make rank 1's
    ttnn.open_mesh_device block on a fabric handshake with mesh 0. Matching the
    gsm8k_onestep pattern (autograd_ctx.open_device) satisfies that handshake
    without dragging in any tensor work.
    """
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config()

    # Open the ttml device solely so the fabric intermesh channel between mesh 0
    # (this rank) and mesh 1 (peer) can be established -- rank 1's open blocks
    # until we do this. Closed in the finally below.
    _open_ttml_device(device_config)

    try:
        # completions_per_batch matches GRPOTrainer._setup:1067:
        #   completions_per_microbatch = per_device_train_batch_size * num_devices
        #   generation_batch_prompts   = completions_per_microbatch * grad_accum
        # We ship completions (not prompts) so the inference rank can size its
        # loader after dividing by num_generations.
        grpo = raw["training_config"]["grpo_config"]
        fa = raw["training_config"]["fully_async_config"]
        num_devices = _num_devices_from_config(device_config)
        completions_per_batch = (
            int(grpo["per_device_train_batch_size"]) * num_devices * int(grpo["gradient_accumulation_steps"])
        )

        wait_step_s: float = float(fa["wait_step_s"])
        weight_bridge_mock_s: float = float(fa["weight_bridge_mock_s"])
        steps: int = int(fa["steps"])

        print(
            f"[mock-training] boot: num_devices={num_devices}, "
            f"per_device_train_batch_size={grpo['per_device_train_batch_size']}, "
            f"gradient_accumulation_steps={grpo['gradient_accumulation_steps']}, "
            f"completions_per_batch={completions_per_batch}, "
            f"wait_step_s={wait_step_s}, weight_bridge_mock_s={weight_bridge_mock_s}, "
            f"steps={steps}",
            flush=True,
        )

        channel = AsyncTrainingEventChannel(peer_rank=TTT_RANK)
        channel.send(AsyncTrainingEvent.TRAINING_BATCH_SIZE, payload=completions_per_batch)
        print(f"[mock-training] sent TRAINING_BATCH_SIZE({completions_per_batch}) to rank {TTT_RANK}", flush=True)

        weights_pushed = False
        for step in range(steps):
            time.sleep(wait_step_s)
            print(f"[mock-training] step {step} done (slept {wait_step_s}s)", flush=True)

            if weights_pushed:
                ev, _payload = channel.wait_for_next_event()
                assert (
                    ev == AsyncTrainingEvent.INFERENCE_RECEIVED_WEIGHTS
                ), f"expected INFERENCE_RECEIVED_WEIGHTS, got {ev.name}"
                print("[mock-training] got INFERENCE_RECEIVED_WEIGHTS ack", flush=True)

            channel.send(AsyncTrainingEvent.TRAINING_ABOUT_TO_SEND_WEIGHTS)
            print("[mock-training] sent TRAINING_ABOUT_TO_SEND_WEIGHTS", flush=True)

            # Mock the actual weight-bridge push. In the follow-up cut this becomes
            # a HostWeightBridge / MeshSocketWeightBridge send_weights call.
            time.sleep(weight_bridge_mock_s)
            channel.send(AsyncTrainingEvent.TRAINING_SENT_WEIGHTS)
            print(
                f"[mock-training] sent TRAINING_SENT_WEIGHTS after mock push of {weight_bridge_mock_s}s",
                flush=True,
            )

            weights_pushed = True

        channel.send(AsyncTrainingEvent.TRAINING_STOPPED)
        print("[mock-training] sent TRAINING_STOPPED, exiting", flush=True)
    finally:
        _close_ttml_device()


def _log(msg: str) -> None:
    """Print a rank-1 progress line with flush so MPI's --tag-output picks it up
    immediately (default stdout buffering hides progress during the long
    trace-capture / HF-download stalls)."""
    print(f"[inference] {msg}", flush=True)


def _ttt_main() -> None:
    """Real inference loop.

    Owns the parent mesh, the ``TttGenerationWorker``, a locally-loaded tokenizer,
    the :class:`DatasetLoader`, and the :class:`AsyncTrainingEventChannel`. Runs
    :meth:`TttGenerationWorker.generate_and_get_log_probs` in a wrap-around loop
    until it receives ``TRAINING_STOPPED``.
    """
    _log("entering _ttt_main")
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    grpo_config = raw["training_config"]["grpo_config"]
    grpo_temperature = float(grpo_config["temperature"])
    num_generations = int(grpo_config["num_generations"])
    max_completion_length = int(grpo_config["max_completion_length"])
    model_id = raw["training_config"]["model_id"]
    rr = raw["remote_rollout_config"]
    seed = int(raw["training_config"].get("seed", 0))

    _log(f"opening parent mesh {tuple(rr['mesh_shape'])}...")
    _t0 = time.perf_counter()
    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*rr["mesh_shape"]),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    _log(f"parent mesh open ({time.perf_counter() - _t0:.1f}s)")

    worker: Any = None
    channel: Optional[AsyncTrainingEventChannel] = None
    try:
        _log(f"resolving stop/pad tokens for {model_id}...")
        _t1 = time.perf_counter()
        stop_token_ids, pad_token_id = qwen3_stop_and_pad(model_id)
        _log(f"stop/pad tokens ready ({time.perf_counter() - _t1:.1f}s)")

        _log("building TttGenerationWorker (may take 30-60s: HF config + ttml Transformer + trace capture)...")
        _t2 = time.perf_counter()
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
            # We keep dummy_weights=True for the first cut: the real weight
            # bridge is deferred, so any completions we generate are gibberish
            # but the plumbing (tokens + logprobs shape) is exercised end-to-end.
            dummy_weights=True,
        )
        _log(f"worker built ({time.perf_counter() - _t2:.1f}s)")

        # TttGenerationWorker doesn't own a tokenizer; DatasetLoader needs one
        # to encode gsm8k prompts, so we load it locally on the TTT rank.
        _log("loading tokenizer via AutoTokenizer.from_pretrained...")
        _t3 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _log(f"tokenizer ready ({time.perf_counter() - _t3:.1f}s)")

        channel = AsyncTrainingEventChannel(peer_rank=TTML_RANK)
        _log("waiting for TRAINING_BATCH_SIZE from rank 0...")
        ev, batch_size_completions = channel.wait_for_next_event()
        assert ev == AsyncTrainingEvent.TRAINING_BATCH_SIZE, f"expected TRAINING_BATCH_SIZE first, got {ev.name}"
        prompts_per_batch = batch_size_completions // num_generations
        assert prompts_per_batch > 0, (
            f"batch_size_completions={batch_size_completions} < num_generations={num_generations}; "
            f"per-batch prompt count would be 0"
        )
        _log(
            f"got TRAINING_BATCH_SIZE({batch_size_completions}); "
            f"prompts_per_batch={prompts_per_batch}, num_generations={num_generations}"
        )

        _log("loading + tokenizing gsm8k dataset...")
        _t4 = time.perf_counter()
        loader = DatasetLoader(
            build_dataset(seed),
            tokenizer=tokenizer,
            prompts_per_batch=prompts_per_batch,
        )
        _log(f"dataset loader ready ({time.perf_counter() - _t4:.1f}s); entering generation loop")

        for iter_idx, prompts in enumerate(loader.iter()):
            # Expand 1:1 with the worker's output slots (each prompt spawns G completions).
            prompts_expanded = [p for p in prompts for _ in range(num_generations)]
            _log(
                f"iter {iter_idx}: starting generate_and_get_log_probs "
                f"({len(prompts_expanded)} prompts, max_new_tokens={max_completion_length})..."
            )
            _t_gen = time.perf_counter()
            completions, logprobs = worker.generate_and_get_log_probs(
                prompts_expanded,
                max_new_tokens=max_completion_length,
            )
            _gen_s = time.perf_counter() - _t_gen

            avg_len = mean(len(c) for c in completions) if completions else 0.0
            shapes_match = all(len(lp) == len(c) for lp, c in zip(logprobs, completions))
            _log(
                f"iter {iter_idx}: generated {len(completions)} completions in {_gen_s:.1f}s "
                f"(avg len={avg_len:.1f}, logprob shapes match={shapes_match})"
            )
            # TODO: ship (completions, logprobs) to the rollout completions queue -- deferred.

            ev_opt = channel.poll()
            if ev_opt is None:
                _log(f"iter {iter_idx}: no pending event, continuing")
                continue

            kind, _payload = ev_opt
            _log(f"iter {iter_idx}: polled event {kind.name}")
            if kind == AsyncTrainingEvent.TRAINING_STOPPED:
                _log("got TRAINING_STOPPED, exiting loop")
                break

            if kind == AsyncTrainingEvent.TRAINING_ABOUT_TO_SEND_WEIGHTS:
                _log("waiting for matching TRAINING_SENT_WEIGHTS...")
                kind2, _ = channel.wait_for_next_event()
                assert (
                    kind2 == AsyncTrainingEvent.TRAINING_SENT_WEIGHTS
                ), f"expected TRAINING_SENT_WEIGHTS after _ABOUT_TO_SEND, got {kind2.name}"
                # TODO: worker.update_weights(pad_dict) once the real bridge is wired.
                _log("would copy weights from bridge pad into the worker here")
                channel.send(AsyncTrainingEvent.INFERENCE_RECEIVED_WEIGHTS)
                _log("sent INFERENCE_RECEIVED_WEIGHTS ack")
            else:
                _log(f"unexpected event {kind.name}; ignoring")
    finally:
        _log("cleaning up (releasing worker + closing mesh)...")
        worker = None
        channel = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)
        _log("clean exit")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"gsm8k_fully_async_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use gsm8k_fully_async/runner.sh."
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
