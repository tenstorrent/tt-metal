# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rank-1 inference loop for the fully-async GRPO example.

Users of the example (``gsm8k_fully_async_training_example.py``) see only the
rank-0 training code plus a one-liner dispatch to :func:`run_inference_loop`.
This module owns everything the rollout worker needs:

  * parent :class:`ttnn.MeshDevice` open + close
  * :class:`TttGenerationWorker` (dummy boot weights + first ``theta_0``
    install via the bridge)
  * :class:`AutoTokenizer` load
  * :class:`DatasetLoader` (prompts + gold answers, wrap-around)
  * :class:`ThreadedWeightBridge` receiver
  * :class:`RolloutQueue` producer
  * :class:`AsyncTrainingEventChannel`

Loop shape (aReal-style, single-version snapshot per :class:`RolloutBatch`):

  1. ``prompts, answers = next(loader.iter())``
  2. Expand ``prompts_x = [p for p in prompts for _ in range(G)]`` and same
     for ``answers_x``.
  3. Snapshot the CURRENTLY-INSTALLED weight version (tracked locally --
     see ``installed_version`` in ``run_inference_loop``) BEFORE any worker
     call so every sub-call inside this batch shares the same theta.
  4. Loop worker sub-calls at ``worker._global_batch_size`` granularity,
     concatenate completions + logprobs. Do NOT poll_weights between
     sub-calls (weights must not shift mid-batch).
  5. Pad the log-probs to fp32 ``[B, max_completion_length]`` and push a
     :class:`RolloutBatch` (``extra={"answer": answers_x}``) to the queue.
  6. Between rollouts, ``bridge.poll_weights()`` -> if a fresh dict is
     available, split per submesh, install via ``worker.update_weights``,
     and bump ``installed_version``.
  7. Break on ``AsyncTrainingEvent.TRAINING_STOPPED``.
"""

from __future__ import annotations

import gc
import time
from statistics import mean
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch

import ttnn
from transformers import AutoTokenizer
from ttml.common.config import load_config

from .async_training_event_channel import AsyncTrainingEvent, AsyncTrainingEventChannel
from .qwen3_ttt_presets import bf16_attn_bfp8_mlp_optimizations, qwen3_stop_and_pad
from .rollout_queue import RolloutBatch, RolloutQueue
from .threaded_weight_bridge import ThreadedWeightBridge
from .ttt_generation_worker import TttGenerationWorker
from .weight_bridge import TTML_RANK


# Signature: (tokenizer, seed) -> (tokenized_prompts, gold_answers), aligned
# 1:1. The consumer's reward functions see ``batch.extra["answer"][i]`` for
# the same-index prompt+completion pair.
DatasetFactory = Callable[[Any, int], Tuple[List[List[int]], List[str]]]


class DatasetLoader:
    """Wrap-around iterator yielding ``(prompts_batch, answers_batch)``
    of length ``prompts_per_batch``. The gold answers travel alongside the
    tokens so the queue producer can put them into ``RolloutBatch.extra``.
    """

    def __init__(
        self,
        prompts: List[List[int]],
        answers: List[str],
        prompts_per_batch: int,
    ) -> None:
        assert prompts_per_batch > 0, f"prompts_per_batch must be > 0 (got {prompts_per_batch})"
        assert len(prompts) == len(answers), f"prompts ({len(prompts)}) / answers ({len(answers)}) length mismatch"
        if len(prompts) < prompts_per_batch:
            raise ValueError(f"dataset has {len(prompts)} prompts, need at least {prompts_per_batch} for one batch")
        self._prompts: List[List[int]] = prompts
        self._answers: List[str] = answers
        self._prompts_per_batch: int = int(prompts_per_batch)

    def iter(self) -> Iterator[Tuple[List[List[int]], List[str]]]:
        i, n = 0, len(self._prompts)
        while True:
            prompts_batch = [self._prompts[(i + j) % n] for j in range(self._prompts_per_batch)]
            answers_batch = [self._answers[(i + j) % n] for j in range(self._prompts_per_batch)]
            i = (i + self._prompts_per_batch) % n
            yield prompts_batch, answers_batch


def _log(msg: str) -> None:
    print(f"[inference] {msg}", flush=True)


def _pad_logprobs(all_logprobs: List[List[float]], max_completion_length: int) -> torch.Tensor:
    """Pack a ragged list-of-lists of per-token sampled log-probs into a
    fixed ``[B, max_completion_length]`` fp32 tensor. Padded positions are
    don't-care; the trainer masks them via the completion loss mask.
    """
    B = len(all_logprobs)
    out = torch.zeros((B, max_completion_length), dtype=torch.float32)
    for u, lp in enumerate(all_logprobs):
        k = min(len(lp), max_completion_length)
        if k > 0:
            out[u, :k] = torch.as_tensor(lp[:k], dtype=torch.float32)
    return out


def _apply_bridge_dict_to_worker(
    bridge_dicts: List[dict],
    worker: TttGenerationWorker,
) -> int:
    """Bring the bridge's parent-mesh recv-pad dict to host once per key,
    replicate on each submesh, and install via
    ``worker.update_weights(per_submesh)``. Returns the number of keys
    installed (0 if the bridge yielded an empty dict, e.g. on peer close).

    Called from INSIDE the ``with bridge.poll_weights()`` /
    ``with bridge.receive_weights()`` block so the bridge's recv-pad lock
    is held across the D->H reads.
    """
    if not bridge_dicts:
        return 0
    parent_dict = bridge_dicts[0]
    if not parent_dict:
        return 0

    # Bring each pad to host once (device 0's copy of the replicated tensor).
    host_dict = {key: ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]) for key, tensor in parent_dict.items()}
    spec_dict = {key: (tensor.dtype, tensor.layout) for key, tensor in parent_dict.items()}

    per_submesh: List[dict] = []
    for submesh in worker.submeshes:
        d = {
            key: ttnn.from_torch(
                host,
                dtype=spec_dict[key][0],
                layout=spec_dict[key][1],
                device=submesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
            )
            for key, host in host_dict.items()
        }
        per_submesh.append(d)

    worker.update_weights(per_submesh)
    return len(host_dict)


def run_inference_loop(
    *,
    config_path: str,
    dataset_factory: DatasetFactory,
    peer_rank: int = TTML_RANK,
    initial_weights_max_wait_s: float = 600.0,
    initial_weights_poll_interval_s: float = 0.5,
) -> None:
    """Rank-1 fully-async inference loop entrypoint.

    Blocks until the peer sends
    :data:`AsyncTrainingEvent.TRAINING_STOPPED`, then closes the bridge,
    queue, and mesh cleanly.
    """
    _log("entering run_inference_loop")
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    raw = load_config(config_path)
    tc = raw["training_config"]
    grpo_cfg = tc["grpo_config"]
    fa_cfg = tc.get("fully_async_config", {})
    rr = raw["remote_rollout_config"]

    grpo_temperature: float = float(grpo_cfg["temperature"])
    num_generations: int = int(grpo_cfg["num_generations"])
    max_completion_length: int = int(grpo_cfg["max_completion_length"])
    model_id: str = tc["model_id"]
    seed: int = int(tc.get("seed", 0))
    rollout_queue_capacity: int = int(fa_cfg.get("rollout_queue_capacity", 2))

    _log(f"opening parent mesh {tuple(rr['mesh_shape'])}...")
    _t0 = time.perf_counter()
    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*rr["mesh_shape"]),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    _log(f"parent mesh open ({time.perf_counter() - _t0:.1f}s)")

    worker: Optional[TttGenerationWorker] = None
    bridge: Optional[ThreadedWeightBridge] = None
    queue: Optional[RolloutQueue] = None
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
            max_batch_size=int(rr["max_batch_size"]),
            max_seq_len=int(rr["max_seq_len"]),
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            temperature=grpo_temperature,
            top_k=0,
            top_p=1.0,
            seed=None,
            # dummy_weights=True; the first bridge dict we install below
            # supplies theta_0 before we enter the generation loop.
            dummy_weights=True,
        )
        _log(f"worker built ({time.perf_counter() - _t2:.1f}s)")

        _log("loading tokenizer via AutoTokenizer.from_pretrained...")
        _t3 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _log(f"tokenizer ready ({time.perf_counter() - _t3:.1f}s)")

        # Bridge FIRST, queue SECOND -- both call
        # ``ttnn.distributed_context_duplicate()`` collectively; the two
        # ranks must call in the same order for the private contexts to
        # line up.
        _log(f"connecting ThreadedWeightBridge receiver to rank {peer_rank}...")
        _t_bridge = time.perf_counter()
        bridge = ThreadedWeightBridge.receiver(peer_rank=peer_rank, mesh_device=parent_mesh)
        bridge.connect()
        _log(f"bridge connected + receiver thread started ({time.perf_counter() - _t_bridge:.1f}s)")

        _log(f"connecting RolloutQueue producer to rank {peer_rank} (capacity={rollout_queue_capacity})...")
        _t_queue = time.perf_counter()
        queue = RolloutQueue.producer(peer_rank=peer_rank, capacity=rollout_queue_capacity)
        queue.connect()
        _log(f"queue connected + producer thread started ({time.perf_counter() - _t_queue:.1f}s)")

        channel = AsyncTrainingEventChannel(peer_rank=peer_rank)

        _log("waiting for TRAINING_BATCH_SIZE from trainer rank...")
        ev, completions_per_batch = channel.wait_for_next_event()
        if ev != AsyncTrainingEvent.TRAINING_BATCH_SIZE:
            raise RuntimeError(f"expected TRAINING_BATCH_SIZE first, got {ev.name}")
        prompts_per_batch = completions_per_batch // num_generations
        if prompts_per_batch <= 0:
            raise ValueError(
                f"completions_per_batch={completions_per_batch} < num_generations={num_generations}; "
                "per-batch prompt count would be 0"
            )
        _log(
            f"got TRAINING_BATCH_SIZE({completions_per_batch}); "
            f"prompts_per_batch={prompts_per_batch}, num_generations={num_generations}, "
            f"worker.global_batch_size={worker._global_batch_size}"
        )

        _log("building tokenised dataset via dataset_factory...")
        _t4 = time.perf_counter()
        tokenised_prompts, gold_answers = dataset_factory(tokenizer, seed)
        loader = DatasetLoader(
            prompts=tokenised_prompts,
            answers=gold_answers,
            prompts_per_batch=prompts_per_batch,
        )
        _log(f"dataset loader ready ({time.perf_counter() - _t4:.1f}s); " f"{len(tokenised_prompts)} prompts total")

        # ---- theta_0 install ------------------------------------------------
        #
        # ``installed_version`` is the LOCAL source of truth for which theta
        # the worker currently holds (0-indexed by install count). It is NOT
        # ``bridge.latest_version()`` because the bridge's recv-pad version
        # counts writes-into-pads, which can outpace installs (the bridge
        # thread may write theta_{k+1} between our polls). Every rollout
        # batch is tagged with ``installed_version``, giving the trainer
        # correct ``(step - 1) - vgen`` staleness accounting.
        installed_version = -1
        _log(f"waiting for initial theta_0 on the weight bridge (up to {initial_weights_max_wait_s:.0f}s)...")
        _t_wait = time.perf_counter()
        while True:
            with bridge.poll_weights() as dicts:
                if dicts is not None:
                    n = _apply_bridge_dict_to_worker(dicts, worker)
                    if n > 0:
                        installed_version += 1
                        _log(
                            f"installed initial theta_{installed_version} ({n} keys, "
                            f"{(time.perf_counter() - _t_wait):.1f}s wait)"
                        )
                        break
            if time.perf_counter() - _t_wait > initial_weights_max_wait_s:
                raise RuntimeError(
                    f"inference: no initial weights received on the bridge after "
                    f"{initial_weights_max_wait_s:.0f}s -- did the trainer boot? "
                    "did ThreadedWeightBridge.connect() collectives line up?"
                )
            time.sleep(initial_weights_poll_interval_s)

        # Handshake AFTER install so the trainer's first `queue.pop` does
        # not race with the worker still applying weights.
        channel.send(AsyncTrainingEvent.INFERENCE_READY)
        _log("sent INFERENCE_READY; entering generation loop")

        # ---- generation loop ------------------------------------------------
        batch_id = 0
        for prompts, answers in loader.iter():
            prompts_x = [p for p in prompts for _ in range(num_generations)]
            answers_x = [a for a in answers for _ in range(num_generations)]
            assert len(prompts_x) == completions_per_batch, (
                f"internal: expanded prompts len {len(prompts_x)} != " f"completions_per_batch {completions_per_batch}"
            )

            # Version snapshot BEFORE any worker sub-call.
            weight_version = installed_version

            _log(
                f"iter {batch_id}: generating {len(prompts_x)} completions "
                f"(weight_version={weight_version}, "
                f"max_new_tokens={max_completion_length})..."
            )
            _t_gen = time.perf_counter()

            all_completions: List[List[int]] = []
            all_logprobs: List[List[float]] = []
            gbs = worker._global_batch_size
            for s in range(0, len(prompts_x), gbs):
                e = min(s + gbs, len(prompts_x))
                sub_completions, sub_logprobs = worker.generate_and_get_log_probs(
                    prompts_x[s:e],
                    max_new_tokens=max_completion_length,
                )
                all_completions.extend(sub_completions)
                all_logprobs.extend(sub_logprobs)

            _gen_s = time.perf_counter() - _t_gen
            avg_len = mean(len(c) for c in all_completions) if all_completions else 0.0
            _log(
                f"iter {batch_id}: generated {len(all_completions)} completions in {_gen_s:.1f}s "
                f"(avg len={avg_len:.1f})"
            )

            logprobs_tensor = _pad_logprobs(all_logprobs, max_completion_length)

            _t_push = time.perf_counter()
            queue.push(
                RolloutBatch(
                    batch_id=batch_id,
                    weight_version=weight_version,
                    prompts=prompts_x,
                    completions=all_completions,
                    logprobs=logprobs_tensor,
                    extra={"answer": list(answers_x)},
                )
            )
            _log(
                f"iter {batch_id}: pushed to rollout queue "
                f"(wait_for_slot={(time.perf_counter() - _t_push):.2f}s, qsize~={queue.qsize()})"
            )

            # Non-blocking poll for fresh weights between rollouts.
            with bridge.poll_weights() as dicts:
                if dicts is not None:
                    _t_apply = time.perf_counter()
                    n = _apply_bridge_dict_to_worker(dicts, worker)
                    if n > 0:
                        installed_version += 1
                        _log(
                            f"iter {batch_id}: installed theta_{installed_version} "
                            f"({n} keys, apply_s={(time.perf_counter() - _t_apply):.2f}s)"
                        )

            # Break on TRAINING_STOPPED.
            ev_opt = channel.poll()
            if ev_opt is not None:
                kind, payload = ev_opt
                if kind == AsyncTrainingEvent.TRAINING_STOPPED:
                    _log("got TRAINING_STOPPED, exiting loop")
                    break
                _log(f"iter {batch_id}: unexpected event {kind.name} payload={payload}")

            batch_id += 1
    finally:
        _log("cleaning up (closing bridge, queue, worker, mesh)...")
        try:
            if bridge is not None:
                bridge.close()
        except Exception as e:  # noqa: BLE001
            _log(f"bridge.close() raised {type(e).__name__}: {e}")
        try:
            if queue is not None:
                queue.close()
        except Exception as e:  # noqa: BLE001
            _log(f"queue.close() raised {type(e).__name__}: {e}")
        worker = None
        gc.collect()
        try:
            ttnn.close_mesh_device(parent_mesh)
        except Exception as e:  # noqa: BLE001
            _log(f"close_mesh_device raised {type(e).__name__}: {e}")
        _log("clean exit")
