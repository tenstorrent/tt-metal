#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO on GSM8K, async-rollout variant.

Two-rank tt-run entrypoint (rank 0 = ttml Qwen3 policy + GRPOTrainer, rank 1
= TttGenerationWorker on the rollout mesh); reward is the sum of five signals
(correctness, xmlcount, soft_format, strict_format, int) dispatched by the
framework and logged as ``{name}_mean`` columns.

Run:
    tt-train/sources/examples/grpo_remote_rollout/gsm8k/runner.sh
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
from ttml.trainers import GRPOTrainer, get_grpo_config
from utils.mpi_rollout import MPIRolloutClient, MPIRolloutServer
from utils.qwen3_grpo_completer import Qwen3CompleterRemoteRollout, Qwen3CompletionCtx
from utils.qwen3_ttt_presets import bf16_attn_bfp8_mlp_optimizations, qwen3_stop_and_pad
from utils.ttt_generation_worker import TttGenerationWorker
from utils.weight_bridge import HostWeightBridge, TTML_RANK, TTT_RANK

CONFIG_REL = "tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_remote_rollout.yaml"
REPO_ROOT = Path(__file__).resolve().parents[5]

DATASET = "openai/gsm8k"
DATASET_SPLIT = "train"

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "Respond in the following format:\n" f"{THINK_OPEN}\n...\n{THINK_CLOSE}\n" f"{ANSWER_OPEN}\n...\n{ANSWER_CLOSE}\n"
)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)


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


def build_dataset(seed: int):
    ds = load_dataset(DATASET, "main", split=DATASET_SPLIT)

    def to_example(row):
        return {
            "prompt": f"{SYSTEM_PROMPT}\nQuestion: {row['question']}\nAnswer:",
            "answer": extract_hash_answer(row["answer"]),
        }

    return ds.shuffle(seed=seed).map(to_example, remove_columns=ds.column_names)


def get_output_dir() -> str:
    return os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_gsm8k_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


class WeightSyncCallback:
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
        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

        dataset = build_dataset(seed=int(raw["training_config"].get("seed", 0)))

        output_dir = get_output_dir()
        grpo_config = get_grpo_config(raw, output_dir=output_dir)
        optimizer_dict = raw["training_config"]["optimizer"]
        transformer_config = get_model_config(raw["training_config"]["model_config"])

        completer = Qwen3CompleterRemoteRollout(
            ctx=Qwen3CompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            mesh_device=mesh_device,
            model_source=model_id,
            inference_client=client,
            enable_ddp=device_config.enable_ddp,
        )

        client.connect()
        completer.push_weights()

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_funcs=REWARD_FUNCS,
            optimizer_dict=optimizer_dict,
            callbacks=[WeightSyncCallback(completer, every=weight_sync_every)],
            model_source=model_id,
        )
        trainer.train()
    finally:
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
        stop_token_ids, pad_token_id = qwen3_stop_and_pad(model_id)

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

        bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=worker.submeshes)

        server = MPIRolloutServer(
            peer_rank=TTML_RANK,
            bridge=bridge,
            generate_fn=worker.generate,
            on_weights_received=worker.update_weights,
        )
        server.connect()
        server.serve_forever()
    finally:
        worker = None
        server = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"gsm8k_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use gsm8k/runner.sh."
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
