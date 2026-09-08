#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real queue-driven fully asynchronous GRPO on GSM8K (two MPI ranks)."""

from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttml
import ttnn
from datasets import load_dataset
from ttml.common.config import DeviceConfig, get_model_config, load_config
from ttml.trainers import FullyAsyncGRPOTrainer, get_grpo_config
from utils.async_weight_bridge import AsyncHostWeightBridge
from utils.fully_async_rollout import FullyAsyncRolloutClient, FullyAsyncRolloutWorker
from utils.mpi_rollout_transport import MPIRolloutTrainerTransport, MPIRolloutWorkerTransport
from utils.qwen3_grpo_completer import Qwen3CompleterRemoteRollout, Qwen3CompletionCtx
from utils.qwen3_ttt_presets import bf16_attn_bfp8_mlp_optimizations, qwen3_stop_and_pad
from utils.ttt_generation_worker import TttGenerationWorker
from utils.weight_bridge import TTML_RANK, TTT_RANK

REPO_ROOT = Path(__file__).resolve().parents[5]
CONFIG_REL = "tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_fully_async.yaml"
SYSTEM_PROMPT = "Respond in the format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>\n"
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _number(value: str) -> str:
    value = value.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return value


def _answer(text: str) -> str | None:
    if "<answer>" not in text:
        return None
    matches = _NUM_RE.findall(text.split("<answer>")[-1].split("</answer>")[0])
    return _number(matches[-1]) if matches else None


def correctness_reward(completions, answer, **_kwargs):
    return [2.0 if _answer(completion) == gold else 0.0 for completion, gold in zip(completions, answer)]


def format_reward(completions, **_kwargs):
    return [
        0.5 if all(tag in value for tag in ("<think>", "</think>", "<answer>", "</answer>")) else 0.0
        for value in completions
    ]


def build_dataset(seed: int):
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    def convert(row):
        return {
            "prompt": f"{SYSTEM_PROMPT}Question: {row['question']}\nAnswer:",
            "answer": _number(row["answer"].split("####")[-1]),
        }

    return dataset.shuffle(seed=seed).map(convert, remove_columns=dataset.column_names)


def _config():
    raw = load_config(os.path.join(REPO_ROOT, CONFIG_REL))
    return DeviceConfig(raw), raw


def _model_id(raw: dict) -> str:
    """Allow a local/custom Qwen3 SFT directory without editing the YAML."""
    return os.environ.get("TT_TRAIN_MODEL_ID", raw["training_config"]["model_id"])


def _trainer_main() -> None:
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.initialize_distributed_context(*sys.argv)
    device_config, raw = _config()
    ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    mesh = ctx.get_device()
    bridge = None
    transport = None
    completer = None
    try:
        model_id = _model_id(raw)
        grpo = get_grpo_config(raw, output_dir=str(REPO_ROOT / "generated/tt-train/gsm8k_fully_async"))
        completer = Qwen3CompleterRemoteRollout(
            Qwen3CompletionCtx(grpo.max_completion_length, grpo.temperature, grpo.num_generations),
            get_model_config(raw["training_config"]["model_config"]),
            mesh_device=mesh,
            model_source=model_id,
            enable_ddp=device_config.enable_ddp,
        )
        bridge = AsyncHostWeightBridge.init_sender(peer_rank=TTT_RANK)
        transport = MPIRolloutTrainerTransport(
            peer_rank=TTT_RANK,
            capacity=int(raw["training_config"]["rollout_queue_capacity"]),
        )
        bridge.connect()
        transport.start()
        client = FullyAsyncRolloutClient(
            transport=transport,
            weight_bridge=bridge,
            weight_export=completer.export_weights,
            num_generations=grpo.num_generations,
        )
        completer.set_rollout_client(client)
        FullyAsyncGRPOTrainer(
            completer=completer,
            dataset=build_dataset(int(raw["training_config"].get("seed", 0))),
            config=grpo,
            reward_funcs=[correctness_reward, format_reward],
            optimizer_dict=raw["training_config"]["optimizer"],
            callbacks=[],
            model_source=model_id,
        ).train()
    finally:
        completer = None
        gc.collect()
        ctx.close_device()


def _rollout_main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    _, raw = _config()
    model_id = _model_id(raw)
    rr = raw["remote_rollout_config"]
    grpo = raw["training_config"]["grpo_config"]
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*rr["mesh_shape"]), offset=ttnn.MeshCoordinate(0, 0))
    worker = None
    try:
        stop_ids, pad_id = qwen3_stop_and_pad(model_id)
        worker = TttGenerationWorker(
            mesh_device=mesh,
            model_source=model_id,
            max_batch_size=rr["max_batch_size"],
            max_seq_len=rr["max_seq_len"],
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
            temperature=float(grpo["temperature"]),
            top_k=32,
            top_p=1.0,
            seed=None,
            return_logprobs=True,
            dummy_weights=True,
        )
        bridge = AsyncHostWeightBridge.init_receiver(peer_rank=TTML_RANK, submeshes=worker.submeshes)
        transport = MPIRolloutWorkerTransport(
            peer_rank=TTML_RANK,
            capacity=int(raw["training_config"]["rollout_queue_capacity"]),
        )
        bridge.connect()
        transport.start()
        FullyAsyncRolloutWorker(
            transport=transport,
            weight_bridge=bridge,
            worker=worker,
            max_new_tokens=int(grpo["max_completion_length"]),
        ).serve_forever()
        bridge.close()
    finally:
        worker = None
        gc.collect()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    if int(ttnn.distributed_context_get_size()) != 2:
        raise RuntimeError("gsm8k fully async requires exactly two tt-run ranks")
    if int(ttnn.distributed_context_get_rank()) == TTML_RANK:
        _trainer_main()
    else:
        _rollout_main()
