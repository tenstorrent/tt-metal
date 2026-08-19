# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile Qwen full-model terminal greedy sampler choices on the target mesh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import HIDDEN_SIZE, MODEL_ID
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import DEFAULT_MAX_BATCH_SIZE, QwenFullModel, _make_sampling_args
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)
from models.common.sampling import SamplingGenerator
from models.common.utility_functions import nearest_32

MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
DEFAULT_OUTPUT = MODEL_DIR / "doc/optimized_full_model/artifacts/terminal_sampler_choices.json"


def _replicate(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _hidden(mesh_device) -> ttnn.Tensor:
    torch.manual_seed(0)
    host = torch.randn((1, 1, 1, HIDDEN_SIZE), dtype=torch.bfloat16)
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate(mesh_device),
    )


def _token_buffer(mesh_device, token: int = 0) -> ttnn.Tensor:
    host = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
    host[0, 0, 0, 0] = int(token)
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate(mesh_device),
    )


def _read_token(token_buffer: ttnn.Tensor) -> int:
    local = ttnn.get_device_tensors(token_buffer)[0]
    return int(ttnn.to_torch(local.cpu()).reshape(-1)[0].item())


def _pad_logits_batch_for_sampling(logits: ttnn.Tensor) -> ttnn.Tensor:
    shape = tuple(int(dim) for dim in logits.shape)
    batch = shape[-2]
    padded_batch = nearest_32(batch)
    if padded_batch == batch:
        return logits
    return ttnn.pad(
        logits,
        padding=[(0, 0), (0, 0), (0, padded_batch - batch), (0, 0)],
        value=0.0,
    )


def _sample_once(sampler: SamplingGenerator, logits: ttnn.Tensor, token_buffer: ttnn.Tensor) -> None:
    sampled = sampler.sample(logits, enable_trace=False, tt_out_tok=token_buffer)
    if isinstance(sampled, tuple):
        sampled = sampled[0]
    ttnn.synchronize_device(sampler.tt_sampling.mesh_device)


def _measure_sampler(
    label: str,
    sampler: SamplingGenerator,
    logits: ttnn.Tensor,
    mesh_device,
    *,
    iterations: int,
) -> dict[str, Any]:
    token_buffer = _token_buffer(mesh_device)
    _sample_once(sampler, logits, token_buffer)
    warm_token = _read_token(token_buffer)
    elapsed = []
    token = warm_token
    for _ in range(iterations):
        start = time.perf_counter()
        _sample_once(sampler, logits, token_buffer)
        elapsed.append(time.perf_counter() - start)
    token = _read_token(token_buffer)
    total_s = sum(elapsed)
    return {
        "label": label,
        "warm_token": int(warm_token),
        "last_token": int(token),
        "iterations": iterations,
        "total_s": total_s,
        "mean_ms": (total_s / iterations) * 1000.0 if iterations > 0 else 0.0,
        "tokens_per_second": iterations / total_s if total_s > 0 else 0.0,
    }


def _force_argmax_sampler(model: QwenFullModel, mesh_device) -> SamplingGenerator:
    args = _make_sampling_args(model.text_config, mesh_device, DEFAULT_MAX_BATCH_SIZE)
    force_args = SimpleNamespace(**vars(args))
    force_args.model_config = {
        **args.model_config,
        "SAMPLING_AG_CONFIG": {
            "allow_force_argmax": True,
            "num_links": 2,
            "topology": ttnn.Topology.Ring,
        },
    }
    return SamplingGenerator(args=force_args, mesh_device=mesh_device, tt_ccl=get_tt_ccl(mesh_device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    add_mesh_device_args(parser)
    args = parser.parse_args()

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        load_start = time.perf_counter()
        model = QwenFullModel.from_hf(
            mesh_device=mesh_device,
            model_id=args.hf_model,
            local_files_only=args.local_files_only,
            load_rope_tables=False,
        )
        load_s = time.perf_counter() - load_start
        hidden = _hidden(mesh_device)
        logits = model.apply_lm_head(hidden)
        padded_logits = _pad_logits_batch_for_sampling(logits)
        host_logits = model.decode_logits_to_torch(logits, batch=1)
        expected = int(torch.argmax(host_logits[0]).item())

        choices: list[dict[str, Any]] = []
        if model.sampling is None:
            raise RuntimeError("terminal sampler profiler requires on-device sampling")
        common = _measure_sampler(
            "common_topk1_composite_gather",
            model.sampling,
            padded_logits,
            mesh_device,
            iterations=args.iterations,
        )
        common["matches_host_argmax"] = common["last_token"] == expected
        choices.append(common)

        try:
            force_sampler = _force_argmax_sampler(model, mesh_device)
            force = _measure_sampler(
                "force_argmax_async_full_vocab_gather",
                force_sampler,
                padded_logits,
                mesh_device,
                iterations=args.iterations,
            )
            force["matches_host_argmax"] = force["last_token"] == expected
            force["force_argmax_active"] = bool(force_sampler.tt_sampling.force_argmax_sampling)
            choices.append(force)
        except Exception as exc:
            rejection = str(exc).split("backtrace:", 1)[0].strip()
            choices.append(
                {
                    "label": "force_argmax_async_full_vocab_gather",
                    "rejected": True,
                    "rejection_summary": rejection,
                    "matches_host_argmax": False,
                }
            )

        valid_choices = [choice for choice in choices if choice.get("matches_host_argmax") and "mean_ms" in choice]
        selected = min(valid_choices, key=lambda choice: choice["mean_ms"])["label"] if valid_choices else None
        payload = {
            "hf_model_id": args.hf_model,
            "mesh_shape": tuple(int(dim) for dim in mesh_device.shape),
            "num_devices": mesh_device.get_num_devices(),
            "load_s": load_s,
            "expected_host_argmax": expected,
            "selected_valid_choice": selected,
            "choices": choices,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    main()
