# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Signposted profiler harness for the Qwen full-model terminal path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import HIDDEN_SIZE, MODEL_ID
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenFullModel
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)
from models.common.utility_functions import nearest_32

MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
DEFAULT_OUTPUT = MODEL_DIR / "doc/full_model/artifacts/terminal_path_profile_summary.json"


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


def _sample(model: QwenFullModel, logits: ttnn.Tensor, token_buffer: ttnn.Tensor) -> None:
    if model.sampling is None:
        raise RuntimeError("terminal-path profile requires on-device sampling")
    sampled = model.sampling.sample(logits, enable_trace=False, tt_out_tok=token_buffer)
    if isinstance(sampled, tuple):
        sampled = sampled[0]


def _measure(label: str, func, mesh_device) -> float:
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    func()
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    print(f"{label}_s={elapsed:.6f}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        token_buffer = _token_buffer(mesh_device)

        logits = model.apply_lm_head(hidden)
        padded_logits = _pad_logits_batch_for_sampling(logits)
        _sample(model, padded_logits, token_buffer)
        ttnn.synchronize_device(mesh_device)

        terminal_s = _measure(
            "terminal_path",
            lambda: _sample(model, _pad_logits_batch_for_sampling(model.apply_lm_head(hidden)), token_buffer),
            mesh_device,
        )

        signpost(header="QWEN_LM_HEAD_ONLY")
        lm_head_s = _measure("lm_head", lambda: model.apply_lm_head(hidden), mesh_device)
        signpost(header="QWEN_LM_HEAD_ONLY_END")

        logits_for_sampler = _pad_logits_batch_for_sampling(model.apply_lm_head(hidden))
        signpost(header="QWEN_SAMPLER_ONLY")
        sampler_s = _measure("sampler", lambda: _sample(model, logits_for_sampler, token_buffer), mesh_device)
        signpost(header="QWEN_SAMPLER_ONLY_END")

        signpost(header="QWEN_FULL_DECODE_TERMINAL")
        terminal_window_s = _measure(
            "terminal_path_signposted",
            lambda: _sample(model, _pad_logits_batch_for_sampling(model.apply_lm_head(hidden)), token_buffer),
            mesh_device,
        )
        signpost(header="QWEN_FULL_DECODE_TERMINAL_END")

        payload = {
            "hf_model_id": args.hf_model,
            "load_s": load_s,
            "terminal_path_s": terminal_s,
            "lm_head_s": lm_head_s,
            "sampler_s": sampler_s,
            "terminal_path_signposted_s": terminal_window_s,
            "sampler_fraction_of_terminal": sampler_s / terminal_s if terminal_s else None,
            "mesh_shape": tuple(int(dim) for dim in mesh_device.shape),
            "num_devices": mesh_device.get_num_devices(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    main()
