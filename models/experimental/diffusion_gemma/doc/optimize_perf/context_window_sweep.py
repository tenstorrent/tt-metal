# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure full-depth DiffusionGemma prefill and steady block speed by context.

One invocation builds a single model-owned contiguous KV cache, then measures
each requested prompt prefix serially. Results are checkpointed after every row
so a long-context run retains completed evidence if a later row fails.

*** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
from models.experimental.diffusion_gemma.tt.commit_batched import batched_commit_enabled
from models.experimental.diffusion_gemma.tt.generate import prefill_prompt_tokens, select_denoise_block_fn
from models.experimental.diffusion_gemma.tt.prefill_moe import (
    _find_supported_experts,
    tuned_prefill_moe_enabled,
)
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    self_conditioning_embedding_prechunk_enabled,
    self_conditioning_logits_l1_mode,
)
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.tt.sparse_moe import tuned_configs_enabled

CANVAS_LENGTH = 256
NUM_BLOCKS = 2
MAX_DENOISE_STEPS = 4
DEFAULT_CHECKPOINT = Path("/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("prompt lengths must be positive comma-separated integers")
    if lengths != sorted(set(lengths)):
        raise argparse.ArgumentTypeError("prompt lengths must be unique and increasing")
    return lengths


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _dram_gib(mesh_device) -> dict:
    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 2**30
    return {
        "used": view.num_banks * view.total_bytes_allocated_per_bank / gib,
        "free": view.num_banks * view.total_bytes_free_per_bank / gib,
        "usable_total": view.num_banks * view.total_bytes_per_bank / gib,
        "banks": int(view.num_banks),
    }


def _prompt_prefix(length: int, vocab_size: int) -> torch.Tensor:
    """Deterministic varied token IDs; every shorter request is a true prefix."""

    positions = torch.arange(length, dtype=torch.int64)
    return ((positions * 1_103_515_245 + 12_345) % (vocab_size - 1) + 1).unsqueeze(0)


def _token_sha256(tokens: torch.Tensor) -> str:
    values = tokens.reshape(-1).tolist()
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _emission_summary(emission) -> dict:
    return {
        "block_idx": int(emission.block_idx),
        "start_pos": int(emission.start_pos),
        "next_pos": int(emission.next_pos),
        "latency_s": float(emission.latency_s),
        "denoise_latency_s": float(emission.denoise_latency_s),
        "commit_latency_s": float(emission.commit_latency_s),
        "denoise_steps": int(emission.num_denoise_steps),
        "halted": bool(emission.halted),
        "token_sha256": _token_sha256(emission.tokens),
    }


def _write_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n")


def _validate_contract(args) -> None:
    expected_blocks = 0 if args.prefill_only else NUM_BLOCKS
    if args.num_blocks != expected_blocks:
        if args.prefill_only:
            raise ValueError("prefill-only comparison requires num_blocks=0")
        raise ValueError(f"comparison table requires exactly {NUM_BLOCKS} blocks")
    if args.canvas_length != CANVAS_LENGTH:
        raise ValueError(f"comparison table requires canvas_length={CANVAS_LENGTH}")
    if not args.prefill_only and args.max_denoise_steps != MAX_DENOISE_STEPS:
        raise ValueError(f"comparison table requires max_denoise_steps={MAX_DENOISE_STEPS}")
    for prompt_len in args.prompt_lengths:
        aligned = ((prompt_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if aligned + args.num_blocks * args.canvas_length > args.max_seq_len:
            raise ValueError(
                f"prompt {prompt_len} aligns to {aligned}; two committed blocks exceed "
                f"max_seq_len={args.max_seq_len}"
            )
    if not args.prefill_only and os.environ.get("DG_SPARSE_MOE") != "1":
        raise ValueError("set DG_SPARSE_MOE=1 for the comparison contract")
    if not args.prefill_only and os.environ.get("DG_DEDUP_ARGMAX") != "1":
        raise ValueError("set DG_DEDUP_ARGMAX=1 for the comparison contract")
    if not tuned_prefill_moe_enabled():
        raise ValueError("tuned prefill MoE must resolve enabled")
    if not args.prefill_only and not tuned_configs_enabled():
        raise ValueError("tuned sparse denoise MoE must resolve enabled")


def run(args) -> dict:
    _validate_contract(args)
    result = {
        "schema_version": 1,
        "status": "running",
        "label": args.label,
        "tt_metal_head": _git_head(args.tt_metal),
        "checkpoint": str(args.checkpoint),
        "hardware": {
            "mesh": args.mesh,
            "mesh_shape": [1, 4],
            "architecture": "Blackhole",
            "tensor_parallel": 4,
        },
        "contract": {
            "num_layers": 30,
            "max_seq_len": args.max_seq_len,
            "prompt_lengths": args.prompt_lengths,
            "prompt_recipe": "token[i] = (i*1103515245 + 12345) % (vocab_size-1) + 1",
            "canvas_length": args.canvas_length,
            "num_blocks": args.num_blocks,
            "prefill_only": args.prefill_only,
            "max_denoise_steps": args.max_denoise_steps,
            "gumbel_mode": "argmax",
            "seed": args.seed,
            "steady_block": 1,
            "timing": "synchronized device wall time",
        },
        "environment": {
            name: os.environ.get(name, "<unset>")
            for name in (
                "DG_PREFILL_MOE_TUNED",
                "DG_SPARSE_MOE",
                "DG_SPARSE_MOE_TUNED",
                "DG_SPARSE_MOE_CAPACITY",
                "DG_DEDUP_ARGMAX",
                "DG_COMMIT_BATCHED",
                "DG_SELFCOND_PRECHUNK_EMBED",
                "DG_SELFCOND_LOGITS_L1",
                "DG_NORM_FULLCANVAS",
                "DG_MOE_L1",
                "DG_DENOISE_TRACED",
                "DG_DENOISE_TRACED_MULTISTEP",
                "DG_DENOISE_EARLY_HALT",
                "DG_PREFIX_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        "resolved_defaults": {
            "prefill_moe_tuned": tuned_prefill_moe_enabled(),
            "sparse_moe_tuned": tuned_configs_enabled(),
            "batched_commit": batched_commit_enabled(),
            "selfcond_prechunk": self_conditioning_embedding_prechunk_enabled(),
            "selfcond_logits_l1": self_conditioning_logits_l1_mode(),
            "denoise_block_fn": select_denoise_block_fn().__name__,
        },
        "model_build": {},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_result(args.output, result)

    mesh_device = _open_mesh_device(args.mesh)
    try:
        result["model_build"]["dram_before"] = _dram_gib(mesh_device)
        build_t0 = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device,
            args.checkpoint,
            tokenizer_kwargs={"local_files_only": True},
            max_seq_len=args.max_seq_len,
            create_kv_cache=True,
        )
        ttnn.synchronize_device(mesh_device)
        result["model_build"]["elapsed_s"] = time.perf_counter() - build_t0
        result["model_build"]["dram_after"] = _dram_gib(mesh_device)
        result["model_build"]["num_layers"] = len(bundle.tt_model.layers)
        if len(bundle.tt_model.layers) != 30:
            raise AssertionError(f"expected full 30-layer model, got {len(bundle.tt_model.layers)}")

        supported_experts = _find_supported_experts(bundle.tt_model)
        if supported_experts is None or len(supported_experts) != 30:
            raise AssertionError("production model did not satisfy the tuned-prefill support guard")
        result["model_build"]["prefill_expert_layers_guarded"] = len(supported_experts)
        _write_result(args.output, result)

        vocab_size = int(bundle.tt_model.hf_config.vocab_size)
        for prompt_len in args.prompt_lengths:
            prompt_tokens = _prompt_prefix(prompt_len, vocab_size)
            if args.prefill_only:
                ttnn.synchronize_device(mesh_device)
                prefill_t0 = time.perf_counter()
                prefill = prefill_prompt_tokens(bundle.tt_model, prompt_tokens)
                ttnn.synchronize_device(mesh_device)
                prefill_s = time.perf_counter() - prefill_t0
                dram_after_prefill = _dram_gib(mesh_device)
                row = {
                    "prompt_len": prompt_len,
                    "cache_len": int(prefill.cache_len),
                    "prompt_token_sha256": _token_sha256(prompt_tokens),
                    "prefill_s": prefill_s,
                    "prefill_tokens_per_s": prompt_len / prefill_s,
                    "dram_after_prefill": dram_after_prefill,
                    "blocks": [],
                    "final_next_pos": int(prefill.cache_len),
                }
                result["rows"].append(row)
                _write_result(args.output, result)
                print(
                    "DG_CONTEXT_PREFILL_ROW "
                    + json.dumps(
                        {
                            "max_seq_len": args.max_seq_len,
                            "prompt_len": prompt_len,
                            "prefill_s": round(prefill_s, 6),
                            "prefill_tokens_per_s": round(prompt_len / prefill_s, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            session = BlockDiffusionServingSession(
                bundle.tt_model,
                bundle.state_dict,
                config=DiffusionConfig(
                    canvas_length=args.canvas_length,
                    max_denoise_steps=args.max_denoise_steps,
                ),
                vocab_size=vocab_size,
                gumbel_mode="argmax",
                seed=args.seed,
                stop_token_ids=[],
            )
            try:
                ttnn.synchronize_device(mesh_device)
                prefill_t0 = time.perf_counter()
                cache_len = session.prefill(prompt_tokens)
                ttnn.synchronize_device(mesh_device)
                prefill_s = time.perf_counter() - prefill_t0
                dram_after_prefill = _dram_gib(mesh_device)

                emissions = []
                synchronized_block_times = []
                for _ in range(args.num_blocks):
                    ttnn.synchronize_device(mesh_device)
                    block_t0 = time.perf_counter()
                    emission = session.decode_block()
                    ttnn.synchronize_device(mesh_device)
                    synchronized_block_times.append(time.perf_counter() - block_t0)
                    emissions.append(emission)

                blocks = [_emission_summary(emission) for emission in emissions]
                for block, synchronized_s in zip(blocks, synchronized_block_times):
                    block["synchronized_wall_s"] = synchronized_s
                steady_s = synchronized_block_times[1]
                row = {
                    "prompt_len": prompt_len,
                    "cache_len": int(cache_len),
                    "prompt_token_sha256": _token_sha256(prompt_tokens),
                    "prefill_s": prefill_s,
                    "prefill_tokens_per_s": prompt_len / prefill_s,
                    "dram_after_prefill": dram_after_prefill,
                    "blocks": blocks,
                    "steady_block_idx": 1,
                    "steady_block_s": steady_s,
                    "steady_output_tokens_per_s": args.canvas_length / steady_s,
                    "steady_denoise_steps": blocks[1]["denoise_steps"],
                    "steady_halted": blocks[1]["halted"],
                    "final_next_pos": int(session.next_pos),
                }
                result["rows"].append(row)
                _write_result(args.output, result)
                print(
                    "DG_CONTEXT_WINDOW_ROW "
                    + json.dumps(
                        {
                            "max_seq_len": args.max_seq_len,
                            "prompt_len": prompt_len,
                            "prefill_s": round(prefill_s, 6),
                            "prefill_tokens_per_s": round(prompt_len / prefill_s, 3),
                            "steady_block_s": round(steady_s, 6),
                            "steady_output_tokens_per_s": round(args.canvas_length / steady_s, 3),
                            "steady_denoise_steps": blocks[1]["denoise_steps"],
                            "steady_halted": blocks[1]["halted"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                session.reset()

        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _close_mesh_device(mesh_device)
        result["completed_prompt_lengths"] = [row["prompt_len"] for row in result["rows"]]
        _write_result(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--prompt-lengths", type=_parse_lengths, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tt-metal", type=Path, default=Path("/home/zni/tt-metal"))
    parser.add_argument("--mesh", default="P150x4")
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--max-denoise-steps", type=int, default=MAX_DENOISE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        "DG_CONTEXT_WINDOW_SWEEP "
        + json.dumps(
            {
                "status": result["status"],
                "max_seq_len": args.max_seq_len,
                "completed_prompt_lengths": result["completed_prompt_lengths"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
