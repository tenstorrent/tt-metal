# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure fixed-K traced block speed across live DiffusionGemma prefix lengths.

The production trace always captures the released 48-step schedule.  This
benchmark leaves that capture unchanged, but replaces the content-dependent halt
decision with a fixed replay count so every row performs the same amount of
denoise work.  The override is process-local and is never part of serving.

One 256K model-owned hybrid KV cache and one 256K reveal trace are reused across
all rows.  A requested 256K context uses a 256K-minus-one-canvas prefix so block
0 remains within the advertised context bound.

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
from models.experimental.diffusion_gemma.tt import generator_vllm
from models.experimental.diffusion_gemma.tt import traced_denoise
from models.experimental.diffusion_gemma.tt.hybrid_kv import (
    attach_model_owned_hybrid_kv,
    model_owned_hybrid_kv_model_kwargs,
)


DEFAULT_CONTEXT_BOUNDS = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)
DEFAULT_CHECKPOINT = Path("/home/zni/dg_models/diffusiongemma-26B-A4B-it")
DEFAULT_TRACE_REGION_SIZE = 2684354560  # 2.5 GiB, validated through 256K on QB2.
CANVAS_LENGTH = 256


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("context bounds must be positive comma-separated integers")
    if lengths != sorted(set(lengths)):
        raise argparse.ArgumentTypeError("context bounds must be unique and increasing")
    return lengths


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _prompt_prefix(length: int, vocab_size: int) -> torch.Tensor:
    """Return deterministic varied IDs without allocating tokenizer-sized text."""

    positions = torch.arange(length, dtype=torch.int64)
    return ((positions * 1_103_515_245 + 12_345) % (vocab_size - 1) + 1).unsqueeze(0)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def _effective_prompt_len(context_bound: int, *, max_seq_len: int, canvas_length: int) -> int:
    if context_bound > max_seq_len:
        raise ValueError(f"context bound {context_bound} exceeds max_seq_len={max_seq_len}")
    return min(context_bound, max_seq_len - canvas_length)


def _configure_environment(args, prompt_lengths: list[int], *, reveal_pmax: int) -> None:
    os.environ["DG_TRACE_REGION_SIZE"] = str(args.trace_region_size)
    os.environ["DG_UPFRONT_CAPTURE"] = "1"
    os.environ["DG_VLLM_GUMBEL_MODE"] = "device"
    os.environ["DG_MODEL_OWNED_HYBRID_KV"] = "1"
    os.environ["DG_DENOISE_REVEAL_PMAX"] = str(reveal_pmax)
    os.environ["DG_UPFRONT_PREFILL_WARMUP_LENS"] = ",".join(str(value) for value in prompt_lengths)
    os.environ["DG_UPFRONT_STRICT_PREFILL_LENS"] = "1"
    # This is a throughput benchmark, not a qualitative run.  A policy-triggered
    # retry would execute more than K denoise steps and invalidate the comparison.
    os.environ["DG_DEGENERACY_POLICY"] = "off"


def run(args) -> dict:
    if not 1 <= args.k <= traced_denoise.UPFRONT_DENOISE_STEPS:
        raise ValueError(f"k must be in [1, {traced_denoise.UPFRONT_DENOISE_STEPS}], got {args.k}")
    if not 1 <= args.warmup_k <= traced_denoise.UPFRONT_DENOISE_STEPS:
        raise ValueError(f"warmup_k must be in [1, {traced_denoise.UPFRONT_DENOISE_STEPS}], got {args.warmup_k}")
    if args.canvas_length != CANVAS_LENGTH:
        raise ValueError(f"this comparison requires canvas_length={CANVAS_LENGTH}")
    if args.max_seq_len % ttnn.TILE_SIZE != 0:
        raise ValueError(f"max_seq_len must be a {ttnn.TILE_SIZE}-token multiple")
    reveal_pmax = args.max_seq_len if args.reveal_pmax is None else int(args.reveal_pmax)
    if reveal_pmax <= 0 or reveal_pmax % ttnn.TILE_SIZE != 0:
        raise ValueError(f"reveal_pmax must be a positive {ttnn.TILE_SIZE}-token multiple")
    if reveal_pmax > args.max_seq_len:
        raise ValueError(f"reveal_pmax={reveal_pmax} exceeds max_seq_len={args.max_seq_len}")

    prompt_lengths = [
        _effective_prompt_len(
            context_bound,
            max_seq_len=args.max_seq_len,
            canvas_length=args.canvas_length,
        )
        for context_bound in args.context_bounds
    ]
    if any(length <= 0 or length % ttnn.TILE_SIZE != 0 for length in prompt_lengths):
        raise ValueError(f"effective prompt lengths must be positive tile multiples, got {prompt_lengths}")
    if len(set(prompt_lengths)) != len(prompt_lengths):
        raise ValueError(f"context bounds collapse to duplicate effective prompt lengths: {prompt_lengths}")
    _configure_environment(args, prompt_lengths, reveal_pmax=reveal_pmax)

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
            "reveal_pmax": reveal_pmax,
            "context_bounds": args.context_bounds,
            "effective_prompt_lengths": prompt_lengths,
            "canvas_length": args.canvas_length,
            "fixed_k": args.k,
            "commit_warmup_k": args.warmup_k,
            "captured_schedule_steps": traced_denoise.UPFRONT_DENOISE_STEPS,
            "gumbel_mode": "device",
            "hybrid_kv": True,
            "block": "block 0 after model-lifetime trace capture",
            "timing": "serving BlockEmission wall time; trace replay includes per-step halt readback",
            "fixed_k_method": (
                "process-local traced_denoise.eval_halt override; production source unchanged; "
                "one warmup block at each start position compiles commit programs before the measured block"
            ),
            "prompt_recipe": "token[i] = (i*1103515245 + 12345) % (vocab_size-1) + 1",
        },
        "environment": {
            name: os.environ.get(name, "<unset>")
            for name in (
                "DG_TRACE_REGION_SIZE",
                "DG_UPFRONT_CAPTURE",
                "DG_VLLM_GUMBEL_MODE",
                "DG_MODEL_OWNED_HYBRID_KV",
                "DG_DENOISE_REVEAL_PMAX",
                "DG_UPFRONT_PREFILL_WARMUP_LENS",
                "DG_UPFRONT_STRICT_PREFILL_LENS",
                "DG_DEGENERACY_POLICY",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        "model_build": {},
        "trace_capture": {},
        "rows": [],
    }
    _write_result(args.output, result)

    metric_events: list[dict] = []
    original_metric = generator_vllm._metric
    original_eval_halt = traced_denoise.eval_halt
    step_limit = {"value": args.warmup_k}

    def collect_metric(event: str, **fields) -> None:
        metric_events.append({"event": event, **fields})
        original_metric(event, **fields)

    def fixed_k_halt(
        mean_entropy: float,
        mismatch: float,
        step: int,
        *,
        threshold: float,
        n_stable: int = 1,
    ) -> bool:
        del mean_entropy, mismatch, threshold, n_stable
        return step + 1 >= step_limit["value"]

    generator_vllm._metric = collect_metric
    traced_denoise.eval_halt = fixed_k_halt

    mesh_device = None
    wrapper = None
    try:
        mesh_device = _open_mesh_device(args.mesh)
        build_t0 = time.perf_counter()
        model_kwargs = {
            "max_seq_len": args.max_seq_len,
            "max_batch_size": 1,
        }
        model_kwargs.update(
            model_owned_hybrid_kv_model_kwargs(
                max_seq_len=args.max_seq_len,
                max_batch_size=1,
            )
        )
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device,
            args.checkpoint,
            tokenizer_kwargs={"local_files_only": True},
            **model_kwargs,
        )
        page_tables_per_layer = attach_model_owned_hybrid_kv(
            bundle.tt_model,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
        )
        ttnn.synchronize_device(mesh_device)
        result["model_build"] = {
            "elapsed_s": time.perf_counter() - build_t0,
            "num_layers": len(bundle.tt_model.layers),
            "dram": generator_vllm._dram_snapshot(mesh_device, synchronize=False),
        }
        if len(bundle.tt_model.layers) != 30:
            raise AssertionError(f"expected full 30-layer model, got {len(bundle.tt_model.layers)}")
        _write_result(args.output, result)

        wrapper = generator_vllm.DiffusionGemmaForCausalLM(
            [bundle.tt_model],
            [bundle.model_args],
            mesh_device,
            dg_state_dict=bundle.state_dict,
            tokenizer=bundle.tokenizer,
            config=DiffusionConfig(),
            gumbel_mode="device",
            max_model_len=args.max_seq_len,
            page_tables_per_layer=page_tables_per_layer,
        )

        capture_t0 = time.perf_counter()
        wrapper.warmup_model_prefill(None, False, True)
        wrapper.warmup_model_prefill(None, True, True)
        ttnn.synchronize_device(mesh_device)
        controller = wrapper._persistent_adapter._upfront_traced_denoise_controller
        result["trace_capture"] = {
            "elapsed_s": time.perf_counter() - capture_t0,
            "stats": controller.stats(),
            "warmup_prompt_lengths": sorted(wrapper._upfront_prefill_warmup_lens),
            "dram": generator_vllm._dram_snapshot(mesh_device, synchronize=False),
        }
        if controller.traces_captured != traced_denoise.UPFRONT_DENOISE_STEPS:
            raise AssertionError(
                f"expected {traced_denoise.UPFRONT_DENOISE_STEPS} traces, got {controller.traces_captured}"
            )
        _write_result(args.output, result)

        vocab_size = int(bundle.tt_model.hf_config.vocab_size)
        for context_bound, prompt_len in zip(args.context_bounds, prompt_lengths):
            prompt_tokens = _prompt_prefix(prompt_len, vocab_size)

            # Commit selects programs from the absolute start position.  Run one
            # cheap block first so the measured K block excludes compilation.
            step_limit["value"] = args.warmup_k
            warmup_event_start = len(metric_events)
            ttnn.synchronize_device(mesh_device)
            warmup_t0 = time.perf_counter()
            wrapper.prefill_forward(prompt_tokens, prompt_lens=[prompt_len])
            ttnn.synchronize_device(mesh_device)
            warmup_wall_s = time.perf_counter() - warmup_t0
            warmup_event = next(
                (
                    event
                    for event in reversed(metric_events[warmup_event_start:])
                    if event.get("event") == "prefill_block0"
                ),
                None,
            )
            if warmup_event is None:
                raise AssertionError(f"missing commit-warmup metric for context {context_bound}")
            if int(warmup_event["denoise_steps"]) != args.warmup_k:
                raise AssertionError(
                    f"context {context_bound} warmup ran {warmup_event['denoise_steps']} steps, "
                    f"expected {args.warmup_k}"
                )
            commit_warmup = {
                "request_wall_s": warmup_wall_s,
                "prefill_s": float(warmup_event["prefill_s"]),
                "block_latency_s": float(warmup_event["block_latency_s"]),
                "denoise_latency_s": float(warmup_event["denoise_latency_s"]),
                "commit_latency_s": float(warmup_event["commit_latency_s"]),
                "denoise_steps": int(warmup_event["denoise_steps"]),
            }
            wrapper.release_request(0)

            step_limit["value"] = args.k
            event_start = len(metric_events)
            ttnn.synchronize_device(mesh_device)
            request_t0 = time.perf_counter()
            block = wrapper.prefill_forward(prompt_tokens, prompt_lens=[prompt_len])
            ttnn.synchronize_device(mesh_device)
            request_wall_s = time.perf_counter() - request_t0

            block_event = next(
                (event for event in reversed(metric_events[event_start:]) if event.get("event") == "prefill_block0"),
                None,
            )
            if block_event is None:
                raise AssertionError(f"missing prefill_block0 metric for context {context_bound}")
            if int(block_event["denoise_steps"]) != args.k:
                raise AssertionError(
                    f"context {context_bound} ran {block_event['denoise_steps']} denoise steps, expected {args.k}"
                )
            block_latency_s = float(block_event["block_latency_s"])
            denoise_latency_s = float(block_event["denoise_latency_s"])
            commit_latency_s = float(block_event["commit_latency_s"])
            row = {
                "context_bound": context_bound,
                "prompt_len": prompt_len,
                "cache_len": int(block_event["cache_len"]),
                "start_pos": int(block_event["start_pos"]),
                "next_pos": int(block_event["next_pos"]),
                "prefill_s": float(block_event["prefill_s"]),
                "ttft_s": float(block_event["ttft_s"]),
                "request_wall_s": request_wall_s,
                "block_latency_s": block_latency_s,
                "denoise_latency_s": denoise_latency_s,
                "commit_latency_s": commit_latency_s,
                "denoise_steps": int(block_event["denoise_steps"]),
                "ms_per_denoise_step": 1000.0 * denoise_latency_s / args.k,
                "tokens_per_block_per_s": args.canvas_length / block_latency_s,
                "halted": bool(block_event["halted"]),
                "committed_sha256": _sha256_tensor(block),
                "dram": block_event["dram"],
                "commit_warmup": commit_warmup,
            }
            result["rows"].append(row)
            _write_result(args.output, result)
            print("DG_CONTEXT_K_ROW " + json.dumps(row, sort_keys=True), flush=True)
            wrapper.release_request(0)

        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        generator_vllm._metric = original_metric
        traced_denoise.eval_halt = original_eval_halt
        if wrapper is not None:
            wrapper.release_persistent_capture()
        if mesh_device is not None:
            _close_mesh_device(mesh_device)
        result["completed_context_bounds"] = [row["context_bound"] for row in result["rows"]]
        _write_result(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-bounds",
        type=_parse_lengths,
        default=list(DEFAULT_CONTEXT_BOUNDS),
        help="comma-separated advertised context bounds; max_seq_len maps to max_seq_len-canvas prefix",
    )
    parser.add_argument("--max-seq-len", type=int, default=262144)
    parser.add_argument(
        "--reveal-pmax",
        type=int,
        default=None,
        help="fixed traced prefix geometry (default: max-seq-len); useful when physical KV must be larger",
    )
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--warmup-k", type=int, default=1)
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACE_REGION_SIZE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tt-metal", type=Path, default=Path("/home/zni/tt-metal"))
    parser.add_argument("--mesh", default="P150x4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="hybrid-kv-upfront-fixed-k-context-sweep")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        "DG_CONTEXT_K_SWEEP "
        + json.dumps(
            {
                "status": result["status"],
                "fixed_k": args.k,
                "completed_context_bounds": result["completed_context_bounds"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
