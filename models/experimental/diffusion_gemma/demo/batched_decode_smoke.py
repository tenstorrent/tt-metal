# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched canvas decode independence check for DiffusionGemma (#47557).

Drives ``B=2`` canvases through **one** denoise loop (batched decision kernels) and asserts the
committed clean-argmax for each canvas is **bit-identical** to running that canvas alone as ``B=1``
— i.e. no cross-canvas leakage. The two canvases are seeded **differently**, so any leakage would
change a committed token. Argmax (RUN-first) sampling + a fixed step budget keep the run
deterministic and per-row independent (no RNG coupling, no batch-coupled early-halt).

This is the model-side batched denoise (the vLLM multi-request wiring is #47488). It does not commit
K/V to the cache (that needs B caches = #47488); the committed argmax tokens are the model output.

Opt-in behind ``DG_BATCH_DECODE=1``. Emits a greppable ``DG_BATCH_DECODE_SUCCESS`` /
``DG_BATCH_DECODE_MISMATCH`` / ``DG_BATCH_DECODE_FAILURE`` line.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt.batched_decode import (
    batched_decode_enabled,
    run_batched_denoise_block,
)
from models.experimental.diffusion_gemma.tt.generate import host_canvas_to_device, tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"),
    )
    parser.add_argument("--mesh", default="P150x4")
    parser.add_argument("--num-layers", type=int, default=2, help="reduced layer count (default 2)")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--prompt", default="Explain diffusion models briefly.")
    parser.add_argument("--batch", type=int, default=2, help="number of canvases (B)")
    parser.add_argument("--canvas-length", type=int, default=64, help="canvas size (32-tile multiple)")
    parser.add_argument("--max-denoising-steps", type=int, default=3)
    parser.add_argument("--mode", default="loop", choices=["loop", "dim0"], help="batched logits mode")
    parser.add_argument("--probe-dim0", action="store_true", help="also try mode=dim0 (non-gating)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metrics-json", default=None)
    return parser


def _host_canvases(batch: int, canvas_len: int, vocab_size: int, seed: int) -> torch.Tensor:
    """B distinct random canvases ``[B, C]`` (different per row so leakage is observable)."""
    rows = []
    for row in range(batch):
        gen = torch.Generator().manual_seed(seed + 1000 * (row + 1))
        rows.append(torch.randint(0, vocab_size, (canvas_len,), generator=gen))
    return torch.stack(rows, dim=0)


def _host_noise(batch: int, canvas_len: int, vocab_size: int, steps: int, seed: int) -> list[torch.Tensor]:
    """Per-step random renoise tokens ``steps × [B, C]`` (distinct per row)."""
    noise = []
    for step in range(steps):
        rows = []
        for row in range(batch):
            gen = torch.Generator().manual_seed(seed + 7919 * (step + 1) + 1000 * (row + 1))
            rows.append(torch.randint(0, vocab_size, (canvas_len,), generator=gen))
        noise.append(torch.stack(rows, dim=0))
    return noise


def _noise_tokens_fn(mesh_device, host_noise: list[torch.Tensor], rows: slice | None = None):
    """Build a ``noise_tokens_fn(step) -> [b, 1, C, 1]`` device hook from host noise."""

    def fn(step: int):
        tokens = host_noise[step]
        if rows is not None:
            tokens = tokens[rows]
        return host_canvas_to_device(mesh_device, tokens.clone())

    return fn


def run(args) -> dict:
    if not batched_decode_enabled():
        raise RuntimeError("set DG_BATCH_DECODE=1 to run the batched canvas decode check (#47557)")

    config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.max_denoising_steps)
    tokenizer_kwargs = {"local_files_only": True} if args.local_files_only else None

    mesh_device = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device, args.checkpoint, tokenizer_kwargs=tokenizer_kwargs, **model_kwargs
        )
        _log_mesh_dram(mesh_device, "post-build")

        vocab_size = int(getattr(bundle.tt_model, "vocab_size", config.vocab_size) or config.vocab_size)
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)

        # Prefill the SHARED prompt (batch=1) and build the denoise adapter.
        session = BlockDiffusionServingSession(
            bundle.tt_model,
            bundle.state_dict,
            config=config,
            tokenizer=bundle.tokenizer,
            gumbel_mode="argmax",
            seed=args.seed,
        )
        cache_len = session.prefill(prompt_tokens)
        adapter = session._logits_fn
        _log_mesh_dram(mesh_device, "post-prefill")

        B = args.batch
        C = args.canvas_length
        canvases = _host_canvases(B, C, vocab_size, args.seed)
        noise = _host_noise(B, C, vocab_size, config.max_denoise_steps, args.seed)

        def batched_committed(mode: str) -> torch.Tensor:
            init_dev = host_canvas_to_device(mesh_device, canvases.clone())
            t0 = time.perf_counter()
            committed = run_batched_denoise_block(
                adapter,
                init_dev,
                config,
                start_pos=cache_len,
                batch=B,
                noise_tokens_fn=_noise_tokens_fn(mesh_device, noise),
                mode=mode,
            )
            return committed, time.perf_counter() - t0

        def standalone_committed(row: int) -> torch.Tensor:
            init_dev = host_canvas_to_device(mesh_device, canvases[row : row + 1].clone())
            committed = run_batched_denoise_block(
                adapter,
                init_dev,
                config,
                start_pos=cache_len,
                batch=1,
                noise_tokens_fn=_noise_tokens_fn(mesh_device, noise, rows=slice(row, row + 1)),
                mode="loop",
            )
            return committed  # [1, C]

        # Batched B canvases through one denoise loop.
        batched, batched_s = batched_committed(args.mode)
        _log_mesh_dram(mesh_device, "post-batched")

        # Standalone B=1 runs (the independence reference).
        standalones = [standalone_committed(row) for row in range(B)]

        matches = []
        mismatch_counts = []
        for row in range(B):
            ref = standalones[row][0]
            got = batched[row]
            eq = bool(torch.equal(got, ref))
            matches.append(eq)
            mismatch_counts.append(int((got != ref).sum()))

        all_match = all(matches)

        # Optional: probe the dim0 mode (non-gating).
        dim0 = {"attempted": False}
        if args.probe_dim0 and args.mode != "dim0":
            dim0["attempted"] = True
            try:
                d0_committed, _ = batched_committed("dim0")
                dim0["ran"] = True
                dim0["matches"] = [bool(torch.equal(d0_committed[r], standalones[r][0])) for r in range(B)]
                dim0["all_match"] = all(dim0["matches"])
            except Exception as exc:  # noqa: BLE001
                dim0["ran"] = False
                dim0["error"] = f"{type(exc).__name__}: {exc}"

        metrics = {
            "batch": B,
            "canvas_length": C,
            "num_layers": args.num_layers,
            "max_denoising_steps": config.max_denoise_steps,
            "mode": args.mode,
            "cache_len": cache_len,
            "prompt_len": int(prompt_tokens.shape[1]),
            "vocab_size": vocab_size,
            "per_row_match": matches,
            "per_row_mismatch_count": mismatch_counts,
            "all_match": all_match,
            "batched_latency_s": batched_s,
            "dim0_probe": dim0,
        }
        session.reset()
        logger.info("[batched_decode_smoke] metrics:\n" + json.dumps(metrics, indent=2))
        if args.metrics_json:
            with open(args.metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _close_mesh_device(mesh_device)


def _marker(metrics: dict) -> str:
    tag = "DG_BATCH_DECODE_SUCCESS" if metrics["all_match"] else "DG_BATCH_DECODE_MISMATCH"
    return (
        f"{tag} batch={metrics['batch']} canvas={metrics['canvas_length']} "
        f"num_layers={metrics['num_layers']} steps={metrics['max_denoising_steps']} "
        f"mode={metrics['mode']} per_row_match={metrics['per_row_match']} "
        f"mismatch_count={metrics['per_row_mismatch_count']} "
        f"batched_latency_s={metrics['batched_latency_s']:.3f}"
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        metrics = run(args)
    except BaseException as exc:  # noqa: BLE001
        logger.error(f"DG_BATCH_DECODE_FAILURE error_type={type(exc).__name__} mesh={args.mesh}")
        raise
    marker = _marker(metrics)
    logger.info(marker)
    print(marker)
    return 0 if metrics["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
