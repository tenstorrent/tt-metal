# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reduced-surface serving driver for the DiffusionGemma vLLM block contract (#47466).

This is the *smallest representative serving target* for stage-09: it drives the
exact block-emission path the vLLM adapter (``tt/generator_vllm.py``) delegates to
— :class:`BlockDiffusionServingSession` — directly on device, **without** the
container-gated vLLM engine. It proves the block-granular contract end-to-end:

- prefill writes prompt K/V and builds the stateful denoise logits fn;
- each decode step emits ONE 256-token block (the on-device Gumbel-max /
  entropy-budget / renoise canvas sampling loop + commit-append);
- absolute position advances by ``canvas_length`` per block;
- a deliberately **non-256-aligned** prompt length exercises the input-alignment
  carve-out (the intrinsic 256-token *output* block granularity is not an input
  constraint).

It reports **per-block** serving metrics (prefill TTFT = prefill + block-0
latency; per-block latency; tokens-per-block throughput) — never a per-token
``1000/mean_tpot_ms``. RUN-first: degenerate output is expected until #48291.

Emits a single greppable ``DG_VLLM_SERVING_SMOKE_SUCCESS ...`` line on success and
``DG_VLLM_SERVING_SMOKE_FAILURE ...`` on any uncaught error.
"""

from __future__ import annotations

import argparse
import json
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
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


def build_arg_parser() -> argparse.ArgumentParser:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"),
        help="HF checkpoint directory or model id",
    )
    parser.add_argument("--mesh", default="P150x4", help="mesh label or ROWSxCOLS (QB2 = P150x4)")
    parser.add_argument("--num-layers", type=int, default=None, help="reduced layer count (default: full 30)")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="served max context (KV/RoPE span)")
    parser.add_argument(
        "--prompt",
        default="Explain what a diffusion language model is in one sentence.",
        help="user prompt (chat-templated); its token length is intentionally non-256-aligned",
    )
    parser.add_argument("--num-blocks", type=int, default=2, help="number of 256-token blocks to emit")
    parser.add_argument("--canvas-length", type=int, default=256, help="output block size (canvas)")
    parser.add_argument("--max-denoising-steps", type=int, default=4, help="denoise steps per block cap")
    parser.add_argument(
        "--gumbel-mode",
        default="argmax",
        choices=["argmax", "chunked", "host", "device"],
        help="sampler memory strategy (argmax/chunked fit full 256K)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-eos-stop",
        action="store_true",
        help="do not halt on committed EOS/stop tokens (surfaces visible non-EOS text for the "
        "qualitative control; mirrors text_demo --disable-eos-stop)",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metrics-json", default=None, help="optional path to dump the per-block metrics JSON")
    return parser


def _success_marker(metrics: dict) -> str:
    return (
        "DG_VLLM_SERVING_SMOKE_SUCCESS "
        f"prompt_len={metrics['prompt_len']} "
        f"prompt_aligned_256={metrics['prompt_aligned_256']} "
        f"cache_len={metrics['cache_len']} "
        f"blocks={metrics['blocks_emitted']} "
        f"tokens={metrics['tokens_emitted']} "
        f"canvas={metrics['canvas_length']} "
        f"ttft_s={metrics['ttft_s']:.3f} "
        f"mean_block_latency_s={metrics['mean_block_latency_s']:.3f} "
        f"tokens_per_block_per_s={metrics['tokens_per_block_per_s']:.2f} "
        f"final_next_pos={metrics['final_next_pos']} "
        f"text_chars={metrics['text_chars']}"
    )


def run(args) -> dict:
    config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.max_denoising_steps)
    tokenizer_kwargs = {"local_files_only": True} if args.local_files_only else None

    mesh_device = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device,
            args.checkpoint,
            tokenizer_kwargs=tokenizer_kwargs,
            **model_kwargs,
        )
        _log_mesh_dram(mesh_device, "post-build")

        # Prompt length is intentionally NOT a multiple of the 256 output block —
        # the adapter must serve any valid prompt length.
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        prompt_len = int(prompt_tokens.shape[1])
        logger.info(f"[serving_smoke] prompt_len={prompt_len} (aligned_256={prompt_len % args.canvas_length == 0})")

        session = BlockDiffusionServingSession(
            bundle.tt_model,
            bundle.state_dict,
            config=config,
            tokenizer=bundle.tokenizer,
            gumbel_mode=args.gumbel_mode,
            seed=args.seed,
            # Empty list disables the EOS/stop halt so degenerate EOS-heavy blocks
            # still emit their non-EOS positions for the qualitative control.
            stop_token_ids=[] if args.disable_eos_stop else None,
        )

        # prefill_forward == prefill + block 0 (TTFT).
        t0 = time.perf_counter()
        cache_len = session.prefill(prompt_tokens)
        first = session.decode_block()
        ttft_s = time.perf_counter() - t0
        _log_mesh_dram(mesh_device, "post-prefill+block0")

        emissions = [first]
        # decode_forward == one block per step.
        for _ in range(1, args.num_blocks):
            if session.finished:
                break
            emissions.append(session.decode_block())

        block_latencies = [e.latency_s for e in emissions]
        decode_block_latencies = block_latencies[1:] if len(block_latencies) > 1 else block_latencies
        tokens_emitted = sum(e.tokens.shape[1] for e in emissions)
        committed = torch.cat([e.tokens for e in emissions], dim=1)

        # Detokenize the concatenated committed blocks (RUN-first: may be degenerate).
        text = decode_generation(
            bundle.tokenizer,
            prompt_tokens,
            # prompt_len here is the position-space (aligned) cache_len used for
            # commit-append and position advancement, matching DeviceGeneration
            # in the one-shot path — not the logical prompt token count.
            _DeviceGenLike(committed, session.cache_len, session.next_pos),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        text_str = text[0] if text else ""

        mean_block_latency_s = sum(decode_block_latencies) / len(decode_block_latencies)
        tokens_per_block_per_s = args.canvas_length / mean_block_latency_s if mean_block_latency_s > 0 else 0.0
        metrics = {
            "prompt_len": prompt_len,
            "prompt_aligned_256": bool(prompt_len % args.canvas_length == 0),
            "cache_len": cache_len,
            "canvas_length": args.canvas_length,
            "max_denoising_steps": args.max_denoising_steps,
            "gumbel_mode": args.gumbel_mode,
            "blocks_emitted": len(emissions),
            "tokens_emitted": tokens_emitted,
            "ttft_s": ttft_s,
            "per_block_latency_s": block_latencies,
            "mean_block_latency_s": mean_block_latency_s,
            "tokens_per_block_per_s": tokens_per_block_per_s,
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "halted_per_block": [e.halted for e in emissions],
            "final_next_pos": session.next_pos,
            "text_chars": len(text_str),
            "text": text_str,
        }
        session.reset()
        logger.info("[serving_smoke] per-block metrics:\n" + json.dumps(metrics, indent=2))
        if args.metrics_json:
            with open(args.metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _close_mesh_device(mesh_device)


class _DeviceGenLike:
    """Minimal DeviceGeneration-shaped view for decode_generation reuse."""

    def __init__(self, generated, prompt_len, next_pos):
        self.generated = generated
        self.prompt_len = prompt_len
        self.next_pos = next_pos
        self.trajectories = []


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        metrics = run(args)
    except BaseException as exc:  # noqa: BLE001 - emit a greppable failure marker then re-raise
        logger.error(f"DG_VLLM_SERVING_SMOKE_FAILURE error_type={type(exc).__name__} mesh={args.mesh}")
        raise
    logger.info(_success_marker(metrics))
    print(_success_marker(metrics))
    if metrics["text"]:
        print("GENERATED:", metrics["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
