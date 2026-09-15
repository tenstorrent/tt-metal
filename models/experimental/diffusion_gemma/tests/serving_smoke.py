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
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import ttnn
from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.hybrid_kv import (
    attach_model_owned_hybrid_kv,
    model_owned_hybrid_kv_model_kwargs,
)
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    self_conditioning_embedding_prechunk_enabled,
    self_conditioning_logits_l1_mode,
)
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.tt.traced_denoise import (
    UPFRONT_DENOISE_STEPS,
    set_default_reveal_pmax,
    upfront_traced_denoise_block,
)


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
        choices=["argmax", "chunked", "device"],
        help="sampler memory strategy (argmax/chunked fit full 256K)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--entropy-stop-threshold",
        type=float,
        default=None,
        help="override DiffusionConfig.entropy_stop_threshold (default 0.005). Pass a negative "
        "value to disable the stable-and-confident early halt so every block runs the full "
        "--max-denoising-steps. Required for a per-step latency A/B: a lever that changes the "
        "numerics also changes where the halt fires, so arms otherwise compare different amounts "
        "of work (observed 48-step arms halting at 9/2/2 steps).",
    )
    parser.add_argument(
        "--disable-eos-stop",
        action="store_true",
        help="do not halt on committed EOS/stop tokens (surfaces visible non-EOS text for the "
        "qualitative control; mirrors text_demo --disable-eos-stop)",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="render the prompt with DiffusionGemma's <|think|> contract (what the GPQA CoT runs "
        "use through vLLM). Omitted = the non-thinking render.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metrics-json", default=None, help="optional path to dump the per-block metrics JSON")
    parser.add_argument(
        "--upfront",
        action="store_true",
        help="exercise the model-lifetime up-front traced denoise path (capture 48 traces once at "
        "startup, then replay) instead of the eager per-step loop. Requires DG_TRACE_REGION_SIZE>0 "
        "and --gumbel-mode device (the only materialized source); forces 48 denoise steps.",
    )
    parser.add_argument(
        "--reveal-pmax",
        type=int,
        default=None,
        help="fixed reveal span for --upfront (default: --max-seq-len rounded down to a tile). "
        "DG_DENOISE_REVEAL_PMAX still wins if set.",
    )
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


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_upfront(args):
    """Validate the --upfront contract and return (denoise_steps, reveal_pmax).

    Mirrors the fail-loud startup contract the vLLM wrapper enforces, so this smoke is a
    faithful (and far cheaper) stand-in for it: the controller captures the released 48-step
    schedule, needs a real trace-region reservation, and needs a materialized full-tensor
    Gumbel source ("chunked" is a descriptor and "argmax" is None, neither of which can be
    refreshed between trace replays).
    """
    if args.gumbel_mode != "device":
        raise ValueError(
            "--upfront requires --gumbel-mode device (the only materialized source), got "
            f"{args.gumbel_mode!r}. 'chunked'/'argmax' are not materialized full-tensor sources."
        )
    if int(os.environ.get("DG_TRACE_REGION_SIZE", "0")) <= 0:
        raise ValueError("--upfront requires DG_TRACE_REGION_SIZE>0 (the mesh is opened with it reserved)")
    if args.max_denoising_steps != UPFRONT_DENOISE_STEPS:
        logger.warning(
            f"[serving_smoke] --upfront captures the released {UPFRONT_DENOISE_STEPS}-step schedule; "
            f"overriding --max-denoising-steps {args.max_denoising_steps}"
        )
    p_max = args.reveal_pmax
    if p_max is None:
        p_max = (int(args.max_seq_len) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return UPFRONT_DENOISE_STEPS, int(p_max)


def run(args) -> dict:
    denoise_steps = args.max_denoising_steps
    reveal_pmax = None
    if args.upfront:
        denoise_steps, reveal_pmax = _resolve_upfront(args)
        # The controller re-reads DG_DENOISE_REVEAL_PMAX; register the resolved span so an
        # unset env var derives from --max-seq-len instead of hard-failing.
        set_default_reveal_pmax(reveal_pmax)
    config_kwargs = {"canvas_length": args.canvas_length, "max_denoise_steps": denoise_steps}
    if args.entropy_stop_threshold is not None:
        config_kwargs["entropy_stop_threshold"] = args.entropy_stop_threshold
    config = DiffusionConfig(**config_kwargs)
    tokenizer_kwargs = {"local_files_only": True} if args.local_files_only else None

    mesh_device = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh_device, "baseline")
        # The model-owned hybrid layout is the only served cache layout.
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        model_kwargs.update(
            model_owned_hybrid_kv_model_kwargs(
                max_seq_len=args.max_seq_len,
                max_batch_size=1,
            )
        )
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device,
            args.checkpoint,
            tokenizer_kwargs=tokenizer_kwargs,
            **model_kwargs,
        )
        page_tables_per_layer = attach_model_owned_hybrid_kv(
            bundle.tt_model,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
        )
        _log_mesh_dram(mesh_device, "post-build")

        # Prompt length is intentionally NOT a multiple of the 256 output block —
        # the adapter must serve any valid prompt length.
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt, enable_thinking=args.enable_thinking)
        prompt_len = int(prompt_tokens.shape[1])
        logger.info(f"[serving_smoke] prompt_len={prompt_len} (aligned_256={prompt_len % args.canvas_length == 0})")

        session = BlockDiffusionServingSession(
            bundle.tt_model,
            bundle.state_dict,
            config=config,
            tokenizer=bundle.tokenizer,
            gumbel_mode=args.gumbel_mode,
            seed=args.seed,
            page_tables_per_layer=page_tables_per_layer,
            # Empty list disables the EOS/stop halt so degenerate EOS-heavy blocks
            # still emit their non-EOS positions for the qualitative control.
            stop_token_ids=[] if args.disable_eos_stop else None,
            denoise_block_fn=upfront_traced_denoise_block if args.upfront else None,
        )

        # prefill_forward == prefill + block 0 (TTFT).
        t0 = time.perf_counter()
        cache_len = session.prefill(prompt_tokens)
        if args.upfront:
            # The controller is created lazily and ONLY during the startup capture phase, so the
            # first block must be marked as that phase (same handshake as the vLLM wrapper's
            # warmup_model_prefill). Every later block replays the captured traces.
            adapter = session._logits_fn
            adapter._upfront_capture_phase = True
            try:
                first = session.decode_block()
            finally:
                delattr(adapter, "_upfront_capture_phase")
            controller = getattr(adapter, "_upfront_traced_denoise_controller", None)
            if controller is None or not getattr(controller, "captured", False):
                raise RuntimeError("--upfront startup denoise did not leave a fully captured controller")
            if not getattr(adapter, "use_reveal_mask", False):
                raise RuntimeError("--upfront trace was not captured with a persistent reveal mask")
        else:
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
        committed_sha256 = hashlib.sha256(committed.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
        per_block_sha256 = [
            hashlib.sha256(emission.tokens.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
            for emission in emissions
        ]

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
        checkpoint = Path(args.checkpoint)
        metrics = {
            "model": "google/diffusiongemma-26B-A4B-it",
            "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": _file_sha256(checkpoint / "config.json"),
            "mesh": args.mesh,
            "mesh_shape": [1, 4] if args.mesh == "P150x4" else None,
            "num_layers": len(bundle.tt_model.layers),
            "max_seq_len": args.max_seq_len,
            "hybrid_kv": True,
            "seed": args.seed,
            "DG_SELFCOND_PRECHUNK_EMBED": os.environ.get("DG_SELFCOND_PRECHUNK_EMBED", "<unset>"),
            "resolved_selfcond_prechunk": self_conditioning_embedding_prechunk_enabled(),
            "DG_SELFCOND_LOGITS_L1": os.environ.get("DG_SELFCOND_LOGITS_L1", "<unset>"),
            "resolved_selfcond_logits_l1": self_conditioning_logits_l1_mode(),
            "DG_TRACE_REGION_SIZE": os.environ.get("DG_TRACE_REGION_SIZE", "<unset>"),
            "TT_METAL_WATCHER": os.environ.get("TT_METAL_WATCHER", "<unset>"),
            "prompt_len": prompt_len,
            "prompt_aligned_256": bool(prompt_len % args.canvas_length == 0),
            "cache_len": cache_len,
            "canvas_length": args.canvas_length,
            "max_denoising_steps": config.max_denoise_steps,
            "gumbel_mode": args.gumbel_mode,
            "upfront": bool(args.upfront),
            "reveal_pmax": reveal_pmax,
            # False => the fixed full-span prefix read borrowed the model-owned cache instead
            # of cloning it per layer per step.
            "prefix_owns_result": getattr(session._logits_fn, "prompt_hidden_by_layer", None) is not None
            and getattr(session._logits_fn.prompt_hidden_by_layer, "owns_result", None),
            "blocks_emitted": len(emissions),
            "tokens_emitted": tokens_emitted,
            "committed_sha256": committed_sha256,
            "per_block_sha256": per_block_sha256,
            "ttft_s": ttft_s,
            "per_block_latency_s": block_latencies,
            "mean_block_latency_s": mean_block_latency_s,
            "tokens_per_block_per_s": tokens_per_block_per_s,
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "halted_per_block": [e.halted for e in emissions],
            "final_next_pos": session.next_pos,
            "trace_stats": session.trace_stats(),
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
