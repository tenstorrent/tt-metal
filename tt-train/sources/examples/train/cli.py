# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line argument parsing + back-compat shims for the training entry point."""

from __future__ import annotations

import argparse
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with grouped help + back-compat aliases for old train_nanogpt.py callers."""
    p = argparse.ArgumentParser(description="NanoGPT/Llama/DeepSeek/Qwen3 training (SFTTrainer)")

    g = p.add_argument_group("Config & data")
    g.add_argument(
        "-c",
        "--config",
        type=str,
        default="training_shakespeare_nanogpt_char.yaml",
        help="Path to training config YAML",
    )
    g.add_argument(
        "--data-path", dest="data_path", type=str, default="", help="Override training data path (default: from config)"
    )
    g.add_argument(
        "--sequence-length",
        dest="sequence_length",
        type=int,
        default=None,
        help="Override sequence length (default: from config)",
    )

    g = p.add_argument_group("Training overrides")
    g.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Override batch size")
    g.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="Override max training steps")
    g.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    g.add_argument(
        "--max-grad-norm",
        dest="max_grad_norm",
        type=float,
        default=None,
        help="Enable gradient clipping with this max-norm",
    )
    g.add_argument(
        "--no-lazy",
        dest="no_lazy",
        action="store_true",
        help="Disable lazy parameter init under FSDP (allocate eagerly, then shard; more peak memory)",
    )

    g = p.add_argument_group("Checkpointing")
    g.add_argument(
        "--checkpoint-dir", dest="checkpoint_dir", type=str, default="", help="Directory for saving/loading checkpoints"
    )
    g.add_argument(
        "--checkpoint-prefix",
        dest="checkpoint_prefix",
        type=str,
        default="model",
        help="Prefix for checkpoint filenames",
    )
    g.add_argument("--resume", type=str, default="", help="Specific checkpoint to resume from (default: auto-detect)")
    g.add_argument("--fresh", action="store_true", help="Skip resume; train from scratch")

    g = p.add_argument_group("Inference")
    g.add_argument("--model-path", dest="model_path", type=str, default="", help="Checkpoint to load for inference")
    g.add_argument("--prompt", type=str, default="", help="Prompt text; with --model-path triggers inference mode")
    g.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=300, help="Tokens to generate")
    g.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy)")
    g.add_argument("--top-k", dest="top_k", type=int, default=40, help="Top-k sampling (0=disabled)")

    g = p.add_argument_group("Diagnostics")
    g.add_argument("--track-memory", dest="track_memory", action="store_true", help="Enable memory tracking callbacks")
    g.add_argument(
        "--print-summary", dest="print_summary", action="store_true", help="Print model layer-by-layer summary"
    )
    g.add_argument(
        "--log-expert-activations",
        dest="log_expert_activations",
        type=str,
        default=None,
        help="DeepSeek-only: CSV path for per-step expert activation probabilities",
    )

    # Silent underscore aliases for old train_nanogpt.py callers. Same dest as the
    # canonical hyphen flag; SUPPRESS hides them from --help.
    for flag, dest, type_name in (
        ("--data_path", "data_path", "str"),
        ("--sequence_length", "sequence_length", "int"),
        ("--batch_size", "batch_size", "int"),
        ("--max_steps", "max_steps", "int"),
        ("--checkpoint_dir", "checkpoint_dir", "str"),
        ("--checkpoint_prefix", "checkpoint_prefix", "str"),
        ("--max_new_tokens", "max_new_tokens", "int"),
        ("--top_k", "top_k", "int"),
        ("--model_path", "model_path", "str"),
    ):
        p.add_argument(
            flag, dest=dest, type={"str": str, "int": int}[type_name], default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
    p.add_argument(
        "--track_memory", dest="track_memory", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--print_summary", dest="print_summary", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    # Deprecated renames — mapped to the canonical name in _apply_backcompat with a stderr warning.
    p.add_argument("--num_epochs", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--clip_grad_norm", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--model_save_path", type=str, default="", help=argparse.SUPPRESS)

    return p


def _apply_backcompat(args: argparse.Namespace) -> None:
    """Translate deprecated flags into canonical (with stderr warnings) and validate inference-mode flag pairing."""
    if args.num_epochs is not None:
        print("warning: --num_epochs is deprecated; use --epochs", file=sys.stderr)
        if args.epochs is None:
            args.epochs = args.num_epochs
    if args.clip_grad_norm is not None:
        print("warning: --clip_grad_norm is deprecated; use --max-grad-norm", file=sys.stderr)
        if args.max_grad_norm is None:
            args.max_grad_norm = args.clip_grad_norm
    if args.model_save_path:
        if args.checkpoint_dir:
            raise SystemExit("error: --model_save_path and --checkpoint-dir are mutually exclusive")
        print(
            "warning: --model_save_path is deprecated; use --checkpoint-dir + --checkpoint-prefix",
            file=sys.stderr,
        )
        dirpart = os.path.dirname(args.model_save_path) or "."
        basepart = os.path.basename(args.model_save_path) or "model"
        args.checkpoint_dir = dirpart
        args.checkpoint_prefix = basepart

    if bool(args.prompt) ^ bool(args.model_path):
        missing = "--model-path" if args.prompt else "--prompt"
        raise SystemExit(f"error: inference mode requires both --prompt and --model-path (missing: {missing})")


def parse_args() -> argparse.Namespace:
    args = _build_parser().parse_args()
    _apply_backcompat(args)
    return args
