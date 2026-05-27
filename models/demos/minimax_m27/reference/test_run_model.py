# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse

from transformers import AutoConfig, AutoTokenizer

from models.demos.minimax_m27.reference.modeling_minimax_m2 import MiniMaxM2ForCausalLM


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check local MiniMax reference model wiring.")
    parser.add_argument(
        "--local-model-path",
        type=str,
        default="/data/minimax_m27",
        help="Path to local MiniMax model directory (config/tokenizer files).",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.local_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path, trust_remote_code=True)
    model = MiniMaxM2ForCausalLM(cfg).eval()

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Reference model class: {model.__class__.__name__}")
    print(f"Hidden size: {cfg.hidden_size}, layers: {cfg.num_hidden_layers}")


if __name__ == "__main__":
    main()
