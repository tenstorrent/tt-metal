# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.runner import GenerationResult, run_generation
from models.demos.deepseek_v3_b1.demo.runtime import TokenCodec, create_model

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DeepSeek-V3-B1 Demo on TT-NN")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of decode steps to run")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer id or local tokenizer path",
    )
    parser.add_argument(
        "--loopback-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HostInterface loopback path (recommended for current B1 bring-up)",
    )
    parser.add_argument("--mesh-height", type=int, default=1, help="Mesh height for open_mesh_device")
    parser.add_argument("--mesh-width", type=int, default=1, help="Mesh width for open_mesh_device")
    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_demo(
    *,
    prompt: str,
    max_new_tokens: int,
    tokenizer_name_or_path: str,
    loopback_mode: bool,
    mesh_height: int,
    mesh_width: int,
    output_stream: TextIO,
) -> GenerationResult:
    logger.info(
        "Starting DeepSeek V3 B1 demo (max_new_tokens={}, loopback_mode={}, mesh_shape={}x{})",
        max_new_tokens,
        loopback_mode,
        mesh_height,
        mesh_width,
    )
    if not is_slow_dispatch():
        raise RuntimeError(
            "DeepSeek V3 B1 demo requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
        )

    logger.info("Loading tokenizer: {}", tokenizer_name_or_path)
    tokenizer = load_tokenizer(tokenizer_name_or_path)
    token_codec = TokenCodec(batch_size=1)
    is_first_decode_chunk = True

    def write_text(text: str) -> None:
        nonlocal is_first_decode_chunk
        if is_first_decode_chunk:
            if prompt:
                output_stream.write(prompt)
            is_first_decode_chunk = False
        output_stream.write(text)
        output_stream.flush()

    mesh_device: ttnn.MeshDevice | None = None
    try:
        logger.info("Opening mesh device")
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_height, mesh_width))
        logger.info("Creating DeepSeekV3 model")
        model = create_model(mesh_device=mesh_device, batch_size=1, loopback_mode=loopback_mode)
        logger.info("Running prefill + decode")
        result = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            make_input_tensor=token_codec.make_input,
            extract_token_id=token_codec.extract_token_id,
            write_text=write_text,
        )
        logger.info(
            "Generation complete (prompt_tokens={}, generated_tokens={})",
            len(result.prompt_token_ids),
            len(result.generated_token_ids),
        )
        return result
    finally:
        if mesh_device is not None:
            logger.info("Closing mesh device")
            ttnn.close_mesh_device(mesh_device)


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer_name_or_path=args.tokenizer,
        loopback_mode=args.loopback_mode,
        mesh_height=args.mesh_height,
        mesh_width=args.mesh_width,
        output_stream=sys.stdout,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
