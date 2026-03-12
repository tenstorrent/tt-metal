# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-27B demo and smoke test using the TTNN hybrid model.

Usage (on a device):
    HF_MODEL=Qwen3.5-27B python models/demos/qwen3_5/demo/demo.py \
        --prompt "The capital of France is" --max-new-tokens 32
"""

import argparse
import os

import pytest
from loguru import logger

import ttnn
from models.demos.qwen3_5.tt.generator import Qwen3_5Generator
from models.demos.qwen3_5.tt.model import Qwen3_5Transformer
from models.tt_transformers.tt.model_config import ModelArgs


def create_model_and_generator(
    mesh_device,
    model_name: str = "Qwen3.5-27B",
    max_batch_size: int = 1,
    max_seq_len: int = 4096,
    dtype=ttnn.bfloat8_b,
    dummy_weights: bool = False,
    max_n_layers: int = None,
):
    """Load Qwen3.5-27B TTNN model and wrap in generator.

    ModelArgs reads the checkpoint path from HF_MODEL env var.
    Set HF_MODEL=Qwen3.5-27B to use the local params at
    models/tt_transformers/model_params/Qwen3.5-27B.
    """
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_name

    model_args = ModelArgs(
        mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )

    weight_cache_path = model_args.model_cache_path if not dummy_weights else None
    if weight_cache_path is not None:
        weight_cache_path.mkdir(parents=True, exist_ok=True)

    state_dict = model_args.load_state_dict()

    tt_model = Qwen3_5Transformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        max_n_layers=max_n_layers,
    )

    from transformers import AutoTokenizer

    tokenizer = None
    if not dummy_weights:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_args.CKPT_DIR)
        except Exception as e:
            logger.warning(f"Tokenizer load failed: {e}")

    generator = Qwen3_5Generator(
        model=[tt_model],
        model_args=model_args,
        mesh_device=mesh_device,
        tokenizer=tokenizer,
    )
    return generator, tokenizer


@pytest.fixture(scope="module")
def mesh_device():
    """Open a device for testing."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize("max_seq_len", [128])
def test_qwen3_5_smoke(mesh_device, max_seq_len):
    """Smoke test: load 4-layer dummy model, run prefill + 4 decode steps."""
    os.environ["HF_MODEL"] = "Qwen3.5-27B"
    generator, _ = create_model_and_generator(
        mesh_device=mesh_device,
        model_name="Qwen3.5-27B",
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat16,
        dummy_weights=True,
        max_n_layers=4,
    )

    prompt_tokens = [1, 450, 7483, 310, 3444, 338]
    generated, perf = generator.generate(prompt_tokens, max_new_tokens=4, temperature=0.0)
    logger.info(f"Generated token ids: {generated}")
    logger.info(f"Perf: {perf}")
    assert len(generated) > 0
    assert perf["ttft_ms"] > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5-27B demo")
    parser.add_argument("--model-name", default="Qwen3.5-27B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("HF_MODEL", args.model_name)
    device = ttnn.open_device(device_id=args.device_id)
    try:
        generator, tokenizer = create_model_and_generator(
            mesh_device=device,
            model_name=args.model_name,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
        )
        if tokenizer is not None:
            prompt_tokens = tokenizer.encode(args.prompt)
        else:
            prompt_tokens = [1, 450, 7483, 310, 3444, 338]
        logger.info(f"Prompt: {args.prompt!r}  ({len(prompt_tokens)} tokens)")
        generated, perf = generator.generate(prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=0.0)
        if tokenizer is not None:
            text = tokenizer.decode(generated)
            logger.info(f"Generated: {text!r}")
        logger.info(f"TTFT={perf['ttft_ms']:.1f}ms | {perf['tok_per_sec']:.1f} tok/s")
    finally:
        ttnn.close_device(device)
