# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 text generation demo — Galaxy mesh (8,4).

Supports:
  - Variable ISL (input sequence length) — any prompt length up to max_seq_len
  - Device-resident KV cache (no host round-trips for KV)
  - Greedy decode with multiple prompts

Usage:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate

    # Run with real checkpoint (paged attention enabled by default):
    pytest models/demos/minimax_m2/demo/text_demo.py -xvs

    # Test with 32k context length:
    MINIMAX_M2_MAX_SEQ_LEN=32768 pytest models/demos/minimax_m2/demo/text_demo.py -xvs

    # Disable paged attention (non-paged mode):
    MINIMAX_M2_PAGED_ATTENTION=0 pytest models/demos/minimax_m2/demo/text_demo.py -xvs

    # Override model path:
    MINIMAX_M2_MODEL_PATH=/path/to/model pytest models/demos/minimax_m2/demo/text_demo.py -xvs

    # Override max generation length:
    MINIMAX_M2_MAX_NEW_TOKENS=128 pytest models/demos/minimax_m2/demo/text_demo.py -xvs
"""

import json
import os
import time

import pytest
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.demos.minimax_m2.reference.generate_goldens import load_and_dequant
from models.demos.minimax_m2.tt.generator import TtMiniMaxGenerator
from models.demos.minimax_m2.tt.model import TtMiniMaxModel
from models.demos.minimax_m2.tt.model_config import MiniMaxM2TTConfig, make_paged_attention_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    "/home/cust-team/models/models--MiniMaxAI--MiniMax-M2.5/" "snapshots/f710177d938eff80b684d42c5aa84b382612f21f"
)
MODEL_PATH = os.environ.get("MINIMAX_M2_MODEL_PATH", DEFAULT_MODEL_PATH)
MAX_SEQ_LEN = int(os.environ.get("MINIMAX_M2_MAX_SEQ_LEN", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("MINIMAX_M2_MAX_NEW_TOKENS", "64"))
USE_PAGED_ATTENTION = os.environ.get("MINIMAX_M2_PAGED_ATTENTION", "1") == "1"
BATCH = 1

MESH_ROWS = 8
MESH_COLS = 4

DEFAULT_PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a land far away, there lived a wise old wizard who",
    "Explain the theory of relativity in simple terms:",
    "Write a Python function to compute the Fibonacci sequence:",
]


# ---------------------------------------------------------------------------
# Demo infrastructure
# ---------------------------------------------------------------------------


def load_model_config(model_path: str) -> MiniMaxM2TTConfig:
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    return MiniMaxM2TTConfig(
        hidden_size=cfg["hidden_size"],
        head_dim=cfg["head_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        num_local_experts=cfg["num_local_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        rotary_dim=cfg["rotary_dim"],
        rope_theta=cfg["rope_theta"],
        rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"],
    )


def load_state_dict(model_path: str) -> dict:
    """Load and dequantize all weight shards."""
    logger.info("Loading model weights from {}", model_path)
    raw = {}
    shard_files = sorted(f for f in os.listdir(model_path) if f.startswith("model-") and f.endswith(".safetensors"))
    for i, shard in enumerate(shard_files):
        logger.info("  [{}/{}] {}", i + 1, len(shard_files), shard)
        raw.update(load_file(os.path.join(model_path, shard)))
    return load_and_dequant(raw)


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)


def generate_for_prompts(
    generator: TtMiniMaxGenerator,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    eos_token_id: int | None = None,
):
    """Run generation for each prompt, measuring per-prompt performance."""
    results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]  # [1, ISL]
        isl = input_ids.shape[1]
        logger.info("Prompt {}/{}: ISL={} tokens", i + 1, len(prompts), isl)

        t0 = time.perf_counter()
        output_ids = generator.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
        t1 = time.perf_counter()

        generated_tokens = output_ids.shape[1] - isl
        decode_time = t1 - t0
        tokens_per_sec = generated_tokens / decode_time if decode_time > 0 else 0

        output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        results.append(
            {
                "prompt": prompt,
                "output": output_text,
                "isl": isl,
                "generated_tokens": generated_tokens,
                "total_time_s": decode_time,
                "tokens_per_sec": tokens_per_sec,
            }
        )

        logger.info(
            "  Generated {} tokens in {:.2f}s ({:.1f} tok/s)",
            generated_tokens,
            decode_time,
            tokens_per_sec,
        )
        logger.info("  Output: {}", output_text[:200] + ("..." if len(output_text) > 200 else ""))
        logger.info("")

    return results


# ---------------------------------------------------------------------------
# Tests / entry points
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    d = ttnn.open_mesh_device(ttnn.MeshShape(MESH_ROWS, MESH_COLS))
    yield d
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def model_artifacts():
    """Load config, state dict, and tokenizer."""
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    config = load_model_config(MODEL_PATH)
    state_dict = load_state_dict(MODEL_PATH)
    tokenizer = load_tokenizer(MODEL_PATH)
    return config, state_dict, tokenizer


@pytest.fixture(scope="module")
def tt_model_and_gen(mesh_device, model_artifacts):
    """Create TT model and generator."""
    config, state_dict, _ = model_artifacts

    paged_attention_config = None
    if USE_PAGED_ATTENTION:
        paged_attention_config = make_paged_attention_config(max_seq_len=MAX_SEQ_LEN)
        logger.info(
            "Building TT model ({} layers, max_seq_len={}, paged=True, blocks={})",
            config.num_hidden_layers,
            MAX_SEQ_LEN,
            paged_attention_config.max_num_blocks,
        )
    else:
        logger.info("Building TT model ({} layers, max_seq_len={})", config.num_hidden_layers, MAX_SEQ_LEN)

    t0 = time.perf_counter()
    model = TtMiniMaxModel(
        mesh_device,
        state_dict,
        config,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH,
        paged_attention_config=paged_attention_config,
    )
    gen = TtMiniMaxGenerator(model, mesh_device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)
    t1 = time.perf_counter()
    logger.info("Model loaded in {:.1f}s", t1 - t0)
    return gen


def test_text_generation(tt_model_and_gen, model_artifacts):
    """End-to-end text generation demo with multiple prompts at different ISLs."""
    gen = tt_model_and_gen
    _, _, tokenizer = model_artifacts

    eos_id = tokenizer.eos_token_id
    prompts = (
        os.environ.get("MINIMAX_M2_PROMPTS", "").split("|") if os.environ.get("MINIMAX_M2_PROMPTS") else DEFAULT_PROMPTS
    )

    results = generate_for_prompts(gen, tokenizer, prompts, MAX_NEW_TOKENS, eos_token_id=eos_id)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "  ISL={:>4d} → {:>3d} tokens in {:.2f}s ({:.1f} tok/s)",
            r["isl"],
            r["generated_tokens"],
            r["total_time_s"],
            r["tokens_per_sec"],
        )

    for r in results:
        assert r["generated_tokens"] > 0, f"No tokens generated for prompt: {r['prompt'][:50]}"
        assert len(r["output"]) > len(r["prompt"]), "Output should be longer than prompt"
