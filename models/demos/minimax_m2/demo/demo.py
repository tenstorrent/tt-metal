# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 Demo - Performance + Coherence

Unified demo that shows:
1. Generated text output (coherence verification)
2. TTFT (time-to-first-token) performance
3. Decode throughput (tokens/sec)

Usage:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate

    # Run with default prompt
    python models/demos/minimax_m2/demo/demo.py

    # Run with custom prompt
    python models/demos/minimax_m2/demo/demo.py --prompt "Write a poem about AI"

    # Run with multiple prompts from file
    python models/demos/minimax_m2/demo/demo.py --prompts-file models/demos/minimax_m2/demo/sample_prompts.json

    # Quick test with 2 layers
    python models/demos/minimax_m2/demo/demo.py --num-layers 2

    # Greedy decoding (no sampling)
    python models/demos/minimax_m2/demo/demo.py --greedy
"""

import argparse
import json
import os
import time

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.minimax_m2.tt.generator import TtMiniMaxGenerator
from models.demos.minimax_m2.tt.model import TtMiniMaxModel
from models.demos.minimax_m2.tt.model_config import MiniMaxM2TTConfig, make_paged_attention_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    "/home/cust-team/models/models--MiniMaxAI--MiniMax-M2.5/" "snapshots/f710177d938eff80b684d42c5aa84b382612f21f"
)

MESH_ROWS = 8
MESH_COLS = 4

DEFAULT_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where robots and humans coexist,",
    "The most important scientific discovery of the 21st century will be",
]


def get_padded_prefill_len(seq_len: int) -> int:
    """Get padded prefill length matching tt_transformers pattern.

    Pads to:
    - 128 if seq_len <= 128
    - 1024 if seq_len <= 1024
    - Next power of 2 if seq_len > 1024
    """
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    else:
        return 2 ** (seq_len - 1).bit_length()


# ---------------------------------------------------------------------------
# Weight Loading
# ---------------------------------------------------------------------------


def dequant_fp8_to_bf16(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize FP8 weight to BF16 using block-wise scales."""
    out_f, in_f = weight.shape
    w = weight.to(torch.bfloat16)
    s = scale_inv.to(torch.bfloat16).repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    s = s[:out_f, :in_f]
    return w * s


class StreamingStateDict:
    """Memory-efficient state dict that loads weights on-demand from safetensors."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.shard_files = sorted(
            f for f in os.listdir(model_path) if f.startswith("model-") and f.endswith(".safetensors")
        )
        self._key_to_shard = {}
        for shard in self.shard_files:
            with safe_open(os.path.join(model_path, shard), framework="pt") as f:
                for key in f.keys():
                    self._key_to_shard[key] = shard
        logger.info("Indexed {} keys across {} shards", len(self._key_to_shard), len(self.shard_files))

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._key_to_shard:
            raise KeyError(f"Key not found: {key}")
        shard_path = os.path.join(self.model_path, self._key_to_shard[key])
        with safe_open(shard_path, framework="pt") as f:
            tensor = f.get_tensor(key)
            scale_key = key + "_scale_inv"
            if tensor.dtype == torch.float8_e4m3fn and scale_key in self._key_to_shard:
                scale_shard = os.path.join(self.model_path, self._key_to_shard[scale_key])
                with safe_open(scale_shard, framework="pt") as sf:
                    scale = sf.get_tensor(scale_key)
                return dequant_fp8_to_bf16(tensor, scale)
            elif tensor.dtype == torch.float8_e4m3fn:
                return tensor.to(torch.bfloat16)
            else:
                return tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_shard

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def load_model_config(model_path: str, num_layers: int | None = None) -> MiniMaxM2TTConfig:
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    return MiniMaxM2TTConfig(
        hidden_size=cfg["hidden_size"],
        head_dim=cfg["head_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        num_hidden_layers=num_layers if num_layers is not None else cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        num_local_experts=cfg["num_local_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        rotary_dim=cfg["rotary_dim"],
        rope_theta=cfg["rope_theta"],
        rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"],
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_demo(
    mesh_device,
    config: MiniMaxM2TTConfig,
    state_dict,
    tokenizer,
    prompts: list[str],
    max_seq_len: int,
    max_new_tokens: int,
    greedy: bool,
):
    """Run demo with performance tracking and text output."""
    results = []

    # Build model
    paged_config = make_paged_attention_config(max_seq_len=max_seq_len)
    logger.info("Building model ({} layers, max_seq_len={}, paged=True)", config.num_hidden_layers, max_seq_len)

    t0 = time.perf_counter()
    model = TtMiniMaxModel(
        mesh_device,
        state_dict,
        config,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        paged_attention_config=paged_config,
    )
    gen = TtMiniMaxGenerator(model, mesh_device, max_seq_len=max_seq_len, batch=1)
    build_time = time.perf_counter() - t0
    logger.info("Model built in {:.1f}s", build_time)

    # Sampling params
    if greedy:
        temperature, top_p, rep_penalty = 0.0, 1.0, 1.0
        logger.info("Using greedy decoding (temperature=0)")
    else:
        temperature, top_p, rep_penalty = 0.7, 0.9, 1.1
        logger.info("Using sampling (temperature={}, top_p={}, rep_penalty={})", temperature, top_p, rep_penalty)

    # Process each prompt
    for i, prompt in enumerate(prompts):
        logger.info("\n" + "=" * 70)
        logger.info("PROMPT {}/{}: {}", i + 1, len(prompts), prompt[:80] + "..." if len(prompt) > 80 else prompt)
        logger.info("=" * 70)

        # Tokenize and pad to prefill bucket (matches tt_transformers)
        input_ids_raw = tokenizer(prompt, return_tensors="pt")["input_ids"]
        isl = input_ids_raw.shape[1]
        padded_isl = get_padded_prefill_len(isl)

        # Pad with zeros (will be ignored due to causal mask)
        if padded_isl > isl:
            padding = torch.zeros(1, padded_isl - isl, dtype=input_ids_raw.dtype)
            input_ids = torch.cat([input_ids_raw, padding], dim=1)
        else:
            input_ids = input_ids_raw

        logger.info("Input: {} tokens (padded to {})", isl, padded_isl)

        # Clear KV cache
        model.clear_kv_caches()

        # Warmup (first prompt only)
        if i == 0:
            logger.info("Warmup run...")
            _ = gen.generate(
                input_ids,
                max_new_tokens=2,
                eos_token_id=tokenizer.eos_token_id,
                use_trace=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
            model.clear_kv_caches()
            ttnn.synchronize_device(mesh_device)

        # Timed generation
        t_start = time.perf_counter()
        output_ids = gen.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            use_trace=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )
        ttnn.synchronize_device(mesh_device)
        t_end = time.perf_counter()

        # Decode output
        output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        generated_text = output_text[len(prompt) :].strip()
        num_generated = output_ids.shape[1] - input_ids.shape[1]

        # Calculate metrics
        total_time = t_end - t_start
        ttft = total_time / (num_generated + 1)  # Approximate: total / (prefill + decode tokens)
        decode_time = total_time - ttft
        tok_per_sec = (num_generated - 1) / decode_time if decode_time > 0 and num_generated > 1 else 0

        result = {
            "prompt": prompt,
            "isl": isl,
            "padded_isl": padded_isl,
            "generated_tokens": num_generated,
            "ttft_ms": ttft * 1000,
            "decode_tok_s": tok_per_sec,
            "total_time_s": total_time,
            "output": generated_text,
        }
        results.append(result)

        # Print output
        logger.info("\n--- OUTPUT ({} tokens in {:.2f}s) ---", num_generated, total_time)
        print(generated_text)
        logger.info("---")
        logger.info("TTFT: {:.1f}ms | Decode: {:.1f} tok/s", ttft * 1000, tok_per_sec)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY ({} layers, {} prompts)", config.num_hidden_layers, len(prompts))
    logger.info("=" * 80)
    logger.info("{:>6} {:>8} {:>8} {:>12} {:>12}", "ISL", "Padded", "Tokens", "TTFT (ms)", "Tok/s")
    logger.info("-" * 80)
    for r in results:
        logger.info(
            "{:>6} {:>8} {:>8} {:>12.1f} {:>12.1f}",
            r["isl"],
            r["padded_isl"],
            r["generated_tokens"],
            r["ttft_ms"],
            r["decode_tok_s"],
        )

    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    avg_toks = sum(r["decode_tok_s"] for r in results) / len(results)
    logger.info("-" * 80)
    logger.info("{:>6} {:>8} {:>8} {:>12.1f} {:>12.1f}", "AVG", "-", "-", avg_ttft, avg_toks)

    return results


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M2.5 Demo")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MINIMAX_M2_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Path to model weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with list of prompts",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for KV cache",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (default: all 62). Use 2-4 for quick tests.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (no sampling)",
    )
    args = parser.parse_args()

    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
            prompts = data if isinstance(data, list) else data.get("prompts", [])
    else:
        prompts = DEFAULT_PROMPTS

    if not os.path.isdir(args.model_path):
        logger.error("Model not found at {}", args.model_path)
        return

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

    # Initialize mesh device
    logger.info("Initializing mesh device ({}, {})", MESH_ROWS, MESH_COLS)
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(MESH_ROWS, MESH_COLS))

    try:
        config = load_model_config(args.model_path, args.num_layers)
        state_dict = StreamingStateDict(args.model_path)

        run_demo(
            mesh_device,
            config,
            state_dict,
            tokenizer,
            prompts,
            args.max_seq_len,
            args.max_new_tokens,
            args.greedy,
        )

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
