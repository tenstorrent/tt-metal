# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate a torch reference cache for `test_prefill_transformer.py` PCC validation,
without any TT device dependency. Runs on any machine that can import torch + ttnn
(no Galaxy / no actual silicon required).

Output filename matches the test's cache key:
    {output_dir}/pretrained_{input_source}_isl{isl}_layers{num_layers}_experts{n_experts}.pt

Reference branch of `load_and_compute_layer_by_layer` is fully device-free; this
script just wires CLI args -> tokenizer -> helper -> save.

Example:
    python -m models.demos.deepseek_v3_d_p.scripts.generate_reference_cache \\
        --model-path /mnt/models/deepseek-ai/DeepSeek-R1-0528 \\
        --num-layers 61 --isl 25600 --input-source longbook_qa_eng \\
        --n-routed-experts 256 \\
        --output-dir /mnt/models/deepseek-prefill-cache/golden
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
from transformers import AutoConfig, AutoTokenizer

INFINITEBENCH_SUBSETS = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}
JSON_PROMPT_SOURCES = {"json_prompts", "abc_1k"}


def _load_tokenizer(model_path: Path) -> AutoTokenizer:
    """Mirror the test's tokenizer fixture without pulling pytest in."""
    if not any(model_path.glob("tokenizer*")):
        raise FileNotFoundError(f"No tokenizer files in {model_path}")
    return AutoTokenizer.from_pretrained(str(model_path), use_fast=True, trust_remote_code=True)


def _resolve_prompt_text(input_source: str) -> str:
    from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
    from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
        ABC_1K_PATH,
        PROMPTS_PATH,
        download_infinitebench_subset,
    )

    if input_source == "json_prompts":
        prompts = load_prompts_from_json(str(PROMPTS_PATH))
        return prompts[0] if isinstance(prompts, list) else prompts
    if input_source == "abc_1k":
        return load_prompts_from_json(str(ABC_1K_PATH))
    if input_source in INFINITEBENCH_SUBSETS:
        cached_path = download_infinitebench_subset(input_source)
        with open(cached_path) as f:
            return json.load(f)["prompt"]
    raise ValueError(f"Unsupported --input-source {input_source!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True, help="Path to HF DeepSeek-R1 weights directory.")
    parser.add_argument("--num-layers", type=int, required=True, help="Layer count to truncate the model to.")
    parser.add_argument("--isl", type=int, required=True, help="Input sequence length (token count, padded).")
    parser.add_argument(
        "--input-source",
        type=str,
        required=True,
        choices=sorted(INFINITEBENCH_SUBSETS | JSON_PROMPT_SOURCES),
        help="Prompt source matching test_prefill_transformer.py.",
    )
    parser.add_argument("--n-routed-experts", type=int, default=256, help="MoE routed-expert count (production: 256).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the reference cache file. Sets TT_DS_PREFILL_HOST_REF_CACHE for save_reference_cache.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if the cache file already exists.")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        logger.error(f"--model-path {args.model_path} is not a directory")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TT_DS_PREFILL_HOST_REF_CACHE"] = str(args.output_dir)

    cache_key = f"pretrained_{args.input_source}_isl{args.isl}_layers{args.num_layers}_experts{args.n_routed_experts}"
    cache_path = args.output_dir / f"{cache_key}.pt"
    if cache_path.exists() and not args.force:
        logger.info(f"Cache already exists at {cache_path}; skipping (pass --force to regenerate).")
        return 0

    # Imports that pull in ttnn/transformer_helpers happen after argparse so --help is fast.
    from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
    from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
        load_and_compute_layer_by_layer,
        save_reference_cache,
        tokenize_prompt_to_isl,
    )

    logger.info(
        f"Generating reference: layers={args.num_layers} isl={args.isl} "
        f"input_source={args.input_source} n_experts={args.n_routed_experts}"
    )
    logger.info(f"Output: {cache_path}")

    # Match the test: monkey-patch the singleton expert count so HF + helpers agree.
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = args.n_routed_experts

    config = AutoConfig.from_pretrained(str(args.model_path), trust_remote_code=True)
    config.max_seq_len = args.isl

    tokenizer = _load_tokenizer(args.model_path)
    prompt_text = _resolve_prompt_text(args.input_source)
    token_ids, attention_mask, _ = tokenize_prompt_to_isl(tokenizer, max_isl=args.isl, prompt_text=prompt_text)
    logger.info(f"Tokenized {args.input_source}: shape={token_ids.shape}, first 10={token_ids[0, :10].tolist()}")

    t0 = time.time()
    result = load_and_compute_layer_by_layer(
        model_path=args.model_path,
        config=config,
        num_layers=args.num_layers,
        token_ids=token_ids,
        attention_mask=attention_mask,
        compute_reference=True,
        build_ttnn_cache=False,
        weight_cache_path=None,
        mesh_device=None,
        seq_len=args.isl,
    )
    elapsed = time.time() - t0
    logger.info(f"Reference computation done in {elapsed:.1f}s")

    if result.ref_snapshots is None or result.ref_kvpe_list is None:
        logger.error("load_and_compute_layer_by_layer returned no reference data; aborting.")
        return 3

    save_reference_cache(cache_key, result.ref_snapshots, result.ref_kvpe_list)
    size_mb = cache_path.stat().st_size / 1024**2
    logger.success(f"Wrote {cache_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
