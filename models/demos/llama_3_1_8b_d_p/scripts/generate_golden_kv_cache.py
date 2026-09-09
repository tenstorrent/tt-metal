#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden trace: the CPU reference forward, run ONCE and saved.

    <out_dir>/
        metadata.json            token_ids (the exact input), n_tokens, num_layers, model info
        kv_cache/
            layer_0.safetensors  key_cache_layer_0 / value_cache_layer_0
            ...                  [1, num_kv_heads, seq_len, head_dim], post-RoPE K and raw V
            layer_31.safetensors

P1 and P2 feed those same token ids to the device and PCC its KV cache against these tensors,
layer by layer. It is the graded artifact of this bring-up.

Three properties worth stating, because each one bites:

* **It needs REAL weights.** This script refuses to run without a safetensors checkpoint. A golden
  trace from random weights proves only that two random-valued pipelines agree.
* **It is per (model, prompt, ISL, depth)** — not per model. A different sequence length or layer
  count needs a different trace, which is why the output dirs are named for prompt and token count.
* **It is not a ttnn trace.** `use_trace` / `trace_region_size` capture a device command buffer for
  replay; that is a perf mechanism with no goldens in it.

## Conventions this file pins

**fp16 on disk.** Recipe §4: every reference and golden in this bring-up is `torch.float16`,
regardless of the checkpoint's bf16 dtype. The donor generator writes bfloat16; that cast is
shape-tuned, not structural, so it is replaced rather than carried over.

**HF head-dim order.** K is stored in the HF half-split layout the reference produces. The DEVICE
cache holds K in Meta interleaved order (q/k are permuted so the rope ops can consume Meta tables),
so the comparison permutes one side — see `tests/galaxy_prefill_kv_pcc.py`. Storing HF order here
keeps the trace a property of the MODEL rather than of this package's device-side choices.

Usage:

    python3 models/demos/llama_3_1_8b_d_p/scripts/generate_golden_kv_cache.py \\
        --max-tokens 5120 \\
        --out-dir /mnt/models/meta-llama/Llama-3.1-8B-Instruct/golden/synthetic_5120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants  # noqa: E402
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefModel  # noqa: E402
from models.demos.llama_3_1_8b_d_p.tt.model_config import ModelArgs, resolve_weights_path  # noqa: E402

# A deterministic, self-contained prompt. Real prose rather than repeated tokens: an input with no
# variety lets a broken position encoding look correct, because every position sees the same content.
SEED_TEXT = """
The bring-up of a large language model on a mesh of accelerators is, at bottom, an exercise in
bookkeeping. The mathematics is settled before any of it begins: a stack of identical decoder
layers, each a self-attention block and a feed-forward block, each wrapped in a normalization and a
residual connection. What is not settled is where each number lives. A tensor that is correct on one
chip and absent on another is not correct. A key that has been rotated twice is indistinguishable
from one rotated once until it is compared against something that knows better. The work of a
bring-up is to build that something, and then to believe it only as far as it has been checked.
Sharding turns arithmetic into geography. The sequence is cut across one axis of the mesh and the
features across another, so that a single matrix multiplication becomes a conversation between
thirty-two devices, each holding a slice of the truth and none holding all of it. The collectives
that stitch those slices back together are not incidental plumbing; they are where the model's
accuracy is won or lost, because a reduction performed in the wrong precision, or over the wrong
axis, produces a number that is plausible, finite, and wrong.
"""


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="output directory for the trace")
    ap.add_argument("--max-tokens", type=int, default=5120, help="sequence length to generate (ISL)")
    ap.add_argument("--weights", default=None, help="checkpoint dir (default: HF_MODEL / the shared store)")
    ap.add_argument("--num-layers", type=int, default=None, help="layers to run (default: all 32)")
    ap.add_argument("--prompt-file", default=None, help="a text file to tokenize instead of the built-in prompt")
    ap.add_argument("--overwrite", action="store_true", help="regenerate even if the trace already exists")
    return ap.parse_args()


def build_token_ids(weights_path: str, n_tokens: int, prompt_file: str | None) -> list[int]:
    """Tokenize the prompt and tile it to exactly `n_tokens`.

    Tiling repeats CONTENT, not a single token, so positions stay distinguishable; the trace is a
    correctness reference, not a benchmark of natural text.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weights_path)
    text = Path(prompt_file).read_text() if prompt_file else SEED_TEXT
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    if not ids:
        raise ValueError("prompt tokenized to nothing")
    while len(ids) < n_tokens:
        ids = ids + ids
    return ids[:n_tokens]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    kv_dir = out_dir / "kv_cache"
    if (out_dir / "metadata.json").exists() and not args.overwrite:
        logger.info(f"{out_dir} already holds a trace; pass --overwrite to regenerate")
        return 0

    weights_path = args.weights or resolve_weights_path(required=True)
    logger.info(f"weights: {weights_path}")

    config = LlamaConfigConstants.from_json()
    num_layers = args.num_layers or config.num_hidden_layers
    if num_layers != config.num_hidden_layers:
        logger.warning(
            f"REDUCED DEPTH: {num_layers} of {config.num_hidden_layers} layers. This trace is a "
            "diagnostic, not a grade — label every number it produces as reduced."
        )

    token_ids = build_token_ids(weights_path, args.max_tokens, args.prompt_file)
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    logger.info(f"{len(token_ids)} tokens, {num_layers} layers")

    state_dict = ModelArgs.load_state_dict(weights_path)
    model = RefModel(config, num_layers=num_layers).to(REF_DTYPE).eval()
    model.load_state_dict(
        {
            name: state_dict[f"model.{name}" if not name.startswith("lm_head") else name].to(REF_DTYPE)
            for name in model.state_dict()
        }
    )
    del state_dict
    logger.info("weights loaded into the reference")

    kv_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with torch.no_grad():
        _, per_layer_kv = model(input_ids, return_kv=True)
    logger.info(f"reference forward: {time.time() - start:.1f}s")

    for layer_idx, (k, v) in enumerate(per_layer_kv):
        save_file(
            {
                f"key_cache_layer_{layer_idx}": k.to(REF_DTYPE).contiguous(),
                f"value_cache_layer_{layer_idx}": v.to(REF_DTYPE).contiguous(),
            },
            str(kv_dir / f"layer_{layer_idx}.safetensors"),
        )
    logger.info(f"wrote {len(per_layer_kv)} layer files to {kv_dir}")

    metadata = {
        "model": "llama_3_1_8b",
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "weights_path": str(weights_path),
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "num_layers": num_layers,
        "reduced_depth": num_layers != config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "dtype": "float16",
        "k_layout": "hf_half_split",  # NOT the device cache's Meta order — see the module docstring
        "k_is_post_rope": True,
        "v_is_raw": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"golden trace written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
