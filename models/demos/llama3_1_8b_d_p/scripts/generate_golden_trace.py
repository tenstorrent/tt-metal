# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the on-disk golden trace the prefill producer PCCs against (recipe P1/P3).

Runs the torch reference on CPU with the REAL checkpoint and writes the layout the shared producer
reads (``prefill_producer.py::_read_slot_kv_and_check_pcc_gqa_per_head``):

    <out_dir>/metadata.json                       {"token_ids": [...], ...}
    <out_dir>/kv_cache/layer_<i>.safetensors      key_cache_layer_<i>, value_cache_layer_<i>

The K written here is in the **HF half-split** convention, NOT the device's Meta-interleaved layout —
the producer applies the head permutation itself when it compares. Writing an already-permuted golden
would double-apply it and look like a subtle numerics failure. (See ``docs/SPEC_NOTES.md`` §4: this is
the field the spec should carry explicitly.)

Usage::

    HF_MODEL=/path/to/Meta-Llama-3.1-8B-Instruct \
      python -m models.demos.llama3_1_8b_d_p.scripts.generate_golden_trace \
        --out /tmp/llama31_8b_trace --seq-len 2048
"""

import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.llama3_1_8b_d_p.reference.config import LlamaConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output trace directory")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=None, help="layer count (default: the model's 32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=os.getenv("HF_MODEL"), help="checkpoint dir (default $HF_MODEL)")
    args = ap.parse_args()

    if not args.checkpoint or not os.path.isdir(args.checkpoint):
        raise SystemExit("set HF_MODEL (or pass --checkpoint) to a Llama-3.1-8B-Instruct checkpoint directory")

    from safetensors.torch import save_file

    from models.demos.llama3_1_8b_d_p.tests.galaxy_prefill_kv_pcc import build_reference_from_checkpoint
    from models.demos.llama3_1_8b_d_p.tt.model_config import ModelArgs

    cfg = LlamaConfig.from_json(Path(args.checkpoint) / "config.json")
    num_layers = args.layers or cfg.num_hidden_layers

    logger.info(f"Loading checkpoint from {args.checkpoint} (HF layout — the golden is HF-convention)")
    state_dict = ModelArgs.load_state_dict(args.checkpoint, convert_to_meta_format=False)

    logger.info(f"Running the CPU reference: {num_layers} layers, {args.seq_len} tokens")
    model = build_reference_from_checkpoint(cfg, state_dict, num_layers)
    del state_dict

    g = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), generator=g)
    with torch.no_grad():
        _, kvs, _ = model(input_ids)

    out = Path(args.out)
    (out / "kv_cache").mkdir(parents=True, exist_ok=True)
    with open(out / "metadata.json", "w") as f:
        json.dump(
            {
                "token_ids": input_ids[0].tolist(),
                "model": "llama3_1_8b_d_p",
                "num_layers": num_layers,
                "seq_len": args.seq_len,
                "seed": args.seed,
                "head_layout": "hf_half_split",
                "checkpoint": str(args.checkpoint),
            },
            f,
        )
    for i, (k, v) in enumerate(kvs):
        save_file(
            {f"key_cache_layer_{i}": k.contiguous(), f"value_cache_layer_{i}": v.contiguous()},
            str(out / "kv_cache" / f"layer_{i}.safetensors"),
        )
    logger.info(f"Golden trace written to {out} ({num_layers} layers, {args.seq_len} tokens)")


if __name__ == "__main__":
    main()
