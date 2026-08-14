# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fresh TT AIME24 generation against the exact existing HF reference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    reference_meta = json.loads((args.reference / "autoregressive_meta.json").read_text())
    if reference_meta["max_new_tokens"] != args.max_new_tokens:
        raise ValueError("HF reference token budget does not match requested TT run")
    prompt_ids = reference_meta["prompt_token_ids"]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"), mesh_device=mesh, max_context=512, batch=1
        )
        tt_tokens = generator.generate(prompt_ids, args.max_new_tokens)
        tt_text = generator.tokenizer.decode(tt_tokens, skip_special_tokens=False)
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.reference / "hf_completion.txt", args.output / "hf_completion.txt")
    (args.output / "tt_completion.txt").write_text(tt_text)
    metadata = dict(reference_meta)
    metadata["tt"] = {"token_ids": tt_tokens, "num_tokens": len(tt_tokens)}
    metadata["hf_reference_reused_from"] = str(args.reference)
    metadata["hf_reference_exact_match"] = {
        "prompt_token_ids": True,
        "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "max_new_tokens": args.max_new_tokens,
    }
    (args.output / "autoregressive_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"tt_num_tokens": len(tt_tokens), "tt_text": tt_text}, indent=2))


if __name__ == "__main__":
    main()
