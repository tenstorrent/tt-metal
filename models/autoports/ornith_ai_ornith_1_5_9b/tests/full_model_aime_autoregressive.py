# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Free-running TT comparison against the exact cached AIME24 HF continuation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from readiness_check.teacher_forcing import TokenAccuracy

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import HF_MODEL, HF_REVISION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    model_dir = Path(__file__).resolve().parents[1]
    reference = model_dir / "readiness_aime24_chat.refpt"
    cache_key = model_dir / "readiness/reference_cache/refpt-3763ed27aeead79f4eca5bc075ecce35.key.json"
    meta = json.loads(cache_key.read_text())
    assert meta["hf_model_id"] == HF_MODEL and meta["revision"] == "hub:" + HF_REVISION
    assert meta["settings"]["chat_template"] and meta["settings"]["gen_len"] == 100
    acc = TokenAccuracy(reference)
    prompt, hf_tokens = acc.get_prompt_token_ids(), acc.gt_gen_1d[0].tolist()
    assert len(hf_tokens) == 100
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(model_dir, mesh, max_seq_len=max(2048, len(prompt) + 100))
        tokens = gen.generate(prompt, 100, enable_trace=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (("hf", hf_tokens), ("tt", tokens)):
            (args.output_dir / (name + "_completion.txt")).write_text(
                gen.tokenizer.decode(value, skip_special_tokens=False)
            )
        text = gen.tokenizer.decode(prompt, skip_special_tokens=False)
        (args.output_dir / "prompt.txt").write_text(text)
        output = {
            "hf_model_id": HF_MODEL,
            "hf_revision": HF_REVISION,
            "prompt_mode": "chat",
            "prompt_text": text,
            "prompt_token_ids": prompt,
            "max_new_tokens": 100,
            "hf": {"token_ids": hf_tokens, "num_tokens": len(hf_tokens)},
            "tt": {"token_ids": tokens, "num_tokens": len(tokens)},
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "reference_cache_key": meta,
            "reference_reused": True,
            "teacher_forcing": False,
            "generation_metrics": gen.last_generation_metrics,
        }
        (args.output_dir / "autoregressive_meta.json").write_text(json.dumps(output, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
