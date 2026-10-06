# SPDX-License-Identifier: Apache-2.0
"""Unchanged packaged autoregressive runner with a pinned local HF architecture."""

import json
import os
from pathlib import Path

import torch
from readiness_check.run_autoregressive import run_autoregressive
from transformers import AutoTokenizer

import ttnn

from ..tt.checkpoint import MODEL_ID, REVISION, SNAPSHOT
from . import hf_model  # Explicit local architecture registration.
from .full_provenance import provenance


def main():
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    out = (
        Path(os.environ["FULL_ARTIFACT_DIR"]) / "readiness_autoregressive"
        if "FULL_ARTIFACT_DIR" in os.environ
        else root / "readiness_autoregressive"
    )
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    prompt = json.loads((root / "doc/full_model/qualitative_prompts.json").read_text())[0]
    prompt_file = out / "rendered_prompt.txt"
    prompt_file.write_text(prompt["rendered"])
    actual = tokenizer.encode(prompt["rendered"].strip(), add_special_tokens=True)
    metadata = dict(
        provenance=provenance(),
        model_id=MODEL_ID,
        revision=REVISION,
        tokenizer_path=str(SNAPSHOT),
        prompt_mode="chat_template_with_runner_whitespace_strip",
        exact_template_ids=prompt["token_ids"],
        runner_ids=actual,
        caveat="Unchanged runner strips terminal assistant newline; exact-template HF/TT shared suite is the main qualitative verdict.",
    )
    (out / "prompt_format.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    try:
        run_autoregressive(
            model_dir=root,
            hf_model_id=str(SNAPSHOT),
            prompt_file=prompt_file,
            mesh_device=mesh,
            output_dir=out,
            max_new_tokens=128,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
