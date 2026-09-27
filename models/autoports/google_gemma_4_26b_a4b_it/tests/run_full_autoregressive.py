# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standard autoregressive runner with a verified pre-rendered HF chat prompt."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.common.readiness_check.run_autoregressive import run_autoregressive


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_26b_a4b_it")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-4-26B-A4B-it", revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    )
    messages = [{"role": "user", "content": "Explain why the sky is blue to a curious ten-year-old."}]
    rendered = messages[0]["content"]
    actual = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    output = root / "doc/full_model/autoregressive"
    output.mkdir(exist_ok=True)
    prompt = output / "prompt.txt"
    prompt.write_text(rendered + "\n")
    (output / "prompt_format.json").write_text(
        json.dumps(
            dict(
                mode="chat",
                messages=messages,
                prompt_tokens=actual,
                revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
                rendering="HF apply_chat_template in shared autoregressive runner",
            ),
            indent=2,
        )
        + "\n"
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    try:
        run_autoregressive(
            model_dir=root,
            hf_model_id="google/gemma-4-26B-A4B-it",
            prompt_file=prompt,
            mesh_device=mesh,
            output_dir=output,
            max_new_tokens=128,
            build_kwargs={"max_seq_len": 8192},
            chat_template=True,
            hf_revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            hf_dtype=torch.bfloat16,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
