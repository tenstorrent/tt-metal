# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-revision CPU HF reference using the packaged readiness implementation.

The generic runner selects AutoModelForCausalLM; this checkpoint registers its
text-capable conditional-generation class with AutoModelForImageTextToText.
Only that loader choice and the immutable local snapshot are adapted here.
"""

import json
import sys
from pathlib import Path

import torch
from readiness_check import generate
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
SNAPSHOT = Path("/mnt/models/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots") / REVISION
ROOT = Path(__file__).resolve().parents[1]


class ExactModel:
    @staticmethod
    def from_pretrained(model_id, **kwargs):
        assert model_id == MODEL_ID
        return AutoModelForImageTextToText.from_pretrained(
            SNAPSHOT, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )


class ExactTokenizer:
    @staticmethod
    def from_pretrained(model_id, **kwargs):
        assert model_id == MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
        render = tokenizer.apply_chat_template

        def legacy_token_list(*args, **kw):
            # transformers >=5.15 defaults to BatchEncoding; readiness expects ids.
            kw.setdefault("return_dict", False)
            return render(*args, **kw)

        tokenizer.apply_chat_template = legacy_token_list
        return tokenizer


if __name__ == "__main__":
    torch.set_num_threads(16)
    generate.AutoModelForCausalLM = ExactModel
    generate.AutoTokenizer = ExactTokenizer
    generate._main()
    path = ROOT / "doc/full_model/reference_metadata.json"
    tokenizer = ExactTokenizer.from_pretrained(MODEL_ID)
    path.write_text(
        json.dumps(
            dict(
                model_id=MODEL_ID,
                revision=REVISION,
                snapshot=str(SNAPSHOT),
                command=sys.argv,
                tokenizer=type(tokenizer).__name__,
                chat_template=tokenizer.chat_template,
                prompt_source="aime24",
                gen_len=100,
                top_k=100,
                hf_loader="AutoModelForImageTextToText",
                dtype="bfloat16",
                device="cpu",
            ),
            indent=2,
        )
        + "\n"
    )
