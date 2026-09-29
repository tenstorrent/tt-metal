"""Run packaged HF reference generation at the exact pinned checkpoint revision."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check.generate import generate_reference
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MODEL = "IFM/K2-Horizon-7B"
HF_REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"


def main():
    torch.set_num_threads(16)
    model_load = AutoModelForCausalLM.from_pretrained
    tokenizer_load = AutoTokenizer.from_pretrained

    def pinned_tokenizer(*args, **kwargs):
        tokenizer = tokenizer_load(*args, revision=HF_REVISION, **kwargs)
        original_chat = tokenizer.apply_chat_template

        def chat(*a, **kw):
            kw.setdefault("return_dict", False)
            return original_chat(*a, **kw)

        tokenizer.apply_chat_template = chat
        return tokenizer

    with (
        patch.object(
            AutoModelForCausalLM,
            "from_pretrained",
            side_effect=lambda *a, **kw: model_load(*a, revision=HF_REVISION, **kw),
        ),
        patch.object(AutoTokenizer, "from_pretrained", side_effect=pinned_tokenizer),
    ):
        path = generate_reference(
            hf_model_id=HF_MODEL,
            prompt_source="aime24",
            chat_template=True,
            gen_len=100,
            top_k=100,
            output_path="models/autoports/ifm_k2_horizon_7b/readiness_aime24_chat.refpt",
        )
    Path("models/autoports/ifm_k2_horizon_7b/doc/full_model/reference_metadata.json").write_text(
        json.dumps(
            {
                "hf_model": HF_MODEL,
                "revision": HF_REVISION,
                "tokenizer_revision": HF_REVISION,
                "prompt_source": "aime24",
                "chat_template": True,
                "generation_length": 100,
                "top_k": 100,
                "command": "python -m models.autoports.ifm_k2_horizon_7b.tests.generate_reference",
                "reference": str(path),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
