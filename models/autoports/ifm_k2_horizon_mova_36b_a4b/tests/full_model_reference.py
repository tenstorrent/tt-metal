# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-revision entrypoint for the standard readiness reference writer."""
from functools import partial
from unittest.mock import patch

import torch
from readiness_check.generate import generate_reference
from transformers import AutoModelForCausalLM, AutoTokenizer

from .run_decoder import MODEL, REVISION, ROOT


def main():
    torch.set_num_threads(16)
    load_tokenizer = AutoTokenizer.from_pretrained

    def tokenizer(*args, **kwargs):
        result = load_tokenizer(*args, revision=REVISION, **kwargs)
        result.apply_chat_template = partial(result.apply_chat_template, return_dict=False)
        return result

    with (
        patch.object(AutoTokenizer, "from_pretrained", tokenizer),
        patch.object(
            AutoModelForCausalLM,
            "from_pretrained",
            partial(
                AutoModelForCausalLM.from_pretrained,
                revision=REVISION,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ),
        ),
    ):
        generate_reference(
            MODEL,
            prompt_source="aime24",
            chat_template=True,
            gen_len=100,
            top_k=100,
            output_path=ROOT / "readiness_aime24_chat.refpt",
        )


if __name__ == "__main__":
    main()
