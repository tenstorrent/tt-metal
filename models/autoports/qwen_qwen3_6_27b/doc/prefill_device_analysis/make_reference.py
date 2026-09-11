# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate a fresh, pinned Qwen3.8 AIME reference on this host's CPU."""

from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from models.common.readiness_check.generate import generate_reference


def main():
    torch.set_num_threads(8)
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    loader = AutoModelForCausalLM.from_pretrained

    def load(*args, **kwargs):
        return loader(*args, **kwargs, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager")

    with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=load):
        generate_reference(
            hf_model_id=str(snapshot),
            prompt_len=128,
            gen_len=100,
            output_path=Path("/tmp/qwen38_prefill_followup_aime100.refpt"),
            top_k=100,
            prompt_source="aime24",
            chat_template=True,
        )


if __name__ == "__main__":
    main()
