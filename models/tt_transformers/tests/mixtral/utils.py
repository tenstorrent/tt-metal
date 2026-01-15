# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

from transformers import AutoConfig


def load_hf_mixtral_config():
    hf_model = os.getenv("HF_MODEL")
    assert hf_model is not None, "Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct"
    config = AutoConfig.from_pretrained(hf_model, local_files_only=os.getenv("CI") == "true")
    return config
