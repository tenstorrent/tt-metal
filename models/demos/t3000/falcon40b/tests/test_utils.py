# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM


def load_hf_model(model_version):
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_version, local_files_only=os.getenv("CI") == "true", low_cpu_mem_usage=True
    )
    # transformers v5 loads from_pretrained in the checkpoint dtype (bf16 for Falcon-40B);
    # force float32 so the CPU reference matches the float32 test inputs (avoids
    # "mixed dtype (CPU)" LayerNorm errors). Restores the pre-v5 default; mirrors the
    # .to(torch.float32) fix applied to the falcon7b tests in #47218.
    hugging_face_reference_model = hugging_face_reference_model.to(torch.float32)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict
