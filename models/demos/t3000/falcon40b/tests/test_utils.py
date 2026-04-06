# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM


def load_hf_model(model_version):
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_version, local_files_only=os.getenv("CI") == "true", low_cpu_mem_usage=True
    )
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict
