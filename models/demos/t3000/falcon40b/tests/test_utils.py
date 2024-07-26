# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)


def load_hf_model(model_location_generator, model_version):
    model_name = model_location_generator(model_version, model_subdir="Falcon")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict
