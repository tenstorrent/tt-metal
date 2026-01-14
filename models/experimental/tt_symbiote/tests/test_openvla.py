# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for OpenVLA model with TTNN backend."""

import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor

from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLama
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_openvla(device):
    """Test OpenVLA model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLama,
        nn.SiLU: TTNNSilu,
    }

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )

    image: Image.Image = Image.new("RGB", (224, 224))
    # Grab image input & format prompt
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeV2)
    inputs = processor(prompt, image).to(dtype=torch.bfloat16)
    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, use_cache=True)
    print(action)
