# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for OpenVLA model with TTNN backend."""

import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLama, TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm, TTNNRMSNorm
from models.experimental.tt_symbiote.modules.attention import LlamaAttention
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager


def test_openvla(device):
    """Test OpenVLA model with TTNN acceleration."""
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLama,
        nn.SiLU: TTNNSilu,
        nn.SiLU: TTNNSilu,
        nn.LayerNorm: TTNNLayerNorm,
        model.language_model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
        model.language_model.model.layers[0].self_attn.__class__: LlamaAttention,
    }
    nn_to_ttnn_2 = {
        nn.Linear: TTNNLinear,
    }
    exclude_list = set(
        [
            "language_model.lm_head",
            "language_model.model.layers.0.mlp.gate_proj",
            "language_model.model.layers.1.mlp.gate_proj",
            "language_model.model.layers.14.mlp.gate_proj",
            "language_model.model.layers.0.mlp.down_proj",
            "language_model.model.layers.13.mlp.down_proj",
            "language_model.model.layers.3.mlp.down_proj",
            "language_model.model.layers.8.mlp.gate_proj",
            "language_model.model.layers.11.mlp.gate_proj",
            "language_model.model.layers.7.mlp.gate_proj",
            "language_model.model.layers.13.mlp.up_proj",
            "language_model.model.layers.5.mlp.gate_proj",
            "language_model.model.layers.7.mlp.down_proj",
            "language_model.model.layers.5.mlp.up_proj",
            "language_model.model.layers.12.mlp.gate_proj",
            "language_model.model.layers.7.mlp.up_proj",
            "language_model.model.layers.4.mlp.gate_proj",
            "language_model.model.layers.3.mlp.gate_proj",
            "language_model.model.layers.9.mlp.gate_proj",
            "language_model.model.layers.6.mlp.gate_proj",
            "language_model.model.layers.10.mlp.gate_proj",
            "language_model.model.layers.5.mlp.down_proj",
            "language_model.model.layers.0.mlp.up_proj",
            "language_model.model.layers.10.mlp.up_proj",
            "language_model.model.layers.2.mlp.up_proj",
            "language_model.model.layers.1.mlp.up_proj",
            "language_model.model.layers.2.mlp.down_proj",
            "language_model.model.layers.8.mlp.up_proj",
            "language_model.model.layers.3.mlp.up_proj",
            "language_model.model.layers.8.mlp.down_proj",
            "language_model.model.layers.9.mlp.down_proj",
            "language_model.model.layers.2.mlp.gate_proj",
            "language_model.model.layers.4.mlp.down_proj",
            "language_model.model.layers.10.mlp.down_proj",
            "language_model.model.layers.11.mlp.down_proj",
            "language_model.model.layers.1.mlp.down_proj",
            "language_model.model.layers.9.mlp.up_proj",
            "language_model.model.layers.11.mlp.up_proj",
            "language_model.model.layers.4.mlp.up_proj",
            "language_model.model.layers.12.mlp.up_proj",
            "language_model.model.layers.18.mlp.gate_proj",
            "language_model.model.layers.6.mlp.up_proj",
            "language_model.model.layers.18.mlp.up_proj",
            "language_model.model.layers.14.mlp.up_proj",
            "language_model.model.layers.30.mlp.gate_proj",
            "language_model.model.layers.22.mlp.gate_proj",
            "language_model.model.layers.16.mlp.gate_proj",
            "language_model.model.layers.13.mlp.gate_proj",
            "language_model.model.layers.24.mlp.gate_proj",
            "language_model.model.layers.30.mlp.up_proj",
            "language_model.model.layers.24.mlp.up_proj",
            "language_model.model.layers.20.mlp.gate_proj",
            "language_model.model.layers.17.mlp.gate_proj",
            "language_model.model.layers.29.mlp.gate_proj",
            "language_model.model.layers.16.mlp.down_proj",
            "language_model.model.layers.12.mlp.down_proj",
            "language_model.model.layers.19.mlp.gate_proj",
            "language_model.model.layers.15.mlp.gate_proj",
            "language_model.model.layers.14.mlp.down_proj",
            "language_model.model.layers.16.mlp.up_proj",
            "language_model.model.layers.20.mlp.down_proj",
            "language_model.model.layers.21.mlp.up_proj",
            "language_model.model.layers.25.mlp.gate_proj",
            "language_model.model.layers.15.mlp.up_proj",
            "language_model.model.layers.31.mlp.gate_proj",
            "language_model.model.layers.27.mlp.up_proj",
            "language_model.model.layers.18.mlp.down_proj",
            "language_model.model.layers.23.mlp.gate_proj",
            "language_model.model.layers.19.mlp.down_proj",
            "language_model.model.layers.27.mlp.gate_proj",
            "language_model.model.layers.28.mlp.gate_proj",
            "language_model.model.layers.20.mlp.up_proj",
            "language_model.model.layers.26.mlp.gate_proj",
            "language_model.model.layers.30.mlp.down_proj",
            "language_model.model.layers.17.mlp.up_proj",
            "language_model.model.layers.15.mlp.down_proj",
            # "language_model.model.layers.22.mlp.up_proj",
            # "language_model.model.layers.6.mlp.down_proj",
            # "language_model.model.layers.19.mlp.up_proj",
            # "language_model.model.layers.27.mlp.down_proj",
            # "language_model.model.layers.29.mlp.down_proj",
            # "language_model.model.layers.24.mlp.down_proj",
            # "language_model.model.layers.31.mlp.up_proj",
            # "language_model.model.layers.25.mlp.up_proj",
            # "language_model.model.layers.25.mlp.down_proj",
            # "language_model.model.layers.21.mlp.gate_proj",
            # "language_model.model.layers.21.mlp.down_proj",
            # "language_model.model.layers.29.mlp.up_proj",
            # "language_model.model.layers.23.mlp.down_proj",
            # "language_model.model.layers.17.mlp.down_proj",
            # "language_model.model.layers.31.mlp.down_proj",
            # "language_model.model.layers.26.mlp.down_proj",
            # "language_model.model.layers.22.mlp.down_proj",
            # "language_model.model.layers.23.mlp.up_proj",
            # "language_model.model.layers.28.mlp.up_proj",
            # "language_model.model.layers.28.mlp.down_proj",
            # "language_model.model.layers.26.mlp.up_proj",
            # "projector.fc2",
            # "projector.fc3",
            # "projector.fc1",
        ]
    )
    image: Image.Image = Image.new("RGB", (224, 224))
    # Grab image input & format prompt
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeV2)
    inputs = processor(prompt, image).to(dtype=torch.bfloat16)

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_list)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn_2, model_config=None)
    set_device(model, device)
    for k, v in tqdm({**modules1, **modules2}.items()):
        v.preprocess_weights()
        if k in exclude_list:
            v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, use_cache=True)
    DispatchManager.clear_timings()
    for i in range(60):
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, use_cache=True)
    DispatchManager.save_stats_to_file("openvla_timing_stats.csv")
    print(action)
