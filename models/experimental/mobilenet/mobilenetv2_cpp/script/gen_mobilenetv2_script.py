# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

from models.demos.vision.classification.mobilenetv2.reference.mobilenetv2 import (
    Conv2dNormActivation,
    InvertedResidual,
    Mobilenetv2,
)


script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
save_path = script_dir / "mobilenet_v2-b0353104.pth"
url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
urllib.request.urlretrieve(url, save_path)

state_dict = torch.load(save_path)
ds_state_dict = {k: v for k, v in state_dict.items()}

torch_model = Mobilenetv2()
new_state_dict = {
    name1: parameter2
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
    if isinstance(parameter2, torch.FloatTensor)
}
torch_model.load_state_dict(new_state_dict)
torch_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
script_module = torch.jit.trace(torch_model, dummy_input)
save_path_pt = script_dir / "mobilenet_v2-b0353104.pt"
script_module.save(save_path_pt)


weights_dir = script_dir / "mobilenet_v2-b0353104_weights"
weights_dir.mkdir(exist_ok=True)


def write_tensor(name, tensor, manifest, layout="ROW_MAJOR"):
    file_name = f"{name}.bin"
    contiguous_tensor = tensor.detach().to(torch.float32).contiguous().cpu()
    contiguous_tensor.numpy().tofile(weights_dir / file_name)
    manifest["tensors"][name] = {
        "file": file_name,
        "shape": list(contiguous_tensor.shape),
        "dtype": "float32",
        "layout": layout,
    }


def fold_batch_norm2d_into_conv2d(conv, bn):
    eps = bn.eps
    weight = conv.weight.detach()
    running_mean = bn.running_mean.detach()
    running_var = bn.running_var.detach()
    scale = bn.weight.detach()
    shift = bn.bias.detach()
    folded_weight = weight * (scale / torch.sqrt(running_var + eps)).view(-1, 1, 1, 1)
    folded_bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    return folded_weight, folded_bias.view(1, 1, 1, -1)


manifest = {"tensors": {}}
conv_bn_counter = 0
counter = 0
for _, module in torch_model.named_modules():
    if isinstance(module, InvertedResidual):
        for idx, submodule in enumerate(module.conv):
            if isinstance(submodule, nn.Conv2d):
                bn = (
                    module.conv[idx + 1]
                    if idx + 1 < len(module.conv) and isinstance(module.conv[idx + 1], nn.BatchNorm2d)
                    else None
                )
                if bn is not None:
                    weight, bias = fold_batch_norm2d_into_conv2d(submodule, bn)
                    write_tensor(f"conv_{counter}_weight", weight, manifest)
                    write_tensor(f"conv_{counter}_bias", bias, manifest)
                    counter += 1
    elif isinstance(module, Conv2dNormActivation):
        if len(module) == 3 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
            weight, bias = fold_batch_norm2d_into_conv2d(module[0], module[1])
            write_tensor(f"fused_conv_{conv_bn_counter}_weight", weight, manifest)
            write_tensor(f"fused_conv_{conv_bn_counter}_bias", bias, manifest)
            conv_bn_counter += 1
    elif isinstance(module, nn.Linear):
        write_tensor("classifier_1_weight", module.weight.detach().T, manifest, layout="TILE")
        write_tensor("classifier_1_bias", module.bias.detach().reshape(1, -1), manifest, layout="TILE")

with open(weights_dir / "manifest.json", "w", encoding="utf-8") as manifest_file:
    json.dump(manifest, manifest_file, indent=2)

if os.path.exists(save_path):
    os.remove(save_path)
