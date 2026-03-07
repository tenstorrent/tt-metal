# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.request
import torch

from models.demos.vision.classification.mobilenetv2.reference.mobilenetv2 import Mobilenetv2

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "mobilenet_v2-b0353104.pth")
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

save_path_pt = os.path.join(script_dir, "mobilenet_v2-b0353104.pt")
script_module.save(save_path_pt)

if os.path.exists(save_path):
    os.remove(save_path)
