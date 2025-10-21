# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import os
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2

weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
if not os.path.exists(weights_path):
    os.system("bash models/demos/mobilenetv2/weights_download.sh")

state_dict = torch.load(weights_path)
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
script_module.save("models/mobilenetv2_cpp/mobilenet_v2-b0353104-script.pt")
