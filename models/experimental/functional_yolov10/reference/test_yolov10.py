# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import requests
import sys
import torch.nn as nn
from pathlib import Path
from models.experimental.functional_yolov10.reference import yolov10_utils
from models.experimental.functional_yolov10.reference.yolov10 import YOLOv10
from tests.ttnn.utils_for_testing import assert_with_pcc

try:
    sys.modules["ultralytics"] = yolov10_utils
    sys.modules["ultralytics.nn.tasks"] = yolov10_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov10_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov10_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov10_utils
except KeyError:
    print("models.experimental.functional_yolov10x.reference.YoloV10x not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def download_yolov10x_weights():
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt"
    weights_path = "models/experimental/functional_yolov10/reference/yolov10x.pt"

    response = requests.get(url)
    if response.status_code == 200:
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print(f"YOLOv10x weights downloaded successfully and saved to {weights_path}")
    else:
        print("Failed to download YOLOv10x weights. Status code:", response.status_code)


def attempt_download(file, repo="ultralytics/assets/"):
    tests = Path(__file__).parent.parent / "tests"
    file = tests / Path(str(file).strip().replace("'", "").lower())
    file = Path(str(file).strip().replace("'", "").lower())
    if not file.exists():
        assets = [
            "yolov10x.pt",
        ]
        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False
            try:
                url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1e6
            except Exception as e:
                assert redundant, "No secondary mirror"
                url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                os.system(f"curl -L {url} -o {file}")
            finally:
                if not file.exists() or file.stat().st_size < 1e6:
                    file.unlink(missing_ok=True)
                return


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        w = Path(__file__).parent.parent / "reference" / "yolov10x.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x(device, use_program_cache, reset_seeds):
    torch_input = torch.rand(1, 3, 640, 640)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    reference_model_output = reference_model(torch_input)[0]

    with torch.inference_mode():
        torch_model_output = torch_model(torch_input)[0]

    assert_with_pcc(reference_model_output, torch_model_output, 1.0)
