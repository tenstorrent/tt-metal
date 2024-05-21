# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import cv2

from models.experimental.functional_yolox_m.reference.yolox_m import YOLOX
from models.experimental.functional_yolox_m.demo.demo_utils import ValTransform, decode_outputs, postprocess, visual


def inference(torch_model, decode_in_inference=True, num_classes=80, confthre=0.25, nmsthre=0.45):
    img_file = "tests/ttnn/integration_tests/yolox_m/dog.jpg"
    img = cv2.imread(img_file)
    img_info = {}

    img_info["raw_img"] = img
    test_size = (640, 640)
    ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    img_info["ratio"] = ratio
    preprocess = ValTransform()
    img, _ = preprocess(img, None, (640, 640))
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    torch_input_tensor = img

    torch_model.eval()

    with torch.no_grad():
        torch_output_tensors = torch_model(torch_input_tensor)

    torch_output_tensor0 = torch_output_tensors[0]
    torch_output_tensor1 = torch_output_tensors[1]
    torch_output_tensor2 = torch_output_tensors[2]

    outputs = [torch_output_tensor0, torch_output_tensor1, torch_output_tensor2]

    hw = [x.shape[-2:] for x in outputs]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

    if decode_in_inference:
        outputs = decode_outputs(outputs, dtype=img.type(), hw=hw)
    outputs = postprocess(outputs, num_classes, confthre, nmsthre, class_agnostic=True)

    result_image = visual(outputs[0], img_info, confthre)
    save_file_name = "./torch_dog.jpg"
    cv2.imwrite(save_file_name, result_image)


def test_yolox_model(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolox")
    if model_path == "models":
        state_dict = torch.load("tests/ttnn/integration_tests/yolox_m/yolox_m.pth", map_location="cpu")
    else:
        weights_pth = str(model_path / "yolox_m.pth")
        state_dict = torch.load(weights_pth)

    state_dict = state_dict["model"]
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(("backbone", "head")))}
    torch_model = YOLOX()
    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    inference(torch_model)
