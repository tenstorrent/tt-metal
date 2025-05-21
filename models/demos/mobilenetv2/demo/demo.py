# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

import ttnn
from models.demos.mobilenetv2.demo.demo_utils import get_batch, get_data_loader
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import create_mobilenetv2_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


def mobilenetv2_imagenet_demo(
    device,
    reset_seeds,
    model_location_generator,
    imagenet_label_dict,
    iterations,
    batch_size,
    res,
    use_pretrained_weight,
):
    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/mobilenetv2/weights_download.sh")

    torch_model = Mobilenetv2()
    if use_pretrained_weight:
        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {
            name1: parameter2
            for (name1, _), (_, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        torch_model.load_state_dict(new_state_dict)
    else:
        state_dict = torch_model.state_dict()

    torch_model.eval()

    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(parameters, device, batchsize=batch_size)
    correct = 0

    for iter in range(iterations):
        predictions = []
        inputs, labels = get_batch(data_loader, res)
        torch_input_tensor = inputs.reshape(batch_size, 3, res, res)
        torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

        ttnn_input_tensor = torch_input_tensor.reshape(
            1,
            1,
            torch_input_tensor.shape[0] * torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
        )

        ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)
        tt_output = ttnn_model(ttnn_input_tensor)
        tt_output = ttnn.from_device(tt_output, blocking=True).to_torch().to(torch.float)
        prediction = tt_output.argmax(dim=-1)

        for i in range(batch_size):
            predicted_label = imagenet_label_dict[prediction[i].item()]
            true_label = imagenet_label_dict[labels[i]]
            predictions.append(predicted_label)
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {true_label} -- Predicted Label: {predicted_label}"
            )
            if true_label == predicted_label:
                correct += 1

        del tt_output, inputs, labels, predictions

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight", [False, True], ids=["pretrained_weight_false", "pretrained_weight_true"]
)
@pytest.mark.parametrize("batch_size, resolution, iterations", [[8, 224, 100]])
def test_mobilenetv2_imagenet_demo(
    device,
    reset_seeds,
    iterations,
    batch_size,
    resolution,
    use_pretrained_weight,
    model_location_generator,
    imagenet_label_dict,
):
    mobilenetv2_imagenet_demo(
        device,
        reset_seeds,
        model_location_generator,
        imagenet_label_dict,
        iterations,
        batch_size,
        resolution,
        use_pretrained_weight,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_mobilenetv2_demo(device, reset_seeds, batch_size=8):
    weights_path = "models/demos/functional_mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/functional_mobilenetv2/weights_download.sh")

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}
    torch_model = Mobilenetv2()

    new_state_dict = {
        name1: parameter2
        for (name1, _), (_, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open("models/experimental/functional_mobilenetv2/demo/images/strawberry.jpg")
    img_t = transform(img)
    img_batch_t = img_t.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    torch_input_tensor = img_batch_t
    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)

    ttnn_model = ttnn_mobilenetv2.MobileNetV2(parameters, device, batchsize=batch_size)
    output_tensor = ttnn_model(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(torch_output_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    with open("models/experimental/functional_mobilenetv2/demo/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(output_tensor, 1)
    percentage = torch.nn.functional.softmax(output_tensor, dim=1)[0] * 100
    logger.info("\033[1m" + f"Predicted class: {classes[index[0]]}")
    logger.info("\033[1m" + f"Probability: {percentage[index[0]].item():.2f}%")

    _, indices = torch.sort(output_tensor, descending=True)
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.91)
