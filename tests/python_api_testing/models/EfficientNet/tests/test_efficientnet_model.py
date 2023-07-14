import os
import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import cv2
import tt_lib
import torch
from loguru import logger
import torchvision
from datasets import load_dataset

from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from python_api_testing.models.EfficientNet.tt.efficientnet_model import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)


def download_images(img_path):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image.save(img_path)


def make_input_tensor():
    img_path = ROOT / "input_image.jpg"
    download_images(img_path)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    image = cv2.imread(str(img_path))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def run_efficientnet_model_test(
    reference_model_class, tt_model_class, pcc=0.99, real_input=False
):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = reference_model_class(pretrained=True)

    torch.manual_seed(0)

    if real_input:
        test_input = make_input_tensor()
    else:
        test_input = torch.rand(1, 3, 224, 224)

    with torch.no_grad():
        refence_model.eval()
        pt_out = refence_model(test_input)

    tt_model = tt_model_class(device)

    test_input = torch2tt_tensor(
        test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
    )

    with torch.no_grad():
        tt_model.eval()
        tt_out = tt_model(test_input)
        tt_out = tt2torch_tensor(tt_out)

    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_efficientnet_model {tt_model_class} Passed!")
    else:
        logger.warning(f"test_efficientnet_model {tt_model_class} Failed!")

    assert does_pass


def test_efficientnet_b0_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b0, efficientnet_b0)


def test_efficientnet_b1_model():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b1, efficientnet_b1, 0.97
    )


def test_efficientnet_b2_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b2, efficientnet_b2)


def test_efficientnet_b3_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b3, efficientnet_b3)


def test_efficientnet_b4_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b4, efficientnet_b4)


def test_efficientnet_b5_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b5, efficientnet_b5)


def test_efficientnet_b6_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b6, efficientnet_b6)


def test_efficientnet_b7_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_b7, efficientnet_b7)


def test_efficientnet_v2_s_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_v2_s, efficientnet_v2_s)


def test_efficientnet_v2_m_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_v2_m, efficientnet_v2_m)


def test_efficientnet_v2_l_model():
    run_efficientnet_model_test(torchvision.models.efficientnet_v2_l, efficientnet_v2_l)


def test_efficientnet_b0_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b0, efficientnet_b0, 0.97, real_input=True
    )


def test_efficientnet_b1_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b1, efficientnet_b1, 0.97, real_input=True
    )


def test_efficientnet_b2_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b2, efficientnet_b2, 0.96, real_input=True
    )


def test_efficientnet_b3_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b3, efficientnet_b3, 0.96, real_input=True
    )


def test_efficientnet_b4_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b4, efficientnet_b4, 0.97, real_input=True
    )


def test_efficientnet_b5_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b5, efficientnet_b5, 0.97, real_input=True
    )


def test_efficientnet_b6_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b6, efficientnet_b6, 0.97, real_input=True
    )


def test_efficientnet_b7_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b7, efficientnet_b7, 0.97, real_input=True
    )


def test_efficientnet_v2_s_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_s, efficientnet_v2_s, 0.97, real_input=True
    )


def test_efficientnet_v2_m_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_m, efficientnet_v2_m, 0.97, real_input=True
    )


def test_efficientnet_v2_l_model_real():
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_l, efficientnet_v2_l, 0.97, real_input=True
    )
