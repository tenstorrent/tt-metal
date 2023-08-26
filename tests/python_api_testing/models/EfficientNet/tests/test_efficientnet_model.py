import tt_lib
import torch
from loguru import logger
import torchvision
from datasets import load_dataset

from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from tt_models.EfficientNet.tt.efficientnet_model import (
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
    reference_efficientnet_lite0,
    reference_efficientnet_lite1,
    reference_efficientnet_lite2,
    reference_efficientnet_lite3,
    reference_efficientnet_lite4,
    efficientnet_lite0,
    efficientnet_lite1,
    efficientnet_lite2,
    efficientnet_lite3,
    efficientnet_lite4,
)


def make_input_tensor(imagenet_sample_input, resize=256, crop=224):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return transform(imagenet_sample_input)


def run_efficientnet_model_test(
    reference_model_class,
    tt_model_class,
    imagenet_sample_input,
    pcc=0.99,
    real_input=False,
    resize=256,
    crop=224,
):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = reference_model_class(pretrained=True)
    torch.manual_seed(0)

    if real_input:
        test_input = make_input_tensor(imagenet_sample_input, resize, crop)
    else:
        test_input = torch.rand(1, 3, crop, crop)

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


def test_efficientnet_b0_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b0, efficientnet_b0, imagenet_sample_input
    )


def test_efficientnet_b1_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b1, efficientnet_b1, imagenet_sample_input, 0.97
    )


def test_efficientnet_b2_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b2, efficientnet_b2, imagenet_sample_input
    )


def test_efficientnet_b3_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b3, efficientnet_b3, imagenet_sample_input
    )


def test_efficientnet_b4_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b4, efficientnet_b4, imagenet_sample_input
    )


def test_efficientnet_b5_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b5, efficientnet_b5, imagenet_sample_input
    )


def test_efficientnet_b6_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b6, efficientnet_b6, imagenet_sample_input
    )


def test_efficientnet_b7_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b7, efficientnet_b7, imagenet_sample_input
    )


def test_efficientnet_v2_s_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_s, efficientnet_v2_s, imagenet_sample_input
    )


def test_efficientnet_v2_m_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_m, efficientnet_v2_m, imagenet_sample_input
    )


def test_efficientnet_v2_l_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_l, efficientnet_v2_l, imagenet_sample_input
    )


def test_efficientnet_lite0_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite0,
        efficientnet_lite0,
        imagenet_sample_input,
        0.99,
        real_input=False,
    )


def test_efficientnet_lite1_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite1,
        efficientnet_lite1,
        imagenet_sample_input,
        0.99,
        real_input=False,
        resize=280,
        crop=240,
    )


def test_efficientnet_lite2_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite2,
        efficientnet_lite2,
        imagenet_sample_input,
        0.99,
        real_input=False,
        resize=300,
        crop=260,
    )


def test_efficientnet_lite3_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite3,
        efficientnet_lite3,
        imagenet_sample_input,
        0.99,
        real_input=False,
        resize=320,
        crop=280,
    )


def test_efficientnet_lite4_model_synt(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite4,
        efficientnet_lite4,
        imagenet_sample_input,
        0.99,
        real_input=False,
        resize=350,
        crop=300,
    )


def test_efficientnet_b0_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b0,
        efficientnet_b0,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_b1_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b1,
        efficientnet_b1,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_b2_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b2,
        efficientnet_b2,
        imagenet_sample_input,
        0.96,
        real_input=True,
    )


def test_efficientnet_b3_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b3,
        efficientnet_b3,
        imagenet_sample_input,
        0.96,
        real_input=True,
    )


def test_efficientnet_b4_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b4,
        efficientnet_b4,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_b5_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b5,
        efficientnet_b5,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_b6_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b6,
        efficientnet_b6,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_b7_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_b7,
        efficientnet_b7,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_v2_s_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_s,
        efficientnet_v2_s,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_v2_m_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_m,
        efficientnet_v2_m,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_v2_l_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        torchvision.models.efficientnet_v2_l,
        efficientnet_v2_l,
        imagenet_sample_input,
        0.97,
        real_input=True,
    )


def test_efficientnet_lite0_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite0,
        efficientnet_lite0,
        imagenet_sample_input,
        0.98,
        real_input=True,
    )


def test_efficientnet_lite1_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite1,
        efficientnet_lite1,
        imagenet_sample_input,
        0.98,
        real_input=True,
        resize=280,
        crop=240,
    )


def test_efficientnet_lite2_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite2,
        efficientnet_lite2,
        imagenet_sample_input,
        0.97,
        real_input=True,
        resize=300,
        crop=260,
    )


def test_efficientnet_lite3_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite3,
        efficientnet_lite3,
        imagenet_sample_input,
        0.97,
        real_input=True,
        resize=320,
        crop=280,
    )


def test_efficientnet_lite4_model_real(imagenet_sample_input):
    run_efficientnet_model_test(
        reference_efficientnet_lite4,
        efficientnet_lite4,
        imagenet_sample_input,
        0.97,
        real_input=True,
        resize=350,
        crop=300,
    )
