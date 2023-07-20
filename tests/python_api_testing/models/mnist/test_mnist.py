from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../../..")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import pytest

import tt_lib
from utility_functions_new import comp_pcc, get_oom_of_float
from mnist import *

_batch_size = 1


def run_mnist_inference(pcc):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=_batch_size)

    # Trained to 68% accuracy in modelzoo
    state_dict = torch.load(f"{Path(__file__).parent}/mnist_model.pt")

    tt_mnist_model = TtMnistModel(device, host, state_dict)
    pytorch_mnist_model = PytorchMnistModel(state_dict)

    with torch.no_grad():
        first_input = next(iter(dataloader))

        x, _ = first_input
        tt_image = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            [1, 1, 1, x.shape[2] * x.shape[3]],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            device,
        )

        # Run one input through the network
        tt_output = tt_mnist_model(tt_image)
        pytorch_out = pytorch_mnist_model(first_input)

        pcc_passing, pcc_output = comp_pcc(pytorch_out, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        assert (
            tt_output.topk(10).indices == pytorch_out.topk(10).indices
        ).all(), "The outputs from device and pytorch must have the same topk indices"

        # Check that the scale of each output is the same
        tt_out_oom = get_oom_of_float(tt_output.view(-1).tolist())
        pytorch_out_oom = get_oom_of_float(pytorch_out.tolist())

        assert (
            tt_out_oom == pytorch_out_oom
        ), "The order of magnitudes of the outputs must be the same"

        tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mnist_inference(pcc):
    run_mnist_inference(pcc)


if __name__ == "__main__":
    run_mnist_inference(0.99)
