import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import tt_lib

from tt_models.utility_functions import torch2tt_tensor, tt2torch_tensor, \
    comp_pcc, \
    get_oom_of_float
from tt_models.mnist.tt.mnist_model import mnist_model


def test_mnist_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)

    # Load model
    tt_model, pt_model = mnist_model(device)

    with torch.no_grad():
        test_input, _ = next(iter(dataloader))
        tt_input = torch2tt_tensor(
            test_input, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        pt_output = pt_model(test_input)
        tt_output = tt_model(tt_input)
        tt_output = tt2torch_tensor(tt_output)

        tt_lib.device.CloseDevice(device)

    pcc_passing, pcc_output = comp_pcc(pt_output, tt_output, 0.99)
    logger.info(f"Output {pcc_output}")

    assert pcc_passing, f"Model output does not meet PCC requirement {0.99}."

    assert (
        tt_output.topk(10).indices == pt_output.topk(10).indices
    ).all(), "The outputs from device and pytorch must have the same topk indices"

    # Check that the scale of each output is the same
    tt_out_oom = get_oom_of_float(tt_output.view(-1).tolist())
    pytorch_out_oom = get_oom_of_float(pt_output.tolist())

    assert (
        tt_out_oom == pytorch_out_oom
    ), "The order of magnitudes of the outputs must be the same"
