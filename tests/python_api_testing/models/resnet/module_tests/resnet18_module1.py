from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from resnet import *


def test_resnet18_module1():
    # inputs
    layer1_input = (1, 64, 64, 64)
    input_shape = layer1_input

    with torch.no_grad():
        # torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module1 = torch_resnet.layer1

        layer1 = _make_layer(BasicBlock, 64, 2, name="layer1", state_dict=state_dict)

        print(torch_module1[0])
        print(layer1[0])

        input = torch.randn(input_shape)

        torch_output = torch_module1(input)
        tt_output = layer1(input)

        print(torch_output.shape, tt_output.shape)
        print(comp_allclose_and_pcc(torch_output, tt_output), "outputs")
