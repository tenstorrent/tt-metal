import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from gpai import gpai
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float
from python_api_testing.fused_ops.linear import Linear as TtLinear


class TtMnistModel(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        # Extract params from state dict
        fc1_weight = pad_weight(state_dict["fc1.weight"])
        fc1_bias = pad_weight(state_dict["fc1.bias"])

        fc2_weight = pad_weight(state_dict["fc2.weight"])
        fc2_bias = pad_weight(state_dict["fc2.bias"])

        fc3_weight = pad_weight(state_dict["fc3.weight"])
        fc3_bias = pad_weight(state_dict["fc3.bias"])

        # Get shapes
        fc1_weight_shape = fc1_weight.shape

        fc2_weight_shape = fc2_weight.shape

        fc3_weight_shape = fc3_weight.shape

        # Tilize params
        fc1_weight = tilize_to_list(fc1_weight)
        fc1_bias = tilize_to_list(fc1_bias)

        fc2_weight = tilize_to_list(fc2_weight)
        fc2_bias = tilize_to_list(fc2_bias)

        fc3_weight = tilize_to_list(fc3_weight)
        fc3_bias = tilize_to_list(fc3_bias)

        self.lin1 = TtLinear(*fc1_weight_shape[-2:], fc1_weight, fc1_bias, device)
        self.lin2 = TtLinear(*fc2_weight_shape[-2:], fc2_weight, fc2_bias, device)
        self.lin3 = TtLinear(*fc3_weight_shape[-2:], fc3_weight, fc3_bias, device)

        # We are doing identity since back to back matmul and activation produces garbage results...
        # probably reading from wrong address
        self.act = gpai.tensor.relu


    def forward(self, X):

        x, labels = X

        # Flatten tensor
        x = x.view(x.shape[0], -1)

        # Pad to tile
        x = pad_activation(x)
        x_ = tilize_to_list(x)

        # x is a pytorch tensor,... need to convert to a buda tensor
        inp = gpai.tensor.Tensor(x_, x.shape, gpai.tensor.DataFormat.FLOAT32, gpai.tensor.Layout.TILE, device)

        lin1_out = self.lin1(inp)
        lin1_out_act = self.act(lin1_out)

        lin2_out = self.lin2(lin1_out_act)

        lin2_out_act = self.act(lin2_out)

        lin3_out = self.lin3(lin2_out_act)
        lin3_out_act = self.act(lin3_out)

        # Softmax on CPU
        lin3_out_cpu = lin3_out_act.to(host)

        # Make pytorch tensor... since we had to pad the output, we need
        # to only retrieve the 10 values that represent actual classes
        lin3_out_cpu_pytorch = torch.Tensor(lin3_out_cpu.data()).reshape(lin3_out_cpu.shape())[:, 0, 0, :10]
        out = torch.nn.functional.softmax(lin3_out_cpu_pytorch)

        return out

class PytorchMnistModel(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.fc1 = torch.nn.Linear(784, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        # Doing identity for same reason as above
        self.act = torch.nn.functional.relu
        self.load_state_dict(state_dict)

    def forward(self, X):
        x, labels = X

        x = x.view(-1)

        lin1_out = self.fc1(x)
        lin1_out_act = self.act(lin1_out)

        lin2_out = self.fc2(lin1_out_act)
        lin2_out_act = self.act(lin2_out)

        lin3_out = self.fc3(lin2_out_act)
        lin3_out_act = self.act(lin3_out)

        out = torch.nn.functional.softmax(lin3_out_act)
        return out


def run_mnist_inference():
    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=1)

    # Trained to 68% accuracy in modelzoo
    state_dict = torch.load(f"{Path(__file__).parent}/mnist_model.pt")

    tt_mnist_model = TtMnistModel(state_dict)
    pytorch_mnist_model = PytorchMnistModel(state_dict)

    first_input = next(iter(dataloader))

    # Run one input through the network
    tt_out = tt_mnist_model(first_input)
    pytorch_out = pytorch_mnist_model(first_input)

    assert (tt_out.topk(10).indices == pytorch_out.topk(10).indices).all(), "The outputs from device and pytorch must have the same topk indices"

    # Check that the scale of each output is the same
    tt_out_oom = get_oom_of_float(tt_out.tolist()[0])
    pytorch_out_oom = get_oom_of_float(pytorch_out.tolist())

    assert tt_out_oom == pytorch_out_oom, "The order of magnitudes of the outputs must be the same"

if __name__ == "__main__":
    # Initialize the device
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    run_mnist_inference()
    gpai.device.CloseDevice(device)
