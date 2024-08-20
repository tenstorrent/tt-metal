# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import torch
import ttnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import ttnn
from models.utility_functions import (
    pad_activation,
    pad_weight,
    tilize_to_list,
    get_oom_of_float,
    is_close,
)

input_dim = 1024
hidden_dim = 256
output_dim = 10
batch_size = 1
eps = 1e-3


class TtMnistModel(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        # Extract params from state dict
        fc1_weight = pad_weight(state_dict["linear1.weight"])
        fc1_bias = pad_weight(state_dict["linear1.bias"])

        fc2_weight = pad_weight(state_dict["linear2.weight"])
        fc2_bias = pad_weight(state_dict["linear2.bias"])

        fc3_weight = pad_weight(state_dict["linear3.weight"])
        fc3_bias = pad_weight(state_dict["linear3.bias"])

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

        # self.lin1 = TtLinear(*fc1_weight_shape[-2:], fc1_weight, fc1_bias, device)
        # self.lin2 = TtLinear(*fc2_weight_shape[-2:], fc2_weight, fc2_bias, device)
        # self.lin3 = TtLinear(*fc3_weight_shape[-2:], fc3_weight, fc3_bias, device)

        self.lin1 = TtLinear(800, 256, fc1_weight, fc1_bias, device)
        self.lin2 = TtLinear(256, 256, fc2_weight, fc2_bias, device)
        self.lin3 = TtLinear(256, 32, fc3_weight, fc3_bias, device)

        # Extract batch norm params from state dict
        gamma1 = pad_weight(state_dict["batchnorm1d_1.weight"])
        beta1 = pad_weight(state_dict["batchnorm1d_1.bias"])

        gamma2 = pad_weight(state_dict["batchnorm1d_2.weight"])
        beta2 = pad_weight(state_dict["batchnorm1d_2.bias"])

        gamma3 = pad_weight(state_dict["batchnorm1d_3.weight"])
        beta3 = pad_weight(state_dict["batchnorm1d_3.bias"])

        # Get shapes

        gamma1_shape = gamma1.shape[3]
        gamma2_shape = gamma2.shape[3]
        gamma3_shape = gamma3.shape[3]

        # Tilize params
        gamma1 = tilize_to_list(gamma1)
        beta1 = tilize_to_list(beta1)

        gamma2 = tilize_to_list(gamma2)
        beta2 = tilize_to_list(beta2)

        gamma3 = tilize_to_list(gamma3)
        beta3 = tilize_to_list(beta3)

        ### running mean and var
        running_mean1 = pad_weight(state_dict["batchnorm1d_1.running_mean"])
        running_var1 = pad_weight(state_dict["batchnorm1d_1.running_var"])

        running_mean2 = pad_weight(state_dict["batchnorm1d_2.running_mean"])
        running_var2 = pad_weight(state_dict["batchnorm1d_2.running_var"])

        running_mean3 = pad_weight(state_dict["batchnorm1d_3.running_mean"])
        running_var3 = pad_weight(state_dict["batchnorm1d_3.running_var"])

        # Get shapes
        print(
            "running mean size before padding:",
            state_dict["batchnorm1d_1.running_mean"].shape,
        )
        print("running mean size after padding:", running_mean1.shape)

        running_mean1_shape = running_mean1.shape
        running_mean2_shape = running_mean2.shape
        running_mean3_shape = running_mean3.shape

        # Tilize params
        running_mean1 = tilize_to_list(running_mean1)
        running_var1 = tilize_to_list(running_var1)

        running_mean2 = tilize_to_list(running_mean2)
        running_var2 = tilize_to_list(running_var2)

        running_mean3 = tilize_to_list(running_mean3)
        running_var3 = tilize_to_list(running_var3)

        ### defining batch norms
        self.batchnorm1d_1 = batchnorm1d_inference(
            gamma1, beta1, running_mean1, running_var1, eps, gamma1_shape, device
        )
        self.batchnorm1d_2 = batchnorm1d_inference(
            gamma2, beta2, running_mean2, running_var2, eps, gamma2_shape, device
        )
        self.batchnorm1d_3 = batchnorm1d_inference(
            gamma3, beta3, running_mean3, running_var3, eps, gamma3_shape, device
        )

        self.TtRelu = ttnn.relu

    # tt forwrd
    def forward(self, X):
        x, labels = X

        # Flatten tensor
        x = x.view(x.shape[0], -1)

        # Pad to tile
        x = pad_activation(x)
        x_ = tilize_to_list(x)

        # x is a pytorch tensor,... need to convert to a buda tensor
        inp = ttnn.Tensor(x_, x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        # breakpoint()
        lin1_out = self.lin1(inp)
        bn1_out = self.batchnorm1d_1(lin1_out)
        relu1_out = self.TtRelu(lin1_out)

        lin2_out = self.lin2(relu1_out)
        bn2_out = self.batchnorm1d_2(lin2_out)
        relu2_out = self.TtRelu(lin2_out)

        lin3_out = self.lin3(relu2_out)
        bn3_out = self.batchnorm1d_3(lin3_out)
        relu3_out = self.TtRelu(lin3_out)

        # Softmax on CPU
        lin3_out_cpu = relu3_out.cpu()

        # Make pytorch tensor... since we had to pad the output, we need
        # to only retrieve the 10 values that represent actual classes
        lin3_out_cpu_pytorch = torch.Tensor(lin3_out_cpu.to_torch())[:, 0, 0, :10]
        out = nn.functional.softmax(lin3_out_cpu_pytorch)

        return out


class PytorchMnistModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, state_dict):
        super(PytorchMnistModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batchnorm1d_1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.batchnorm1d_2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.batchnorm1d_3 = nn.BatchNorm1d(output_dim)
        self.relu3 = nn.ReLU()

        self.load_state_dict(state_dict)

    def forward(self, X):
        x, labels = X
        x = x.view(x.shape[0], -1)

        lin1_out = self.linear1(x)
        bn1_out = self.batchnorm1d_1(lin1_out)
        relu1_out = self.relu1(bn1_out)

        lin2_out = self.linear2(relu1_out)
        bn2_out = self.batchnorm1d_2(lin2_out)
        relu2_out = self.relu1(bn2_out)

        lin3_out = self.linear3(relu2_out)
        bn3_out = self.batchnorm1d_3(lin3_out)
        relu3_out = self.relu3(bn3_out)

        out = nn.functional.softmax(relu3_out)

        return out


def run_mnist_inference():
    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Trained to 63% accuracy
    state_dict = torch.load(f"{Path(__file__).parent}/lfs/synthetic_grads/bn1d_32.pt")

    tt_mnist_model = TtMnistModel(state_dict)
    pytorch_mnist_model = PytorchMnistModel(input_dim, hidden_dim, output_dim, state_dict)
    pytorch_mnist_model.eval()

    first_input = next(iter(dataloader))

    # Run one input through the network
    tt_out = tt_mnist_model(first_input)
    pytorch_out = pytorch_mnist_model(first_input)

    print("tt_out:", tt_out)
    print("pytorch_out:", pytorch_out)
    # assert (tt_out.topk(10).indices == pytorch_out.topk(10).indices).all(), "The outputs from device and pytorch must have the same topk indices"
    print("tt out topk:", tt_out.topk(10))
    print("pytorch out topk:", pytorch_out.topk(10))

    # Check that the scale of each output is the same
    tt_out_oom = get_oom_of_float(tt_out.tolist()[0])
    pytorch_out_oom = get_oom_of_float(pytorch_out.tolist()[0])

    close_or_far = is_close(pytorch_out, tt_out)
    print("close or far?", close_or_far)
    # breakpoint()
    # assert tt_out_oom == pytorch_out_oom, "The order of magnitudes of the outputs must be the same"


def test_run_mnist_inference(device):
    run_mnist_inference()
