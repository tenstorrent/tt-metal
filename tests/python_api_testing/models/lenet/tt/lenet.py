import tt_lib
import torch
import torch.nn as nn
from loguru import logger

from lenet_utils import load_torch_lenet
from utility_functions_new import torch2tt_tensor, tt2torch_tensor

class TtLeNet5(nn.Module):
    def __init__(self, num_classes, device, host, state_dict):
        super().__init__()
        self.device = device
        self.host = host

        # conv1_weight = state_dict["layer1.0.weight"]
        # conv1_bias = state_dict["layer1.0.bias"].tolist()
        # self.conv1_params = [6, 1, 5, 5, 1, 1, 0, 0, 1, 1]
        # self.conv1_on_tt = TtConv(conv1_weight.reshape(-1).tolist(), self.conv1_params, self.device, conv1_bias)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv1.weight = nn.Parameter(state_dict["layer1.0.weight"])
        self.conv1.bias = nn.Parameter(state_dict["layer1.0.bias"])

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm1.weight = nn.Parameter(state_dict["layer1.1.weight"])
        self.batch_norm1.bias = nn.Parameter(state_dict["layer1.1.bias"])
        self.batch_norm1.running_mean = nn.Parameter(
            state_dict["layer1.1.running_mean"]
        )
        self.batch_norm1.running_var = nn.Parameter(state_dict["layer1.1.running_var"])
        self.batch_norm1.num_batches_tracked = nn.Parameter(
            state_dict["layer1.1.num_batches_tracked"], requires_grad=False
        )

        self.relu1 = tt_lib.tensor.relu

        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2_weight = state_dict["layer2.0.weight"]
        # conv2_bias = state_dict["layer2.0.bias"].tolist()
        # self.conv2_params = [16, 6, 5, 5, 1, 1, 0, 0, 1, 1]
        # self.conv2_on_tt = TtConv(conv2_weight.reshape(-1).tolist(), self.conv2_params, self.device, conv2_bias)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv2.weight = nn.Parameter(state_dict["layer2.0.weight"])
        self.conv2.bias = nn.Parameter(state_dict["layer2.0.bias"])

        self.batch_norm2 = nn.BatchNorm2d(16)
        self.batch_norm2.weight = nn.Parameter(state_dict["layer2.1.weight"])
        self.batch_norm2.bias = nn.Parameter(state_dict["layer2.1.bias"])
        self.batch_norm2.running_mean = nn.Parameter(
            state_dict["layer2.1.running_mean"]
        )
        self.batch_norm2.running_var = nn.Parameter(state_dict["layer2.1.running_var"])
        self.batch_norm2.num_batches_tracked = nn.Parameter(
            state_dict["layer2.1.num_batches_tracked"], requires_grad=False
        )

        self.relu2 = tt_lib.tensor.relu

        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        fc_weights = state_dict[f"fc.weight"]
        self.fc_weights = torch2tt_tensor(
            fc_weights.reshape(list((1, 1) + fc_weights.shape)),
            self.device,
            tt_lib.tensor.Layout.ROW_MAJOR
        )
        fc_bias = state_dict[f"fc.bias"]
        self.fc_bias = torch2tt_tensor(
            fc_bias.reshape(list((1, 1, 1) + fc_bias.shape)),
            self.device,
            tt_lib.tensor.Layout.ROW_MAJOR
        )

        fc1_weights = state_dict[f"fc1.weight"]
        self.fc1_weights = torch2tt_tensor(
            fc1_weights.reshape(list((1, 1) + fc1_weights.shape)),
            self.device,
            tt_lib.tensor.Layout.ROW_MAJOR
        )
        fc1_bias = state_dict[f"fc1.bias"]
        self.fc1_bias = torch2tt_tensor(
            fc1_bias.reshape(list((1, 1, 1) + fc1_bias.shape)),
            self.device,
            tt_lib.tensor.Layout.ROW_MAJOR
        )

        fc2_weights = state_dict[f"fc2.weight"]
        self.fc2_weights = torch2tt_tensor(
            fc2_weights.reshape(list((1, 1) + fc2_weights.shape)),
            self.device,
            tt_lib.tensor.Layout.ROW_MAJOR
        )
        fc2_bias = state_dict[f"fc2.bias"]
        self.fc2_bias = torch2tt_tensor(
            fc2_bias.reshape(list((1, 1, 1) + fc2_bias.shape)),
            self.device,
            tt_lib.tensor.Layout.ROW_MAJOR
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = tt2torch_tensor(x, self.host)

        out = self.conv1(x)  # HOST
        out = self.batch_norm1(out)  # HOST

        tt_tensor = torch2tt_tensor(out, self.device, tt_lib.tensor.Layout.ROW_MAJOR)

        out = self.relu1(tt_tensor)  # DEVICE

        out = tt2torch_tensor(out, self.host)

        out = self.maxp1(out)  # HOST
        out = self.conv2(out)  # HOST
        out = self.batch_norm2(out)  # HOST

        tt_tensor = torch2tt_tensor(out, self.device, tt_lib.tensor.Layout.ROW_MAJOR)

        out = self.relu2(tt_tensor)  # DEVICE

        out = tt2torch_tensor(out, self.host)

        out = self.maxp2(out)  # HOST
        out = out.reshape(out.size(0), 1, 1, -1)  # HOST: modifed to have 4 dims

        tt_tensor = torch2tt_tensor(out, self.device, tt_lib.tensor.Layout.ROW_MAJOR)

        # fc
        weight_T = tt_lib.tensor.transpose(self.fc_weights)
        output = tt_lib.tensor.matmul(tt_tensor, weight_T)
        out = tt_lib.tensor.bcast(
            output,
            self.fc_bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
        )
        # relu 2
        out = self.relu2(out)
        # fc1
        weight_T = tt_lib.tensor.transpose(self.fc1_weights)
        output = tt_lib.tensor.matmul(out, weight_T)
        out = tt_lib.tensor.bcast(
            output,
            self.fc1_bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
        )
        # relu 2
        out = self.relu2(out)
        # fc2
        weight_T = tt_lib.tensor.transpose(self.fc2_weights)
        output = tt_lib.tensor.matmul(out, weight_T)
        out = tt_lib.tensor.bcast(
            output,
            self.fc2_bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
        )

        return out


def _lenet5(num_classes, device, host, state_dict) -> TtLeNet5:
    return TtLeNet5(num_classes, device, host, state_dict)


def lenet5(num_classes, device, host, model_location_generator) -> TtLeNet5:
    pt_model_path = model_location_generator("tt_dnn-models/LeNet/model.pt")
    _, state_dict = load_torch_lenet(pt_model_path, num_classes)
    model = _lenet5(num_classes, device, host, state_dict)
    return model
