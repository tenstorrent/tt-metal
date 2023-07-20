from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import tt_lib
from utils import tt_linear, get_shape
from utility_functions_new import torch2tt_tensor
from tt_lib.fallback_ops import fallback_ops

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout2 = nn.Dropout(0.5) # commented out since makes no different in inference
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class TtConvNet(nn.Module):
    def __init__(self, device=None, state_dict=None) -> None:
        super().__init__()
        self.device = device

        conv1_weight = state_dict["conv1.weight"]
        self.tt_conv1_weight = tt_lib.tensor.Tensor(
            conv1_weight.reshape(-1).tolist(),
            get_shape(conv1_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        conv1_bias = state_dict["conv1.bias"]
        self.tt_conv1_bias = tt_lib.tensor.Tensor(
            conv1_bias.reshape(-1).tolist(),
            get_shape(conv1_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        conv2_weight = state_dict["conv2.weight"]
        self.tt_conv2_weight = tt_lib.tensor.Tensor(
            conv2_weight.reshape(-1).tolist(),
            get_shape(conv2_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        conv2_bias = state_dict["conv2.bias"]
        self.tt_conv2_bias = tt_lib.tensor.Tensor(
            conv2_bias.reshape(-1).tolist(),
            get_shape(conv2_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear1_weights = state_dict[f"fc1.weight"]
        linear1_weights = tt_lib.tensor.Tensor(
            linear1_weights.reshape(-1).tolist(),
            get_shape(linear1_weights.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear1_bias = state_dict[f"fc1.bias"]
        linear1_bias = tt_lib.tensor.Tensor(
            linear1_bias.reshape(-1).tolist(),
            get_shape(linear1_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear2_weights = state_dict[f"fc2.weight"]
        linear2_weights = tt_lib.tensor.Tensor(
            linear2_weights.reshape(-1).tolist(),
            get_shape(linear2_weights.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear2_bias = state_dict[f"fc2.bias"]
        linear2_bias = tt_lib.tensor.Tensor(
            linear2_bias.reshape(-1).tolist(),
            get_shape(linear2_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        self.fc1 = tt_linear(linear1_weights, linear1_bias, self.device)
        self.fc2 = tt_linear(linear2_weights, linear2_bias, self.device)

    def forward(self, tt_x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        out = fallback_ops.conv2d(tt_x, self.tt_conv1_weight, self.tt_conv1_bias)

        out = tt_lib.tensor.relu(out)

        # Convert TT tensor to Torch tensor
        out = out.cpu()
        out = torch.Tensor(out.data()).reshape(out.shape())

        out = F.max_pool2d(out, 2)  # On CPU
        out = fallback_ops.conv2d(out, self.tt_conv2_weight, self.tt_conv2_bias)

        out = tt_lib.tensor.relu(out)

        # Convert TT tensor to Torch tensor
        out = out.cpu()
        out = torch.Tensor(out.data()).reshape(out.shape())

        out = F.max_pool2d(out, 2)  # On CPU

        out = torch.flatten(out, start_dim=1).unsqueeze(1).unsqueeze(1)

        tt_tensor = tt_lib.tensor.Tensor(
            out.reshape(-1).tolist(),
            out.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )

        out = self.fc1(tt_tensor)
        out = tt_lib.tensor.relu(out)
        out = self.fc2(out)  # On Device

        out = fallback_ops.softmax(out)

        return out


def load_torch(weka_path):
    model = ConvNet().to("cpu")
    checkpoint = torch.load(weka_path, map_location=torch.device("cpu"))
    model.eval()
    checkpoint = {"state_dict": model.state_dict()}
    return model, checkpoint["state_dict"]


def prep_data():
    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.05,), std=(0.05,)),
            ]
        ),
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.05,), std=(0.05,)),
            ]
        ),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return test_dataset, test_loader
