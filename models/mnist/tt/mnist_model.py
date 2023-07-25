from pathlib import Path
import torch
import tt_lib

from tt_lib.fallback_ops import fallback_ops
from tt_lib.fused_ops.softmax import softmax as tt_softmax
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.mnist.reference.mnist import MnistModel


class TtMnistModel(torch.nn.Module):
    def __init__(self, device, state_dict):
        super().__init__()
        self.device = device

        # Extract params from state dict
        self.fc1_weight = torch2tt_tensor(
            state_dict["fc1.weight"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.fc1_bias = torch2tt_tensor(
            state_dict["fc1.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.fc2_weight = torch2tt_tensor(
            state_dict["fc2.weight"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.fc2_bias = torch2tt_tensor(
            state_dict["fc2.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.fc3_weight = torch2tt_tensor(
            state_dict["fc3.weight"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.fc3_bias = torch2tt_tensor(
            state_dict["fc3.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.fc1_weight = tt_lib.tensor.transpose(self.fc1_weight)
        self.fc2_weight = tt_lib.tensor.transpose(self.fc2_weight)
        self.fc3_weight = tt_lib.tensor.transpose(self.fc3_weight)

    def forward(self, x):
        # tt_lib.tensor.reshape throws an assertion RuntimeError: TT_ASSERT @ libs/tt_dnn/op_library/reshape/reshape_op.cpp:295: input_tensor_a.shape()[3] % TILE_WIDTH == 0 && W % TILE_WIDTH == 0 info:
        # Operand/target width must be a multiple of 32. So using fallback_ops.reshape.
        x = tt_lib.fallback_ops.reshape(x, x.shape()[0], 1, 1, 784)

        x = tt_lib.tensor.matmul(x, self.fc1_weight)
        x = tt_lib.tensor.bcast(
            x, self.fc1_bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
        x = tt_lib.tensor.relu(x)

        x = tt_lib.tensor.matmul(x, self.fc2_weight)
        x = tt_lib.tensor.bcast(
            x, self.fc2_bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
        x = tt_lib.tensor.relu(x)

        x = tt_lib.tensor.matmul(x, self.fc3_weight)
        x = tt_lib.tensor.bcast(
            x, self.fc3_bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
        x = tt_lib.tensor.relu(x)

        x = tt_lib.fused_ops.softmax.softmax(x)
        # x = fallback_ops.softmax(x, -1)
        return x


def _mnist_model(device, state_dict) -> TtMnistModel:
    return TtMnistModel(device, state_dict)


def mnist_model(device) -> TtMnistModel:
    # Trained to 68% accuracy in modelzoo
    state_dict = torch.load(f"{Path(__file__).parent}/mnist_model.pt")

    tt_model = _mnist_model(device, state_dict)
    pt_model = MnistModel(state_dict)

    return tt_model, pt_model
