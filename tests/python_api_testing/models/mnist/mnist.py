from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../../..")

import torch

from libs import tt_lib
from utility_functions import get_oom_of_float
from utils import tt_linear, get_shape


class TtMnistModel(torch.nn.Module):
    def __init__(self, device=None, host=None, state_dict=None) -> None:
        super().__init__()
        self.device = device
        self.host = host

        # Extract params from state dict
        fc1_weight = state_dict["fc1.weight"]
        fc1_weight_tt = tt_lib.tensor.Tensor(
            fc1_weight.reshape(-1).tolist(),
            get_shape(fc1_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )
        fc1_bias = state_dict["fc1.bias"]
        fc1_bias_tt = tt_lib.tensor.Tensor(
            fc1_bias.reshape(-1).tolist(),
            get_shape(fc1_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )

        fc2_weight = state_dict["fc2.weight"]
        fc2_weight_tt = tt_lib.tensor.Tensor(
            fc2_weight.reshape(-1).tolist(),
            get_shape(fc2_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )
        fc2_bias = state_dict["fc2.bias"]
        fc2_bias_tt = tt_lib.tensor.Tensor(
            fc2_bias.reshape(-1).tolist(),
            get_shape(fc2_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )

        fc3_weight = state_dict["fc3.weight"]
        fc3_weight_tt = tt_lib.tensor.Tensor(
            fc3_weight.reshape(-1).tolist(),
            get_shape(fc3_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )
        fc3_bias = state_dict["fc3.bias"]
        fc3_bias_tt = tt_lib.tensor.Tensor(
            fc3_bias.reshape(-1).tolist(),
            get_shape(fc3_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )

        self.lin1 = tt_linear(fc1_weight_tt, fc1_bias_tt, self.device)
        self.lin2 = tt_linear(fc2_weight_tt, fc2_bias_tt, self.device)
        self.lin3 = tt_linear(fc3_weight_tt, fc3_bias_tt, self.device)

        # We are doing identity since back to back matmul and activation produces garbage results...
        # probably reading from wrong address
        self.act = tt_lib.tensor.relu

    def forward(self, X):
        x, labels = X

        # x is a pytorch tensor,... need to convert to a TT_tensor
        inp = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            [1, 1, 1, x.shape[2] * x.shape[3]],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )

        lin1_out = self.lin1(inp)
        lin1_out_act = self.act(lin1_out)

        lin2_out = self.lin2(lin1_out_act)
        lin2_out_act = self.act(lin2_out)

        lin3_out = self.lin3(lin2_out_act)
        lin3_out_act = self.act(lin3_out)

        # Softmax on CPU
        lin3_out_cpu = lin3_out_act.to(self.host)

        # Make pytorch tensor... since we had to pad the output, we need
        # to only retrieve the 10 values that represent actual classes
        lin3_out_cpu_pytorch = torch.Tensor(lin3_out_cpu.data()).reshape(
            lin3_out_cpu.shape()
        )[:, 0, 0, :10]
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
