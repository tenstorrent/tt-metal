import torch
import tt_lib
from models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


def TtLinear(
    x: tt_lib.tensor.Tensor,
    weight: tt_lib.tensor.Tensor,
    bias: tt_lib.tensor.Tensor = None,
) -> tt_lib.tensor.Tensor:
    weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x


class TtClassifier(torch.nn.Module):
    def __init__(
        self,
        state_dict=None,
        device=None,
        host=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.host = host

        self.classifier1_weight = torch_to_tt_tensor_rm(
            state_dict["classifier.0.weight"], device, put_on_device=False
        )
        self.classifier1_bias = torch_to_tt_tensor_rm(
            state_dict["classifier.0.bias"], device, put_on_device=False
        )
        self.classifier2_weight = torch_to_tt_tensor_rm(
            state_dict["classifier.3.weight"], device, put_on_device=False
        )

        self.classifier2_bias = torch_to_tt_tensor_rm(
            state_dict["classifier.3.bias"], device, put_on_device=False
        )
        self.scale_activation = torch.nn.Hardswish()

    def forward(self, input: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        linear_1 = TtLinear(input, self.classifier1_weight, self.classifier1_bias)
        scale = tt_to_torch_tensor(linear_1, self.host)
        scale = self.scale_activation(scale)
        scale = torch_to_tt_tensor_rm(scale, self.device)
        linear_2 = TtLinear(scale, self.classifier2_weight, self.classifier2_bias)
        return linear_2
