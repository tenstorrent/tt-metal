import torch
from torch import nn
import tt_lib
from python_api_testing.models.llama.llama_utils import tt2torch_tensor, torch2tt_tensor


class TtLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        layer_position,
        hidden_size,
        eps=1e-6,
    ):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()

        self.device = device
        self.variance_epsilon = eps
        self.state_dict = state_dict

        # hadle constant variance_epsilon
        self.variance_epsilon_const = tt_lib.tensor.Tensor(
            [self.variance_epsilon] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            self.device,
        )

        # check if it is final norm layer
        if layer_num is not None:
            pytorch_weights = self.state_dict[
                f"{base_url}.{layer_num}.{layer_position}.weight"
            ]
        else:
            pytorch_weights = self.state_dict[f"model.norm.weight"]

        # get weights
        pytorch_weights = pytorch_weights.repeat(1, 1, 32, 1)
        self.weight = torch2tt_tensor(pytorch_weights, self.device)

    def forward(self, hidden_states):
        # handle variance in PyTorch
        torch_hidden_states = tt2torch_tensor(hidden_states)
        variance = torch_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        variance = variance.repeat(1, 1, 1, 32)
        variance = torch2tt_tensor(variance, self.device)

        # Pytorch implementation for: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Product 2: torch.rsqrt(variance + self.variance_epsilon)
        tmp = tt_lib.tensor.bcast(
            variance,
            self.variance_epsilon_const,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
        )

        # Product 1 * Product 2: hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        tmp = tt_lib.tensor.recip(tt_lib.tensor.sqrt(tmp))
        hidden_states = tt_lib.tensor.bcast(
            hidden_states,
            tmp,
            tt_lib.tensor.BcastOpMath.MUL,
            tt_lib.tensor.BcastOpDim.W,
        )

        # weight * hidden_states
        result = tt_lib.tensor.bcast(
            hidden_states,
            self.weight,
            tt_lib.tensor.BcastOpMath.MUL,
            tt_lib.tensor.BcastOpDim.H,
        )

        return result
