import torch
import tt_lib
from python_api_testing.models.t5.t5_utils import tt2torch_tensor, torch2tt_tensor


# class T5LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
#         # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
#         # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
#         # half-precision inputs is done in fp32

#         variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)

#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         t = torch.rsqrt(variance + self.variance_epsilon)

#         # convert into half-precision if necessary
#         if self.weight.dtype in [torch.float16, torch.bfloat16]:
#             hidden_states = hidden_states.to(self.weight.dtype)
#         return self.weight * hidden_states


class TtT5LayerNorm(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.variance_epsilon = config["layer_norm_epsilon"]
        self.device = device

        # hadle constant variance_epsilon
        self.variance_epsilon_const = tt_lib.tensor.Tensor(
            [self.variance_epsilon] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.TILE,
            self.device,
        )

        # get weights
        pytorch_weights = state_dict[f"{base_address}.weight"]
        pytorch_weights = pytorch_weights.repeat(1, 1, 32, 1)

        self.weight = torch2tt_tensor(pytorch_weights, device)

    def forward(self, hidden_states):
        # variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        torch_hidden_states = tt2torch_tensor(hidden_states)
        variance = torch_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        variance = variance.repeat(1, 1, 1, 32)
        variance = torch2tt_tensor(variance, self.device)

        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        tmp = tt_lib.tensor.bcast(
            variance,
            self.variance_epsilon_const,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
        )
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
