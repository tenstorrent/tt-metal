import torch
from torch import nn
from libs import tt_lib as ttl
from python_api_testing.models.llama.llama_utils import tt2torch_tensor, torch2tt_tensor


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class TtLlamaRMSNorm(nn.Module):
    def __init__(self, device, state_dict, base_url, layer_num, layer_position, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()

        self.device = device
        self.variance_epsilon = eps
        # get weights
        self.state_dict = state_dict
        # check if it is final norm layer
        if layer_num is not None:
            pytorch_weights = self.state_dict[f"{base_url}.{layer_num}.{layer_position}.weight"]
        else:
            pytorch_weights = self.state_dict[f"model.norm.weight"]

        pytorch_weights = pytorch_weights.repeat(1, 1, 32, 1)
        self.weight = torch2tt_tensor(pytorch_weights, self.device)

    def forward(self, hidden_states):
        # handle variance in PyTorch
        torch_hidden_states = tt2torch_tensor(hidden_states)
        variance = torch_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        variance = variance.repeat(1, 1, 1, 32)
        tt_variance = torch2tt_tensor(variance, self.device)

        # Pytorch implementation for: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # handle constant variance_epsilon
        tt_variance_epsilon_const = ttl.tensor.Tensor(
            [self.variance_epsilon] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            self.device
        )

        # Product 2: torch.rsqrt(variance + self.variance_epsilon)
        op_add = ttl.tensor.bcast(tt_variance, tt_variance_epsilon_const, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)
        term_2 = ttl.tensor.recip(ttl.tensor.sqrt(op_add))

        # Product 1 * Product 2: hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = ttl.tensor.bcast(hidden_states, term_2, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W)

        # weight * hidden_states
        result = ttl.tensor.bcast(hidden_states, self.weight, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H)
        return result
