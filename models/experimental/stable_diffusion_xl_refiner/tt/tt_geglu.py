import ttnn
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_linear_params,
)


class TtGEGLU:
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        weights = state_dict[f"{self.module_path}.proj.weight"]
        bias = state_dict[f"{self.module_path}.proj.bias"]
        w1, w2 = weights.chunk(2, dim=0)
        b1, b2 = bias.chunk(2, dim=0)

        self.w1 = w1.unsqueeze(0).unsqueeze(0)
        self.w2 = w2.unsqueeze(0).unsqueeze(0)
        self.b1 = b1
        self.b2 = b2

        # TODO: check dtype
        self.value_weights, self.value_bias = prepare_linear_params(device, w1, b1, ttnn.bfloat16)
        self.gate_weights, self.gate_bias = prepare_linear_params(device, w2, b2, ttnn.bfloat16)

    def forward(self, input_tensor):
        hidden_states = ttnn.linear(
            input_tensor,
            self.value_weights,
            bias=self.value_bias,
        )

        gate = ttnn.linear(
            input_tensor,
            self.gate_weights,
            bias=self.gate_bias,
        )

        gate = ttnn.gelu(gate)

        hidden_states = ttnn.mul_(hidden_states, gate, use_legacy=True)
        return hidden_states
