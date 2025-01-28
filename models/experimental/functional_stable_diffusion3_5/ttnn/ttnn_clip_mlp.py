# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ttnn_QuickGELUActivation:
    def __call__(self, input):
        return input * ttnn.sigmoid(1.702 * input)


ACT2FN = {
    # "gelu": GELUActivation,
    # "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    # "gelu_fast": FastGELUActivation,
    # "gelu_new": NewGELUActivation,
    # "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    # "gelu_pytorch_tanh": PytorchGELUTanh,
    # "gelu_accurate": AccurateGELUActivation,
    # "laplace": LaplaceActivation,
    # "leaky_relu": nn.LeakyReLU,
    # "linear": LinearActivation,
    # "mish": MishActivation,
    "quick_gelu": ttnn_QuickGELUActivation(),
    # "relu": nn.ReLU,
    # "relu2": ReLUSquaredActivation,
    # "relu6": nn.ReLU6,
    # "sigmoid": nn.Sigmoid,
    # "silu": nn.SiLU,
    # "swish": nn.SiLU,
    # "tanh": nn.Tanh,
}


class ttnn_CLIPMLP:
    def __init__(self, config):
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = ttnn.linear
        self.fc2 = ttnn.linear

    def __call__(self, hidden_states: ttnn.Tensor, parameters=None) -> ttnn.Tensor:
        hidden_states = self.fc1(
            hidden_states,
            parameters.fc1.weight,
            bias=parameters.fc1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(
            hidden_states,
            parameters.fc2.weight,
            bias=parameters.fc2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return hidden_states
