import ttnn


class TtPatchTSMixerGatedAttention:
    def __init__(self, device, base_address: str, parameters: dict):
        self.device = device
        self.base_address = base_address

        self.weight = parameters[f"{base_address}.attn_layer.weight"]
        self.bias = parameters[f"{base_address}.attn_layer.bias"]

    def __call__(self, x):
        # x: TTNN tensor, last dimension = d_model
        y = ttnn.linear(x, self.weight, self.bias)
        w = ttnn.softmax(y, dim=-1)
        return ttnn.multiply(x, w)
