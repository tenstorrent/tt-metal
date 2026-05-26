import torch
from safetensors import safe_open
import ttnn

def convert_lfm2_weights(model_path, device):
    parameters = {}
    with safe_open(model_path, framework="pt") as f:
        parameters["embed_tokens"] = {"weight": ttnn.from_torch(f.get_tensor("model.embed_tokens.weight"), device=device)}
        parameters["layers"] = []
        for i in range(16):
            layer_params = {}
            parameters["layers"].append(layer_params)
        parameters["norm"] = {"weight": ttnn.from_torch(f.get_tensor("model.norm.weight"), device=device)}
    return parameters