# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_linear_params,
)


class WeightLoader:
    def __init__(self, component_instance, state_dict, module_path="", **kwargs):
        component_class_name = component_instance.__class__.__name__

        # Map component class names to weight loader classes
        loader_mapping = {
            "TtResnetBlock2D": ResNetWeightLoader,
            "TtBasicTransformerBlock": TransformerBlockWeightLoader,
            "TtAttention": AttentionWeightLoader,
            "TtFeedForward": FeedForwardWeightLoader,
            "TtGEGLU": GEGLUWeightLoader,
            "TtUNet2DConditionModel": UNetWeightLoader,
            "TtDownsample2D": DownSampleWeightLoader,
            "TtUpsample2D": UpSampleWeightLoader,
            "TtTransformer2DModel": Transformer2DModelWeightLoader,
        }

        # Create the appropriate delegate weight loader with standardized parameters
        if component_class_name in loader_mapping:
            loader_class = loader_mapping[component_class_name]
            device = component_instance.device if hasattr(component_instance, "device") else None

            if loader_class == ResNetWeightLoader:
                delegate_loader = loader_class(
                    state_dict, module_path, device, use_conv_shortcut=kwargs.get("use_conv_shortcut", False)
                )
            else:
                delegate_loader = loader_class(state_dict, module_path, device)
        else:
            raise ValueError(f"No weight loader found for component class: {component_class_name}")

        # Store the delegate loader for dynamic attribute access
        self._delegate_loader = delegate_loader

        # Forward all weight properties and methods from delegate to this instance
        for attr_name in dir(delegate_loader):
            if not attr_name.startswith("_"):
                attr_value = getattr(delegate_loader, attr_name)
                # Forward both properties (non-callable) and custom methods (callable)
                # But exclude built-in methods like __class__, __dict__, etc.
                if not callable(attr_value) or hasattr(delegate_loader.__class__, attr_name):
                    setattr(self, attr_name, attr_value)

    # Dynamic attribute access forwarding
    def __getattr__(self, name):
        if hasattr(self, "_delegate_loader") and hasattr(self._delegate_loader, name):
            return getattr(self._delegate_loader, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ResNetWeightLoader:
    def __init__(self, state_dict, module_path, device, use_conv_shortcut=False):
        self.module_path = module_path
        self.device = device
        self.use_conv_shortcut = use_conv_shortcut
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        # Norm weights
        self.norm_weights_1 = state_dict[f"{self.module_path}.norm1.weight"]
        self.norm_bias_1 = state_dict[f"{self.module_path}.norm1.bias"]
        self.norm_weights_2 = state_dict[f"{self.module_path}.norm2.weight"]
        self.norm_bias_2 = state_dict[f"{self.module_path}.norm2.bias"]

        # Conv weights
        self.conv_weights_1 = state_dict[f"{self.module_path}.conv1.weight"]
        self.conv_bias_1 = state_dict[f"{self.module_path}.conv1.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.conv_weights_2 = state_dict[f"{self.module_path}.conv2.weight"]
        self.conv_bias_2 = state_dict[f"{self.module_path}.conv2.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Time embedding weights
        self.time_emb_weights = state_dict[f"{self.module_path}.time_emb_proj.weight"]
        self.time_emb_bias = state_dict[f"{self.module_path}.time_emb_proj.bias"]

        # Shortcut weights
        if f"{self.module_path}.conv_shortcut.weight" in state_dict:
            self.conv_weights_3 = state_dict[f"{self.module_path}.conv_shortcut.weight"]
            self.conv_bias_3 = (
                state_dict[f"{self.module_path}.conv_shortcut.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            self.has_conv_shortcut = True
        else:
            self.conv_weights_3 = None
            self.conv_bias_3 = None
            self.has_conv_shortcut = False


class TransformerBlockWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        self.norm_weights_1 = state_dict[f"{self.module_path}.norm1.weight"]
        self.norm_bias_1 = state_dict[f"{self.module_path}.norm1.bias"]

        self.norm_weights_2 = state_dict[f"{self.module_path}.norm2.weight"]
        self.norm_bias_2 = state_dict[f"{self.module_path}.norm2.bias"]

        self.norm_weights_3 = state_dict[f"{self.module_path}.norm3.weight"]
        self.norm_bias_3 = state_dict[f"{self.module_path}.norm3.bias"]


class AttentionWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        self.q_weights = state_dict[f"{self.module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        self.k_weights = state_dict[f"{self.module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        self.v_weights = state_dict[f"{self.module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)
        self.out_weights = state_dict[f"{self.module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        self.out_bias = state_dict[f"{self.module_path}.to_out.0.bias"]


class FeedForwardWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        # GEGLU weights (handled by separate loader)
        self.net_0_proj_weight = state_dict[f"{self.module_path}.net.0.proj.weight"]
        self.net_0_proj_bias = state_dict[f"{self.module_path}.net.0.proj.bias"]

        # Final linear layer weights
        self.net_2_weight = state_dict[f"{self.module_path}.net.2.weight"].unsqueeze(0).unsqueeze(0)
        self.net_2_bias = state_dict[f"{self.module_path}.net.2.bias"]

    def prepare_linear_params(self, dtype):
        # Prepare net.2 linear parameters
        self.linear_weights, self.linear_bias = prepare_linear_params(
            self.device, self.net_2_weight, self.net_2_bias, dtype
        )


class GEGLUWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        proj_weight = state_dict[f"{self.module_path}.proj.weight"]
        proj_bias = state_dict[f"{self.module_path}.proj.bias"]

        # Chunk weights and bias into two parts (value and gate)
        w1, w2 = proj_weight.chunk(2, dim=0)
        b1, b2 = proj_bias.chunk(2, dim=0)

        self.w1 = w1.unsqueeze(0).unsqueeze(0)
        self.w2 = w2.unsqueeze(0).unsqueeze(0)
        self.b1 = b1
        self.b2 = b2

    def prepare_linear_params(self, dtype):
        # Prepare value projection parameters (w1, b1)
        self.value_weights, self.value_bias = prepare_linear_params(self.device, self.w1, self.b1, dtype)

        # Prepare gate projection parameters (w2, b2)
        self.gate_weights, self.gate_bias = prepare_linear_params(self.device, self.w2, self.b2, dtype)


class UNetWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        # For main UNet, module_path is empty
        prefix = f"{self.module_path}." if self.module_path else ""

        # Conv input weights
        self.conv_in_weight = state_dict[f"{prefix}conv_in.weight"]
        self.conv_in_bias = state_dict[f"{prefix}conv_in.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Conv output weights
        self.conv_out_weight = state_dict[f"{prefix}conv_out.weight"]
        self.conv_out_bias = state_dict[f"{prefix}conv_out.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Norm out weights
        self.conv_norm_out_weight = state_dict[f"{prefix}conv_norm_out.weight"]
        self.conv_norm_out_bias = state_dict[f"{prefix}conv_norm_out.bias"]


class DownSampleWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        self.conv_weight = state_dict[f"{self.module_path}.conv.weight"]
        self.conv_bias = state_dict[f"{self.module_path}.conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)


class UpSampleWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        self.conv_weight = state_dict[f"{self.module_path}.conv.weight"]
        self.conv_bias = state_dict[f"{self.module_path}.conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)


class Transformer2DModelWeightLoader:
    def __init__(self, state_dict, module_path, device):
        self.module_path = module_path
        self.device = device
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        # Norm weights
        self.norm_weight = state_dict[f"{self.module_path}.norm.weight"]
        self.norm_bias = state_dict[f"{self.module_path}.norm.bias"]

        # Projection weights
        self.proj_in_weight = state_dict[f"{self.module_path}.proj_in.weight"].unsqueeze(0).unsqueeze(0)
        self.proj_in_bias = state_dict[f"{self.module_path}.proj_in.bias"]
        self.proj_out_weight = state_dict[f"{self.module_path}.proj_out.weight"].unsqueeze(0).unsqueeze(0)
        self.proj_out_bias = state_dict[f"{self.module_path}.proj_out.bias"]

    def prepare_linear_params(self, dtype):
        # Prepare input projection parameters
        self.weights_in, self.bias_in = prepare_linear_params(
            self.device, self.proj_in_weight, self.proj_in_bias, dtype
        )

        # Prepare output projection parameters
        self.weights_out, self.bias_out = prepare_linear_params(
            self.device, self.proj_out_weight, self.proj_out_bias, dtype
        )
