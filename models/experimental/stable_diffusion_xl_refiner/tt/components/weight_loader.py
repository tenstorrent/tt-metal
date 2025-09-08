class ResNetWeightLoader:
    def __init__(self, state_dict, module_path, use_conv_shortcut=False):
        self.module_path = module_path
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


# Not used for now, might move param prepare functions into loaders
class TransformerBlockWeightLoader:
    def __init__(self, state_dict, module_path):
        self.module_path = module_path
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        self.norm_weights_1 = state_dict[f"{self.module_path}.norm1.weight"]
        self.norm_bias_1 = state_dict[f"{self.module_path}.norm1.bias"]

        self.norm_weights_2 = state_dict[f"{self.module_path}.norm2.weight"]
        self.norm_bias_2 = state_dict[f"{self.module_path}.norm2.bias"]

        self.norm_weights_3 = state_dict[f"{self.module_path}.norm3.weight"]
        self.norm_bias_3 = state_dict[f"{self.module_path}.norm3.bias"]
