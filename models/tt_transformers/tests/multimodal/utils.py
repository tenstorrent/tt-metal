import os

import torch.nn.functional as F


def load_partial_weights(auto_model, weights_path, layer_prefix):
    partial_state_dict = {}
    model = auto_model.from_pretrained(weights_path, torch_dtype="auto", local_files_only=os.getenv("CI") == "true")
    weights = model.state_dict()
    keys = weights.keys()
    for key in keys:
        if layer_prefix in key:
            # Caution it may cause potential failures. In future versions and different formats the below prefix may change
            key_name = key[len(layer_prefix) :]
            partial_state_dict.update({key_name: weights[key]})
    return partial_state_dict


def expand_num_tokens_to_mult8(tensor):
    num_padding_patches = (8 - (tensor.shape[-2] % 8)) % 8
    # Compute padding tuple for pad function
    padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
    # Pad the tensor
    tensor = F.pad(tensor, padding, mode="constant", value=0)
    slice_index = -num_padding_patches if num_padding_patches > 0 else 0
    return tensor, slice_index


def contract_num_tokens_from_mult8(tensor, slice_index):
    if slice_index == 0:
        return tensor
    return tensor[:, :, :slice_index, :]
