# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from pathlib import Path

import regex as re
import torch
from loguru import logger
from tqdm import tqdm

from models.tt_transformers.tt.load_checkpoints import load_sharded_checkpoints

# get the absolute path to this file
_file_abs_path = os.path.abspath(__file__)

# [INFO]: because HF does not export convert_mllama_weight_to_hf.py (the line below fails), we need to copy the functions here
# from transformers.models.mllama.convert_mllama_weight_to_hf import is_param_different_across_shards, convert_old_keys_to_new_keys, get_concat_dim, cross_attention_layers_shift, self_attention_layers_shift

# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"text_model.layers.(\d+).attention.wqkv.weight": r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"text_model.norm.weight":                                                                  r"language_model.model.norm.weight",
    r"text_model.output.weight":                                                                r"language_model.lm_head.weight",
    r"text_model.tok_embeddings":                                                               r"language_model.model.embed_tokens",
    r"text_model.learnable_embedding":                                                          r"language_model.model.learnable_embedding",
    r"text_model.rope.freqs":                                                                   None, # meaning we skip it and don't want it
    # For every cross attention layer, the layer needs to be updated
    r"text_model.cross_attention_layers.(\d+).gate_attn":                                       r"language_model.model.layers.\1.cross_attn_attn_gate",
    r"text_model.cross_attention_layers.(\d+).gate_ffwd":                                       r"language_model.model.layers.\1.cross_attn_mlp_gate",
    # special key, wqkv needs to be split afterwards
    r"text_model.cross_attention_layers.(\d+).attention.w(q|k|v|o)":                            r"language_model.model.layers.\1.cross_attn.\2_proj",
    r"text_model.cross_attention_layers.(\d+).attention.(q|k)_norm":                            r"language_model.model.layers.\1.cross_attn.\2_norm",
    r"text_model.cross_attention_layers.(\d+).attention_norm.weight":                           r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).attention.wk.layer_norm_weight":                  r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.w1.weight":                          r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.w2.weight":                          r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.w3.weight":                          r"language_model.model.layers.\1.mlp.up_proj.weight",
    r"text_model.cross_attention_layers.(\d+).ffn_norm.weight":                                 r"language_model.model.layers.\1.post_attention_layernorm.weight",
    # self attention layers
    r"text_model.layers.(\d+).attention.w(q|k|v|o).weight":                                     r"language_model.model.layers.\1.self_attn.\2_proj.weight",
    r"text_model.layers.(\d+).attention_norm.weight":                                           r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.layers.(\d+).feed_forward.w1.":                                                r"language_model.model.layers.\1.mlp.gate_proj.",
    r"text_model.layers.(\d+).feed_forward.w2.":                                                r"language_model.model.layers.\1.mlp.down_proj.",
    r"text_model.layers.(\d+).feed_forward.w3.":                                                r"language_model.model.layers.\1.mlp.up_proj.",
    r"text_model.layers.(\d+).ffn_norm.weight":                                                 r"language_model.model.layers.\1.post_attention_layernorm.weight",
    # Vision encoder mapping
    r"vision_model.vision_encoder.conv1._linear":                                               r"vision_model.patch_embedding",
    r'vision_model.vision_projection.':                                                         r"multi_modal_projector.",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wq":    r"vision_model.\1.layers.\2.self_attn.q_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wk":    r"vision_model.\1.layers.\2.self_attn.k_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wv":    r"vision_model.\1.layers.\2.self_attn.v_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wo":    r"vision_model.\1.layers.\2.self_attn.o_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_fc":   r"vision_model.\1.layers.\2.mlp.fc1",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_proj": r"vision_model.\1.layers.\2.mlp.fc2",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_1":       r"vision_model.\1.layers.\2.input_layernorm",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_2":       r"vision_model.\1.layers.\2.post_attention_layernorm",
    r"vision_model.vision_encoder.global_transformer.resblocks.(\d+).(gate_ffn|gate_attn)":     r"vision_model.global_transformer.layers.\1.\2",
    r'vision_model.vision_encoder.ln_(pre|post).(weight|bias)':                                 r'vision_model.vision_encoder.layernorm_\1.\2',
    r'vision_model.vision_encoder.positional_embedding\b':                                      r'vision_model.gated_positional_embedding.embedding',
    r'vision_model.vision_encoder.gated_positional_embedding\b':                                r'vision_model.gated_positional_embedding.tile_embedding.weight',
    r'vision_model.vision_encoder.gated_positional_embedding_gate':                             r'vision_model.gated_positional_embedding.gate',
    r"vision_model.vision_encoder.pre_tile_pos_embed.embedding":                                r"vision_model.pre_tile_positional_embedding.embedding.weight",
    r"vision_model.vision_encoder.post_tile_pos_embed.embedding":                               r"vision_model.post_tile_positional_embedding.embedding.weight",
    r"vision_model.vision_encoder.pre_tile_pos_embed.gate":                                     r"vision_model.pre_tile_positional_embedding.gate",
    r"vision_model.vision_encoder.post_tile_pos_embed.gate":                                    r"vision_model.post_tile_positional_embedding.gate",
    r"vision_model.vision_encoder.(?=\w)":                                                      r"vision_model.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def is_param_different_across_shards(key):
    """
    Return `True` if the parameter is different across checkpoint shards
    and needs to be concatenated.
    """
    patterns = [r"vision_model.patch_embedding.weight",r"vision_model.(transformer|global_transformer).layers.(\d+).self_attn.(q|k|v|o)_proj.weight",r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc1.(weight|bias)",r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc2.weight",  r"multi_modal_projector.(weight|bias)",r"language_model.model.embed_tokens.weight",r"language_model.lm_head.weight",r"language_model.model.layers.(\d+).self_attn.(q|k|v|o)_proj.weight",r"language_model.model.layers.(\d+).cross_attn.(q|k|v|o)_proj.weight",r"language_model.model.layers.(\d+).mlp.(up|down|gate)_proj.weight",r"language_model.model.learnable_embedding.weight"]  # fmt: skip
    return any(re.search(pattern, key) for pattern in patterns)


def get_concat_dim(key):
    """
    Return the dimension to concatenate the weights on.
    """
    concat_dim_1 = [
        r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc2.weight",
        r"vision_model.(transformer|global_transformer).layers.(\d+).self_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).cross_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).self_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).mlp.down_proj.weight",
    ]
    if any(re.search(pattern, key) for pattern in concat_dim_1):
        return 1
    return 0


def hf_mllama_save_tensor_states(loaded: list, model_name: str, params: dict, tensor_states_path: str) -> None:
    # [INFO] copied just enough code from transformers.models.mllama.convert_mllama_weight_to_hf.write_model to get the tensor states
    text_num_layers = params["n_layers"]
    cross_attention_num_layers = params["vision_num_cross_attention_layers"]

    cross_attention_frequency = math.ceil(text_num_layers / cross_attention_num_layers)
    text_num_total_layers = text_num_layers + cross_attention_num_layers
    cross_attention_layers_shift = list(
        range(cross_attention_frequency - 1, text_num_total_layers, cross_attention_frequency + 1)
    )
    self_attention_layers_shift = [k for k in range(text_num_total_layers) if k not in cross_attention_layers_shift]

    all_keys = list(loaded[0].keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    tensor_states = {}
    for key in all_keys:
        new_key = new_keys[key]

        # In the original model, self-attention layers and cross-attention layers are different lists of layers.
        # In the converted model, they are merged into one list with corresponding index shift to preserve the order.
        if ("cross_attention" in key or "text_model.layers" in key) and "language_model" in new_key:
            shift = cross_attention_layers_shift if "cross_attention" in key else self_attention_layers_shift
            new_key = re.sub(r"layers.(\d+).", lambda _match: f"layers.{shift[int(_match.groups()[0])]}.", new_key)

        concat_dim = get_concat_dim(new_key)

        current_parameter = [chunk.pop(key).contiguous().clone() for chunk in loaded]
        is_sharded = is_param_different_across_shards(new_key)
        if not is_sharded:
            full_parameter = current_parameter[0]
        else:
            full_parameter = torch.cat(current_parameter, dim=concat_dim)

        tensor_states.update(
            {
                key: {
                    # NOTE: uncomment key-value pairs for debugging
                    # "sharded": is_sharded,
                    # "concat_dim": concat_dim,
                    "full_shape": full_parameter.shape,
                    # "shard_shapes": [p.shape for p in current_parameter],
                }
            }
        )

    # save tensor_states to json file
    with open(tensor_states_path, "w") as f:
        json.dump(tensor_states, f, indent=4)


# [INFO]: start of copied function from known good implementation of load_sharded_checkpoints for the following models:
# "Llama3.2-1B"
# "Llama3.2-3B"
# "Llama3.1-8B"
# "Llama3.2-11B"
# "Llama3.1-70B"
def load_sharded_checkpoints_orig(checkpoints, n_layers):
    checkpoint = {}
    logger.info(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for (
            key,
            value,
        ) in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            if key in checkpoint:
                checkpoint[key] += [value]
            else:
                checkpoint[key] = [value]
        del loaded_ckpt

    # concat checkpoint values
    for key, value in checkpoint.items():
        if len(value) == 1 or "norm" in key:
            checkpoint[key] = value[0]
        else:
            if key == "tok_embeddings.weight" or key == "output.weight":
                assert value[0].shape[1] == 8192  # FIXME: do we need this hardcoded shape?
                # Concatenate along dimension 0 for llama3 token embeddings weight and lm head
                checkpoint[key] = torch.cat(value, dim=0)
            else:
                # cat_dim is index of the smallest dimension in value[0].shape
                cat_dim = torch.argmin(torch.tensor(value[0].shape))
                checkpoint[key] = torch.cat(value, dim=cat_dim)

    return checkpoint


# [INFO]: end of copied function from known good implementation of load_sharded_checkpoints


def write_tensor_states(loaded: dict, tensor_states_path: str) -> None:
    tensor_states = {}
    for key, value in loaded.items():
        tensor_states.update(
            {
                key: {
                    "full_shape": list(value.shape),
                }
            }
        )

    # save tensor_states to json file
    with open(tensor_states_path, "w") as f:
        json.dump(tensor_states, f, indent=4)


# [INFO] Focus on testing 70B and 90B models because they are the only ones that require shared checkpoints
def test_load_checkpoints():
    # make ModelArgs object with empty mesh_device for its ability to recognize the model name
    input_base_path = os.getenv("LLAMA_DIR")
    assert input_base_path, "LLAMA_DIR must be set to indicate the path to the model checkpoints"
    logger.info(f"Checkpoint directory: {input_base_path}")
    # [INFO] we can hardcode this check because we only test 70B and 90B models atm
    is_70b = "Llama" in input_base_path and "70B" in input_base_path
    is_90b = "Llama" in input_base_path and "90B" in input_base_path and "Vision" in input_base_path
    assert (
        is_70b or is_90b
    ), "this test is only needed for models with sharded checkpoints (only 70B and 90B models atm)"
    model_name = "Llama3.1-70B-Instruct" if is_70b else "Llama3.2-90B-Vision-Instruct"
    model_names_with_hf_golden_func = ("Llama3.2-90B-Vision-Instruct",)
    model_names_with_orig_golden_func = ("Llama3.1-70B-Instruct",)

    with open(os.path.join(input_base_path, "params.json"), "r") as f:
        params = json.load(f)

    checkpoints = sorted(Path(input_base_path).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {input_base_path}"
    print(f"\nFetching all parameters from the checkpoint at {input_base_path}...")

    tensor_states_path = Path(_file_abs_path).parent.parent / "model_params" / model_name / "tensor_states.json"
    if not os.path.exists(tensor_states_path):
        print(f"tensor states not found at {tensor_states_path}. Generating...")
        num_shards = len(checkpoints)
        if model_name in model_names_with_hf_golden_func:
            loaded = [
                torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu", mmap=True)
                for i in range(num_shards)
            ]
            hf_mllama_save_tensor_states(loaded, model_name, params, tensor_states_path)
        else:
            assert (
                model_name in model_names_with_orig_golden_func
            ), f"model {model_name} must have a golden function implemented"
            loaded = load_sharded_checkpoints_orig(checkpoints, n_layers=None)
            write_tensor_states(loaded, tensor_states_path)

    assert os.path.exists(tensor_states_path), f"tensor states must now be available at {tensor_states_path}"
    print(f"loading tensor states from {tensor_states_path}")
    tensor_states = json.load(open(tensor_states_path, "r"))

    # use tt-transformers' load_sharded_checkpoints function to load the model
    state_dict = load_sharded_checkpoints(checkpoints, n_layers=None)  # n_layers=None will load all layers

    assert len(state_dict) == len(
        tensor_states
    ), f"number of keys in state_dict ({len(state_dict)}) does not match number of keys in tensor_states ({len(tensor_states)})"

    for key, value in state_dict.items():
        assert key in tensor_states, f"key {key} not found in tensor_states"
        assert (
            list(value.shape) == tensor_states[key]["full_shape"]
        ), f"shape mismatch for key {key}: {value.shape} != {tensor_states[key]['full_shape']}"
