#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import msgpack
import msgpack_numpy  # required for serializing numpy arrays
from typing import Dict, Any, List
import functools
import argparse
import re
from collections import Counter
import ipdb
import json
import os
import sys

msgpack_numpy.patch()


@ipdb.iex
def tweak_and_dump_tokenizer(args):
    """
    Get the tokenizer JSON and modify the decoder part to remove the strip clause.
    This function loads the tokenizer from the HF model, modifies its configuration,
    and saves it to a JSON file.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    assert args.dump_tokenizer_path is not None

    # Get the tokenizer configuration as a dictionary
    tokenizer_config = tokenizer.backend_tokenizer.to_str(pretty=True)
    tokenizer_dict = json.loads(tokenizer_config)

    # Remove the strip decoder from the decoders
    if "decoder" in tokenizer_dict and "decoders" in tokenizer_dict["decoder"]:
        decoders = tokenizer_dict["decoder"]["decoders"]
        for i, decoder in enumerate(decoders):
            if decoder == {"type": "Strip", "content": " ", "start": 1, "stop": 0}:
                decoders.pop(i)
                break

    # a few tweaks to match with the tokenizer.json uploaded to HF to make the
    # tokenizer compatible with our setup. for some reason, the transformers
    # library doesn't do this stuff automatically when it exports the tokenizer.
    if "model" in tokenizer_dict and "ignore_merges" in tokenizer_dict["model"]:
        del tokenizer_dict["model"]["ignore_merges"]

    # Convert merges to joined format
    if "model" in tokenizer_dict and "merges" in tokenizer_dict["model"]:
        merges = tokenizer_dict["model"]["merges"]
        new_merges = []
        for merge in merges:
            new_merges.append(" ".join(merge))
        tokenizer_dict["model"]["merges"] = new_merges

    output_dir = os.path.dirname(args.dump_tokenizer_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer_path = args.dump_tokenizer_path
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_dict, f)

    print(f"Modified tokenizer saved to {tokenizer_path}")


@ipdb.iex
def dump_model(args):
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model)

    def fix_hf_state_dict_for_rope(head_dim=64):
        def unpermute_proj(w: np.ndarray, n_heads: int = 32) -> np.ndarray:
            """
            Permutes the rows of the weight matrix `w` based on head structure using NumPy.
            Assuming `w` has `R` rows and `n_heads`, it divides the rows into `n_heads`
            chunks of size `D = R // n_heads`. Within each chunk, it interleaves the
            first `D // 2` rows with the second `D // 2` rows.

            Example: If a chunk has rows [r0, r1, r2, r3] (D=4), the output order for
            that chunk will be [r0, r2, r1, r3].

            This is intended to convert Hugging Face projection weights (Q/K) to a
            layout compatible with Meta-style interleaved RoPE application.

            Args:
                w: Input weight matrix (NumPy array), shape [R, C]. Typically [dim, dim].
                n_heads: Number of heads corresponding to the rows R.

            Returns:
                np.ndarray: The permuted weight matrix with the same shape as `w`.
            """
            if type(w) is not np.ndarray:
                w = w.cpu().numpy()
            R, C = w.shape  # R = dim (e.g., hidden_size)
            D = R // n_heads  # D = head_dim
            assert R % n_heads == 0, "Number of rows R must be divisible by n_heads."
            assert D % 2 == 0, "Rows per head (D) must be even."

            # Reshape the row dimension R into (n_heads, 2, D // 2).
            # The dimension of size 2 separates the first half of rows (index 0)
            # and the second half of rows (index 1) within each D-sized chunk.
            # Shape becomes: [n_heads, 2, D // 2, C]
            w_reshaped = w.reshape(n_heads, 2, D // 2, C)

            # Transpose the 'half' dimension (axis 1) and the 'index within half'
            # dimension (axis 2).
            # Shape becomes: [n_heads, D // 2, 2, C]
            # For head `h` and index `i` (0..D//2-1):
            # - w_transposed[h, i, 0, :] contains original row `h*D + i` (from 1st half)
            # - w_transposed[h, i, 1, :] contains original row `h*D + D//2 + i` (from 2nd half)
            w_transposed = w_reshaped.transpose(0, 2, 1, 3)  # Swap axes 1 and 2

            # Reshape back to [R, C] by flattening the first three dimensions (h, i, k).
            # The new row index `r_new = h*D + i*2 + k`.
            # - For k=0 (even indices): `r_new = h*D + 2*i`, gets data from original row `h*D + i`.
            # - For k=1 (odd indices): `r_new = h*D + 2*i + 1`, gets data from original row `h*D + D//2 + i`.
            # This creates the interleaved order: [row 0, row D/2, row 1, row D/2+1, ...] for each chunk.
            w_interleaved = w_transposed.reshape(R, C)

            return w_interleaved

        def convert_hf_qkv_to_meta_format(loaded_weights, head_dim):
            """Convert HuggingFace QKV weights to Meta format for RoPE compatibility."""
            converted_dict = {}
            for key, tensor in loaded_weights.items():
                if any(f"{pfx}.weight" in key or f"{pfx}.bias" in key for pfx in ["q_proj", "k_proj"]):
                    print(f"Permuting {key}")
                    n_heads = tensor.shape[0] // head_dim
                    permuted = unpermute_proj(tensor, n_heads)
                    assert permuted.shape == tensor.shape, "permuted shape doesn't match original projection shape!"
                    converted_dict[key] = permuted
                else:
                    converted_dict[key] = tensor
            return converted_dict

        return convert_hf_qkv_to_meta_format(hf_model.state_dict(), head_dim)

    if not args.meta_style:
        print("Using interleaved state dict")
        head_dim = hf_model.config.hidden_size // hf_model.config.num_attention_heads
        hf_state_dict = fix_hf_state_dict_for_rope(head_dim)
    else:
        print("Using non-interleaved state dict")
        hf_state_dict = hf_model.state_dict()

    def tt_to_hf_key(key: str) -> str | List[str] | None:
        non_block_map = {
            "llama/llama/tok_emb/weight/value/data": "model.embed_tokens.weight",
            "llama/llama/ln_fc/gamma/value/data": "model.norm.weight",
            "llama/llama/fc/weight/value/data": "lm_head.weight",
        }
        if key in non_block_map:
            return non_block_map[key]

        # block
        tt_prefix_pattern = r"llama/llama/llama_block_(\d+)/(.*)"
        tt_prefix_match = re.match(tt_prefix_pattern, key)
        if tt_prefix_match:
            layer_num = tt_prefix_match.group(1)
            hf_prefix = f"model.layers.{layer_num}."
        else:
            return None

        rest = tt_prefix_match.group(2)

        block_key_map = {
            "attention_norm/gamma/value/data": "input_layernorm.weight",
            "mlp_norm/gamma/value/data": "post_attention_layernorm.weight",
            "attention/q_linear/weight/value/data": "self_attn.q_proj.weight",
            "attention/kv_linear/weight/value/data": ["self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            "attention/out_linear/weight/value/data": "self_attn.o_proj.weight",
            "mlp/w1/weight/value/data": "mlp.gate_proj.weight",
            "mlp/w2/weight/value/data": "mlp.down_proj.weight",
            "mlp/w3/weight/value/data": "mlp.up_proj.weight",
        }

        if rest in block_key_map:
            if isinstance(block_key_map[rest], list):
                prefixed = [hf_prefix + k for k in block_key_map[rest]]
                return prefixed
            else:
                return hf_prefix + block_key_map[rest]
        else:
            return None

    def update_single_key(existing_state: Dict[str, Any], key: str, new_value: np.ndarray) -> bool:
        """
        Updates a key in existing_state with new_value.

        All stored values are linear (1D) arrays. For special keys (token embedding,
        fc head weight, etc.) we allow padding:

        - For embedding keys: new_value is a 2D array of shape (n_tokens, emb_dim).
          The stored value is a flat array representing a matrix with shape
          (expected_n_tokens, emb_dim). If new_value has fewer tokens, pad with zeros.

        - For all other keys, the flattened new_value must match the stored volume.
        """
        # In the file, the weight is stored as a linear array in the second element.
        old_value = existing_state[key][1]
        old_array = np.array(old_value) if isinstance(old_value, list) else old_value
        if not isinstance(new_value, np.ndarray):
            new_value = new_value.cpu().numpy().astype(np.float32)

        # Define keys that require special treatment (may need padding)
        embedding_keys = {
            "llama/llama/tok_emb/weight/value/data",
            "llama/llama/fc/weight/value/data",
        }

        if key in embedding_keys:
            (n_tokens, emb_dim) = new_value.shape
            # new_value is expected to be 2D: (n_tokens, emb_dim)
            if new_value.ndim != 2:
                raise ValueError(f"Expected new_value for key '{key}' to be 2D, got shape {new_value.shape}")
            assert n_tokens == hf_model.config.vocab_size
            # Determine expected number of tokens from the stored flat array.
            if old_array.size % emb_dim != 0:
                raise ValueError(
                    f"Stored size for key '{key}' ({old_array.size}) is not divisible by emb_dim ({emb_dim})."
                )
            expected_tokens = old_array.size // emb_dim
            if n_tokens < expected_tokens:
                pad_tokens = expected_tokens - n_tokens
                pad = np.zeros((pad_tokens, emb_dim), dtype=new_value.dtype)
                new_value = np.concatenate([new_value, pad], axis=0)
            elif n_tokens > expected_tokens:
                raise ValueError(
                    f"New value for key '{key}' has more tokens than expected: new {new_value.shape[0]} vs expected {expected_tokens}"
                )

            new_value_flat = new_value.flatten()
            existing_state[key][1] = new_value_flat.tolist()
            return True

        # For all other keys, require an exact match in total number of elements.
        new_value_flat = new_value.flatten()
        if old_array.size != new_value_flat.size:
            raise ValueError(
                f"Mismatch in size for key '{key}': existing volume {old_array.size} != new volume {new_value_flat.size}"
            )
        existing_state[key][1] = new_value_flat.tolist()
        return True

    def import_hf_weights(init_state: Dict[str, Any], hf_model_config, hf_state_dict):
        init_shapes = {key: np.array(value).shape for (key, (_, value)) in init_state.items()}
        init_keys = set(init_state.keys())

        def flat_tup(tup):
            def product(s):
                return functools.reduce(lambda x, y: x * y, s)

            return (product(tup),)

        for key in init_keys:
            assert set(init_state.keys()) == init_keys
            hf_key = tt_to_hf_key(key)
            if hf_key is None:
                continue
            if isinstance(hf_key, list):
                hf_values = [hf_state_dict[k] for k in hf_key]
                head_dim = hf_model_config.head_dim
                emb_dim = hf_model_config.hidden_size
                if not all(p.shape == (hf_model_config.num_key_value_heads * head_dim, emb_dim) for p in hf_values):
                    raise ValueError(f"Mismatch in shape for {key}: {hf_values[0].shape} vs. {init_shapes[key]}(tt)")
                # M -> N linear repesented by n x m matrix, hence concate on dim -2 == 0.
                hf_value = np.concatenate(hf_values, axis=-2)
            else:
                hf_value = hf_state_dict[hf_key]
            hf_shape = flat_tup(hf_value.shape)
            assert hf_shape == init_shapes[key], f"shape mismatch for {key}: {hf_shape}(hf) vs. {init_shapes[key]}(tt)"

            res = update_single_key(init_state, key, hf_value)
            assert res, f"failed to update key: {key}"

    print(f"Loading initial state from {args.input_path}")
    with open(args.input_path, "rb") as f:
        tt_state = msgpack.unpack(f, raw=False)

    init_keys = Counter(tt_state.keys())
    print("Importing hf weights")
    import_hf_weights(tt_state, hf_model.config, hf_state_dict)
    new_keys = Counter(tt_state.keys())
    assert new_keys == init_keys, f"Keys mismatch after conversion: {new_keys} vs {init_keys}"
    print("Saving converted init_state")
    with open(args.output_path, "wb") as f:
        msgpack.pack(tt_state, f)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="data/tinyllama_init.msgpack",
        help="Path to the dumped original weights file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="data/tinyllama_exported.msgpack",
        help="Path to the output weights file",
    )
    parser.add_argument("--hf_model", type=str, default="TinyLlama/TinyLlama_v1.1", help="name of the HF model")
    parser.add_argument(
        "--meta_style",
        action="store_true",
        help="The model is in the Meta style (QKV projections have dims interleaved.)",
    )
    parser.add_argument("-t", "--dump_tokenizer_path", type=str, default=None, help="Path to the output tokenizer file")
    parser.set_defaults(meta_style=False)

    args = parser.parse_args()

    if args.dump_tokenizer_path is not None:
        tweak_and_dump_tokenizer(args)
        sys.exit(0)

    if args.input_path is not None and args.output_path is not None:
        dump_model(args)
    elif not args.dump_tokenizer_path:
        print("Nothing to do. Please either specify --dump_tokenizer_path or both of --input_path and --output_path.")
        sys.exit(1)
    elif any([args.input_path, args.output_path]) and not all([args.input_path, args.output_path]):
        print("Note: both of input_path and output_path are required to export the weights.")
        sys.exit(1)
