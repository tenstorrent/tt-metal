# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import argparse
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


# Helper function to recreate Mixtral state dictionary.
# Reads the consolidated weights provided in HuggingFace,
# separates the 8 experts and saves the updated dict into a new single file.
def repack_mixtral_weights(ckpt_dir, repack_dir):
    state_dict = {}
    # Set dummy_weights to True to avoid going through asserts that check for repack weights file (repack_weights.pt)
    model_args = TtModelArgs(dummy_weights=True)
    consolidated_weights_path = lambda i: str(ckpt_dir + f"/consolidated.{i:02d}.pt")

    for i in range(1 + (model_args.n_layers - 1) // 4):
        print(f"Loading consolidated.0{i}.pt...")
        state_dict_i = torch.load(consolidated_weights_path(i), map_location="cpu")
        state_dict.update(state_dict_i)

    repack_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }

    base_address = "feed_forward."
    for l in range(model_args.n_layers):
        print(f"Updating layer {l}...")
        pre = f"layers.{l}."
        repack_state_dict[pre + base_address + "gate.weight"] = repack_state_dict[pre + "block_sparse_moe.gate.weight"]
        del repack_state_dict[pre + "block_sparse_moe.gate.weight"]

        w1 = repack_state_dict[pre + "block_sparse_moe.w1"].contiguous().clone()
        w2 = repack_state_dict[pre + "block_sparse_moe.w2"].contiguous().clone()
        w3 = repack_state_dict[pre + "block_sparse_moe.w3"].contiguous().clone()
        ffn_dim = 14336
        for i in range(8):
            repack_state_dict[pre + base_address + f"experts.{i}.w1.weight"] = (
                w1[ffn_dim * i : ffn_dim * (i + 1), :].contiguous().clone()
            )
            repack_state_dict[pre + base_address + f"experts.{i}.w2.weight"] = (
                w2[ffn_dim * i : ffn_dim * (i + 1), :].T.clone().contiguous()
            )
            repack_state_dict[pre + base_address + f"experts.{i}.w3.weight"] = (
                w3[ffn_dim * i : ffn_dim * (i + 1), :].contiguous().clone()
            )
        repack_state_dict.pop(pre + "block_sparse_moe.w1")
        repack_state_dict.pop(pre + "block_sparse_moe.w2")
        repack_state_dict.pop(pre + "block_sparse_moe.w3")
    print(f"Saving repacked weights to {repack_dir}/repack_weights.pt")
    torch.save(repack_state_dict, repack_dir + "/repack_weights.pt")


if __name__ == "__main__":
    # Take in command line arguments
    parser = argparse.ArgumentParser(description="Repack mixtral8x7b weights")
    parser.add_argument("ckpt_dir", type=str, help="input checkpoint directory")
    parser.add_argument("out_dir", type=str, help="output repack directory")
    args = parser.parse_args()
    repack_mixtral_weights(args.ckpt_dir, args.out_dir)
