# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.gpt_oss.reference.hf_utils import get_state_dict, load_model_uninitialized, load_tokenizer

local_model_path = "models/demos/gpt_oss/reference"
local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")
torch_state_dict_path = os.path.join(local_weights_path, "torch_state_dict.pt")


def main():
    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = load_tokenizer(local_weights_path)
    print("Tokenizer loaded successfully")

    with torch.no_grad():
        # Load the model with uninitialized weights
        print("Loading uninitialized model from repo version")
        model = load_model_uninitialized(local_model_path)
        model.eval()
        print("Model loaded successfully")

        # Load the model weights
        print("Loading model weights")
        weights_dict = get_state_dict(local_weights_path, dtype=torch.float32)  # prefix="model.layers.0.self_attn."

        model.load_state_dict(weights_dict, strict=True)
        print("Model weights loaded successfully")

        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        print("Running the model")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Increased to allow for reasoning and final response
        )
        print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


if __name__ == "__main__":
    main()
