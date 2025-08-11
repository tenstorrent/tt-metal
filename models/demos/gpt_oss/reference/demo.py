# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.gpt_oss.reference.hf_utils import (
    convert_bf16_to_fp32,
    load_model_uninitialized,
    load_model_weights,
    load_tokenizer,
)

local_model_path = "models/demos/gpt_oss/reference"
local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/localdev/avora/gpt-oss-20b-BF16")


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
        weights_dict = load_model_weights(local_weights_path)
        weights_dict = convert_bf16_to_fp32(weights_dict)

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
