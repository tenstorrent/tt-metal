# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer

from models.demos.gpt_oss.reference.hf_utils import load_model_uninitialized, load_model_weights

local_model_path = "/localdev/avora/tt-metal/models/demos/gpt_oss/reference"
local_weights_path = "/localdev/avora/gpt-oss-20b"


def main():
    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    print("Tokenizer loaded successfully")

    # Load the model with uninitialized weights
    print("Loading uninitialized model from repo version")
    model = load_model_uninitialized()
    model.eval()
    print("Model loaded successfully")

    # Load the model weights
    print("Loading model weights")
    weights_dict = load_model_weights(local_weights_path)

    model.load_state_dict(weights_dict, strict=False)
    print("Model weights loaded successfully")

    breakpoint()
    messages = [
        {"role": "user", "content": "Hi!"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


if __name__ == "__main__":
    main()
