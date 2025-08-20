# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

from models.demos.deepseek_v3.reference.deepseek_reference_outputs_gen import *


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A script to trace the IO of the deepseek model.")
    parser.add_argument("local_model_path", type=str, help="Path to the local model directory.")
    parser.add_argument("prompt", type=str, help="Prompt to generate outputs for.")
    return parser


def main():
    # Parse the sysargs
    parser = create_parser()
    args = parser.parse_args()

    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = load_tokenizer(args.local_model_path)
    print("Tokenizer loaded successfully")

    with torch.no_grad():
        # Load the model with uninitialized weights
        print("Loading uninitialized model")
        model = load_model_uninitialized(args.local_model_path)
        model.eval()
        print("Model loaded successfully")

        # Load the model weights
        print("Loading model weights")
        weights_dict = load_model_weights(args.local_model_path)
        add_dynamic_weight_loading_hooks(model, weights_dict)
        print("Model weights loaded successfully")

        # Run the model
        model_inputs = tokenizer(args.prompt, return_tensors="pt")
        print("Running the model")
        outputs = model.generate(**model_inputs, max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)


if __name__ == "__main__":
    main()
