# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import bz2
import os
import argparse
import time
from models.demos.llama3.tt.llama_common import HostEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from loguru import logger


def generate_reference_outputs(total_length, output_file):
    # Load the model arguments
    model_args = TtModelArgs(mesh_device=None)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Load the model state dict
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Initialize the reference model
    reference_model = Transformer(model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()  # Set to evaluation mode

    # Initialize HostEmbedding
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # Load the book text and encode tokens
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    prompt_file = os.path.join(current_file_dir, "tale-of-two-cities.txt.bz2")

    with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        text = f.read()

    # Encode text to tokens
    encoded_tokens = tokenizer.encode(text, bos=True, eos=False)[:total_length]
    encoded_tokens_tensor = torch.tensor(encoded_tokens).unsqueeze(0)  # Shape [1, seq_len]

    all_top5_tokens = []
    segment_top1_correct = []
    segment_top5_correct = []
    segment_accuracies = []
    segment_summaries = []

    print(f"{'ETA':<8}{'Progress':<15}{'Correct':<8}{'Actual':<15}{'Top 5 Predictions':<75}")
    print("-" * 121)

    start_time = None
    with torch.no_grad():
        for i in range(total_length):
            pt_decode_input = embd(encoded_tokens_tensor[:, i]).view(1, 1, -1)

            ref_output = reference_model(pt_decode_input, start_pos=i)

            if i < len(encoded_tokens) - 1:
                next_token = encoded_tokens[i + 1]
            else:
                next_token = torch.argmax(ref_output, dim=-1).item()

            # Compute top-5 predictions for the current token
            probs = torch.softmax(ref_output, dim=-1)
            top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)
            top5_indices = top5_indices.squeeze()
            all_top5_tokens.append(top5_indices)

            # Record top1 and top5 correctness
            segment_top1_correct.append(top5_indices[0] == next_token)
            segment_top5_correct.append(next_token in top5_indices)

            sanitize = lambda x: x.replace("\n", "").replace("\r", "").replace("\x0c", "")
            actual_token = tokenizer.decode([next_token])
            top5_tokens = [tokenizer.decode([t.item()]) for t in top5_indices]
            correct = "x" if segment_top1_correct[-1] else ("-" if segment_top5_correct[-1] else " ")
            top5_str = " ".join(f"{t:<14}" for t in top5_tokens)
            actual_token = sanitize(actual_token)
            top5_str = sanitize(top5_str)

            # Calculate ETA and progress
            if start_time:
                elapsed_time = time.time() - start_time
                tokens_per_second = i / elapsed_time
                remaining_tokens = total_length - 1 - i
                eta_seconds = remaining_tokens / tokens_per_second
                eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = ""
                start_time = time.time()

            progress_str = f"{i+1}/{total_length}"

            print(f"{eta_str:<8}{progress_str:<15}{correct:<8}{actual_token:<15}{top5_str}")

            # Calculate and store segment accuracies every 100 tokens or at the end
            if (i + 1) % 100 == 0 or i == total_length - 1:
                top1_acc = sum(segment_top1_correct) / len(segment_top1_correct) * 100
                top5_acc = sum(segment_top5_correct) / len(segment_top5_correct) * 100
                segment_accuracies.append((top1_acc, top5_acc))
                segment_summaries.append(
                    f"Tokens {i-len(segment_top1_correct)+1}-{i+1}: Top-1 Accuracy: {top1_acc:.0f} %, Top-5 Accuracy: {top5_acc:.0f} %"
                )
                segment_top1_correct = []
                segment_top5_correct = []

    # Convert list to tensor
    all_top5_tokens = torch.stack(all_top5_tokens)

    # Save the data
    data = {
        "top5_tokens": all_top5_tokens,
        "reference_tokens": encoded_tokens_tensor,
    }

    torch.save(data, output_file)
    logger.info(f"Saved reference outputs to {output_file}")

    # Print all segment accuracy summaries as a table
    print("\nSegment Accuracy Summaries:")
    print(f"{'Tokens':<15}{'Top-1 Accuracy':<20}{'Top-5 Accuracy':<20}")
    print("-" * 55)
    for i, (top1_acc, top5_acc) in enumerate(segment_accuracies):
        start_token = i * 100 + 1
        end_token = min((i + 1) * 100, total_length)
        print(f"{f'{start_token}-{end_token}':<15}{f'{top1_acc:.2f}%':<20}{f'{top5_acc:.2f}%':<20}")

    # Calculate overall accuracy
    overall_top1_acc = sum(acc[0] for acc in segment_accuracies) / len(segment_accuracies)
    overall_top5_acc = sum(acc[1] for acc in segment_accuracies) / len(segment_accuracies)
    print("-" * 55)
    print(f"{'Overall':<15}{f'{overall_top1_acc:.2f}%':<20}{f'{overall_top5_acc:.2f}%':<20}")


# New main function with argparse
def main():
    parser = argparse.ArgumentParser(description="Generate reference outputs for LLaMA accuracy testing.")
    parser.add_argument("--total_length", type=int, default=1500, help="Total length of tokens to process")
    parser.add_argument(
        "--output_file", type=str, default="reference_outputs.pt", help="Output file path for reference data"
    )
    args = parser.parse_args()

    generate_reference_outputs(total_length=args.total_length, output_file=args.output_file)


if __name__ == "__main__":
    main()
