# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import bz2
import os
import argparse
import time
from models.demos.llama3_subdevices.tt.llama_common import HostEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_reference_outputs(total_length, output_file, hf_model_name=None):
    if hf_model_name:
        # HuggingFace path
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.float32)
        model.eval()
    else:
        # Original path - load reference model
        model_args = TtModelArgs(mesh_device=None)
        model_args.max_seq_len = total_length
        tokenizer = Tokenizer(model_args.tokenizer_path)

        state_dict = model_args.load_state_dict()
        state_dict_prefix = model_args.get_state_dict_prefix("", None)
        reference_state_dict = {
            k[len(state_dict_prefix) :]: v
            for k, v in state_dict.items()
            if (
                any([f"{state_dict_prefix}layers.{i}." in k for i in range(model_args.n_layers)])
                or any(
                    [
                        f"{state_dict_prefix}{name}" in k
                        for name in ["tok_embeddings.weight", "norm.weight", "output.weight"]
                    ]
                )
            )
        }
        model = Transformer(model_args)
        model.load_state_dict(reference_state_dict)
        model.eval()

        # Initialize HostEmbedding
        embd = HostEmbedding(model_args)
        embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # Load the book text and encode tokens
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    prompt_file = os.path.join(current_file_dir, "tale-of-two-cities.txt.bz2")

    with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        text = f.read()

    # Modify token encoding based on model type
    if hf_model_name:
        encoded_tokens = tokenizer.encode(text, add_special_tokens=True)[:total_length]
    else:
        encoded_tokens = tokenizer.encode(text, bos=True, eos=False)[:total_length]

    encoded_tokens_tensor = torch.tensor(encoded_tokens).unsqueeze(0)  # Shape [1, seq_len]

    print(f"{'Progress':<15}{'Correct':<8}{'Actual':<15}{'Top 5 Predictions':<75}")
    print("-" * 113)

    # Initialize lists to store results
    all_top1_correct = []
    all_top5_correct = []
    all_top5_tokens = []
    segment_accuracies = []
    chunk_size = 1024

    with torch.no_grad():
        for chunk_start in range(0, total_length - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_length)
            # Get input and target chunks, ensuring they have matching lengths
            chunk_tokens = encoded_tokens_tensor[:, chunk_start:chunk_end]
            chunk_next_tokens = encoded_tokens[chunk_start + 1 : chunk_end + 1]
            actual_chunk_size = min(len(chunk_tokens[0]), len(chunk_next_tokens))

            # Trim input chunk if needed
            chunk_tokens = chunk_tokens[:, :actual_chunk_size]

            # Process chunk based on model type
            if hf_model_name:
                outputs = model(chunk_tokens)
                ref_output = outputs.logits
            else:
                pt_decode_input = embd(chunk_tokens).view(1, actual_chunk_size, -1)
                ref_output = model(pt_decode_input, start_pos=chunk_start)

            # Compute top-5 predictions
            probs = torch.softmax(ref_output, dim=-1)
            _, chunk_top5_tokens = torch.topk(probs, k=5, dim=-1)  # Shape: [1, chunk_size, 5]
            chunk_top5_tokens = chunk_top5_tokens.squeeze(0)  # Shape: [chunk_size, 5]

            # Get next tokens tensor, ensuring same length as predictions
            chunk_next_tokens_tensor = torch.tensor(chunk_next_tokens[:actual_chunk_size])

            # Calculate correctness
            chunk_top1_correct = chunk_top5_tokens[:, 0] == chunk_next_tokens_tensor
            chunk_top5_correct = torch.any(chunk_top5_tokens == chunk_next_tokens_tensor.unsqueeze(1), dim=1)

            # Store results
            all_top1_correct.extend(chunk_top1_correct.tolist())
            all_top5_correct.extend(chunk_top5_correct.tolist())
            all_top5_tokens.append(chunk_top5_tokens)

            # Print predictions for this chunk
            for i in range(len(chunk_next_tokens)):
                global_pos = chunk_start + i
                next_token = chunk_next_tokens[i]

                sanitize = lambda x: x.replace("\n", "").replace("\r", "").replace("\x0c", "")
                actual_token = sanitize(tokenizer.decode([next_token]))
                top5_tokens = [sanitize(tokenizer.decode([t.item()])) for t in chunk_top5_tokens[i]]
                correct = "x" if chunk_top1_correct[i] else ("-" if chunk_top5_correct[i] else " ")
                top5_str = " ".join(f"{t:<14}" for t in top5_tokens)

                progress_str = f"{global_pos+1}/{total_length-1}"
                print(f"{progress_str:<15}{correct:<8}{actual_token:<15}{top5_str}")

                # Calculate and store segment accuracies every 100 tokens
                if (global_pos + 1) % 100 == 0 or global_pos == total_length - 2:
                    start_idx = (global_pos // 100) * 100
                    end_idx = min(start_idx + 100, len(all_top1_correct))
                    segment_top1_acc = sum(all_top1_correct[start_idx:end_idx]) / (end_idx - start_idx) * 100
                    segment_top5_acc = sum(all_top5_correct[start_idx:end_idx]) / (end_idx - start_idx) * 100
                    if len(segment_accuracies) <= global_pos // 100:
                        segment_accuracies.append((segment_top1_acc, segment_top5_acc))

        # Concatenate all top5 tokens into a single tensor
        all_top5_tokens = torch.cat(all_top5_tokens, dim=0)  # Shape: [total_tokens, 5]

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
    parser.add_argument("--total_length", type=int, default=1024, help="Total length of tokens to process")
    parser.add_argument(
        "--output_file", type=str, default="reference_outputs.pt", help="Output file path for reference data"
    )
    parser.add_argument("--model", type=str, help="Optional: HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-hf')")
    args = parser.parse_args()

    generate_reference_outputs(total_length=args.total_length, output_file=args.output_file, hf_model_name=args.model)


if __name__ == "__main__":
    main()
