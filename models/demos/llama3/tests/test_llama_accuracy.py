# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import bz2
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_common import (
    get_single_rot_mat,
    get_prefill_rot_mat,
    prepare_inputs_ttnn,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    HostEmbedding,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer


@torch.no_grad()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("prefill_len", [0])
@pytest.mark.parametrize("decode_len", [1000])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_model_accuracy(mesh_device, prefill_len, decode_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    mesh_device.enable_async(True)

    # Load model args and tokenizer
    model_args = TtModelArgs(mesh_device)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Load state_dict for TT model
    logger.info("Loading weights...")
    state_dict = model_args.load_state_dict()
    logger.info("Finished loading weights...")

    # Load the reference data
    reference_data_file = "ref-tokens-1b-1000.pt"
    reference_data = torch.load(reference_data_file)
    reference_tokens = reference_data["reference_tokens"]
    top5_tokens = reference_data["top5_tokens"]

    N = prefill_len + decode_len
    input_ids = reference_tokens[:, : N + 1]  # Shape [1, N+1]

    # Initialize TT model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )
    # Initialize embedding
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # Skip prefill if prefill_len is 0
    if prefill_len > 0:
        # For prefill, process prefill_len tokens
        batch_size = 1
        prefill_ids = input_ids[:, :prefill_len]  # Shape [1, prefill_len]
        pt_prefill_input = embd(prefill_ids).view(batch_size, prefill_len, -1)

        # Pre-compute the rotational embedding matrix and send to device
        rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=prefill_len)
        transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
        transformation_mats = ttnn.from_torch(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run prefill
        decode_input = prepare_inputs_ttnn_prefill(
            pt_prefill_input,
            mesh_device,
        )
        tt_out = tt_model(decode_input, None, rot_mats, transformation_mats, user_id=0, mode="prefill")
        ttnn.deallocate(tt_out)

    # Start decoding
    generation_start_pos = prefill_len
    generation_length = decode_len
    current_pos = ttnn.from_torch(
        torch.tensor([generation_start_pos]),
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        mesh_device,
        model_args.num_devices,
        start_pos=max(0, generation_start_pos - 1),
    )

    # Print table header
    print(f"{'Progress':<15}{'Correct':<8}{'True':<15}{'Actual':<15}{'Top 5 Predictions':<75}")
    print("-" * 128)

    top1_correct = []
    top5_correct = []
    errors = []  # New list to store error information

    for i in range(generation_length):
        # Input is reference token at each step
        ref_token = input_ids[0, prefill_len + i].item()
        # Get the true next token (if available)
        true_token = input_ids[0, prefill_len + i + 1].item() if i < generation_length - 1 else None
        # Convert to torch tensor
        ref_token = torch.tensor([[ref_token]], dtype=torch.int32)  # Shape [1,1]
        # Get embedding
        pt_decode_input = embd(ref_token).view(1, 1, -1)
        # Prepare input for TT model
        decode_input = prepare_inputs_ttnn(
            pt_decode_input,
            model_args.dim,
            mesh_device,
        )
        # Run TT model
        tt_out = tt_model(decode_input, current_pos, rot_mat=current_rot_mat)

        if tt_model.args.num_devices > 1:
            tt_out_gathered = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.deallocate(tt_out)
        else:
            tt_out_gathered = tt_out
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True)
        ttnn.deallocate(tt_out_gathered)
        tt_out_tok = ttnn.argmax(tt_out_rm, dim=3, use_multicore=True)
        tt_argmax_token = ttnn.to_torch(tt_out_tok, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
            0, 0, 0, 0
        ]
        ttnn.deallocate(tt_out_rm)
        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        ttnn.plus_one(current_pos)

        # Reset rotation matrix every 100 iterations
        if i % 100 == 0:  # Doing this every 100 iterations as in demo takes top5 from 99% ->
            current_rot_mat, rot_matrix_reset = get_single_rot_mat(
                model_args.head_dim,
                mesh_device,
                model_args.num_devices,
                start_pos=generation_start_pos + i,
                on_host=False,
            )

        # Get reference top5 tokens and probabilities for this position
        ref_top5_tokens = top5_tokens[prefill_len + i]

        # Check top-1 and top-5 accuracy
        top1_match = tt_argmax_token.item() == ref_top5_tokens[0].item()
        top1_correct.append(top1_match)
        top5_match = tt_argmax_token in ref_top5_tokens
        top5_correct.append(top5_match)
        true_match = (
            tt_argmax_token.item() == input_ids[0, prefill_len + i + 1].item() if i < generation_length - 1 else False
        )

        # Store error information if top5 is incorrect
        if not top5_match:
            context_start = max(0, prefill_len + i - 9)
            context_tokens = input_ids[0, context_start : prefill_len + i + 1]
            context_text = tokenizer.decode(context_tokens.tolist())
            incorrect_token = tokenizer.decode([tt_argmax_token])
            expected_tokens = [tokenizer.decode([t]) for t in ref_top5_tokens]
            errors.append(
                {
                    "position": prefill_len + i,
                    "context": context_text,
                    "incorrect": incorrect_token,
                    "expected": expected_tokens,
                    "predicted_id": tt_argmax_token.item(),
                    "expected_ids": ref_top5_tokens.tolist(),
                }
            )

        # Decode tokens to text
        tt_argmax_text = tokenizer.decode([tt_argmax_token])
        true_text = tokenizer.decode([true_token]) if true_token is not None else "N/A"
        ref_top5_text = [tokenizer.decode([t]) for t in ref_top5_tokens]

        # Prepare table row
        sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes
        progress_str = f"{i+1}/{generation_length}"
        correct = "x" if top1_match else ("-" if top5_match else ("!" if true_match else " "))
        tt_argmax_text = sanitize(tt_argmax_text)
        true_text = sanitize(true_text)
        ref_top5_str = " ".join(f"{sanitize(t):<14}" for t in ref_top5_text)

        # Print table row
        print(f"{progress_str:<15}{correct:<8}{true_text:<15}{tt_argmax_text:<15}{ref_top5_str}")

    # Compute accuracies over every 100 tokens
    num_tokens = len(top1_correct)
    num_segments = (num_tokens + 99) // 100
    for seg in range(num_segments):
        start = seg * 100
        end = min(start + 100, num_tokens)
        top1_acc = 100 * sum(top1_correct[start:end]) / (end - start)
        top5_acc = 100 * sum(top5_correct[start:end]) / (end - start)
        max_width = len(str(decode_len))
        print(
            f"Tokens {start:{max_width}d}-{end:{max_width}d}: Top-1 accuracy: {top1_acc:3.0f} %, Top-5 accuracy: {top5_acc:3.0f} %"
        )

    # Report total accuracies
    total_top1_acc = 100 * sum(top1_correct) / num_tokens
    total_top5_acc = 100 * sum(top5_correct) / num_tokens
    print(
        f"Total tokens {num_tokens}: Top-1 accuracy: {total_top1_acc:3.0f} %, Top-5 accuracy: {total_top5_acc:3.0f} %"
    )

    # Print error summary
    print("\nError Summary (only showing errors where reference top-1 matches true token):")
    print("-" * 120)
    for error in errors:
        true_token = input_ids[0, error["position"] + 1].item()
        if error["expected_ids"][0] == true_token:
            sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes
            context = sanitize(error["context"])
            incorrect = sanitize(error["incorrect"])
            expected = " | ".join(sanitize(t) for t in error["expected"])
            true_word = sanitize(tokenizer.decode([true_token]))
            print(f"{error['position']}: {context}[{incorrect}] != [{expected}], true: [{true_word}]")

    # Optionally, you can add assertions to check if the accuracies meet expectations
    assert total_top1_acc > 0.2, f"Top-1 accuracy {total_top1_acc:.4f} is too low"
    assert total_top5_acc > 0.5, f"Top-5 accuracy {total_top5_acc:.4f} is too low"
