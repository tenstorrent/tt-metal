# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_common import (
    HostEmbedding,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from pathlib import Path


def get_accuracy_thresholds(model_name: str, device_name: str, optimizations: LlamaOptimizations):
    """Parse accuracy thresholds from PERF.md for the given model, optimization mode, and device."""
    # Get model size (e.g., "1b", "3b", etc.)
    model_size = model_name.split("-")[1].lower()

    # Read PERF.md
    perf_file = Path(__file__).parent.parent / "PERF.md"
    with open(perf_file, "r") as f:
        content = f.read()

    # Split into sections based on optimization mode
    sections = content.split("## ")
    target_section = next(s for s in sections if s.startswith(f"LlamaOptimizations.{optimizations.__name__}\n"))

    # Parse the table and find the row for our model and device
    rows = [
        line.split("|")[1:]  # Each row starts with a separator
        for line in target_section.replace(" ", "").split("\n")
        if f"|{model_size}|{device_name}|" in line
    ]
    if not rows:
        raise ValueError(
            f"Could not find accuracy data for {model_size} on {device_name} in {optimizations.__name__} mode"
        )

    assert (
        len(rows) == 1
    ), f"Found multiple rows for {model_size} on {device_name} in {optimizations.__name__} mode in PERF.md"
    row = rows[0]
    top1_acc = float(row[2].strip())
    top5_acc = float(row[3].strip())

    # Allow for rounding
    return top1_acc - 0.5, top5_acc - 0.5


@torch.no_grad()
@pytest.mark.parametrize(
    "min_top1_acc, min_top5_acc",  # Max seqlen should be at least prefill_len + decode_len
    ((91, 99),),
)
@pytest.mark.parametrize(
    "prefill_len, decode_len, max_seq_len",  # Max seqlen should be at least prefill_len + decode_len
    ((512, 128, 1024),),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "optimizations",
    [
        pytest.param(LlamaOptimizations.performance, id="performance"),
    ],
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "use_reference_file",
    [
        # pytest.param(True, id="reference_file"),
        pytest.param(True, id="reference_text"),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_tt_model_acc(
    prefill_len,
    decode_len,
    max_seq_len,
    batch_size,
    min_top1_acc,
    min_top5_acc,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    use_reference_file,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    is_ci_env,
):
    if is_ci_env and not use_reference_file:
        pytest.skip("CI test only runs vs reference file")

    dtype = ttnn.bfloat8_b

    # Load model args and tokenizer
    model_args = TtModelArgs(
        mesh_device, optimizations=optimizations, max_batch_size=batch_size, max_seq_len=max_seq_len, instruct=True
    )

    # model_args.n_layers = 1

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Load state_dict for TT model
    logger.info("Loading weights...")
    state_dict = model_args.load_state_dict()
    logger.info("Finished loading weights...")

    # Load the reference data
    model_size = model_args.model_name.split("-")[1].lower()  # e.g., "1b", "3b", "8b", "70b"

    if use_reference_file:
        # Existing reference file loading logic
        reference_data_file = "models/tt_transformers/tests/reference_outputs/Llama3.1-70B-Instruct.refpt"
        logger.info(f"Loading reference data from {reference_data_file}")
        assert os.path.exists(reference_data_file)
        reference_data = torch.load(reference_data_file)
        reference_tokens = reference_data["reference_tokens"]
        top5_tokens = reference_data["top5_tokens"]
    else:
        # Load and encode the reference text
        # current_file_path = os.path.dirname(os.path.abspath(__file__))
        # prompt_file = os.path.join(current_file_path, "tale-of-two-cities.txt.bz2")
        # with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        #     text = f.read()
        text = "This is a test. It's important to conduct tests to ensure everything is functioning correctly. Whether it's a new software application, a scientific experiment, or a simple task, testing helps us identify any issues and make improvements. When we test, we learn about the strengths and weaknesses of what we're working with, allowing us to make necessary adjustments. In the end, testing leads to better outcomes and higher quality results. So, let's proceed with this test and see what we discover. Remember, every test is a step towards perfection. In academic and professional settings, tests and assessments are crucial for validating knowledge and skills. They offer insights into areas that require further development and help establish benchmarks for progress. From standardized tests in education to quality assurance in manufacturing, the principle of testing spans across various fields, underlining"
        # Encode text to tokens
        encoded_tokens = tokenizer.encode(text, bos=True, eos=False)
        total_length = prefill_len + decode_len + 1
        reference_tokens = torch.tensor(encoded_tokens[:total_length]).unsqueeze(0)
        top5_tokens = None  # Will be computed during inference

    N = prefill_len + decode_len
    input_ids = reference_tokens[:, : N + 1]  # Shape [1, N+1]

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Initialize TT model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    # Initialize embedding
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})
    current_pos_tensor = ttnn.from_torch(
        torch.tensor([0 for b in range(model_args.max_batch_size)]),
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    current_pos = torch.tensor([0 for b in range(model_args.max_batch_size)])
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    # Skip prefill if prefill_len is 0
    if prefill_len > 0:
        logger.info(f"Starting prefill...")
        for i in range(prefill_len):
            # Input is reference token at each step
            ref_token = input_ids[0, i].item()
            # Convert to torch tensor
            ref_token = torch.tensor([[ref_token]], dtype=torch.int32)  # Shape [1,1]
            # Get embedding
            pt_decode_input = embd(ref_token).view(1, 1, -1).expand(32, -1, -1)
            # Prepare input for TT model
            decode_input = model_args.prepare_residual_tensor_decode(
                pt_decode_input,
                model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            )
            rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

            tt_out = tt_model(
                decode_input,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )

            tt_out_gathered = tt_model.tt_ccl.line_all_gather(
                tt_out[0],
                dim=3,
                num_links=2,
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SAMPLING",
            )
            tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True, sub_core_grids=sub_core_grids)
            tt_out_tok = ttnn.argmax(  # FIXME When ttnn.argmax supports multicore, avoid falling back to host
                tt_out_rm, dim=3, keepdim=True, use_multicore=True, sub_core_grids=sub_core_grids
            )
            tt_out_tok = ttnn.to_torch(
                tt_out_tok,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(3, 1) if model_args.is_galaxy else (1, 3),
                    mesh_shape=model_args.cluster_shape,
                ),
            )[0, 0, :32, 0].view(32, 1)
            print("output", tokenizer.decode([ref_token]), tokenizer.decode([tt_out_tok.squeeze(1).tolist()[0]]))
            ttnn.plus_one(
                current_pos_tensor,
                sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            )

            current_pos += 1
    # Start decoding
    logger.info(f"Starting decode...")
    generation_start_pos = prefill_len
    generation_length = decode_len

    # Initial positions
    decoding_pos = [generation_start_pos] * model_args.max_batch_size
    current_pos = torch.tensor([decoding_pos[b] for b in range(model_args.max_batch_size)])

    # Get cos/sin matrices for the current position of each user
    rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

    # Print table header
    if use_reference_file:
        logger.info(f"{'Progress':<15}{'Correct':<8}{'True':<15}{'Actual':<15}{'Top 5 Predictions':<75}")
    else:
        logger.info(f"{'Progress':<15}{'Correct':<8}{'True':<15}{'Top 5 Predictions':<75}")
    logger.info("-" * 113)

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
        pt_decode_input = embd(ref_token).view(1, 1, -1).expand(32, -1, -1)
        # Prepare input for TT model
        decode_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_out_gathered = tt_model.tt_ccl.line_all_gather(
            tt_out[0], dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="SAMPLING"
        )
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True, sub_core_grids=sub_core_grids)
        tt_out_tok = ttnn.argmax(  # FIXME When ttnn.argmax supports multicore, avoid falling back to host
            tt_out_rm, dim=3, keepdim=True, use_multicore=True, sub_core_grids=sub_core_grids
        )
        if not use_reference_file:
            tt_logits = ttnn.to_torch(
                tt_out_rm,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(2, 1),
                    mesh_shape=model_args.cluster_shape,
                ),
            )[0, 0, 0, :]
        ttnn.deallocate(tt_out_rm)

        tt_argmax_token = ttnn.to_torch(
            tt_out_tok,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(3, 1),
                mesh_shape=model_args.cluster_shape,
            ),
        )[0, 0, 0, 0]

        ttnn.plus_one(
            current_pos_tensor,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

        # Update rot_mats for next iteration
        current_pos += 1
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

        # Modify the accuracy checking section when using reference text
        if not use_reference_file:
            # Get probabilities from model output
            probs = torch.softmax(tt_logits, dim=-1)
            _, tt_top5_tokens = torch.topk(probs, k=5, dim=-1)

            # Check against actual next token
            true_token = input_ids[0, prefill_len + i + 1].item()
            top1_match = tt_argmax_token.item() == true_token
            top5_match = true_token in tt_top5_tokens
            ref_top5_text = [tokenizer.decode([t]) for t in tt_top5_tokens]
        else:
            # Existing logic for reference file comparison
            ref_top5_tokens = top5_tokens[prefill_len + i]
            top1_match = tt_argmax_token.item() == ref_top5_tokens[0].item()
            top5_match = tt_argmax_token in ref_top5_tokens
            ref_top5_text = [tokenizer.decode([t]) for t in ref_top5_tokens]

        # Check top-1 and top-5 accuracy
        top1_correct.append(top1_match)
        top5_correct.append(top5_match)
        true_match = (
            tt_argmax_token.item() == input_ids[0, prefill_len + i + 1].item() if i < generation_length - 1 else False
        )

        # Store error information vs reference model if top5 is incorrect
        if use_reference_file and not top5_match:
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

        sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes

        # Decode tokens to text
        tt_argmax_text = tokenizer.decode([tt_argmax_token])
        true_text = tokenizer.decode([true_token]) if true_token is not None else "N/A"

        # Prepare table row
        progress_str = f"{i+1}/{generation_length}"
        correct = "x" if top1_match else ("-" if top5_match else ("!" if true_match else " "))
        tt_argmax_text = sanitize(tt_argmax_text)
        true_text = sanitize(true_text)
        ref_top5_str = " ".join(f"{sanitize(t):<14}" for t in ref_top5_text)

        # Print table row
        if use_reference_file:
            logger.info(f"{progress_str:<15}{correct:<8}{true_text:<15}{tt_argmax_text:<15}{ref_top5_str}")
        else:
            logger.info(f"{progress_str:<15}{correct:<8}{true_text:<15}{ref_top5_str}")

    # Compute accuracies over every 100 tokens
    num_tokens = len(top1_correct)
    num_segments = (num_tokens + 99) // 100
    for seg in range(num_segments):
        start = seg * 100
        end = min(start + 100, num_tokens)
        top1_acc = 100 * sum(top1_correct[start:end]) / (end - start)
        top5_acc = 100 * sum(top5_correct[start:end]) / (end - start)
        max_width = len(str(decode_len))
        logger.info(
            f"Tokens {start:{max_width}d}-{end:{max_width}d}: Top-1 accuracy: {top1_acc:3.0f} %, Top-5 accuracy: {top5_acc:3.0f} %"
        )

    # Report total accuracies
    total_top1_acc = 100 * sum(top1_correct) / num_tokens
    total_top5_acc = 100 * sum(top5_correct) / num_tokens
    logger.info(
        f"Total tokens {num_tokens}: Top-1 accuracy: {total_top1_acc:3.0f} %, Top-5 accuracy: {total_top5_acc:3.0f} %"
    )

    # Only show error summary when using reference files
    if use_reference_file:
        logger.info("\nError Summary (only showing errors where reference top-1 matches true token):")
        logger.info("-" * 120)
        for error in errors:
            true_token = input_ids[0, error["position"] + 1].item()
            if error["expected_ids"][0] == true_token:
                sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes
                context = sanitize(error["context"])
                incorrect = sanitize(error["incorrect"])
                expected = " | ".join(sanitize(t) for t in error["expected"])
                true_word = sanitize(tokenizer.decode([true_token]))
                logger.info(f"{error['position']}: {context}[{incorrect}] != [{expected}], true: [{true_word}]")

    # Get accuracy thresholds from PERF.md
    # min_top1_acc, min_top5_acc = get_accuracy_thresholds(
    #     model_args.model_name,
    #     model_args.device_name,
    #     optimizations,s
    # )

    tt_model.tt_ccl.close()

    logger.info(f"Top-1: {total_top1_acc:.0f}% | Top-5: {total_top5_acc:.0f}%")
    assert (
        total_top1_acc >= min_top1_acc
    ), f"Top-1 accuracy {total_top1_acc:.1f}% is too low (expected >={min_top1_acc}%)"
    assert (
        total_top5_acc >= min_top5_acc
    ), f"Top-5 accuracy {total_top5_acc:.1f}% is too low (expected >={min_top5_acc}%)"
