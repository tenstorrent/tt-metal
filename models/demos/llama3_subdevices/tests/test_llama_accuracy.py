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
from models.demos.llama3_subdevices.tt.sampling import TTSampling
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from tqdm import tqdm

is_RING_6U = os.environ.get("RING_6U", "0") == "1"


@torch.no_grad()
@pytest.mark.parametrize(
    "min_top1_acc, min_top5_acc",  # Max seqlen should be at least prefill_len + decode_len
    ((91, 99),),
)
@pytest.mark.parametrize(
    "prefill_len, decode_len, max_seq_len",  # Max seqlen should be at least prefill_len + decode_len
    ((512, 511, 128 * 1024),),
)
@pytest.mark.parametrize(
    "sampling_params",
    [{"top_k": 1, "top_p": 0.00, "seed": 42}],
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
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 23887872,
            "worker_l1_size": 1344544,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if is_RING_6U else ttnn.FabricConfig.FABRIC_1D,
        }
    ],
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
    sampling_params,
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
        enable_prefetcher_performance_mode=True,
    )
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        sampling_params=sampling_params,
        tt_ccl=tt_model.tt_ccl,
    )
    # Initialize embedding
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})
    current_pos = torch.tensor([0 for b in range(model_args.max_batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    # Get the first input tensors
    _, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

    ref_token = input_ids[0, 0].item()  # First token
    ref_token = torch.tensor([[ref_token]], dtype=torch.int32)
    pt_decode_input = embd(ref_token).view(1, 1, -1).expand(32, -1, -1)

    decode_input = model_args.prepare_residual_tensor_decode(
        pt_decode_input,
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    def run_model():
        rot_mats = tt_model.rope_setup.get_rm_rot_mats(rot_mat_idxs)

        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Sampling
        tt_out_tok = tt_sampling(tt_out[0])

        # Update the idxs
        ttnn.plus_one(
            current_pos_tensor,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

        return tt_out_tok, tt_out[0]

    # Compile the model
    logger.info("Compiling model...")
    tt_out_tok, tt_out = run_model()

    # Capturing trace
    logger.info("Capturing trace...")

    tt_model.tt_ccl.reset_gather_and_buffer_idx()

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    tt_out_tok, tt_out = run_model()

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Reset the decoding position for the proper run of the model
    current_pos_reset = ttnn.from_torch(
        current_pos,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # Reset the current position and output token tensors for the real decode run
    ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos_tensor)
    rot_mat_idxs_reset = tt_model.rope_setup.get_rm_rot_idxs(current_pos, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_mat_idxs_reset, rot_mat_idxs)

    ttnn.synchronize_device(mesh_device)

    # Skip prefill if prefill_len is 0
    if prefill_len > 0:
        logger.info(f"Starting prefill...")
        for i in tqdm(range(prefill_len)):
            # Input is reference token at each step
            ref_token = input_ids[0, i].item()
            # Convert to torch tensor
            ref_token = torch.tensor([[ref_token]], dtype=torch.int32)  # Shape [1,1]
            # Get embedding
            pt_decode_input = embd(ref_token).view(1, 1, -1).expand(32, -1, -1)
            # Prepare input for TT model
            decode_input_new = model_args.prepare_residual_tensor_decode(
                pt_decode_input, model_args.model_config["DECODE_RESIDUAL_MEMCFG"], on_host=True
            )

            # Run the trace with a new input
            ttnn.copy_host_to_device_tensor(decode_input_new, decode_input)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    # Start decoding
    logger.info(f"Starting decode...")
    generation_length = decode_len

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
        decode_input_new = model_args.prepare_residual_tensor_decode(
            pt_decode_input, model_args.model_config["DECODE_RESIDUAL_MEMCFG"], on_host=True
        )

        # Run the trace with a new input
        ttnn.copy_host_to_device_tensor(decode_input_new, decode_input)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        if not use_reference_file:
            # Convert ttnn tensor to torch tensor
            tt_logits = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=model_args.cluster_shape),
            )[0, 0, 0, : model_args.vocab_size]

        tt_argmax_token = ttnn.to_torch(
            tt_out_tok,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(3, 1),
                mesh_shape=model_args.cluster_shape,
            ),
        )[0, 0, 0, 0]

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

    tt_model.tt_ccl.close()

    logger.info(f"Top-1: {total_top1_acc:.0f}% | Top-5: {total_top5_acc:.0f}%")
    assert (
        total_top1_acc >= min_top1_acc
    ), f"Top-1 accuracy {total_top1_acc:.1f}% is too low (expected >={min_top1_acc}%)"
    assert (
        total_top5_acc >= min_top5_acc
    ), f"Top-5 accuracy {total_top5_acc:.1f}% is too low (expected >={min_top5_acc}%)"
