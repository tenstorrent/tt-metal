# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import json
import pytest
from loguru import logger
from time import time
import sys
from tqdm import tqdm
import numpy as np

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
    get_single_rot_mat,
    sample,
    cache_attention,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.datasets.llm_dataset_utils import (
    prepare_textgen_dataset,
    prepare_textgen_dataloader,
    calculate_acc_metrics,
    verify_acc_metrics,
)


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@torch.no_grad()
def run_test_perplexity(
    mesh_device,
    batch_size,
    llm_mode,
    max_seq_len,
    num_samples,
    expected_acc_metrics,
    instruct_mode=False,  # For now, only run accuracy check for general weights
    stride=None,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
):
    assert batch_size == 32, "Batch size must be 32"
    dtype = ttnn.bfloat8_b
    embed_on_host = True  # embedding and argmax on host
    seqlen = 1  # Generating one token per user at a time

    # Flag to compile the ref model and run the accuracy metrics on it
    validate_ref_model = False

    # Accuracy metrics
    running_nll, running_top1_acc, running_top5_acc = 0.0, 0.0, 0.0

    if validate_ref_model:
        ref_running_nll, ref_running_top1_acc, ref_running_top5_acc = 0.0, 0.0, 0.0

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(mesh_device.get_device(0), instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)
    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    model_args.n_layers = 32  # Full model

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_textgen_dataset(dataset_name, dataset_config, split)
    if instruct_mode:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encodings = torch.tensor(tokenizer.encode("[INST] " + dataset + " [/INST]"))
    else:
        encodings = torch.tensor(tokenizer.encode(dataset))
    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride)

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)
    # If not using the full model, remove the layers that are not used
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(model_args.n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)

    # Embedding on host
    if embed_on_host:
        embd = Emb()
        embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Load TTNN mixtral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
        rotary_on_host=True,
    )

    if validate_ref_model:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)
        reference_model.eval()

    if not embed_on_host:
        tt_embds = TtMixtralEmbedding(
            mesh_device=mesh_device,
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )

    logger.info("Finished loading weights to device.")

    # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        tt_model.mesh_device,
    )

    generation_start_pos = 0

    cache_attention(
        mesh_device,
        state_dict,
        model_args,
        current_rot_mat,
        rot_matrix,
        dtype,
    )

    logger.info("Starting inference...")
    for input_ids, labels in tqdm(dataloader, desc="Evaluating batches"):
        if llm_mode == "prefill":
            # TODO Add prefill support
            assert "Prefill mode not yet supported"
        elif llm_mode == "decode":
            logits = []
            if validate_ref_model:
                ref_logits = []

            for kv_cache_len in tqdm(range(max_seq_len), desc="Decoding tokens for current batch"):
                # Convert input_id into ttnn input
                pt_decode_input = embd(input_ids[:, kv_cache_len]).view(batch_size, seqlen, -1)

                start_pos = generation_start_pos + kv_cache_len
                current_pos = start_pos

                if embed_on_host:
                    decode_input_11BH = prepare_inputs_ttnn(
                        pt_decode_input,
                        model_args.dim,
                        start_pos,
                        model_args,
                        tt_model.mesh_device,
                    )
                else:
                    assert "Only embedding on host is supported for now!"

                # Run ttnn mixtral model
                tt_logits = tt_model(decode_input_11BH, start_pos, current_pos)

                if embed_on_host:
                    # Convert ttnn tensor to torch tensor
                    pt_logits = (
                        ttnn.to_torch(tt_logits, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))[0]
                        .squeeze(1)
                        .view(32, seqlen, -1)
                        .detach()
                        .float()
                    )[:batch_size, ...]
                else:
                    assert "Only embedding on host is supported for now!"

                logits.append(pt_logits.view(-1, 1, model_args.vocab_size))

                if validate_ref_model:
                    positions = torch.LongTensor([start_pos])
                    ref_pt_logits = reference_model(pt_decode_input, positions).detach().float()
                    ref_logits.append(ref_pt_logits.view(-1, 1, model_args.vocab_size))

        logits = torch.cat(logits, dim=1)
        # Re-shape logits and labels and calculate metrics
        logits = logits.view(batch_size * max_seq_len, model_args.vocab_size)  # batch_size * max_seq_len, vocab_size
        labels = labels.view(-1)  # batch_size * max_seq_len

        # Calculate accuracy metrics
        nll, top1_acc, top5_acc = calculate_acc_metrics(logits, labels)
        running_nll += nll
        running_top1_acc += top1_acc
        running_top5_acc += top5_acc

        if validate_ref_model:
            ref_logits = torch.cat(ref_logits, dim=1)
            # Re-shape logits and labels and calculate metrics
            ref_logits = ref_logits.view(
                batch_size * max_seq_len, model_args.vocab_size
            )  # batch_size * max_seq_len, vocab_size

            # Calculate accuracy metrics
            ref_nll, ref_top1_acc, ref_top5_acc = calculate_acc_metrics(ref_logits, labels)
            ref_running_nll += ref_nll
            ref_running_top1_acc += ref_top1_acc
            ref_running_top5_acc += ref_top5_acc

    # Validate accuracy metrics against the expected values
    nll = running_nll / len(dataloader)
    ppl = np.exp(nll)
    top1_acc = running_top1_acc / len(dataloader)
    top5_acc = running_top5_acc / len(dataloader)
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")

    calculated_acc_metrics = {"ppl": ppl, "top1_acc": top1_acc, "top5_acc": top5_acc}
    verify_acc_metrics(calculated_acc_metrics, expected_acc_metrics)

    if validate_ref_model:
        ref_nll = ref_running_nll / len(dataloader)
        ref_ppl = np.exp(ref_nll)
        ref_top1_acc = ref_running_top1_acc / len(dataloader)
        ref_top5_acc = ref_running_top5_acc / len(dataloader)
        logger.info(f"Ref Negative log-likelihood: {nll:.4f}")
        logger.info(f"Ref Perplexity: {ppl:.4f}")
        logger.info(f"Ref Top-1 accuracy: {top1_acc:.4f}")
        logger.info(f"Ref Top-5 accuracy: {top5_acc:.4f}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "llm_mode, max_seq_len, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        # TODO Add prefill accuracy tests
        # ("prefill", 128, 64, -, -, -),
        # ("prefill", 1024, 64, -, -, -),
        # ("prefill", 2048, 64, -, -, -),
        # ("prefill", 4096, 64, -, -, -),
        ("decode", 128, 64, 8.80, 0.52, 0.75),
        # ("decode", 1024, 64, 5.10, 0.62, 0.83),
        # ("decode", 2048, 64, 4.23, 0.64, 0.85),
        # ("decode", 4096, 32, 10.59, 0.49, 0.73),
    ),
    ids=[
        # "prefill_128",
        # "prefill_1024",
        # "prefill_2048",
        # "prefill_4096",
        "decode_128",
        # "decode_1024",
        # "decode_2048",
        # "decode_4096",
    ],
)
def test_mixtral_perplexity(
    t3k_mesh_device,
    use_program_cache,
    reset_seeds,
    llm_mode,
    max_seq_len,
    num_samples,
    expected_ppl,
    expected_top1,
    expected_top5,
):
    assert (
        llm_mode == "decode"
    ), "Only decode mode is supported for now"  # TODO Add prefill support when it reaches main

    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)

    return run_test_perplexity(
        mesh_device=t3k_mesh_device,
        batch_size=32,
        llm_mode=llm_mode,
        max_seq_len=max_seq_len,
        num_samples=num_samples,
        expected_acc_metrics={"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5},
    )
