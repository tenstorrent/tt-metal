# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from loguru import logger
from typing import Tuple, Callable
import json

import numpy as np
import torch
from tqdm import tqdm

from models.demos.wormhole.qwen2_7b.reference.tokenizer import Tokenizer
from models.demos.wormhole.qwen2_7b.reference.model import Transformer, Emb
from models.demos.wormhole.qwen2_7b.tt.model_config import TtModelArgs

import ttnn
from models.demos.wormhole.qwen2_7b.tt.qwen2_model import TtTransformer
from models.demos.wormhole.qwen2_7b.tt.qwen2_common import (
    precompute_freqs,
    freqs_to_rotation_matrix,
    load_safetensor_weights,
    prepare_inputs_ttnn,
)

from models.datasets.llm_dataset_utils import (
    prepare_textgen_dataset,
    prepare_textgen_dataloader,
    calculate_acc_metrics,
    verify_acc_metrics,
)


def calculate_perplexity(compute_logits_fn: Callable, dataloader, vocab_size) -> Tuple[float, float, float, float]:
    running_nll, running_top1_acc, running_top5_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Evaluating batches"):
            batch, seqlen = input_ids.shape
            logits = compute_logits_fn(input_ids, seqlen)
            logits = logits.view(batch * seqlen, vocab_size)
            labels = labels.view(-1)
            nll, top1_acc, top5_acc = calculate_acc_metrics(logits, labels)
            running_nll += nll
            running_top1_acc += top1_acc
            running_top5_acc += top5_acc

    nll = running_nll / len(dataloader)
    ppl = np.exp(nll)
    top1_acc = running_top1_acc / len(dataloader)
    top5_acc = running_top5_acc / len(dataloader)
    return float(nll), float(ppl), float(top1_acc), float(top5_acc)


@pytest.mark.parametrize(
    "batch_size, max_seq_len, num_samples",
    [
        # Combinations for general weights
        (8, 64, 64)
    ],
)
def test_qwen2_reference_perplexity(batch_size: int, max_seq_len: int, num_samples: int):
    torch.manual_seed(0)

    logger.info("Preparing dataset")
    dataset = prepare_textgen_dataset("wikitext", "wikitext-2-raw-v1", "test")

    model_args = TtModelArgs("")

    tokenizer = Tokenizer(model_args.tokenizer_path)
    encodings = torch.tensor(tokenizer.encode(dataset))

    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride=None)

    logger.info("Loading weights...")
    state_dict = load_safetensor_weights(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        )
    }
    logger.info("Loading weights finished!")

    reference_model = Transformer(args=model_args)

    partial_state_dict = {k[len("model.") :]: v for k, v in state_dict.items() if "layers." in k}
    partial_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    partial_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    reference_model.load_state_dict(partial_state_dict)

    embd = Emb(model_args.vocab_size, model_args.dim, tokenizer.pad_id)
    embd.load_state_dict({"emb.weight": state_dict["model.embed_tokens.weight"]})

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    def decode(input_ids, seqlen: int):
        logits = []
        for idx in range(seqlen):
            next_token = input_ids[:, idx].unsqueeze(-1)  # (B, 1)
            next_token_emb = embd(next_token).view(input_ids.shape[0], 1, -1)

            freqs_cis_i = freqs_cis[idx, :].unsqueeze(0)
            positions = torch.tensor([idx])
            next_token_logits = reference_model(next_token_emb, freqs_cis_i, positions)
            logits.append(next_token_logits)
        return torch.cat(logits, dim=1)

    compute_logits = decode

    logger.info(f"Evaluating perplexity (batch_size={batch_size} max_seq_len={max_seq_len} num_samples={num_samples})")
    start = time.time()
    nll, ppl, top1_acc, top5_acc = calculate_perplexity(compute_logits, dataloader, reference_model.args.vocab_size)
    end = time.time()

    logger.info(f"Perplexity evaluation time: {(end - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")


@pytest.mark.parametrize(
    "batch_size, max_seq_len, num_samples",
    [
        # Combinations for general weights
        (8, 64, 64)
    ],
)
def test_qwen2_perplexity(device: str, batch_size: int, max_seq_len: int, num_samples: int):
    torch.manual_seed(0)

    logger.info("Preparing dataset")
    dataset = prepare_textgen_dataset("wikitext", "wikitext-2-raw-v1", "test")

    model_args = TtModelArgs(device)

    tokenizer = Tokenizer(model_args.tokenizer_path)
    encodings = torch.tensor(tokenizer.encode(dataset))

    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride=None)

    logger.info("Loading weights...")
    state_dict = load_safetensor_weights(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        )
    }
    logger.info("Loading weights finished!")

    embd = Emb(model_args.vocab_size, model_args.dim, tokenizer.pad_id)
    embd.load_state_dict({"emb.weight": state_dict["model.embed_tokens.weight"]})

    logger.info("Loading weights to device...")

    dtype = ttnn.bfloat8_b

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    rot_emb_matrix = freqs_to_rotation_matrix(cos, sin)

    rot_emb_matrix_list = []
    for i in range(rot_emb_matrix.shape[0]):
        rot_emb_matrix_list.append(
            ttnn.from_torch(
                rot_emb_matrix[i, :, :].unsqueeze(0).unsqueeze(0), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
            )
        )  # ttnn.bfloat16

    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
        rot_mat=rot_emb_matrix_list,
        start_pos=0,
    )
    logger.info("Finished loading weights to device. Starting inference...")

    def decode(input_ids, seqlen: int):
        logits = []
        for idx in range(seqlen):
            next_token = input_ids[:, idx].unsqueeze(-1)  # (B, 1)
            next_token_emb = embd(next_token).view(input_ids.shape[0], 1, -1)

            decode_input, current_pos = prepare_inputs_ttnn(
                next_token_emb,
                idx,
                model_args.dim,
                model_args.sliding_window,
                tt_model.device,
            )

            tt_out_logits = tt_model(decode_input, current_pos)

            # [batch, seq, hidden_dim]
            torch_out = ttnn.to_torch(tt_out_logits)
            next_token_logits = torch_out.permute(2, 1, 0, 3).squeeze(1)[:batch_size, :, :]
            logits.append(next_token_logits)
        return torch.cat(logits, dim=1)

    compute_logits = decode

    logger.info(f"Evaluating perplexity (batch_size={batch_size} max_seq_len={max_seq_len} num_samples={num_samples})")
    start = time.time()
    nll, ppl, top1_acc, top5_acc = calculate_perplexity(compute_logits, dataloader, model_args.vocab_size)
    end = time.time()

    logger.info(f"Perplexity evaluation time: {(end - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")
