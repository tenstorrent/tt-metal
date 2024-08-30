# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from loguru import logger
from typing import Tuple, Callable

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

import ttnn

from models.demos.wormhole.mamba.reference.prefill_decode_model import Mamba, MambaPretrainedModelName
from models.demos.wormhole.mamba.tt import model_config
from models.demos.wormhole.mamba.tt.model_config import ModelMode
from models.demos.wormhole.mamba.tt.mamba_model import MambaTT

from models.datasets.llm_dataset_utils import (
    prepare_textgen_dataset,
    prepare_textgen_dataloader,
    calculate_acc_metrics,
    verify_acc_metrics,
)
from models.utility_functions import skip_for_grayskull


def calculate_perplexity(
    compute_logits_fn: Callable, reset_model_states_fn: Callable, dataloader, vocab_size
) -> Tuple[float, float, float, float]:
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

            reset_model_states_fn()

    nll = running_nll / len(dataloader)
    ppl = np.exp(nll)
    top1_acc = running_top1_acc / len(dataloader)
    top5_acc = running_top5_acc / len(dataloader)
    return float(nll), float(ppl), float(top1_acc), float(top5_acc)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "model_version, mode, batch_size, max_seq_len, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        ("state-spaces/mamba-2.8b", ModelMode.DECODE, 32, 64, 64, 21.6, 0.39, 0.65),
        ("state-spaces/mamba-2.8b", ModelMode.DECODE, 32, 128, 64, 15.4, 0.44, 0.69),
        ("state-spaces/mamba-2.8b", ModelMode.PREFILL, 32, 64, 64, 21.6, 0.39, 0.65),
        ("state-spaces/mamba-2.8b", ModelMode.PREFILL, 32, 128, 64, 15.4, 0.44, 0.69),
    ),
)
def test_mamba_reference_perplexity(
    model_version: MambaPretrainedModelName,
    mode: ModelMode,
    batch_size: int,
    max_seq_len: int,
    num_samples: int,
    expected_ppl: int,
    expected_top1: int,
    expected_top5: int,
    is_ci_env,
    reset_seeds,
):
    torch.manual_seed(0)

    if is_ci_env:
        pytest.skip("Disabled on CI due to long execution time")
    else:
        logger.warning("Mamba CPU reference test is currently disabled on CI due to long execution times")

    logger.info("Preparing dataset")
    dataset = prepare_textgen_dataset("wikitext", "wikitext-2-raw-v1", "test")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    encodings = tokenizer(dataset, return_tensors="pt")["input_ids"].squeeze(0)
    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride=None)

    reference_model = Mamba.from_pretrained(model_version, batch_size=batch_size)
    reference_model.args.mode = mode
    reference_model.eval()

    def decode(input_ids, seqlen: int):
        logits = []
        for idx in range(seqlen):
            next_token = input_ids[:, idx].unsqueeze(-1)  # (B, 1)
            next_token_logits = reference_model(next_token)
            logits.append(next_token_logits)
        return torch.cat(logits, dim=1)

    def prefill(input_ids, _: int):
        return reference_model(input_ids)

    if mode == ModelMode.DECODE:
        compute_logits = decode
    else:
        compute_logits = prefill

    logger.info(f"Evaluating perplexity (batch_size={batch_size} max_seq_len={max_seq_len} num_samples={num_samples})")
    start = time.time()
    nll, ppl, top1_acc, top5_acc = calculate_perplexity(
        compute_logits, reference_model.initialize_states, dataloader, reference_model.args.vocab_size
    )
    end = time.time()

    logger.info(f"Perplexity evaluation time: {(end - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")

    calculated_acc_metrics = {"ppl": ppl, "top1_acc": top1_acc, "top5_acc": top5_acc}
    expected_acc_metrics = {"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5}
    verify_acc_metrics(calculated_acc_metrics, expected_acc_metrics)


@pytest.mark.timeout(1200)
@skip_for_grayskull("Mamba not supported on Grayskull")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_version, mode, batch_size, max_seq_len, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        ("state-spaces/mamba-2.8b", ModelMode.DECODE, 32, 64, 64, 27.2, 0.378, 0.620),
        ("state-spaces/mamba-2.8b", ModelMode.DECODE, 32, 128, 64, 19.5, 0.410, 0.667),
        ("state-spaces/mamba-2.8b", ModelMode.PREFILL, 1, 64, 64, 24.7, 0.375, 0.632),
        ("state-spaces/mamba-2.8b", ModelMode.PREFILL, 1, 128, 64, 18.6, 0.415, 0.670),
    ),
)
def test_mamba_perplexity(
    device: ttnn.Device,
    model_version: MambaPretrainedModelName,
    mode: ModelMode,
    batch_size: int,
    max_seq_len: int,
    num_samples: int,
    expected_ppl: int,
    expected_top1: int,
    expected_top5: int,
    use_program_cache,
    get_tt_cache_path,
    reset_seeds,
):
    torch.manual_seed(0)

    logger.info("Preparing dataset")
    dataset = prepare_textgen_dataset("wikitext", "wikitext-2-raw-v1", "test")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    encodings = tokenizer(dataset, return_tensors="pt")["input_ids"].squeeze(0)
    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride=None)

    reference_model = Mamba.from_pretrained(model_version, batch_size=batch_size)
    reference_model.args.mode = mode
    reference_model.eval()

    if mode == ModelMode.DECODE:
        config = model_config.create_model_config(batch_size, reference_model.args.d_model, mode=mode, seq_len=1)

        start = time.time()
        model = MambaTT(reference_model, device, config, tt_cache_path=get_tt_cache_path(model_version))
        logger.info(f"Finished initializing Mamba (took {time.time() - start:.3f} sec)")

        def decode(input_ids, seqlen: int):
            logits = []
            for idx in range(seqlen):
                next_token = input_ids[:, idx].unsqueeze(-1)  # (B, 1)
                next_token_logits = model(next_token)
                logits.append(next_token_logits)
            return torch.cat(logits, dim=1)

        compute_logits = decode
        reset = model.reset
    else:
        prefill_chunk_size = max_seq_len
        config = model_config.create_model_config(
            batch_size, reference_model.args.d_model, mode=mode, seq_len=prefill_chunk_size
        )
        start = time.time()
        model = MambaTT(reference_model, device, config, tt_cache_path=get_tt_cache_path(model_version))
        logger.info(f"Finished initializing Mamba (took {time.time() - start:.3f} sec)")

        def prefill(input_ids, _: int):
            return model(input_ids)  # assumes input fits into single chunk

        compute_logits = prefill
        reset = model.reset

    logger.info(f"Evaluating perplexity (batch_size={batch_size} max_seq_len={max_seq_len} num_samples={num_samples})")
    start = time.time()
    nll, ppl, top1_acc, top5_acc = calculate_perplexity(
        compute_logits, reset, dataloader, reference_model.args.vocab_size
    )
    end = time.time()

    logger.info(f"Perplexity evaluation time: {(end - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")

    calculated_acc_metrics = {"ppl": ppl, "top1_acc": top1_acc, "top5_acc": top5_acc}
    expected_acc_metrics = {"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5}
    verify_acc_metrics(calculated_acc_metrics, expected_acc_metrics)
