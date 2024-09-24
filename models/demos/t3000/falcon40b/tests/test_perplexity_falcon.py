# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import numpy as np
import ttnn
from ttnn import ConcatMeshToTensor
from models.demos.t3000.falcon40b.tt.falcon_common import (
    PytorchFalconCausalLM,
)
from models.demos.t3000.falcon40b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.t3000.falcon40b.tt.model_config import get_model_config
from models.demos.t3000.falcon40b.tests.test_utils import load_hf_model
from models.datasets.llm_dataset_utils import (
    prepare_textgen_dataset,
    prepare_textgen_dataloader,
    calculate_acc_metrics,
    verify_acc_metrics,
)
from models.utility_functions import is_wormhole_b0, tt_tensors_to_torch_tensors


def calculate_perplexity(
    model, dataloader, llm_mode, batch_size, seq_len, kv_cache, configuration, mesh_device, use_hf_model=False
):
    if llm_mode == "prefill" and not use_hf_model:
        assert batch_size == 1
    use_cache = True
    running_nll, running_top1_acc, running_top5_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Evaluating batches"):
            if llm_mode == "prefill":
                if not use_hf_model:
                    user_id = 0
                    (
                        tt_prefill_input_ids,
                        tt_prefill_attention_mask,
                    ) = model.model_preprocessing(
                        "prefill", input_ids[user_id::batch_size], 0, num_input_tokens=seq_len
                    )
                    tt_logits, kv_cache = model(
                        input_ids=tt_prefill_input_ids,
                        llm_mode="prefill",
                        attention_mask=tt_prefill_attention_mask,
                        user_id=user_id,
                        layer_past=kv_cache,
                        layer_past_len=0,
                        use_cache=use_cache,
                    )
                    # Get outputs from all devices
                    logits = ttnn.to_torch(
                        tt_logits, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1)
                    )
                    # Deallocate tt tensors
                    tt_prefill_input_ids.deallocate()
                    tt_prefill_attention_mask.deallocate()
                    tt_logits.deallocate()
                else:  # huggingface model
                    logits, _ = model(input_ids=input_ids, use_cache=use_cache, return_dict=False)

            elif llm_mode == "decode":
                logits = []
                layer_present = None
                for kv_cache_len in tqdm(range(seq_len), desc="Decoding tokens for current batch"):
                    decode_ids = input_ids[:, kv_cache_len].view(batch_size, 1)
                    if not use_hf_model:
                        (
                            tt_decode_input_ids,
                            tt_decode_attention_mask,
                        ) = model.model_preprocessing(
                            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
                        )
                        tt_logits, kv_cache = model(
                            input_ids=tt_decode_input_ids,
                            llm_mode="decode",
                            attention_mask=tt_decode_attention_mask,
                            layer_past=kv_cache,
                            layer_past_len=kv_cache_len,
                            use_cache=use_cache,
                        )
                        # Get outputs from all devices
                        logits_cur = ttnn.to_torch(
                            tt_logits, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1)
                        )
                        logits.append(logits_cur.view(-1, 1, configuration.vocab_size))
                        # Deallocate tt tensors
                        tt_decode_input_ids.deallocate()
                        tt_decode_attention_mask.deallocate()
                        tt_logits.deallocate()
                    else:  # huggingface model
                        logits_cur, layer_present = model(
                            input_ids=decode_ids, past_key_values=layer_present, use_cache=use_cache, return_dict=False
                        )
                        logits.append(logits_cur)

                logits = torch.cat(logits, dim=1)

            # Re-shape logits and labels and calculate metrics
            logits = logits.view(batch_size * seq_len, configuration.vocab_size)
            labels = labels.view(-1)
            nll, top1_acc, top5_acc = calculate_acc_metrics(logits, labels)
            running_nll += nll
            running_top1_acc += top1_acc
            running_top5_acc += top5_acc

    nll = running_nll / len(dataloader)
    ppl = np.exp(nll)
    top1_acc = running_top1_acc / len(dataloader)
    top5_acc = running_top5_acc / len(dataloader)
    return nll, ppl, top1_acc, top5_acc


def run_test_perplexity(
    llm_mode,
    batch_size,
    max_seq_len,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    mesh_device,
    num_samples,
    expected_acc_metrics,
    stride=None,
    model_version="tiiuae/falcon-40b-instruct",
    num_layers=60,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    use_hf_model=False,
):
    # Set random reproducible seed
    torch.manual_seed(0)

    # Load HF model
    logger.info("Loading HuggingFace model...")
    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_textgen_dataset(dataset_name, dataset_config, split)
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    encodings = tokenizer(dataset, return_tensors="pt")["input_ids"].squeeze(0)
    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride)

    if not use_hf_model:
        # Load tt-metal model config
        input_shape = [batch_size, max_seq_len]
        model_config = get_model_config(
            model_config_str, llm_mode, input_shape, num_devices=len(mesh_device.get_devices())
        )
        tt_cache_path = get_tt_cache_path(
            model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
        )

        # Load tt-metal model
        logger.info("Moving weights (all layers) to device; might take some time...")
        model = TtFalconCausalLM(
            mesh_device,
            state_dict,
            "",
            num_layers,
            configuration,
            max_seq_len,
            model_config,
            tt_cache_path,
            use_global_cos_sin_cache=True,
        )
        for device in mesh_device.get_devices():
            ttnn.synchronize_device(device)

        # Initialize kvcache
        logger.info("Initializing kvcache...")
        kv_cache = model.initialize_kv_cache()
    else:
        # model = pytorch_FalconCausalLM
        model = hugging_face_reference_model
        kv_cache = None

    # Evaluate perplexity
    logger.info("Evaluating perplexity...")
    start = time.time()
    nll, ppl, top1_acc, top5_acc = calculate_perplexity(
        model,
        dataloader,
        llm_mode,
        batch_size,
        max_seq_len,
        kv_cache,
        configuration,
        mesh_device,
        use_hf_model=use_hf_model,
    )
    logger.info(f"Perplexity evaluation time: {(time.time() - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")

    # Verify metrics against targets
    calculated_acc_metrics = {"ppl": ppl, "top1_acc": top1_acc, "top5_acc": top5_acc}
    verify_acc_metrics(calculated_acc_metrics, expected_acc_metrics)


@pytest.mark.parametrize(
    "llm_mode, batch_size, max_seq_len, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        ("prefill", 32, 128, 64, 12.67, 0.47, 0.71),
        ("prefill", 32, 1024, 64, 7.21, 0.55, 0.79),
        ("prefill", 32, 2048, 64, 9.81, 0.50, 0.74),  # TODO: run for falcon40b
        ("decode", 64, 128, 64, 12.67, 0.47, 0.71),
        ("decode", 64, 1024, 64, 7.21, 0.55, 0.79),
        ("decode", 64, 2048, 64, 9.81, 0.50, 0.74),  # TODO: run for falcon40b
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq1024",
        "prefill_seq2048",
        "decode_128",
        "decode_1024",
        "decode_2048",
    ],
)
def test_perplexity_huggingface(
    llm_mode,
    batch_size,
    max_seq_len,
    num_samples,  # Total number of prompts to evaluate (all if None)
    expected_ppl,
    expected_top1,
    expected_top5,
    model_location_generator,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skipping HF reference test in CI environment")

    run_test_perplexity(
        llm_mode,
        batch_size,
        max_seq_len,
        None,
        model_location_generator,
        None,
        None,
        num_samples,
        {"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5},
        use_hf_model=True,
    )


@pytest.mark.parametrize(
    "llm_mode, batch_size, max_seq_len, model_config_str, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        ("prefill", 1, 128, "BFLOAT8_B-DRAM", 64, 12.74, 0.47, 0.71),
        ("prefill", 1, 1024, "BFLOAT8_B-DRAM", 64, 7.25, 0.55, 0.78),
        ("prefill", 1, 2048, "BFLOAT8_B-DRAM", 64, 6.55, 0.56, 0.80),
        ("decode", 32, 128, "BFLOAT8_B-SHARDED", 64, 13.91, 0.46, 0.71),
        ("decode", 32, 1024, "BFLOAT8_B-SHARDED", 64, 7.79, 0.54, 0.78),
        ("decode", 32, 2048, "BFLOAT8_B-SHARDED", 64, 6.96, 0.55, 0.79),  # TODO: Hangs on CI
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq1024",
        "prefill_seq2048",
        "decode_128",
        "decode_1024",
        "decode_2048",
    ],
)
def test_perplexity(
    llm_mode,
    batch_size,
    max_seq_len,
    model_config_str,
    num_samples,  # Total number of prompts to evaluate (all if None)
    expected_ppl,
    expected_top1,
    expected_top5,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
):
    assert is_wormhole_b0(), "This test is only for Wormhole B0"

    if llm_mode == "decode" and max_seq_len > 128:
        pytest.skip("Decode mode is hanging for seqlen > 128")

    run_test_perplexity(
        llm_mode,
        batch_size,
        max_seq_len,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        t3k_mesh_device,
        num_samples,
        {"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5},
    )
