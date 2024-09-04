# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import numpy as np
import ttnn
from models.demos.falcon7b_common.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon7b_common.tt.model_config import get_model_config
from models.demos.falcon7b_common.tests.test_utils import initialize_kv_cache, load_hf_model
from models.datasets.llm_dataset_utils import (
    prepare_textgen_dataset,
    prepare_textgen_dataloader,
    calculate_acc_metrics,
    verify_acc_metrics,
)
from models.utility_functions import tt_tensors_to_torch_tensors


def calculate_perplexity(
    model, dataloader, llm_mode, batch_size, seq_len, kv_cache, configuration, use_hf_model=False, mesh_device=None
):
    if not use_hf_model:
        assert mesh_device is not None
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
                    logits = tt_tensors_to_torch_tensors(tt_logits, mesh_device, concat_dim=0).squeeze(1)
                    # Deallocate tt tensors
                    tt_prefill_input_ids.deallocate()
                    if isinstance(tt_prefill_attention_mask, ttnn.Tensor):
                        tt_prefill_attention_mask.deallocate()
                    elif isinstance(tt_prefill_attention_mask, list):
                        for tt_attention_mask_element in tt_prefill_attention_mask:
                            tt_attention_mask_element.deallocate()
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
                        logits_cur = tt_tensors_to_torch_tensors(tt_logits, mesh_device, concat_dim=2).squeeze(1)
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
    model_version="tiiuae/falcon-7b-instruct",
    num_layers=32,
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
        model_config = get_model_config(model_config_str, max_seq_len, batch_size)
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
            max_seq_len,
        )

        # Initialize kvcache
        logger.info("Initializing kvcache...")
        kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, mesh_device)
    else:
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
        use_hf_model=use_hf_model,
        mesh_device=mesh_device,
    )
    logger.info(f"Perplexity evaluation time: {(time.time() - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    logger.info(f"Top-1 accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 accuracy: {top5_acc:.4f}")

    # Verify metrics against targets
    calculated_acc_metrics = {"ppl": ppl, "top1_acc": top1_acc, "top5_acc": top5_acc}
    verify_acc_metrics(calculated_acc_metrics, expected_acc_metrics)
