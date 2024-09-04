# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import os
import json
import re
import torch
import torch.nn.functional as F

import datasets
import pytest
from loguru import logger

from models.demos.t3000.llama2_70b.demo.demo import (
    build_generator,
    construct_arg,
)
from datetime import datetime
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
)
from models.demos.t3000.llama2_70b.demo.eval import (
    wikitext_detokenizer,
    calculate_perplexity,
)


@dataclass
class EvalDataArgs:
    dataset: str = "wikitext"
    split: str = "test"
    config: str = "wikitext-2-raw-v1"
    stride: int = 128
    sample_len: int = 128
    num_samples: int = 128
    perplexity_score: float = 5.4


def main(args, eval_data_args):
    # Set random reproducible seed
    torch.manual_seed(0)
    model_args = args.model
    tt_args = args.tt
    data_args = args.data

    generator = build_generator(model_args, tt_args)

    # Load the model and tokenizer
    model, tokenizer = generator.model, generator.tokenizer

    # Dataset preparation
    dataset = datasets.load_dataset(
        eval_data_args.dataset, eval_data_args.config, split=eval_data_args.split, ignore_verifications=True
    )
    text = wikitext_detokenizer("\n".join(dataset["text"]))
    encodings = tokenizer.encode(text, bos=True, eos=False)  # not prepending bos

    # args for perplexity calculation
    max_length = model_args.max_seq_len
    stride = eval_data_args.stride if eval_data_args.stride else max_length
    assert stride <= max_length, "stride cannot be larger than max_length"
    seq_len = eval_data_args.sample_len
    num_samples = eval_data_args.num_samples
    max_batch_size = model_args.max_batch_size
    assert num_samples > 0, "num_samples must be greater than 0"
    assert seq_len + (num_samples - 1) * stride <= len(encodings), (
        "total length of token decoded must be less than the length of the dataset, \
        the maximum allowed num_samples is: %d"
        % (len(encodings) - seq_len)
        // stride
        + 1
    )

    perplexity = calculate_perplexity(
        model_args,
        tt_args,
        data_args,
        eval_data_args,
        model,
        tokenizer,
        seq_len,
        max_length,
        stride,
        encodings,
        num_samples,
        max_batch_size,
    )

    base_path = "models/demos/t3000/llama2_70b/scripts/llama_perplexity_runs/"
    os.makedirs(base_path, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dump perplexity and max_seq_len to a JSON file with timestamp and max_length in the file name
    filename = os.path.join(
        base_path,
        f"perplexity_{model_args.llama_version}_{model_args.implementation}_{model_args.num_layers}L_"
        f"{eval_data_args.sample_len}_{data_args.max_output_tokens}_{timestamp}.json",
    )
    result = {
        "model": model_args.llama_version,
        "perplexity": perplexity.item(),
        "seq_len": eval_data_args.sample_len,
        "gen_length": data_args.max_output_tokens,
    }
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(result, f)

    logger.info("Perplexity: %f" % perplexity)
    if perplexity < eval_data_args.perplexity_score:
        logger.info("Perplexity is less than the threshold")
    else:
        assert False, "Perplexity is greater than the threshold"

    return perplexity


@pytest.mark.parametrize(
    "llama_version",
    [
        ("llama3"),
    ],
)
@pytest.mark.parametrize("num_layers", [80], ids=["80L"])
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices",
    [
        (
            "tt",
            False,
            8,
        ),
    ],
    ids=["tt-70b"],
)
@pytest.mark.parametrize(
    "top_p, top_k, temperature",
    [
        (1, 1, 1.0),
    ],
    ids=["greedy"],
)
@pytest.mark.parametrize(  # sample_len => prefill length, num_samples => decode length
    "dataset, split, config, stride, sample_len, max_output_tokens, num_samples, perplexity_score",
    [
        ("wikitext", "test", "wikitext-2-raw-v1", 128, 128, 128, 128, 5.4),
        ("wikitext", "test", "wikitext-2-raw-v1", 128, 2048, 128, 128, 3.4313),
    ],
    ids=["wikitext-128-128", "wikitext-2k-128"],
)
def test_LlamaModel_demo(
    # model args
    implementation,
    skip_model_load,
    num_layers,
    # Generation args
    max_output_tokens,
    top_p,
    top_k,
    temperature,
    # TT args
    t3k_mesh_device,
    n_devices,
    # Dataset args
    dataset,
    split,
    config,
    stride,
    sample_len,
    num_samples,
    perplexity_score,
    llama_version,
    use_program_cache,
):
    logger.info("Running LlamaModel perplexity test")
    ## Get model config
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )

    check_mesh_device(t3k_mesh_device, model_config)

    for i in t3k_mesh_device.get_device_ids():
        device = t3k_mesh_device.get_device(i)
        device.enable_async(True)

    args = construct_arg(
        implementation=implementation,
        llama_version=llama_version,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        chat=False,
        mesh_device=t3k_mesh_device,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=False,
    )

    eval_data_args = EvalDataArgs(
        dataset=dataset,
        split=split,
        config=config,
        stride=stride,
        sample_len=sample_len,
        num_samples=num_samples,
        perplexity_score=perplexity_score,
    )

    main(args, eval_data_args)
