# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import json
import re
import torch
import torch.nn.functional as F

from time import time
import datasets
import pytest
from loguru import logger

from tqdm import tqdm
from models.demos.t3000.llama2_70b.demo.demo import run_decode, build_generator
from datetime import datetime
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_device_mesh,
)


def main(args):
    # Set random reproducible seed
    torch.manual_seed(0)
    generator = build_generator(args)

    # Load the model and tokenizer
    model, tokenizer = generator.model, generator.tokenizer

    # Dataset preparation
    dataset = datasets.load_dataset(args.dataset, args.config, split=args.split, ignore_verifications=True)
    text = wikitext_detokenizer("\n".join(dataset["text"]))
    encodings = tokenizer.encode(text, bos=True, eos=False)  # not prepending bos

    # args for perplexity calculation
    max_length = args.max_seq_len
    stride = args.stride if args.stride else max_length
    assert stride <= max_length, "stride cannot be larger than max_length"
    seq_len = args.sample_len
    num_samples = args.num_samples
    max_batch_size = args.max_batch_size
    assert num_samples > 0, "num_samples must be greater than 0"
    assert seq_len + (num_samples - 1) * stride <= len(encodings), (
        "total length of token decoded must be less than the length of the dataset, \
        the maximum allowed num_samples is: %d"
        % (len(encodings) - seq_len)
        // stride
        + 1
    )

    perplexity = calculate_perplexity(
        args, model, tokenizer, seq_len, max_length, stride, encodings, num_samples, max_batch_size
    )

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dump perplexity and max_seq_len to a JSON file with timestamp and max_length in the file name
    filename = f"models/demos/t3000/llama2_70b/scripts/llama_perplexity_runs/perplexity_{args.llama_version}_{args.implementation}_{args.num_layers}L_{args.sample_len}_{args.num_tokens}_{timestamp}.json"
    result = {
        "model": args.llama_version,
        "perplexity": perplexity.item(),
        "seq_len": args.sample_len,
        "gen_length": args.num_tokens,
    }
    with open(filename, "w") as f:
        json.dump(result, f)

    logger.info("Perplexity: %f" % perplexity)
    if perplexity < args.perplexity_score:
        logger.info("Perplexity is less than the threshold")
    else:
        assert False, "Perplexity is greater than the threshold"

    return perplexity


def calculate_perplexity(args, model, tokenizer, seq_len, max_len, stride, encodings, num_samples, max_batch_size):
    start = time()
    eval_inputs = []
    eval_labels = []
    total_len = len(encodings)

    # construct perplexity calculation inputs
    for i, begin_loc in enumerate(range(0, total_len, stride)):
        end_loc = begin_loc + seq_len + args.num_tokens - 1
        if end_loc >= total_len:
            raise ValueError(
                "The dataset is too small to decode the number of samples requested with the given stride."
            )
        inputs = encodings[begin_loc:end_loc]
        labels = encodings[begin_loc + 1 : end_loc + 1]
        eval_inputs.append(inputs)
        eval_labels.append(labels)
        if i == num_samples - 1:
            break

    # batch perplexity calculation inputs
    dataset = torch.utils.data.TensorDataset(torch.tensor(eval_inputs), torch.tensor(eval_labels))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=max_batch_size, shuffle=False)

    # Run PPL eval
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    nlls = []
    tokens_seen = 0
    with torch.no_grad():
        # Loop over the dataset
        for sample in tqdm(dataloader):
            tokens, labels = sample
            outputs = run_decode(
                args=args,
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=tokens,
                prompts=None,
                return_full_logits=True,
            )

            all_text, logits = outputs
            vocab_size = logits.shape[-1]
            loss = loss_func(logits.view(-1, vocab_size), labels.view(-1))
            neg_log_likelihood = loss.to("cpu").float() * seq_len * max_batch_size
            tokens_seen += seq_len * max_batch_size
            nlls.append(neg_log_likelihood)

    loss = torch.stack(nlls).sum() / tokens_seen
    ppl = torch.exp(loss)

    logger.info(f"Evaluation execution time:\t{time() - start}")
    logger.info("Loss: %f" % loss)

    return ppl


def wikitext_detokenizer(string):
    """From Megatron-DeepSpeed/tasks/zeroshot_gpt/detokenizer.py"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def get_model_max_sequence_length(model):
    for c in ["n_positions", "seq_length"]:
        if hasattr(model.config, c):
            return getattr(model.config, c)
    return None


class Args:
    def __init__(
        self,
        # model args
        implementation="meta",
        ckpt_dir="/home/llama-data-repacked-2/llama-2-70b/",
        tokenizer_path="/home/llama-data/tokenizer.model",
        skip_model_load=False,
        max_batch_size=32,
        num_layers=None,
        max_seq_len=4096,
        # Generation args
        num_tokens=1,
        prompts_file="models/demos/t3000/llama2_70b/demo/data/multi_prompt.json",
        output_at_end=True,
        top_p=1,
        top_k=1,
        temperature=1.0,
        # TT args
        device_mesh=None,
        n_devices=8,
        emulated=False,
        cache_path=None,
        decode_only=False,
        # Dataset args
        dataset="wikitext",
        split="test",
        config="wikitext-2-raw-v1",
        stride=128,
        sample_len=128,
        num_samples=32,
        perplexity_score=5.4,
        llama_version="llama3",
    ):
        self.implementation = implementation
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.skip_model_load = skip_model_load
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.prompts_file = prompts_file
        self.output_at_end = output_at_end
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.device_mesh = device_mesh
        self.n_devices = n_devices
        self.emulated = emulated
        self.cache_path = cache_path
        self.decode_only = decode_only
        self.dataset = dataset
        self.split = split
        self.config = config
        self.stride = stride
        self.sample_len = sample_len
        self.num_samples = num_samples
        self.perplexity_score = perplexity_score
        self.llama_version = llama_version


def construct_arg(**kwargs):
    return Args(**kwargs)


@pytest.mark.timeout(240000)
@pytest.mark.parametrize(
    "llama_version",
    (
        ("llama2"),
        ("llama3"),
    ),
)
@pytest.mark.parametrize("num_layers", (1, 2, 10, 80), ids=["1L", "2L", "10L", "80L"])
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices",
    [
        (
            "tt",
            False,
            8,
        ),
        (
            "meta",
            False,
            8,
        ),
    ],
    ids=["tt-70b", "meta-70b"],
)
@pytest.mark.parametrize(
    "top_p, top_k, temperature",
    [
        (1, 1, 1.0),
        (0.9, 10, 1.0),
    ],
    ids=["greedy", "sampling"],
)
@pytest.mark.parametrize(  # sample_len => prefill length, num_samples => decode length
    "dataset, split, config, stride, sample_len, num_tokens, num_samples, perplexity_score",
    [
        ("wikitext", "test", "wikitext-2-raw-v1", 128, 128, 1, 128, 5.4),
        ("wikitext", "test", "wikitext-2-raw-v1", 128, 2048, 1, 128, 3.4313),
        ("wikitext", "test", "wikitext-2-raw-v1", 128, 128, 128, 128, 5.4),
        ("wikitext", "test", "wikitext-2-raw-v1", 128, 2048, 128, 128, 3.4313),
    ],
    ids=["wikitext-128-0", "wikitext-2k-0", "wikitext-128-128", "wikitext-2k-128"],
)
def test_LlamaModel_demo(
    # model args
    implementation,
    skip_model_load,
    num_layers,
    # Generation args
    num_tokens,
    top_p,
    top_k,
    temperature,
    # TT args
    t3k_device_mesh,
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
    logger.info("Running LlamaModel demo")
    ## Get model config

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )

    check_device_mesh(t3k_device_mesh, model_config)

    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_async(True)

    args = construct_arg(
        implementation=implementation,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        num_tokens=num_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        device_mesh=t3k_device_mesh,
        n_devices=n_devices,
        cache_path=cache_path,
        dataset=dataset,
        split=split,
        config=config,
        stride=stride,
        sample_len=sample_len,
        num_samples=num_samples,
        perplexity_score=perplexity_score,
        llama_version=llama_version,
    )
    main(args)
