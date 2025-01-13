# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
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
from models.demos.t3000.llama2_70b.demo.demo import (
    build_generator,
    construct_arg,
    initialize_inputs,
    get_sampling_func,
)
from datetime import datetime
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
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


def prepare_next_input_eval(tokenizer, tokens, input_text_mask, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token

    prev_pos = cur_pos

    return tokens, prev_pos


def run_forward(
    model_args,
    tt_args,
    data_args,
    eval_data_args,
    model,
    tokenizer,
    prompt_tokens,
    prompts,
    return_logits=False,
    return_full_logits=False,
):
    """
    return_logits: return the logits for the last token
    return_full_logits: return the logits for all tokens
    """
    assert not (return_logits and return_full_logits), "return_logits and return_full_logits cannot both be true"

    # decode arguments
    bsz = model_args.max_batch_size
    output_tokens = data_args.max_output_tokens

    sampling_func = get_sampling_func(data_args.top_k, data_args.top_p, data_args.temperature)

    total_len = min(model_args.max_kv_context_len, eval_data_args.sample_len + output_tokens)
    assert total_len <= model_args.max_kv_context_len
    assert total_len - 1 == prompt_tokens.size(1), "Prompt tokens must be of length total_len"

    # prepare inputs
    tokens, input_text_mask, _ = initialize_inputs(tokenizer, prompt_tokens, bsz, total_len)
    prev_pos = 0

    # some profiling and logging
    full_logits = []

    # Prefill up to sample_len, then decode for output_tokens-1 num tokens.
    for cur_pos in range(eval_data_args.sample_len, eval_data_args.sample_len + output_tokens):
        logger.info(f"EVAL: Inference from token {prev_pos} to {cur_pos}")
        input_tokens = tokens[:, prev_pos:cur_pos]

        if prev_pos == 0:  # Prefill
            logits = []
            for b in range(bsz):
                logits.append(model.prefill_forward_single_user(input_tokens[b : b + 1], prev_pos, b))
            logits = torch.cat(logits, dim=0)
        else:  # Decode
            logits = model.forward(input_tokens, prev_pos)

        next_logits = logits[:, -1, :]  # batch, vocab of last token
        next_token = sampling_func(next_logits)

        tokens, prev_pos = prepare_next_input_eval(tokenizer, tokens, input_text_mask, cur_pos, next_token)

        full_logits.append(logits.clone().detach())

    full_logits = torch.cat(full_logits, dim=1)
    return full_logits


def calculate_perplexity(
    model_args,
    tt_args,
    data_args,
    eval_data_args,
    model,
    tokenizer,
    seq_len,
    max_len,
    stride,
    encodings,
    num_samples,
    max_batch_size,
):
    start = time()
    eval_inputs = []
    eval_labels = []
    total_len = len(encodings)

    # construct perplexity calculation inputs
    for i, begin_loc in enumerate(range(0, total_len, stride)):
        end_loc = begin_loc + seq_len + data_args.max_output_tokens - 1
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
            logits = run_forward(
                model_args,
                tt_args,
                data_args,
                eval_data_args,
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=tokens,
                prompts=None,
                return_full_logits=True,
            )

            vocab_size = logits.shape[-1]
            logger.info(f"logits shape: {logits.shape}")
            loss = loss_func(logits.view(-1, vocab_size), labels.view(-1))
            neg_log_likelihood = loss.to("cpu").float() * logits.shape[1] * logits.shape[0]
            tokens_seen += logits.shape[1] * logits.shape[0]
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
    ],
    ids=["greedy"],
)
@pytest.mark.parametrize(  # sample_len => prefill length, num_samples => decode length
    "dataset, split, config, stride, sample_len, max_output_tokens, num_samples, perplexity_score",
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

    t3k_mesh_device.enable_async(True)

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
