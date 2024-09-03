# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import os
import json
import torch
import torch.nn.functional as F

from time import time
import pytest
from loguru import logger

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from transformers.generation.utils import top_k_top_p_filtering
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from models.demos.t3000.llama2_70b.tt.llama_common import load_llama_state_dict
from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import ChatFormat
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
    string_similarity_score,
)


@dataclass
class ModelArgs:
    implementation: str = "meta"
    llama_version: str = None
    ckpt_dir: str = "/home/llama-data-repacked-2/llama-2-70b/"
    tokenizer_path: str = "/home/llama-data/tokenizer.model"
    skip_model_load: bool = False
    max_batch_size: int = 32
    num_layers: int = None
    max_seq_len: int = 4096
    max_kv_context_len: int = 4096


@dataclass
class TTArgs:
    mesh_device: object = None
    n_devices: int = 8
    emulated: bool = False
    cache_path: str = None
    decode_only: bool = False


@dataclass
class DataArgs:
    max_output_tokens: int = 128
    prompts_file: str = "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json"
    output_at_end: bool = True
    top_p: float = 1
    top_k: int = 1
    temperature: float = 1.0
    chat: bool = False
    sample_len: int = None
    ground_truth: str = None


@dataclass
class DemoArgs:
    model: ModelArgs
    tt: TTArgs
    data: DataArgs


def construct_arg(**kwargs):
    model_args = ModelArgs(**{k: v for k, v in kwargs.items() if hasattr(ModelArgs, k)})
    tt_args = TTArgs(**{k: v for k, v in kwargs.items() if hasattr(TTArgs, k)})
    data_args = DataArgs(**{k: v for k, v in kwargs.items() if hasattr(DataArgs, k)})
    return DemoArgs(model=model_args, tt=tt_args, data=data_args)


def main(args):
    # Set random reproducible seed
    torch.manual_seed(0)

    model_args = args.model
    tt_args = args.tt
    data_args = args.data

    # Load ground truth if available
    if data_args.ground_truth:
        if not os.path.exists(data_args.ground_truth):
            logger.info(f"Ground truth file {data_args.ground_truth} does not exist.")
            data_args.ground_truth = None
        else:
            ground_truth_outputs = json.load(open(data_args.ground_truth, "r"))

            if len(ground_truth_outputs) == 0:
                logger.info("Ground truth outputs are empty")
                data_args.ground_truth = None
            else:
                logger.info(f"Loaded {len(ground_truth_outputs)} ground truth outputs")

    generator = build_generator(model_args, tt_args)

    # Load the model and tokenizer
    model, tokenizer = generator.model, generator.tokenizer

    tokenized, prompts = load_prompts_file(model_args, data_args, tokenizer)

    # Run decode
    with torch.no_grad():
        all_text = run_decode(
            model_args, tt_args, data_args, model=model, tokenizer=tokenizer, prompt_tokens=tokenized, prompts=prompts
        )

        if data_args.output_at_end:
            with open(
                f"models/demos/t3000/llama2_70b/demo/data/{model_args.llama_version}_demo_user_output.json", "w"
            ) as f:  # Open a file for writing
                output_json = json.dumps(all_text, indent=4)
                f.write(output_json)

    # Check against ground truth
    if data_args.ground_truth:
        scores = string_similarity_score(ground_truth_outputs, all_text)

        match = sum(scores) == len(scores)
        if not match:
            incorrect_indices = [i for i, score in enumerate(scores) if score < 1]
            logger.info(f"Output does not match ground truth at indices {incorrect_indices}")
            for idx in incorrect_indices:
                print(f"User {idx}: \n\tBad Output: {all_text[idx]}")

            assert match, "Output must match ground truth!"

        logger.info("Output matches ground truth!")


def build_generator(model_args, tt_args):
    generator = Llama.build(
        ckpt_dir=model_args.ckpt_dir,
        tokenizer_path=model_args.tokenizer_path,
        max_seq_len=model_args.max_seq_len,
        max_batch_size=model_args.max_batch_size,
        skip_model_load=model_args.skip_model_load,
        n_layers=1 if model_args.implementation == "tt" else model_args.num_layers,
    )

    state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

    if model_args.implementation == "tt":
        generator.model = TtLlamaModelForGeneration(
            configuration=generator.model.params,
            state_dict=state_dict,
            model_args=model_args,
            tt_args=tt_args,
        )
    return generator


def get_sampling_func(top_k, top_p, temperature):
    if top_k == 1:
        return lambda x: torch.argmax(x, dim=-1).reshape(-1)  # TODO: remove :, -1 since outer code already does that
    else:
        return lambda x: top_pk_logits_efficient(x, p=top_p, k=top_k, temperature=temperature).reshape(-1)


def load_prompts_file(model_args, data_args, tokenizer):
    # Load prompts from json
    prompts = json.load(open(data_args.prompts_file))
    # Encode the prompt
    if data_args.chat:
        formatter = ChatFormat(tokenizer)
        tokenized = [formatter.encode_dialog_prompt(dialog) for dialog in prompts]
    else:
        tokenized = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    if len(tokenized) > model_args.max_batch_size:
        logger.info(
            f"Warning: prompts file contains {len(tokenized)} prompts, but max batch size is {model_args.max_batch_size}. Only first {model_args.max_batch_size} are decoded."
        )
        tokenized = tokenized[: model_args.max_batch_size]
        prompts = prompts[: model_args.max_batch_size]

    return tokenized, prompts


def initialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu").clone().detach()
    eos_reached = torch.tensor([False] * bsz, device="cpu")
    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask, eos_reached


def prepare_next_input(tokenizer, tokens, input_text_mask, finished_mask, prompt_lens, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token

    eos_reached = (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
    prev_pos = cur_pos

    return tokens, eos_reached, prev_pos


def run_decode(
    model_args,
    tt_args,
    data_args,
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

    prompt_lens = [len(t) for t in prompt_tokens]
    min_prompt_len = min(prompt_lens) if not tt_args.decode_only else 1
    max_prompt_len = max(prompt_lens)
    assert max_prompt_len <= model_args.max_kv_context_len
    total_len = min(model_args.max_kv_context_len, max_prompt_len + output_tokens)
    assert total_len <= model_args.max_kv_context_len
    if total_len != max_prompt_len + output_tokens:
        logger.warning(
            f"Requested more output tokens than allowed by model. Truncating to {total_len - max_prompt_len} output tokens."
        )

    # prepare inputs
    tokens, input_text_mask, finished_mask = initialize_inputs(tokenizer, prompt_tokens, bsz, total_len)
    prev_pos = 0

    # some profiling and logging
    latencies = []
    full_logits = []

    for cur_pos in range(min_prompt_len, total_len):
        start = time()
        input_tokens = tokens[:, prev_pos:cur_pos]
        logits = model.forward(input_tokens, prev_pos)

        next_logits = logits[:, -1, :]  # batch, vocab of last token
        next_token = sampling_func(next_logits)

        tokens, eos_reached, prev_pos = prepare_next_input(
            tokenizer, tokens, input_text_mask, finished_mask, prompt_lens, cur_pos, next_token
        )
        latencies.append(time() - start)

        if all(eos_reached):
            break

        # Decode the entire sequence generated so far and log it
        for user_id in range(max(0, bsz - 3), bsz):
            text = tokenizer.decode(tokens[user_id, : cur_pos + 1].tolist())
            logger.info(f"Loop {cur_pos} user {user_id}: {text}\n")

        if return_full_logits:
            full_logits.append(logits.clone().detach())

    latency_printout(latencies, model_args, total_len - min_prompt_len)
    output = get_all_text(tokenizer, tokens, prompt_tokens, output_tokens)

    if return_logits:
        output = (output, logits)
    elif return_full_logits:
        full_logits = torch.cat(full_logits, dim=1)
        output = (output, full_logits)
    return output


def latency_printout(latencies, model_args, generated_len):
    latencies = [
        latency for token_pos, latency in enumerate(latencies) if token_pos % 32 != 0
    ]  # We recompute program_cache for multiples of 32
    overall_time = sum(latencies)
    overall_tokens = model_args.max_batch_size * len(latencies)
    warmup_batch = 2
    # Skip initial warmup batch
    if len(latencies) > warmup_batch:
        overall_time -= sum(latencies[:warmup_batch])
        overall_tokens -= warmup_batch * model_args.max_batch_size
        latencies = latencies[warmup_batch:]

    mean_latency = sum(latencies) / len(latencies) if len(latencies) > 0 else 0

    tokens_per_second = 1 / mean_latency if mean_latency != 0 else 0
    overall_tokens_per_second = overall_tokens / overall_time if overall_time != 0 else 0
    tokens_per_second_per_user = (
        overall_tokens_per_second / model_args.max_batch_size if model_args.max_batch_size != 0 else 0
    )
    throughput = 1000 * overall_time / overall_tokens if overall_tokens != 0 else 0

    logger.info(f"Overall throughput: {throughput:.1f} ms @ {overall_tokens_per_second:.1f} tokens/s")
    logger.info(f"Tokens per second per user: {tokens_per_second_per_user:.1f} tokens/s/u")
    logger.info(f"User latency: {1000 * mean_latency:.1f} ms @ {tokens_per_second:.1f} tokens/s")


def get_all_text(tokenizer, tokens, prompt_tokens, max_gen_len):
    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        try:
            # cut to max gen len
            start = 0
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        except IndexError:
            logger.info(f"Index out of range for sequence {i}, returning entire sequence.")
            pass

        # cut to eos tok if any
        if tokenizer.eos_id in toks:
            eos_idx = toks.index(tokenizer.eos_id)
            toks = toks[:eos_idx]
        out_tokens.append(toks)

    all_text = [tokenizer.decode(toks) for toks in out_tokens]
    return all_text


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


@pytest.mark.timeout(240000)
@pytest.mark.parametrize(
    "llama_version",
    (("llama2"),),
)
@pytest.mark.parametrize(
    "chat, prompts_file",
    (
        (True, "models/demos/t3000/llama2_70b/demo/data/multi_prompt_chat.json"),
        (False, "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json"),
    ),
    ids=("chat_completion", "text_completion"),
)
@pytest.mark.parametrize("decode_only", (True, False), ids=("decode_only", "prefill_decode"))
@pytest.mark.parametrize("num_layers", (1, 2, 10, 80), ids=("1L", "2L", "10L", "80L"))
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices",
    (
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
    ),
    ids=("tt-70b-T3000", "meta-70b"),
)
@pytest.mark.parametrize(
    "max_output_tokens, output_at_end, top_p, top_k, temperature",
    (
        (128, True, 1, 1, 1.0),
        (128, True, 0.9, 10, 1.0),
    ),
    ids=("greedy", "sampling"),
)
@pytest.mark.parametrize(
    "ground_truth",
    ("models/demos/t3000/llama2_70b/demo/data/llama2_ground_truth.json", None),
    ids=("check_enabled", "check_disabled"),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    ((32, 2048), (16, 8192), (1, 128 * 1024)),
    ids=("short_context", "long_context", "128k_context"),
)
def test_LlamaModel_demo(
    # model args
    implementation,
    skip_model_load,
    num_layers,
    # Generation args
    max_output_tokens,
    prompts_file,
    output_at_end,
    top_p,
    top_k,
    temperature,
    chat,
    # TT args
    t3k_mesh_device,
    n_devices,
    decode_only,
    llama_version,
    ground_truth,
    max_batch_size,
    max_context_len,
    use_program_cache,
):
    logger.info("Running LlamaModel demo")
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
        max_batch_size=max_batch_size,
        max_kv_context_len=max_context_len,
        max_output_tokens=max_output_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        chat=chat,
        mesh_device=t3k_mesh_device,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=decode_only,
        ground_truth=ground_truth,
    )
    main(args)
