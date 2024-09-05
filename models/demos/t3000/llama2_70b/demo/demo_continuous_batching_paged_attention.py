# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from queue import Queue
from dataclasses import dataclass
import os
import json
import torch
import torch.nn.functional as F

from time import time
import pytest
from loguru import logger
import ttnn
from ttnn import ReplicateTensorToMesh

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
class PagedAttentionConfig:
    block_size = 64
    max_num_blocks = 2048


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
    paged_attention_config = PagedAttentionConfig()

    generator = build_generator(model_args, tt_args, paged_attention_config)

    # Load the model and tokenizer
    model, tokenizer = generator.model, generator.tokenizer

    tokenized, prompts = load_prompts_file(model_args, data_args, tokenizer)

    # Run decode
    with torch.no_grad():
        all_text = run_decode(
            model_args,
            tt_args,
            data_args,
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=tokenized,
            prompts=prompts,
            paged_attention_config=paged_attention_config,
        )

        if data_args.output_at_end:
            with open(
                f"models/demos/t3000/llama2_70b/demo/data/{model_args.llama_version}_demo_user_output.json", "w"
            ) as f:  # Open a file for writing
                output_json = json.dumps(all_text, indent=4)
                f.write(output_json)


def build_generator(model_args, tt_args, paged_attention_config=None):
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
            paged_attention_config=paged_attention_config,
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

    logger.info(f"Loaded {len(tokenized)} prompts from {data_args.prompts_file}")
    # if len(tokenized) > model_args.max_batch_size:
    #     logger.info(
    #         f"Warning: prompts file contains {len(tokenized)} prompts, but max batch size is {model_args.max_batch_size}. Only first {model_args.max_batch_size} are decoded."
    #     )
    #     tokenized = tokenized[: model_args.max_batch_size]
    #     prompts = prompts[: model_args.max_batch_size]

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


def initialize_prefill_input(tokenizer, prompt_tokens):
    padded_len = 128 if len(prompt_tokens) <= 128 else 2048
    assert padded_len >= len(prompt_tokens)

    tokens = torch.full((1, padded_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").clone().detach()

    return tokens, len(prompt_tokens)


def initialize_decode_input(token_inputs, batch_token_indices):
    tokens = torch.tensor(token_inputs, dtype=torch.long, device="cpu").unsqueeze(1)
    indices = torch.tensor(batch_token_indices, dtype=torch.long, device="cpu")
    return tokens, indices


def prepare_next_input(tokenizer, tokens, input_text_mask, finished_mask, prompt_lens, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token

    eos_reached = (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
    prev_pos = cur_pos

    return tokens, eos_reached, prev_pos


def is_batch_full(batch_valid):
    return all(batch_valid)


def is_batch_empty(batch_valid):
    return not any(batch_valid)


def run_decode(
    model_args,
    tt_args,
    data_args,
    model,
    tokenizer,
    prompt_tokens,
    prompts,
    paged_attention_config,
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

    # prompt_lens = [len(t) for t in prompt_tokens]
    # min_prompt_len = min(prompt_lens) if not tt_args.decode_only else 1
    # max_prompt_len = max(prompt_lens)
    # assert max_prompt_len <= model_args.max_kv_context_len
    # total_len = min(model_args.max_kv_context_len, max_prompt_len + output_tokens)
    # assert total_len <= model_args.max_kv_context_len
    # if total_len != max_prompt_len + output_tokens:
    #     logger.warning(
    #         f"Requested more output tokens than allowed by model. Truncating to {total_len - max_prompt_len} output tokens."
    #     )

    # # prepare inputs
    # tokens, input_text_mask, finished_mask = initialize_inputs(tokenizer, prompt_tokens, bsz, total_len)
    # prev_pos = 0

    # some profiling and logging
    latencies = []
    full_logits = []

    # intialize continuous batching data structures
    prompts_q = Queue()
    output_q = []
    for user_id, (p, t) in enumerate(zip(prompts, prompt_tokens)):
        prompts_q.put((user_id, p, t))

    # TODO: When I merge paged_update_cache, use -1. Until then, use 0
    batch_token_indices = [0 for _ in range(model_args.max_batch_size)]
    batch_valid = [False for _ in range(model_args.max_batch_size)]
    batch_token_inputs = [0 for _ in range(model_args.max_batch_size)]
    batch_prompt_text = [None for _ in range(model_args.max_batch_size)]
    batch_token_outputs = [None for _ in range(model_args.max_batch_size)]
    batch_user_ids = [None for _ in range(model_args.max_batch_size)]

    # breakpoint()
    MAX_GEN_LENGTH = 180

    # set up paged attention
    # first pass: create a static page table for all users
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    static_page_table = reverse_permutation.reshape(bsz, paged_attention_config.max_num_blocks // bsz)
    page_table_tt = ttnn.as_tensor(
        static_page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(model.mesh_device),
    )
    page_table_tt = ttnn.to_device(page_table_tt, model.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    while True:
        logger.info(f"Current batch valid: {batch_valid}")
        logger.info(f"Current batch token indices: {batch_token_indices}")
        if not is_batch_full(batch_valid) and not prompts_q.empty():
            # Find first invalid slot
            user_id, prompt_text, prompt_tokens = prompts_q.get()
            batch_idx = batch_valid.index(False)
            logger.info(f"Prefilling user into batch idx {batch_idx}")
            batch_token_outputs[batch_idx] = prompt_tokens
            prompt_tokens, prompt_len = initialize_prefill_input(tokenizer, prompt_tokens)

            logits = model.prefill_forward_single_user(prompt_tokens, 0, batch_idx, page_table=page_table_tt)
            # breakpoint() # are we getting the right shape back? are we indexing the right index?
            next_logits = logits[:, prompt_len - 1, :]  # 1, seq_len, vocab -> 1, vocab
            next_token = sampling_func(next_logits).item()  # shape = (1,)

            # Update datastructures
            batch_token_indices[batch_idx] = prompt_len
            batch_valid[batch_idx] = True
            batch_token_inputs[batch_idx] = next_token
            batch_prompt_text[batch_idx] = prompt_text
            batch_token_outputs[batch_idx].append(next_token)
            batch_user_ids[batch_idx] = user_id

        elif not is_batch_empty(batch_valid):
            # Decode iteration
            tokens_tensor, indices_tensor = initialize_decode_input(batch_token_inputs, batch_token_indices)
            # What is the shape of tokens_tensor???
            # breakpoint()
            logger.info(f"Decoding batch with indices {batch_token_indices}")
            # TODO: call model.decode? Is it switching model_configs correctly?
            logits = model.decode_forward(tokens_tensor, indices_tensor, page_table=page_table_tt)
            next_logits = logits[:, -1, :]  # batch, vocab of last token
            next_token = sampling_func(next_logits)

            # Update datastructures
            for i in range(len(batch_valid)):
                if batch_valid[i]:
                    batch_token_inputs[i] = next_token[i].item()
                    batch_token_outputs[i].append(next_token[i].item())
                    batch_token_indices[i] += 1

                    if batch_token_indices[i] > MAX_GEN_LENGTH:
                        user_id = batch_user_ids[i]
                        logger.info(
                            f"User {user_id} in batch slot {i} has reached max gen length. Removing from batch."
                        )
                        # TODO: Print user output or save it somewhere
                        user_text = tokenizer.decode(batch_token_outputs[i])
                        logger.info(f"User {user_id} output: {user_text}")
                        output_q.append((user_id, batch_prompt_text[i], user_text))
                        batch_valid[i] = False
                        batch_token_inputs[i] = 0
                        batch_token_indices[i] = 0
                        batch_token_outputs[i] = None
                        batch_prompt_text[i] = None
                        batch_user_ids[i] = None

        else:
            logger.info("All users have finished. Exiting.")
            break

    output_q.sort(key=lambda x: x[0])
    for out in output_q:
        user_id, prompt_text, user_text = out
        logger.info(f"User {user_id} prompt: {prompt_text}")
        logger.info(f"User {user_id} output: {user_text}")


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
    (("llama3"),),
)
@pytest.mark.parametrize(
    "chat, prompts_file",
    (
        (True, "models/demos/t3000/llama2_70b/demo/data/multi_prompt_chat.json"),
        (False, "models/demos/t3000/llama2_70b/demo/data/multi_prompt_large.json"),
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
    ),
    ids=("tt-70b-T3000",),
)
@pytest.mark.parametrize(
    "max_output_tokens, output_at_end, top_p, top_k, temperature",
    (
        (128, True, 1, 1, 1.0),
        # (128, True, 0.9, 10, 1.0),
    ),
    ids=("greedy",),  # "sampling"
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 2048),
        # (16, 8192),
    ),
    ids=(
        "short_context",
        # "long_context"
    ),
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
    )
    main(args)
