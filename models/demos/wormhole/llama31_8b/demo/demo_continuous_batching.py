# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from queue import Queue
from dataclasses import dataclass
import os
import json
import torch
import torch.nn.functional as F
import ttnn

from time import time
import pytest
from loguru import logger

from transformers.generation.utils import top_k_top_p_filtering
from models.demos.wormhole.llama31_8b.tt.llama_common import (
    prepare_inputs_ttnn,
    sample,
    get_rotation_mat_batched,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    encode_prompt_llama_instruct,
    HostEmbedding,
)
from models.demos.wormhole.llama31_8b.tt.llama_model import TtTransformer
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs


@dataclass
class TTArgs:
    tt_model_args: object = None
    device: object = None
    n_devices: int = 1
    decode_only: bool = False
    dtype: object = ttnn.bfloat8_b


@dataclass
class DataArgs:
    max_output_tokens: int = 128
    prompts_file: str = "models/demos/wormhole/llama31_8b/demo/input_data.json"
    output_at_end: bool = True
    top_p: float = 1
    top_k: int = 1
    temperature: float = 1.0
    instruct_mode: bool = False
    sample_len: int = None
    ground_truth: str = None


@dataclass
class DemoArgs:
    tt: TTArgs
    data: DataArgs


def construct_arg(**kwargs):
    tt_args = TTArgs(**{k: v for k, v in kwargs.items() if hasattr(TTArgs, k)})
    data_args = DataArgs(**{k: v for k, v in kwargs.items() if hasattr(DataArgs, k)})
    return DemoArgs(tt=tt_args, data=data_args)


def main(args):
    # Set random reproducible seed
    torch.manual_seed(0)

    model_args = args.tt.tt_model_args
    tt_args = args.tt
    data_args = args.data

    tokenizer = Tokenizer(model_args.tokenizer_path)

    input_prompts = load_inputs(data_args.prompts_file)

    input_prompts_tokens = preprocess_inputs(input_prompts, tokenizer, data_args.instruct_mode)

    tt_model, tt_embd, host_embd = load_tt_model(model_args, tt_args.device, tt_args.dtype)

    # Run decode
    with torch.no_grad():
        all_text = run_decode(
            model_args,
            tt_args,
            data_args,
            model=tt_model,
            tt_embed=tt_embd,
            host_embed=host_embd,
            tokenizer=tokenizer,
            prompt_tokens=input_prompts_tokens,
            prompts=input_prompts,
        )


# load from json, return as a list
def load_inputs(user_input):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    in_prompt = []
    for i in range(len(user_input)):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs(input_prompts, tokenizer, instruct):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

    return encoded_prompts


def load_tt_model(model_args, device, dtype):
    logger.info("Loading weights...")
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    logger.info("Loading weights finished!")

    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
    )
    tt_embd = TtLlamaEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    logger.info("Finished loading weights to device. Starting inference...")

    return tt_model, tt_embd, embd


def get_sampling_func(top_k, top_p, temperature):
    if top_k == 1:
        return lambda x: torch.argmax(x, dim=-1).reshape(-1)  # TODO: remove :, -1 since outer code already does that
    else:
        return lambda x: top_pk_logits_efficient(x, p=top_p, k=top_k, temperature=temperature).reshape(-1)


def initialize_prefill_input(tokenizer, prompt_tokens):
    padded_len = 128 if len(prompt_tokens) <= 128 else 2048
    assert padded_len >= len(prompt_tokens)

    tokens = torch.full((1, padded_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").clone().detach()

    return tokens, len(prompt_tokens)


def initialize_decode_input(token_inputs, batch_token_indices):
    tokens = torch.tensor(token_inputs, dtype=torch.long, device="cpu").unsqueeze(0)
    indices = torch.tensor(batch_token_indices, dtype=torch.long, device="cpu")
    return tokens, indices


def is_batch_full(batch_valid):
    return all(batch_valid)


def is_batch_empty(batch_valid):
    return not any(batch_valid)


def prepare_input_prefill(model_args, device, prompt_tokens, embed):
    prefill_seq_len = prompt_tokens.shape[-1]
    # embed input tokens
    prefill_input = embed(prompt_tokens).view(1, prefill_seq_len, -1)
    prefill_input = prepare_inputs_ttnn_prefill(
        prefill_input,
        device,
    )

    # rotary matrix
    rot_mats_prefill = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, device, seq_len=prefill_seq_len)

    # transformation matrix
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return prefill_input, rot_mats_prefill, transformation_mats


def prepare_input_decode(model_args, device, tokens_tensor, tt_embed, host_embed, indices_tensor):
    # embed input tokens
    tt_out_tok = torch.zeros(1, 32)
    tt_out_tok[:, : tokens_tensor.size(1)] = tokens_tensor
    if tt_embed is not None:
        tt_out_tok = ttnn.from_torch(
            tt_out_tok,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_decode_input = tt_embed(tt_out_tok)
    else:
        tt_decode_input = host_embed(tt_out_tok).view(1, model_args.max_batch_size, -1)
        tt_decode_input = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            device,
        )

    # cur pos tensor
    current_pos_tensor = ttnn.from_torch(torch.tensor(indices_tensor), device=device, dtype=ttnn.int32)

    # rotary matrix
    rot_cache_idxs = torch.maximum(
        indices_tensor, torch.tensor(0, dtype=torch.int64)
    )  # Ensure position indices are non-negative
    rot_mat = get_rotation_mat_batched(model_args.rot_emb, rot_cache_idxs, 1, batch=model_args.max_batch_size)
    assert rot_mat.size() == (1, model_args.max_batch_size, model_args.head_dim, model_args.head_dim)

    rot_mat = ttnn.as_tensor(
        rot_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    rot_mat = ttnn.to_device(rot_mat, device, memory_config=model_args.model_config["ROT_MAT_MEMCONFIG"])

    return tt_decode_input, current_pos_tensor, rot_mat


def run_decode(
    model_args,
    tt_args,
    data_args,
    model,
    tt_embed,
    host_embed,
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

    sampling_func = get_sampling_func(data_args.top_k, data_args.top_p, data_args.temperature)

    # intialize continuous batching data structures
    prompts_q = Queue()
    output_q = []
    for user_id, (p, t) in enumerate(zip(prompts, prompt_tokens)):
        # Put global user id, prompt text, and prompt tokens into queue
        prompts_q.put((user_id, p, t))

    """
    Datastructures for continuous batching
    Each datastructure is a list of size max_batch_size. The contents of each list are:
        batch_token_indices: token index
        batch_valid: slot is occupied/valid
        batch_token_inputs: token input (always a single token)
        batch_prompt_text: prompt text
        batch_token_outputs: token outputs (grows as the user generates more tokens)
        batch_user_ids: global user id
    """
    batch_token_indices = [0 for _ in range(model_args.max_batch_size)]
    batch_valid = [False for _ in range(model_args.max_batch_size)]
    batch_token_inputs = [0 for _ in range(model_args.max_batch_size)]
    batch_prompt_text = [None for _ in range(model_args.max_batch_size)]
    batch_token_outputs = [None for _ in range(model_args.max_batch_size)]
    batch_user_ids = [None for _ in range(model_args.max_batch_size)]

    MAX_GEN_LENGTH = data_args.max_output_tokens

    while True:
        logger.info(f"Current batch valid: {batch_valid}")
        logger.info(f"Current batch token indices: {batch_token_indices}")
        if not is_batch_full(batch_valid) and not prompts_q.empty():
            """
            Prefill Step:
                If the batch is not full and the prompts queue is not empty,
                we will prefill one user and insert it into the batch.
            """
            # Find first invalid slot
            user_id, prompt_text, prompt_tokens = prompts_q.get()
            batch_idx = batch_valid.index(False)
            logger.info(f"Prefilling user into batch idx {batch_idx}")
            batch_token_outputs[batch_idx] = prompt_tokens
            prompt_tokens, prompt_len = initialize_prefill_input(tokenizer, prompt_tokens)
            prefill_input, rot_mats_prefill, transformation_mats = prepare_input_prefill(
                model_args, tt_args.device, prompt_tokens, host_embed
            )
            tt_out = model(
                prefill_input,
                0,  # Current position
                None,
                rot_mats_prefill,
                transformation_mats,
                user_id=batch_idx,
                mode="prefill",
            )
            ttnn.deallocate(tt_out)
            # we avoid lm head in prefill because it is too slow
            # instead, we save the last token for first iteration of decode
            next_token = prompt_tokens.squeeze()[prompt_len - 1]
            # Update datastructures
            batch_token_indices[batch_idx] = prompt_len - 1
            batch_valid[batch_idx] = True
            batch_token_inputs[batch_idx] = next_token
            batch_prompt_text[batch_idx] = prompt_text
            batch_user_ids[batch_idx] = user_id

        elif not is_batch_empty(batch_valid):
            """
            Decode Step:
                If the batch is not empty, we have users in the decode batch
                to process. Do one decode iteration and then update datastructures.
            """
            # Decode iteration
            tokens_tensor, indices_tensor = initialize_decode_input(batch_token_inputs, batch_token_indices)
            decode_input, current_pos_tensor, current_rot_mat = prepare_input_decode(
                model_args, tt_args.device, tokens_tensor, tt_embed, host_embed, indices_tensor
            )
            logger.info(f"Decoding batch with indices {batch_token_indices}")
            tt_out = model(decode_input, current_pos_tensor, current_pos_tensor, rot_mat=current_rot_mat)
            tt_out_rm = ttnn.untilize(tt_out, use_multicore=True)
            ttnn.deallocate(tt_out)
            ttnn.deallocate(current_rot_mat)
            logits = ttnn.to_torch(tt_out_rm).squeeze(0)  # (1, 1, batch, vocab) -> (1, batch, vocab)
            next_logits = logits[-1, :, :]  # batch, vocab of last token
            next_token = sampling_func(next_logits)

            # Update datastructures
            for i in range(len(batch_valid)):
                if batch_valid[i]:
                    batch_token_inputs[i] = next_token[i].item()
                    batch_token_outputs[i].append(next_token[i].item())
                    batch_token_indices[i] += 1

                    if batch_token_indices[i] > MAX_GEN_LENGTH:
                        # In this demo, stop decoding only if the user has hit the maximum generation length
                        user_id = batch_user_ids[i]
                        logger.info(
                            f"User {user_id} in batch slot {i} has reached max gen length. Removing from batch."
                        )
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

    # Log all outputs
    output_q.sort(key=lambda x: x[0])
    for out in output_q:
        user_id, prompt_text, user_text = out
        logger.info(f"User {user_id} prompt: {prompt_text}")
        logger.info(f"User {user_id} output: {user_text}")


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
    "instruct_mode, prompts_file",
    (
        (True, "models/demos/wormhole/llama31_8b/demo/input_data_questions.json"),
        (False, "models/demos/wormhole/llama31_8b/demo/input_data.json"),
    ),
    ids=("chat_completion", "text_completion"),
)
@pytest.mark.parametrize("num_layers", (1, 8, 32), ids=("1L", "8L", "32L"))
@pytest.mark.parametrize(
    "max_output_tokens, output_at_end, top_p, top_k, temperature",
    ((128, True, 1, 1, 1.0),),
    ids=("greedy",),  # "sampling"
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    ((4, 8192),),
    ids=("batch_4",),
)
def test_LlamaModel_demo(
    # model args
    num_layers,
    # Generation args
    max_output_tokens,
    prompts_file,
    output_at_end,
    top_p,
    top_k,
    temperature,
    instruct_mode,
    # TT args
    device,
    llama_version,
    max_batch_size,
    max_context_len,
    use_program_cache,
):
    logger.info("Running LlamaModel demo")
    ## Get model config

    # Load model args, weights, and tokenizer
    tt_model_args = TtModelArgs(device, instruct=instruct_mode, max_batch_size=max_batch_size)
    tt_model_args.max_seq_len = max_context_len
    tt_model_args.kv_seq_len = max_context_len
    tt_model_args.sliding_window = max_context_len
    tt_model_args.n_layers = num_layers

    args = construct_arg(
        # Generation args
        max_output_tokens=max_output_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        instruct_mode=instruct_mode,
        # TT args
        n_devices=1,
        device=device,
        tt_model_args=tt_model_args,
        decode_only=False,
        dtype=ttnn.bfloat8_b,
    )
    main(args)
