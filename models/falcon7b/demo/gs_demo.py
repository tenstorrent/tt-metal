# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import tt_lib
import torch
from loguru import logger

from transformers import AutoTokenizer

from models.falcon7b.tt.falcon_causallm import TtFalconCausalLM

from models.falcon7b.reference.hf_modeling_falcon import FalconForCausalLM
from models.falcon7b.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, nearest_32
import time

END_OF_TEXT = 11
SPACE = 204

def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids


def test_gs_demo_kv():
    torch.manual_seed(0)

    tt_lib.program_cache.enable()

    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    start = time.time()
    model_version = "tiiuae/falcon-7b-instruct"
    model_config = get_model_config("BFLOAT16-DRAM")
    tt_cache_path = get_tt_cache_path(model_version)

    batch_size = 32
    num_layers = 32
    num_max_tokens = 192
    max_input_tokens = 32
    max_seq_len = nearest_32(num_max_tokens)

    input_prompts = [
        "write a short poem about London in English",
        "write a short poem about Madrid in Spanish",
        "write a short poem about Paris in French",
    ]

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_version)

    logger.info("Tokenizing inputs")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=max_input_tokens, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    tokenized_inputs_nopad = tokenizer(
        input_prompts, padding=False, max_length=max_input_tokens, add_special_tokens=False, return_tensors="pt"
    )

    num_users = len(tokenized_inputs_nopad["input_ids"])
    num_input_tokens = len(tokenized_inputs_nopad["input_ids"][0])
    for input_prompt in tokenized_inputs_nopad["input_ids"]:
        assert len(input_prompt) == num_input_tokens
    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens: {num_input_tokens}")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_version)
    hugging_face_reference_model.eval()

    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    post_processor = partial(post_process)

    # Prepare input ------------------------------------------------------------------------
    base_url = ""
    max_position_embeddings = max_seq_len
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True

    logger.info("Creating TT Model")
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )
    logger.info("Created TT Model")

    logger.info("Setting up inputs and attention masks")

    kv_cache = ()
    k_cache = torch.zeros(batch_size, 1, max_position_embeddings, head_dim)
    v_cache = torch.zeros(batch_size, 1, max_position_embeddings, head_dim)
    for _ in range(num_layers):
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        kv_cache += ((tt_k_cache, tt_v_cache),)

    # PREFILL
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    for user_id in range(num_users):
        logger.info(f"Falcon prefill for user {user_id} only")

        (
            tt_prefill_embeddings,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("prefill", prefill_ids[user_id:user_id+1], 0, num_input_tokens=num_input_tokens)
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_prefill_embeddings,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache,
            layer_past_len=0,
            use_cache=use_cache,
        )
        tt_prefill_embeddings.deallocate()
        if tt_prefill_attention_mask is not None:
            tt_prefill_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()
        user_output_ids = post_processor(logits=logits, index=num_input_tokens-1)
        output_ids[user_id] = user_output_ids
    logger.info("finished prefill stage")

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for output_prompt in output_prompts:
        logger.info(f"output::: {output_prompt}")

    zeroed_out_kv_cache = ()
    for tt_k_cache, tt_v_cache in kv_cache:
        k_cache = tt2torch_tensor(tt_k_cache)
        v_cache = tt2torch_tensor(tt_v_cache)
        k_cache[:, :, num_input_tokens:] = 0
        v_cache[:, :, num_input_tokens:] = 0
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        zeroed_out_kv_cache += ((tt_k_cache, tt_v_cache),)
    kv_cache = zeroed_out_kv_cache

    kv_cache_len = num_input_tokens  # This will increment by one after each decode

    end_prefill = time.time()
    logger.info(f"Prefill Run Time: {round((end_prefill - start), 2)}")

    # DECODE
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    prompt_is_done = [False] * num_users
    for output_token_index in range(num_max_tokens - num_input_tokens):
        decode_start = time.time()

        logger.info(f"Falcon decode token {output_token_index} for {batch_size} users")
        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
        )
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_decode_embeddings.deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

        decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)

        for user_id, user_decode_id in enumerate(decode_ids[:num_users]):
            if user_decode_id == END_OF_TEXT:
                prompt_is_done[user_id] = True
            if prompt_is_done[user_id]:
                decode_ids[user_id] = SPACE

        if all(prompt_is_done):
            break

        generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
        kv_cache_len += 1

        output_prompts = tokenizer.batch_decode(generated_ids.tolist())
        for output_prompt in output_prompts:
            logger.info(f"output::: {output_prompt}")

        decode_end = time.time()
        logger.info(f"Decode #{output_token_index} Run Time: {round((decode_end - decode_start), 2)}")

    end = time.time()
    logger.info(f"Total Model Run Time: {round((end - start), 2)}")

    output_prompts = tokenizer.batch_decode(generated_ids.tolist())

    tt_lib.device.CloseDevice(device)
    device = None

    for input_prompt, output_prompt in zip(input_prompts, output_prompts):
        logger.info(f"input: {input_prompt}")
        logger.info(f"output: {output_prompt}")

    tt_lib.program_cache.disable_and_clear()
