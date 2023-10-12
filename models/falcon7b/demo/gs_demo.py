# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import tt_lib
import torch
from loguru import logger

from transformers import AutoTokenizer

from models.falcon7b.tt.falcon_causallm import TtFalconCausalLM

from models.falcon7b.reference.hf_modeling_falcon import FalconConfig
from models.falcon7b.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import time

END_OF_TEXT = 11
SPACE = 204

def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids

def preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    tokenized_inputs_nopad = tokenizer(
        input_prompts, padding=False, max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )

    num_users = len(tokenized_inputs_nopad["input_ids"])
    num_input_tokens = len(tokenized_inputs_nopad["input_ids"][0])
    for input_prompt in tokenized_inputs_nopad["input_ids"]:
        assert len(input_prompt) == num_input_tokens
    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens

def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        kv_cache += ((tt_k_cache, tt_v_cache),)
    return kv_cache

def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def test_gs_demo_kv():
    torch.manual_seed(0)

    tt_lib.program_cache.enable()

    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    model_version = "tiiuae/falcon-7b-instruct"
    model_config = get_model_config("BFLOAT16-DRAM")
    tt_cache_path = get_tt_cache_path(model_version)

    batch_size = 32
    num_layers = 32
    max_seq_len = 256

    configuration = FalconConfig(
        **{
            "_name_or_path": "tiiuae/falcon-7b-instruct",
            "alibi": False,
            "apply_residual_connection_post_layernorm": False,
            "architectures": [
                "FalconForCausalLM"
            ],
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_falcon.FalconConfig",
                "AutoModel": "modeling_falcon.FalconModel",
                "AutoModelForCausalLM": "modeling_falcon.FalconForCausalLM",
                "AutoModelForQuestionAnswering": "modeling_falcon.FalconForQuestionAnswering",
                "AutoModelForSequenceClassification": "modeling_falcon.FalconForSequenceClassification",
                "AutoModelForTokenClassification": "modeling_falcon.FalconForTokenClassification"
            },
            "bias": False,
            "bos_token_id": 11,
            "eos_token_id": 11,
            "hidden_dropout": 0.0,
            "hidden_size": 4544,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "falcon",
            "multi_query": True,
            "new_decoder_architecture": False,
            "num_attention_heads": 71,
            "num_hidden_layers": 32,
            "num_kv_heads": 71,
            "parallel_attn": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 65024
        }
    )

    # State dict is needed for embeddings
    state_dict = {"transformer.word_embeddings.weight": torch.load(tt_cache_path / "embedding.pt")}

    logger.info("Loading TT model weights")
    base_url = ""
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
    )
    logger.info("Loaded TT model weights")

    input_prompts = [
        "write a short poem about London in English",
        "write a short poem about Madrid in Spanish",
        "write a short poem about Madrid in English",
        "write a short poem about Paris in English",
        "write a short poem about Paris in French",
        "write a short poem about Istanbul in English",
        "write a short poem about Shanghai in English",
        "write a short poem about Lagos in English",
        "what is the capital of USA? ",
        "what is the capital of Canada? ",
        "what is the capital of UK? ",
        "what is the capital of Germany? ",
        "what is the capital of France? ",
        "what is the capital of Japan? ",
        "what is the capital of India? ",
        "what is the capital of China? ",
        "what is the currency of Cuba? ",
        "what is the currency of Lebanon? ",
        "what is the currency of Brazil? ",
        "what is the currency of Australia? ",
        "what is the currency of Jamaica? ",
        "what is the currency of Egypt? ",
        "what is the currency of Uzbekistan? ",
        "what is the currency of Argentina? ",
        "describe the geographic location of London in UK",
        "describe the geographic location of Toronto in Canada",
        "describe the geographic location of Madrid in Spain",
        "describe the geographic location of Paris in France",
        "describe the geographic location of Rome in Italy",
        "describe the geographic location of Istanbul in Turkey",
        "describe the geographic location of Shanghai in China",
        "describe the geographic location of Lagos in Nigeria",
    ]

    logger.info("Tokenizing inputs")
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len)

    logger.info("Initializing KV cache")
    kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device)

    logger.info("Running prefill stage")
    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    for user_id in range(num_users):
        prefill_start = time.time()
        logger.info(f"Running prefill for user {user_id}")

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

        prefill_end = time.time()
        logger.info(f"Finished prefill for user {user_id} in {round((prefill_end - prefill_start), 2)}s")

    logger.info("Finished prefill stage")

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)
    print_output_prompts(generated_ids, tokenizer, 3)

    logger.info("Running decode stage")
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]
    for output_token_index in range(max_seq_len - num_input_tokens):
        logger.info(f"Running decode for token {output_token_index}")
        decode_start = time.time()

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

        print_output_prompts(generated_ids, tokenizer, 3)

        decode_end = time.time()
        logger.info(f"Finished decode for token {output_token_index} in {round((decode_end - decode_start), 2)}")

    logger.info("Finished decode stage")

    print_output_prompts(generated_ids, tokenizer)

    tt_lib.device.CloseDevice(device)
    tt_lib.program_cache.disable_and_clear()
