# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

os.environ.setdefault("GEMMA4_MODEL_PATH", "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it")

from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig


def run_generation_with_generator(
    mesh_device,
    model_path,
    prompts,
    max_new_tokens=16,
    max_seq_len=1024,
    num_layers=None,
    enable_prefill_trace=False,
    enable_decode_trace=False,
):
    page_params = {"page_block_size": 64, "page_max_num_blocks": max_seq_len // 64}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
    )
    if enable_prefill_trace:
        generator.warmup_model_prefill(
            kv_cache=tt_kv_cache,
            enable_trace=True,
            can_sample_on_device=False,
            non_greedy_decoding_on_device=False,
        )
    if enable_decode_trace:
        generator.warmup_model_decode(
            kv_cache=tt_kv_cache,
            enable_trace=True,
            max_batch_size=1,
            num_blocks=paged_attention_config.max_num_blocks,
            can_sample_on_device=False,
            non_greedy_decoding_on_device=False,
            read_from_device=False,
        )

    generated = []
    for prompt in prompts:
        if tokenizer.chat_template:
            chat_result = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = chat_result["input_ids"].squeeze(0)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

        prompt_tokens = input_ids.unsqueeze(0)
        prompt_lens = torch.tensor([input_ids.shape[0]], dtype=torch.long)

        prefill_logits = generator.prefill_forward_text(
            tokens=prompt_tokens,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=prompt_lens,
            enable_trace=enable_prefill_trace,
        )
        out_tok = torch.argmax(prefill_logits, dim=-1).to(torch.int32)
        generated_tokens = [int(out_tok[0, 0].item())]
        current_pos = torch.tensor([input_ids.shape[0]], dtype=torch.int32)

        for iteration in range(max_new_tokens - 1):
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=enable_decode_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                reset_batch=(iteration == 0),
            )
            out_tok = torch.argmax(logits, dim=-1).to(torch.int32)
            token_id = int(out_tok[0, 0].item())
            generated_tokens.append(token_id)
            current_pos += 1
            if token_id == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logger.info("Prompt: {}", prompt)
        logger.info("Generated tokens: {}", generated_tokens)
        logger.info("Generated text: {}", generated_text)
        generated.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "generated_tokens": generated_tokens,
                "full_text": prompt + generated_text,
            }
        )

    return generated


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
def test_gemma4_generator_meaningful_output(mesh_device, model_path):
    results = run_generation_with_generator(
        mesh_device=mesh_device,
        model_path=model_path,
        prompts=["The capital of France is"],
        max_new_tokens=8,
        max_seq_len=1024,
        enable_prefill_trace=True,
        enable_decode_trace=True,
    )

    assert len(results) == 1
    generated_text = results[0]["generated_text"].strip()
    assert generated_text, "Generator returned no text"
    assert (
        "Paris" in generated_text or "capital" in generated_text
    ), f"Generated text is not meaningful enough: {generated_text!r}"
