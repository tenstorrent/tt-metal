# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full-depth text prefill last-token logits PCC for Mistral-Small-3.1-24B.

Modeled on ``models/tt_transformers/tests/test_model_prefill.py``: real tokens from
*Tale of Two Cities*, all 40 decoder layers + norm + lm_head, sweeping ``seq_len`` up
to the model context window. HF reference calls ``language_model`` + ``lm_head`` directly.
"""

import bz2
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole
from models.experimental.mistral_24b.tests.pipeline_tests.test_end2end import (
    fabric_1d_trace_device_params,
    setup_vision_model_args,
)
from models.experimental.mistral_24b.tt.generator import MistralGenerator
from models.experimental.mistral_24b.tt.model import MistralTransformer as Transformer
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.model_config import DecodersPrecision

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)
PROMPT_FILE = os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/tale-of-two-cities.txt.bz2")


def scale_page_params(page_params, seq_len, batch_size):
    """Scale paged KV blocks so prefill of ``seq_len`` tokens fits."""
    block_size = page_params["page_block_size"]
    num_blocks = max(page_params["page_max_num_blocks"], -(-seq_len // block_size))
    num_blocks = -(-num_blocks // batch_size) * batch_size
    return {"page_block_size": block_size, "page_max_num_blocks": num_blocks}


def hf_prefill_last_token_logits(reference_model, inputs_embeds, vocab_size):
    """Text-only HF prefill: language_model + lm_head, return last-token logits."""
    batch_size, seq_len, _ = inputs_embeds.shape
    position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1)
    outputs = reference_model.model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    logits = reference_model.lm_head(outputs.last_hidden_state)
    return logits[:, -1:, :vocab_size]


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "seq_len",
    (1024,),
    # (128, 256, 1024, 3072, 4096, 8192, 16384, 32768, 65536, 131072),
    ids=["1k"],
    # ids=["128", "256", "1k", "3k", "4k", "8k", "16k", "32k", "64k", "128k"],
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128 * 1024,),
    ids=["max128k"],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "device_params",
    fabric_1d_trace_device_params(num_command_queues=2),
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P150x4": (1, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_text_prefill_logits(seq_len, max_seq_len, batch_size, page_params, mesh_device, reset_seeds, is_ci_env):
    """Prefill last-token logits PCC vs HF for increasing sequence lengths (full 40-layer depth)."""
    if is_ci_env and seq_len != 1024:
        pytest.skip("CI runs prefill logits at seq_len=1024 only.")

    pcc_required = 0.97
    dtype = ttnn.bfloat8_b

    page_params = scale_page_params(page_params, seq_len, batch_size)
    logger.info(
        f"Paged KV cache: {page_params['page_max_num_blocks']} blocks x "
        f"{page_params['page_block_size']} (seq_len={seq_len})"
    )

    optimizations = lambda margs: DecodersPrecision.accuracy(margs.n_layers, margs.model_name)
    model_args, instruct = setup_vision_model_args("instruct", max_seq_len, batch_size, mesh_device, optimizations)

    with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
        prompt = f.read()
    encoded_prompt = model_args.encode_prompt(prompt, instruct=instruct)[:seq_len]
    assert len(encoded_prompt) == seq_len, f"Prompt corpus shorter than seq_len={seq_len}"
    logger.info(f"Prefill length: {seq_len} tokens")

    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    reference_model = model_args.reference_transformer(wrap=False, load_checkpoint=True)
    reference_model.eval()
    embd = model_args.reference_embedding()
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )

    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [[layer.attention.layer_past for layer in tt_model.layers]]
    generator = MistralGenerator([tt_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)

    encoded_prompt_tensor = torch.tensor(encoded_prompt)
    tt_prefill_input = encoded_prompt_tensor.unsqueeze(0)

    logger.info("Running TT prefill...")
    tt_prefill_logits = generator.prefill_forward_text(
        tt_prefill_input,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[seq_len],
    )

    logger.info("Running HF prefill reference...")
    pt_prefill = embd(encoded_prompt_tensor).view(batch_size, seq_len, -1)
    ref_logits = hf_prefill_last_token_logits(reference_model, pt_prefill, model_args.vocab_size)

    passing, pcc_val = comp_pcc(ref_logits, tt_prefill_logits[:, :, : model_args.vocab_size], pcc_required)
    logger.info(comp_allclose(ref_logits, tt_prefill_logits[:, :, : model_args.vocab_size]))
    logger.info(f"Prefill last-token logits PCC (seq_len={seq_len}): {pcc_val}")

    assert passing, f"Prefill logits PCC {pcc_val} below {pcc_required} for seq_len={seq_len}."
