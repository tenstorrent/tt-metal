# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This file is a unit test for validating the Mistral-24B Text Decoder path.
"""

import os
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.mistral_24b.tests.pipeline_tests.test_end2end import setup_vision_model_args
from models.experimental.mistral_24b.tt.model import MistralTransformer as Transformer
from models.experimental.mistral_24b.tt.generator import MistralGenerator

# Mesh trace region (bytes) by architecture.
TRACE_REGION_SIZE_WORMHOLE = 30_000_000  # 30 MiB
TRACE_REGION_SIZE_BLACKHOLE = 35_000_000  # 35 MiB


def fabric_1d_trace_device_params(*, num_command_queues: int = 1):
    from models.common.utility_functions import is_wormhole_b0

    trace_region_size = TRACE_REGION_SIZE_WORMHOLE if is_wormhole_b0() else TRACE_REGION_SIZE_BLACKHOLE
    return [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": trace_region_size,
            "num_command_queues": num_command_queues,
        }
    ]


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P150x4": (1, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    fabric_1d_trace_device_params(num_command_queues=2),  # Arch-adaptive trace region: 30 MiB WH / 35 MiB BH.
    indirect=True,
)
def test_text_decoder(mesh_device):
    pcc_required = 0.99
    dtype = ttnn.bfloat8_b

    weights = "instruct"
    max_seq_len = 1024 * 8
    batch_size = 1
    page_params = {"page_block_size": 32, "page_max_num_blocks": 1024}
    optimizations = lambda margs: DecodersPrecision.accuracy(margs.n_layers, margs.model_name)

    model_args, _ = setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations)

    ##### Reference model output (Torch) #####
    reference_model = model_args.reference_transformer(wrap=False, load_checkpoint=True)
    reference_model.eval()

    # ##### TT Model: MistralGenerator (text decode) #####
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    state_dict = model_args.load_state_dict()
    text_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size,
        paged_attention_config.max_num_blocks // model_args.max_batch_size,
    )

    generator = MistralGenerator([text_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
    tt_kv_cache = [[layer.attention.layer_past for layer in text_model.layers]]

    # Real single-token decode input — first token of a text prompt.
    prompt_text = "The capital of France is"
    encoded = model_args.tokenizer.encode(prompt_text)
    out_tok = torch.full((batch_size,), encoded[0], dtype=torch.long)
    current_pos = torch.zeros(batch_size, dtype=torch.long)

    # Reference logits at the same (token, position) as the decode step below.
    ref_hidden = reference_model.model(
        input_ids=out_tok.view(batch_size, 1),
        position_ids=current_pos.view(batch_size, 1),
    )[0]
    ref_logits = reference_model.lm_head(ref_hidden[:, -1, :])[:, : model_args.vocab_size].float()

    decode_logits_out = generator.decode_forward(
        out_tok,
        current_pos,
        enable_trace=True,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )
    tt_logits, _ = decode_logits_out
    tt_logits_torch = tt_logits.reshape(batch_size, -1)[:, : model_args.vocab_size]

    passing, pcc_message = comp_pcc(ref_logits, tt_logits_torch, pcc_required)

    logger.info(comp_allclose(ref_logits, tt_logits_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}. {pcc_message}"
