# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Full end-to-end PCC parity test for Janus-Pro.

Unlike ``test_e2e_hybrid`` (TT vision + torch decoder), here the WHOLE pipeline
runs on device: TT vision tower (encoder + aligner) -> host masked_scatter fusion
-> TT LLaMA-style decoder -> LM head. The next-token logits are compared against a
full HF ``JanusForConditionalGeneration`` forward.

Because the pipeline compounds vision (bf16), aligner, 24 decoder layers (bf8) and
the LM head, the PCC bar is lower than the per-stage tests (vision ~0.95, decoder
~0.99). The threshold below is a starting point; tune it from the first real run.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.janus_pro.tt.janus_pro_e2e_model import TtJanusProModel
from models.experimental.janus_pro.tt.model_config import ModelArgs


def _round_up(x, m):
    return ((x + m - 1) // m) * m


@torch.no_grad()
@pytest.mark.timeout(1200)  # full 7B decoder + vision tower + first-run compile exceed the 300s default
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_e2e(mesh_device, dummy_weights, reset_seeds, ensure_gc):
    pcc_required = 0.95
    dtype = ttnn.bfloat8_b  # decoder dtype (7B must fit); vision tower stays bf16 inside the model

    # ---- inputs: random image + a prompt with exactly num_image_tokens placeholders ----
    bsz = 1
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=1024, dummy_weights=dummy_weights, cache_hf=True)
    in_channels = model_args.vision_in_channels
    image_size = model_args.vision_chunk_size
    num_image_tokens = model_args.mm_tokens_per_image

    hf_model = model_args.reference_vision_transformer(wrap=False)
    hf_model.eval()
    image_token_id = hf_model.config.image_token_id

    pixel_values = torch.rand((bsz, in_channels, image_size, image_size))

    text_prefix = torch.tensor([[1, 100, 200]], dtype=torch.long)
    image_block = torch.full((bsz, num_image_tokens), image_token_id, dtype=torch.long)
    text_suffix = torch.tensor([[300, 400]], dtype=torch.long)
    input_ids = torch.cat([text_prefix, image_block, text_suffix], dim=1)  # [1, 581]
    real_len = input_ids.shape[1]
    last_token_idx = real_len - 1
    assert int((input_ids == image_token_id).sum()) == num_image_tokens

    # ---- reference (full torch HF): vision + aligner + masked_scatter + decoder + lm head ----
    ref_logits = hf_model(input_ids=input_ids, pixel_values=pixel_values.float()).logits
    ref_last = ref_logits[0, last_token_idx, :]

    # ---- TT model: pad the prompt so prefill/get_last_token align ----
    # The prefill sequence length must be a multiple of prefill_len_cutoff (512 on
    # Blackhole, 1024 on Wormhole): the MLP reshapes activations into
    # [1, seq_len // prefill_len_cutoff, prefill_len_cutoff, -1], which is only valid
    # when seq_len divides evenly. This also satisfies attention's multiple-of-128 rule.
    padded_len = _round_up(real_len, model_args.prefill_len_cutoff)
    assert padded_len <= model_args.max_seq_len
    input_ids_padded = torch.zeros((bsz, padded_len), dtype=torch.long)
    input_ids_padded[:, :real_len] = input_ids

    state_dict = model_args.load_state_dict()
    tt_model = TtJanusProModel(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        vision_dtype=ttnn.bfloat16,
    )

    tt_input, rot_mats_global, rot_mats_local, tt_page_table, *_ = tt_model.prepare_inputs_prefill(
        input_ids_padded, pixel_values=pixel_values
    )

    get_last_token = (last_token_idx // 32) * 32
    tt_out = tt_model.ttnn_prefill_forward(
        tt_input,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        user_id=0,
        page_table=tt_page_table,
        get_last_token=get_last_token,
    )

    # tt_out: [1, 1, 32, vocab] (norm + lm head already applied on device); pick the last token's row.
    tt_logits = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    row = last_token_idx - get_last_token
    tt_last = tt_logits[0, 0, row, : model_args.vocab_size]

    passing, pcc_message = comp_pcc(ref_last, tt_last, pcc_required)
    logger.info(comp_allclose(ref_last, tt_last))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required}. Check Warnings!"
