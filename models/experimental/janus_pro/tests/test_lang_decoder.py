# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end prefill parity test for the Janus Pro language decoder.

Proves that the shared tt_transformers ``Transformer`` reproduces the HF
``model.language_model`` (a LLaMA-style ``LlamaModel``) when driven with Janus
Pro weights (HF_MODEL=deepseek-community/Janus-Pro-7B). A single prefill forward
through all decoder layers is compared to the HF reference by PCC on the final
hidden states (post output-norm, matching ``LlamaModel.last_hidden_state``).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer


@torch.no_grad()
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_lang_decoder_inference(seq_len, mesh_device, reset_seeds, ensure_gc):
    pcc_required = 0.99
    dtype = ttnn.bfloat8_b

    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max(seq_len, 512), cache_hf=True)
    state_dict = model_args.load_state_dict()

    # Reference: HF language_model (LlamaModel) restricted from the full Janus Pro model.
    reference_model = model_args.reference_language_model()
    reference_model.eval()

    # TT decoder stack: the shared tt_transformers Transformer, built from Janus weights.
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )
    logger.info(f"Model with {model_args.n_layers} layers loaded.")

    # One prefill forward with the same random tokens on both models.
    input_ids = torch.randint(0, model_args.vocab_size, (1, seq_len))

    tt_input, rot_mats_global, rot_mats_local, tt_page_table, *_ = tt_model.prepare_inputs_prefill(input_ids)
    tt_out = tt_model.ttnn_prefill_forward(
        tt_input,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        user_id=0,
        page_table=tt_page_table,
        get_last_token=-1,
    )
    # ttnn_prefill_forward returns pre-norm hidden states; apply the output norm to
    # match LlamaModel.last_hidden_state.
    tt_out = tt_model.norm(
        tt_out, mode=Mode.PREFILL, norm_config=model_args.get_norm_config("lm_head", Mode.PREFILL, tt_model.prefetcher)
    )
    tt_hidden = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        0, 0, :seq_len, : model_args.dim
    ]

    reference_output = reference_model(input_ids=input_ids, use_cache=False).last_hidden_state[0]

    passing, pcc_message = comp_pcc(reference_output, tt_hidden, pcc_required)
    logger.info(comp_allclose(reference_output, tt_hidden))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC {pcc_message} is below required {pcc_required}. Check Warnings!"
