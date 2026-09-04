# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.device import is_blackhole as ttnn_is_blackhole

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    require_single_device,
    to_torch,
    to_ttnn_ids,
)
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_ID = "BAAI/bge-m3"
PCC_THRESHOLD = 0.94

# bf8_b holds the 0.94 PCC gate at every sequence length on Blackhole. On
# Wormhole the SDPA reduction over very long sequences (S8192) accumulates more
# bf8 quantization error and falls below the gate (~0.90), so fall back to bf16
# there. Blackhole keeps bf8_b everywhere; Wormhole uses bf8_b up to 4096.
_BF8_MAX_SEQ_LEN_WORMHOLE = 4096


def _dtype_for(device, seq_len):
    if ttnn_is_blackhole(device) or seq_len <= _BF8_MAX_SEQ_LEN_WORMHOLE:
        return ttnn.bfloat8_b
    return ttnn.bfloat16


@pytest.fixture(scope="module")
def model_artifacts(model_location_generator):
    transformers = pytest.importorskip("transformers")
    model_id_or_path = str(model_location_generator(MODEL_ID, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
    ).eval()

    backbone = hf_model.roberta if hasattr(hf_model, "roberta") else hf_model
    state_dict = hf_model.state_dict()
    return backbone, state_dict, model_id_or_path


def _run_full_end_to_end(device, model_artifacts, batch_size, seq_len):
    """Shared body: end-to-end HF-vs-TT PCC for one (batch_size, seq_len).

    bf8_b on Blackhole at every length; on Wormhole bf8_b up to S4096 and bf16
    beyond (see _dtype_for). Gated at PCC_THRESHOLD=0.94.
    """
    require_single_device(device)
    backbone, state_dict, model_id_or_path = model_artifacts

    model_args, tt_model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dtype=_dtype_for(device, seq_len),
        state_dict=state_dict,
        hf_model_name=model_id_or_path,
    )
    torch.manual_seed(42)
    input_ids = torch.randint(low=0, high=model_args.vocab_size, size=(batch_size, seq_len), dtype=torch.long)
    non_pad_token_id = (int(model_args.pad_token_id) + 1) % model_args.vocab_size
    input_ids[input_ids == model_args.pad_token_id] = non_pad_token_id
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = None
    with torch.no_grad():
        reference_output = (
            backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=None,
                return_dict=True,
            )
            .last_hidden_state.unsqueeze(1)
            .to(torch.float32)
        )

    tt_output = tt_model.forward(
        input_ids=to_ttnn_ids(input_ids, device),
        attention_mask=attention_mask,
        token_type_ids=to_ttnn_ids(token_type_ids, device),
    )
    tt_output_torch = to_torch(tt_output, expected_shape=(batch_size, 1, seq_len, model_args.dim))

    assert_pcc(reference_output, tt_output_torch, PCC_THRESHOLD)


@pytest.mark.slow
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{s}" for s in SEQUENCE_LENGTHS])
def test_model_full_end_to_end(device, model_artifacts, seq_len, reset_seeds):
    """End-to-end HF-vs-TT PCC for batch 1 x all sequence lengths (bf8_b).

    This is the CI-facing variant: batch 1 only (9 runs) to stay within the CI
    time budget. The larger-batch sweep lives in
    `test_model_full_end_to_end_multibatch`. Gated at PCC_THRESHOLD=0.94.
    Filter combos with -k, e.g. `-k "S512"`.
    """
    _run_full_end_to_end(device, model_artifacts, batch_size=1, seq_len=seq_len)
