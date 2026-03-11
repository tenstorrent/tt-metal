# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
BATCH_SIZE = 1
PCC_THRESHOLD = 0.94


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


@pytest.mark.slow
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_model_full_end_to_end(device, model_artifacts, seq_len):
    require_single_device(device)
    backbone, state_dict, model_id_or_path = model_artifacts

    model_args, tt_model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=BATCH_SIZE,
        max_seq_len=seq_len,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
        hf_model_name=model_id_or_path,
    )

    torch.manual_seed(42)
    input_ids = torch.randint(low=0, high=model_args.vocab_size, size=(BATCH_SIZE, seq_len), dtype=torch.long)
    non_pad_token_id = (int(model_args.pad_token_id) + 1) % model_args.vocab_size
    input_ids[input_ids == model_args.pad_token_id] = non_pad_token_id
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = input_ids.ne(model_args.pad_token_id).long()

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
        attention_mask=to_ttnn_ids(attention_mask, device),
        token_type_ids=to_ttnn_ids(token_type_ids, device),
    )
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, model_args.dim))

    assert_pcc(reference_output, tt_output_torch, PCC_THRESHOLD)
