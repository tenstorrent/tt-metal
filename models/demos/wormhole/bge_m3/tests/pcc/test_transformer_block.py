# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.wormhole.bge_m3.tt.encoder import BgeM3TransformerBlock

MODEL_ID = "BAAI/bge-m3"
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
MLP_SIZE = 4096
BATCH_SIZE = 1
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def first_layer_artifacts(model_location_generator):
    transformers = pytest.importorskip("transformers")
    model_id_or_path = model_location_generator(MODEL_ID, download_if_ci_v2=True, ci_v2_timeout_in_s=1800)

    hf_config = transformers.AutoConfig.from_pretrained(model_id_or_path)
    hf_config.num_hidden_layers = 1
    hf_model = transformers.AutoModel.from_pretrained(
        model_id_or_path,
        config=hf_config,
        torch_dtype=torch.bfloat16,
    ).eval()

    backbone = hf_model.roberta if hasattr(hf_model, "roberta") else hf_model
    assert backbone.config.hidden_size == HIDDEN_SIZE
    assert backbone.config.num_attention_heads == NUM_HEADS
    assert backbone.config.intermediate_size == MLP_SIZE

    hf_layer = backbone.encoder.layer[0]
    state_dict = {f"roberta.encoder.layer.0.{key}": value for key, value in hf_layer.state_dict().items()}
    args = SimpleNamespace(
        dim=HIDDEN_SIZE,
        n_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=MLP_SIZE,
        norm_eps=backbone.config.layer_norm_eps,
    )
    return hf_layer, state_dict, args


@pytest.mark.slow
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_transformer_block_vs_hf_first_layer(device, first_layer_artifacts, seq_len):
    require_single_device(device)
    hf_layer, state_dict, args = first_layer_artifacts

    torch.manual_seed(42)
    hidden_states = torch.randn((BATCH_SIZE, seq_len, HIDDEN_SIZE), dtype=torch.bfloat16)

    with torch.no_grad():
        reference_output = hf_layer(hidden_states=hidden_states, attention_mask=None)[0].unsqueeze(1).to(torch.float32)

    tt_block = BgeM3TransformerBlock(
        args=args,
        mesh_device=device,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
        layer_num=0,
    )
    tt_output = tt_block.forward(
        hidden_states=to_ttnn_tensor(hidden_states.unsqueeze(1), device),
        attention_mask=None,
    )
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    assert_pcc(reference_output, tt_output_torch, PCC_THRESHOLD)
