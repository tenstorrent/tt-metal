# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel (TP=2) end-to-end PCC for BGE-M3.

Targets the long-context serving shape (batch 12, seq_len 8192) running on a
1x2 N300 mesh (mesh shape (2, 1)). Attention shards Q/K/V by heads and MLP is
row/column parallel; both stages all-reduce their output projection. The final
hidden state is replicated across the two chips, so we compare one replica
against the HF reference. Gated at PCC_THRESHOLD=0.94 to match the single-chip
gate in test_model.py.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_ID = "BAAI/bge-m3"
PCC_THRESHOLD = 0.93

# TP=2 serving shape.
TP_BATCH_SIZE = 12
TP_SEQ_LEN = 8192


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


def _ids_to_seqsharded(input_ids, mesh_device):
    # Shard on the sequence dim (tensor dim 1) across the 2-chip mesh.
    return ttnn.from_torch(
        input_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["tp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_model_tp2_b12_s8192(mesh_device, model_artifacts, reset_seeds):
    """End-to-end HF-vs-TT PCC for the TP=2 B12/S8192 shape on a 1x2 N300."""
    assert tuple(mesh_device.shape) == (2, 1), "TP=2 test requires a (2, 1) mesh"
    assert mesh_device.get_num_devices() == 2, "TP=2 test requires exactly 2 chips"

    backbone, state_dict, model_id_or_path = model_artifacts

    model_args, tt_model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=TP_BATCH_SIZE,
        max_seq_len=TP_SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        hf_model_name=model_id_or_path,
    )

    torch.manual_seed(42)
    input_ids = torch.randint(low=0, high=model_args.vocab_size, size=(TP_BATCH_SIZE, TP_SEQ_LEN), dtype=torch.long)
    non_pad_token_id = (int(model_args.pad_token_id) + 1) % model_args.vocab_size
    input_ids[input_ids == model_args.pad_token_id] = non_pad_token_id
    token_type_ids = torch.zeros_like(input_ids)
    # RoBERTa-style absolute position ids (host-computed over the full sequence;
    # the sequence-parallel path forbids device cumsum over a sharded sequence).
    pad = int(model_args.pad_token_id)
    nonpad = (input_ids != pad).to(torch.long)
    position_ids = (torch.cumsum(nonpad, dim=1) * nonpad + pad).to(torch.long)

    with torch.no_grad():
        reference_output = (
            backbone(
                input_ids=input_ids,
                attention_mask=None,
                token_type_ids=token_type_ids,
                position_ids=None,
                return_dict=True,
            )
            .last_hidden_state.unsqueeze(1)
            .to(torch.float32)
        )

    tt_output = tt_model.forward(
        input_ids=_ids_to_seqsharded(input_ids, mesh_device),
        attention_mask=None,
        token_type_ids=_ids_to_seqsharded(token_type_ids, mesh_device),
        position_ids=_ids_to_seqsharded(position_ids, mesh_device),
    )

    # Output is sharded on the sequence dim (tensor dim 2) across mesh axis 0.
    # Concatenate the two shards to recover the full [B, 1, S, D].
    # mesh axis 0 (size 2, the shard axis) -> tensor dim 2 (sequence);
    # mesh axis 1 (size 1) -> tensor dim 3 (no-op concat of a single shard).
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(2, 1))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=composer).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(TP_BATCH_SIZE, 1, TP_SEQ_LEN, model_args.dim)

    passing, msg = comp_pcc(reference_output, tt_output_torch, PCC_THRESHOLD)
    print(f"GATE_PCC_TP2 bf8 B{TP_BATCH_SIZE} S{TP_SEQ_LEN} = {msg}")
    assert passing, f"TP=2 PCC gate failed: {msg}"
