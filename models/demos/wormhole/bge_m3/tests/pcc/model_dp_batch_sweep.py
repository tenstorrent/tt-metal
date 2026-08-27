# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-swept DP=2 end-to-end HF-vs-TT PCC for BGE-M3 (batch-scaling campaign).

Parameterized over local batch and variant via env vars so the driver can run one
(batch, variant, seed) per subprocess:

  BGE_LOCAL_BATCH   local batch per chip (1,2,3,4,6); global = 2*local
  BGE_VARIANT       stock_dram | jit_dram | jit_l1_<name>
  BGE_PCC_SEED      seed (default 42)
  BGE_L1_HANDOFF    (jit_l1 only) handoff name -> exported for the model path

Prints:  GATE_PCC_SWEEP b=<lb> variant=<v> seed=<s> impl=<i> pcc=<x> pass=<0/1>

Dense, non-causal, bidirectional over all 8192 tokens; no mask; scale 1.0; batch
sharded on dim 0; same input construction as model_dp.py (pad replaced).
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_ID = "BAAI/bge-m3"
PCC_THRESHOLD = 0.93
SEQ_LEN = 8192


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


def _ids_to_batchsharded(input_ids, mesh_device):
    return ttnn.from_torch(
        input_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_model_dp2_batch_sweep(mesh_device, model_artifacts, reset_seeds):
    local_batch = int(os.environ.get("BGE_LOCAL_BATCH", "6"))
    variant = os.environ.get("BGE_VARIANT", "jit_dram")
    seed = int(os.environ.get("BGE_PCC_SEED", "42"))
    global_batch = 2 * local_batch
    use_jit = variant.startswith("jit")
    if use_jit:
        os.environ.setdefault("BGE_REQUIRE_ENCODER_SDPA", "1")

    assert tuple(mesh_device.shape) == (2, 1)
    assert mesh_device.get_num_devices() == 2

    backbone, state_dict, model_id_or_path = model_artifacts

    model_args, tt_model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=global_batch,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        hf_model_name=model_id_or_path,
        data_parallel=True,
        use_experimental_encoder_sdpa=use_jit,
    )
    assert tt_model._data_parallel

    torch.manual_seed(seed)
    input_ids = torch.randint(low=0, high=model_args.vocab_size, size=(global_batch, SEQ_LEN), dtype=torch.long)
    non_pad = (int(model_args.pad_token_id) + 1) % model_args.vocab_size
    input_ids[input_ids == model_args.pad_token_id] = non_pad
    token_type_ids = torch.zeros_like(input_ids)
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
        input_ids=_ids_to_batchsharded(input_ids, mesh_device),
        attention_mask=None,
        token_type_ids=_ids_to_batchsharded(token_type_ids, mesh_device),
        position_ids=_ids_to_batchsharded(position_ids, mesh_device),
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(2, 1))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=composer).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(global_batch, 1, SEQ_LEN, model_args.dim)

    passing, msg = comp_pcc(reference_output, tt_output_torch, PCC_THRESHOLD)
    impl = "jit_encoder" if use_jit else "stock_ttnn"
    print(
        f"GATE_PCC_SWEEP b={local_batch} variant={variant} seed={seed} impl={impl} pcc={msg} pass={int(bool(passing))}"
    )
    assert passing, f"DP=2 batch-sweep PCC gate failed b={local_batch} variant={variant}: {msg}"
