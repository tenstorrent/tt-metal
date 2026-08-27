# SPDX-License-Identifier: Apache-2.0
"""Multi-seed full-model PCC: experimental (non-FP32-dest SDPA) vs stock, to
confirm the PCC improvement is robust across inputs (not seed-42-specific).
Builds BOTH models once, runs several random-token batches through each, and
reports HF-vs-TT PCC per seed for both. Gated to the DP2 B12 S8192 shape.
"""
import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

DP_BATCH_SIZE, DP_SEQ_LEN = 12, 8192
SEEDS = [7, 123]  # reduced: HF ref forward is slow on CPU
MODEL_ID = "BAAI/bge-m3"


@pytest.fixture(scope="module")
def model_artifacts(model_location_generator):
    transformers = pytest.importorskip("transformers")
    model_id_or_path = str(model_location_generator(MODEL_ID, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.bfloat16).eval()
    backbone = hf_model.roberta if hasattr(hf_model, "roberta") else hf_model
    return backbone, hf_model.state_dict(), model_id_or_path


def _ids_to_batchsharded(input_ids, mesh_device):
    return ttnn.from_torch(
        input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
def test_model_dp2_multiseed(mesh_device, model_artifacts, reset_seeds):
    backbone, state_dict, model_id = model_artifacts
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(2, 1))

    def build(exp):
        ma, m, _ = create_tt_model(
            mesh_device=mesh_device, max_batch_size=DP_BATCH_SIZE, max_seq_len=DP_SEQ_LEN,
            dtype=ttnn.bfloat8_b, state_dict=state_dict, hf_model_name=model_id,
            data_parallel=True, use_experimental_encoder_sdpa=exp,
        )
        return ma, m

    ma, tt_stock = build(False)
    _, tt_exp = build(True)
    pad = int(ma.pad_token_id)
    nonpad_id = (pad + 1) % ma.vocab_size

    print("\nseed |   stock PCC   |   exp PCC")
    for seed in SEEDS:
        torch.manual_seed(seed)
        ids = torch.randint(0, ma.vocab_size, (DP_BATCH_SIZE, DP_SEQ_LEN), dtype=torch.long)
        ids[ids == pad] = nonpad_id
        tti = torch.zeros_like(ids)
        nonpad = (ids != pad).to(torch.long)
        pos = (torch.cumsum(nonpad, 1) * nonpad + pad).to(torch.long)
        with torch.no_grad():
            ref = backbone(input_ids=ids, attention_mask=None, token_type_ids=tti,
                           position_ids=None, return_dict=True).last_hidden_state.unsqueeze(1).to(torch.float32)

        def run(m):
            o = m.forward(
                input_ids=_ids_to_batchsharded(ids, mesh_device), attention_mask=None,
                token_type_ids=_ids_to_batchsharded(tti, mesh_device),
                position_ids=_ids_to_batchsharded(pos, mesh_device),
            )
            t = ttnn.to_torch(o, mesh_composer=composer).to(torch.float32)
            return t.reshape(DP_BATCH_SIZE, 1, DP_SEQ_LEN, ma.dim)

        _, s_msg = comp_pcc(ref, run(tt_stock), 0.9)
        _, e_msg = comp_pcc(ref, run(tt_exp), 0.9)
        print(f"{seed:>4} | {s_msg} | {e_msg}", flush=True)
