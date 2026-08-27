# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Calibration: does the ALREADY-SHIPPING K-bf4 also show low min per-token
cosine vs a K-bf8 variant? This calibrates the min-cos metric used to judge
Wi-bf4 (#233/#234). If shipping K-bf4 also has min-cos ~0.3, min-cos is just this
model's normal quantization spread and Wi-bf4 is not specially bad. If K-bf4
stays ~0.95, then Wi-bf4's 0.24 is genuinely disqualifying.

Compares TT(K-bf4, shipping) vs TT(K-bf8, BGE_NO_KBF4=1), same metric as
wi_bf4_quality.py. Fast (no HF forward).
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_ID = "BAAI/bge-m3"
DP_BATCH_SIZE = 12
DP_SEQ_LEN = 8192
SEEDS = [42, 7, 2024]


@pytest.fixture(scope="module")
def sd():
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval()
    return hf.state_dict(), MODEL_ID


def _shard(ids, dev):
    return ttnn.from_torch(
        ids.to(torch.int32), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0), memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_kbf4_calib(mesh_device, sd, reset_seeds):
    state_dict, model_id = sd
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(2, 1))

    def build():
        ma, m, _ = create_tt_model(
            mesh_device=mesh_device, max_batch_size=DP_BATCH_SIZE, max_seq_len=DP_SEQ_LEN,
            dtype=ttnn.bfloat8_b, state_dict=state_dict, hf_model_name=model_id,
            data_parallel=True, use_experimental_encoder_sdpa=True,
        )
        return ma, m

    # shipping = K bf4; variant = K bf8. NO_KBF4 read at forward time (attention.py).
    ma, tt = build()
    pad = int(ma.pad_token_id)
    nonpad_id = (pad + 1) % ma.vocab_size

    def run(ids, tti, pos):
        o = tt.forward(
            input_ids=_shard(ids, mesh_device), attention_mask=None,
            token_type_ids=_shard(tti, mesh_device), position_ids=_shard(pos, mesh_device),
        )
        return ttnn.to_torch(o, mesh_composer=composer).to(torch.float32).reshape(DP_BATCH_SIZE, 1, DP_SEQ_LEN, ma.dim)

    print("\nseed | PCC(kbf4|kbf8) | all mean/min | hi-norm mean/min", flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        ids = torch.randint(0, ma.vocab_size, (DP_BATCH_SIZE, DP_SEQ_LEN), dtype=torch.long)
        ids[ids == pad] = nonpad_id
        tti = torch.zeros_like(ids)
        nonpad = (ids != pad).to(torch.long)
        pos = (torch.cumsum(nonpad, 1) * nonpad + pad).to(torch.long)

        os.environ["BGE_NO_KBF4"] = "0"  # shipping: K bf4
        a = run(ids, tti, pos)
        os.environ["BGE_NO_KBF4"] = "1"  # variant: K bf8
        b = run(ids, tti, pos)
        os.environ["BGE_NO_KBF4"] = "0"

        _, pcc_msg = comp_pcc(a, b, 0.9)
        af, bf = a.reshape(-1, ma.dim), b.reshape(-1, ma.dim)
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=-1)
        norms = af.norm(dim=-1)
        keep = norms >= torch.quantile(norms, 0.25)
        ch = cos[keep]
        print(
            f"{seed:>4} | {pcc_msg} | {cos.mean().item():.4f}/{cos.min().item():.4f}"
            f" | {ch.mean().item():.4f}/{ch.min().item():.4f}",
            flush=True,
        )
