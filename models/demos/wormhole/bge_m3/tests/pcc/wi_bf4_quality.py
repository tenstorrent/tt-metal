# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TT-vs-TT quality delta: committed bf8 baseline vs Wi-activation-bf4.

Question for the 12-emb/s gate decision: how much does Wi-bf4 (the -10ms lever)
change the DELIVERED embedding relative to the config we ALREADY ship (bf8)?
PCC-vs-HF already bakes in bf8 quantization error we accept today; the relevant
quality delta for a *further* optimization is TT(bf8) vs TT(bf4). No HF forward
-> fast enough to sweep many seeds (anti-overfit: judge on the distribution, not
one seed).

Reports, per seed, over the full B12/S8192 output:
  - PCC(bf4 vs bf8)                : correlation of the two TT outputs
  - mean per-token cosine sim      : the retrieval-relevant metric (embeddings
                                     are compared by cosine in practice)
  - min per-token cosine sim       : worst-token degradation

Run: BGE_WI_BF4 is toggled internally per config.
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
SEEDS = [42, 7, 123, 2024, 99]


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
def test_wi_bf4_quality(mesh_device, sd, reset_seeds):
    state_dict, model_id = sd
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(2, 1))

    def build():
        ma, m, _ = create_tt_model(
            mesh_device=mesh_device, max_batch_size=DP_BATCH_SIZE, max_seq_len=DP_SEQ_LEN,
            dtype=ttnn.bfloat8_b, state_dict=state_dict, hf_model_name=model_id,
            data_parallel=True, use_experimental_encoder_sdpa=True,
        )
        return ma, m

    # Build both models once. BGE_WI_BF4 is read at weight-build time, so set the
    # env BEFORE each build and keep the two model objects side by side.
    os.environ["BGE_WI_BF4"] = "0"
    ma, tt_bf8 = build()
    os.environ["BGE_WI_BF4"] = "1"
    _, tt_bf4 = build()
    os.environ["BGE_WI_BF4"] = "0"

    pad = int(ma.pad_token_id)
    nonpad_id = (pad + 1) % ma.vocab_size

    def run(m, ids, tti, pos):
        o = m.forward(
            input_ids=_shard(ids, mesh_device), attention_mask=None,
            token_type_ids=_shard(tti, mesh_device), position_ids=_shard(pos, mesh_device),
        )
        t = ttnn.to_torch(o, mesh_composer=composer).to(torch.float32)
        return t.reshape(DP_BATCH_SIZE, 1, DP_SEQ_LEN, ma.dim)

    print("\nseed | PCC(bf4|bf8) | mean_cos | min_cos", flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        ids = torch.randint(0, ma.vocab_size, (DP_BATCH_SIZE, DP_SEQ_LEN), dtype=torch.long)
        ids[ids == pad] = nonpad_id
        tti = torch.zeros_like(ids)
        nonpad = (ids != pad).to(torch.long)
        pos = (torch.cumsum(nonpad, 1) * nonpad + pad).to(torch.long)

        a = run(tt_bf8, ids, tti, pos)
        b = run(tt_bf4, ids, tti, pos)
        _, pcc_msg = comp_pcc(a, b, 0.9)
        # per-token cosine similarity over the hidden dim
        af = a.reshape(-1, ma.dim)
        bf = b.reshape(-1, ma.dim)
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=-1)
        # CONTROL: condition on token norm. Random-id S8192 has degenerate near-
        # zero-norm rows where cosine is meaningless. Restrict to content tokens
        # (norm >= 25th percentile of the bf8 baseline norms).
        norms = af.norm(dim=-1)
        thr = torch.quantile(norms, 0.25)
        keep = norms >= thr
        ch = cos[keep]
        n = int(keep.sum())
        print(
            f"{seed:>4} | {pcc_msg} | all mean/min={cos.mean().item():.4f}/{cos.min().item():.4f}"
            f" | hi-norm(n={n}) mean/min={ch.mean().item():.4f}/{ch.min().item():.4f}"
            f" | norm min/p25/med={norms.min().item():.2f}/{thr.item():.2f}/{norms.median().item():.2f}",
            flush=True,
        )
