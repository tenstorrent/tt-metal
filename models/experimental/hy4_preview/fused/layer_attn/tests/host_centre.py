# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: what a score centre would remove at the cut (PLAN.md section 21.5). The reference indexer on the
corrected goldens' P3000 case, full layers 0, 1, 5, the 6 decode rows; the device's two roundings emulated on exact
operands: every head's relu(q . k) packed to BF16 before the head sum, and the score packed to BF16 (minus a centre)."""
import os
import torch

os.environ.setdefault(
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml",
)
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors

bf = lambda t: t.bfloat16().float()
golden = hf_harness.find_decode_golden(6, 3000)
K = 2048
for layer in (0, 1, 5):
    names = ["wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"]
    idx = (
        Hy4Indexer(IndexerConfig())
        .eval()
        .load_weights(dict(zip(names, hy4_tensors([f"model.layers.{layer}.self_attn.indexer.{n}" for n in names]))))
    )
    hf, cache = golden["layers"][layer], golden["cache"][layer]
    pos = torch.tensor(golden["meta"]["positions"])
    cos, sin = rope_cos_sin(pos, AttentionConfig())
    out = {
        k: []
        for k in ("final", "final+centre", "heads", "heads+final", "heads+final+centre", "heads+final+centre(prev)")
    }
    with torch.no_grad():
        q = idx.queries(hf["q_resid"].float(), cos, sin)
        w = idx.head_weights(hf["attn_in"].float())
        prev_cut = None
        for j, p in enumerate(pos.tolist()):
            keys = cache["index_key"][: p + 1].float()
            terms = torch.relu(q[j].float() @ keys.T)  # [32, n]
            exact = w[j] @ terms
            ref = set(exact.topk(K).indices.tolist())
            cut = exact.topk(K).values[-1]
            lossy = w[j] @ bf(terms)
            diff = lambda s: K - len(ref & set(s.topk(K).indices.tolist()))
            out["final"].append(diff(bf(exact)))
            out["final+centre"].append(diff(bf(exact - bf(cut))))
            out["heads"].append(diff(lossy))
            out["heads+final"].append(diff(bf(lossy)))
            out["heads+final+centre"].append(diff(bf(lossy - bf(cut))))
            c = bf(cut) if prev_cut is None else prev_cut  # the cut of the step before (the first: its own)
            out["heads+final+centre(prev)"].append(diff(bf(lossy - c)))
            prev_cut = bf(cut)
            if j == 0:
                print(
                    f"layer {layer} row 0: cut {float(cut):+.4f}, best {float(exact.max()):+.4f}, scores within 1/64 of |cut| of the cut: {int(((exact - cut).abs() <= cut.abs() / 64).sum())}, within 1e-3: {int(((exact - cut).abs() <= 1e-3).sum())}"
                )
    for k, v in out.items():
        print(f"layer {layer} picks of {K} that differ from the reference's, rows 0..5 [{k}]: {v}")
