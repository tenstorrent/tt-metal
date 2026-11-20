# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: what a score centre would remove at the cut (PLAN.md sections 21.5, 21.6). The reference indexer on a
corrected decode golden (default: the 6-layer P3000 case; argv: layers, prompt, then optionally the layers to look
at), every full-indexer layer the golden caches keys of, its decode rows; the device's two roundings emulated on exact
operands: every head's relu(q . k) packed to BF16 before the head sum, and the score packed to BF16 (minus a centre).
Per row: picks of the top-2,048 that are not the reference's.

    HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved python host_centre.py [78 3000]

"trunc": the packs as truncations (the packer's default) instead of round-to-nearest."""
import os, sys, torch

os.environ.setdefault(
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml",
)
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors

near = lambda t: t.bfloat16().float()


def trunc(t):
    """float32 -> BF16 by dropping the low 16 bits (toward zero)."""
    return (t.float().contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32)


def tf32(t):
    """float32 -> the FPU's 19-bit operand (e8m10) by dropping the low 13 bits: what a Float32 buffer gives a matmul
    (fused/README.md, second batch), i.e. the head terms if they were packed to Float32 instead of BF16."""
    return (t.float().contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)


args = [a for a in sys.argv[1:] if a != "trunc"]
bf = trunc if "trunc" in sys.argv else near
n_layers, prompt = (int(args[0]), int(args[1])) if len(args) >= 2 else (6, 3000)
golden = hf_harness.find_decode_golden(n_layers, prompt)
assert golden is not None, "no such golden under HY4_GOLDEN_DIR"
K = 2048
cache_all = golden["cache"]
items = cache_all.items() if isinstance(cache_all, dict) else enumerate(cache_all)
full = sorted(l for l, c in items if isinstance(c, dict) and c.get("index_key") is not None)
layers = [int(a) for a in args[2:]] or full
print(
    f"golden L{n_layers} P{prompt}: full-indexer layers with cached keys {full}; looking at {layers}; packs: {'truncate' if bf is trunc else 'nearest'}"
)
names = ["wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"]
KEYS = (
    "final",
    "final+centre",
    "heads",
    "heads+final",
    "heads+final+centre",
    "heads+final+centre(prev)",
    "heads(e8m10)+final+centre(prev)",
)
total = {k: [] for k in KEYS}
crowd = []
for layer in layers:
    idx = (
        Hy4Indexer(IndexerConfig())
        .eval()
        .load_weights(dict(zip(names, hy4_tensors([f"model.layers.{layer}.self_attn.indexer.{n}" for n in names]))))
    )
    hf, cache = golden["layers"][layer], golden["cache"][layer]
    pos = torch.tensor(golden["meta"]["positions"])
    cos, sin = rope_cos_sin(pos, AttentionConfig())
    out = {k: [] for k in KEYS}
    with torch.no_grad():
        q = idx.queries(hf["q_resid"].float(), cos, sin)
        w = idx.head_weights(hf["attn_in"].float())
        prev_cut = None
        for j, p in enumerate(pos.tolist()):
            keys = cache["index_key"][: p + 1].float()
            terms = torch.relu(q[j].float() @ keys.T)  # [32, n]
            exact = w[j] @ terms
            top = exact.topk(K)
            ref, cut = set(top.indices.tolist()), top.values[-1]
            lossy = w[j] @ bf(terms)
            diff = lambda s: K - len(ref & set(s.topk(K).indices.tolist()))
            out["final"].append(diff(bf(exact)))
            out["final+centre"].append(diff(bf(exact - bf(cut))))
            out["heads"].append(diff(lossy))
            out["heads+final"].append(diff(bf(lossy)))
            out["heads+final+centre"].append(diff(bf(lossy - bf(cut))))
            c = bf(cut) if prev_cut is None else prev_cut  # the cut of the step before (the first: its own)
            out["heads+final+centre(prev)"].append(diff(bf(lossy - c)))
            out["heads(e8m10)+final+centre(prev)"].append(diff(bf(w[j] @ tf32(terms) - c)))
            prev_cut = bf(cut)
            crowd.append(
                (
                    layer,
                    j,
                    float(cut),
                    float(exact.max()),
                    int(((exact - cut).abs() <= cut.abs() / 64).sum()),
                    keys.shape[0],
                )
            )
    for k, v in out.items():
        total[k] += v
    row0 = [c for c in crowd if c[0] == layer][0]
    print(
        f"layer {layer}: row 0 cut {row0[2]:+.3f} best {row0[3]:+.3f} scores within |cut|/64 of the cut {row0[4]} of {row0[5]} keys; "
        + "; ".join(f"[{k}] {v}" for k, v in out.items())
    )
rows = len(total["final"])
print(
    f"ALL {len(layers)} layers x {rows // max(len(layers), 1)} rows = {rows} rows, picks of {K} that are not the reference's: "
    + "; ".join(f"[{k}] mean {sum(v) / rows:.2f} max {max(v)}" for k, v in total.items())
)
