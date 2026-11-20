# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the scan's score centre at AgentX context lengths, on REAL keys (PLAN.md section 21.6). Layer 0 is a
full-indexer layer whose keys and queries depend on nothing but the tokens themselves (embedding -> the layer's
iHC pre-mix -> input norm -> wk / q_a / weights_proj): its index keys of ANY context length are a few matmuls on the
host. The pipeline is first checked against the corrected P3000 golden (its input ids -> its cached keys of layer 0
and its q_resid rows); then, on a long token stream (this tree's own documents and sources through the model's
tokenizer: agent-like text), for each context length N the 6 decode rows at positions N .. N + 5: picks of the top-2,048
that are not the reference's under the device's roundings (host_centre.py's emulation).

    HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved python host_centre_long.py [contexts...]
"""
import os, sys, glob, torch
import torch.nn.functional as F

os.environ.setdefault(
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml",
)
from models.experimental.hy4_preview.fused.ihc_site.op import golden_pre
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rms_norm, rope_cos_sin
from models.experimental.hy4_preview.reference.ihc import IHCConfig, ihc_expand
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors

K, LAYER, CHUNK = 2048, 0, 4096
near = lambda t: t.bfloat16().float()
tf32 = lambda t: (t.float().contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)
P = f"model.layers.{LAYER}."
(embed,) = hy4_tensors(["model.embed_tokens.weight"])
site = hy4_tensors([P + "hc_attn_layer.hc_pre." + n for n in ("hc_fn", "hc_base", "hc_scale")])
gamma, q_a, q_norm = hy4_tensors(
    [P + "input_layernorm.weight", P + "self_attn.q_a_proj.weight", P + "self_attn.q_a_layernorm.weight"]
)
names = ["wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"]
idx = (
    Hy4Indexer(IndexerConfig())
    .eval()
    .load_weights(dict(zip(names, hy4_tensors([P + "self_attn.indexer." + n for n in names]))))
)
icfg, acfg = IHCConfig(dim=embed.shape[1]), AttentionConfig()
# the layer's attention (reference/attention_decode.py), for what the picks that differ do to the sublayer's output F
from models.experimental.hy4_preview.reference.attention import Hy4Attention
from models.experimental.hy4_preview.reference.attention_decode import Hy4AttentionDecode

attention = Hy4Attention(acfg).eval()
with torch.no_grad():
    a_names = [
        P + "self_attn." + n + ("" if n == "learnable_sink_param" else ".weight") for n in Hy4Attention.PARAMETER_NAMES
    ]
    for name, value in zip(Hy4Attention.PARAMETER_NAMES, hy4_tensors(a_names)):
        getattr(attention, name).copy_(value.reshape(getattr(attention, name).shape))
decode = Hy4AttentionDecode(attention, None)


def layer_input(ids):
    """ids [T] -> x [T, 6144]: the layer's attention input (after the iHC pre-mix and the input norm)."""
    streams = ihc_expand(embed[ids].float()[None], icfg.n)[0]  # [T, 4, C]
    y, _, _, _ = golden_pre(icfg, *site, streams)
    return rms_norm(y, gamma.float(), acfg.norm_eps)


def keys_of(ids, first=0):
    out = []
    with torch.no_grad():
        for a in range(0, len(ids), CHUNK):
            part = ids[a : a + CHUNK]
            cos, sin = rope_cos_sin(torch.arange(first + a, first + a + len(part)), acfg)
            out.append(idx.keys(layer_input(part), cos, sin))
    return torch.cat(out)


def latents_of(ids):
    """The layer's cache rows (k_pass | k_rope) of every position, [T, 576]."""
    out = []
    with torch.no_grad():
        for a in range(0, len(ids), CHUNK):
            part = ids[a : a + CHUNK]
            cos, sin = rope_cos_sin(torch.arange(a, a + len(part)), acfg)
            _, k_pass, k_rope = attention.latent(layer_input(part), cos, sin)
            out.append(torch.cat([k_pass, k_rope], dim=-1))
    return torch.cat(out)


def f_of(ids, p, latents, picks):
    """The sublayer's output F [6144] of the token at position p attending over the cache rows ``picks``."""
    with torch.no_grad():
        x = layer_input(ids[p : p + 1])
        token = decode.new_token(x, p)
        return decode.stages(x, token, latents[picks])["out"][0].double()


def rows_of(ids, positions):
    """(queries [S, 32, 128], head weights [S, 32]) of the tokens at ``positions``."""
    with torch.no_grad():
        x = layer_input(ids[positions])
        cos, sin = rope_cos_sin(positions, acfg)
        q_resid = rms_norm(F.linear(x, q_a.float()), q_norm.float(), acfg.norm_eps)
        return idx.queries(q_resid, cos, sin), idx.head_weights(x)


# ---- the pipeline against the golden
golden = hf_harness.find_decode_golden(6, 3000)
g_ids = golden["input_ids"].long()
g_keys = golden["cache"][LAYER]["index_key"].float()
mine = keys_of(g_ids[: g_keys.shape[0]])
rel = float((mine - g_keys).pow(2).sum().sqrt() / g_keys.pow(2).sum().sqrt())
pos = torch.tensor(golden["meta"]["positions"])
q, w = rows_of(g_ids, pos)
exact_g = Hy4Indexer.scores(q, w, g_keys)
exact_m = Hy4Indexer.scores(q, w, mine)
shared = [
    len(set(exact_g[j, : p + 1].topk(K).indices.tolist()) & set(exact_m[j, : p + 1].topk(K).indices.tolist()))
    for j, p in enumerate(pos.tolist())
]
rec = [
    len(set(golden["layers"][LAYER]["topk_indices"][j].tolist()) & set(exact_m[j, : p + 1].topk(K).indices.tolist()))
    for j, p in enumerate(pos.tolist())
]
print(
    f"pipeline check on the P3000 golden, layer {LAYER}: host keys vs the golden's cached keys rel rms {rel:.2e}; picks shared between "
    f"scores on the golden's keys and on the host's {shared} of {K}; with the golden's recorded picks {rec}"
)

# ---- a long token stream
from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.path.join(os.environ["HY4_HF_MODEL"], "tokenizer.json"))
contexts = [int(a) for a in sys.argv[1:]] or [3000, 45609, 92682, 215110, 535958]
need = max(contexts) + 6
root = "generated/hy4_preview/tt-metal"
files = sorted(glob.glob("generated/hy4_preview/*.md")) + sorted(
    glob.glob(root + "/models/experimental/hy4_preview/**/*.md", recursive=True)
)
files += sorted(glob.glob(root + "/models/experimental/hy4_preview/**/*.py", recursive=True)) + sorted(
    glob.glob(root + "/ttnn/cpp/ttnn/operations/**/*.cpp", recursive=True)
)
ids = []
for f in files:
    try:
        ids += tok.encode(open(f, errors="ignore").read(), add_special_tokens=False).ids
    except Exception:
        continue
    if len(ids) >= need:
        break
assert len(ids) >= need, (len(ids), need)
ids = torch.tensor(ids[:need])
print(f"token stream: {len(ids)} tokens from {len(files)} candidate files (documents, then sources)")
all_keys = keys_of(ids)
all_latents = latents_of(ids)
TOTAL, F_ALL = {}, {}
KEYS = (
    "final (BF16 score)",
    "today (heads BF16 + final)",
    "centre = the cut",
    "centre = previous step's cut",
    "heads alone (the floor of a centre)",
    "heads as e8m10 + centre(prev)",
    "heads as e8m10, no centre",
)
for n in contexts:
    positions = torch.arange(n, n + 6)
    q, w = rows_of(ids, positions)
    out = {k: [] for k in KEYS}
    f_rows = {}
    crowd, prev = [], None
    for j, p in enumerate(positions.tolist()):
        keys = all_keys[: p + 1]
        terms = torch.relu(q[j].float() @ keys.T)
        exact = w[j] @ terms
        top = exact.topk(K)
        ref, cut = set(top.indices.tolist()), top.values[-1]
        lossy = w[j] @ near(terms)
        diff = lambda s: K - len(ref & set(s.topk(K).indices.tolist()))
        c = near(cut) if prev is None else prev
        out[KEYS[0]].append(diff(near(exact)))
        out[KEYS[1]].append(diff(near(lossy)))
        out[KEYS[2]].append(diff(near(lossy - near(cut))))
        out[KEYS[3]].append(diff(near(lossy - c)))
        out[KEYS[4]].append(diff(lossy))
        out[KEYS[5]].append(diff(near(w[j] @ tf32(terms) - c)))
        out[KEYS[6]].append(diff(near(w[j] @ tf32(terms))))
        # what the picks that differ do to F: the same token attending over each selection
        f_ref = f_of(ids, p, all_latents, top.indices)
        for name, scores in (
            ("today", near(lossy)),
            ("centre(prev)", near(lossy - c)),
            ("Float32 head terms + centre(prev)", near(w[j] @ tf32(terms) - c)),
        ):
            f = f_of(ids, p, all_latents, scores.topk(K).indices)
            rel = float((f - f_ref).pow(2).sum().sqrt() / f_ref.pow(2).sum().sqrt())
            pc = float(torch.corrcoef(torch.stack((f, f_ref)))[0, 1])
            f_rows.setdefault(name, []).append((rel, pc))
        prev = near(cut)
        step = 2.0 ** (torch.floor(torch.log2(cut.abs())) - 7)  # a BF16 step at the cut
        crowd.append((float(cut), float(exact.max()), int(((exact - cut).abs() <= step).sum())))
    for k in KEYS:
        TOTAL.setdefault(k, []).extend(out[k] if n > 3000 else [])
    for k, v in f_rows.items():
        F_ALL.setdefault(k, []).extend(v if n > 3000 else [])
    print(
        f"context {n}: F with a selection against F with the reference's selection, per row (relative rms; 1 - pcc): "
        + "; ".join(
            f"[{k}] rel rms {[f'{r:.1e}' for r, _ in v]} 1-pcc max {max(1 - c for _, c in v):.1e}"
            for k, v in f_rows.items()
        )
    )
    print(
        f"context {n}: cut {crowd[0][0]:+.3f}, best {crowd[0][1]:+.3f}, scores within one BF16 step of the cut {[c[2] for c in crowd]}; "
        + "; ".join(f"[{k}] {v}" for k, v in out.items())
    )
rows = len(next(iter(TOTAL.values())))
print(
    f"ALL contexts above 3,000 ({rows} rows), picks of {K} that are not the reference's: "
    + "; ".join(f"[{k}] mean {sum(v) / max(rows, 1):.1f} max {max(v)}" for k, v in TOTAL.items())
)
print(
    "ALL contexts above 3,000: F against F with the reference's selection: "
    + "; ".join(
        f"[{k}] rel rms mean {sum(r for r, _ in v) / len(v):.2e} max {max(r for r, _ in v):.2e}, 1 - pcc max {max(1 - c for _, c in v):.2e}"
        for k, v in F_ALL.items()
    )
)
