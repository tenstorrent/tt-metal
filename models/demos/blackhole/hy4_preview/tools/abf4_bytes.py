"""Exact per-chip weight bytes of the attention-side tensors per format / sharding (Hy4Config shapes). python abf4_bytes.py"""
import json

B8, B4 = (
    1088 / 1024,
    576 / 1024,
)  # bytes per element (tile = 1024 mantissa + 64 exponent bytes; bfp8_b 1088, bfp4_b 576)
GiB = 2**30
c = json.load(open("/mnt/tt-data/Hy4-Preview/config.json"))
D, H, ql, kvl = c["hidden_size"], c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
nope, rope, vd = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
ni, di, I, SI = (
    c["index_n_heads"],
    c["index_head_dim"],
    c["intermediate_size"],
    c["moe_intermediate_size"] * c.get("n_shared_experts", 1),
)
NL, NMTP = c["num_hidden_layers"], c["num_nextn_predict_layers"]
n_full = sum(t == "full" for t in c["indexer_types"]) + NMTP  # MTP has its own indexer
n_attn = NL + NMTP
n_shared = NL - 1 + NMTP  # layers 1..77 + MTP
# elements per tensor (whole), shard factor over the 8 columns (1 = replicated)
T = {
    "q_a": (D * ql, 8),
    "q_b": (ql * H * (nope + rope), 8),
    "kv_a": (D * (kvl + rope), 1),
    "kv_b": (H * (nope + vd) * kvl, 8),
    "gate": (D * H * vd, 8),
    "o_proj": (H * vd * D, 8),
}
IDX = {"idx_wq_b": (ql * ni * di, 1), "idx_wk": (D * di, 1), "idx_wp": (D * ni, 1)}
SH = {"sh_gate": (D * SI, 8), "sh_up": (D * SI, 8), "sh_down": (SI * D, 8)}
DENSE = {"d_gate": (D * I, 8), "d_up": (D * I, 8), "d_down": (I * D, 8)}


def per_chip(items, count, bpe, rows_shard=1, rep_split=False):
    tot = 0
    for n, (el, s) in items.items():
        e = el / s
        if rows_shard > 1 and (s > 1 or rep_split):
            e /= rows_shard  # 32-way: also split over the 4 mesh rows
        tot += e * count * bpe
    return tot / GiB


rows = {}
for name, items, cnt in (
    ("attn(q_a,q_b,kv_a,kv_b,gate,o_proj)", T, n_attn),
    ("indexer(21 full+MTP)", IDX, n_full),
    ("shared expert x78", SH, n_shared),
    ("dense L0 mlp", DENSE, 1),
):
    rows[name] = (
        per_chip(items, cnt, B8),
        per_chip(items, cnt, B4),
        per_chip(items, cnt, B8, 4),
        per_chip(items, cnt, B4, 4),
        per_chip(items, cnt, B4, 4, True),
    )
print("component | bfp8 TP8 | bfp4 TP8 | bfp8 32-way | bfp4 32-way | bfp4 32-way incl replicated split  (GiB per chip)")
for k, v in rows.items():
    print(f"{k:40s} " + " ".join(f"{x:8.3f}" for x in v))
tot = [sum(v[i] for v in rows.values()) for i in range(5)]
print(f"{'TOTAL':40s} " + " ".join(f"{x:8.3f}" for x in tot))
print("saved vs bfp8 TP8:", " ".join(f"{tot[0]-x:6.3f}" for x in tot))
print(
    "\nper tensor (all layers), GiB per chip: bfp8 TP8 | bfp4 TP8 | saving bfp8->bfp4 | extra saving of 32-way at bfp4"
)
for items, cnt in ((T, n_attn), (IDX, n_full), (SH, n_shared), (DENSE, 1)):
    for n, it in items.items():
        a, b = per_chip({n: it}, cnt, B8), per_chip({n: it}, cnt, B4)
        print(f"{n:10s} {a:7.3f} {b:7.3f} {a-b:7.3f} {b - per_chip({n: it}, cnt, B4, 4, True):7.3f}")
