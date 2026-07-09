#!/usr/bin/env python3
"""Per-call attribution: assign each execution-ordered device op call to a semantic block
and an internal op node, with its REAL merged duration. Greedy template walk with alias
handling for ttnn-relabeled CCL ops. Output feeds both the semantic node durations and the
expanded intra-block dataflow."""
import glob, os, sys, json
import pandas as pd

sys.path.insert(0, os.getcwd())
from models.tt_transformers.tests.test_utils import merge_device_rows

DUR = "DEVICE KERNEL DURATION [ns]"
SP = os.path.dirname(os.path.abspath(__file__))
# BRANCH report dirs per (variant, mode, scenario), from variant-aware discover.py (keys "v/m/s").
BRANCH_TOTALS = os.path.join(SP, "totals_e71a1b0a666.json")
RPT = {}  # (variant, mode, scenario) -> report dir
for _k, _v in json.load(open(BRANCH_TOTALS)).items():
    _variant, _mode, _scen = _k.split("/")
    RPT[(_variant, _mode, _scen)] = _v["dir"]
VARIANTS = sorted({k[0] for k in RPT})
SCENS = ["warm", "cold", "long"]

# alias groups: a template code accepts any code in its group (ttnn relabels)
AG = {"AllGatherAsyncDeviceOperation", "AllBroadcastDeviceOperation"}
RS = {"ReduceScatterMinimalAsyncDeviceOperation", "ReduceScatterDeviceOperation"}
ROPE = {"RotaryEmbeddingIndexedDeviceOperation", "RotaryEmbeddingLlamaDeviceOperation"}


def C(x):
    return {x}


# ordered internal-op template per mode: (block, node_id, node_label, {accepted op codes})
SPARSE_TMPL = [
    ("s1", "s1.qa", "q_a_proj matmul", C("MatmulDeviceOperation")),
    ("s1", "s1.rs", "TP reduce-scatter", RS),
    ("s1", "s1.ag", "TP all-gather", AG),
    ("s1", "s1.norm", "q_a RMSNorm", C("LayerNormDeviceOperation")),
    ("s2", "s2.wk", "indexer wk matmul", C("MatmulDeviceOperation")),
    ("s2", "s2.rs", "TP reduce-scatter", RS),
    ("s2", "s2.ag", "TP all-gather", AG),
    ("s2", "s2.norm", "indexer k-norm", C("LayerNormDeviceOperation")),
    ("s2", "s2.sl1", "slice rope", C("SliceDeviceOperation")),
    ("s2", "s2.sl2", "slice nope", C("SliceDeviceOperation")),
    ("s2", "s2.perm", "rope permute matmul", C("MatmulDeviceOperation")),
    ("s2", "s2.rope", "rotary (block-cyclic)", ROPE),
    ("s2", "s2.cat", "concat pe·nope", C("ConcatDeviceOperation")),
    ("s2", "s2.tc", "typecast → bf8", C("TypecastDeviceOperation")),
    ("s2", "s2.wr", "update index K-cache", C("UpdatePaddedKvCacheDeviceOperation")),
    # SP×TP seq-sharded indexer (commit c7f7dc30f3b). wq_b is replicated (all H_idx heads on-chip) so the
    # score head-sum is complete on-chip — the old TP logit all-reduce (tilize/RS/AG/untilize over the
    # ~end_pos-wide logits) is GONE. Instead the query seq is TP-sharded (mesh_partition qr), and only the
    # small top-k indices are gathered back over TP after top-k.
    ("s3", "s3.mpq", "TP seq-shard qr (mesh_partition)", C("MeshPartitionDeviceOperation")),
    ("s3", "s3.wqb", "indexer wq_b matmul", C("MatmulDeviceOperation")),
    ("s3", "s3.heads", "create heads", C("NlpCreateHeadsDeviceOperation")),
    ("s3", "s3.sl1", "slice rope", C("SliceDeviceOperation")),
    ("s3", "s3.sl2", "slice nope", C("SliceDeviceOperation")),
    ("s3", "s3.rperm", "rope permute matmul", C("MatmulDeviceOperation")),
    ("s3", "s3.rope", "rotary q", ROPE),
    ("s3", "s3.cat", "concat q", C("ConcatDeviceOperation")),
    ("s3", "s3.wproj", "weights_proj matmul", C("MatmulDeviceOperation")),
    ("s3", "s3.wrs", "weights reduce-scatter", RS),
    ("s3", "s3.wag", "weights all-gather", AG),
    ("s3", "s3.mul", "scale multiply", C("BinaryNgDeviceOperation")),
    ("s3", "s3.mpw", "TP seq-shard weights (mesh_partition)", C("MeshPartitionDeviceOperation")),
    ("s3", "s3.pm", "permute weights", C("PermuteDeviceOperation")),
    ("s3", "s3.gk", "gather index K (SP)", AG),
    ("s3", "s3.score", "indexer_score_dsa", C("IndexerScoreDeviceOperation")),
    ("s3", "s3.topk", "top-k indices", C("TopkLargeIndicesDeviceOperation")),
    ("s3", "s3.idxag", "TP gather top-k indices", AG),
    ("s3", "s3.idxcat", "index gather concat", C("ConcatDeviceOperation")),
    ("s4", "s4.qb", "q_b_proj matmul", C("MatmulDeviceOperation")),
    ("s4", "s4.heads", "create heads", C("NlpCreateHeadsDeviceOperation")),
    ("s4", "s4.sl1", "slice nope", C("SliceDeviceOperation")),
    ("s4", "s4.sl2", "slice rope", C("SliceDeviceOperation")),
    ("s4", "s4.absorb", "wkv_b1 absorb matmul", C("MatmulDeviceOperation")),
    ("s4", "s4.rope", "rotary q", ROPE),
    ("s4", "s4.cat", "concat q", C("ConcatDeviceOperation")),
    ("s5", "s5.kva", "kv_a_proj matmul", C("MatmulDeviceOperation")),
    ("s5", "s5.ag", "TP all-gather (dim1)", AG),
    ("s5", "s5.frnc", "fast_reduce_nc", C("FastReduceNCDeviceOperation")),
    ("s5", "s5.sl1", "slice nope", C("SliceDeviceOperation")),
    ("s5", "s5.sl2", "slice rope", C("SliceDeviceOperation")),
    ("s5", "s5.norm", "kv-norm", C("LayerNormDeviceOperation")),
    ("s5", "s5.rope", "rotary kv", ROPE),
    ("s5", "s5.cat", "concat kvpe", C("ConcatDeviceOperation")),
    ("s5", "s5.ut", "to ROW_MAJOR (cache fmt)", C("UntilizeDeviceOperation")),
    ("s6", "s6.wr", "update KVPE cache", C("UpdatePaddedKvCacheDeviceOperation")),
    ("s7", "s7.ag", "gather KVPE prefix (SP)", AG),
    ("s8", "s8.q2rm", "q → ROW_MAJOR", C("UntilizeDeviceOperation")),
    ("s8", "s8.sdpa", "sparse_sdpa (top-k)", C("SparseSDPAOperation")),
    ("s8", "s8.o2tile", "out → TILE", C("TilizeDeviceOperation")),
    ("s9", "s9.wkvb2", "wkv_b2 matmul", C("MatmulDeviceOperation")),
    ("s10", "s10.cat", "concat heads", C("NLPConcatHeadsDeviceOperation")),
    ("s10", "s10.o", "o_proj matmul", C("MatmulDeviceOperation")),
    ("s10", "s10.rs", "TP reduce-scatter", RS),
]
DENSE_TMPL = [
    ("d1", "d1.qa", "q_a_proj matmul", C("MatmulDeviceOperation")),
    ("d1", "d1.rs", "TP reduce-scatter", RS),
    ("d1", "d1.ag", "TP all-gather", AG),
    ("d1", "d1.norm", "q_a RMSNorm", C("LayerNormDeviceOperation")),
    ("d2", "d2.qb", "q_b_proj matmul", C("MatmulDeviceOperation")),
    ("d2", "d2.heads", "create heads", C("NlpCreateHeadsDeviceOperation")),
    ("d2", "d2.sl1", "slice nope", C("SliceDeviceOperation")),
    ("d2", "d2.sl2", "slice rope", C("SliceDeviceOperation")),
    ("d2", "d2.absorb", "wkv_b1 absorb matmul", C("MatmulDeviceOperation")),
    ("d2", "d2.rope", "rotary q", ROPE),
    ("d2", "d2.cat", "concat q", C("ConcatDeviceOperation")),
    ("d3", "d3.kva", "kv_a_proj matmul", C("MatmulDeviceOperation")),
    ("d3", "d3.ag", "TP all-gather (dim1)", AG),
    ("d3", "d3.frnc", "fast_reduce_nc", C("FastReduceNCDeviceOperation")),
    ("d3", "d3.sl1", "slice nope", C("SliceDeviceOperation")),
    ("d3", "d3.sl2", "slice rope", C("SliceDeviceOperation")),
    ("d3", "d3.norm", "kv-norm", C("LayerNormDeviceOperation")),
    ("d3", "d3.rope", "rotary kv", ROPE),
    ("d3", "d3.cat", "concat kvpe", C("ConcatDeviceOperation")),
    ("d3", "d3.tc", "typecast → bf8", C("TypecastDeviceOperation")),
    ("d4", "d4.wr", "update KVPE cache", C("UpdatePaddedKvCacheDeviceOperation")),
    ("d5", "d5.ring", "ring_mla (RingJointSDPA)", C("RingJointSDPADeviceOperation")),
    ("d6", "d6.wkvb2", "wkv_b2 matmul", C("MatmulDeviceOperation")),
    ("d7", "d7.cat", "concat heads", C("NLPConcatHeadsDeviceOperation")),
    ("d7", "d7.o", "o_proj matmul", C("MatmulDeviceOperation")),
    ("d7", "d7.rs", "TP reduce-scatter", RS),
]
# GLM sparse = deepseek SPARSE_TMPL with variant deltas (confirmed from the glm_5_1 warm-sparse dump):
#  - interleaved indexer RoPE (index_rope_interleave=True) → _rope_perm is None → NO rope-permute matmul,
#    so drop s2.perm and s3.rperm.
#  - 16 heads/chip (< 32) → _needs_head_to_seq_reshard fires → s8 gains the thin-head transpose: an
#    all-gather(q, dim1) + mesh_partition BEFORE sparse_sdpa, and mesh_partition + all-gather + mesh_partition
#    AFTER. The two AllGatherAsync reshards (~1.1 ms + ~1.5 ms) dominate — larger than sparse_sdpa itself.
_GLM_S8 = [
    ("s8", "s8.agq", "thin-head q all-gather (dim1)", AG),
    ("s8", "s8.mp1", "reshard q (mesh_partition)", C("MeshPartitionDeviceOperation")),
    ("s8", "s8.q2rm", "q → ROW_MAJOR", C("UntilizeDeviceOperation")),
    ("s8", "s8.mpidx", "reshard indices (mesh_partition)", C("MeshPartitionDeviceOperation")),
    ("s8", "s8.sdpa", "sparse_sdpa (top-k)", C("SparseSDPAOperation")),
    ("s8", "s8.o2tile", "out → TILE", C("TilizeDeviceOperation")),
    ("s8", "s8.ago", "thin-head out all-gather (dim2)", AG),
    ("s8", "s8.mp2", "reshard out (mesh_partition)", C("MeshPartitionDeviceOperation")),
]
GLM_SPARSE_TMPL = [n for n in SPARSE_TMPL if n[0] != "s8" and n[1] not in ("s2.perm", "s3.rperm")] + _GLM_S8
GLM_SPARSE_TMPL.sort(key=lambda n: int(n[0][1:]))  # keep block order (s1..s10); stable within block

TMPL = {
    ("deepseek_v32", "sparse"): SPARSE_TMPL,
    ("deepseek_v32", "dense"): DENSE_TMPL,
    ("glm_5_1", "sparse"): GLM_SPARSE_TMPL,
    ("glm_5_1", "dense"): DENSE_TMPL,
}
LABELS = {}
BLOCK_OF = {}
for _tp in TMPL.values():
    for blk, nid, lab, codes in _tp:
        LABELS[nid] = lab
        BLOCK_OF[nid] = blk

# once-per-forward structural ops: pin to their node so same-code CCL noise can't displace them
ANCHOR_CODES = {
    "IndexerScoreDeviceOperation",
    "TopkLargeIndicesDeviceOperation",
    "SparseSDPAOperation",
    "RingJointSDPADeviceOperation",
    "FastReduceNCDeviceOperation",
    "BinaryNgDeviceOperation",
    "NLPConcatHeadsDeviceOperation",
    "TypecastDeviceOperation",
}


def anchor_map(tmpl):
    m = {}
    for i, (blk, nid, lab, codes) in enumerate(tmpl):
        if len(codes) == 1 and next(iter(codes)) in ANCHOR_CODES:
            m[next(iter(codes))] = i
    return m


def walk_forward(calls, tmpl):
    """calls: list of (code,dur) in exec order → dict node_id->[durs] (+ misc buckets).
    Match is scoped to the CURRENT block (earliest unconsumed matching node); if the current
    block has no match, allow a transition to the immediate next block; otherwise the call is a
    composite of the current block (misc). Never jumps 2+ blocks ahead, so repeated op codes
    (e.g. an extra relabeled all-gather) stay in-block instead of pulling a later node early."""
    N = len(tmpl)
    node = {}
    ptr = 0
    anch = anchor_map(tmpl)
    order = {}
    oi = 0
    # Node ids consumed by the anchor branch (pinned by code, label-only). They must NOT hold up the
    # sequential pointer, so ptr auto-skips them: otherwise ptr would stall on score/topk/mul and the
    # in-order tail (idxag, gather, etc.) would fall through to composites.
    anchor_nids = {tmpl[i][1] for i in anch.values()}

    def rec(nid, dur):
        nonlocal oi
        node.setdefault(nid, []).append(dur)
        order.setdefault(nid, oi)
        oi += 1

    def skip_anchors(p):
        while p < N and tmpl[p][1] in anchor_nids:
            p += 1
        return p

    for code, dur in calls:
        if code in anch:  # pin structural anchor to its node — LABEL ONLY, do not advance ptr.
            # (the profiler can list an async anchor like topk out of program order; advancing ptr
            #  here would orphan the in-order ops that follow it and race the whole tail.)
            idx = anch[code]
            rec(tmpl[idx][1], dur)
            continue
        ptr = skip_anchors(ptr)
        if ptr >= N:
            nid = f"{tmpl[-1][0]}.misc:{code.replace('DeviceOperation','').replace('Operation','')}"
            rec(nid, dur)
            continue
        cur = tmpl[ptr][0]
        # candidate 1: STRICT — only the (non-anchor) node exactly at ptr. Scanning ahead within the
        # block (the old behaviour) let an unmodeled composite whose code equals a LATER named node
        # (e.g. a rope-internal MeshPartition matching s3.mpw, or an RS-internal Concat matching
        # s3.idxcat) jump the pointer and orphan every node between — block sums stay exact but the
        # distribution scrambles. Strict-at-ptr keeps composites in-block; ptr only advances on a
        # real, in-order match. (Verified: every template node is present and in program order for the
        # DeepSeek block-cyclic path on this box, so no legitimate node is skipped.)
        if code in tmpl[ptr][3]:
            rec(tmpl[ptr][1], dur)
            ptr += 1
            continue
        # candidate 2: transition ONLY if the call matches the ENTRY op of the immediate next block
        # (rare safety for a genuinely-absent node); skip that block's leading anchors too.
        k = ptr
        while k < N and tmpl[k][0] == cur:
            k += 1
        k = skip_anchors(k)
        if k < N and code in tmpl[k][3]:
            rec(tmpl[k][1], dur)
            ptr = k + 1
            continue
        # otherwise: composite of the current block
        nid = f"{cur}.misc:{code.replace('DeviceOperation','').replace('Operation','')}"
        rec(nid, dur)
    return node, order


def ordered_calls(df):
    return list(zip(df["OP CODE"].tolist(), pd.to_numeric(df[DUR], errors="coerce").fillna(0).tolist()))


out = {v: {"sparse": {}, "dense": {}} for v in VARIANTS}  # variant -> mode -> scenario -> agg
for (variant, mode, scen), rp in RPT.items():
    f = glob.glob(rp + "/ops_perf_results_*.csv")[0]
    df0 = pd.read_csv(f, low_memory=False)
    mk = df0[df0["OP TYPE"] == "signpost"]["OP CODE"]
    a = mk[mk == "start"].index[0]
    b = mk[mk == "stop"].index[0]
    region = df0.iloc[a + 1 : b].copy()
    region[DUR] = pd.to_numeric(region[DUR], errors="coerce")
    ri = region.reset_index(drop=True)
    starts = list(ri.index[ri["OP CODE"] == "MLA_START"])
    bounds = starts + [len(ri)]
    agg = {}
    for i in range(len(starts)):
        seg = merge_device_rows(ri.iloc[bounds[i] + 1 : bounds[i + 1]])
        nd, od = walk_forward(ordered_calls(seg), TMPL[(variant, mode)])
        for nid, ds in nd.items():
            e = agg.setdefault(nid, {"total_ns": 0.0, "count": 0, "order": od[nid]})
            e["total_ns"] += sum(ds)
            e["count"] += len(ds)
            e["order"] = min(e["order"], od[nid])
    out[variant][mode][scen] = agg

# assemble: per (variant,mode,scenario): block totals + node list
result = {"nodes": {}, "blocks": {}}
for variant in out:
    result["nodes"][variant] = {"sparse": {}, "dense": {}}
    result["blocks"][variant] = {"sparse": {}, "dense": {}}
    for mode in out[variant]:
        for scen, agg in out[variant][mode].items():
            blocks = {}
            nodes = []
            for nid, e in agg.items():
                misc = ".misc:" in nid
                blk = nid.split(".misc:")[0] if misc else BLOCK_OF.get(nid, nid.split(".")[0])
                lab = (nid.split(".misc:")[1] + " (composite)") if misc else LABELS.get(nid, nid)
                nodes.append(
                    {
                        "id": nid,
                        "block": blk,
                        "label": lab,
                        "total_ns": e["total_ns"],
                        "count": e["count"],
                        "misc": misc,
                        "order": e["order"],
                    }
                )
                blocks[blk] = blocks.get(blk, 0.0) + e["total_ns"]
            result["nodes"][variant][mode][scen] = sorted(nodes, key=lambda x: -x["total_ns"])
            result["blocks"][variant][mode][scen] = blocks

json.dump(result, open(SP + "/percall.json", "w"))

# ---- validation ----
data = json.load(open(SP + "/perf_data.json"))
ANCHORS = {
    "sparse": [
        ("s8.sdpa", "SparseSDPAOperation"),
        ("s3.score", "IndexerScoreDeviceOperation"),
        ("s3.topk", "TopkLargeIndicesDeviceOperation"),
    ],
    "dense": [("d5.ring", "RingJointSDPADeviceOperation")],
}
print("=== validation: block sums vs scenario total; anchor ops vs summary ===")
for variant in VARIANTS:
    for mode in ["sparse", "dense"]:
        for scen in SCENS:
            if (variant, mode, scen) not in RPT:
                continue
            blks = result["blocks"][variant][mode][scen]
            s = sum(blks.values())
            tot = data["variants"][variant]["modes"][mode][scen]["total_ns"]
            nd = {n["id"]: n for n in result["nodes"][variant][mode][scen]}
            astr = []
            for nid, code in ANCHORS[mode]:
                got = nd.get(nid, {}).get("total_ns", 0)
                ref = next(
                    (o["total_ns"] for o in data["variants"][variant]["modes"][mode][scen]["ops"] if o["op"] == code), 0
                )
                astr.append(f"{nid.split('.')[1]} {got/1e6:.2f}vs{ref/1e6:.2f}")
            flag = "" if abs(s - tot) < 1 else "  <<< MISMATCH"
            print(
                f"{variant:12} {mode}/{scen}: sum {s/1e6:8.2f} / total {tot/1e6:8.2f}  diff {abs(s-tot):.0f}ns | {astr}{flag}"
            )

# eyeball GLM s3/s8 per-node distribution (the variant-specific blocks)
print("\n=== glm_5_1 sparse/warm s3+s8 node distribution ===")
for n in sorted(result["nodes"]["glm_5_1"]["sparse"]["warm"], key=lambda x: x["order"]):
    if n["block"] in ("s3", "s8"):
        print(
            f"   {n['block']:3} {n['id']:16} {n['label'][:32]:32} {n['total_ns']/1e3:8.1f}us x{n['count']} {'MISC' if n['misc'] else ''}"
        )
print("wrote percall.json")
