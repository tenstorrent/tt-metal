# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Augmented decode roofline calculator for Qwen3.6-27B on Blackhole Galaxy.

CPU-only. No device required. Computes, per token (B=1 decode):

    t_token  ~=  t_roofline(weights + KV + GDN-state)         # hard DRAM-BW floor
               + N_ccl_total * t_ccl                          # communication latency
               + socket_hops * t_socket                       # pipeline boundary hops

The roofline term is a hard lower bound (bytes / aggregate-bandwidth). The CCL
term is what the roofline OMITS and is the real decode bottleneck; its
coefficients are anchored to two hardware measurements from this repo:

  * G=4 single-axis MLP, traced+warmed: 0.342 ms/layer @ bf8
    (0.12 ms DRAM floor -> ~0.22 ms is 2 CCLs -> ~0.11 ms/CCL)
  * TP=32 current impl (2D), measured full-model decode: ~24 tok/s (~41.7 ms/tok)
    (1.5 ms weight floor -> ~40 ms is CCL/dispatch over ~6-8 CCL/layer -> ~0.08-0.1 ms/CCL)

Both anchors imply a roughly RING-SIZE-INDEPENDENT per-CCL latency ~0.1 ms at
these decode sizes (B=1 is launch/sync-latency bound, not bandwidth bound), which
is the central finding of the G-sweep.

Run:
    python models/demos/qwen3_6_galaxy_v2/tests/g4_roofline_calc.py
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- Qwen3.6-27B geometry (from HF config.json text_config) ----
H = 5120  # hidden_size
I = 17408  # intermediate_size
N_LAYERS = 64
N_FULL_ATTN = 16  # full_attention_interval=4 -> 16 of 64
N_GDN = N_LAYERS - N_FULL_ATTN  # 48 linear_attention layers
VOCAB = 248320

# Full attention
N_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
ATTN_OUTPUT_GATE = True  # attn_output_gate=true

# Linear attention (Gated DeltaNet)
GDN_CONV_KERNEL = 4
GDN_KEY_HEAD_DIM = 128
GDN_NUM_KEY_HEADS = 16
GDN_VALUE_HEAD_DIM = 128
GDN_NUM_VALUE_HEADS = 48

BYTES = {"bf16": 2.0, "bf8": 1.0, "bf4": 0.5}

# ---- Hardware ----
BW_CHIP = 512 * (1024**3)  # 512 GiB/s per chip (project constant; validate vs BH spec)

# ---- CCL model (anchored to measured numbers above) ----
T_CCL_MS = 0.10  # per-CCL latency at B=1 decode (~ring-independent, launch/sync bound)
T_SOCKET_MS = 0.03  # per pipeline-boundary D2D socket hop (small)
# CCLs per layer by parallelization style:
#   2D TP (TP=32): MLP(RS+AG+AR ~3) + attn(QKV AG/RS + out AR ~3-4) ~= 7
#   1D TP (G-stage pipeline): MLP(AG in + RS out = 2) + attn(~2-3) ~= 4
N_CCL_PER_LAYER_2D = 7
N_CCL_PER_LAYER_1D = 4


# ---- Weight parameter counts (analytic, per decode read) ----
def mlp_params_per_layer() -> int:
    return 3 * H * I  # gate + up + down


def full_attn_params_per_layer() -> int:
    q = H * (N_HEADS * HEAD_DIM)
    k = H * (N_KV_HEADS * HEAD_DIM)
    v = H * (N_KV_HEADS * HEAD_DIM)
    o = (N_HEADS * HEAD_DIM) * H
    gate = (N_HEADS * HEAD_DIM) * H if ATTN_OUTPUT_GATE else 0
    return q + k + v + o + gate


def gdn_params_per_layer() -> int:
    qk = 2 * (GDN_NUM_KEY_HEADS * GDN_KEY_HEAD_DIM)  # q,k channels
    v = GDN_NUM_VALUE_HEADS * GDN_VALUE_HEAD_DIM
    in_proj = H * (qk + v)  # project hidden -> q,k,v
    conv = (qk + v) * GDN_CONV_KERNEL  # depthwise causal conv (small)
    out_gate = H * v  # swish output gate
    out_proj = v * H
    return in_proj + conv + out_gate + out_proj


def lm_head_params() -> int:
    return VOCAB * H  # read once per token for the final projection


def decode_weight_params() -> int:
    layers = (
        mlp_params_per_layer() * N_LAYERS + full_attn_params_per_layer() * N_FULL_ATTN + gdn_params_per_layer() * N_GDN
    )
    return layers + lm_head_params()


# ---- KV / state bytes per token ----
def kv_bytes(seqlen: int, kv_dtype: str = "bf16") -> float:
    # full-attn layers only; read whole cache for S positions
    per = 2 * N_KV_HEADS * HEAD_DIM * seqlen  # K + V
    return per * N_FULL_ATTN * BYTES[kv_dtype]


def gdn_state_bytes(state_dtype: str = "bf16") -> float:
    # recurrent state [num_v_heads, key_head_dim, value_head_dim] + conv state
    recurrent = GDN_NUM_VALUE_HEADS * GDN_KEY_HEAD_DIM * GDN_VALUE_HEAD_DIM
    conv = (2 * GDN_NUM_KEY_HEADS * GDN_KEY_HEAD_DIM + GDN_NUM_VALUE_HEADS * GDN_VALUE_HEAD_DIM) * (GDN_CONV_KERNEL - 1)
    rw = 2  # read + write each token
    return (recurrent + conv) * N_GDN * BYTES[state_dtype] * rw


@dataclass
class Roofline:
    weight_ms: float
    kv_ms: float
    state_ms: float
    roofline_ms: float
    ccl_ms: float
    socket_ms: float
    total_ms: float
    roofline_tok_s: float
    total_tok_s: float


def compute(
    n_chips: int,
    weight_dtype: str,
    *,
    seqlen: int = 4096,
    kv_dtype: str = "bf16",
    state_dtype: str = "bf16",
    ccl_per_layer: int = N_CCL_PER_LAYER_1D,
    socket_hops: int = 0,
    t_ccl_ms: float = T_CCL_MS,
) -> Roofline:
    agg_bw = n_chips * BW_CHIP
    w_ms = decode_weight_params() * BYTES[weight_dtype] / agg_bw * 1000.0
    kv_ms = kv_bytes(seqlen, kv_dtype) / agg_bw * 1000.0
    st_ms = gdn_state_bytes(state_dtype) / agg_bw * 1000.0
    roof = w_ms + kv_ms + st_ms
    ccl_ms = ccl_per_layer * N_LAYERS * t_ccl_ms
    sock_ms = socket_hops * T_SOCKET_MS
    total = roof + ccl_ms + sock_ms
    return Roofline(
        weight_ms=w_ms,
        kv_ms=kv_ms,
        state_ms=st_ms,
        roofline_ms=roof,
        ccl_ms=ccl_ms,
        socket_ms=sock_ms,
        total_ms=total,
        roofline_tok_s=1000.0 / roof,
        total_tok_s=1000.0 / total,
    )


# Config presets: (label, n_chips_per_TP_group, ccl_per_layer, socket_hops)
#   TP=32 : one 32-wide TP group, 2D CCLs, no pipeline hops
#   G=8/4/2 : pipeline of 32/G stages, 1D TP per stage, (stages-1) socket hops
PRESETS = [
    ("TP=32 (current 2D)", 32, N_CCL_PER_LAYER_2D, 0),
    ("G=8  pipe x4", 8, N_CCL_PER_LAYER_1D, 3),
    ("G=4  pipe x8", 4, N_CCL_PER_LAYER_1D, 7),
    ("G=2  pipe x16", 2, N_CCL_PER_LAYER_1D, 15),
]


def print_sheet(seqlen: int = 4096) -> None:
    params = decode_weight_params()
    print("=" * 92)
    print("Qwen3.6-27B DECODE ROOFLINE (B=1)  —  per-token, seqlen =", seqlen)
    print(
        f"decode weight params (layers + lm_head): {params/1e9:.2f} B  "
        f"(MLP {mlp_params_per_layer()*N_LAYERS/1e9:.1f}B, "
        f"GDN {gdn_params_per_layer()*N_GDN/1e9:.1f}B, "
        f"full-attn {full_attn_params_per_layer()*N_FULL_ATTN/1e9:.1f}B, "
        f"lm_head {lm_head_params()/1e9:.1f}B)"
    )
    print(
        f"per-chip DRAM BW: {BW_CHIP/1e9:.1f} GB/s ;  t_ccl={T_CCL_MS} ms ;  " f"KV/state read minor at seqlen={seqlen}"
    )
    print("=" * 92)
    hdr = f"{'config':<20} {'prec':<5} {'roof_ms':<9} {'tok/s_roof':<11} {'ccl_ms':<8} {'sock':<6} {'tot_ms':<8} {'tok/s':<8} {'70?':<4}"
    for prec in ("bf16", "bf8", "bf4"):
        print("-" * 92)
        print(hdr)
        for label, nchips, ccl_pl, hops in PRESETS:
            r = compute(nchips, prec, seqlen=seqlen, ccl_per_layer=ccl_pl, socket_hops=hops)
            ok = "YES" if r.total_tok_s >= 70 else "no"
            print(
                f"{label:<20} {prec:<5} {r.roofline_ms:<9.2f} {r.roofline_tok_s:<11.0f} "
                f"{r.ccl_ms:<8.2f} {r.socket_ms:<6.2f} {r.total_ms:<8.2f} {r.total_tok_s:<8.0f} {ok:<4}"
            )
    print("=" * 92)
    print("roof = hard DRAM-bandwidth floor (bytes/agg-BW).  tot = roof + CCL + socket.")
    print("CCL term dominates everywhere -> bandwidth (and thus G) is NOT the lever; CCL count is.")
    print("=" * 92)


# ---- CCL fusion scenarios (TP=32, the config actually running) ----
# Per-layer CCL inventory of the CURRENT decode path (USE_PREFETCHER=False),
# read from the qwen3_6_galaxy_v2 tt/ source:
#   norm (x2 fused_rms_minimal):              2  (already fused: rmsnorm+AG)
#   MLP: w1 RS + w3 RS + ff AG + w2 AR(=RS+AG interleaved):  ~5
#   mixer GDN out-proj all_reduce (48 lyr):   ~2  (+ input gather)
#   mixer full-attn (16 lyr): rs_create_heads + all_gather_concat + WO AR: ~3
# Weighted avg over 64 layers ~= 7 CCL/layer (matches the TP=32 ~24 tok/s anchor).
FUSION_SCENARIOS = [
    (
        "current (USE_PREFETCHER=off)",
        7.0,
        "separate w1/w3 linear + 2 RS; w2 AR decomposes to RS+AG on interleaved input",
    ),
    (
        "enable double_matmul_rs (MLP)",
        5.5,
        "fuse w1+w3 matmul+RS via llama_rs_matmul (needs global_cb/prefetcher on BH)",
    ),
    ("+ keep w2 sharded (1-op AR)", 4.5, "feed w2 all_reduce a sharded input -> all_reduce_async (1 op, not RS+AG)"),
    (
        "+ fuse QKV/WO create-heads",
        3.5,
        "llama_rs_create_heads + all_gather_concat already exist; ensure both on qwen36 path",
    ),
]


def print_fusion_table(prec: str = "bf8") -> None:
    print("=" * 92)
    print(f"TP=32 CCL-FUSION PROJECTION  (prec={prec}, seqlen=4096)  —  target 70 tok/s = 14.3 ms")
    print("=" * 92)
    print(f"{'scenario':<34} {'ccl/lyr':<8} {'ccl_ms':<8} {'tot_ms':<8} {'tok/s':<7} {'70?':<4}")
    print("-" * 92)
    for label, ccl_pl, _note in FUSION_SCENARIOS:
        r = compute(32, prec, ccl_per_layer=ccl_pl, socket_hops=0)
        ok = "YES" if r.total_tok_s >= 70 else "no"
        print(f"{label:<34} {ccl_pl:<8.1f} {r.ccl_ms:<8.2f} {r.total_ms:<8.2f} {r.total_tok_s:<7.0f} {ok:<4}")
    print("-" * 92)
    # break-even: how few CCL/layer to hit 70 tok/s?
    floor = compute(32, prec, ccl_per_layer=0, socket_hops=0).roofline_ms
    budget_ms = 1000.0 / 70.0 - floor
    max_ccl = budget_ms / (N_LAYERS * T_CCL_MS)
    print(
        f"to hit 70 tok/s: CCL budget = {budget_ms:.2f} ms -> <= {max_ccl:.2f} CCL/layer "
        f"(or cut t_ccl from {T_CCL_MS} to {budget_ms/(N_LAYERS*7):.3f} ms at 7 CCL/layer)"
    )
    for label, ccl_pl, note in FUSION_SCENARIOS:
        print(f"  - {label}: {note}")
    print("=" * 92)


if __name__ == "__main__":
    print_sheet(seqlen=4096)
    print()
    print_fusion_table("bf8")
