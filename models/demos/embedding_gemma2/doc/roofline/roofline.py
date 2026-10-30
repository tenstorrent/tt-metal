#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Practical roofline for google/embeddinggemma-2 prefill on one Blackhole P150, 12 x 10 grid, batch 1. CPU only.

    python3 roofline.py                 # per-shape table + op-family tables (text 2048, text 32768, vision 1x2560)
    python3 roofline.py --families all  # op-family tables for every shape

Method (JAX scaling book, roofline chapter), per op family:
    compute_ideal = FLOPs / peak FLOP/s (peak at the family's math fidelity)
    memory_ideal  = bytes moved / DRAM bandwidth
    label         = COMPUTE if FLOPs / bytes > ridge (peak / bandwidth), else MEMORY
    practical     = max(compute_ideal / 0.5, memory_ideal / 0.8)
Shape practical roofline = sum of the families' practical times.

Bytes: every op reads its inputs and writes its outputs once (activations bf16), weights once in the dtype of the
selected precision config. Attention reads Q, K, V and writes O once (scores stay on chip); sliding layers count only
the visible 32 x 32 blocks of the |q - k| <= 512 band. Norms, eltwise and rotary are MEMORY (bytes in + out).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRECISION = HERE.parent / "datatype_sweep" / "selected_precision_config.json"

# ------------------------------------------------------------------------------------------- hardware
CORES = 120  # 12 x 10 grid (Galaxy target)
CLK_HZ = 1.35e9  # AICLK
FLOP_PER_CYCLE_LOFI = 4096  # per core, 2 * 8 * 16 * 16 (tech_reports/GEMM_FLOPS; roofline session roofline.py)
PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}  # math-fidelity passes per multiply
DRAM_BPS = 512e9  # P150 DRAM bandwidth (datasheet)
COMPUTE_CEIL = 0.5  # practical share of peak compute
MEMORY_CEIL = 0.8  # practical share of DRAM bandwidth
BYTES = {"bf16": 2.0, "fp32": 4.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
ACT = BYTES["bf16"]


def peak_flops(fidelity: str) -> float:
    return CORES * CLK_HZ * FLOP_PER_CYCLE_LOFI / PASSES[fidelity]


# ------------------------------------------------------------------------------------------- measured
# Sustained ms per traced forward at 12 x 10 (doc/prefill/04-attention/README.md,
# doc/prefill_towers/02-attention/README.md).
MEASURED = [
    ("text", 128, 4.837),
    ("text", 256, 5.512),
    ("text", 512, 7.130),
    ("text", 1024, 11.279),
    ("text", 2048, 19.235),
    ("text", 4096, 39.701),
    ("text", 8192, 90.479),
    ("text", 16384, 207.514),
    ("text", 32768, 526.523),
    ("text", 65536, 1577.429),
    ("vision", 2560, 15.686),
    ("audio", 256, 7.267),
    ("audio", 768, 11.141),
]
FAMILY_SHAPES = [("text", 2048), ("text", 32768), ("vision", 2560)]


# ------------------------------------------------------------------------------------------- ledger
class Ledger:
    """Per op family: FLOPs, bytes and ideal compute seconds (each op at its own fidelity)."""

    def __init__(self):
        self.f = OrderedDict()

    def _g(self, fam):
        return self.f.setdefault(fam, {"flops": 0.0, "bytes": 0.0, "compute_s": 0.0})

    def matmul(self, fam, M, K, N, wdtype, fid, out=ACT, weight=True):
        g = self._g(fam)
        fl = 2.0 * M * K * N
        g["flops"] += fl
        g["compute_s"] += fl / peak_flops(fid)
        g["bytes"] += M * K * ACT + M * N * out + (K * N * BYTES[wdtype] if weight else 0.0)

    def attn(self, fam, flops, nbytes, fid):
        g = self._g(fam)
        g["flops"] += flops
        g["compute_s"] += flops / peak_flops(fid)
        g["bytes"] += nbytes

    def mem(self, fam, nbytes):
        self._g(fam)["bytes"] += nbytes

    def elt(self, fam, M, width, n_in=1, n_out=1, dtype=ACT):
        """Eltwise / norm pass over [M, width]: n_in reads and n_out writes."""
        self.mem(fam, (n_in + n_out) * M * width * dtype)


def finish(led: Ledger) -> OrderedDict:
    rows = OrderedDict()
    for fam, g in led.f.items():
        mem_s = g["bytes"] / DRAM_BPS
        c_s = g["compute_s"]
        label = "COMPUTE" if c_s > mem_s else "MEMORY"  # same as intensity > ridge at the family's peak
        eff_peak = g["flops"] / c_s if c_s else peak_flops("HiFi2")
        rows[fam] = {
            "flops": g["flops"],
            "bytes": g["bytes"],
            "intensity": g["flops"] / g["bytes"] if g["bytes"] else 0.0,
            "ridge": eff_peak / DRAM_BPS,
            "label": label,
            "compute_ms": c_s * 1e3,
            "memory_ms": mem_s * 1e3,
            "ideal_ms": max(c_s, mem_s) * 1e3,
            "practical_ms": max(c_s / COMPUTE_CEIL, mem_s / MEMORY_CEIL) * 1e3,
            "limit": "compute 50 %" if c_s / COMPUTE_CEIL > mem_s / MEMORY_CEIL else "memory 80 %",
        }
    return rows


def band_blocks(S: int, radius: int) -> int:
    """32 x 32 (q, k) blocks per head that touch the bidirectional band |q - k| <= radius."""
    T = S // 32
    r = (radius + 31) // 32  # tile distance d is visible when 32 * d - 31 <= radius
    r = max(d for d in range(r + 1) if 32 * d - 31 <= radius)
    return sum(min(T - 1, i + r) - max(0, i - r) + 1 for i in range(T))


# ------------------------------------------------------------------------------------------- towers
QKVO, MLP, ATTN, ELT, EMB, HEAD = "QKV/O proj", "MLP", "attention", "norms/eltwise/rotary", "embedding/PLE/stem", "head"


def text(hf, prec, S):
    tc, P = hf["text_config"], prec["text"]
    w = lambda k: (P["weights"][k]["dtype"], P["weights"][k]["fidelity"])
    H, I, Pd, NL, nq = (
        tc["hidden_size"],
        tc["intermediate_size"],
        tc["hidden_size_per_layer_input"],
        tc["num_hidden_layers"],
        tc["num_attention_heads"],
    )
    sfid = P["sdpa"]["fidelity"]
    M = S
    L = Ledger()
    L.mem(EMB, M * H * ACT)  # embedding gather (rows out)
    L.matmul(EMB, M, H, NL * Pd, *w("ple_proj"))  # projection-only PLE, once
    for i, kind in enumerate(tc["layer_types"]):
        over = tc.get("per_layer_config", {}).get(f"{i:02d}", {})
        hd = over.get("head_dim", tc["head_dim"])
        nkv = over.get("num_key_value_heads", tc["num_key_value_heads"])
        qw, kvw = nq * hd, nkv * hd
        L.elt(ELT, M, Pd)  # PLE slice norm
        L.elt(ELT, M, H)  # input norm
        L.matmul(QKVO, M, H, qw + 2 * kvw, *w("attn_qkv"))
        L.elt(ELT, M, qw + 2 * kvw)  # q / k / v norms
        L.elt(ELT, M, qw + kvw)  # RoPE on q, k
        L.mem(ELT, 2 * M * hd * ACT)  # cos / sin tables
        keys = band_blocks(S, tc["sliding_window"]) * 32 * 32 / S if kind == "sliding_attention" else S
        fl = 2 * 2.0 * M * keys * hd * nq  # Q.K^T and P.V over visible keys
        L.attn(ATTN, fl, (2 * M * qw + 2 * M * kvw) * ACT, sfid)  # Q, K, V in, O out
        L.matmul(QKVO, M, qw, H, *w("attn_o"))
        L.elt(ELT, M, H)  # post-attention norm
        L.elt(ELT, M, H, 2, 1)  # residual add
        L.elt(ELT, M, H)  # pre-FF norm
        L.matmul(MLP, M, H, I, *w("mlp_gate_up"))
        L.matmul(MLP, M, H, I, *w("mlp_gate_up"))
        L.elt(ELT, M, I, 2, 1)  # gelu(gate) * up
        L.matmul(MLP, M, I, H, *w("mlp_down"))
        L.elt(ELT, M, H)  # post-FF norm
        L.elt(ELT, M, H, 2, 1)  # residual add
        L.matmul(EMB, M, H, Pd, *w("ple_block"))  # PLE gate
        L.elt(ELT, M, Pd, 2, 1)  # gelu(gate) * pli
        L.matmul(EMB, M, Pd, H, *w("ple_block"))  # PLE proj
        L.elt(ELT, M, H)  # post-PLE norm
        L.elt(ELT, M, H, 2, 1)  # residual add
        L.elt(ELT, M, H)  # layer scalar
    L.elt(ELT, M, H)  # final norm
    L.matmul(HEAD, M, H, tc["embedding_dim"], *w("final_proj"))
    L.elt(HEAD, M, tc["embedding_dim"], 1, 0)  # mean pool read
    return L


def vision(hf, prec, S):
    vc, P = hf["vision_config"], prec["vision"]
    w = lambda k: (P["weights"][k]["dtype"], P["weights"][k]["fidelity"])
    H, I, NL, nh, hd = (
        vc["hidden_size"],
        vc["intermediate_size"],
        vc["num_hidden_layers"],
        vc["num_attention_heads"],
        vc["head_dim"],
    )
    sfid = P["sdpa"]["fidelity"]
    k2 = vc["pooling_kernel_size"] ** 2
    S_out = -(-(S // k2) // 32) * 32
    M = S
    pin = 3 * vc["patch_size"] ** 2
    L = Ledger()
    L.mem(EMB, M * pin * BYTES["fp32"])  # fp32 pixels
    L.matmul(EMB, M, pin, H, *w("patch_embed"))
    L.mem(EMB, 2 * M * H * ACT)  # position-table gathers
    L.elt(EMB, M, H, 3, 1)  # + pos_x + pos_y
    for _ in range(NL):
        L.elt(ELT, M, H)  # input norm
        L.matmul(QKVO, M, H, 3 * nh * hd, *w("attn_qkv"))
        L.elt(ELT, M, 3 * nh * hd)  # q / k / v norms
        L.elt(ELT, M, 2 * nh * hd)  # axial RoPE on q, k
        L.mem(ELT, 2 * M * hd * ACT)  # cos / sin tables
        L.attn(ATTN, 2 * 2.0 * M * S * hd * nh, 4 * M * nh * hd * ACT, sfid)  # dense, one image
        L.matmul(QKVO, M, nh * hd, H, *w("attn_o"))
        L.elt(ELT, M, H)  # post-attention norm
        L.elt(ELT, M, H, 2, 1)  # residual
        L.elt(ELT, M, H)  # pre-FF norm
        L.matmul(MLP, M, H, I, *w("mlp_gate_up"))
        L.matmul(MLP, M, H, I, *w("mlp_gate_up"))
        L.elt(ELT, M, I, 2, 1)  # gelu(gate) * up
        L.matmul(MLP, M, I, H, *w("mlp_down"))
        L.elt(ELT, M, H)  # post-FF norm
        L.elt(ELT, M, H, 2, 1)  # residual
    # pooler: one-hot [S_out, S] @ h at HiFi4; the matrix is a forward input
    L.matmul(HEAD, S_out, S, H, "bf16", "HiFi4")
    L.elt(HEAD, S_out, H)  # projector norm
    L.matmul(HEAD, S_out, H, hf["text_config"]["hidden_size"], *w("projector"))
    return L


def audio(hf, prec, S):
    ac, P = hf["audio_config"], prec["audio"]
    w = lambda k: (P["weights"][k]["dtype"], P["weights"][k]["fidelity"])
    H, NL, nh = ac["hidden_size"], ac["num_hidden_layers"], ac["num_attention_heads"]
    hd, FF, c0, c1 = H // nh, 4 * H, *ac["subsampling_conv_channels"]
    afid = P["attention"]["fidelity"]
    M, Tf = S, 4 * S  # tokens, mel frames
    L = Ledger()
    # subsampling stem: conv0 1 -> 128 (3 x 3, s2) + LayerNorm, conv1 128 -> 32 (3 x 3, s2) + LayerNorm, proj
    L.mem(EMB, Tf * 128 * BYTES["fp32"])  # mel input
    pos0, pos1 = (Tf // 2) * 64, (Tf // 4) * 32
    L.matmul(EMB, pos0, 9, c0, *w("subsample_conv"))
    L.elt(EMB, pos0, c0, 1, 1)  # LayerNorm + relu
    L.matmul(EMB, pos1, 9 * c0, c1, *w("subsample_conv"))
    L.elt(EMB, pos1, c1, 1, 1)
    L.matmul(EMB, M, 32 * c1, H, *w("subsample_proj"))

    def ffn():
        L.elt(ELT, M, H)  # pre-norm
        L.matmul(MLP, M, H, FF, *w("ffn_up"))
        L.elt(ELT, M, FF)  # silu
        L.matmul(MLP, M, FF, H, *w("ffn_down"))
        L.elt(ELT, M, H)  # post-norm
        L.elt(ELT, M, H, 2, 1)  # residual (x 0.5)

    for _ in range(NL):
        ffn()
        L.elt(ELT, M, H)  # attention pre-norm
        L.matmul(QKVO, M, H, 3 * H, *w("attn_qkv"))
        # band attention as the model evaluates it: 32-query blocks x 64-key context, + rel-position Q.E^T
        blocks = nh * (S // 32)
        fl = blocks * (2.0 * 32 * 64 * hd * 2 + 2.0 * 32 * 32 * hd)
        L.attn(ATTN, fl, 4 * M * H * ACT, afid)
        L.matmul(QKVO, M, H, H, *w("attn_o"))
        L.elt(ELT, M, H)  # post-norm
        L.elt(ELT, M, H, 2, 1)  # residual
        # light conv: norm, W_start (H -> 2H), GLU, depthwise k5, norm, silu, W_end, residual
        L.elt(ELT, M, H)
        L.matmul(QKVO, M, H, 2 * H, *w("lconv_start"))
        L.elt(ELT, M, H, 2, 1)  # GLU
        L.elt(ELT, M, H)  # depthwise conv
        L.elt(ELT, M, H)  # conv norm
        L.elt(ELT, M, H)  # silu
        L.matmul(QKVO, M, H, H, *w("lconv_end"))
        L.elt(ELT, M, H, 2, 1)  # residual
        ffn()
        L.elt(ELT, M, H)  # norm_out
    L.matmul(HEAD, M, H, ac["output_proj_dims"], *w("output_proj"), out=BYTES[P["weights"]["output_proj"]["out_dtype"]])
    L.elt(HEAD, M, ac["output_proj_dims"])  # embedder norm
    L.matmul(HEAD, M, ac["output_proj_dims"], hf["text_config"]["hidden_size"], *w("projector"))
    return L


TOWERS = {"text": text, "vision": vision, "audio": audio}


# ------------------------------------------------------------------------------------------- report
def shape_name(tower, S):
    return f"text {S}" if tower == "text" else f"{tower} 1x{S}"


def summarize(fams):
    total = sum(r["practical_ms"] for r in fams.values())
    by = {"COMPUTE": 0.0, "MEMORY": 0.0}
    for r in fams.values():
        by[r["label"]] += r["practical_ms"]
    dom = max(by, key=by.get)
    return total, dom, by[dom] / total


def find_hf_config() -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    hits = sorted(glob.glob(os.path.join(hub, "models--google--embeddinggemma-2", "snapshots", "*", "config.json")))
    if not hits:
        raise SystemExit("HF config.json for google/embeddinggemma-2 not found; pass --config")
    return Path(hits[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="HF config.json (default: the HF cache snapshot)")
    ap.add_argument("--precision", default=str(PRECISION))
    ap.add_argument("--families", choices=["default", "all"], default="default")
    a = ap.parse_args()
    cfg = Path(a.config) if a.config else find_hf_config()
    hf = json.loads(cfg.read_text())
    prec = json.loads(Path(a.precision).read_text())

    print(f"config {cfg}")
    print(f"precision {Path(a.precision).name} ({prec.get('config_id')})")
    print()
    print("| Hardware number | Value | Source |")
    print("|---|---|---|")
    print(f"| Active cores | {CORES} (12 x 10) | Galaxy target grid (brief) |")
    print(f"| AICLK | {CLK_HZ / 1e9:.2f} GHz | P150 nominal; roofline session `roofline.py` CLK_HZ |")
    print(
        f"| Matmul per core | {FLOP_PER_CYCLE_LOFI} FLOP / cycle at LoFi, / fidelity passes (HiFi2 2, HiFi4 4) | roofline session `roofline.py` FLOP_PER_CYCLE_LOFI; `tech_reports/GEMM_FLOPS` (2 x 8 x 16 x 16) |"
    )
    print(
        f"| Peak compute | LoFi {peak_flops('LoFi') / 1e12:.1f}, HiFi2 {peak_flops('HiFi2') / 1e12:.1f}, HiFi4 {peak_flops('HiFi4') / 1e12:.1f} TFLOP/s | cores x clock x FLOP / cycle |"
    )
    print(f"| DRAM bandwidth | {DRAM_BPS / 1e9:.0f} GB/s | P150 datasheet (brief) |")
    print(
        f"| Ridge point | HiFi2 {peak_flops('HiFi2') / DRAM_BPS:.0f}, HiFi4 {peak_flops('HiFi4') / DRAM_BPS:.0f} FLOP / byte | peak / bandwidth |"
    )
    print(
        f"| Practical ceilings | compute {COMPUTE_CEIL:.0%} of peak (HiFi2 {COMPUTE_CEIL * peak_flops('HiFi2') / 1e12:.1f} TFLOP/s), memory {MEMORY_CEIL:.0%} of bandwidth ({MEMORY_CEIL * DRAM_BPS / 1e9:.0f} GB/s); practical ridge HiFi2 {COMPUTE_CEIL * peak_flops('HiFi2') / (MEMORY_CEIL * DRAM_BPS):.0f} FLOP / byte | person's rule of thumb |"
    )
    print("| Bytes per element | bfp8 1.0625, bfp4 0.5625, bf16 2, fp32 4 | 1088 / 576 B per 32 x 32 tile |")
    print()
    print(
        "| Shape | Practical roofline ms | Measured ms | % of practical (practical / measured) | Dominant bound (share of practical) |"
    )
    print("|---|---|---|---|---|")
    all_fams = {}
    for tower, S, meas in MEASURED:
        fams = finish(TOWERS[tower](hf, prec, S))
        all_fams[(tower, S)] = fams
        total, dom, share = summarize(fams)
        print(
            f"| {shape_name(tower, S)} | {total:.3f} | {meas:.3f} | {100 * total / meas:.0f} % | {dom} ({100 * share:.0f} %) |"
        )

    shapes = FAMILY_SHAPES if a.families == "default" else [(t, s) for t, s, _ in MEASURED]
    for key in shapes:
        fams = all_fams[key]
        total, _, _ = summarize(fams)
        print()
        print(f"**{shape_name(*key)}** (practical {total:.3f} ms)")
        print()
        print(
            "| Op family | GFLOP | MB moved | FLOP / byte | Ridge | Label | Ideal ms | Practical ms | Practical limit |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for fam, r in fams.items():
            print(
                f"| {fam} | {r['flops'] / 1e9:.1f} | {r['bytes'] / 1e6:.1f} | {r['intensity']:.0f} | {r['ridge']:.0f} | "
                f"{r['label']} | {r['ideal_ms']:.3f} | {r['practical_ms']:.3f} | {r['limit']} |"
            )


if __name__ == "__main__":
    main()
