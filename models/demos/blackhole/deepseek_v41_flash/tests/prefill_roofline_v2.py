"""CPU-only prefill roofline per 512-token window (prefill_opt). python tests/prefill_roofline_v2.py
Config from /mnt/tt-data/ssinghal/deepseek-v41-flash-1017/config.json. Layer types (compress_ratios in the config: 0 = window only, 2 = ratio-2 compressor + indexer
on layers 2/8/14, 1 = uncompressed latents + indexer on 20/24/28/32/36; layers 41.. are MTP/drafter and are not prefilled).
Peaks: tech_reports/GEMM_FLOPS/GEMM_FLOPS.md (BH 1.35 GHz, 130 Tensix = 13x10; per-engine LoFi 5.4 / HiFi2 2.7 / HiFi4 1.35 TFLOP/s; measured best 580 TFLOP/s
on P150 (bf8 HiFi2 / bf4 LoFi)). DRAM: 511.6 GB/s per chip measured (blaze_meas/RESULTS.md)."""
D, H, HD, QL, OG, OL, WIN = 5120, 64, 512, 1280, 8, 1024, 128
FF, NE, K, NS = 2304, 384, 6, 1
IH, ID, TOPK = 32, 128, 512
CHIPS, ROWS, COLS = 32, 4, 8
PEAK = {"LoFi": 130 * 5.4, "HiFi2": 130 * 2.7, "HiFi4": 130 * 1.35}  # TFLOP/s per chip, all 130 cores, datasheet
DRAM = 511.6e9
RATIO = [0, 0] + [2] * 18 + [1] * 20
IDX = {2, 8, 14, 20, 24, 28, 32, 36}
BPP = 1.0625  # bfp8 bytes per parameter


def flops_layer(L, pos):
    """FLOP per token of layer L at prompt position pos, by group."""
    r = RATIO[L]
    g = {}
    g["moe_routed"] = K * 3 * D * FF * 2
    g["moe_shared"] = NS * 3 * D * FF * 2
    g["router"] = 2 * D * NE
    g["attn_proj"] = 2 * (D * QL + QL * H * HD + D * HD + (H * HD // OG) * OG * OL + OG * OL * D)
    g["mhc"] = 2 * (2 * (4 * D) * 24)
    nk = pos // r if r else 0
    g["sdpa"] = 4 * H * HD * (WIN + min(TOPK, nk))
    g["compressor"] = 0 if r == 0 else 2 * D * (2 * HD if r == 2 else HD)
    g["indexer"] = (2 * QL * IH * ID + 2 * D * IH + 2 * IH * ID * nk + 2 * D * ID) if L in IDX else 0
    return g


def window(pos, tokens_per_row, layers):
    """FLOP per CHIP of one window (all FLOP split evenly over the 32 chips: MoE expert-parallel, attention TP over the 8 columns x tokens over the 4 rows)."""
    tot = {}
    for L in layers:
        for k, v in flops_layer(L, pos).items():
            tot[k] = tot.get(k, 0) + v * tokens_per_row * ROWS / CHIPS
    return tot


def dram_floor_layer_bytes(tokens_per_row):
    """Bytes per chip per layer: routed experts (all 384 active at >= 1024 tokens/row: 24.6k assignments), shared expert + attention weights TP over 8 columns."""
    routed = NE * 3 * D * FF * BPP / CHIPS
    shared = NS * 3 * D * FF * BPP / COLS
    attn = (D * QL + QL * H * HD + D * HD + (H * HD // OG) * OG * OL + OG * OL * D) * BPP / COLS
    return routed, shared, attn


if __name__ == "__main__":
    A, Bz = list(range(0, 21)), list(range(21, 40))
    for tpr, name in (
        (1024, "Up=2 (B=8): 1024 tokens/row, 4096/window"),
        (512, "Up=1 (B=4/16): 512 tokens/row, 2048/window"),
    ):
        print(f"\n== {name}")
        routed, shared, attn = dram_floor_layer_bytes(tpr)
        db = (routed + shared + attn) / DRAM
        print(
            f"DRAM floor per layer per window: routed {routed / 1e6:.0f} MB + shared {shared / 1e6:.1f} + attn {attn / 1e6:.1f} -> {db * 1e3:.2f} ms"
        )
        for pos in (4096, 32768, 131072):
            for lab, lay in (("A (L0-20)", A), ("B (L21-39)", Bz)):
                w = window(pos, tpr, lay)
                tot = sum(w.values())
                rows = ", ".join(f"{k} {v / tot * 100:.0f}%" for k, v in sorted(w.items(), key=lambda kv: -kv[1])[:4])
                t = {f: tot / (PEAK[f] * 1e12) * 1e3 for f in PEAK}
                dram = len(lay) * db * 1e3
                print(
                    f"pos {pos:6d} {lab:11s}: {tot / 1e9:7.1f} GFLOP/chip; compute-only at 100%: LoFi {t['LoFi']:.2f} / HiFi2 {t['HiFi2']:.2f} ms; "
                    f"at 65% HiFi2 {t['HiFi2'] / 0.65:.2f} ms; DRAM-weights floor {dram:.1f} ms | mix: {rows}"
                )
    print("\nMeasured (OPTIMIZATIONS 8.6): trace A 190 ms (Up=2, 21 layers), A+B 330 ms.")
