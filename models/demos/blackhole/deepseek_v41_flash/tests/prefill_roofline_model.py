"""CPU-only PREFILL roofline of DeepSeek-V4.1-Flash on the 4x8 Blackhole Galaxy (32 chips). python tests/prefill_roofline_model.py
Peaks (tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: 1.35 GHz, 130 compute cores, per-engine LoFi 5.4 / HiFi2 2.7 / HiFi4 1.35 TFLOP/s;
measured best matmul ~580 TFLOP/s on P150): PEAK below; EFF = achievable fraction (assumption, GEMM report: up to 96% on ideal shapes)."""
D, L, H, HD, QL, OG, OL, WIN = 5120, 40, 64, 512, 1280, 8, 1024, 128
FF, NE, K, NS = 2304, 384, 6, 1
CHIPS = 32
PEAK = {"LoFi": 700.0, "HiFi2": 350.0, "HiFi4": 175.0}  # TFLOP/s per chip, 130 cores
EFF = 0.5  # fraction of peak reached by real multi-op layers (assumption; the measured DSv3 prefill block numbers are far lower)
DRAM = 450.0  # GB/s effective per chip (measured by the decode roofline)
ratios = [0, 0, 2] + [2] * 17 + [1] * 20  # compress ratio per layer (config.compress_ratios[:40])
idx_layers = [2, 8, 14, 20, 24, 28, 32, 36]

moe = (K + NS) * 3 * D * FF * 2  # routed top-6 + shared, FLOP/token/layer
attn_proj = 2 * (D * (QL + HD) + QL * H * HD + (H * HD // OG) * OG * OL + OG * OL * D)
router = 2 * D * NE
mhc = 2 * 2 * (4 * D) * 24 + 2 * 4 * 24 * 8  # two mHC: mixes matmul [4D -> 24] + collapse/expand
engram = 2 * 6144 * 3200 * 2 if False else 2 * 6144 * 5120
comp = lambda r: 0 if r == 0 else 2 * D * (2 * HD if r == 2 else HD)  # compressor projection


def attn_core(ctx, r, topk=512):
    """per query token SDPA FLOPs: window + (all compressed entries if <= topk else topk). QK + PV, 64 heads, d=512 (K==V)."""
    n = WIN + min(ctx // r if r else 0, topk)
    return 2 * 2 * H * HD * n


def idx_flops(ctx, r):
    """indexer scoring per query token at position ~ctx: 32 heads x 128 dim over ctx/r keys, + q proj (QL x 32x128) + weights proj."""
    return 2 * 32 * 128 * (ctx // r) + 2 * QL * 32 * 128 + 2 * D * 32


def per_token(ctx):
    t = {
        "routed+shared MoE": moe * L,
        "attn projections": attn_proj * L,
        "router": router * L,
        "mHC": mhc * L,
        "compressors": sum(comp(r) for r in ratios),
        "Engram dev (2 layers)": engram * 2,
        "head (last token only)": 0,
    }
    t["SDPA core (avg ctx/2)"] = sum(attn_core(ctx // 2, r) for r in ratios)
    # indexer: layers whose r>0 and are index sources (+ the readers re-use the top-k)
    t["indexer scoring (avg ctx/2)"] = sum(idx_flops(ctx // 2, ratios[l]) for l in idx_layers if ratios[l])
    return t


if __name__ == "__main__":
    print("FLOP/token by component (GFLOP), 40 layers:")
    for ctx in (4096, 65536, 262144, 1 << 20):
        t = per_token(ctx)
        tot = sum(t.values())
        print(f"\nctx {ctx}: total {tot / 1e9:.1f} GFLOP/token")
        for k, v in t.items():
            print(f"  {k:30s} {v / 1e9:8.3f}  ({100 * v / tot:4.1f}%)")
        for fid, pk in PEAK.items():
            r = pk * EFF * 1e12 * CHIPS / tot
            print(
                f"  -> {fid}: peak {pk:.0f} x EFF {EFF}: {r:9.0f} tok/s all-mesh; 1M tokens in {1 << 20 and (1 << 20) / r:7.1f} s"
            )
    # expert weight sweep: all weights read once per MoE call per chip
    wbytes = NE * 3 * D * FF * 1.0625 / CHIPS
    print(
        f"\nexpert weights/chip/layer bfp8 {wbytes / 1e6:.0f} MB -> sweep {wbytes / DRAM / 1e6:.2f} ms at {DRAM} GB/s"
    )
    for toks in (128, 512, 2048, 8192, 16384, 32768):
        flops = toks * moe / CHIPS
        for fid in ("LoFi", "HiFi2"):
            tc = flops / (PEAK[fid] * EFF * 1e12) * 1e3
            print(
                f"  MoE call of {toks:6d} tokens (global): compute {tc:6.2f} ms ({fid}) vs sweep {wbytes / DRAM / 1e6:.2f} ms -> {'DRAM' if tc < wbytes / DRAM / 1e6 else 'compute'}-bound; "
                f"tokens/expert avg {toks * K / NE:.1f}"
            )
