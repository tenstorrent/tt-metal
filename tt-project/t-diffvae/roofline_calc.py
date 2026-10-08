"""#257 DiffVAE 1080p 145f 4x8 roofline. Per chip, per stage-5 block unless noted.

Assumptions (edit and rerun):
  BH chip: 130 Tensix cores at 1.35 GHz, 4096 FLOP/cycle/core at LoFi (= the 774 TF
  FP8 spec at 140 cores), HiFi2 half that. DRAM 512 GB/s spec; the SDPA K/V stream
  saturates at ~270-290 GB/s chip-wide (tt-metal #56691), NA measured 224 GB/s (#253).
"""

CORES, CLK = 130, 1.35e9
PEAK_LOFI = CORES * CLK * 4096
PEAK_HIFI2 = PEAK_LOFI / 2
DRAM_STREAM = 280e9
TILE = 2048  # bf16 32x32

# Stage-5 geometry per chip: S5_2D splits H/4 x W/8; all 4 heads on chip.
T, H, W, HALO = 145, 272 // 4, 480 // 8, 5
SITES = T * H * W
HEADS, HD, D = 4, 64, 256
CHUNKS, KBRICKS = 9308, 112  # measured (#253): query chunks per chip, key bricks per chunk
BRICK_TILES = HD // 32  # one 32-site brick of one head = 2 tiles

kv_read = CHUNKS * HEADS * 2 * KBRICKS * BRICK_TILES * TILE
kv_unique = T * (H + 2 * HALO) * (W + 2 * HALO) * D * 2 * 2
na_flop = CHUNKS * HEADS * 64 * KBRICKS * 32 * HD * 2 * 2
na_exact_flop = SITES * HEADS * 11**3 * HD * 2 * 2
# TT flash SDPA at d64: matmul ~40% of cycles, softmax (exp, max, sum, rescale) the rest.
na_compute = na_flop / PEAK_HIFI2 / 0.4


def reads_per_chunk(group_h, group_w):
    """Key bricks fetched per query chunk when a core keeps the union of a (1, gh, gw)
    group of query chunks resident: 7 x (gh+3) x (gw+3) bricks for gh*gw chunks."""
    return 7 * (group_h + 3) * (group_w + 3) / (group_h * group_w)


print(f"stage-5 sites/chip {SITES:,}")
print(f"NA K/V read {kv_read/1e9:.1f} GB, unique {kv_unique/1e9:.2f} GB, re-read x{kv_read/kv_unique:.0f}")
print(f"NA DRAM time at {DRAM_STREAM/1e9:.0f} GB/s: {kv_read/DRAM_STREAM*1e3:.0f} ms (measured 152.7)")
print(f"NA FLOP {na_flop:.2e} (exact {na_exact_flop:.2e}, x{na_flop/na_exact_flop:.1f} brick over-gather)")
print(f"NA compute floor (HiFi2, 40% mm share): {na_compute*1e3:.1f} ms")
for name, gh, gw in [("1-D W-row ring", 1, 15), ("2-D 4x15 group", 4, 15), ("full plane", 17, 15)]:
    r = reads_per_chunk(gh, gw)
    t = kv_read * r / KBRICKS / DRAM_STREAM
    print(f"  {name:16s}: {r:5.1f} bricks/chunk -> DRAM {t*1e3:5.1f} ms, NA ~{max(t, na_compute)*1e3:5.1f} ms")
    if name.startswith("1-D"):
        print(f"  {'':16s}  + bf8 K/V: DRAM {t/2*1e3:5.1f} ms, NA ~{max(t/2, na_compute)*1e3:5.1f} ms")

lin = SITES * (3 * D * D + D * D + 12 * D * D) * 2
act = SITES * D * 2
print(f"stage-5 linears {lin:.2e} FLOP -> {lin/PEAK_HIFI2*1e3:.1f} ms HiFi2; one 256-ch tensor {act/1e6:.0f} MB")
print(f"  unfused SwiGLU intermediate r+w {SITES*2048*2*2/1e9:.1f} GB -> {SITES*2048*4/DRAM_STREAM*1e3:.0f} ms")

# Det stages: (sites over the whole video, dim, blocks, chips sharing the work today)
det = {"stage1": (21 * 34 * 60, 2048, 4, 1), "stage2": (21 * 68 * 120, 1024, 6, 8),
       "stage3": (41 * 68 * 120, 512, 4, 8), "stage4": (81 * 136 * 240, 512, 2, 8)}
for k, (s, d, n, ways) in det.items():
    f = s * 16 * d * d * 2 * n
    print(f"{k}: {f:.1e} FLOP; HiFi2 {f/ways/PEAK_HIFI2*1e3:5.1f} ms at {ways}-way split today, "
          f"{f/32/PEAK_HIFI2*1e3:4.1f} ms at 32-way")
