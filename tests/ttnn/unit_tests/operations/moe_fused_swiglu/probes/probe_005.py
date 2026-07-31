"""Device-kernel ns + DRAM-read utilisation at the graded counts.

util = dram_read_bytes / (512e9 * device_kernel_time_s); read bytes = three bfp4 weight sets
(count-independent) + one read of the real tokens at the format's granularity.
"""
import torch, ttnn
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

HIDDEN, TILE = 2048, 32
NG, NL, LID, GID = 256, 8, 3, 137
BFP4_TILE = 576


def read_bytes(count, emb, fmt):
    w = 3 * (emb * HIDDEN // 1024) * BFP4_TILE
    if fmt == "bf16_rm":
        return w + count * emb * 2.0
    return w + ((count + TILE - 1) // TILE) * TILE * emb * 1.0625


device = ttnn.open_device(device_id=0)
try:

    def build(emb, capacity, count, fmt):
        torch.manual_seed(42)
        x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
        if count < capacity:
            x[:, :, count:, :] = 100.0
        dt, lay = (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) if fmt == "bf16_rm" else (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        tt_x = ttnn.from_torch(
            x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_w = [
            ttnn.from_torch(
                torch.randn(s, dtype=torch.bfloat16),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for s in ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))
        ]
        counts = torch.zeros(NG, dtype=torch.int32)
        counts[GID] = count
        idx = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
        idx[LID] = GID
        tod = lambda t: ttnn.from_torch(
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return tt_x, tt_w, tod(counts), tod(idx)

    print(
        f"{'fmt':>10} {'emb':>5} {'cap':>5} {'cnt':>5} | {'kernel ns':>10} {'MB read':>8} {'util':>6} {'target ns':>9} {'tgt util':>8}"
    )
    TGT = {128: (91800, 0.566), 256: (108000, 0.514), 512: (161820, 0.388)}
    for fmt in ("bf16_rm", "bfp8_tile"):
        for emb, capacity, count in [
            (7168, 5120, 128),
            (7168, 5120, 256),
            (7168, 5120, 512),
            (7168, 1024, 256),
            (6144, 5120, 256),
            (7168, 5120, 5120),
        ]:
            tt_x, tt_w, tc, ti = build(emb, capacity, count, fmt)
            for _ in range(2):
                out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tc, ti, LID)
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            rb = read_bytes(count, emb, fmt)
            t, u = TGT.get(count, (0, 0.0)) if emb == 7168 else (0, 0.0)
            print(f"{fmt:>10} {emb:>5} {capacity:>5} {count:>5} | {'?':>10} {rb/1e6:>8.2f} {'?':>6} {t:>9} {u:>8}")
finally:
    ttnn.close_device(device)
