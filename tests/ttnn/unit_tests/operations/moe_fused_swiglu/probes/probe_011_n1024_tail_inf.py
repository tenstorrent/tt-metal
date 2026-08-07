"""Localize the hidden=1024/count=257 non-finite stress result.

This deliberately reproduces the determinism harness's seed and hostile x-padding, then varies
only count, padding contents, activation format, and output format.  It reports non-finites by
token row and output tile and checks whether repeated failure masks are identical.
"""

import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

EMB, HIDDEN, CAPACITY = 7168, 1024, 1024
GRID = (11, 8)
TILE = 32
NUM_GLOBAL, NUM_LOCAL, LOCAL_ID, GLOBAL_ID = 256, 8, 3, 137


def report(label, got, semantic_count):
    got = got.float()
    bad = ~torch.isfinite(got)
    row_bad = bad.any(dim=1)
    tile_bad = bad.reshape(got.shape[0], EMB // TILE, TILE).any(dim=2)
    rows = torch.where(row_bad)[0].tolist()
    tile_coords = torch.nonzero(tile_bad, as_tuple=False)
    finite = got[~bad]
    real_bad = int(bad[:semantic_count].sum())
    pad_bad = int(bad[semantic_count : ((semantic_count + TILE - 1) // TILE) * TILE].sum())
    print(
        f"{label:34s} bad={int(bad.sum()):7d} real_bad={real_bad:7d} pad_bad={pad_bad:7d} "
        f"rows={rows[:12]}{'...' if len(rows) > 12 else ''} finite_max={finite.abs().max().item():.4e}",
        flush=True,
    )
    if tile_coords.numel():
        coords = [(int(r), int(c)) for r, c in tile_coords[:24]]
        per_row = [(r, int(bad[r].sum())) for r in rows[:12]]
        print(f"  first bad (row, output-tile): {coords}", flush=True)
        print(f"  bad elements per first rows: {per_row}", flush=True)
    return got, bad


device = ttnn.open_device(device_id=0)
try:
    # Exact stress-harness RNG order: x first, then gate/up/down weights.
    torch.manual_seed(42)
    x_random = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.float32).to(torch.bfloat16)
    weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))]
    x_sentinel = x_random.clone()
    x_sentinel[:, :, 257:, :] = 100.0
    x_zero = x_random.clone()
    x_zero[:, :, 257:, :] = 0.0

    gu_mc, down_mc = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    tt_w = [
        ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        for w, mc in zip(weights, (gu_mc, gu_mc, down_mc))
    ]
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL for i in range(NUM_LOCAL)], dtype=torch.int32)
    idx[LOCAL_ID] = GLOBAL_ID
    tt_idx = ttnn.from_torch(
        idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    x_specs = {
        "random_bfp8": (x_random, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        "sentinel_bfp8": (x_sentinel, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        "zero_bfp8": (x_zero, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        "sentinel_bf16rm": (x_sentinel, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    }
    tt_x = {
        name: ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for name, (x, dtype, layout) in x_specs.items()
    }

    def run(name, count, out_dtype=ttnn.bfloat8_b):
        counts = torch.zeros(NUM_GLOBAL, dtype=torch.int32)
        counts[GLOBAL_ID] = count
        tt_counts = ttnn.from_torch(
            counts,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = moe_fused_swiglu(tt_x[name], *tt_w, tt_counts, tt_idx, LOCAL_ID, core_grid=GRID, dtype=out_dtype)
        written = ((count + TILE - 1) // TILE) * TILE
        host = ttnn.to_torch(out)[0, 0, :written].clone()
        ttnn.deallocate(out)
        ttnn.deallocate(tt_counts)
        suffix = "bf16out" if out_dtype == ttnn.bfloat16 else "bfp8out"
        return report(f"{name} M={count} {suffix}", host, count)

    base_256, _ = run("sentinel_bfp8", 256)
    random_257, _ = run("random_bfp8", 257)
    random_288, _ = run("random_bfp8", 288)
    zero_257, _ = run("zero_bfp8", 257)
    sentinel_runs = [run("sentinel_bfp8", 257) for _ in range(3)]
    sentinel_bf16rm, _ = run("sentinel_bf16rm", 257)
    sentinel_bf16out, _ = run("sentinel_bfp8", 257, ttnn.bfloat16)

    print("\nCross-run comparisons", flush=True)
    print(f"  M256 == M257 first 256 rows: {torch.equal(base_256, sentinel_runs[0][0][:256])}", flush=True)
    print(f"  random M257 == M288 all 288 rows: {torch.equal(random_257, random_288)}", flush=True)
    for i in range(1, len(sentinel_runs)):
        print(
            f"  sentinel run0/run{i}: bad_mask_equal={torch.equal(sentinel_runs[0][1], sentinel_runs[i][1])} "
            f"values_equal={torch.equal(sentinel_runs[0][0], sentinel_runs[i][0])}",
            flush=True,
        )
    print(
        f"  bfp8-vs-bf16RM bad masks equal: {torch.equal(sentinel_runs[0][1], ~torch.isfinite(sentinel_bf16rm))}",
        flush=True,
    )
    print(f"  BF16 output has bad values: {bool((~torch.isfinite(sentinel_bf16out)).any())}", flush=True)

    # One-row arithmetic references are enough to price the hostile padding without a 288-row
    # host matmul. Use round-tripped BFP4 weights so this is the actual stored weight scale.
    qweights = [ttnn.to_torch(w).float() for w in tt_w]
    for label, row in (("real row 256", x_random[0, 0, 256].float()), ("sentinel row", torch.full((EMB,), 100.0))):
        gate = row @ qweights[0]
        up = row @ qweights[1]
        h = F.silu(gate) * up
        y = h @ qweights[2]
        print(
            f"  host {label:12s}: gate|max|={gate.abs().max().item():.4e} "
            f"up|max|={up.abs().max().item():.4e} h|max|={h.abs().max().item():.4e} "
            f"y|max|={y.abs().max().item():.4e} finite={bool(torch.isfinite(y).all())}",
            flush=True,
        )

    for tensor in (*tt_w, *tt_x.values(), tt_idx):
        ttnn.deallocate(tensor)
finally:
    ttnn.close_device(device)
