import tests.ttnn.unit_tests.operations.tilize._bench_tilize as B
import torch, ttnn

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def _crs(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def H(shape, n):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _L1, ttnn.ShardSpec(_crs(n), (shape[-2] // n, shape[-1]), _ROW)
    )


CASES = [
    # (shape, cores, dir) — the R1-recorded crossover rows, read transfer = wt_chunk*64 B
    ([1, 1, 512, 64], 4, "dram_to_shard"),  # shard width 64 -> wt_chunk 2 -> 128 B reads
    ([1, 1, 2048, 256], 8, "dram_to_shard"),  # shard width 256 -> wt_chunk 8 -> 512 B reads
    ([1, 1, 512, 64], 4, "shard_to_dram"),
    ([1, 1, 1024, 128], 8, "dram_to_shard"),  # width 128 -> wt_chunk 4 -> 256 B reads
]
for shape, n, direction in CASES:
    cfg = H(shape, n)
    inc, outc = (None, cfg) if direction == "dram_to_shard" else (cfg, None)
    row = {}
    for zc in (1, 0):
        row[zc] = B._measure(
            device,
            shape,
            ttnn.bfloat16,
            in_mem_config=inc,
            out_mem_config=outc,
            levers=dict(zero_copy=zc),
            label=f"{shape}/{n}/{direction}/zc={zc}",
        )
    print(f"RESULT {shape} x{n} {direction}: on={row[1]} off={row[0]} ratio={row[0]/row[1]:.2f}x")
ttnn.close_device(device)
