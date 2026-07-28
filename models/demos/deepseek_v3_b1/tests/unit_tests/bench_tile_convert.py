# SPDX-License-Identifier: Apache-2.0
"""How expensive is it to bridge tt_transformers' [32,32] decode activations to the
tiny-tile [1,32] format that dram_streaming_matmul needs?

dram_streaming_matmul reaches 453-500 GB/s on bfp4 weights at m=1, versus 308 GB/s
for the production dram_sharded kernel at m=32 -- worth ~6.2 ms/token on gemma2-9B
1xP150. But tt_transformers decode carries batch-1 activations padded to a full
32-row tile, so every matmul would need a format bridge on the way in and out.
There are ~420 such bridges per token (5 matmuls x 42 layers x in+out), so if a
bridge costs more than ~7 us the entire gain is eaten.

A [1,32] tile has [1,16] faces, i.e. 32 contiguous elements -- byte-identical to a
row-major row. So the inbound bridge is really "extract row 0", and the outbound one
is "place a row into row 0 of a padded tile". This measures the candidates.

Shapes are gemma2-9B: K=3584 (FF1/FF3 and QKV input), N=14336 (FF1/FF3 output).
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

K = 3584
N = 14336
REPS = 200


def bench(fn, device, reps=REPS):
    """Median-of-3 wall time per call, with the device drained each round.

    Includes host dispatch, so it is an upper bound: production decode runs under
    trace, where dispatch is hidden. Anything cheap here is cheap for real.
    """
    fn()
    ttnn.synchronize_device(device)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(device)
        el = (time.perf_counter() - t0) / reps
        best = el if best is None else min(best, el)
    return best * 1e6  # us


def report(name, us, note=""):
    logger.info(f"CONV {name:44s} {us:8.2f} us  {note}")


@pytest.mark.parametrize("width", [K, N], ids=[f"K{K}", f"N{N}"])
def test_inbound(device, width):
    """[1,1,32,width] TILE bf16  ->  [1,1,1,width] contiguous row."""
    logger.info(f"===== INBOUND  width={width} =====")
    torch_x = torch.randn(1, 1, 32, width)

    dram = ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    l1 = ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    for tag, src in (("dram", dram), ("l1", l1)):
        # A) full untilize then slice row 0
        try:

            def a():
                rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT)
                out = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 1, width])
                ttnn.deallocate(rm)
                ttnn.deallocate(out)

            report(f"{tag} A untilize+slice", bench(a, device))
        except Exception as e:
            report(f"{tag} A untilize+slice", float("nan"), f"FAILED {type(e).__name__}: {str(e)[:90]}")

        # B) slice row 0 while still tiled, then untilize the 1-row result
        try:

            def b():
                s = ttnn.slice(src, [0, 0, 0, 0], [1, 1, 1, width])
                out = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(s)
                ttnn.deallocate(out)

            report(f"{tag} B slice+untilize", bench(b, device))
        except Exception as e:
            report(f"{tag} B slice+untilize", float("nan"), f"FAILED {type(e).__name__}: {str(e)[:90]}")

        # C) untilize_with_unpadding -- one op instead of two
        try:

            def c():
                out = ttnn.untilize_with_unpadding(src, [0, 0, 0, width - 1])
                ttnn.deallocate(out)

            report(f"{tag} C untilize_with_unpadding", bench(c, device))
        except Exception as e:
            report(f"{tag} C untilize_with_unpadding", float("nan"), f"FAILED {type(e).__name__}: {str(e)[:90]}")

    ttnn.deallocate(dram)
    ttnn.deallocate(l1)


@pytest.mark.parametrize("width", [N], ids=[f"N{N}"])
def test_outbound(device, width):
    """[1,1,1,width] row  ->  [1,1,32,width] TILE bf16 (row 0 populated)."""
    logger.info(f"===== OUTBOUND width={width} =====")
    row = ttnn.from_torch(
        torch.randn(1, 1, 1, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # D) tilize with padding straight to a 32-row tile
    try:

        def d():
            out = ttnn.tilize_with_val_padding(row, [1, 1, 32, width], 0.0)
            ttnn.deallocate(out)

        report("D tilize_with_val_padding", bench(d, device))
    except Exception as e:
        report("D tilize_with_val_padding", float("nan"), f"FAILED {type(e).__name__}: {str(e)[:90]}")

    # E) plain tilize of the single row (stays 1 row tall -- only valid if the
    #    consumer tolerates a 1-row tensor, which stock ops do not)
    try:

        def e_():
            out = ttnn.tilize(row)
            ttnn.deallocate(out)

        report("E tilize (1 row)", bench(e_, device))
    except Exception as e:
        report("E tilize (1 row)", float("nan"), f"FAILED {type(e).__name__}: {str(e)[:90]}")

    ttnn.deallocate(row)


def test_replicate(device):
    """The streaming kernel wants in0 replicated: one copy of [1,K] per compute core,
    HEIGHT_SHARDED. Measure that broadcast separately from the format change."""
    logger.info("===== REPLICATE in0 across compute cores =====")
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    n_cores = len(cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])

    src = ttnn.from_torch(
        torch.randn(1, 1, n_cores, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    try:

        def f():
            out = ttnn.to_memory_config(src, mc)
            ttnn.deallocate(out)

        report(f"F reshard to {n_cores}-core height-sharded", bench(f, device))
    except Exception as e:
        report("F reshard", float("nan"), f"FAILED {type(e).__name__}: {str(e)[:90]}")
    ttnn.deallocate(src)
