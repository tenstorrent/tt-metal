# Cost of moving a DP8 x TP4 (row-group) prefill step's KV cache into the decode layout with ttnn.point_to_point.
# With TP4 on mesh rows, prompt p (lane c) runs on row r; chip (r, j) holds 2 of its 8 kv heads, while decode wants head k on
# chip (k, c). Staged per step, the move is: every chip (r, j != c) sends its 2 heads (all 80 layers) along the row to
# (r, c), then (r, c) sends head k down the column to (k, c). At 4K one head for 80 layers is 80 * 4096 * 128 * 2 B
# (bf8 K and V) ~ 84 MB. Measures single transfers and the whole 80-transfer pattern.
#   pytest models/demos/llama3_70b_galaxy/tests/test_kv_move_bench.py -s       env: KM_ROWS (10240 = one 4K head)
import os
import time

import pytest
from loguru import logger

import ttnn

W = 8192


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_kv_move_bench(mesh_device, device_params):
    mesh = mesh_device
    rows = int(os.environ.get("KM_ROWS") or 10240)  # [rows, 8192] bf8 = one head's K and V for 80 layers at 4K
    mb = rows * W * 2 / 1e6  # bf16: point_to_point rejects block-float tiles
    one = ttnn.zeros(
        (1, 1, rows, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    two = ttnn.zeros(
        (1, 1, 2 * rows, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out1 = ttnn.zeros(
        (1, 1, rows, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out2 = ttnn.zeros(
        (1, 1, 2 * rows, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    C = ttnn.MeshCoordinate

    def timed(name, fn, n=3):
        fn()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) / n * 1e3
        logger.info(f"KVMOVE {name}: {ms:.2f} ms")
        return ms

    def p2p(t, s, r, out, topo=ttnn.Topology.Linear):
        return ttnn.point_to_point(t, C(s), C(r), topology=topo, output_tensor=out)

    res = {}
    try:
        res["row1"] = timed(f"1 transfer {mb:.0f} MB, 1 hop along a row", lambda: p2p(one, (0, 0), (0, 1), out1))
        res["col1"] = timed(f"1 transfer {mb:.0f} MB, 1 hop down a column", lambda: p2p(one, (0, 0), (1, 0), out1))
        res["col4"] = timed(
            f"1 transfer {mb:.0f} MB, 4 hops down a column (Linear)", lambda: p2p(one, (0, 0), (4, 0), out1)
        )
        try:
            res["col4r"] = timed(
                f"1 transfer {mb:.0f} MB, 4 hops down a column (Ring)",
                lambda: p2p(one, (0, 0), (4, 0), out1, ttnn.Topology.Ring),
            )
        except Exception as e:
            logger.warning(f"KVMOVE ring column ERROR {str(e)[:200]}")
        res["row8"] = timed(
            f"8 transfers {mb:.0f} MB on 8 different rows at once",
            lambda: [p2p(one, (r, 0), (r, 1), out1) for r in range(8)],
        )

        def step_pattern():
            # prompt on row r belongs to lane c = r % 4; heads gathered at (r, c), then fanned out down column c
            for r in range(8):
                c = r % 4
                for j in range(4):
                    if j != c:
                        p2p(two, (r, j), (r, c), out2)
            for r in range(8):
                c = r % 4
                for k in range(8):
                    if k != r:
                        p2p(one, (r, c), (k, c), out1)

        res["step"] = timed(
            "whole step pattern (24 row transfers of 2 heads + 56 column transfers of 1 head)", step_pattern, n=2
        )
    except Exception as e:
        logger.warning(f"KVMOVE ERROR {str(e)[:400]}")
    logger.info(
        "KVMOVE SUMMARY "
        + " ".join(f"[{k}]={v:.2f}ms" for k, v in res.items())
        + (f" | 1-hop bandwidth {mb / res['row1']:.1f} GB/s" if "row1" in res else "")
    )
