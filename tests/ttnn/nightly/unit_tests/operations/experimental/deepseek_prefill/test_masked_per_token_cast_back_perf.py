# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time perf sweep for masked_per_token_cast_back across dispatch-buffer fill levels.

Uses the real DeepSeek dispatch-buffer dimensions (capacity M = 5*1024*8 = 40960 rows, emb_dim
H = 7168) and varies how much of the buffer is valid (the packed prefix), from ~empty (best case)
to fully valid (worst case). A tracy `signpost` is emitted before each scenario so the generated
ops_perf CSV can be sliced per scenario. A baseline unmasked per_token_cast_back over the full
buffer is measured too, to show what the masked op saves.

Run under the tracy profiler to get DEVICE KERNEL DURATION:

    source python_env/bin/activate
    python -m tracy -r -v -o /localdev/nostojic/masked_perf \
        -m pytest tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/\
test_masked_per_token_cast_back_perf.py -s

The CSV lands in <output>/ops_perf_results_*.csv; parse rows whose OP CODE contains
"PerTokenCastBack", keyed by the preceding signpost header.
"""

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost

from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.use_module_device

BLOCK_W = 128
TILE = 32
E4M3_MAX = 448.0

# Real dispatch-buffer geometry.
CAPACITY = 5 * 1024 * 8  # 40960 rows (ISL 5*1024 * dispatch capacity 8)
H = 7168  # emb_dim -> H / 128 = 56 scale blocks per row

ITERS = 10  # measured invocations per scenario (program cache warm after the first)

# (label, valid_rows) — valid_rows is tile-aligned already.
# Small-fill points exercise the flattened compute-block split (few tile-rows, many cores); the
# ~10K point is the headline regime where the whole grid is already busy (should be unchanged).
SCENARIOS = [
    ("r32_1tr", 32),  # 1 tile-row -> 56 compute-blocks -> 56 cores (was 1 core under row split)
    ("r96_3tr", 96),  # 3 tile-rows -> 168 compute-blocks -> all 110 cores
    ("r1024_1K", 1024),  # 32 tile-rows -> 1792 compute-blocks (was 32 cores under row split)
    ("r1280", 1280),  # 40 tile-rows -> 2240 compute-blocks; all 110 cores busy
    ("r4096_10pct", 4096),  # 128 tile-rows; row split already fills the grid here
    ("r10240_10K", 10240),  # 320 tile-rows, the headline number (unchanged by this optimization)
]


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


def _make_u32(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_masked_decompress_perf_sweep(device):
    torch.manual_seed(0)

    # Build the FP8 input + scale ONCE (shared across scenarios; only the metadata/counts change,
    # and the op only touches the valid prefix). This keeps DRAM usage bounded.
    x = (torch.randn(CAPACITY, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    e4m3_tt = ttnn.from_torch(
        x.float(),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale = (torch.rand(CAPACITY, H // BLOCK_W) * 2.0).to(torch.float32)
    scale_tt = ttnn.from_torch(
        scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    logger.info(f"perf sweep: capacity={CAPACITY} H={H} num_cores={num_cores}")

    # experts_per_chip = 1, identity table: single expert region [0, valid_rows).
    table_tt = _make_u32(device, [0])
    region_tt = _make_u32(device, [0])

    def run_masked(valid_rows):
        counts_tt = _make_u32(device, [valid_rows])
        out = ttnn.experimental.deepseek_prefill.masked_per_token_cast_back(
            e4m3_tt, scale_tt, region_tt, counts_tt, table_tt, experts_per_chip=1, output_dtype=ttnn.bfloat16
        )
        ttnn.synchronize_device(device)
        return out

    # Warm the program cache (all scenarios share one program hash: identical shapes/attrs).
    run_masked(TILE)

    # ---- Masked op: sweep fill levels ----
    for label, valid_rows in SCENARIOS:
        signpost(header=f"MASKED:{label}:valid_rows={valid_rows}")
        for _ in range(ITERS):
            run_masked(valid_rows)

    # ---- Baseline: unmasked per_token_cast_back always processes the FULL buffer ----
    signpost(header="BASELINE:unmasked_full")
    for _ in range(ITERS):
        out = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.bfloat16)
        ttnn.synchronize_device(device)

    logger.info("perf sweep done")


@pytest.mark.parametrize("label, valid_rows", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_masked_decompress_single(device, label, valid_rows):
    """Single-variant version so one fill level can be isolated with -k (e.g. -k r10240_10K).

    Builds only the valid-prefix-sized FP8 input + scale (M = ceil_tile(valid_rows)), so it is fast
    and light. Emits one signpost and runs ITERS measured invocations.
    """
    torch.manual_seed(0)
    m = ((valid_rows + TILE - 1) // TILE) * TILE

    x = (torch.randn(m, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    e4m3_tt = ttnn.from_torch(
        x.float(),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale = (torch.rand(m, H // BLOCK_W) * 2.0).to(torch.float32)
    scale_tt = ttnn.from_torch(
        scale, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    table_tt = _make_u32(device, [0])
    region_tt = _make_u32(device, [0])
    counts_tt = _make_u32(device, [valid_rows])

    grid = device.compute_with_storage_grid_size()
    logger.info(f"single {label}: valid_rows={valid_rows} M={m} H={H} num_cores={grid.x * grid.y}")

    def run():
        out = ttnn.experimental.deepseek_prefill.masked_per_token_cast_back(
            e4m3_tt, scale_tt, region_tt, counts_tt, table_tt, experts_per_chip=1, output_dtype=ttnn.bfloat16
        )
        ttnn.synchronize_device(device)
        return out

    run()  # warm program cache
    signpost(header=f"MASKED:{label}:valid_rows={valid_rows}")
    for _ in range(ITERS):
        run()
    logger.info(f"single {label} done")
