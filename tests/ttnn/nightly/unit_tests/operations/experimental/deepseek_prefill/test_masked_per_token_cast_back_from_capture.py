# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay masked_per_token_cast_back over REAL per-layer dispatch metadata captured from the prefill
transformer, and measure device time under tracy.

The capture side (TtMoe, gated by TT_DS_CAPTURE_DISPATCH_META_DIR) dumps one file per MoE layer:

    layer_<idx>_dispatch_meta.pt = {
        "layer_idx", "experts_per_chip", "num_routed_experts", "seq_len_per_chip",
        "emb_dim",                    # H (dispatch buffer width; scale width = H // 128)
        "dispatch_buffer_capacity",   # M (dispatch buffer row count)
        "mesh_shape", "num_devices",  # e.g. (8, 4), 32
        "expert_token_counts",        # [num_devices, num_routed_experts]
        "expert_region_offsets",      # [num_devices, num_routed_experts]
        "global_expert_idx_table",    # [num_devices, experts_per_chip]
    }

Only the per-expert valid-token metadata drives the op (it is compute-bound and data-independent), so
the FP8 buffer + scale are faked here; their content does not affect timing. For each (layer, device)
we rebuild that chip's exact valid prefix and run the op, emitting a tracy signpost so the ops_perf CSV
can be sliced per (layer, device).

Run:

    export TT_DS_CAPTURE_DISPATCH_META_DIR=/path/to/capture   # dir with layer_*_dispatch_meta.pt
    source python_env/bin/activate
    python -m tracy -r -v -m -p pytest \
        tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/\
test_masked_per_token_cast_back_from_capture.py -s

Optional env knobs:
    MASKED_CAPTURE_ITERS       measured invocations per (layer, device)  [default 10]
    MASKED_CAPTURE_MAX_DEVICES cap devices per layer (e.g. 4) for a quick run  [default all]
    MASKED_CAPTURE_DEDUP       "1" to run one representative per distinct valid-row count  [default 0]
"""

import glob
import os

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

CAPTURE_DIR = os.getenv("TT_DS_CAPTURE_DISPATCH_META_DIR", "")
ITERS = int(os.getenv("MASKED_CAPTURE_ITERS", "10"))
MAX_DEVICES = int(os.getenv("MASKED_CAPTURE_MAX_DEVICES", "0"))  # 0 = all
DEDUP = os.getenv("MASKED_CAPTURE_DEDUP", "0").lower() in ("1", "true", "yes")


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


def _ceil_tile(n):
    return ((int(n) + TILE - 1) // TILE) * TILE


def _make_u32(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _valid_prefix(counts_row, offsets_row, table_row):
    """This chip's valid dispatch-buffer prefix = max over its local experts of
    region_offset[g] + ceil_tile(counts[g]) (mirrors the reader kernel)."""
    prefix = 0
    for g in table_row.tolist():
        end = int(offsets_row[g]) + _ceil_tile(int(counts_row[g]))
        if end > prefix:
            prefix = end
    return prefix


def _layer_files():
    if not CAPTURE_DIR:
        return []
    return sorted(
        glob.glob(os.path.join(CAPTURE_DIR, "layer_*_dispatch_meta.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )


def test_masked_decompress_from_capture(device):
    """Sweep every captured MoE layer x chip, run the masked op on that chip's real valid prefix,
    and measure device time under tracy (one signpost per (layer, device))."""
    files = _layer_files()
    if not files:
        pytest.skip(
            "No capture files found. Set TT_DS_CAPTURE_DISPATCH_META_DIR to a dir containing "
            "layer_*_dispatch_meta.pt (produced by running the prefill transformer with the same env "
            "var set)."
        )

    logger.info(f"masked-from-capture: {len(files)} layer file(s) in {CAPTURE_DIR}")

    for path in files:
        meta = torch.load(path, map_location="cpu")
        layer_idx = int(meta["layer_idx"])
        H = int(meta["emb_dim"])
        counts = meta["expert_token_counts"]  # [num_devices, num_routed_experts]
        offsets = meta["expert_region_offsets"]  # [num_devices, num_routed_experts]
        table = meta["global_expert_idx_table"]  # [num_devices, experts_per_chip]
        num_devices = counts.shape[0]

        dev_range = range(num_devices if MAX_DEVICES <= 0 else min(MAX_DEVICES, num_devices))
        seen_prefixes = set()

        for dev in dev_range:
            counts_row = counts[dev]
            offsets_row = offsets[dev]
            table_row = table[dev]
            experts_per_chip = table_row.numel()

            prefix = _valid_prefix(counts_row, offsets_row, table_row)
            if prefix <= 0:
                logger.info(f"layer {layer_idx} dev {dev}: 0 valid rows, skipping")
                continue
            if DEDUP and prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)

            # Fake FP8 buffer + scale sized to this chip's valid prefix (M = prefix). The op only
            # touches [0, prefix); rows beyond it are never read, so this is faithful for timing while
            # keeping host/device memory bounded. Content is irrelevant (data-independent op).
            m = _ceil_tile(prefix)
            x = torch.zeros(m, H, dtype=torch.float32)
            e4m3_tt = ttnn.from_torch(
                x,
                dtype=ttnn.fp8_e4m3,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            scale = torch.ones(m, H // BLOCK_W, dtype=torch.float32)
            scale_tt = ttnn.from_torch(
                scale,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # region_offsets / counts are indexed by GLOBAL expert id, so pass the full-length rows;
            # the table (this chip's local->global map) selects which entries the op actually reads.
            region_tt = _make_u32(device, offsets_row.to(torch.int32).tolist())
            counts_tt = _make_u32(device, counts_row.to(torch.int32).tolist())
            table_tt = _make_u32(device, table_row.to(torch.int32).tolist())

            def run():
                out = ttnn.experimental.deepseek_prefill.masked_per_token_cast_back(
                    e4m3_tt,
                    scale_tt,
                    region_tt,
                    counts_tt,
                    table_tt,
                    experts_per_chip=experts_per_chip,
                    output_dtype=ttnn.bfloat16,
                )
                ttnn.synchronize_device(device)
                return out

            logger.info(
                f"layer {layer_idx} dev {dev}: valid_rows={prefix} (M={m}, {prefix // TILE} tile-rows), "
                f"experts_per_chip={experts_per_chip}, H={H}"
            )
            run()  # warm program cache for this shape
            signpost(header=f"MASKED:layer{layer_idx}:dev{dev}:valid_rows={prefix}")
            for _ in range(ITERS):
                run()

            ttnn.deallocate(e4m3_tt)
            ttnn.deallocate(scale_tt)
            ttnn.deallocate(region_tt)
            ttnn.deallocate(counts_tt)
            ttnn.deallocate(table_tt)

    logger.info("masked-from-capture sweep done")
