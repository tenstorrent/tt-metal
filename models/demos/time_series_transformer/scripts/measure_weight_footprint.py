# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Measures the L1 footprint of TST model weights to gate Change 3
(L1 memory config for weights).

Weights are loaded via tt.tst_model.load_weights(), which produces
ttnn.bfloat16 tensors (2 bytes/element, confirmed in _to_ttnn). This
script walks the resulting weights dict, computes per-tensor and
per-group byte sizes from tensor.shape (no assumptions about padded
widths -- reads actual shapes off the loaded ttnn tensors), and
compares cumulative footprint against the device's real L1 capacity,
queried live rather than hardcoded:

    ttnn.get_max_worker_l1_unreserved_size() -> bytes per core
    device.compute_with_storage_grid_size()  -> core grid

ttnn.L1_MEMORY_CONFIG is TensorMemoryLayout.INTERLEAVED + BufferType.L1,
meaning an L1-resident tensor striped this way draws from the pooled
per-core budget across the full grid (per_core_bytes * num_cores), not
a single core's budget alone. This script reports both the pooled
ceiling and the per-core figure so the decision is visible either way.

Usage:
    ARCH_NAME=wormhole_b0 python3 scripts/measure_weight_footprint.py
"""
import sys
from pathlib import Path

from transformers import TimeSeriesTransformerForPrediction

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt.tst_model import load_weights  # noqa: E402

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
BYTES_PER_ELEMENT_BFLOAT16 = 2


def _tensor_bytes(t: ttnn.Tensor) -> int:
    numel = 1
    for d in t.shape:
        numel *= d
    return numel * BYTES_PER_ELEMENT_BFLOAT16


def _walk_group(name, obj, rows, skipped):
    """Recursively walk a weights dict/subdict, recording (path, bytes) leaves.

    Non-tensor leaves (e.g. weights["dist_type"] = "student_t", a plain
    metadata string, not a weight) are skipped and reported separately
    rather than silently miscounted or crashing on .shape access.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_group(f"{name}.{k}", v, rows, skipped)
    elif hasattr(obj, "shape"):
        rows.append((name, _tensor_bytes(obj)))
    else:
        skipped.append((name, type(obj).__name__, repr(obj)[:60]))


def main():
    print("Loading HuggingFace reference model (for config + state_dict only)...")
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model.eval()

    device = ttnn.open_device(device_id=0)
    try:
        l1_per_core = ttnn.get_max_worker_l1_unreserved_size()
        grid = device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        l1_pooled = l1_per_core * num_cores

        print(f"\nDevice L1 capacity (queried live, not hardcoded):")
        print(f"  per-core unreserved L1: {l1_per_core:,} bytes ({l1_per_core / 1024:.1f} KB)")
        print(f"  grid: {grid.x} x {grid.y} = {num_cores} cores")
        print(
            f"  pooled L1 (INTERLEAVED, per_core * num_cores): {l1_pooled:,} bytes ({l1_pooled / 1024 / 1024:.2f} MB)"
        )

        print("\nLoading TST weights via tt.tst_model.load_weights()...")
        weights = load_weights(hf_model, device)

        rows = []
        skipped = []
        _walk_group("weights", weights, rows, skipped)

        if skipped:
            print(f"\nSkipped {len(skipped)} non-tensor metadata leaf/leaves (not weights):")
            for name, typ, preview in skipped:
                print(f"  {name} ({typ}) = {preview}")

        total_bytes = sum(b for _, b in rows)

        print(f"\n{'Tensor path':<55}{'Bytes':>12}{'KB':>10}")
        print("-" * 77)
        for path, b in sorted(rows, key=lambda r: -r[1]):
            print(f"{path:<55}{b:>12,}{b / 1024:>10.2f}")

        print("-" * 77)
        print(f"{'TOTAL':<55}{total_bytes:>12,}{total_bytes / 1024:>10.2f}")
        print(f"{'TOTAL (MB)':<55}{'':<12}{total_bytes / 1024 / 1024:>10.3f}")

        # Per-layer-group rollups, matching the Stage 2 issue checklist
        # (value/feature embeddings, encoder self-attn, decoder self+cross-attn,
        # FFN, distribution head).
        group_totals = {}
        for path, b in rows:
            # weights.encoder.layers.0.qkv_weight -> group key "encoder.layers.N.*"
            parts = path.split(".")
            if len(parts) >= 3 and parts[2] == "layers":
                group_key = f"{parts[1]}.layers"
            else:
                group_key = ".".join(parts[1:3]) if len(parts) >= 3 else path
            group_totals[group_key] = group_totals.get(group_key, 0) + b

        print(f"\n{'Group':<30}{'Bytes':>12}{'MB':>10}{'% of pooled L1':>18}")
        print("-" * 70)
        for group, b in sorted(group_totals.items(), key=lambda r: -r[1]):
            pct = 100.0 * b / l1_pooled
            print(f"{group:<30}{b:>12,}{b / 1024 / 1024:>10.3f}{pct:>17.4f}%")

        print("\n" + "=" * 70)
        fits_pooled = total_bytes <= l1_pooled
        fits_single_core = total_bytes <= l1_per_core
        pct_pooled = 100.0 * total_bytes / l1_pooled
        print(f"Total weight footprint: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.3f} MB)")
        print(
            f"Fits in pooled L1 (INTERLEAVED across {num_cores} cores)? {fits_pooled} ({pct_pooled:.4f}% of {l1_pooled / 1024 / 1024:.2f} MB)"
        )
        print(f"Fits in a single core's L1 (worst case, no interleaving)? {fits_single_core}")
        print("=" * 70)

        if fits_pooled:
            print(
                "\nRESULT: Full weight set fits comfortably in pooled L1. "
                "Footprint is NOT the blocker for Change 3 -- proceed with "
                "wiring ttnn.L1_MEMORY_CONFIG onto weight tensors and measure "
                "actual latency impact (L1 also holds activations/KV cache/trace "
                "buffers during decode, so real contention -- not raw footprint -- "
                "is the thing to verify next via test_single_sequence_latency)."
            )
        else:
            print(
                "\nRESULT: Weight footprint EXCEEDS pooled L1 capacity. "
                "Change 3 cannot move all weights to L1 as-is -- select a subset "
                "(e.g. hot-path decoder self-attn only) or keep DRAM_MEMORY_CONFIG."
            )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
