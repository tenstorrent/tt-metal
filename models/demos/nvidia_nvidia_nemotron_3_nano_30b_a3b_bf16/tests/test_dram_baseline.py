#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: measure DRAM usage baseline after device open, then after embedding upload."""

import os
import sys

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn


def _dram_stats(mesh, label=""):
    """Dump DRAM stats by provoking an intentional OOM on a tiny 1-byte allocation and catching the stats."""
    print(f"\n=== DRAM stats: {label} ===")
    try:
        # DumpDeviceMemoryState writes to files - use it to get stats
        for chip_id in range(4):
            dev = mesh.get_device(chip_id)
            ttnn.DumpDeviceMemoryState(dev, f"/tmp/dram_stats_chip{chip_id}_")
            print(f"  Chip {chip_id}: stats written to /tmp/dram_stats_chip{chip_id}_buffers.csv")
    except Exception as e:
        print(f"  DumpDeviceMemoryState error: {e}")


def main():
    from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _rep, close_device_tp4, open_device_tp4

    print("Opening 4-chip mesh...")
    mesh = open_device_tp4()
    ttnn.synchronize_device(mesh)
    print("Device opened.")

    _dram_stats(mesh, "after_device_open")

    # Upload just the embedding weight
    print("\nUploading embedding weight [131072, 2688] bf16...")
    wc = WeightCache()
    emb = wc["backbone.embeddings.weight"].bfloat16()
    print(f"  Embedding shape: {emb.shape}, size: {emb.numel() * 2 / 1e6:.1f} MB")
    emb_tt = _rep(emb, mesh)
    ttnn.synchronize_device(mesh)
    print("  Uploaded.")

    _dram_stats(mesh, "after_embedding_upload")

    print("\nDone. Closing mesh.")
    close_device_tp4(mesh)


if __name__ == "__main__":
    main()
