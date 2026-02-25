# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark: measure factory, build, and launch times separately."""

import time
import pytest
import torch
import ttnn

from models.experimental.ops.descriptors.fusion import Sequential
from models.experimental.ops.descriptors.normalization import rms_norm
from models.experimental.ops.descriptors import composite


def _tt(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _cores(x1, y1, x2, y2):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestBuildTiming:
    def test_build_timing_breakdown(self, device):
        """Measure factory, build(), and launch() separately after JIT warmup."""
        torch.manual_seed(42)
        hidden = 128
        core_range = _cores(0, 0, 3, 1)
        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        # ---- JIT warmup (first build+launch populates kernel cache) ----
        r1 = rms_norm.rms_norm(
            _tt(torch_input, device), core_range_set=core_range, weight=_tt(torch_w, device), epsilon=1e-5
        )
        r2 = rms_norm.rms_norm(
            r1.output_tensors[0], core_range_set=core_range, weight=_tt(torch_w, device), epsilon=1e-5
        )
        fused = Sequential(r1, r2).build(device)
        composite.launch([fused])

        # ---- Timed: fresh tensors each iteration ----
        print(f"\n{'='*60}")
        print("Fresh tensors each iteration (factory + build + launch):")
        for trial in range(5):
            t0 = time.perf_counter()
            r1 = rms_norm.rms_norm(
                _tt(torch_input, device), core_range_set=core_range, weight=_tt(torch_w, device), epsilon=1e-5
            )
            r2 = rms_norm.rms_norm(
                r1.output_tensors[0], core_range_set=core_range, weight=_tt(torch_w, device), epsilon=1e-5
            )
            t1 = time.perf_counter()
            fused = Sequential(r1, r2).build(device)
            t2 = time.perf_counter()
            outputs = composite.launch([fused])
            t3 = time.perf_counter()
            print(
                f"  Trial {trial}: factory={1000*(t1-t0):.2f}ms  build={1000*(t2-t1):.2f}ms  launch={1000*(t3-t2):.2f}ms  total={1000*(t3-t0):.2f}ms"
            )

        # ---- Reuse same FusedOp (no rebuild) ----
        print()
        print("Reuse same FusedOp (no rebuild, no factory):")
        for trial in range(5):
            t0 = time.perf_counter()
            outputs = composite.launch([fused])
            t1 = time.perf_counter()
            print(f"  Trial {trial}: launch={1000*(t1-t0):.2f}ms")
        print(f"{'='*60}")
