# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prefill's stage hang watcher, started by arm_hang_detector: python hang_watcher.py <probe.json>

ttnn.stream_hang_detection's watcher, with prefill's warm-up and verdict label. Prefill's run loop takes a chunk in
only after its previous push finished and sends the activation before the indices, which
is what the default rules rely on. Before the first real chunk, the trace capture may send a warm-up chunk on from
every rank with a D2D output, and every later rank takes it in through its D2D input like a real chunk; consumed()
evens out the difference.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


def _load_base_watcher() -> ModuleType:
    """The base watcher, loaded by file instead of importing ttnn.stream_hang_detection.watcher that would run ttnn/__init__.py."""
    path = Path(importlib.util.find_spec("ttnn").submodule_search_locations[0]) / "stream_hang_detection" / "watcher.py"
    spec = importlib.util.spec_from_file_location("_stream_hang_watcher", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass fails without this
    spec.loader.exec_module(module)
    return module


watcher = _load_base_watcher()


@dataclass
class PrefillStreamHangWatcher(watcher.StreamHangWatcher):
    stage_idx: int
    layers: str  # "20-39"
    warmup_out: int  # chunks sent before the first real one
    warmup_in: int  # chunks taken in before the first real one

    def label(self) -> str:
        return f"PREFILL HANG: stage {self.stage_idx} (layers {self.layers})"

    def limit_s(self, dev00: watcher.DeviceCounters) -> float:
        # Until the first real chunk is out, the trace capture can hold the stage still for seconds.
        warmed = self.produced(dev00) >= 1 + self.warmup_out
        return self.settings.timeout_s if warmed else self.settings.first_chunk_timeout_s

    def consumed(self, device: watcher.DeviceCounters) -> int:
        """Warm-up sent but never taken in counts as consumed too, once the first output is out."""
        warmup = self.warmup_out - self.warmup_in if self.produced(device) >= 1 else 0
        return super().consumed(device) + warmup

    def counts(self, device: watcher.DeviceCounters) -> str:
        counts = super().counts(device)
        if self.warmup_out or self.warmup_in:
            counts += f", warmup out={self.warmup_out} in={self.warmup_in}"
        return counts


if __name__ == "__main__":
    sys.exit(PrefillStreamHangWatcher.main(sys.argv[1]))
