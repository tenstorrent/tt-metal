# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Isolation probe: does _overlay_kv_device's output tensor actually equal
the real write-then-read-cache path, BEFORE any attention/softmax/FFN math
runs? probe_fused_trace.py's full correctness check failed with
max_abs_diff ~0.80-0.90 at EVERY step (including step 0), using a REAL
non-trivial embedding from _prepare_dec_step_cpu_1tok. All prior probes
that validated _overlay_kv_device used an all-ZERO dec_input, which may
have masked a broadcast/tile-padding bug that only shows up with real
(non-zero) magnitudes.

This probe builds a real step-0 embedding exactly like production does,
then directly compares:
  Path A (ground truth): _extract_and_write_kv into a zeroed cache, then
      read the cache back via ttnn.to_layout(..., TILE_LAYOUT) -- the
      exact read _attend_from_cache does.
  Path B (under test): _overlay_kv_device on a SEPARATE zeroed cache,
      returning k_full/v_full directly without any cache write.
No attention, softmax, cross-attn, or FFN math is involved -- if these
two diverge here, the bug is in the overlay construction (ttnn.mul
selector broadcast, or the ROW_MAJOR/TILE conversion) itself, not in
anything downstream.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.demos.time_series_transformer.tt.tst_attention import allocate_kv_cache  # noqa: E402
