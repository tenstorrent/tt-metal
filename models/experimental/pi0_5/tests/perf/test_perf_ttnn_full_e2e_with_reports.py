# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end with ttnn-visualizer report capture.

Uses `ttnn.graph.full_graph_capture(report_path)` — the canonical context
manager for producing a complete report (op DAG + buffer trace + Python
I/O sidecar) scoped to a single inference. It handles:
  * switching to slow-dispatch (`enable_fast_runtime_mode=False`)
  * `begin_graph_capture` / `end_graph_capture`
  * Python I/O recording + stack traces + detailed buffer tracing
  * writing the final JSON on exit + restoring all flags

Why this approach beats the earlier `manage_config` toggle and the global
`TTNN_CONFIG_OVERRIDES=enable_graph_report=true`:
  * `manage_config` flips the runtime flag but does NOT install the
    per-op capture buffer that `begin_graph_capture` does. With manage
    alone, ops emit "fast" without recording — so the final report
    contains only whatever ran before the manage_config block (i.e.,
    the 34 setup ops we saw, not the ~3000 ops of one sample_actions).
  * Setting `enable_graph_report=true` globally breaks `ttnn.from_torch`
    during weight upload (the graph tracer wraps the torch.Tensor input
    and the C++ pybind parser rejects the wrapper —
    `parse_py_tensor: ndarray_import failed`).
  * `full_graph_capture` is scoped, only wraps the inference, and uses
    the proper begin/end pair so the JSON actually contains the ops.

How to run:
    TT_METAL_HOME=/home/tt-admin/sdawle/pi0/tt-metal \
    TT_VISIBLE_DEVICES=0 \
    PI0_UPSTREAM_MASKS=1 \
    QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 \
    PYTHONPATH=/home/tt-admin/sdawle/pi0/tt-metal:/storage/sdawle/openpi/src \
    /home/tt-admin/sdawle/pi0/tt-metal/python_env/bin/python -m pytest \
      models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_with_reports.py \
      -x -s --no-header

NOTE: Do NOT set TTNN_CONFIG_OVERRIDES with enable_graph_report=true —
full_graph_capture handles the toggling internally and avoids the
init-time from_torch breakage. Times printed below are NOT comparable
to perf runs (slow-dispatch + tracing dominates).

Output goes to `generated/ttnn/reports/pi0_5_e2e_<timestamp>/report.json`
(and sidecar `.python_io.json` next to it). Import into ttnn-visualizer
via `ttnn-visualizer --report-dir <that dir>`.
"""

import os
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"

NUM_WARMUP = 0
NUM_ITERS = 1
LANG_SEQ_LEN = 256
SEED = 0
TRACE_REGION_SIZE = 80_000_000
# Production pi0.5 LIBERO passes 3 images to SigLIP. See [[pi05-siglip-bs3-production]].
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "2"))

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(cfg, device, num_cameras: int = NUM_CAMERAS):
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    images_ttnn = [
        ttnn.from_torch(
            im, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for im in images
    ]
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return images_ttnn, img_masks, lang_tokens_ttnn, lang_masks_ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_full_e2e_with_reports(device):
    """One sample_actions call captured via ttnn.graph.full_graph_capture."""
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig.from_checkpoint(CHECKPOINT_DIR)
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print("✅ Model loaded")

    images_ttnn, img_masks, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(cfg, device)
    print(f"   num_cameras={len(images_ttnn)} (SigLIP runs bs={len(images_ttnn)} via concat)")

    # Pick the report destination. Put it under generated/ttnn/reports/ to
    # match the convention the visualizer's auto-discovery expects.
    report_root = Path("generated/ttnn/reports") / f"pi0_5_e2e_{time.strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "report.json"
    print(f"\n🎬 Capturing inference graph to {report_path}")

    times_ms: List[float] = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        with torch.no_grad(), ttnn.graph.full_graph_capture(str(report_path)):
            _ = model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=None,
            )
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   call {i + 1:2d}: {elapsed_ms:7.2f} ms  (with full_graph_capture overhead)")

    avg = statistics.mean(times_ms)
    print("\n" + "=" * 72)
    print("  PI0.5 TTNN E2E + VISUALIZER REPORT")
    print("=" * 72)
    print(f"   Per-call avg:        {avg:7.2f} ms   (NOT comparable to perf runs)")
    print(f"   Report directory:    {report_root.resolve()}")
    print(f"   Main report file:    {report_path.name}")
    print(f"   Python I/O sidecar:  {report_path.stem}.python_io.json")
    print(f"   Inspect with:        ttnn-visualizer --report-dir {report_root.resolve()}")
    print("=" * 72)
    assert avg > 0
    assert report_path.exists(), f"Expected report at {report_path}"
    assert report_path.stat().st_size > 0, "Report file is empty — capture failed"
