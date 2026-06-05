# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Isolated device-perf harness for the DINO neck and neck + pre_transformer_tt.

Bypasses the Swin-L backbone (random activations, real weights) so the device
profiler only sees neck / pre_transformer ops. These are perf benchmarks, not
correctness tests — PCC is covered by tests/pcc/test_ttnn_dino_e2e.py.

Benchmarks:
  test_bench_neck           - neck only
  test_bench_neck_pretrans  - neck + pre_transformer_tt (the full hand-off)

Env toggles (all default off unless noted):
  TT_METAL_DEVICE_PROFILER=1  drain the device profiler after the timed region
  DINO_NECK_FLATTEN=1         neck emits flattened (B, H*W, C) ROW_MAJOR and
                              pre_transformer consumes it directly, vs the legacy
                              NCHW permute/reshape round-trip
  DINO_RM_CONCAT=0            disable the row-major sequence concat in
                              pre_transformer_tt (default ON in tt_dino.py)
  DINO_TRACE=1                build TtDINO in trace_mode (caches positional
                              encoding) — matches the production runtime path

Reproduce a capture and bucket it by OP CODE:

  TT_METAL_DEVICE_PROFILER=1 DINO_TRACE=1 DINO_NECK_FLATTEN=1 \\
    python -m tracy -r -p -v --no-runtime-analysis \\
    -m pytest models/experimental/dino_5scale_swin_l/tests/perf/_bench_neck.py \\
    -k test_bench_neck_pretrans -sv

  # tracy writes generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv;
  # run this file as a script to print a per-OP-CODE device-kernel breakdown
  # scoped to the signpost("start"/"stop") region:
  python models/experimental/dino_5scale_swin_l/tests/perf/_bench_neck.py \\
    generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv
"""

import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.dino_5scale_swin_l.common import NECK_IN_CHANNELS, NECK_OUT_CHANNELS
from loguru import logger


# Backbone feature shapes feeding the neck (NHWC), at 800x1333 input.
_NECK_INPUT_HWC = [
    (200, 334, 192),
    (100, 167, 384),
    (50, 84, 768),
    (25, 42, 1536),
]


def _get_ckpt():
    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    ckpt = ckpt_dir / "dino_5scale_swin_l.pth"
    if not ckpt.is_file():
        ckpt = ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    return str(ckpt)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_bench_neck(device, reset_seeds):
    ckpt_path = _get_ckpt()
    if not Path(ckpt_path).is_file():
        pytest.skip("Checkpoint not found")

    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import load_neck_weights, _resolve_state_dict
    from models.experimental.dino_5scale_swin_l.tt.tt_neck import TtDINONeck

    sd = _resolve_state_dict(ckpt_path)
    neck_params = load_neck_weights(sd, device)
    neck = TtDINONeck(
        device,
        neck_params,
        in_channels=tuple(NECK_IN_CHANNELS),
        out_channels=NECK_OUT_CHANNELS,
    )

    def make_features():
        feats = []
        for H, W, C in _NECK_INPUT_HWC:
            t = torch.rand(1, H, W, C)
            feats.append(
                ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        return feats

    flatten = os.environ.get("DINO_NECK_FLATTEN") == "1"
    logger.info(f"Benchmarking neck path: {'FLATTEN (B,HW,C)' if flatten else 'NCHW (default)'}")
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    try:
        from tracy import signpost
    except Exception:

        def signpost(_):
            pass

    def run_once():
        feats = make_features()
        out = neck(feats, flatten=flatten)
        return out[0] if flatten else out

    # Warm-up (JIT compile + program cache) so the profiled call is steady-state.
    _ = run_once()
    ttnn.synchronize_device(device)

    signpost("start")
    outs = run_once()
    ttnn.synchronize_device(device)
    signpost("stop")
    if profiling:
        ttnn.ReadDeviceProfiler(device)

    logger.info(f"neck produced {len(outs)} levels")
    for i, o in enumerate(outs):
        logger.info(f"  level {i}: {list(o.shape)}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_bench_neck_pretrans(device, reset_seeds):
    """Neck + pre_transformer_tt together, to capture the full round-trip win.

    NCHW path: neck(flatten=False) -> pre_transformer_tt(outs)
    Flatten path: neck(flatten=True) -> pre_transformer_tt(outs, shapes)
    Toggle with DINO_NECK_FLATTEN=1.
    """
    ckpt_path = _get_ckpt()
    if not Path(ckpt_path).is_file():
        pytest.skip("Checkpoint not found")

    from models.experimental.dino_5scale_swin_l.common import (
        NUM_QUERIES,
        NUM_CLASSES,
        NUM_LEVELS,
        ENCODER_EMBED_DIMS,
        ENCODER_NUM_HEADS,
        ENCODER_NUM_POINTS,
        ENCODER_NUM_LAYERS,
        DECODER_NUM_LAYERS,
    )
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import (
        load_neck_weights,
        load_encoder_weights,
        load_decoder_weights,
        _resolve_state_dict,
    )
    from models.experimental.dino_5scale_swin_l.tt.tt_dino import TtDINO

    sd = _resolve_state_dict(ckpt_path)
    neck_params = load_neck_weights(sd, device)
    encoder_params = load_encoder_weights(sd, device)
    decoder_params = load_decoder_weights(sd, device)
    del sd

    model = TtDINO(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        device=device,
        backbone_params=None,
        neck_params=neck_params,
        attn_masks=None,
        num_queries=NUM_QUERIES,
        num_classes=NUM_CLASSES,
        num_levels=NUM_LEVELS,
        embed_dims=ENCODER_EMBED_DIMS,
        num_heads=ENCODER_NUM_HEADS,
        num_points=ENCODER_NUM_POINTS,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        decoder_num_layers=DECODER_NUM_LAYERS,
        pe_temperature=20,
        in_channels=(192, 384, 768, 1536),
        trace_mode=os.environ.get("DINO_TRACE") == "1",
    )

    def make_features():
        feats = []
        for H, W, C in _NECK_INPUT_HWC:
            t = torch.rand(1, H, W, C)
            feats.append(
                ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        return feats

    flatten = os.environ.get("DINO_NECK_FLATTEN") == "1"
    logger.info(f"Benchmarking neck+pre_transformer path: {'FLATTEN' if flatten else 'NCHW (default)'}")
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    try:
        from tracy import signpost
    except Exception:

        def signpost(_):
            pass

    def run_once():
        feats = make_features()
        if flatten:
            outs, shapes = model.neck(feats, flatten=True)
            pre = model.pre_transformer_tt(outs, shapes)
        else:
            outs = model.neck(feats, flatten=False)
            pre = model.pre_transformer_tt(outs)
        return pre

    _ = run_once()
    ttnn.synchronize_device(device)

    signpost("start")
    pre = run_once()
    ttnn.synchronize_device(device)
    signpost("stop")
    if profiling:
        ttnn.ReadDeviceProfiler(device)

    logger.info(f"feat_flatten: {list(pre['feat_flatten'].shape)}")


def _bucket_ops_csv(path):
    """Print a per-OP-CODE device-kernel breakdown from a tracy ops CSV.

    Scopes to the signpost("start"/"stop") rows (OP TYPE == signpost) so warm-up
    ops outside the timed region are excluded.
    """
    import csv as _csv
    import collections

    with open(path) as f:
        rows = list(_csv.DictReader(f))

    start_i = end_i = None
    for i, r in enumerate(rows):
        oc = (r.get("OP CODE") or "").strip().lower()
        if oc == "start" and start_i is None:
            start_i = i
        if oc == "stop":
            end_i = i
    if start_i is not None and end_i is not None:
        region = rows[start_i + 1 : end_i]
        scope = f"signpost region rows {start_i + 1}..{end_i}"
    else:
        region = rows
        scope = "ALL rows (no signpost found)"

    dur_col = "DEVICE KERNEL DURATION [ns]"
    agg = collections.defaultdict(lambda: [0, 0.0])  # op -> [count, total_ns]
    for r in region:
        oc = (r.get("OP CODE") or "").strip()
        if not oc or oc.lower() in ("start", "stop"):
            continue
        try:
            d = float(r.get(dur_col) or 0)
        except ValueError:
            d = 0.0
        agg[oc][0] += 1
        agg[oc][1] += d

    total = sum(v[1] for v in agg.values())
    print(f"Scope: {scope}")
    print(f"Ops in region (excl signposts): {sum(v[0] for v in agg.values())}")
    print(f"Total device kernel time: {total / 1e6:.3f} ms\n")
    print(f"{'OP CODE':36}{'count':>7}{'total_ms':>12}{'%':>8}{'avg_us':>10}")
    for oc, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        pct = 100.0 * t / total if total else 0.0
        avg_us = (t / c) / 1e3 if c else 0.0
        print(f"{oc:36}{c:>7}{t / 1e6:>12.3f}{pct:>8.1f}{avg_us:>10.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python _bench_neck.py <ops_perf_results.csv>")
        sys.exit(1)
    _bucket_ops_csv(sys.argv[1])
