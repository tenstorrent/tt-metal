# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sequence-level benchmark for the DROID-SLAM NN pipeline.

Exercises the full NN stack the paper defines on a real image sequence
from the TUM RGB-D ``rgbd_dataset_freiburg3_cabinet`` sample:

    fnet + cnet → all-pairs correlation pyramid → ``num_iters``
    rounds of (corr lookup, motion encode, ConvGRU, delta/weight/eta/upmask).

Two independent measurements are reported:

1. **Accuracy**: a lockstep run of the pure-torch reference and the
   on-device TtDroidNet share a coord trajectory. At each iteration
   both consume the *same* (net, inp, corr, motion) inputs and we
   compare per-output PCC. This isolates NN precision from compound
   coord-drift bf16 would otherwise introduce across 12 feedback
   iterations.

2. **Throughput**: a stand-alone tt-only run that threads its own
   state through the iteration (as a production deployment would),
   measured in keyframe-windows per second.

SLAM pieces (SE3 pose state, Bundle Adjustment, loop closure) are
outside the scope — they require CUDA-only ``lietorch`` and
``droid_backends`` which cannot be built on this Tenstorrent box
(see ``.omc/autopilot/spec.md``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from models.experimental.droid_slam.reference.corr_ref import CorrBlock, coords_grid
from models.experimental.droid_slam.reference.droid_net_ref import DroidNet
from models.experimental.droid_slam.tt.droid_net_tt import TtDroidNet


DEFAULT_WEIGHTS = Path("/home/ttuser/experiments/droid_slam/DROID-SLAM/droid.pth")
TUM_SEQ = Path(
    "/home/ttuser/experiments/droid_slam/DROID-SLAM/data/rgbd_dataset_freiburg3_cabinet/rgb"
)

BATCH = 1
NUM_KEYFRAMES = 4
HEIGHT = 240
WIDTH = 320
NUM_ITERS = int(os.environ.get("DROID_ITERS", "12"))
WARMUP_ITERS = 2
TIMED_ITERS = 3
KEYFRAME_STRIDE = 30  # TUM is 30 Hz; step by a second per keyframe


def _materialize(t):
    return t._materialize() if hasattr(t, "_materialize") else t


def _load_reference_weights(model: DroidNet, weights_path: Path) -> None:
    if not weights_path.exists():
        pytest.skip(f"droid.pth not found at {weights_path}")
    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    remapped = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        remapped[k] = v
    missing, _ = model.load_state_dict(remapped, strict=False)
    assert not missing, f"Missing weights: {missing[:8]}..."


def _load_tum_keyframes(num_keyframes: int, stride: int = KEYFRAME_STRIDE) -> torch.Tensor:
    if not TUM_SEQ.exists():
        pytest.skip(f"TUM sample not found at {TUM_SEQ}")
    files = sorted(p for p in TUM_SEQ.iterdir() if p.suffix.lower() == ".png")
    if len(files) < num_keyframes * stride:
        pytest.skip(f"TUM sample has {len(files)} frames, need {num_keyframes * stride}")
    from PIL import Image

    frames = []
    for k in range(num_keyframes):
        im = Image.open(str(files[k * stride])).convert("RGB").resize((WIDTH, HEIGHT))
        arr = np.asarray(im, dtype=np.float32)
        arr = arr[:, :, ::-1].copy()  # RGB -> BGR for DROID's reversed normalize
        frames.append(arr)
    stacked = np.stack(frames, axis=0)
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0).contiguous()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


def _reset_tt_update_state(tt_model: TtDroidNet) -> None:
    """Bypass the extract_features net/inp cache so the next
    ``tt_model.update`` call uploads from its torch arguments.
    """
    import ttnn as _ttnn

    if getattr(tt_model, "_update_trace_id", None) is not None:
        _ttnn.release_trace(tt_model.device, tt_model._update_trace_id)
        tt_model._update_trace_id = None
    tt_model._cached_net_tt = None
    tt_model._cached_inp_tt = None
    tt_model._cached_shape = None
    tt_model._update_cache_hit_seen = False
    tt_model._update_trace_outputs = None
    tt_model._update_trace_key = None


def _run_lockstep(
    ref: DroidNet,
    tt_model: TtDroidNet,
    images: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    num_iters: int,
) -> dict:
    """Run both paths synchronized on a shared coord trajectory.

    Each iteration: ref produces (delta, weight, eta, upmask, new_net);
    tt is fed the same (net, inp, corr, motion) and produces its own
    outputs for PCC comparison. Ref's outputs advance the state —
    tt's don't — so corr inputs stay aligned across all iterations.
    """
    b, nk, _, h, w = images.shape
    ht8, wd8 = h // 8, w // 8
    device = images.device

    ref_f, ref_n_all, ref_i_all = ref.extract_features(images)
    # Warm tt's feature caches so the throughput run below has them ready.
    tt_f_lazy, _, _ = tt_model.extract_features(images)
    tt_f = _materialize(tt_f_lazy)

    # From this point we want tt.update to take its `net`/`inp` from
    # the torch args (not the extract cache).
    _reset_tt_update_state(tt_model)

    # Share the correlation volume between both sides (built from ref
    # fmaps). This isolates per-iteration NN precision from the
    # ~0.994 fnet-precision drift.
    corr_fn = CorrBlock(ref_f[:, ii], ref_f[:, jj])
    coords0 = coords_grid(ht8, wd8, device=device, dtype=images.dtype)
    coords1 = coords0[None, None].expand(b, ii.shape[0], ht8, wd8, 2).contiguous()
    target = coords1.clone()

    r_net = ref_n_all[:, ii].contiguous()
    r_inp = ref_i_all[:, ii].contiguous()

    running = {k: 0.0 for k in ("net", "delta", "weight", "eta", "upmask")}
    final_ref = None
    final_tt = None
    for _ in range(num_iters):
        corr = corr_fn(coords1)
        resd = target - coords1
        flow = coords1 - coords0[None, None]
        motion = torch.cat([flow, resd], dim=-1).permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        r_net_out, r_delta, r_weight, r_eta, r_upmask = ref.update(
            r_net, r_inp, corr, motion, ii
        )
        t_net_l, t_delta_l, t_weight_l, t_eta_l, t_upmask_l = tt_model.update(
            r_net, r_inp, corr, motion, ii
        )
        t_net = _materialize(t_net_l)
        t_delta = _materialize(t_delta_l)
        t_weight = _materialize(t_weight_l)
        t_eta = _materialize(t_eta_l)
        t_upmask = _materialize(t_upmask_l)

        running["net"] += _pcc(r_net_out, t_net)
        running["delta"] += _pcc(r_delta, t_delta)
        running["weight"] += _pcc(r_weight, t_weight)
        running["eta"] += _pcc(r_eta, t_eta)
        running["upmask"] += _pcc(r_upmask, t_upmask)

        r_net = r_net_out
        target = coords1 + r_delta
        coords1 = target

        final_ref = (r_net_out, r_delta, r_weight, r_eta, r_upmask)
        final_tt = (t_net, t_delta, t_weight, t_eta, t_upmask)

    avg = {k: v / num_iters for k, v in running.items()}
    return {
        "fmaps_pcc": _pcc(ref_f, tt_f),
        "avg_pccs": avg,
        "final_pccs": {
            "net": _pcc(final_ref[0], final_tt[0]),
            "delta": _pcc(final_ref[1], final_tt[1]),
            "weight": _pcc(final_ref[2], final_tt[2]),
            "eta": _pcc(final_ref[3], final_tt[3]),
            "upmask": _pcc(final_ref[4], final_tt[4]),
        },
    }


def _run_tt_only(
    tt_model: TtDroidNet,
    images: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    initial_net: torch.Tensor,
    initial_inp: torch.Tensor,
    num_iters: int,
) -> None:
    """Throughput-oriented tt-only forward (no ref calls, no PCC).

    Feeds the materialized cnet state through the UpdateModule the way
    a production DROID-SLAM deployment would — ``net`` updated each
    step, ``inp`` held fixed.
    """
    b, nk, _, h, w = images.shape
    ht8, wd8 = h // 8, w // 8
    device = images.device

    fmaps_lazy, _, _ = tt_model.extract_features(images)
    fmaps = _materialize(fmaps_lazy)
    _reset_tt_update_state(tt_model)

    corr_fn = CorrBlock(fmaps[:, ii], fmaps[:, jj])
    coords0 = coords_grid(ht8, wd8, device=device, dtype=images.dtype)
    coords1 = coords0[None, None].expand(b, ii.shape[0], ht8, wd8, 2).contiguous()
    target = coords1.clone()

    net = initial_net[:, ii].contiguous()
    inp = initial_inp[:, ii].contiguous()

    for _ in range(num_iters):
        corr = corr_fn(coords1)
        resd = target - coords1
        flow = coords1 - coords0[None, None]
        motion = torch.cat([flow, resd], dim=-1).permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        net_l, delta_l, _, _, _ = tt_model.update(net, inp, corr, motion, ii)
        net = _materialize(net_l)
        delta = _materialize(delta_l)
        target = coords1 + delta
        coords1 = target


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 6 * 1024 * 1024}],
    indirect=True,
)
def test_droid_slam_sequence(device):
    weights_path = Path(os.environ.get("DROID_WEIGHTS", str(DEFAULT_WEIGHTS)))

    ref = DroidNet()
    _load_reference_weights(ref, weights_path)
    ref.eval()
    tt_model = TtDroidNet(device=device, reference=ref)

    images = _load_tum_keyframes(NUM_KEYFRAMES)
    ii = torch.arange(0, NUM_KEYFRAMES - 1, dtype=torch.long)
    jj = ii + 1

    # --- Accuracy (lockstep) ---
    acc = _run_lockstep(ref, tt_model, images, ii, jj, NUM_ITERS)
    min_avg = min(acc["avg_pccs"].values())
    accuracy = 100.0 * min_avg

    # --- Throughput (tt-only, realistic iterative forward) ---
    # Feed tt the materialized cnet output as the starting state.
    with torch.no_grad():
        _, ref_n, ref_i = ref.extract_features(images)

    for _ in range(WARMUP_ITERS):
        _run_tt_only(tt_model, images, ii, jj, ref_n, ref_i, NUM_ITERS)

    windows_total = BATCH * TIMED_ITERS
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        _run_tt_only(tt_model, images, ii, jj, ref_n, ref_i, NUM_ITERS)
    elapsed = time.perf_counter() - t0
    inference_speed = windows_total / elapsed

    peak_dram_mb = 0.0
    try:
        import ttnn

        peak_dram_mb = float(ttnn.get_memory_config_peak_usage(device)) / (1024 * 1024)
    except Exception:
        peak_dram_mb = 0.0

    print(f"inference_speed {inference_speed:.4f} windows_per_sec")
    print(f"accuracy {accuracy:.4f} percent")
    print(f"peak_dram {peak_dram_mb:.2f} MB")
    print(f"fmaps_pcc {acc['fmaps_pcc']:.4f}")
    print("avg PCC per iter:", {k: f"{v:.4f}" for k, v in acc["avg_pccs"].items()})
    print("final-iter PCC:", {k: f"{v:.4f}" for k, v in acc["final_pccs"].items()})

    assert accuracy >= 99.0, f"accuracy {accuracy:.2f} below 99% threshold"
