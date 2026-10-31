# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip GDN decode: Qwen36GatedDeltaNet vs TPGatedDeltaNet at TP=1.

With one device, qwen36/tt/layer.py builds Qwen36GatedDeltaNet (the 9B single-device module).
TPGatedDeltaNet, with the fused [qkv|z|a|b] in-projection, the L1-resident decode chain and the
fused recurrence, is only built when num_devices > 1. This test loads one real GDN layer, builds
both modules on one chip, checks that they agree over several decode steps, and times one traced
decode step of each (TP=1 with the composite recurrence and with the fused recurrence).

Run on one chip:
    MESH_DEVICE=P150 HF_MODEL=<HF checkpoint dir> \
      pytest models/demos/blackhole/bonsai2/tp1/test_gdn_decode_tp1.py -s
Set GDN_TP1_RESULTS=<path.json> to also write the numbers to a file.
"""
import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, load_gdn_layer, model_path
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

STEPS = 8
REPLAYS = 200
REPEATS = 5
MIN_PCC = 0.99
PROFILE_STEPS = 4

DEVICE_PARAMS = [{"l1_small_size": max(24576, GDN_CONV1D_L1_SMALL_SIZE), "trace_region_size": 64 * 1024 * 1024}]


def _single_device_keys(sd, key_dim):
    """HF linear_attn.* keys -> the remapped keys Qwen36GatedDeltaNet loads (as weight_mapping.py does)."""
    out = {k.removeprefix("linear_attn."): v for k, v in sd.items()}
    out["qkv_proj.weight"] = out.pop("in_proj_qkv.weight")
    conv = out.pop("conv1d.weight")
    out["q_conv.weight"] = conv[:key_dim]
    out["k_conv.weight"] = conv[key_dim : 2 * key_dim]
    out["v_conv.weight"] = conv[2 * key_dim :]
    return out


def _to_dev(mesh_device, x):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def _traced_ms(mesh_device, step, x_tt):
    """Median per-replay wall time of one captured decode step, in ms. Each step gets its own copy of
    x_tt because the GDN modules may free their input (in the model it is a fresh norm output)."""

    def run():
        return step(ttnn.clone(x_tt, memory_config=ttnn.L1_MEMORY_CONFIG))

    ttnn.deallocate(run())  # compile
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        out = run()
    except Exception:
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3 / REPLAYS)
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    return statistics.median(times)


def _build(mesh_device, max_seq_len=256):
    """Both GDN modules for the first linear-attention layer, on one chip."""
    os.environ.setdefault("HF_MODEL", model_path())
    torch.manual_seed(0)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
    assert args.num_devices == 1, "run on a single chip (MESH_DEVICE=P150)"
    tp1 = args.tp1_args(mesh_device)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    n_gdn = sum(t == "linear_attention" for t in args.attention_type_list)
    sd = load_gdn_layer(args.CKPT_DIR, li)
    old = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), _single_device_keys(sd, args.linear_q_dim))
    tp = TPGatedDeltaNet(mesh_device, tp1, load_gdn_weights_tp(mesh_device, sd, tp1), None)
    tp._fuse_agmm = False
    return args, li, n_gdn, old, tp


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_decode_tp1(mesh_device):
    args, li, n_gdn, old, tp = _build(mesh_device)

    def run_old(x_tt):
        return old.forward(x_tt, mode="recurrent")

    def run_tp(x_tt):
        return tp.forward_decode(x_tt)

    def host(out):
        return ttnn.to_torch(out).float().reshape(-1, args.dim)[:1]

    pccs = []
    for _ in range(STEPS):
        x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
        o_old, o_tp = run_old(_to_dev(mesh_device, x)), run_tp(_to_dev(mesh_device, x))
        pccs.append(compute_pcc(host(o_old), host(o_tp)))
        ttnn.deallocate(o_old)
        ttnn.deallocate(o_tp)
    logger.info(f"layer {li}: single-device vs TP=1 decode PCC per step {[round(p, 5) for p in pccs]}")

    x_tt = _to_dev(mesh_device, torch.randn(1, 1, args.dim, dtype=torch.bfloat16))
    clone_ms = _traced_ms(mesh_device, lambda x: x, x_tt)
    logger.info(f"input clone alone: {clone_ms:.4f} ms (subtracted below)")
    ms = {}
    for name, step in (("single_device", run_old), ("tp1_composite", run_tp), ("tp1_fused", run_tp)):
        tp.use_fused_recurrent_decode = name == "tp1_fused"
        ms[name] = _traced_ms(mesh_device, step, x_tt) - clone_ms
        logger.info(f"{name:>14}: {ms[name]:.4f} ms/layer, x{n_gdn} GDN layers = {ms[name] * n_gdn:.2f} ms/token")
    results = {
        "layer": li,
        "gdn_layers": n_gdn,
        "pcc": pccs,
        "ms_per_layer": ms,
        "clone_ms": clone_ms,
        "hf_model": os.environ["HF_MODEL"],
    }
    if os.environ.get("GDN_TP1_RESULTS"):
        with open(os.environ["GDN_TP1_RESULTS"], "w") as f:
            json.dump(results, f, indent=2)
    assert min(pccs) > MIN_PCC, f"single-device and TP=1 decode disagree: {pccs}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_decode_tp1_profile(mesh_device):
    """PROFILE_STEPS eager decode steps of each path between tracy signposts, for a per-op device profile:
        TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -v --no-web-server -m pytest <this file>::test_gdn_decode_tp1_profile
    then summarize the ops_perf_results CSV with profile_summary.py."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        pytest.skip("needs the device profiler (TT_METAL_DEVICE_PROFILER=1 under python -m tracy)")
    from tracy import signpost

    args, _, _, old, tp = _build(mesh_device)
    x_tt = _to_dev(mesh_device, torch.randn(1, 1, args.dim, dtype=torch.bfloat16))
    steps = (
        ("single_device", lambda x: old.forward(x, mode="recurrent")),
        ("tp1_composite", tp.forward_decode),
        ("tp1_fused", tp.forward_decode),
    )
    # Warm every path up before the first signpost, so each region holds only its own PROFILE_STEPS steps.
    for name, step in steps:
        tp.use_fused_recurrent_decode = name == "tp1_fused"
        for _ in range(2):
            ttnn.deallocate(step(ttnn.clone(x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)))
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    for name, step in steps:
        tp.use_fused_recurrent_decode = name == "tp1_fused"
        signpost(name)
        for _ in range(PROFILE_STEPS):
            ttnn.deallocate(step(ttnn.clone(x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)))
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
    signpost("end")
