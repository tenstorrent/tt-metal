import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers import audio_resample
from models.tt_dit.layers.audio_ops import Conv1dViaConv3d, ConvTranspose1dViaConv3d, _AlignedOutConv1d
from models.tt_dit.models.audio_vae.vocoder_ltx import AMPBlock1, Vocoder
from models.tt_dit.parallel.config import AudioTCParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.ltx.test_audio_components_ltx import (
    _MAIN_VOCODER_CFG,
    _build_torch_stage_b,
    _diffusers_vocoder_state_to_tt,
    _tt_vocoder_cfg,
    _vocoder_mel,
)

_PER_CONV = {}  # (Cin, Cout, K, stride, T_pad) -> list of ms


def _wrap_conv(mod):
    orig = mod.forward

    def timed(x, *a, **k):
        ttnn.synchronize_device(mod.mesh_device)
        Cin = int(x.shape[2])
        T_pad = int(x.shape[1])
        t = time.perf_counter()
        r = orig(x, *a, **k)
        ttnn.synchronize_device(mod.mesh_device)
        dt = (time.perf_counter() - t) * 1000
        key = (Cin, mod.unpadded_out_channels, mod.kernel_size[0], mod.stride[0], T_pad)
        _PER_CONV.setdefault(key, []).append(dt)
        return r

    mod.forward = timed


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vocoder_per_conv(mesh_device, device_params):
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1), channel_parallel=ParallelFactor(factor=2, mesh_axis=0)
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    def _walk(m):
        yield m
        for _, c in m.named_children():
            yield from _walk(c)

    for mod in _walk(tt_voc):
        if isinstance(mod, (Conv1dViaConv3d, _AlignedOutConv1d)):
            _wrap_conv(mod)

    mel = _vocoder_mel()
    _ = tt_voc(mel)
    ttnn.synchronize_device(mesh)  # warm
    _PER_CONV.clear()

    for _ in range(3):
        _ = tt_voc(mel)
    ttnn.synchronize_device(mesh)

    # Aggregate
    rows = []
    for k, v in _PER_CONV.items():
        Cin, Cout, K, stride, T_pad = k
        mean_ms = sum(v) / len(v)
        cnt_per_fwd = len(v) / 3
        rows.append((mean_ms * cnt_per_fwd, Cin, Cout, K, stride, T_pad, mean_ms, cnt_per_fwd))
    rows.sort(reverse=True)
    print("\n========== PER-SHAPE Conv1d/3d DEVICE TIME (avg/forward) ==========", flush=True)
    print(
        f"{'TotMs':>8s} {'Cin':>5s} {'Cout':>5s} {'K':>3s} {'s':>2s} {'T_pad':>6s} {'ms/call':>8s} {'n':>5s}",
        flush=True,
    )
    total = 0
    for tot_ms, Cin, Cout, K, stride, T_pad, mean_ms, cnt in rows[:30]:
        total += tot_ms
        print(f"{tot_ms:8.2f} {Cin:5d} {Cout:5d} {K:3d} {stride:2d} {T_pad:6d} {mean_ms:8.3f} {cnt:5.0f}", flush=True)
    print(f"\nShown total: {total:.1f} ms; full conv total: {sum(r[0] for r in rows):.1f} ms", flush=True)


_CAT = {}


def _wrap_cat(mod, label):
    orig = mod.forward

    def timed(*a, **k):
        ttnn.synchronize_device(mod.mesh_device)
        t = time.perf_counter()
        r = orig(*a, **k)
        ttnn.synchronize_device(mod.mesh_device)
        _CAT.setdefault(label, []).append((time.perf_counter() - t) * 1000)
        return r

    mod.forward = timed


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vocoder_categories(mesh_device, device_params):
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1), channel_parallel=ParallelFactor(factor=2, mesh_axis=0)
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    def _walk(m):
        yield m
        for _, c in m.named_children():
            yield from _walk(c)

    for mod in _walk(tt_voc):
        if not hasattr(mod, "mesh_device"):
            continue
        if isinstance(mod, (Conv1dViaConv3d, _AlignedOutConv1d)):
            _wrap_cat(mod, "Conv1d/3d (leaf)")
        elif isinstance(mod, ConvTranspose1dViaConv3d):
            _wrap_cat(mod, "ConvTranspose1d (incl inner conv)")
        elif isinstance(mod, audio_resample.UpSample1d):
            _wrap_cat(mod, "UpSample1d (anti-alias)")
        elif isinstance(mod, audio_resample.DownSample1d):
            _wrap_cat(mod, "DownSample1d (anti-alias)")
        elif isinstance(mod, audio_resample.LowPassFilter1d):
            _wrap_cat(mod, "LowPassFilter1d")
        elif isinstance(mod, audio_resample.Activation1d):
            _wrap_cat(mod, "Activation1d (anti-alias incl up+down)")
        elif isinstance(mod, AMPBlock1):
            _wrap_cat(mod, "AMPBlock1 (incl acts+convs)")
    mel = _vocoder_mel()
    _ = tt_voc(mel)
    ttnn.synchronize_device(mesh)
    _CAT.clear()
    t0 = time.perf_counter()
    for _ in range(3):
        _ = tt_voc(mel)
    ttnn.synchronize_device(mesh)
    total_wall = (time.perf_counter() - t0) * 1000 / 3
    print(f"\n========== CATEGORY DEVICE TIME (avg of 3 forwards) ==========", flush=True)
    print(f"{'Category':45s} {'n/fwd':>8s} {'ms/call':>10s} {'Total ms':>12s}", flush=True)
    for cat in sorted(_CAT, key=lambda k: sum(_CAT[k]), reverse=True):
        tot = sum(_CAT[cat]) / 3
        cnt = len(_CAT[cat]) / 3
        per = tot / cnt
        print(f"{cat:45s} {cnt:8.0f} {per:10.3f} {tot:12.2f}", flush=True)
    print(f"\nFull forward wall time: {total_wall:.2f} ms", flush=True)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vocoder_devicetime(mesh_device, device_params):
    # Run under: python -m tracy -p -r -m pytest <this>::test_prof_vocoder_devicetime
    # Flushes the on-device profiler buffer after each AMPBlock1 (~300 ops/window) to stay
    # under the 1000-zone tracy buffer limit; the CSV then has every op's DEVICE FW DURATION.
    # Sum of DEVICE FW DURATION over one forward = true device-active time; compare to the
    # host wall (printed) to get the host-dispatch-bound fraction (the trace-mode ceiling).
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    # Periodic profiler flush after each AMPBlock1 to bound the device zone buffer.
    def _walk(m):
        yield m
        for _, c in m.named_children():
            yield from _walk(c)

    for mod in _walk(tt_voc):
        if isinstance(mod, AMPBlock1):
            orig = mod.forward

            def timed(*a, _orig=orig, **k):
                r = _orig(*a, **k)
                ttnn.ReadDeviceProfiler(mesh)
                return r

            mod.forward = timed

    mel = _vocoder_mel()
    _ = tt_voc(mel)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)

    # Single profiled forward.
    t0 = time.perf_counter()
    _ = tt_voc(mel)
    ttnn.synchronize_device(mesh)
    host_wall = (time.perf_counter() - t0) * 1000
    ttnn.ReadDeviceProfiler(mesh)
    print(f"\nSINGLE_FORWARD_HOST_WALL_MS={host_wall:.2f}", flush=True)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 300000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vocoder_trace(mesh_device, device_params):
    # Capture the pure-device vocoder graph (_forward_device) and replay it, vs eager.
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    mel = _vocoder_mel()

    # Eager reference (full path) — populates lazy caches (snake shards, CCL buffers, tpad masks).
    host_ref = tt_voc(mel)

    # Eager timing of the device graph alone (host prep/post excluded).
    x_eager = tt_voc._host_to_device(mel)
    ttnn.synchronize_device(mesh)
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        y = tt_voc._forward_device(tt_voc._host_to_device(mel))
        ttnn.synchronize_device(mesh)
    eager_ms = (time.perf_counter() - t0) * 1000 / N

    # Persistent input + warmup compile on a separate buffer.
    x_in = tt_voc._host_to_device(mel)
    _ = tt_voc._forward_device(tt_voc._host_to_device(mel))
    ttnn.synchronize_device(mesh)

    # Capture.
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    y_tr = tt_voc._forward_device(x_in)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)

    # Replay timing (same input buffer).
    t0 = time.perf_counter()
    for _ in range(N):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
    trace_ms = (time.perf_counter() - t0) * 1000 / N

    # Correctness: traced output vs eager reference.
    host_tr = tt_voc._device_to_host(y_tr)
    ttnn.release_trace(mesh, tid)

    max_abs = (host_ref - host_tr).abs().max().item()

    print(
        f"\nTRACE eager_device={eager_ms:.2f}ms trace_replay={trace_ms:.2f}ms "
        f"speedup={eager_ms/trace_ms:.2f}x max|Δ|(trace vs eager)={max_abs:.3e}",
        flush=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 300000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_forward_traced_correctness(mesh_device, device_params):
    # Capture with mel_A, replay with a DIFFERENT mel_B, require == eager forward(mel_B).
    # This proves the persistent-input copy works (not just replaying captured data).
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    torch.manual_seed(1)
    mel_A = torch.randn(1, 2, 120, 64, dtype=torch.float32) * 0.5
    torch.manual_seed(2)
    mel_B = torch.randn(1, 2, 120, 64, dtype=torch.float32) * 0.5

    _ = tt_voc.forward_traced(mel_A)  # capture
    out_B_traced = tt_voc.forward_traced(mel_B)  # replay with new input
    out_B_eager = tt_voc(mel_B)  # eager reference
    tt_voc.release_trace()

    max_abs = (out_B_eager - out_B_traced).abs().max().item()
    logger.info(f"forward_traced replay(mel_B) vs eager(mel_B) max|Δ| = {max_abs:.3e}")
    print(f"\nTRACED_CORRECTNESS max|Δ|={max_abs:.3e}", flush=True)
    assert max_abs < 5e-3, f"traced replay diverged from eager: {max_abs:.3e}"


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 300000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_full_forward_wall(mesh_device, device_params):
    # Full vocoder forward (host prep + device + readback) wall: eager vs forward_traced.
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))
    mel = _vocoder_mel()
    N = 10
    # Eager full forward (warm once).
    _ = tt_voc(mel)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = tt_voc(mel)
    eager_ms = (time.perf_counter() - t0) * 1000 / N
    # Traced full forward (capture on first call).
    _ = tt_voc.forward_traced(mel)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = tt_voc.forward_traced(mel)
    traced_ms = (time.perf_counter() - t0) * 1000 / N
    tt_voc.release_trace()
    print(
        f"\nFULL_FORWARD eager={eager_ms:.2f}ms traced={traced_ms:.2f}ms speedup={eager_ms/traced_ms:.2f}x", flush=True
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 300000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_vocoder_with_bwe_traced(mesh_device, device_params):
    # Validate the VocoderWithBWE.use_trace wiring: traced main vocoder + eager BWE == fully eager.
    from models.tt_dit.tests.models.ltx.test_audio_components_ltx import _build_torch_stage_c, _build_tt_stage_c

    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_full = _build_torch_stage_c(seed=42)
    tt_full = _build_tt_stage_c(mesh, parallel_config=pc, ccl_manager=ccl)
    from models.tt_dit.tests.models.ltx.test_audio_components_ltx import _diffusers_vocoder_with_bwe_state_to_tt

    tt_full.load_torch_state_dict(_diffusers_vocoder_with_bwe_state_to_tt(torch_full.state_dict()))

    mel = _vocoder_mel()
    tt_full.use_trace = False
    out_eager = tt_full(mel)
    tt_full.use_trace = True
    _ = tt_full(mel)  # capture
    out_traced = tt_full(mel)  # replay
    tt_full.release_trace()
    max_abs = (out_eager - out_traced).abs().max().item()
    print(f"\nVOC_BWE_TRACED max|Δ|(traced vs eager)={max_abs:.3e}", flush=True)
    assert max_abs < 5e-3, f"VocoderWithBWE traced diverged: {max_abs:.3e}"


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_eager_wall_singleaxis(mesh_device, device_params):
    # Clean eager wall (bare forward, 1 sync, 10-iter) at single-axis T=4 — works on baseline
    # source too (uses only public forward + ParallelFactor). For the apples-to-apples table.
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = ParallelFactor(factor=4, mesh_axis=1)
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))
    mel = _vocoder_mel()
    _ = tt_voc(mel)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = tt_voc(mel)
    print(f"\nEAGER_SA_WALL={(time.perf_counter()-t0)*1000/10:.2f}ms", flush=True)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 300000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_forward_traced_multishape(mesh_device, device_params):
    # Two distinct input shapes both resident in the LRU trace cache; each replays bit-exact
    # vs eager. Then max_traces=1 evicts the LRU shape.
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = Vocoder(
        mesh_device=mesh,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=pc,
        ccl_manager=ccl,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    torch.manual_seed(1)
    mel_a = torch.randn(1, 2, 120, 64, dtype=torch.float32) * 0.5  # T-pad -> 128
    torch.manual_seed(2)
    mel_b = torch.randn(1, 2, 200, 64, dtype=torch.float32) * 0.5  # T-pad -> 256 (distinct shape)
    ref_a, ref_b = tt_voc(mel_a), tt_voc(mel_b)

    _ = tt_voc.forward_traced(mel_a)  # capture A
    _ = tt_voc.forward_traced(mel_b)  # capture B (A stays resident)
    assert len(tt_voc._traces) == 2, f"expected 2 resident traces, got {len(tt_voc._traces)}"
    da = (ref_a - tt_voc.forward_traced(mel_a)).abs().max().item()  # replay A
    db = (ref_b - tt_voc.forward_traced(mel_b)).abs().max().item()  # replay B
    print(f"\nMULTITRACE replay max|Δ| A={da:.3e} B={db:.3e} resident={len(tt_voc._traces)}", flush=True)
    assert da < 5e-3 and db < 5e-3

    tt_voc.release_trace()
    tt_voc._max_traces = 1
    _ = tt_voc.forward_traced(mel_a)
    _ = tt_voc.forward_traced(mel_b)  # evicts A
    assert len(tt_voc._traces) == 1, f"max_traces=1 should evict; got {len(tt_voc._traces)}"
    print(f"MULTITRACE max_traces=1 -> resident={len(tt_voc._traces)} (A evicted)", flush=True)
    tt_voc.release_trace()


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_bwe_per_conv(mesh_device, device_params):
    # Per-conv device time of the BWE generator at the real 6s length, with the T_out_block
    # each conv got and whether its (Cin,Cout,K) is tuned in _FP32_BLOCKINGS.
    from models.tt_dit.utils.conv3d import _FP32_BLOCKINGS

    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1), channel_parallel=ParallelFactor(factor=2, mesh_axis=0)
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    torch_full = _build_torch_stage_c(seed=42)
    tt_full = _build_tt_stage_c(mesh, parallel_config=pc, ccl_manager=ccl)
    tt_full.load_torch_state_dict(_diffusers_vocoder_with_bwe_state_to_tt(torch_full.state_dict()))

    rows = {}

    def wrap(mod):
        orig = mod.forward
        Cout = mod.unpadded_out_channels
        K = mod.kernel_size[0]
        tob = getattr(mod.conv_config, "T_out_block", None)

        def timed(x, *a, **k):
            ttnn.synchronize_device(mod.mesh_device)
            Cin, T_pad = int(x.shape[2]), int(x.shape[1])
            t = time.perf_counter()
            r = orig(x, *a, **k)
            ttnn.synchronize_device(mod.mesh_device)
            dt = (time.perf_counter() - t) * 1000
            rows.setdefault((Cin, Cout, K, T_pad, tob), []).append(dt)
            return r

        mod.forward = timed

    def walk(m):
        yield m
        for _, c in m.named_children():
            yield from walk(c)

    for mod in walk(tt_full.bwe_generator):
        if isinstance(mod, (Conv1dViaConv3d, _AlignedOutConv1d)):
            wrap(mod)

    mel = _vocoder_mel(int(os.environ.get("T_FRAMES", "601")))
    _ = tt_full(mel)
    ttnn.synchronize_device(mesh)
    rows.clear()
    for _ in range(3):
        _ = tt_full(mel)
    ttnn.synchronize_device(mesh)

    print("\n===== BWE generator per-conv (avg/fwd) =====", flush=True)
    print(f"{'ms/fwd':>8} {'Cin':>5} {'Cout':>5} {'K':>3} {'T_pad':>6} {'Tblk':>5} {'tuned?':>7}", flush=True)
    agg = []
    for (Cin, Cout, K, T_pad, tob), v in rows.items():
        ms = sum(v) / 3
        tuned = (aligned := None) or ((Cin, _round32(Cout), (K, 1, 1)) in _FP32_BLOCKINGS) or ((Cin, Cout, (K, 1, 1)) in _FP32_BLOCKINGS)
        agg.append((ms, Cin, Cout, K, T_pad, tob, tuned))
    for ms, Cin, Cout, K, T_pad, tob, tuned in sorted(agg, reverse=True):
        print(f"{ms:8.2f} {Cin:5} {Cout:5} {K:3} {T_pad:6} {str(tob):>5} {'yes' if tuned else 'NO':>7}", flush=True)
    print(f"\nBWE conv total: {sum(r[0] for r in agg):.1f} ms/fwd", flush=True)


def _round32(c):
    return ((c + 31) // 32) * 32
