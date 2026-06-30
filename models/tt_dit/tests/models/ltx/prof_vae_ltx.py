# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time profiling for the LTX-2 video VAE decoder at production length.

The standalone parity test (test_vae_ltx.py) builds a torch reference and runs
it on host; at 145 frames / 1080p that forward needs tens of GB of host RAM and
is impractical. These harnesses build the torch decoder only to source random
weights, load them into the TT decoder, and run the TT forward alone — so the
shape matches production (decode_latents calls the decoder once with all 19
latent frames) without paying for the host reference.

Defaults: 145 frames @ 1088x1920 on a 2x4 mesh (h_axis=0, w_axis=1), the
distilled-pipeline production config. Override via NUM_FRAMES / HEIGHT / WIDTH.
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.models.vae.vae_ltx import LTXCausalConv3d, LTXDepthToSpaceUpsample, LTXResnetBlock3D, LTXVideoDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.ltx.test_vae_ltx import (
    _LTX_PROD_DECODER_BLOCKS,
    _diffusers_decoder_state_to_tt,
    _require_diffusers_ltx_vae,
    _TorchLTXVideoDecoder,
)

_NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "145"))
_HEIGHT = int(os.environ.get("HEIGHT", "1088"))
_WIDTH = int(os.environ.get("WIDTH", "1920"))
# LTX_USE_FUSED=0 forces the all-standalone baseline; 1 (default) lets the per-shape
# win-table route the conv-heavy s3/s4 layers through the fused neighbor_pad_conv3d op.
_USE_FUSED = os.environ.get("LTX_USE_FUSED", "1") != "0"


def _walk(m):
    yield m
    for _, c in m.named_children():
        yield from _walk(c)


def _build_tt_decoder(mesh):
    """Production decoder with random weights; torch module built for its
    state_dict only (no host forward)."""
    vae_mods = _require_diffusers_ltx_vae()
    if not vae_mods["ltx2"]:
        pytest.skip("LTX-2 decoder profiling requires diffusers autoencoder_kl_ltx2")

    torch.manual_seed(42)
    torch_decoder = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=vae_mods,
    )
    torch_decoder.eval()
    state = torch_decoder.state_dict()
    # Identity per-channel stats so denorm is a no-op (random weights give garbage stats).
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_decoder.load_state_dict(state)

    # Match the BH (2,4) production config (create_pipeline device_configs): num_links=2.
    # NP_LINKS overrides it to probe the W-transport bound (more links = more parallel bandwidth;
    # only helps if the transport is bandwidth-bound, not the documented per-stick-latency bound).
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear, num_links=int(os.environ.get("NP_LINKS", "2")))
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh.shape)[0], mesh_axis=0),
        width_parallel=ParallelFactor(factor=tuple(mesh.shape)[1], mesh_axis=1),
    )
    tt_decoder = LTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        num_frames=_NUM_FRAMES,
        height=_HEIGHT,
        width=_WIDTH,
        mesh_device=mesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_fused=_USE_FUSED,
    )
    tt_decoder.load_torch_state_dict(_diffusers_decoder_state_to_tt(torch_decoder.state_dict()))
    return tt_decoder


def _latent():
    latent_frames = (_NUM_FRAMES - 1) // 8 + 1
    return torch.randn(1, 128, latent_frames, _HEIGHT // 32, _WIDTH // 32, dtype=torch.float32)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vae_ltx_per_conv(mesh_device, device_params):
    # Per-conv device time (sync-wrapped wall), with the C_in_block each conv3d got.
    # Identifies the heaviest conv shapes and whether their blocking is the generic fallback.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)

    rows = {}

    def wrap(mod):
        orig = mod.forward
        key0 = (mod.unpadded_in_channels, mod.unpadded_out_channels, mod.kernel_size, mod.stride)
        cib = getattr(mod.conv_config, "C_in_block", None)

        def timed(x, *a, **k):
            ttnn.synchronize_device(mod.mesh_device)
            t = time.perf_counter()
            r = orig(x, *a, **k)
            ttnn.synchronize_device(mod.mesh_device)
            dt = (time.perf_counter() - t) * 1000
            rows.setdefault(key0 + (int(x.shape[1]), cib), []).append(dt)
            return r

        mod.forward = timed

    for mod in _walk(tt_decoder):
        if isinstance(mod, LTXCausalConv3d):
            wrap(mod)

    latent = _latent()
    _ = tt_decoder(latent)  # warm program cache
    rows.clear()
    _ = tt_decoder(latent)

    print(f"\n===== LTX VAE per-conv device time ({_NUM_FRAMES}f@{_HEIGHT}x{_WIDTH}) =====", flush=True)
    print(f"{'ms/fwd':>8} {'Cin':>5} {'Cout':>5} {'kernel':>10} {'stride':>10} {'Tin':>4} {'Cblk':>5}", flush=True)
    agg = []
    for (cin, cout, k, s, tin, cib), v in rows.items():
        agg.append((sum(v) / len(v), cin, cout, k, s, tin, cib))
    total = 0.0
    for ms, cin, cout, k, s, tin, cib in sorted(agg, reverse=True):
        total += ms
        print(f"{ms:8.2f} {cin:5} {cout:5} {str(k):>10} {str(s):>10} {tin:4} {str(cib):>5}", flush=True)
    print(f"\nConv total: {total:.1f} ms/fwd", flush=True)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vae_ltx_devicetime(mesh_device, device_params):
    # Run under: python -m tracy -p -r -m pytest <this>::test_prof_vae_ltx_devicetime
    # Flushes the on-device profiler after each resnet/upsample block to stay under the
    # 1000-zone tracy buffer; the CSV then holds every op's DEVICE FW DURATION. Sum over one
    # forward = true device-active time. Host wall is printed for the dispatch-bound fraction.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)

    for mod in _walk(tt_decoder):
        if isinstance(mod, (LTXResnetBlock3D, LTXDepthToSpaceUpsample)):
            orig = mod.forward

            def timed(*a, _orig=orig, **k):
                r = _orig(*a, **k)
                ttnn.ReadDeviceProfiler(mesh)
                return r

            mod.forward = timed

    latent = _latent()
    _ = tt_decoder(latent)  # warm program cache
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    # tracy signpost
    from tracy import signpost

    signpost("start")
    t0 = time.perf_counter()
    _ = tt_decoder(latent)
    ttnn.synchronize_device(mesh)
    host_wall = (time.perf_counter() - t0) * 1000
    ttnn.ReadDeviceProfiler(mesh)
    signpost("stop")
    print(f"\nSINGLE_FORWARD_HOST_WALL_MS={host_wall:.2f}", flush=True)


_TRACE_ITERS = int(os.environ.get("TRACE_ITERS", "10"))


def _prepare_decode_input(tt_decoder, latent):
    """Replicate LTXVideoDecoder.forward's host upload → device-sharded sample_tt + logical dims, so the
    device-only decode (decode_device) can be captured as a trace with the host I/O kept outside."""
    from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_width
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    pc = tt_decoder.parallel_config
    sample = latent.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    sample, logical_h = conv_pad_height(sample, pc.height_parallel.factor)
    sample, logical_w = conv_pad_width(sample, pc.width_parallel.factor)
    sample_tt = typed_tensor_2dshard(
        sample,
        tt_decoder.mesh_device,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    return sample_tt, logical_h, logical_w


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vae_ltx_trace(mesh_device, device_params):
    """TRUE traced-decode WALL: capture the device-only decode as one ttnn trace and replay it, so the
    wall/iter is the real e2e device time with no host-dispatch gap. This is the metric the per-shape
    routing should be derived from (device-FW MIN is optimistic, the isolated-op bench is pessimistic).

    LTX_USE_FUSED=1 (default) = hybrid per-shape routing; =0 = all-standalone. Run both to compare.
    """
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)

    # Skip-ablation: identity-patch an op type so the traced-wall drop = its TRUE e2e contribution
    # (immune to the per-block-flush device-FW inflation that distorts op-share absolutes). Output is
    # garbage under ablation — this measures device time only. LTX_ABLATE=norm zeroes the RMSNorm cost.
    ablate = os.environ.get("LTX_ABLATE", "")
    if "norm" in ablate:
        for m in _walk(tt_decoder):
            if isinstance(m, RMSNorm):
                m.forward = lambda x, **kw: x
    if "binary" in ablate:
        # Skip every elementwise add/mul (residual adds + denorm + W-mask); arg0 is always the full
        # activation, so returning it is shape-safe. Measures BinaryNg's true e2e contribution.
        ttnn.add = lambda a, b, *p, **kw: a
        ttnn.multiply = lambda a, b, *p, **kw: a

    latent = _latent()
    sample_tt, logical_h, logical_w = _prepare_decode_input(tt_decoder, latent)

    # Warmup: cold-compile every program in the decode (trace capture requires cached programs).
    _ = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.synchronize_device(mesh)

    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(_TRACE_ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    wall_ms = (time.perf_counter() - t0) * 1000 / _TRACE_ITERS
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)

    mode = "hybrid(fused)" if _USE_FUSED else "all-standalone"
    print(
        f"\nTRACED_DECODE_WALL_MS={wall_ms:.2f}  mode={mode}  frames={_NUM_FRAMES}  {_HEIGHT}x{_WIDTH}  "
        f"iters={_TRACE_ITERS}",
        flush=True,
    )


# Per-op trace bench on the *LTX* conv (LTXCausalConv3d) — the production op, unlike the Wan-op bench in
# wan2_2/test_neighbor_pad_conv3d_fused_perf.py. Compares the fused NpConv3d vs the LTX standalone NP+conv
# per shape, in isolation. More representative than the Wan bench (right standalone path: LTX's
# persistent-buffer NP + temporal-pad concats), but still ISOLATED — the e2e routing metric is the
# whole-decode trace above. Fused-deployed shapes + a couple standalone for contrast; full LTX 2x4 dims.
# (id, C_in, C_out, T, H_full, W_full, deployed)
_LTX_OP_SHAPES = [
    ("s3_res", 256, 256, 147, 136, 240, "fused(halo_last)"),
    ("s3_chg", 256, 512, 147, 136, 240, "fused(halo_last)"),
    ("s4_res", 128, 128, 147, 272, 480, "fused(halo_last)"),
    ("s4_out", 128, 48, 147, 272, 480, "fused(force_spatial)"),
    ("s1_res", 512, 512, 39, 68, 120, "standalone"),
    ("s2_res", 512, 512, 75, 136, 240, "standalone"),
]


def _build_ltx_conv(mesh, c_in, c_out, T, H, W, *, use_fused):
    from models.tt_dit.utils.conv3d import ConvDims

    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh.shape)[0], mesh_axis=0),
        width_parallel=ParallelFactor(factor=tuple(mesh.shape)[1], mesh_axis=1),
    )
    ccl = CCLManager(mesh, topology=ttnn.Topology.Linear, num_links=2)
    conv = LTXCausalConv3d(
        c_in,
        c_out,
        kernel_size=3,
        mesh_device=mesh,
        parallel_config=pc,
        ccl_manager=ccl,
        conv_dims=ConvDims(T=T, H=H // tuple(mesh.shape)[0], W=W // tuple(mesh.shape)[1]),
        use_fused=use_fused,
    )
    torch.manual_seed(0)
    w = torch.randn(conv.out_channels, conv.in_channels, 3, 3, 3, dtype=torch.float32) * 0.01
    conv.load_torch_state_dict({"weight": w, "bias": torch.zeros(conv.out_channels)})
    # Bypass the hybrid MIN_T threshold so the fused path is exercised for every shape (matches decode use).
    if use_fused and conv._needs_halo and (conv.conv_config.halo_last or conv.conv_config.force_spatial_parallel):
        conv._use_fused = True
    return conv, pc


def _ltx_conv_input(mesh, pc, c_in, T, H, W):
    from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_width
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    torch.manual_seed(42)
    x = torch.randn(1, c_in, T, H, W, dtype=torch.float32).permute(0, 2, 3, 4, 1)  # B,T,H,W,C
    x, lh = conv_pad_height(x, pc.height_parallel.factor)
    x, lw = conv_pad_width(x, pc.width_parallel.factor)
    x = typed_tensor_2dshard(
        x,
        mesh,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    return x, lh, lw


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_bench_ltx_op(mesh_device, device_params):
    """Trace-mode per-op table on the LTX conv: fused NpConv3d vs LTX standalone NP+conv, per shape.

    Isolated (not e2e) but uses the production op/standalone path — unlike the Wan-op bench. The e2e
    routing metric remains the whole-decode trace (test_prof_vae_ltx_trace)."""
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    rows = []
    for sid, c_in, c_out, T, H, W, deployed in _LTX_OP_SHAPES:
        row = {"cid": sid, "ci": c_in, "co": c_out, "t": T, "hw": f"{H // 2}x{W // 4}", "dep": deployed}
        for key, uf in (("sa", False), ("f", True)):
            try:
                conv, pc = _build_ltx_conv(mesh, c_in, c_out, T, H, W, use_fused=uf)
                x, lh, lw = _ltx_conv_input(mesh, pc, c_in, T, H, W)
                conv(x, causal=False, logical_h=lh, logical_w=lw)  # warmup + cold-compile
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                conv(x, causal=False, logical_h=lh, logical_w=lw)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(_TRACE_ITERS):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                row[key] = (time.perf_counter() - t0) * 1e6 / _TRACE_ITERS
                ttnn.release_trace(mesh, tid)
            except Exception as e:  # noqa: BLE001
                row[f"{key}_err"] = type(e).__name__
                print(f"{sid} {'fused' if uf else 'standalone'} FAILED: {str(e)[:160]}", flush=True)
        rows.append(row)
        print(f"LTX-OP-BENCH {sid}: standalone={row.get('sa')} fused={row.get('f')}", flush=True)

    cid_w = max(len("config_id"), max(len(r["cid"]) for r in rows))
    hdr = (
        f"{'config_id':<{cid_w}}  {'C_in':>5} {'C_out':>5} {'T':>4} {'HxW(dev)':>9} "
        f"{'standalone us':>13} {'fused us':>10} {'speedup':>8}  {'deployed':<18}"
    )
    box = "=" * len(hdr)
    lines = [box, "LTX-op isolated trace bench (BH 2x4)", box, hdr, "-" * len(hdr)]
    for r in rows:
        sa = f"{r['sa']:>13.1f}" if r.get("sa") is not None else f"{r.get('sa_err', 'n/a'):>13}"
        fu = f"{r['f']:>10.1f}" if r.get("f") is not None else f"{r.get('f_err', 'n/a'):>10}"
        sp = f"{r['sa'] / r['f']:>7.2f}x" if (r.get("sa") and r.get("f")) else f"{'-':>8}"
        lines.append(
            f"{r['cid']:<{cid_w}}  {r['ci']:>5} {r['co']:>5} {r['t']:>4} {r['hw']:>9} {sa} {fu} {sp}  {r['dep']:<18}"
        )
    lines += [box, "speedup = standalone/fused (>1.0 => fusion faster); isolated per-op, not e2e."]
    print("\n" + "\n".join(lines), flush=True)


# GATE for the dual-output add+norm fusion: the dual-output op == fused-pre-add norm (already exists via
# residual_input_tensor) + one extra write of the sum. So the recoverable saving is bounded by
# [add + plain_norm] - [fused_norm_with_residual]. If that gap is small, the dual-output build (kernel +
# device op + writer + decode refactor) is not worth it. Trace-mode wall, existing ops, no build.
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_bench_norm_residual_gate(mesh_device, device_params):
    import time as _t

    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))

    def trace_time(run, n=30):
        run()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = _t.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (_t.perf_counter() - t0) * 1e6 / n
        ttnn.release_trace(mesh, tid)
        return us

    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    for rows, C in [(32768, 256), (16384, 512), (65536, 128)]:
        h = ttnn.from_torch(
            torch.randn(1, rows, C),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        r = ttnn.from_torch(
            torch.randn(1, rows, C),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )

        def status_quo():
            s = ttnn.add(r, h)
            ttnn.experimental.dit_rms_norm_unary_fused(s, epsilon=1e-8, compute_kernel_config=ckc)

        def fused_resid():
            ttnn.experimental.dit_rms_norm_unary_fused(
                h, epsilon=1e-8, residual_input_tensor=r, compute_kernel_config=ckc
            )

        sq = trace_time(status_quo)
        fr = trace_time(fused_resid)
        # dual-output ~= fr + 1 activation write; status quo = sq. Recoverable = sq - fr (upper bound).
        print(
            f"NORM-RESID-GATE rows={rows} C={C}: add+norm={sq:.1f}us  fused_resid_norm={fr:.1f}us  "
            f"recoverable_upper_bound={sq - fr:.1f}us ({100 * (sq - fr) / sq:.0f}% of add+norm)",
            flush=True,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(r)


# Isolation PCC for the dual-output RMSNorm: validate (1) the teed bf16 sum == input + residual
# (catches the fp32->bf16 dual-pack data-format hazard), and (2) the normed output still matches a
# torch RMSNorm reference with the 2nd output wired. No decode/diffusers needed.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dual_output_rmsnorm_isolation(mesh_device):
    from models.common.utility_functions import comp_pcc

    mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    eps = 1e-8
    torch.manual_seed(0)
    all_ok = True
    # TILE input works at all widths incl C=1024 (Wt=32); the RM-input path hangs at Wt=32 (wiki §4i).
    for rows, C in [(2048, 256), (4096, 128), (1024, 512), (512, 1024)]:
        h_t = torch.randn(1, rows, C)
        r_t = torch.randn(1, rows, C)
        sum_ref = h_t + r_t
        var = sum_ref.pow(2).mean(-1, keepdim=True)
        normed_ref = sum_ref * torch.rsqrt(var + eps)

        h = ttnn.from_torch(
            h_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )
        r = ttnn.from_torch(
            r_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )

        normed, sm = ttnn.experimental.dit_rms_norm_unary_fused_residual_sum(
            h, r, epsilon=eps, compute_kernel_config=ckc
        )

        sm_tt = ttnn.to_torch(sm).float()
        normed_tt = ttnn.to_torch(normed).float()
        ok_sum, pcc_sum = comp_pcc(sum_ref, sm_tt, 0.999)
        ok_norm, pcc_norm = comp_pcc(normed_ref, normed_tt, 0.999)
        all_ok = all_ok and ok_sum and ok_norm
        print(
            f"DUAL-OUT-ISO rows={rows} C={C}: sum_pcc={pcc_sum} ({'OK' if ok_sum else 'FAIL'})  "
            f"norm_pcc={pcc_norm} ({'OK' if ok_norm else 'FAIL'})",
            flush=True,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(r)
        ttnn.deallocate(normed)
        ttnn.deallocate(sm)
    assert all_ok, "dual-output RMSNorm isolation PCC failed"


# Isolation PERF gate for the decode refactor: does the dual-output op (normed + sum, one extra bf16
# pack) actually beat the status quo [add + plain_norm]? If not, wiring it into the decode is wasted.
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_bench_dual_output_perf(mesh_device, device_params):
    import time as _t

    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))

    def trace_time(run, n=30):
        run()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = _t.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (_t.perf_counter() - t0) * 1e6 / n
        ttnn.release_trace(mesh, tid)
        return us

    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    for rows, C in [(32768, 256), (16384, 512), (65536, 128)]:
        h = ttnn.from_torch(
            torch.randn(1, rows, C),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        r = ttnn.from_torch(
            torch.randn(1, rows, C),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )

        def status_quo():
            s = ttnn.add(r, h)
            ttnn.experimental.dit_rms_norm_unary_fused(s, epsilon=1e-8, compute_kernel_config=ckc)

        def dual_out():
            ttnn.experimental.dit_rms_norm_unary_fused_residual_sum(h, r, epsilon=1e-8, compute_kernel_config=ckc)

        sq = trace_time(status_quo)
        du = trace_time(dual_out)
        print(
            f"DUAL-OUT-PERF rows={rows} C={C}: add+norm={sq:.1f}us  dual_output={du:.1f}us  "
            f"speedup={100 * (sq - du) / sq:.0f}%",
            flush=True,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(r)


# RM-realistic gate: the LTX resnet stream is ROW_MAJOR (convs need RM). The dual-output op is TILE-only,
# so fusing it into the chain costs to_layout(TILE) on h + to_layout(RM) on normed per block. This models
# the real per-block-boundary cost: [RM add + RM norm] vs [convert + TILE dual-output + convert back].
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_bench_dual_output_rm(mesh_device, device_params):
    import time as _t

    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))

    def trace_time(run, n=30):
        run()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = _t.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (_t.perf_counter() - t0) * 1e6 / n
        ttnn.release_trace(mesh, tid)
        return us

    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    act = ttnn.UnaryOpType.SILU
    for rows, C in [(32768, 256), (65536, 128)]:
        # h and residual arrive ROW_MAJOR (conv outputs).
        h_rm = ttnn.from_torch(
            torch.randn(1, rows, C),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        r_rm = ttnn.from_torch(
            torch.randn(1, rows, C),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        r_tile = ttnn.to_layout(r_rm, ttnn.TILE_LAYOUT)

        def status_quo_rm():
            s = ttnn.add(r_rm, h_rm)
            n = ttnn.experimental.dit_rms_norm_unary_fused(s, epsilon=1e-8, compute_kernel_config=ckc, activation=act)
            ttnn.to_layout(n, ttnn.ROW_MAJOR_LAYOUT)

        def fused_rm_native():
            # RM-native: h stays RM (tilized internally), residual is TILE (prev block's sum), normed
            # comes back RM, sum stays TILE — no caller-side to_layout.
            normed, _sm = ttnn.experimental.dit_rms_norm_unary_fused_residual_sum(
                h_rm, r_tile, epsilon=1e-8, compute_kernel_config=ckc, activation=act
            )

        sq = trace_time(status_quo_rm)
        fr = trace_time(fused_rm_native)
        print(
            f"DUAL-OUT-RM rows={rows} C={C}: RM[add+norm]={sq:.1f}us  RM-native[dual]={fr:.1f}us  "
            f"speedup={100 * (sq - fr) / sq:.0f}% ({'WIN' if fr < sq else 'LOSS'})",
            flush=True,
        )
        ttnn.deallocate(h_rm)
        ttnn.deallocate(r_rm)
        ttnn.deallocate(r_tile)


# RM-native isolation PCC: input h ROW_MAJOR (conv output), residual TILE (prev block's sum). The op
# tilizes h internally, emits normed as RM (for conv1) and the sum as TILE (next block's residual) — no
# caller-side conversions. Validates the RM compute path + RM writer's 2nd (TILE) output.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dual_output_rmsnorm_rm_isolation(mesh_device):
    from models.common.utility_functions import comp_pcc

    mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    eps = 1e-8
    torch.manual_seed(0)
    all_ok = True
    # C=1024 is the LTX mid-block width (Wt=32); covered here after the RM+fused in0_t deadlock fix.
    for rows, C in [(2048, 256), (4096, 128), (512, 1024)]:
        h_t = torch.randn(1, rows, C)
        r_t = torch.randn(1, rows, C)
        sum_ref = h_t + r_t
        var = sum_ref.pow(2).mean(-1, keepdim=True)
        normed_ref = sum_ref * torch.rsqrt(var + eps)

        h = ttnn.from_torch(
            h_t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )
        r = ttnn.from_torch(
            r_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )

        normed, sm = ttnn.experimental.dit_rms_norm_unary_fused_residual_sum(
            h, r, epsilon=eps, compute_kernel_config=ckc
        )

        assert normed.layout == ttnn.ROW_MAJOR_LAYOUT, f"normed must be RM, got {normed.layout}"
        assert sm.layout == ttnn.TILE_LAYOUT, f"sum must be TILE, got {sm.layout}"
        sm_tt = ttnn.to_torch(sm).float()
        normed_tt = ttnn.to_torch(normed).float()
        ok_sum, pcc_sum = comp_pcc(sum_ref, sm_tt, 0.999)
        ok_norm, pcc_norm = comp_pcc(normed_ref, normed_tt, 0.999)
        all_ok = all_ok and ok_sum and ok_norm
        print(
            f"DUAL-OUT-RM-ISO rows={rows} C={C}: sum_pcc={pcc_sum} ({'OK' if ok_sum else 'FAIL'})  "
            f"norm_pcc={pcc_norm} ({'OK' if ok_norm else 'FAIL'})",
            flush=True,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(r)
        ttnn.deallocate(normed)
        ttnn.deallocate(sm)
    assert all_ok, "RM-native dual-output isolation PCC failed"


# TT-vs-TT decode PCC: the mid-block norm-add fusion must not change the decode output. Traced
# decode_device (untraced full forward x2 is too slow), fusion OFF vs ON, same decoder/weights/input.
# Catches any (h, residual) chain-rewiring bug; bf16-sum vs bf16-add should hold PCC ~1.0.
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_dual_output_decode_pcc(mesh_device, device_params):
    import models.tt_dit.models.vae.vae_ltx as vae_mod
    from models.common.utility_functions import comp_pcc
    from models.tt_dit.utils.tensor import fast_device_to_host

    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)
    torch.manual_seed(123)
    latent = _latent()
    sample_tt, logical_h, logical_w = _prepare_decode_input(tt_decoder, latent)
    concat_dims = [None, None]
    concat_dims[tt_decoder.parallel_config.height_parallel.mesh_axis] = 2
    concat_dims[tt_decoder.parallel_config.width_parallel.mesh_axis] = 3

    def decode_to_host():
        # Single untraced decode_device + gather. No trace: capture would re-dispatch the whole decode a
        # second time (untraced), doubling the cost for a one-shot comparison.
        out = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
        ttnn.synchronize_device(mesh)
        host = fast_device_to_host(out, mesh, concat_dims, ccl_manager=tt_decoder.ccl_manager)
        ttnn.deallocate(out)
        return host

    vae_mod._FUSE_MIDBLOCK_NORM_ADD = False
    ref = decode_to_host()
    vae_mod._FUSE_MIDBLOCK_NORM_ADD = True
    fused = decode_to_host()

    ok, pcc = comp_pcc(ref, fused, 0.99)
    print(f"DECODE-FUSE-PCC: pcc={pcc} ({'OK' if ok else 'FAIL'})", flush=True)
    assert ok, f"mid-block norm-add fusion changed the decode output: {pcc}"


# Fast wiring check: validate the mid-block (h, residual) deferral on ONE mid-block at its real
# production input shape (1,19,34,60,1024 → H/W fractured 2x4 → 17x15 per device, logical 34x60), instead
# of a full untraced decode (~23 min each). The input is synthesized directly (no decode, no abort), so
# the device queue is clean and a hang is attributable to a single run. The chain rewiring is
# shape-independent, so this gives the same correctness signal as a full-decode PCC, far faster.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_dual_output_midblock_pcc(mesh_device, device_params):
    import models.tt_dit.models.vae.vae_ltx as vae_mod
    from models.common.utility_functions import comp_pcc
    from models.tt_dit.models.vae.vae_ltx import LTXUNetMidBlock3D
    from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)
    pc = tt_decoder.parallel_config

    mids = [m for m in tt_decoder.up_blocks if isinstance(m, LTXUNetMidBlock3D)]
    assert mids, "no LTXUNetMidBlock3D in the decoder"
    target = mids[0]
    c = 1024  # first mid-block channels, from a captured decode run (input was 1,19,17,15,1024 per device)
    n_blocks = len(target.res_blocks)
    logical_h, logical_w = 34, 60  # first mid-block's stage dims (captured from a prior decode run)
    print(f"MIDBLOCK: first mid-block {n_blocks} res_blocks, C={c}, logical {logical_h}x{logical_w}", flush=True)

    torch.manual_seed(123)
    full = torch.randn(1, 19, logical_h, logical_w, c)  # (B, T, H, W, C), H/W fractured across the mesh

    def make_input():
        return typed_tensor_2dshard(
            full,
            mesh,
            shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    concat_dims = [None, None]
    concat_dims[pc.height_parallel.mesh_axis] = 2
    concat_dims[pc.width_parallel.mesh_axis] = 3

    def run(fuse):
        vae_mod._FUSE_MIDBLOCK_NORM_ADD = fuse
        out = target(make_input(), causal=False, logical_h=logical_h, logical_w=logical_w)
        ttnn.synchronize_device(mesh)
        host = fast_device_to_host(out, mesh, concat_dims, ccl_manager=tt_decoder.ccl_manager)
        ttnn.deallocate(out)
        return host

    ref = run(False)
    print("MIDBLOCK: unfused run done", flush=True)
    fused = run(True)
    print("MIDBLOCK: fused run done", flush=True)
    ok, pcc = comp_pcc(ref, fused, 0.99)
    print(f"MIDBLOCK-FUSE-PCC: pcc={pcc} ({'OK' if ok else 'FAIL'})  blocks={n_blocks}", flush=True)
    assert ok, f"mid-block norm-add fusion changed the output: {pcc}"


# Bisect the RM Wt=32 hang: does the BASE fused-residual norm (RM input + TILE residual, NO output_residual_sum)
# hang at C=1024? If yes -> the base RM+FUSE_PRE_ADD path is the culprit (my validate-relax exposed it).
# If no -> my dual-output additions (cb_x_out tee + RM-writer drain) are. (1,1) submesh, ~15s.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_base_rm_residual_wt32(mesh_device):
    from models.common.utility_functions import comp_pcc

    mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    eps = 1e-8
    torch.manual_seed(0)
    rows, C = 512, 1024
    h_t = torch.randn(1, rows, C)
    r_t = torch.randn(1, rows, C)
    ref = h_t + r_t
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + eps)
    h = ttnn.from_torch(
        h_t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
    )
    r = ttnn.from_torch(
        r_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
    )
    print("BASE-RM-WT32: calling base fused-residual norm (no dual output)...", flush=True)
    normed = ttnn.experimental.dit_rms_norm_unary_fused(
        h, epsilon=eps, residual_input_tensor=r, compute_kernel_config=ckc
    )
    ttnn.synchronize_device(mesh)
    ok, pcc = comp_pcc(ref, ttnn.to_torch(normed).float(), 0.999)
    print(f"BASE-RM-WT32: completed, pcc={pcc} ({'OK' if ok else 'FAIL'})", flush=True)
    assert ok


# Validate the dual-output op WITH SiLU activation (the mid-block norm1 uses SiLU). normed must be
# SiLU(rms(h+r)); the sum output must stay RAW (h+r, no activation). Isolation tests above used no
# activation, so this checks the activation path doesn't leak into the sum.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dual_output_activation_isolation(mesh_device):
    import torch.nn.functional as F

    from models.common.utility_functions import comp_pcc

    mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    eps = 1e-8
    torch.manual_seed(0)
    all_ok = True
    for rows, C, rm in [(512, 1024, True), (2048, 256, False)]:
        h_t = torch.randn(1, rows, C)
        r_t = torch.randn(1, rows, C)
        sum_ref = h_t + r_t
        normed_ref = sum_ref * torch.rsqrt(sum_ref.pow(2).mean(-1, keepdim=True) + eps)
        normed_ref = F.silu(normed_ref)
        layout = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
        h = ttnn.from_torch(h_t, layout=layout, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh)
        r = ttnn.from_torch(
            r_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )
        normed, sm = ttnn.experimental.dit_rms_norm_unary_fused_residual_sum(
            h, r, epsilon=eps, compute_kernel_config=ckc, activation=ttnn.UnaryOpType.SILU
        )
        ok_s, p_s = comp_pcc(sum_ref, ttnn.to_torch(sm).float(), 0.999)
        ok_n, p_n = comp_pcc(normed_ref, ttnn.to_torch(normed).float(), 0.999)
        all_ok = all_ok and ok_s and ok_n
        print(
            f"ACT-ISO rows={rows} C={C} rm={rm}: RAW_sum_pcc={p_s} ({'OK' if ok_s else 'FAIL'})  "
            f"silu_norm_pcc={p_n} ({'OK' if ok_n else 'FAIL'})",
            flush=True,
        )
    assert all_ok


# Localize the mid-block 0.876: compare block-0's output computed two ways at real conv-produced values —
# (a) block0.forward materialized add(x0, h0), (b) block1.norm1.forward_residual_sum(h0, x0).sum = h0+x0.
# If these match (~0.999), the op's sum is correct and the 0.876 is downstream fp32-vs-bf16 amplification
# through the (random) convs. If low, the op's sum is wrong at conv values.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_midblock_sum_debug(mesh_device, device_params):
    from models.common.utility_functions import comp_pcc
    from models.tt_dit.models.vae.vae_ltx import LTXUNetMidBlock3D
    from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)
    pc = tt_decoder.parallel_config
    target = [m for m in tt_decoder.up_blocks if isinstance(m, LTXUNetMidBlock3D)][0]
    b0, b1 = target.res_blocks[0], target.res_blocks[1]
    lh, lw, c = 34, 60, 1024
    torch.manual_seed(123)
    full = torch.randn(1, 19, lh, lw, c)
    cd = [None, None]
    cd[pc.height_parallel.mesh_axis] = 2
    cd[pc.width_parallel.mesh_axis] = 3

    def mk():
        return typed_tensor_2dshard(
            full,
            mesh,
            shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    # (a) materialized block-0 output
    x1_add = b0(mk(), causal=False, logical_h=lh, logical_w=lw)
    ttnn.synchronize_device(mesh)
    x1_add_h = fast_device_to_host(x1_add, mesh, cd, ccl_manager=tt_decoder.ccl_manager)
    # (b) deferred halves: h0, x0; then block1 norm1 sum = h0 + x0
    h0, x0t = b0.forward_deferred(mk(), None, causal=False, logical_h=lh, logical_w=lw)
    _normed, x1_sum = b1.norm1.forward_residual_sum(h0, x0t, compute_kernel_config=b1.norm_compute_kernel_config)
    ttnn.synchronize_device(mesh)
    x1_sum_h = fast_device_to_host(
        ttnn.to_layout(x1_sum, ttnn.ROW_MAJOR_LAYOUT), mesh, cd, ccl_manager=tt_decoder.ccl_manager
    )
    ok, pcc = comp_pcc(x1_add_h.float(), x1_sum_h.float(), 0.99)
    print(f"MIDBLOCK-SUM-DEBUG: x1(add) vs x1(op-sum) pcc={pcc} ({'OK' if ok else 'FAIL'})", flush=True)


# Does the op's sum break at the 5D mid-block shape (1,19,17,15,1024) where the -2 dim (15) is NOT
# tile-aligned (padded 15->32)? Clean inputs, so this isolates "non-aligned -2 dim" from "conv values".
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dual_output_5d_shape(mesh_device):
    from models.common.utility_functions import comp_pcc

    mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    eps = 1e-8
    torch.manual_seed(0)
    for shape, in_layout in [
        ((1, 19, 17, 15, 1024), ttnn.ROW_MAJOR_LAYOUT),
        ((1, 19, 17, 15, 1024), ttnn.TILE_LAYOUT),
        ((1, 19, 17, 32, 1024), ttnn.ROW_MAJOR_LAYOUT),
    ]:
        h_t = torch.randn(*shape)
        r_t = torch.randn(*shape)
        sum_ref = h_t + r_t
        normed_ref = sum_ref * torch.rsqrt(sum_ref.pow(2).mean(-1, keepdim=True) + eps)
        h = ttnn.from_torch(
            h_t, layout=in_layout, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )
        r = ttnn.from_torch(
            r_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh
        )
        normed, sm = ttnn.experimental.dit_rms_norm_unary_fused_residual_sum(
            h, r, epsilon=eps, compute_kernel_config=ckc
        )
        _, ps = comp_pcc(sum_ref, ttnn.to_torch(sm).float(), 0.999)
        _, pn = comp_pcc(normed_ref, ttnn.to_torch(normed).float(), 0.999)
        # Base op (no dual output, no cb_x_out tee) at the same shape — isolates base RM+FUSE_PRE_ADD.
        base = ttnn.experimental.dit_rms_norm_unary_fused(
            h, epsilon=eps, residual_input_tensor=r, compute_kernel_config=ckc
        )
        _, pb = comp_pcc(normed_ref, ttnn.to_torch(base).float(), 0.999)
        il = "RM " if in_layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"
        al = "ALIGNED" if shape[-2] % 32 == 0 else "NOT-aligned"
        print(f"5D-SHAPE {shape} in={il} -2={al}: sum_pcc={ps}  norm_pcc={pn}  BASE_norm_pcc={pb}", flush=True)
