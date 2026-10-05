# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""#134 device A/B of LTX_SDPA_MM_LOFI (ring SDPA QK^T and softmax @ V matmuls at LoFi).

Arms (T134_ARMS picks the ones this process runs):
  REF  - pre-change build (blx03 ~/fasth3/t48: no #57979/#58032/#58223, no knob). Runs in its own
         process first and leaves its outputs in T134_OUT.
  OFF  - this branch's build, LTX_SDPA_MM_LOFI unset. Must be bit-identical to REF.
  LOFI - this branch's build, LTX_SDPA_MM_LOFI=1.

Phase A, per arm and shape (default S2 = (19,34,60)), AV block 0 of the 22B checkpoint on the Linear 2x4
sp1/tp0 layout, carved from the full mesh:
  T134_BLOCK  traced ms per block (median of 3 laps x T134_REPLAYS replays).
  T134_RING   each ring SDPA call of the block traced on its own, ms per call, and their sum
              (T134_RING_SUM). The call alternates its AG semaphore pair with the ping-pong partner,
              as back-to-back blocks do.
  T134_CMP    block outputs vs REF and LOFI vs OFF: bit-identical flag, PCC, PSNR, max abs diff.
              PSNR is 20*log10(max|ref| / rmse): activations have no fixed peak.
  T134_RING_CMP  the isolated ring SDPA output, LOFI vs OFF.
Phase B (T134_TORCH=1): video-only block vs the diffusers torch block at S1 (scaled random weights),
per arm: T134_TORCH (PCC and relative RMSE vs torch) and T134_TORCH_CMP (vs REF, LOFI vs OFF).
T134_GATE off_identical_to_ref is the default-path check: the cherry-picks must not move knob-off output.
"""

import gc
import math
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
import models.tt_dit.tests.models.ltx.test_transformer_ltx as T
from models.tt_dit.models.transformers.ltx import attention_ltx as _A
from models.tt_dit.utils.tracing import Tracer

_ROOT = os.environ["TT_METAL_HOME"]
# Fail at collection, before the mesh opens, if the models or ttnn come from another tree than the build.
for _mod in (_A, ttnn):
    assert os.path.realpath(_mod.__file__).startswith(os.path.realpath(_ROOT)), f"{_mod.__file__} not under {_ROOT}"

ARMS = os.environ.get("T134_ARMS", "OFF,LOFI").split(",")
assert set(ARMS) <= {"REF", "OFF", "LOFI"} and ARMS, ARMS
OUT = os.environ.get("T134_OUT", "/var/tmp/fasth3/t134")
SHAPES = {"S1": (19, 17, 30), "S2": (19, 34, 60)}
RUN_SHAPES = os.environ.get("T134_SHAPES", "S2").split(",")
N_REPLAY = int(os.environ.get("T134_REPLAYS", "10"))
N_LAPS = 3
SP_AXIS, TP_AXIS = 1, 0
ATTNS = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
KNOB = "LTX_SDPA_MM_LOFI"
ERRORS = []

_SD = {}
_orig_load = T._load_22b_state_dict


def _load_cached(num_layers, checkpoint_path):
    key = (num_layers, checkpoint_path)
    if key not in _SD:
        _SD[key] = _orig_load(num_layers, checkpoint_path)
    return None if _SD[key] is None else dict(_SD[key])


T._load_22b_state_dict = _load_cached

_RING = ttnn.transformer.ring_joint_scaled_dot_product_attention


class _RingCapture:
    """Pass-through for the ring SDPA op that records each call's arguments while active."""

    def __init__(self):
        self.active = False
        self.calls = []

    def __call__(self, *args, **kwargs):
        if self.active:
            self.calls.append((args, kwargs))
        return _RING(*args, **kwargs)


def _set_knob(arm):
    if arm == "LOFI":
        os.environ[KNOB] = "1"
    else:
        os.environ.pop(KNOB, None)


def _cmp(ref, got):
    identical = ref.shape == got.shape and torch.equal(ref, got)
    r, g = ref.flatten().double(), got.flatten().double()
    rmse = (r - g).pow(2).mean().sqrt().item()
    pcc = 1.0 if identical else torch.corrcoef(torch.stack([r, g]))[0, 1].item()
    psnr = math.inf if rmse == 0 else 20 * math.log10(r.abs().max().item() / rmse)
    return dict(identical=identical, pcc=pcc, psnr=psnr, maxabs=(r - g).abs().max().item())


def _fmt(c):
    return f"identical={c['identical']} pcc={c['pcc']:.6f} psnr_db={c['psnr']:.2f} maxabs={c['maxabs']:.4g}"


def _mm_fidelity(config):
    return getattr(config, "matmul_math_fidelity", None)


def _check_knob_reached(block, arm):
    """Every SDPA config of every attention carries the arm's matmul fidelity."""
    want = ttnn.MathFidelity.LoFi if arm == "LOFI" else None
    seen = []
    for name in ATTNS:
        a = getattr(block, name, None)
        if a is None:
            continue
        assert bool(getattr(a, "sdpa_mm_lofi", False)) == (arm == "LOFI"), f"{arm}: {name}.sdpa_mm_lofi"
        configs = [a.sdpa_program_config, a.ring_sdpa_program_config, a.cross_ring_sdpa_program_config]
        configs += list(a._ring_pc_by_n.values()) + list(a._sdpa_pc_by_shape.values())
        seen += [_mm_fidelity(c) for c in configs if c is not None]
    assert seen and all(f == want for f in seen), f"{arm}: matmul fidelities {set(map(str, seen))}, want {want}"


def _timed_replays(mesh, call):
    laps = []
    out = None
    for _ in range(N_LAPS):
        t0 = time.perf_counter()
        for _ in range(N_REPLAY):
            out = call()
        ttnn.synchronize_device(mesh)
        laps.append((time.perf_counter() - t0) / N_REPLAY * 1e3)
    return statistics.median(laps), laps, out


def _partner_semaphores(ccl, kwargs):
    """The other half of the AG ping-pong pool on the call's axis (the pool holds two sets)."""
    pool = ccl.ag_ping_pong_semaphores[kwargs["cluster_axis"]]
    sems = kwargs["multi_device_global_semaphore"]
    half = len(pool) // 2
    if pool[0] is sems[0]:
        return list(pool[half:])
    if pool[half] is sems[0]:
        return list(pool[:half])
    raise LookupError("captured ring SDPA semaphores are not in the AG ping-pong pool")


def _concat_bhne(mesh, t):
    dims = [None, None]
    dims[TP_AXIS], dims[SP_AXIS] = 1, 2
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=dims, mesh_shape=tuple(mesh.shape)))


def _ring_alone(mesh, ccl, arm, name, idx, call):
    """Traces one captured ring SDPA call (A, then B on the partner semaphores) and times it alone."""
    args, kwargs = call
    kwargs_b = dict(kwargs, multi_device_global_semaphore=_partner_semaphores(ccl, kwargs))

    def pair():
        return (*_RING(*args, **kwargs), *_RING(*args, **kwargs_b))

    tracer = Tracer(pair, device=mesh, prep_run=True, clone_prep_inputs=False)
    _ = tracer(traced=True)
    ttnn.synchronize_device(mesh)
    ms, laps, outs = _timed_replays(mesh, lambda: tracer(traced=True))
    ms, laps = ms / 2, [x / 2 for x in laps]
    out = _concat_bhne(mesh, outs[0])
    tracer.release_trace()
    del tracer, outs
    pc = kwargs["program_config"]
    kind = "cross" if kwargs.get("is_cross") else "self"
    logger.info(
        f"T134_RING arm={arm} {name} call={idx} kind={kind} q={tuple(args[0].shape)} k={tuple(args[1].shape)} "
        f"logical_n={kwargs.get('logical_n')} q_chunk={pc.q_chunk_size} k_chunk={pc.k_chunk_size} "
        f"mm_fidelity={_mm_fidelity(pc)} ms_per_call={ms:.3f} laps={[round(x, 3) for x in laps]}"
    )
    return ms, out


def _arm(mesh, arm, name):
    F, H, W = SHAPES[name]
    _set_knob(arm)
    try:
        block, kw, video_N_real, audio_N_real = T._build_block_trace_setup(
            mesh_device=mesh,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            num_links=2,
            topology=ttnn.Topology.Linear,
            F=F,
            H=H,
            W=W,
            checkpoint_variant="fast",
        )
    finally:
        _set_knob("OFF")
    _check_knob_reached(block, arm)

    cap = _RingCapture()
    ttnn.transformer.ring_joint_scaled_dot_product_attention = cap
    try:
        cap.active = True
        _ = block(**kw)
        cap.active = False
        ttnn.synchronize_device(mesh)
        assert cap.calls, "the block made no ring SDPA call"
        want = ttnn.MathFidelity.LoFi if arm == "LOFI" else None
        assert all(_mm_fidelity(c[1]["program_config"]) == want for c in cap.calls), f"{arm}: ring call fidelity"

        tracer = Tracer(block.forward, device=mesh, prep_run=False, clone_prep_inputs=False)
        _ = tracer(**kw, traced=True)
        ttnn.synchronize_device(mesh)
        ms, laps, out = _timed_replays(mesh, lambda: tracer(**kw, traced=True))
        dims = [None, None]
        dims[SP_AXIS], dims[TP_AXIS] = 2, 3
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=dims, mesh_shape=tuple(mesh.shape))
        v = ttnn.to_torch(out[0], mesh_composer=comp).squeeze(0)[:, :video_N_real]
        a = ttnn.to_torch(out[1], mesh_composer=comp).squeeze(0)[:, :audio_N_real]
        tracer.release_trace()
        del tracer, out
        logger.info(
            f"T134_BLOCK arm={arm} {name} F,H,W={F},{H},{W} ms_per_block={ms:.3f} "
            f"laps={[round(x, 3) for x in laps]} replays={N_REPLAY} ring_calls={len(cap.calls)}"
        )
        assert torch.isfinite(v.float()).all() and torch.isfinite(a.float()).all(), f"{arm}: non-finite output"

        ring_ms, ring_out = [], []
        if os.environ.get("T134_RING", "1") == "1":
            ccl = block.attn1.ccl_manager
            try:
                for idx, call in enumerate(cap.calls):
                    rms, rout = _ring_alone(mesh, ccl, arm, name, idx, call)
                    ring_ms.append(rms)
                    ring_out.append(rout)
            except Exception as e:  # the block numbers above still feed the comparisons
                logger.exception(f"T134_FAIL ring {arm} {name}: {e}")
                ERRORS.append(f"ring {arm} {name}: {e}")
            logger.info(
                f"T134_RING_SUM arm={arm} {name} ring_ms={sum(ring_ms):.3f} block_ms={ms:.3f} "
                f"share={sum(ring_ms) / ms * 100:.1f}% per_call={[round(x, 3) for x in ring_ms]}"
            )
    finally:
        ttnn.transformer.ring_joint_scaled_dot_product_attention = _RING
        cap.calls.clear()
    del block, kw, cap
    gc.collect()
    return dict(ms=ms, v=v, a=a, ring_ms=ring_ms, ring_out=ring_out)


def _phase_a(mesh, name, ref):
    res = {}
    for arm in ARMS:
        res[arm] = _arm(mesh, arm, name)
    if "REF" in res:
        r = res["REF"]
        torch.save(dict(v=r["v"], a=r["a"], ms=r["ms"], ring_ms=r["ring_ms"]), os.path.join(OUT, f"ref_{name}.pt"))
        logger.info(f"T134_SAVED ref_{name}.pt ms={r['ms']:.3f} ring_ms={sum(r['ring_ms']):.3f}")
    if ref is not None:
        res.setdefault("REF", dict(ms=ref["ms"], v=ref["v"], a=ref["a"], ring_ms=ref["ring_ms"]))
    gates = {}
    for arm in ("OFF", "LOFI"):
        if arm in res and "REF" in res and arm in ARMS:
            cv, ca = _cmp(res["REF"]["v"], res[arm]["v"]), _cmp(res["REF"]["a"], res[arm]["a"])
            logger.info(
                f"T134_CMP {name} {arm}_vs_REF ms={res[arm]['ms']:.3f} ref_ms={res['REF']['ms']:.3f} "
                f"ring_ms={sum(res[arm]['ring_ms']):.3f} ref_ring_ms={sum(res['REF']['ring_ms']):.3f} "
                f"video: {_fmt(cv)} | audio: {_fmt(ca)}"
            )
            if arm == "OFF":
                gates["off_identical_to_ref"] = cv["identical"] and ca["identical"]
    if "OFF" in res and "LOFI" in res:
        off, lofi = res["OFF"], res["LOFI"]
        cv, ca = _cmp(off["v"], lofi["v"]), _cmp(off["a"], lofi["a"])
        logger.info(f"T134_CMP {name} LOFI_vs_OFF video: {_fmt(cv)} | audio: {_fmt(ca)}")
        ring_off, ring_lofi = sum(off["ring_ms"]), sum(lofi["ring_ms"])
        logger.info(
            f"T134_AB {name} block_off_ms={off['ms']:.3f} block_lofi_ms={lofi['ms']:.3f} "
            f"delta_ms={lofi['ms'] - off['ms']:+.3f} ({(lofi['ms'] / off['ms'] - 1) * 100:+.2f}%) "
            f"ring_off_ms={ring_off:.3f} ring_lofi_ms={ring_lofi:.3f} ring_delta_ms={ring_lofi - ring_off:+.3f}"
        )
        for idx, (ro, rl) in enumerate(zip(off["ring_out"], lofi["ring_out"])):
            logger.info(f"T134_RING_CMP {name} call={idx} LOFI_vs_OFF {_fmt(_cmp(ro, rl))}")
    return gates


def _phase_b(mesh, ref):
    F, H, W = SHAPES["S1"]
    captured = {}
    ref_cache = {}
    orig_ref = T._diffusers_video_block_ref
    orig_aq = T.assert_quality

    def ref_cached(*args, **kwargs):
        if "out" not in ref_cache:
            ref_cache["out"] = orig_ref(*args, **kwargs)
        return ref_cache["out"]

    def capture(ref_out, got, **_kwargs):
        r, g = ref_out.float(), got.float()
        captured["pcc"] = torch.corrcoef(torch.stack([r.flatten().double(), g.flatten().double()]))[0, 1].item()
        captured["rel_rmse"] = ((r - g).pow(2).mean().sqrt() / r.std()).item()
        captured["out"] = got

    T._diffusers_video_block_ref = ref_cached
    T.assert_quality = capture
    outs = {}
    try:
        for arm in ARMS:
            _set_knob(arm)
            torch.manual_seed(0)
            captured.clear()
            T.test_ltx_transformer_block(
                mesh_device=mesh,
                sp_axis=SP_AXIS,
                tp_axis=TP_AXIS,
                num_links=2,
                topology=ttnn.Topology.Linear,
                is_fsdp=False,
                F=F,
                H=H,
                W=W,
                has_audio=False,
                run_pcc=True,
                checkpoint_variant="fast",
                reset_seeds=None,
            )
            outs[arm] = captured["out"]
            logger.info(
                f"T134_TORCH arm={arm} F,H,W={F},{H},{W} pcc_vs_torch={captured['pcc']:.6f} "
                f"rel_rmse_vs_torch={captured['rel_rmse']:.5f}"
            )
            gc.collect()
    finally:
        _set_knob("OFF")
        T._diffusers_video_block_ref = orig_ref
        T.assert_quality = orig_aq
    if "REF" in outs:
        torch.save(outs["REF"], os.path.join(OUT, "ref_torch_S1.pt"))
    if ref is not None:
        outs.setdefault("REF", ref)
    for arm in ("OFF", "LOFI"):
        if arm in ARMS and "REF" in outs:
            logger.info(f"T134_TORCH_CMP {arm}_vs_REF {_fmt(_cmp(outs['REF'], outs[arm]))}")
    if "OFF" in outs and "LOFI" in outs:
        logger.info(f"T134_TORCH_CMP LOFI_vs_OFF {_fmt(_cmp(outs['OFF'], outs['LOFI']))}")


def _load_ref(fname):
    path = os.path.join(OUT, fname)
    if "REF" in ARMS or not os.path.exists(path):
        return None
    return torch.load(path)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_sdpa_mm_lofi_ab(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    os.makedirs(OUT, exist_ok=True)
    logger.info(f"T134_START arms={ARMS} shapes={RUN_SHAPES} root={_ROOT} replays={N_REPLAY}")
    errors, gates = ERRORS, {}
    for name in RUN_SHAPES:
        try:
            gates.update({f"{k}_{name}": v for k, v in _phase_a(mesh, name, _load_ref(f"ref_{name}.pt")).items()})
        except Exception as e:  # keep the other phases' numbers if one fails
            logger.exception(f"T134_FAIL phase_a {name}: {e}")
            errors.append(f"phase_a {name}: {e}")
    if os.environ.get("T134_TORCH", "1") == "1":
        try:
            _phase_b(mesh, _load_ref("ref_torch_S1.pt"))
        except Exception as e:
            logger.exception(f"T134_FAIL phase_b: {e}")
            errors.append(f"phase_b: {e}")
    for k, v in gates.items():
        logger.info(f"T134_GATE {k}={v}")
    assert not errors, errors
