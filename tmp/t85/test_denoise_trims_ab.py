# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""#85 device A/B of the opt-in denoise trims on a 2x4 submesh (Linear, sp1/tp0, real 22B block-0 weights).

Per shape, one block is built and traced once per arm: base, v2a (LTX_V2A_SKIP_PAD_MUL), adaln
(LTX_BATCH_ADALN_ADDS: the block takes pre-added table rows sliced from a stack, as the model
loop does; the per-step batched add is timed separately), all. Logs T85_BLOCK per arm and T85_AB
(delta, bit identity vs base). T85_STACK times the six (48, coeff, 1, D) batched adds one step adds
and checks a stacked row equals the per-block add. LTX_AGMM_K2048 does not trigger at TP=2
(A2V to_out K=2048 N=2048 here) and needs a 4x8 Ring, so it is not exercised.
"""

import gc
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
import models.tt_dit.tests.models.ltx.test_transformer_ltx as T
from models.tt_dit.models.transformers.ltx import transformer_ltx as TL
from models.tt_dit.utils.tracing import Tracer

_HERE = os.path.dirname(os.path.abspath(__file__))
assert TL.__file__.startswith(os.path.join(_HERE, "src")), f"t85 models/tt_dit not on path: {TL.__file__}"

SHAPES = {"S1": (19, 17, 30), "S2": (19, 34, 60)}
N_REPLAY = int(os.environ.get("T85_REPLAYS", "10"))
N_LAPS = 3
ARMS = {"base": (0, 0), "v2a": (1, 0), "adaln": (0, 1), "all": (1, 1)}
TEMB_KEYS = {
    "v": "video_temb",
    "pv": "video_prompt_temb",
    "a": "audio_temb",
    "pa": "audio_prompt_temb",
    "av": "av_ca_temb",
    "ava": "av_ca_audio_temb",
}


def _r4(t):
    return ttnn.reshape(t, (1, t.shape[0], 1, t.shape[3]))


def _pre_stacks(block, kw, nb=1):
    """(nb, coeff, 1, D) table+temb stacks built from block 0's tables (nb copies)."""
    out = {}
    for key, attr in TL.LTXTransformerModel._ADALN_TABLES:
        tab = _r4(getattr(block, attr).data)
        stack = ttnn.concat([tab] * nb, dim=0) if nb > 1 else tab
        out[key] = (stack, _r4(kw[TEMB_KEYS[key]]))
    return out


def _gather(mesh, out, video_N_real, audio_N_real):
    dims = [None, None]
    dims[1], dims[0] = 2, 3
    comp = ttnn.ConcatMesh2dToTensor(mesh, dims=dims, mesh_shape=tuple(mesh.shape))
    v = ttnn.to_torch(out[0], mesh_composer=comp).squeeze(0)[:, :video_N_real]
    a = ttnn.to_torch(out[1], mesh_composer=comp).squeeze(0)[:, :audio_N_real]
    return v, a


def _time(tracer, kw, mesh):
    laps = []
    for _ in range(N_LAPS):
        t0 = time.perf_counter()
        for _ in range(N_REPLAY):
            out = tracer(**kw, traced=True)
        ttnn.synchronize_device(mesh)
        laps.append((time.perf_counter() - t0) / N_REPLAY * 1e3)
    return statistics.median(laps), laps, out


def _arm(mesh, block, kw, name, video_N_real, audio_N_real):
    v2a, adaln = ARMS[name]
    TL.LTX_V2A_SKIP_PAD_MUL = bool(v2a)
    try:
        if adaln:
            # Pre-added rows live outside the trace (the model adds them once per step); the slices
            # stay inside it, as in the model loop.
            stacks = {k: ttnn.add(s, t) for k, (s, t) in _pre_stacks(block, kw).items()}

            def fwd(**k):
                pre = {key: TL._slice_block_row(st, 0) for key, st in stacks.items()}
                return block.forward(**k, adaln_pre=pre)

        else:
            fwd = block.forward
        _ = fwd(**kw)
        ttnn.synchronize_device(mesh)
        tracer = Tracer(fwd, device=mesh, prep_run=False, clone_prep_inputs=False)
        _ = tracer(**kw, traced=True)
        ttnn.synchronize_device(mesh)
        ms, laps, out = _time(tracer, kw, mesh)
        v, a = _gather(mesh, out, video_N_real, audio_N_real)
        tracer.release_trace()
        del tracer, out
        gc.collect()
    finally:
        TL.LTX_V2A_SKIP_PAD_MUL = False
    logger.info(f"T85_BLOCK arm={name} ms_per_block={ms:.3f} laps={[round(x, 3) for x in laps]} replays={N_REPLAY}")
    assert torch.isfinite(v.float()).all() and torch.isfinite(a.float()).all()
    return ms, v, a


def _stack_bench(mesh, block, kw, nb=48):
    src = _pre_stacks(block, kw, nb=nb)

    def step():
        return [ttnn.add(s, t) for s, t in src.values()]

    _ = step()
    ttnn.synchronize_device(mesh)
    tracer = Tracer(step, device=mesh, prep_run=False, clone_prep_inputs=False)
    _ = tracer(traced=True)
    ttnn.synchronize_device(mesh)
    ms, laps, outs = _time(tracer, {}, mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    exact = True
    for (key, attr), got in zip(TL.LTXTransformerModel._ADALN_TABLES, outs):
        row = TL._slice_block_row(got, nb - 1)
        ref = ttnn.chunk(getattr(block, attr).data + kw[TEMB_KEYS[key]], row.__len__(), dim=0)
        for g, r in zip(row, ref):
            exact &= torch.equal(ttnn.to_torch(g, mesh_composer=comp), ttnn.to_torch(r, mesh_composer=comp))
    tracer.release_trace()
    logger.info(f"T85_STACK nb={nb} ms_per_step_6adds={ms:.3f} laps={[round(x, 3) for x in laps]} row_exact={exact}")
    return ms, exact


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_denoise_trims_ab(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    errors = []
    for shape in os.environ.get("T85_SHAPES", "S1,S2").split(","):
        F, H, W = SHAPES[shape]
        try:
            block, kw, video_N_real, audio_N_real = T._build_block_trace_setup(
                mesh_device=mesh,
                sp_axis=1,
                tp_axis=0,
                num_links=2,
                topology=ttnn.Topology.Linear,
                F=F,
                H=H,
                W=W,
                checkpoint_variant="fast",
            )
            assert kw["video_padding_mask"] is not None, "no video SP padding at this shape"
            kw["video_kv_logical_n"] = video_N_real
            res = {}
            for name in ARMS:
                try:
                    res[name] = _arm(mesh, block, kw, name, video_N_real, audio_N_real)
                except Exception as e:
                    logger.exception(f"T85_FAIL {shape} {name}: {e}")
                    errors.append(f"{shape} {name}: {e}")
            if "base" in res:
                ms0, v0, a0 = res["base"]
                for name, (ms, v, a) in res.items():
                    if name == "base":
                        continue
                    logger.info(
                        f"T85_AB {shape} arm={name} base_ms={ms0:.3f} ms={ms:.3f} delta_ms={ms - ms0:+.3f} "
                        f"({(ms / ms0 - 1) * 100:+.2f}%) bit_identical={torch.equal(v0, v) and torch.equal(a0, a)} "
                        f"maxabs_v={(v0.float() - v.float()).abs().max().item():.4g} "
                        f"maxabs_a={(a0.float() - a.float()).abs().max().item():.4g}"
                    )
            if shape == "S1":
                try:
                    _stack_bench(mesh, block, kw)
                except Exception as e:
                    logger.exception(f"T85_FAIL stack: {e}")
                    errors.append(f"stack: {e}")
            del block, kw, res
            gc.collect()
        except Exception as e:
            logger.exception(f"T85_FAIL {shape}: {e}")
            errors.append(f"{shape}: {e}")
    assert not errors, errors
