# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""#110 block A/B of the gate fold (LTX_FUSE_GATE_ON_DEVICE) and LTX_FUSE_NORM_ADALN on a 2x4 submesh.

Same layout as #85: Linear, sp axis 1 / tp axis 0, real block-0 weights, S1 and S2 shapes, traced.
Arms: base, gate, adaln, both. LTX_FUSE_NORM_ADALN is read at construction, so each shape builds
two blocks (adaln 0/1) and times each one unfolded, then folded.

The fold only engages on Ring (a 2x4 has no wraparound link), and the Linear path's
minimal_matmul_split splits evenly, ignoring chunk_sizes. So the gate arms force the fold and
run the fused projection as one minimal_matmul plus per-chunk slices. That keeps the fused
numerics (one matmul, the Q/QKV compute config for the gate columns) but not the Ring timing:
on Ring the unfused gate is its own all-gather-matmul, on Linear only a small matmul on an
already-gathered input. The gate arms' ms are a Linear lower bound, their PCC is the real check.
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
from models.tt_dit.layers import linear as L
from models.tt_dit.models.transformers.ltx import transformer_ltx as TL
from models.tt_dit.utils.tracing import Tracer

_TREE = os.environ["T110_TREE"]
assert TL.__file__.startswith(_TREE), f"models/tt_dit not from {_TREE}: {TL.__file__}"

SHAPES = {"S1": (19, 17, 30), "S2": (19, 34, 60)}
N_REPLAY = int(os.environ.get("T110_REPLAYS", "10"))
N_LAPS = 3
ATTNS = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")

_orig_col_forward = L.ColParallelLinear.forward


def _col_forward(self, x, compute_kernel_config=None, parallel_config=None, dtype=None, **kw):
    sizes = self.chunk_sizes
    if not sizes or len(set(sizes)) == 1 or self.ccl_manager.topology != ttnn.Topology.Linear:
        return _orig_col_forward(
            self, x, compute_kernel_config=compute_kernel_config, parallel_config=parallel_config, dtype=dtype, **kw
        )
    assert parallel_config is None and not kw, "Linear attention gathers before Q/QKV"
    assert self.activation_fn is None and self.fused_activation_fn is None and not self.fuse_swiglu
    x = L.maybe_cast_activation(x, self.activation_dtype)
    dtype = L.resolve_output_dtype(dtype, x)
    weight = self.weight.data
    M, K, N = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
    out = ttnn.experimental.minimal_matmul(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=self.bias.data if self.bias is not None else None,
        config=L.get_matmul_config(M, K, N, L.get_matmul_core_grid(self.mesh_device), None),
        compute_kernel_config=compute_kernel_config or self.compute_config,
        dtype=dtype,
    )
    tp = tuple(self.mesh_device.shape)[self.mesh_axis]
    shape = list(out.shape)
    chunks, start = [], 0
    for w in sizes:
        w //= tp
        chunks.append(ttnn.slice(out, [0] * (len(shape) - 1) + [start], shape[:-1] + [start + w]))
        start += w
    assert start == shape[-1], (start, shape)
    ttnn.deallocate(out)
    return chunks


L.ColParallelLinear.forward = _col_forward


def _fold(block):
    n = 0
    for name in ATTNS:
        attn = getattr(block, name, None)
        if attn is None:
            continue
        attn.can_fold_gate_on_device = attn.apply_gated_attention and not attn.fuse_gate
        attn.fold_gate_on_device()
        n += attn._folded_proj is not None
    return n


def _gather(mesh, out, video_N_real, audio_N_real):
    dims = [None, None]
    dims[1], dims[0] = 2, 3
    comp = ttnn.ConcatMesh2dToTensor(mesh, dims=dims, mesh_shape=tuple(mesh.shape))
    v = ttnn.to_torch(out[0], mesh_composer=comp).squeeze(0)[:, :video_N_real]
    a = ttnn.to_torch(out[1], mesh_composer=comp).squeeze(0)[:, :audio_N_real]
    return v, a


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _arm(mesh, block, kw, name, video_N_real, audio_N_real):
    _ = block.forward(**kw)
    ttnn.synchronize_device(mesh)
    tracer = Tracer(block.forward, device=mesh, prep_run=False, clone_prep_inputs=False)
    _ = tracer(**kw, traced=True)
    ttnn.synchronize_device(mesh)
    laps = []
    for _ in range(N_LAPS):
        t0 = time.perf_counter()
        for _ in range(N_REPLAY):
            out = tracer(**kw, traced=True)
        ttnn.synchronize_device(mesh)
        laps.append((time.perf_counter() - t0) / N_REPLAY * 1e3)
    ms = statistics.median(laps)
    v, a = _gather(mesh, out, video_N_real, audio_N_real)
    tracer.release_trace()
    del tracer, out
    gc.collect()
    logger.info(f"T110_BLOCK arm={name} ms_per_block={ms:.3f} laps={[round(x, 3) for x in laps]} replays={N_REPLAY}")
    assert torch.isfinite(v.float()).all() and torch.isfinite(a.float()).all()
    return ms, v, a


def _build(mesh, F, H, W, adaln):
    os.environ["LTX_FUSE_NORM_ADALN"] = str(adaln)
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
    finally:
        os.environ["LTX_FUSE_NORM_ADALN"] = "0"
    assert block._fuse_norm_adaln == bool(adaln)
    kw["video_kv_logical_n"] = video_N_real
    return block, kw, video_N_real, audio_N_real


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_gate_adaln_ab(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    errors = []
    for shape in os.environ.get("T110_SHAPES", "S1,S2").split(","):
        F, H, W = SHAPES[shape]
        res = {}
        for adaln in (0, 1):
            try:
                block, kw, video_N_real, audio_N_real = _build(mesh, F, H, W, adaln)
                plain, folded = ("adaln", "both") if adaln else ("base", "gate")
                res[plain] = _arm(mesh, block, kw, plain, video_N_real, audio_N_real)
                n = _fold(block)
                logger.info(f"T110_FOLD {shape} adaln={adaln} folded={n}/{len(ATTNS)}")
                assert n == len(ATTNS), f"folded {n}/{len(ATTNS)}"
                res[folded] = _arm(mesh, block, kw, folded, video_N_real, audio_N_real)
            except Exception as e:
                logger.exception(f"T110_FAIL {shape} adaln={adaln}: {e}")
                errors.append(f"{shape} adaln={adaln}: {e}")
            finally:
                block = kw = None
                gc.collect()
        if "base" in res:
            ms0, v0, a0 = res["base"]
            for name in ("gate", "adaln", "both"):
                if name not in res:
                    continue
                ms, v, a = res[name]
                logger.info(
                    f"T110_AB {shape} arm={name} base_ms={ms0:.3f} ms={ms:.3f} delta_ms={ms - ms0:+.3f} "
                    f"({(ms / ms0 - 1) * 100:+.2f}%) pcc_v={_pcc(v0, v):.7f} pcc_a={_pcc(a0, a):.7f} "
                    f"bit_identical={torch.equal(v0, v) and torch.equal(a0, a)} "
                    f"maxabs_v={(v0.float() - v.float()).abs().max().item():.4g} "
                    f"maxabs_a={(a0.float() - a.float()).abs().max().item():.4g}"
                )
        del res
        gc.collect()
    assert not errors, errors
