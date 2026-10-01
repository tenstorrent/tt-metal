# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""#70 device check of LTX_FUSE_GATE_ON_DEVICE (t51 @80882b291c2), plus an LTX_FUSE_NORM_ADALN A/B.

Phase 1 (fold): AV block 0 with real weights, TP=4. Block U loads the unfused layout and folds its
gates on device; block F loads with LTX_FUSE_GATE=1. Every per-device fused weight/bias shard of U
must equal F's bit for bit. No forward: both fused paths need the Ring all-gather-matmul, and a
2x4 submesh has no wraparound link. Bit-identical weights mean the folded forward is the
LTX_FUSE_GATE forward.

Phase 2 (norm+adaln, T70_PHASE2=1): AV block 0 on the Linear 2x4 sp1/tp0 layout, traced, flag
off then on. Reports ms/block for each arm and the PCC between arms.
"""

import gc
import os
import time

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
import models.tt_dit.tests.models.ltx.test_transformer_ltx as T
from models.tt_dit.utils.tracing import Tracer
from models.tt_dit.models.transformers.ltx import attention_ltx as _A

_HERE = os.path.dirname(os.path.abspath(__file__))
# Fail at collection, before the mesh opens, if blx03's tree shadows t51's models/tt_dit.
assert _A.__file__.startswith(os.path.join(_HERE, "src")), f"t51 models/tt_dit not on path: {_A.__file__}"
_ATTNS = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
_BLOCK0 = {}


def _load_block0(num_layers, checkpoint_path):
    """Read only block 0 (safe_open), so the job does not load the whole 46 GB checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    if not _BLOCK0:
        pre = "model.diffusion_model.transformer_blocks.0."
        with safe_open(checkpoint_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith(pre):
                    _BLOCK0[k[len("model.diffusion_model.") :]] = f.get_tensor(k)
    return dict(_BLOCK0)


T._load_22b_state_dict = _load_block0
with open(os.path.join(_HERE, "block_setup_src.py")) as _f:
    exec(compile(_f.read(), "block_setup_src.py", "exec"), T.__dict__)


def _shards(t, mesh):
    # Every device's shard stacked on dim 0, replicas included, so the check covers all 8 chips.
    return [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))]


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _phase1_fold(mesh):
    ckpt = T._resolve_checkpoint_22b("fast")
    block_sd = {k[len("transformer_blocks.0.") :]: v for k, v in _load_block0(1, ckpt).items()}
    ccl = T._make_ccl_manager(mesh, 2, ttnn.Topology.Ring)
    pc = T._make_parallel_config(mesh, 0, 1)

    os.environ["LTX_FUSE_GATE"] = "0"
    u = T._make_tt_block(mesh_device=mesh, ccl_manager=ccl, parallel_config=pc, is_fsdp=False, has_audio=True)
    u.load_torch_state_dict(block_sd, strict=False)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for name in _ATTNS:
        getattr(u, name).fold_gate_on_device()
    ttnn.synchronize_device(mesh)
    fold_ms = (time.perf_counter() - t0) * 1e3

    os.environ["LTX_FUSE_GATE"] = "1"
    f = T._make_tt_block(mesh_device=mesh, ccl_manager=ccl, parallel_config=pc, is_fsdp=False, has_audio=True)
    f.load_torch_state_dict(block_sd, strict=False)
    os.environ["LTX_FUSE_GATE"] = "0"

    bad, folded = [], 0
    for name in _ATTNS:
        ua, fa = getattr(u, name), getattr(f, name)
        if ua._folded_proj is None:
            logger.info(f"T70 {name}: not folded (can_fold={ua.can_fold_gate_on_device} fuse_gate_ref={fa.fuse_gate})")
            if fa.fuse_gate:
                bad.append(f"{name}: LTX_FUSE_GATE fuses it but the on-device fold skipped it")
            continue
        folded += 1
        ref = fa.to_qkv if fa.is_self else fa.to_q
        src = ua.to_qkv if ua.is_self else ua.to_q
        if src.weight._data is not None or ua.to_gate_logits.weight._data is not None:
            bad.append(f"{name}: unfused weights not released")
        for p in ("weight", "bias"):
            got, want = getattr(ua._folded_proj, p), getattr(ref, p)
            if (got is None) != (want is None):
                bad.append(f"{name}.{p}: presence differs")
                continue
            if got is None:
                continue
            gs, ws = _shards(got.data, mesh), _shards(want.data, mesh)
            same = len(gs) == len(ws) and all(g.shape == w.shape and torch.equal(g, w) for g, w in zip(gs, ws))
            logger.info(f"T70 {name}.{p}: shard {tuple(gs[0].shape)} x{len(gs)} bit_identical={same}")
            if not same:
                diff = max((g.float() - w.float()).abs().max().item() for g, w in zip(gs, ws) if g.shape == w.shape)
                bad.append(f"{name}.{p}: not bit-identical (max abs diff {diff})")
    logger.info(f"T70_FOLD folded={folded}/{len(_ATTNS)} fold_ms={fold_ms:.1f} mismatches={len(bad)}")
    for b in bad:
        logger.error(f"T70_FOLD_BAD {b}")
    del u, f
    gc.collect()
    return folded, bad


def _phase2_arm(mesh, fuse, F, H, W, n_replay):
    os.environ["LTX_FUSE_NORM_ADALN"] = str(fuse)
    block, kw, video_N_real, audio_N_real = T._build_block_trace_setup(
        mesh_device=mesh,
        sp_axis=1,
        tp_axis=0,
        num_links=2,
        topology=ttnn.Topology.Linear,
        is_fsdp=False,
        F=F,
        H=H,
        W=W,
        has_audio=True,
    )
    os.environ["LTX_FUSE_NORM_ADALN"] = "0"
    assert block._fuse_norm_adaln == bool(fuse)
    _ = block(**kw)
    ttnn.synchronize_device(mesh)
    tracer = Tracer(block.forward, device=mesh, prep_run=False, clone_prep_inputs=False)
    _ = tracer(**kw, traced=True)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n_replay):
        out = tracer(**kw, traced=True)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) / n_replay * 1e3
    dims = [None, None]
    dims[1], dims[0] = 2, 3
    comp = ttnn.ConcatMesh2dToTensor(mesh, dims=dims, mesh_shape=tuple(mesh.shape))
    v = ttnn.to_torch(out[0], mesh_composer=comp).squeeze(0)[:, :video_N_real].float()
    a = ttnn.to_torch(out[1], mesh_composer=comp).squeeze(0)[:, :audio_N_real].float()
    tracer.release_trace()
    del tracer, block, kw, out
    gc.collect()
    logger.info(f"T70_ADALN fuse={fuse} F,H,W={F},{H},{W} ms_per_block={ms:.2f} replays={n_replay}")
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
    return ms, v, a


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_fold_gate_device_check(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    folded, bad = _phase1_fold(mesh)

    if os.environ.get("T70_PHASE2", "1") == "1":
        # F=10 at 34x60 over sp=4 puts 5100 video tokens on each chip, near stage 2's 4845 on the 4x8.
        F, H, W, n = 10, 34, 60, 5
        ms0, v0, a0 = _phase2_arm(mesh, 0, F, H, W, n)
        ms1, v1, a1 = _phase2_arm(mesh, 1, F, H, W, n)
        logger.info(
            f"T70_ADALN_AB off_ms={ms0:.2f} on_ms={ms1:.2f} delta_ms={ms1 - ms0:+.2f} ({(ms1 / ms0 - 1) * 100:+.1f}%) "
            f"pcc_video={_pcc(v0, v1):.6f} pcc_audio={_pcc(a0, a1):.6f} "
            f"maxabs_video={(v0 - v1).abs().max().item():.4g}"
        )

    assert folded > 0, "no attention was folded"
    assert not bad, f"fold mismatches: {bad}"
