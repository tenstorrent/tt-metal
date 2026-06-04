# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4-vs-TP1 unit test for the dots.ocr vision SwiGLU MLP.

This is the whole MLP issued once per vision block in ``dots_ocr_vision.py``
(``TTNNDotsVisionMLP.forward``)::

    gate = fc1(x)                         # [M, I]      MatmulDeviceOperation
    up   = fc3(x)                         # [M, I]      MatmulDeviceOperation
    h    = silu(gate) * up                # [M, I]      BinaryNgDeviceOperation (SILU fused)
    out  = fc2(h)                         # [M, H]      MatmulDeviceOperation

The production module runs this *replicated* (every chip holds the full weights
and computes the full MLP). This test contrasts that TP1 baseline with the
standard tensor-parallel MLP layout on a real TP{TP_DEGREE} mesh and checks the
final output PCC against the torch reference.

dots.ocr vision MLP shape (input ``[1, 1, M, H]``), from Wormhole production profiler
(``MatmulDeviceOperation`` M x K x N)::

    M (seq, bucket)  = 11264     fc1/fc3: 11264 x 1536 x 4224  (BFP8 gate)
    H (hidden)       = 1536      fc3:     11264 x 1536 x 4224  (BFP4 up)
    I (intermediate) = 4224      mul:     silu(gate) * up      (BFP8)
                                 fc2:     11264 x 4224 x 1536  (BFP8 x BFP4 => BFP4 on WH)

How TP4 maps onto a SwiGLU MLP
------------------------------
The canonical Megatron split, with exactly ONE collective (the down-proj reduce):

    fc1 / fc3  (column-parallel) : shard the intermediate (output) dim -> each chip
                                   owns ``I/TP`` columns. No collective; x is replicated.
    silu*up    (elementwise)     : runs per-chip on the chip's ``I/TP`` slice.
    fc2        (row-parallel)    : shard the intermediate (contraction K) dim -> each
                                   chip produces a PARTIAL ``[M, H]``; an all-reduce
                                   (sum) across chips yields the full output.

``I/TP = 1056`` (33 tiles) and ``H/TP = 384`` (12 tiles) are both tile-aligned, so
the column split, the K split, and the reduce_scatter on H all stay tile-clean.

The all-reduce is decomposed into ``reduce_scatter(dim=3) + all_gather(dim=3)`` —
the trace-compatible CCL path the production decoder linears use (see
``linear.py``); it needs fabric, so this test enables ``FABRIC_1D_RING`` for TP>1.

TP1 replicates the full weights to every chip and runs the full ``I``-wide MLP, so
the mesh wall-time equals one device's full-MLP time (true TP1). Both paths share a
single submesh. Each path's output is PCC-checked against the float torch reference
(``fc2(silu(fc1(x)) * fc3(x))``); after the TP4 all-reduce every chip holds the same
full output, so device-0 readback is the full result.

Run::

    # Wormhole 4-chip (profiler shows full M=11264 per matmul)
    MESH_DEVICE=N150x4 pytest models/experimental/tt_symbiote/tests/test_vision_mlp_tp_perf.py -s

    # Blackhole 4-chip (per-chip M is M/TP in the profiler)
    MESH_DEVICE=P150x4 pytest models/experimental/tt_symbiote/tests/test_vision_mlp_tp_perf.py -s

Tunables (env):
    VISION_MLP_M          seq / M dim. Default 11264 (the vision bucket).
    VISION_MLP_TP_DEGREE  tensor-parallel degree. Default 4.
    VISION_MLP_WARMUP     warmup iterations. Default 3.
    VISION_MLP_ITERS      measured iterations. Default 10.
    VISION_MLP_PCC        PCC gate. Default 0.97 (bf8 weights + fused SILU are lossy).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_n_dev, skip_for_n_or_less_dev


# ---------------------------------------------------------------------------
# Canonical MLP dims — Wormhole production profiler (one vision block, full M).
# ---------------------------------------------------------------------------

MLP_M = 11264
MLP_H = 1536
MLP_I = 4224


# ---------------------------------------------------------------------------
# Mesh / device wiring — 1x4 line mesh on WH (N150x4) or BH (P150x4).
# ---------------------------------------------------------------------------

MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def _default_mesh_device_name():
    return "P150x4" if is_blackhole() else "N150x4"


def _resolve_mesh_device_shape():
    mesh_device = os.environ.get("MESH_DEVICE", _default_mesh_device_name())
    if mesh_device in MESH_DEVICE_MAP:
        return MESH_DEVICE_MAP[mesh_device]
    num = len(ttnn.get_device_ids())
    return (1, num)


def _ccl_settings():
    """CCL along the TP axis on a 1x4 mesh."""
    # WH production / symbiote linear.py use Ring; Linear also works on WH meshes.
    cluster_axis = 1
    topology = ttnn.Topology.Ring if is_blackhole() else ttnn.Topology.Linear
    return cluster_axis, topology


DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


DOTS_OCR_LOCAL_PATH = _resolve_model_path()


def _mlp_device_params():
    # The TP4 down-proj reduce uses reduce_scatter + all_gather, which need fabric.
    sh = _resolve_mesh_device_shape()
    num_devices = sh[0] * sh[1] if isinstance(sh, tuple) else int(sh)
    params = {
        "trace_region_size": 0,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if num_devices > 1 else ttnn.FabricConfig.DISABLED,
    }
    if is_blackhole():
        params["dispatch_core_axis"] = ttnn.DispatchCoreAxis.COL
    return params


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


M_DIM = int(os.environ.get("VISION_MLP_M", str(MLP_M)))
TP_DEGREE = int(os.environ.get("VISION_MLP_TP_DEGREE", "4"))
WARMUP = int(os.environ.get("VISION_MLP_WARMUP", "0"))
ITERS = int(os.environ.get("VISION_MLP_ITERS", "1"))
PCC = float(os.environ.get("VISION_MLP_PCC", "0.97"))

# Match the production module's dtype choices (preprocess_weights_impl uses bf8
# weights; forward emits gate bf8 / up bf4 / mul bf8). The down-proj *output* is
# bf8 here (not the module's bf4) so the cross-device all-reduce sum stays clean.
_W_DT = ttnn.bfloat8_b
_GATE_DT = ttnn.bfloat8_b
_UP_DT = ttnn.bfloat4_b
_MUL_DT = ttnn.bfloat8_b
_DOWN_DT = ttnn.bfloat8_b


# ---------------------------------------------------------------------------
# Weight prep. ttnn.linear computes ``x @ W`` with W laid out [K, N], so the
# torch Linear weight [out, in] is transposed to [in, out] before upload.
#   fc1/fc3 : [H, I]  -> column-parallel shards N=I on dim -1
#   fc2     : [I, H]  -> row-parallel    shards K=I on dim -2
# ---------------------------------------------------------------------------


def _to_w(w_torch):
    """torch Linear weight [out, in] -> tt-layout [in, out] (transposed)."""
    return w_torch.t().contiguous()


def _to_bias(b_torch):
    """torch Linear bias [out] -> [1, out] for ttnn.linear's bias arg."""
    return b_torch.reshape(1, -1).contiguous()


def _upload(t, mesh, mapper, dtype=_W_DT, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _mlp_block(x, w1, b1, w3, b3, w2, b2, *, reduce, mesh):
    """The dots.ocr vision MLP: ``fc2(silu(fc1(x)) * fc3(x))``.

    ``reduce=True`` (TP path) all-reduces the row-parallel down-proj partial via
    reduce_scatter + all_gather and adds the (replicated) fc2 bias afterwards.
    """
    cluster_axis, ccl_topology = _ccl_settings()
    mem = ttnn.DRAM_MEMORY_CONFIG
    gate = ttnn.linear(x, w1, bias=b1, dtype=_GATE_DT, memory_config=mem)
    up = ttnn.linear(x, w3, bias=b3, dtype=_UP_DT, memory_config=mem)
    h = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        fast_and_approximate_mode=True,
        dtype=_MUL_DT,
        memory_config=mem,
    )
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    # In the TP path fc2 is row-parallel -> its bias is added once, after the
    # reduce; pass bias=None here and fold it in below.
    out = ttnn.linear(h, w2, bias=None if reduce else b2, dtype=_DOWN_DT, memory_config=mem)
    ttnn.deallocate(h)

    if reduce:
        out = ttnn.reduce_scatter(
            out,
            dim=3,
            num_links=1,
            cluster_axis=cluster_axis,
            memory_config=mem,
            topology=ccl_topology,
        )
        out = ttnn.all_gather(
            out,
            dim=3,
            num_links=1,
            cluster_axis=cluster_axis,
            memory_config=mem,
            topology=ccl_topology,
        )
        if b2 is not None:
            out = ttnn.add(out, b2, dtype=_DOWN_DT, memory_config=mem)
    return out


def _time_block(device, fn, *, warmup, iters):
    """Mean per-iteration device latency (s); enqueue iters back-to-back then one sync."""
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    for o in outs:
        ttnn.deallocate(o)
    return elapsed / iters


# ---------------------------------------------------------------------------
# The benchmark.
# ---------------------------------------------------------------------------


@skip_for_n_or_less_dev(3)
@skip_for_n_dev(8, reason_str="P150x8 requires all devices active for fabric; use a 4-chip box")
@pytest.mark.parametrize("device_params", [_mlp_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize("m_dim", [M_DIM], ids=[f"M{M_DIM}"])
def test_vision_mlp_tp4_vs_tp1_perf(mesh_device, m_dim):
    """Run the dots.ocr vision SwiGLU MLP in TP1 (replicated, full-width) vs
    TP{TP_DEGREE} (column-parallel fc1/fc3 + row-parallel fc2 + all-reduce), at the
    production ``M=11264`` vision bucket, and PCC-check both against torch."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from tests.ttnn.utils_for_testing import comp_pcc

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    mesh_devices = int(mesh_device.get_num_devices())
    if mesh_devices < TP_DEGREE:
        pytest.skip(f"TP{TP_DEGREE} needs >= {TP_DEGREE} devices, mesh has {mesh_devices}")

    # ----------------------------------------------------------------------
    # Pull the real dots.ocr vision MLP weights + dims (and a float reference).
    # ----------------------------------------------------------------------
    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if hasattr(vision_config, attr):
                setattr(vision_config, attr, 1)
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "dots.ocr vision tower should expose blocks/layers"
    hf_mlp = blocks[0].mlp  # DotsSwiGLUFFN

    H = int(hf_mlp.fc1.in_features)
    I = int(hf_mlp.fc1.out_features)
    assert m_dim == MLP_M, f"M={m_dim} != production bucket {MLP_M}"
    assert H == MLP_H, f"H={H} != Wormhole profiler hidden {MLP_H}"
    assert I == MLP_I, f"I={I} != Wormhole profiler intermediate {MLP_I}"
    assert I % TP_DEGREE == 0, f"intermediate {I} must split TP={TP_DEGREE} ways"
    assert (I // TP_DEGREE) % 32 == 0, f"I/TP={I // TP_DEGREE} must be tile-aligned (32)"
    assert (H // TP_DEGREE) % 32 == 0, f"H/TP={H // TP_DEGREE} must be tile-aligned for the reduce_scatter"

    # Float torch reference off the same weights.
    x_torch = torch.randn(1, 1, m_dim, H, dtype=torch.bfloat16)
    ref = hf_mlp.float()(x_torch.float()).reshape(1, 1, m_dim, H)

    # Transposed weights / biases (torch host tensors; sharded at upload time).
    w1_t, w3_t, w2_t = _to_w(hf_mlp.fc1.weight.data), _to_w(hf_mlp.fc3.weight.data), _to_w(hf_mlp.fc2.weight.data)
    b1_t = _to_bias(hf_mlp.fc1.bias.data) if hf_mlp.fc1.bias is not None else None
    b3_t = _to_bias(hf_mlp.fc3.bias.data) if hf_mlp.fc3.bias is not None else None
    b2_t = _to_bias(hf_mlp.fc2.bias.data) if hf_mlp.fc2.bias is not None else None
    del hf_model

    # One submesh of width TP_DEGREE serves BOTH paths (an overlapping [1,1] +
    # [1,TP] pair would alias device 0 and corrupt its data).
    tp_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, TP_DEGREE))
    rep = ttnn.ReplicateTensorToMesh(tp_mesh)
    shard_n = ttnn.ShardTensorToMesh(tp_mesh, dim=-1)  # column-parallel: split N=I
    shard_k = ttnn.ShardTensorToMesh(tp_mesh, dim=-2)  # row-parallel: split K=I

    x_dev = _upload(x_torch, tp_mesh, rep, dtype=ttnn.bfloat8_b)

    def _readback_full(dev_t):
        return ttnn.to_torch(ttnn.get_device_tensors(dev_t)[0]).float().reshape(ref.shape)

    # ---- TP1: full weights replicated; each chip runs the full I-wide MLP. ----
    w1_1, w3_1, w2_1 = _upload(w1_t, tp_mesh, rep), _upload(w3_t, tp_mesh, rep), _upload(w2_t, tp_mesh, rep)
    b1_1 = _upload(b1_t, tp_mesh, rep) if b1_t is not None else None
    b3_1 = _upload(b3_t, tp_mesh, rep) if b3_t is not None else None
    b2_1 = _upload(b2_t, tp_mesh, rep) if b2_t is not None else None

    out1 = _mlp_block(x_dev, w1_1, b1_1, w3_1, b3_1, w2_1, b2_1, reduce=False, mesh=tp_mesh)
    o1 = _readback_full(out1)
    ttnn.deallocate(out1)
    ok1, pcc1 = comp_pcc(ref, o1, PCC)
    assert ok1, f"TP1 MLP PCC {pcc1} < {PCC}"
    tp1_lat = _time_block(
        tp_mesh,
        lambda: _mlp_block(x_dev, w1_1, b1_1, w3_1, b3_1, w2_1, b2_1, reduce=False, mesh=tp_mesh),
        warmup=WARMUP,
        iters=ITERS,
    )

    # ---- TP{TP_DEGREE}: column-parallel fc1/fc3, row-parallel fc2, all-reduce. ----
    w1_n, w3_n, w2_n = _upload(w1_t, tp_mesh, shard_n), _upload(w3_t, tp_mesh, shard_n), _upload(w2_t, tp_mesh, shard_k)
    b1_n = _upload(b1_t, tp_mesh, shard_n) if b1_t is not None else None
    b3_n = _upload(b3_t, tp_mesh, shard_n) if b3_t is not None else None
    # fc2 bias is replicated (added once, post-reduce).
    b2_n = _upload(b2_t, tp_mesh, rep) if b2_t is not None else None

    outN = _mlp_block(x_dev, w1_n, b1_n, w3_n, b3_n, w2_n, b2_n, reduce=True, mesh=tp_mesh)
    # After the all-reduce every chip holds the same full [1,1,M,H] output.
    shards = ttnn.get_device_tensors(outN)
    assert len(shards) == TP_DEGREE, f"expected {TP_DEGREE} shards, got {len(shards)}"
    pcc_min = 1.0
    for d, sht in enumerate(shards):
        od = ttnn.to_torch(sht).float().reshape(ref.shape)
        okd, pccd = comp_pcc(ref, od, PCC)
        assert okd, f"TP{TP_DEGREE} dev{d} MLP PCC {pccd} < {PCC}"
        pcc_min = min(pcc_min, float(pccd))
    ttnn.deallocate(outN)
    tp_lat = _time_block(
        tp_mesh,
        lambda: _mlp_block(x_dev, w1_n, b1_n, w3_n, b3_n, w2_n, b2_n, reduce=True, mesh=tp_mesh),
        warmup=WARMUP,
        iters=ITERS,
    )

    # ----------------------------------------------------------------------
    # Report.
    # ----------------------------------------------------------------------
    width = 100
    print("\n" + "=" * width)
    print(f"dots.ocr vision SwiGLU MLP perf  —  TP1 vs TP{TP_DEGREE}  ({WARMUP} warmup, {ITERS} iters)")
    print("=" * width)
    print(f"  shape   x[1,1,{m_dim},{H}]  fc1/fc3->[{H},{I}]  fc2->[{I},{H}]")
    print(f"  TP1 matmuls (profiler M x K x N, full I={I}):")
    print(f"    fc1 / fc3 : {m_dim} x {H} x {I}")
    print(f"    fc2       : {m_dim} x {I} x {H}")
    m_per_chip = m_dim // TP_DEGREE
    print(f"  TP{TP_DEGREE} matmuls (shard I and K; profiler M is often M/TP={m_per_chip} per chip):")
    print(f"    fc1 / fc3 : {m_per_chip} x {H} x {I // TP_DEGREE}")
    print(f"    fc2       : {m_per_chip} x {I // TP_DEGREE} x {H}")
    print(
        f"  TP{TP_DEGREE} layout  col-parallel fc1/fc3 (I/{TP_DEGREE}={I // TP_DEGREE})  +  "
        f"row-parallel fc2 (K/{TP_DEGREE})  +  all-reduce(H)"
    )
    print("-" * width)
    print(f"  {'path':<10}{'us':>12}{'speedup':>12}{'PCC':>12}")
    print(f"  {'TP1':<10}{tp1_lat * 1e6:>12.2f}{'1.00x':>12}{pcc1:>12.4f}")
    speedup = tp1_lat / tp_lat if tp_lat > 0 else float("nan")
    print(f"  {'TP'+str(TP_DEGREE):<10}{tp_lat * 1e6:>12.2f}{speedup:>11.2f}x{pcc_min:>12.4f}")
    print("-" * width)
    print("  fc1/fc3 column-parallel (shard intermediate) -> per-chip silu*up -> fc2 row-parallel")
    print("  (shard K) -> reduce_scatter+all_gather all-reduce. Only one collective per MLP.")
    print("=" * width)
