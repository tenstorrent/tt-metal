# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Routed-expert device dtype study (dev-spec D-B, bead 8y7.9): bfp8 vs bfp4 on real layer 2 (KV + index source)
and real layer 0 (sliding-window only), same G1 bars for both dtypes.

* MoE: ``TtV41Moe`` on one production 5120-token chunk of the layer's real MoE input (``expert_dtype_reference``:
  the block oracle's ``ffn_in``, tiled); G1 bars routed >= 0.96, final >= 0.982 (plus gate / shared bars, which
  do not depend on the expert dtype); bit-identical repeat; warm device time; DRAM allocated per chip by the MoE.
* Block: ``TtV41Block`` alone on the block oracle (2048 tokens, one chunk, teacher-forced ``x_in`` / ``pre_in``,
  BF16 KV); G1 bar >= 0.98 real; bit-identical repeat; warm device time; DRAM per chip.

Results are logged as ``V41_DTYPE_MOE`` / ``V41_DTYPE_BLOCK`` JSON lines. Device MoE weight caches are shared with
``test_block_v41`` (same cache root per schedule); their file names carry the dtype, and a dtype-specific marker
records a completed build.
"""

import json
import time
from dataclasses import asdict

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import WEIGHT_CACHE, _pack, _pcc, _unpack
from models.demos.deepseek_v3_d_p.tests.v41.test_moe_v41 import BARS as MOE_BARS
from models.demos.deepseek_v3_d_p.tests.v41.test_moe_v41 import _metrics, _run
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

EXPERT_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
LAYERS = [2, 0]
BLOCK_BAR_REAL = 0.98
TIMED_ITERS = 3
MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]
# test_block_v41's config at its real-weight oracle (candidate count matched to the oracle)
BLOCK_CONFIG = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": 96})


def _dram_bytes_per_chip(mesh_device) -> int:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _weights(ckpt, layer: int, dtype: str, spec=None):
    """``test_block_v41``'s cache root for the layer's schedule (or ``spec``), the layer's weights (routed experts
    only when their ``dtype`` cache is incomplete) and the marker to touch after a completed build."""
    spec = spec or R.block_spec(layer)
    identity = orc._digest(asdict(spec.args), spec.seed, str(spec.checkpoint), orc._reference_digest(synthetic=False))
    root = WEIGHT_CACHE / f"real-{identity}"
    root.mkdir(parents=True, exist_ok=True)
    init_checker(root)
    marker = root / f"layer_{layer}.{dtype}.complete"
    # test_block_v41's marker records a completed build at TtV41Block's default dtype (bfp8)
    cached = marker.exists() or (dtype == "bfp8" and (root / f"layer_{layer}.complete").exists())
    start = time.perf_counter()
    w = load_layer_dense(ckpt, layer) if cached else load_layer(ckpt, layer)
    logger.info(f"layer {layer} weights loaded (routed experts: {not cached}) {time.perf_counter() - start:.1f}s")
    return root, w, marker


def _timed(fn, mesh_device, iters: int = TIMED_ITERS) -> float:
    """Mean warm device time (ms) of ``fn`` with a device synchronize around the loop."""
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - start) * 1e3 / iters


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("expert_dtype", list(EXPERT_DTYPES))
@pytest.mark.parametrize("layer", LAYERS, ids=[f"L{l}" for l in LAYERS])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_moe_expert_dtype(mesh_device, device_params, layer, expert_dtype):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    ref = R.moe_parts(layer)
    root, w, marker = _weights(ckpt, layer, expert_dtype)
    base = _dram_bytes_per_chip(mesh_device)
    start = time.perf_counter()
    moe = TtV41Moe(
        mesh_device,
        C,
        layer,
        w,
        R.CHUNK,
        routed_expert_weights_dtype=EXPERT_DTYPES[expert_dtype],
        weight_cache_path=root,
    )
    marker.touch()
    del w
    built = _dram_bytes_per_chip(mesh_device)
    logger.info(f"MoE L{layer} {expert_dtype} built {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    first = _run(moe, mesh_device, ref["x"])
    repeat = _run(moe, mesh_device, ref["x"])
    logger.info(f"MoE L{layer} {expert_dtype} two forwards {time.perf_counter() - start:.1f}s")
    for name in ("final", "routed", "shared", "logits", "weights", "indices"):
        assert torch.isfinite(first[name].float()).all(), f"non-finite {name}"
        assert torch.equal(first[name], repeat[name]), f"{name} differs between repeated runs"

    shape = tuple(mesh_device.shape)
    tt_x = ttnn.from_torch(
        ref["x"][None, None],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
    )
    ms = _timed(lambda: moe(tt_x), mesh_device)
    report = {
        "mesh": list(shape),
        "layer": layer,
        "block_type": C.block_type(layer).value,
        "expert_dtype": expert_dtype,
        "tokens": R.CHUNK,
        "metrics": _metrics(ref, first) | {"ms": round(ms, 2)},
        "dram_gb_per_chip_moe": round((built - base) / 1e9, 3),
        "dram_gb_per_chip_total": round(built / 1e9, 3),
    }
    logger.info(f"V41_DTYPE_MOE {json.dumps(report)}")
    failed = {k: v for k, v in report["metrics"].items() if k in MOE_BARS and v < MOE_BARS[k]}
    assert not failed, f"below G1 bars {MOE_BARS}: {failed}; {report}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("expert_dtype", list(EXPERT_DTYPES))
@pytest.mark.parametrize("layer", LAYERS, ids=[f"L{l}" for l in LAYERS])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_block_expert_dtype(mesh_device, device_params, layer, expert_dtype):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    spec = R.block_spec(layer)
    start = time.perf_counter()
    result = orc.oracle(spec, orc.random_tokens(spec))
    rec = result["blocks"][layer]
    logger.info(f"block oracle loaded {time.perf_counter() - start:.1f}s")
    seq, cfg, shape = R.ORACLE_SEQ, BLOCK_CONFIG, tuple(mesh_device.shape)
    tp, n = shape[1], cfg.HC_MULT
    root, w, marker = _weights(ckpt, layer, expert_dtype)
    base = _dram_bytes_per_chip(mesh_device)
    start = time.perf_counter()
    block = TtV41Block(
        mesh_device,
        cfg,
        layer,
        w,
        seq,
        routed_expert_weights_dtype=EXPERT_DTYPES[expert_dtype],
        weight_cache_path=root,
    )
    marker.touch()
    del w
    built = _dram_bytes_per_chip(mesh_device)
    logger.info(f"block L{layer} {expert_dtype} built {time.perf_counter() - start:.1f}s")

    def to_device(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims),
        )

    x = to_device(_pack(rec["x_in"].float(), tp), (2, 3))
    pre = to_device(rec["pre_in"].float()[None, None], (2, None))
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    def run():
        state = V41PrefillState(mesh_device, cfg, seq, seq, [layer], kv_format=MlaKvCacheFormat.BF16_RM)
        x_out, pre_out = block(x, pre, state, seq)
        return state, _unpack(down(x_out)[0, 0], n, tp), down(pre_out)[0, 0, :, :n]

    start = time.perf_counter()
    _, x_out, pre_out = run()
    _, x_rep, pre_rep = run()
    logger.info(f"block L{layer} {expert_dtype} two forwards {time.perf_counter() - start:.1f}s")
    assert torch.isfinite(x_out).all() and torch.isfinite(pre_out).all()
    deterministic = torch.equal(x_out, x_rep) and torch.equal(pre_out, pre_rep)
    state = V41PrefillState(mesh_device, cfg, seq, seq, [layer], kv_format=MlaKvCacheFormat.BF16_RM)
    ms = _timed(lambda: block(x, pre, state, seq), mesh_device)
    report = {
        "mesh": list(shape),
        "layer": layer,
        "block_type": C.block_type(layer).value,
        "expert_dtype": expert_dtype,
        "tokens": seq,
        "block_out": _pcc(rec["x_out"], x_out),
        "pre_mix": _pcc(rec["pre_out"], pre_out),
        "deterministic": deterministic,
        "ms": round(ms, 2),
        "dram_gb_per_chip_block": round((built - base) / 1e9, 3),
        "dram_gb_per_chip_total": round(built / 1e9, 3),
    }
    logger.info(f"V41_DTYPE_BLOCK {json.dumps(report)}")
    assert deterministic, report
    assert report["block_out"] >= BLOCK_BAR_REAL, report
