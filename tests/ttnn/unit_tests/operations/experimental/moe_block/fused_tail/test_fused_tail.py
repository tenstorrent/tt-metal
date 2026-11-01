# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block.fused_tail on device.

* test_fused_tail_bit_exact: every saved reference case (dump_ref_fused_tail.py, run in the Python reference tree)
  bit for bit, on three calls (a program-cache miss, then hits: the persistent counters carry over between calls),
  first the calls bit-identical to each other; a latent model's (K3, the engine) output by PCC >= ENGINE_PCC.
  Needs $MOE_BLOCK_REFS (the cases' root) and the case's mesh; skips otherwise.
* test_fused_tail_host_time: the host cost of a call on a program-cache hit (N back-to-back untraced calls, the
  enqueue only), GLM-5.3 (TailA2A) and K3 (TailEngine) at chunk 5120 on the Galaxy 8x4. No reference data needed.

    MOE_BLOCK_REFS=/data/mstaletovic/jobs/port/refs pytest tests/ttnn/unit_tests/operations/experimental/moe_block/fused_tail/test_fused_tail.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ref_harness
from ref_harness import assert_bit_exact, assert_pcc, case_dir, from_mesh, list_cases, load_case, to_mesh  # noqa: E402

mb = ttnn.experimental.moe_block
OP = "fused_tail"
ENGINE_PCC = 0.9995  # (a latent model's tail, K3, against the reference: ~1 bf16 ulp off nearly everywhere)


def _fabric_params(fabric: str, payload: int) -> dict:
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload
    return {
        "fabric_config": getattr(ttnn.FabricConfig, fabric),
        "fabric_router_config": router,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }


def _resources(mesh_device, dims: dict, chunk_tokens: int, num_links: int = 2):
    md = mb.MoEModelDims(
        num_experts=dims["num_experts"],
        top_k=dims["top_k"],
        emb_dim=dims["emb_dim"],
        routed_expert_hidden_dim=dims["routed_expert_hidden_dim"],
        latent_dim=dims["latent_dim"],
        latent_norm=dims["latent_norm"],
        norm_eps=dims["norm_eps"],
    )
    return mb.MoEBlockResources(mesh_device, md, mb.MoEBlockConfig(chunk_tokens=chunk_tokens, num_links=num_links))


def _host_weights(path: Path):
    w = torch.load(path)
    return ttnn.from_torch(w["latent_up"].float()), ttnn.from_torch(w["latent_norm"].float())


def _case_params():
    """One param a saved case (its mesh and fabric from meta.json); a skipped one when there is none."""
    if "MOE_BLOCK_REFS" not in os.environ:
        return [pytest.param(None, None, None, marks=pytest.mark.skip(reason="MOE_BLOCK_REFS not set"), id="no-refs")]
    out = []
    for case in list_cases(OP):
        meta = json.loads((case_dir(OP, case) / "meta.json").read_text())
        cfg = meta["config"]
        params = _fabric_params(cfg["fabric_config"], cfg["fabric_payload_bytes"])
        out.append(pytest.param(tuple(meta["mesh_shape"]), params, case, id=case))
    return out or [pytest.param(None, None, None, marks=pytest.mark.skip(reason="no fused_tail cases"), id="no-cases")]


@pytest.mark.parametrize("mesh_device, device_params, case", _case_params(), indirect=["mesh_device", "device_params"])
def test_fused_tail_bit_exact(mesh_device, device_params, case):
    meta, ref = load_case(OP, case)
    R, C = meta["mesh_shape"]
    cfg = meta["config"]
    res = _resources(mesh_device, cfg["dims"], cfg["chunk_tokens"], cfg["num_links"])
    try:
        y = to_mesh(ref, "y_own", mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        se = to_mesh(ref, "shared_expert", mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        w = None
        if cfg["dims"]["latent_dim"]:
            up, gn = _host_weights(case_dir(OP, case) / meta["weights"])
            w = mb.prepare_fused_tail_weights(up, gn, resources=res, dtype=ttnn.bfloat8_b)
        # (three calls: the first builds the program, the others hit the cache - bit-identical run to run first,
        # independent of the reference, then each against it)
        calls = []
        for _ in range(3):
            out = mb.fused_tail(y, se, resources=res, w_up=w, math_fidelity=ttnn.MathFidelity.HiFi2)
            calls.append({ch: {"output": t} for ch, t in from_mesh(out, (R, C)).items()})
            ttnn.deallocate(out)
        for call in (1, 2):
            assert_bit_exact(calls[call], calls[0], ["output"])
        logger.info(f"fused_tail {case}: deterministic over 3 calls")
        for call, got in enumerate(calls):
            if cfg["dims"]["latent_dim"]:  # (the engine's output differs from the reference's in its last bits - TODO)
                assert_pcc(got, ref, ["output"], ENGINE_PCC)
                logger.info(f"fused_tail {case}: call {call} PCC >= {ENGINE_PCC}")
            else:
                assert_bit_exact(got, ref, ["output"])
                logger.info(f"fused_tail {case}: call {call} bit-exact")
    finally:
        res.release()


# (model, dims, fabric payload: the block's - the combine's row of the routed width)
HOST_TIME_MODELS = {
    "glm53": (
        dict(num_experts=256, top_k=8, emb_dim=6144, routed_expert_hidden_dim=2048, latent_dim=None),
        12352,
    ),
    "k3": (
        dict(num_experts=896, top_k=16, emb_dim=7168, routed_expert_hidden_dim=3072, latent_dim=3584),
        7232,
    ),
}


@pytest.mark.parametrize(
    "mesh_device, device_params, model",
    [
        pytest.param((8, 4), _fabric_params("FABRIC_2D_TORUS_XY", payload), model, id=model)
        for model, (_, payload) in HOST_TIME_MODELS.items()
    ],
    indirect=["mesh_device", "device_params"],
)
def test_fused_tail_host_time(mesh_device, device_params, model):
    dims, _ = HOST_TIME_MODELS[model]
    dims = dict(dims, latent_norm=True, norm_eps=1e-5)
    R, C = tuple(mesh_device.shape)
    chunk = 5120
    rows = chunk // R
    S, H = rows // C, dims["emb_dim"]
    res = _resources(mesh_device, dims, chunk)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    zeros = lambda shape, layout: ttnn.from_torch(  # noqa: E731
        torch.zeros(shape), dtype=ttnn.bfloat16, layout=layout, device=mesh_device, mesh_mapper=rep
    )
    try:
        if dims["latent_dim"]:
            Hr = dims["latent_dim"]
            y = zeros((1, 1, S, Hr), ttnn.ROW_MAJOR_LAYOUT)
            se = zeros((1, 1, rows, H // C), ttnn.TILE_LAYOUT)
            w = mb.prepare_fused_tail_weights(
                ttnn.from_torch(torch.zeros(H, Hr)), ttnn.from_torch(torch.ones(Hr)), resources=res
            )
        else:
            y = zeros((1, 1, S, H), ttnn.ROW_MAJOR_LAYOUT)
            se = zeros((1, 1, S, H), ttnn.TILE_LAYOUT)
            w = None
        call = lambda output=None: mb.fused_tail(y, se, resources=res, w_up=w, output=output)  # noqa: E731
        for _ in range(3):  # (the program built and cached)
            ttnn.deallocate(call())
        ttnn.synchronize_device(mesh_device)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(call())
        t_alloc = (time.perf_counter() - t0) / n
        ttnn.synchronize_device(mesh_device)
        out = call()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(n):
            call(out)
        t_pre = (time.perf_counter() - t0) / n
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"fused_tail {model} {R}x{C} chunk {chunk}: host {t_alloc * 1e6:.1f} us/call (output allocated a call), "
            f"{t_pre * 1e6:.1f} us/call (preallocated output), {n} untraced calls on a program-cache hit"
        )
    finally:
        res.release()
