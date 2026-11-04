# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block.fused_tail on device.

* test_fused_tail_bit_exact: every saved reference case bit for bit, on three calls (a program-cache miss, then
  hits: the persistent counters carry over between calls), first the calls bit-identical to each other (a latent model's case, K3: W_up diag(g) on the bf8 grid, the same device
  weights from the reference's fp32 -> bf16 -> bf8 upload and this op's fp32 -> bf8 packing).
  Needs $MOE_BLOCK_REFS (the cases' root) and the case's mesh; skips otherwise.

    MOE_BLOCK_REFS=/data/mstaletovic/jobs/port/refs pytest tests/ttnn/unit_tests/operations/experimental/moe_block/fused_tail/test_fused_tail.py
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ref_harness
from ref_harness import assert_bit_exact, case_dir, from_mesh, list_cases, load_case, to_mesh  # noqa: E402

mb = ttnn.experimental.moe_block
OP = "fused_tail"


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
            assert_bit_exact(got, ref, ["output"])
            logger.info(f"fused_tail {case}: call {call} bit-exact")
    finally:
        res.release()
