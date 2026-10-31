# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block.routed_experts_combine on device, bit-exact against the reference cases
(dump_ref_routed_experts_combine.py; combine_snap 2^-14). Needs the case's mesh and fabric; skips without them, without
reference data ($MOE_BLOCK_REFS) or before the flat routed expert's builder is integrated into the op (FLAT_INTEGRATED).
"""

import os
import sys
from pathlib import Path

import pytest

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ref_harness import REFS_ROOT, assert_bit_exact, from_mesh, list_cases, load_case, to_mesh  # noqa: E402

OP = "routed_experts_combine"
mb = ttnn.experimental.moe_block

# SEAM(flat): the op takes the flat expert's tensors (xg, counts, regions, token_index, w_gu, w_d, w_rd, gidx, done)
# once its builder is in (flat_routed_expert/device/flat_program.hpp); until then it has no y writers to run.
FLAT_INTEGRATED = False

CASES = list_cases(OP) if os.environ.get("MOE_BLOCK_REFS") or REFS_ROOT.exists() else []
U32, BF16 = ttnn.uint32, ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT


def _resources(mesh_device, meta):
    cfg = meta["config"]
    dims = mb.MoEModelDims(num_experts=cfg["E"], top_k=cfg["K"], emb_dim=cfg["Hr"], routed_expert_hidden_dim=cfg["I"])
    conf = mb.MoEBlockConfig(chunk_tokens=cfg["chunk_tokens"], combine_snap=cfg["combine_snap"])
    return mb.MoEBlockResources.create(mesh_device, dims, conf)


@pytest.mark.skipif(not FLAT_INTEGRATED, reason="SEAM(flat): the flat routed expert's builder is not integrated yet")
@pytest.mark.skipif(not CASES, reason="no reference cases (dump_ref_routed_experts_combine.py, $MOE_BLOCK_REFS)")
@pytest.mark.parametrize("case", CASES or ["none"])
@pytest.mark.parametrize("device_params", [dict(fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_routed_experts_combine(mesh_device, case):
    meta, ref = load_case(OP, case)
    if list(mesh_device.shape) != meta["mesh_shape"]:
        pytest.skip(f"case {case} needs a {meta['mesh_shape']} mesh")
    cfg = meta["config"]
    res = _resources(mesh_device, meta)
    try:
        plan = to_mesh(ref, "plan", mesh_device, dtype=U32, layout=RM)
        table = to_mesh(ref, "table", mesh_device, dtype=U32, layout=RM)
        combine = mb.CombineConfig()
        combine.n_acc = cfg["n_acc"]
        combine.mode = mb.CombineMode.AccOnPorts if cfg["acc_on_ports"] else mb.CombineMode.PortsApart
        combine.arena_bytes = cfg["arena_bytes"]
        cells = [ttnn.CoreCoord(x, y) for x, y in cfg["cells"]]
        writers = [ttnn.CoreCoord(x, y) for x, y in cfg["writers"]]
        out = mb.routed_experts_combine(
            plan, table, res, cells=cells, writers=writers, writer_words_offset=cfg["arena_bytes"], combine=combine
        )
        got = from_mesh(out, tuple(meta["mesh_shape"]))
        assert_bit_exact({ch: {"y_own": t} for ch, t in got.items()}, ref, ["y_own"])
    finally:
        res.release()
