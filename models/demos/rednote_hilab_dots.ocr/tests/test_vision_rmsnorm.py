# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: TtVisionRMSNorm vs the golden reference tensor on a 1x4 mesh.

Golden input/output/weight produced by reference/test_functional.py (PCC 1.0
vs the official HF dots.ocr RMSNorm with the real blocks.0.norm1 weight).
Output is replicated across the mesh, so we compare ONE device's copy.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
GOLDEN = MODEL_DIR / "reference" / "golden" / "vision_rmsnorm.pt"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_tt_vision_rmsnorm", MODEL_DIR / "tt" / "vision_rmsnorm.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtVisionRMSNorm = _mod.TtVisionRMSNorm


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_rmsnorm_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    x, ref_out, weight, eps = golden["input"], golden["output"], golden["weight"], golden["eps"]
    block = TtVisionRMSNorm(mesh_device, {"weight": weight}, eps=eps)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Fast runtime mode skips per-op sub-graph capture; disable it so the
    # calltrace actually records the ttnn ops executed.
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture()
        out_tt = block.forward(x_tt)
        captured = ttnn.graph.end_graph_capture()

    # Persist the traced ttnn op list for the orchestrator's no-shortcuts guard.
    calltrace = list(ttnn.graph.extract_calltrace(captured))
    traced = sorted({op.replace("::", ".") for op in calltrace if op.replace("::", ".").startswith("ttnn.")})
    side = MODEL_DIR / "tt" / "vision_rmsnorm.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # Replicated output: compare a single device's copy.
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    assert out.shape == ref_out.shape, f"{out.shape} != {ref_out.shape}"

    pcc = _pcc(ref_out, out)
    print(f"PCC(vision_rmsnorm) = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
