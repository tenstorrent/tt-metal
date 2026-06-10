# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: TtTextRMSNorm vs the golden reference tensor on a 1x4 mesh.

Golden input/output/weight produced by reference/test_functional.py (PCC 1.0
vs the official HF Qwen2RMSNorm with the real layers.0 input_layernorm
weight). Both paths are verified:

- replicated fused ttnn.rms_norm (output replicated -> compare ONE device);
- distributed pre/all_gather/post over a hidden-sharded input (output stays
  hidden-sharded -> concat shards on the hidden dim).
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
GOLDEN = MODEL_DIR / "reference" / "golden" / "text_rmsnorm.pt"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_rmsnorm", MODEL_DIR / "tt" / "text_rmsnorm.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextRMSNorm = _mod.TtTextRMSNorm


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_text_rmsnorm_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    x, ref_out, weight, eps = golden["input"], golden["output"], golden["weight"], golden["eps"]
    # [1, seq, dim] -> [1, 1, seq, dim] (distributed pre/post ops want 4D).
    x4, ref4 = x.unsqueeze(1), ref_out.unsqueeze(1)
    block = TtTextRMSNorm(mesh_device, {"weight": weight}, eps=eps)

    x_repl = ttnn.from_torch(
        x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_shard = ttnn.from_torch(
        x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    # Fast runtime mode skips per-op sub-graph capture; disable it so the
    # calltrace actually records the ttnn ops executed.
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture()
        out_repl = block.forward(x_repl)
        out_dist = block.forward_distributed(x_shard)
        captured = ttnn.graph.end_graph_capture()

    # Persist the traced ttnn op list for the orchestrator's no-shortcuts guard.
    calltrace = list(ttnn.graph.extract_calltrace(captured))
    traced = sorted({op.replace("::", ".") for op in calltrace if op.replace("::", ".").startswith("ttnn.")})
    side = MODEL_DIR / "tt" / "text_rmsnorm.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # Replicated output: compare a single device's copy.
    out_r = ttnn.to_torch(ttnn.get_device_tensors(out_repl)[0]).float()
    assert out_r.shape == ref4.shape, f"{out_r.shape} != {ref4.shape}"
    pcc_repl = _pcc(ref4, out_r)
    print(f"PCC(text_rmsnorm, replicated fused) = {pcc_repl:.6f}")

    # Distributed output: still hidden-sharded -> concat shards on dim 3.
    out_d = ttnn.to_torch(out_dist, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
    assert out_d.shape == ref4.shape, f"{out_d.shape} != {ref4.shape}"
    pcc_dist = _pcc(ref4, out_d)
    print(f"PCC(text_rmsnorm, distributed) = {pcc_dist:.6f}")

    assert pcc_repl > 0.99, f"replicated PCC {pcc_repl:.6f} < 0.99"
    assert pcc_dist > 0.99, f"distributed PCC {pcc_dist:.6f} < 0.99"
