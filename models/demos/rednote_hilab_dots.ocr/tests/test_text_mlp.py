# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: TtTextMLP vs the golden reference tensor on a 1x4 mesh.

Golden input/output produced by reference/test_functional.py (PCC 1.0 vs the
official HF Qwen2MLP with the real layers.0.mlp weights). Weights are
re-loaded from the HF checkpoint here, as the golden stores only activations.
The block is TP-sharded per the parallelism plan (gate/up column-parallel,
down row-parallel + all-reduce). The all-reduced output is replicated, so we
compare ONE device's copy.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
GOLDEN = MODEL_DIR / "reference" / "golden" / "text_mlp.pt"
REPO = "rednote-hilab/dots.ocr"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_mlp", MODEL_DIR / "tt" / "text_mlp.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextMLP = _mod.TtTextMLP


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _load_weights(prefix, keys):
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in keys:
        full = f"{prefix}.{k}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_text_mlp_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    x, ref_out = golden["input"], golden["output"]
    _, seq, dim = x.shape

    sd = _load_weights("model.layers.0.mlp", ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])
    block = TtTextMLP(mesh_device, sd)

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq, dim),
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
    side = MODEL_DIR / "tt" / "text_mlp.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # All-reduced (replicated) output: compare a single device's copy.
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float().reshape(seq, dim)
    assert out.shape == ref_out.reshape(seq, dim).shape, f"{out.shape} != {ref_out.shape}"

    pcc = _pcc(ref_out.reshape(seq, dim), out)
    print(f"PCC(text_mlp) = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
