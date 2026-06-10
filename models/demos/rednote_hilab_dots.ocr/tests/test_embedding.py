# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: TtEmbedding vs the golden reference tensor on a 1x4 mesh.

Golden input ids / output produced by reference/test_functional.py (PCC 1.0 vs
torch.nn.Embedding with the real model.embed_tokens.weight). The golden does
not store the 151936x1536 table, so it is loaded from the checkpoint here.
Output is replicated across the mesh after all_gather, so we compare ONE
device's copy.
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
GOLDEN = MODEL_DIR / "reference" / "golden" / "embedding.pt"
REPO = "rednote-hilab/dots.ocr"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_tt_embedding", MODEL_DIR / "tt" / "embedding.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtEmbedding = _mod.TtEmbedding


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _load_embed_weight():
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(snap / idx[key], framework="pt") as f:
        return f.get_tensor(key).float()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_embedding_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    ids, ref_out = golden["input"], golden["output"]  # [1, 128] int64, [1, 128, 1536] fp32
    weight = _load_embed_weight()

    block = TtEmbedding(mesh_device, {"weight": weight})

    # tt_transformers convention: tokens flattened to [1, 1, 1, batch*seq],
    # uint32 ROW_MAJOR, replicated across the mesh.
    ids_tt = ttnn.from_torch(
        ids.reshape(1, 1, 1, -1).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Fast runtime mode skips per-op sub-graph capture; disable it so the
    # calltrace actually records the ttnn ops executed.
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture()
        out_tt = block.forward(ids_tt)
        captured = ttnn.graph.end_graph_capture()

    # Persist the traced ttnn op list for the orchestrator's no-shortcuts guard.
    calltrace = list(ttnn.graph.extract_calltrace(captured))
    traced = sorted({op.replace("::", ".") for op in calltrace if op.replace("::", ".").startswith("ttnn.")})
    side = MODEL_DIR / "tt" / "embedding.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # Replicated output (post all_gather): compare a single device's copy.
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    out = out.reshape(ref_out.shape)
    pcc = _pcc(ref_out, out)
    print(f"PCC(embedding) = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
