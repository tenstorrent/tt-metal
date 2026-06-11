# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: TtLMHead vs the golden reference tensor on a 1x4 mesh.

Golden input/output produced by reference/test_functional.py (PCC 1.0 vs
torch.nn.Linear with the real lm_head.weight). The weight is re-loaded from
the HF checkpoint here, as the golden stores only activations. The block is
vocab-sharded per the parallelism plan (column-parallel, 151936/4 = 37984
logits per chip); the sharded logits are recombined on host with
ConcatMeshToTensor(dim=-1) per tp-guidance.
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
GOLDEN = MODEL_DIR / "reference" / "golden" / "lm_head.pt"
REPO = "rednote-hilab/dots.ocr"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_tt_lm_head", MODEL_DIR / "tt" / "lm_head.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtLMHead = _mod.TtLMHead


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
def test_lm_head_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    x, ref_out = golden["input"], golden["output"]
    _, seq, dim = x.shape
    vocab = ref_out.shape[-1]

    sd = _load_weights("lm_head", ["weight"])
    block = TtLMHead(mesh_device, sd)

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
    side = MODEL_DIR / "tt" / "lm_head.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # Vocab-sharded logits: concat the per-chip slices on host (tp-guidance).
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    out = out.float().reshape(seq, vocab)
    assert out.shape == ref_out.reshape(seq, vocab).shape, f"{out.shape} != {ref_out.shape}"

    pcc = _pcc(ref_out.reshape(seq, vocab), out)
    print(f"PCC(lm_head) = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_lm_head_decode_pcc(mesh_device):
    """Decode posture (occupancy REDO, tick-66): bf16 [1,1,1,H] row, bfp8 weight.

    The production traced decode body projects ONE bf16 final-norm row per
    token. Gates the per-row logits PCC AND greedy-argmax exactness over 8
    fresh rows (the AR-loop consumer is argmax — a PCC-passing row that flips
    its argmax would silently derail generation; this is what disqualified
    the bfloat4_b weight lever: PCC 0.9884, 6/16 flips).
    """
    golden = torch.load(GOLDEN)
    sd = _load_weights("lm_head", ["weight"])
    block = TtLMHead(mesh_device, sd)
    w = sd["weight"]

    rows = golden["input"].reshape(-1, golden["input"].shape[-1])
    worst = 1.0
    for step in range(8):
        x = rows[step : step + 1].reshape(1, 1, 1, -1)
        ref = x.reshape(1, -1) @ w.T
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out_tt = block.forward(x_tt)
        out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float().reshape(1, -1)
        pcc = _pcc(ref, out)
        worst = min(worst, pcc)
        assert pcc > 0.99, f"decode row {step}: PCC {pcc:.6f} < 0.99"
        assert int(out.argmax()) == int(ref.argmax()), f"decode row {step}: greedy argmax flipped"
    print(f"PCC(lm_head decode, worst of 8 rows) = {worst:.6f}")
