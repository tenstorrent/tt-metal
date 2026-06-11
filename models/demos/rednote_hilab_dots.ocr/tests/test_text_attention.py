# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests: TtTextAttention vs golden/reference on a 1x4 mesh.

test_text_attention_pcc: golden at seq=128 (fp32 explicit path, layer-0
weights; PCC 1.0 vs HF reference). The block is TP-sharded (3 Q heads/chip,
kv_replication=2, row-parallel o_proj + all-reduce); all-reduced output is
replicated so we compare ONE device's copy.

test_text_attention_sdpa_long_seq: seq=1280 (> 1024 -> bf16 flash SDPA path).
Uses real layer-0 weights; reference is the fp32 torch causal GQA.  Measured
PCC ~0.962 (seq=1280) and ~0.955 (P32=2912); PCC does NOT improve with seq
because the Qwen2 attention sink (K[0] at ±3122) affects every query row.
Gate: PCC > 0.95 (regression fence; real correctness gate is the E2E WER test).
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
GOLDEN = MODEL_DIR / "reference" / "golden" / "text_attention.pt"
REPO = "rednote-hilab/dots.ocr"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_attention", MODEL_DIR / "tt" / "text_attention.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextAttention = _mod.TtTextAttention


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
def test_text_attention_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    x, ref_out = golden["input"], golden["output"]
    cos, sin = golden["cos"], golden["sin"]
    _, seq, dim = x.shape

    sd = _load_weights(
        "model.layers.0.self_attn",
        [
            "q_proj.weight",
            "q_proj.bias",
            "k_proj.weight",
            "k_proj.bias",
            "v_proj.weight",
            "v_proj.bias",
            "o_proj.weight",
        ],
    )
    block = TtTextAttention(mesh_device, sd, num_heads=12, num_kv_heads=2)

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq, dim).float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = block.prepare_rope(cos, sin)
    causal_mask = block.prepare_causal_mask(seq)

    # Fast runtime mode skips per-op sub-graph capture; disable it so the
    # calltrace actually records the ttnn ops executed.
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture()
        out_tt = block.forward(x_tt, rot_mats, causal_mask)
        captured = ttnn.graph.end_graph_capture()

    # Persist the traced ttnn op list for the orchestrator's no-shortcuts guard.
    calltrace = list(ttnn.graph.extract_calltrace(captured))
    traced = sorted({op.replace("::", ".") for op in calltrace if op.replace("::", ".").startswith("ttnn.")})
    side = MODEL_DIR / "tt" / "text_attention.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # All-reduced (replicated) output: compare a single device's copy.
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float().reshape(seq, dim)
    assert out.shape == ref_out.reshape(seq, dim).shape, f"{out.shape} != {ref_out.shape}"

    pcc = _pcc(ref_out.reshape(seq, dim), out)
    print(f"PCC(text_attention) = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"


def _load_ref_functional():
    name = "dots_ocr_reference_functional"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODEL_DIR / "reference" / "functional.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_text_attention_sdpa_long_seq(mesh_device):
    """bf16 flash SDPA path (seq=1280 > 1024) vs fp32 torch reference.

    The Qwen2 attention sink (layer-0 K[0] dominates every Q-row softmax at
    ±3122 logits) limits bf16 SDPA PCC to ~0.962 at seq=1280 and ~0.955 at
    production P32=2912; PCC does NOT improve with longer seq because every
    query row is affected (not just 1/S rows). The real correctness gate is
    the E2E WER test; the PCC threshold here is a regression fence.
    """
    _ref_mod = _load_ref_functional()

    SEQ = 1280
    DIM = 1536

    sd = _load_weights(
        "model.layers.0.self_attn",
        [
            "q_proj.weight",
            "q_proj.bias",
            "k_proj.weight",
            "k_proj.bias",
            "v_proj.weight",
            "v_proj.bias",
            "o_proj.weight",
        ],
    )
    cos, sin = _ref_mod.text_rope_cos_sin(torch.arange(SEQ).unsqueeze(0))

    torch.manual_seed(42)
    x = torch.randn(1, SEQ, DIM)

    ref_out = _ref_mod.text_attention_forward(x.float(), sd, cos, sin)

    block = TtTextAttention(mesh_device, sd, num_heads=12, num_kv_heads=2)
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, SEQ, DIM).float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = block.prepare_rope(cos, sin)
    causal_mask = block.prepare_causal_mask(SEQ)  # returns None for SEQ > 1024

    out_tt = block.forward(x_tt, rot_mats, causal_mask)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float().reshape(SEQ, DIM)

    pcc = _pcc(ref_out.reshape(SEQ, DIM), out)
    print(f"PCC(text_attention_sdpa_long_seq, seq={SEQ}) = {pcc:.6f}")
    assert pcc > 0.95, f"PCC {pcc:.6f} < 0.95"
