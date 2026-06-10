# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: TtVisionTransformer vs the golden reference tensor on a 1x4 mesh.

Golden input (pre-flattened patches), output and grid_thw produced by
reference/test_functional.py (PCC 1.0 vs the official HF DotsVisionTransformer
with ALL real fp32 vision_tower weights). Weights are re-loaded from the HF
checkpoint here, as the golden stores only activations. Rope tables and
cu_seqlens are computed on host from grid_thw exactly as the reference does.
Output is replicated across the mesh, so we compare ONE device's copy, sliced
back to the unpadded merged length (seq // m^2 rows).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
GOLDEN = MODEL_DIR / "reference" / "golden" / "vision_transformer.pt"
REPO = "rednote-hilab/dots.ocr"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location(
    "dots_ocr_tt_vision_transformer", MODEL_DIR / "tt" / "vision_transformer.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtVisionTransformer = _mod.TtVisionTransformer

_ref_spec = importlib.util.spec_from_file_location("dots_ocr_ref_functional", MODEL_DIR / "reference" / "functional.py")
_ref = importlib.util.module_from_spec(_ref_spec)
sys.modules[_ref_spec.name] = _ref
_ref_spec.loader.exec_module(_ref)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _load_vision_tower_weights():
    """All vision_tower.* tensors from the HF checkpoint, prefix stripped."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    prefix = "vision_tower."
    out = {}
    by_file = {}
    for full, fname in idx.items():
        if full.startswith(prefix):
            by_file.setdefault(fname, []).append(full)
    for fname, keys in by_file.items():
        with safe_open(snap / fname, framework="pt") as f:
            for full in keys:
                out[full[len(prefix) :]] = f.get_tensor(full).float()
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_transformer_pcc(mesh_device):
    golden = torch.load(GOLDEN)
    x, ref_out, grid_thw = golden["input"], golden["output"], golden["grid_thw"]
    seq, patch_dim = x.shape
    num_heads = 12
    spatial_merge_size = 2

    sd = _load_vision_tower_weights()
    # fp32 weights + fp32 input: 42 stacked blocks compound bf16 weight/
    # activation quantization error below the 0.99 PCC bar; only the
    # bf16-only rope/SDPA attention core dips to bf16 (see tt module docs).
    model = TtVisionTransformer(mesh_device, sd, num_layers=42, num_heads=num_heads, dtype=ttnn.float32)

    # Host-side rope table + window boundaries, exactly as the reference
    # (ARCHITECTURE.md hybrid_notes: rot_pos_emb / cu_seqlens stay on host).
    head_dim = sd["blocks.0.attn.qkv.weight"].shape[-1] // num_heads
    rope = _ref.vision_rot_pos_emb(grid_thw, head_dim=head_dim, spatial_merge_size=spatial_merge_size)
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # Pad seq to a multiple of 128 (qwen25_vl prefill convention); cu_seqlens
    # keeps the UNPADDED boundaries so pad rows are never attended to.
    padded_seq = ((seq + 127) // 128) * 128
    x_pad = torch.cat([x, torch.zeros(padded_seq - seq, patch_dim)], dim=0)

    x_tt = ttnn.from_torch(
        x_pad.reshape(1, 1, padded_seq, patch_dim),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = model.prepare_rope(rope, padded_seq)
    cu_tt = model.prepare_cu_seqlens(cu_seqlens)

    # Fast runtime mode skips per-op sub-graph capture; disable it so the
    # calltrace actually records the ttnn ops executed.
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture()
        out_tt = model.forward(x_tt, rot_mats, cu_tt)
        captured = ttnn.graph.end_graph_capture()

    # Persist the traced ttnn op list for the orchestrator's no-shortcuts guard.
    calltrace = list(ttnn.graph.extract_calltrace(captured))
    traced = sorted({op.replace("::", ".") for op in calltrace if op.replace("::", ".").startswith("ttnn.")})
    side = MODEL_DIR / "tt" / "vision_transformer.traced_ops.json"
    side.write_text(json.dumps(traced, indent=2) + "\n")
    print(f"traced ops: {traced}")

    # Replicated output: compare a single device's copy; pad rows merge into
    # trailing rows only, so keep the first seq // m^2 merged rows.
    merged_seq = seq // spatial_merge_size**2
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()[:merged_seq]
    assert out.shape == ref_out.shape, f"{out.shape} != {ref_out.shape}"

    pcc = _pcc(ref_out, out)
    print(f"PCC(vision_transformer) = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
