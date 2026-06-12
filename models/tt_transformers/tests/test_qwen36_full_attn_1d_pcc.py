# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-layer PCC test for the 1D-TP Qwen3.6 full-attention block.

Builds ONE ``TtQwen36FullAttention`` (a ``full_attention`` layer — layer 3 in
the canonical (3 linear + 1 full)×16 pattern) on an 8-chip 1D tensor-parallel
mesh, feeds it a prefill hidden state + mRoPE cos/sin, and compares the module
output to the CPU-reference ``GatedAttention`` block (from
``models/demos/qwen3_6_galaxy/reference/qwen36.py``). PCC > 0.99.

Exercises the qwen3.6 full-attn quirks: fused q|gate projection + de-interleave,
per-head zero-centered qk_norm, partial RoPE (first 64 of head_dim 256), sigmoid
output gate, KV pad 4→8 (1 KV head/chip), and the single out-proj all_reduce.

Run (framework path — conftest ``mesh_device`` fixture, NO --noconftest)::

    export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        && export MESH_DEVICE=P150x8 \\
        && export HF_MODEL=Qwen/Qwen3.6-27B \\
        && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest \\
            models/tt_transformers/tests/test_qwen36_full_attn_1d_pcc.py -v -s
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    os.environ.get(
        "QWEN36_SNAPSHOT",
        "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
        "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
)

_B = 1
_T_PREFILL = int(os.environ.get("QWEN36_PCC_T_PREFILL", "128"))
_H = 5120
_PCC_THRESH = 0.99
_LAYER_IDX = 3  # full_attention layer in the canonical pattern


_MESH_DEVICE_PARAM = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), (1, 8))

_framework_mesh = pytest.mark.parametrize("mesh_device", [_MESH_DEVICE_PARAM], indirect=True)
_framework_device_params = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)


def _load_layer_state_dict(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    needed = [k for k in weight_map if k.startswith(prefix)]
    files = sorted({weight_map[k] for k in needed})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _mrope_cos_sin(T: int):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    return build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )  # cos, sin: [T, 64]


def _cpu_reference_full_attn(layer_sd_hf: dict, x: torch.Tensor) -> torch.Tensor:
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))
    attn = GatedAttention(config).eval()

    pfx = f"model.language_model.layers.{_LAYER_IDX}.self_attn."
    ref_sd = {k[len(pfx) :]: v.float() for k, v in layer_sd_hf.items() if k.startswith(pfx)}
    missing, _ = attn.load_state_dict(ref_sd, strict=False)
    assert not [m for m in missing if "proj" in m or "norm" in m], f"missing ref weights: {missing}"

    T = x.shape[1]
    cos, sin = _mrope_cos_sin(T)
    causal = torch.zeros(1, 1, T, T)
    causal = causal.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))
    with torch.no_grad():
        out, _ = attn(x.float(), cos, sin, attention_mask=causal)  # cos/sin already [1, T, rd]
    return out  # [B, T, H]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def _build_args(mesh):
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh, max_batch_size=_B, max_seq_len=max(_T_PREFILL, 2048))
    assert getattr(args, "is_qwen36", False), "ModelArgs did not detect qwen3.6 — check HF_MODEL"
    return args


def _send_replicated_hidden(t: torch.Tensor, mesh):
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, B * T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _send_cos_sin(cos, sin, mesh):
    # cos/sin arrive [1, T, rd]; the class expects 4D [1, 1, T, rd] (slices dim -2).
    def up(t):
        _, T, rd = t.shape
        return ttnn.from_torch(
            t.reshape(1, 1, T, rd),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    return up(cos), up(sin)


def _gather_replicated_output(tt_tensor, mesh, T: int):
    # FRACTURED (reduce_scattered) output: each device holds its H/tp column slice.
    out = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    out = out[..., :_H]
    out = out.reshape(-1, _H)[:T]
    return out.float()


@pytest.mark.hardware
@_framework_device_params
@_framework_mesh
def test_qwen36_full_attn_1d_prefill_pcc(mesh_device):
    """1D-TP full-attention — prefill (T>1) block PCC vs CPU GatedAttention."""
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.qwen36_full_attention import TtQwen36FullAttention

    layer_sd = _load_layer_state_dict(_SNAPSHOT, _LAYER_IDX)
    print(f"[FullAttn-1D/prefill] loaded {len(layer_sd)} layer-{_LAYER_IDX} weights")

    args = _build_args(mesh_device)
    tt_ccl = TT_CCL(mesh_device)
    attn = TtQwen36FullAttention(
        mesh_device=mesh_device,
        args=args,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat8_b,
        state_dict=layer_sd,
        tt_ccl=tt_ccl,
    )

    torch.manual_seed(43)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
    out_ref = _cpu_reference_full_attn(layer_sd, x_cpu)
    print(f"[FullAttn-1D/prefill] CPU ref shape {tuple(out_ref.shape)}")

    cos, sin = _mrope_cos_sin(_T_PREFILL)
    cos_tt, sin_tt = _send_cos_sin(cos, sin, mesh_device)
    x_tt = _send_replicated_hidden(x_cpu, mesh_device)
    tt_out = attn.forward(x_tt, current_pos=None, rot_mats=(cos_tt, sin_tt), mode="prefill")
    tt_cpu = _gather_replicated_output(tt_out, mesh_device, _T_PREFILL)
    print(f"[FullAttn-1D/prefill] TT out shape {tuple(tt_cpu.shape)}")

    pcc = _pcc(tt_cpu, out_ref[0, :_T_PREFILL, :])
    print(f"[FullAttn-1D/prefill] PCC = {pcc:.6f} (thresh {_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"prefill PCC {pcc:.4f} < {_PCC_THRESH}"
