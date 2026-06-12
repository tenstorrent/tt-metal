# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-layer PCC test for the 1D-TP Qwen3.6 GatedDeltaNet attention.

Builds ONE ``TtQwen36GDNAttention`` (layer 0) on an 8-chip 1D tensor-parallel
mesh (``MESH_DEVICE=p15x8`` → mesh shape ``(1, 8)``), feeds it a decode-step
input and a prefill input, and compares the module output to the CPU-reference
``GatedDeltaNet`` (the layer-0 GDN block from
``models/demos/qwen3_6_galaxy/reference/qwen36.py``). PCC > 0.99.

The reference is the GDN BLOCK only (no residual / norm / MLP) — this test
exercises the attention module in isolation, so it feeds the SAME post-
attention-norm hidden state to both the reference block and the TT module and
compares the raw block outputs.

Weights are loaded directly from the raw HF safetensors snapshot
(``AutoModelForCausalLM`` is broken for this model_type — see MEMORY).

Run (framework path — uses the conftest ``mesh_device`` fixture, NO --noconftest)::

    export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        && export MESH_DEVICE=P150x8 \\
        && export HF_MODEL=Qwen/Qwen3.6-27B \\
        && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest \\
            models/tt_transformers/tests/test_qwen36_gdn_1d_pcc.py -v -s
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
_LAYER_IDX = 0


# ---------------------------------------------------------------------------
# Mesh: use the tt_transformers framework path — the conftest ``mesh_device``
# fixture + the demo's exact ``device_params`` marker. NO hand-rolled
# open_mesh_device / set_fabric_config: the framework picks the right fabric
# for the physical mesh (and tolerates the box's degraded eth links the way the
# qwen3-32b demo does) and tears it down cleanly. The two markers below mirror
# models/tt_transformers/demo/simple_text_demo.py verbatim.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Weight loading (raw safetensors — AutoModelForCausalLM broken for this model)
# ---------------------------------------------------------------------------


def _load_layer_state_dict(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    """Load the raw HF GDN tensors for ``layer_idx`` from the safetensors shards."""
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


# ---------------------------------------------------------------------------
# CPU reference: the GatedDeltaNet block (layer 0), float32
# ---------------------------------------------------------------------------


def _cpu_reference_gdn(layer_sd_hf: dict, x: torch.Tensor) -> torch.Tensor:
    """Run the reference ``GatedDeltaNet`` block at float32; return [B, T, H]."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))

    gdn = GatedDeltaNet(config).eval()

    # Map raw HF keys (model.language_model.layers.N.linear_attn.<rest>) to the
    # reference module's own state_dict (it consumes bare in_proj_qkv.weight etc.).
    pfx = f"model.language_model.layers.{_LAYER_IDX}.linear_attn."
    ref_sd = {k[len(pfx) :]: v.float() for k, v in layer_sd_hf.items() if k.startswith(pfx)}
    missing, unexpected = gdn.load_state_dict(ref_sd, strict=False)
    assert not [m for m in missing if not m.startswith("conv1d.bias")], f"missing ref weights: {missing}"

    with torch.no_grad():
        out, _conv, _rec = gdn(x.float())
    return out  # [B, T, H]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------------
# TT module construction + I/O helpers
# ---------------------------------------------------------------------------


def _build_args(mesh):
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh, max_batch_size=_B, max_seq_len=max(_T_PREFILL, 2048))
    assert getattr(args, "is_qwen36", False), "ModelArgs did not detect qwen3.6 — check HF_MODEL"
    return args


def _send_replicated_hidden(t: torch.Tensor, mesh):
    """Upload a full-H hidden state [B, T, H] replicated across the TP chips.

    For 1D-TP GDN the input projection has K == full H, so every chip needs the
    complete H (replicated), unlike the galaxy 2D-TP col-sharded contract.
    """
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, B * T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _gather_replicated_output(tt_tensor, mesh, T: int):
    """Reassemble the FRACTURED (reduce_scattered) output: each device holds its
    H/tp column slice; concat along the last dim → full [.., H]."""
    out = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    out = out[..., :_H]
    out = out.reshape(-1, _H)[:T]
    return out.float()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
@_framework_device_params
@_framework_mesh
def test_qwen36_gdn_1d_prefill_pcc(mesh_device):
    """1D-TP GDN attention — prefill (T>1) block PCC vs CPU GatedDeltaNet."""
    tp8_mesh = mesh_device
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.qwen36_gdn_attention import TtQwen36GDNAttention

    layer_sd = _load_layer_state_dict(_SNAPSHOT, _LAYER_IDX)
    print(f"[GDN-1D/prefill] loaded {len(layer_sd)} layer-{_LAYER_IDX} weights")

    args = _build_args(tp8_mesh)
    tt_ccl = TT_CCL(tp8_mesh)
    attn = TtQwen36GDNAttention(
        mesh_device=tp8_mesh,
        args=args,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat8_b,
        state_dict=layer_sd,
        tt_ccl=tt_ccl,
    )
    print(
        f"[GDN-1D/prefill] built (tp_size={attn.tp_size} n_v/chip={attn.n_v_per_chip} "
        f"n_k/chip={attn.n_k_per_chip} q/chip={attn.q_per_chip} v/chip={attn.v_per_chip})"
    )

    torch.manual_seed(7)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    out_ref = _cpu_reference_gdn(layer_sd, x_cpu)
    print(f"[GDN-1D/prefill] CPU ref shape {tuple(out_ref.shape)}")

    x_tt = _send_replicated_hidden(x_cpu, tp8_mesh)
    tt_out = attn.forward(x_tt, mode="prefill")
    tt_cpu = _gather_replicated_output(tt_out, tp8_mesh, _T_PREFILL)
    print(f"[GDN-1D/prefill] TT out shape {tuple(tt_cpu.shape)}")

    pcc = _pcc(tt_cpu, out_ref[0, :_T_PREFILL, :])
    print(f"[GDN-1D/prefill] PCC = {pcc:.6f} (thresh {_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"prefill PCC {pcc:.4f} < {_PCC_THRESH}"


@pytest.mark.hardware
@_framework_device_params
@_framework_mesh
def test_qwen36_gdn_1d_decode_zerostate_pcc(mesh_device):
    """1D-TP GDN decode (T=1) from ZERO state — isolates the decode forward
    (recurrent core + projections) from the prefill→decode state handoff.

    Both sides start from a cold (zero) conv + recurrent state and take a single
    decode step on the same random token. If this PASSES but the primed-decode
    test fails, the bug is in the prefill→decode state writeback/layout. If this
    FAILS, the bug is in the decode forward itself.
    """
    tp8_mesh = mesh_device
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.qwen36_gdn_attention import TtQwen36GDNAttention

    layer_sd = _load_layer_state_dict(_SNAPSHOT, _LAYER_IDX)
    args = _build_args(tp8_mesh)
    tt_ccl = TT_CCL(tp8_mesh)
    attn = TtQwen36GDNAttention(
        mesh_device=tp8_mesh,
        args=args,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat8_b,
        state_dict=layer_sd,
        tt_ccl=tt_ccl,
    )

    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))
    gdn = GatedDeltaNet(config).eval()
    pfx = f"model.language_model.layers.{_LAYER_IDX}.linear_attn."
    ref_sd = {k[len(pfx) :]: v.float() for k, v in layer_sd.items() if k.startswith(pfx)}
    gdn.load_state_dict(ref_sd, strict=False)

    torch.manual_seed(11)
    x_step = torch.randn(_B, 1, _H, dtype=torch.bfloat16)
    with torch.no_grad():
        out_ref_step, _c, _r = gdn(x_step.float())  # zero state both sub-states

    attn.clear_state()  # zero dn_state + conv_state buffers
    x_step_tt = ttnn.from_torch(
        x_step.reshape(1, 1, 1, _H),
        device=tp8_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(tp8_mesh),
    )
    tt_out_step = attn.forward(x_step_tt, mode="decode")
    tt_step_cpu = _gather_replicated_output(tt_out_step, tp8_mesh, 1)
    pcc = _pcc(tt_step_cpu, out_ref_step[0, :1, :])
    print(f"[GDN-1D/decode-zerostate] PCC = {pcc:.6f} (thresh {_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"zero-state decode PCC {pcc:.4f} < {_PCC_THRESH}"


@pytest.mark.hardware
@_framework_device_params
@_framework_mesh
def test_qwen36_gdn_1d_decode_pcc(mesh_device):
    """1D-TP GDN attention — decode (T=1) block PCC vs CPU GatedDeltaNet.

    Primes the recurrent + conv state with a short prefill, then takes one
    decode step. The CPU reference threads the same conv/recurrent state across
    the prefill→decode boundary, so the comparison is end-to-end consistent.
    """
    tp8_mesh = mesh_device
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.qwen36_gdn_attention import TtQwen36GDNAttention

    layer_sd = _load_layer_state_dict(_SNAPSHOT, _LAYER_IDX)

    args = _build_args(tp8_mesh)
    tt_ccl = TT_CCL(tp8_mesh)
    attn = TtQwen36GDNAttention(
        mesh_device=tp8_mesh,
        args=args,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat8_b,
        state_dict=layer_sd,
        tt_ccl=tt_ccl,
    )

    # --- CPU reference: prime with a prefill, keep its state, then 1 decode --
    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))
    gdn = GatedDeltaNet(config).eval()
    pfx = f"model.language_model.layers.{_LAYER_IDX}.linear_attn."
    ref_sd = {k[len(pfx) :]: v.float() for k, v in layer_sd.items() if k.startswith(pfx)}
    gdn.load_state_dict(ref_sd, strict=False)

    torch.manual_seed(11)
    T_prime = 32
    x_prime = torch.randn(_B, T_prime, _H, dtype=torch.bfloat16)
    x_step = torch.randn(_B, 1, _H, dtype=torch.bfloat16)

    with torch.no_grad():
        _out_p, conv_state, rec_state = gdn(x_prime.float())
        out_ref_step, _conv2, _rec2 = gdn(x_step.float(), conv_state=conv_state, recurrent_state=rec_state)
    print(f"[GDN-1D/decode] CPU ref step shape {tuple(out_ref_step.shape)}")

    # --- TT: prime the persistent state with the same prefill, then 1 decode -
    attn.clear_state()
    x_prime_tt = _send_replicated_hidden(x_prime, tp8_mesh)
    _ = attn.forward(x_prime_tt, mode="prefill")  # writes dn_state / conv_state buffers

    # Decode input is the [1, 1, R, H] tile-padded-row layout the backbone uses;
    # at B=1, R=1 it's a single token in the row slot.
    x_step_tt = ttnn.from_torch(
        x_step.reshape(1, 1, 1, _H),
        device=tp8_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(tp8_mesh),
    )
    tt_out_step = attn.forward(x_step_tt, mode="decode")
    tt_step_cpu = _gather_replicated_output(tt_out_step, tp8_mesh, 1)
    print(f"[GDN-1D/decode] TT out step shape {tuple(tt_step_cpu.shape)}")

    pcc = _pcc(tt_step_cpu, out_ref_step[0, :1, :])
    print(f"[GDN-1D/decode] PCC = {pcc:.6f} (thresh {_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"decode PCC {pcc:.4f} < {_PCC_THRESH}"
