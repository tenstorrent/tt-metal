# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-7 — Layer-0 DeltaNet (linear_attention) prefill PCC test on BH GLX 8×4.

Builds a 1-layer TtTransformer (layer 0 of the hybrid [lin,lin,lin,full]×16
schedule, i.e. ``linear_attention``), then directly forwards a small
T-token sequence through the DeltaNet block on device and compares the
output to the CPU PyTorch reference ``GatedDeltaNet`` block from
``models/demos/qwen3_6_galaxy/reference/qwen36.py``. PCC threshold > 0.99.

Block-level pattern (mirrors v1 ``test_qwen36_deltanet.py``): we feed the
DeltaNet a full-H replicated random hidden state (i.e. pre-norm), which
matches the layout DeltaNet expects via its ``row_shard_out`` weight
sharding. The full v2 TtTransformer prefill forward path goes through a
sharded attention_norm output (col-sharded at dim 3) that's incompatible
with the qwen36 attention/DeltaNet ``dims=(None, 3)`` weight layout —
that's a pending V2-decoder refactor tracked separately. For V2-7 we
lock down the DeltaNet block's math at the same call surface as v1.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_layer0_deltanet_forward_pcc.py \\
            -v -s
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
_T_PREFILL = 128  # must match TT_CCL prefill ``support_seqlens`` (4096/2048/1024/128)
_H = 5120
_PCC_THRESH = 0.99


# ---------------------------------------------------------------------------
# Mesh fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    """Load the HF tensors required to construct a 1-layer TtTransformer."""
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        f"model.language_model.layers.{layer_idx}.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _deltanet_short_weights(state_dict: dict, layer_idx: int) -> dict:
    """Extract the linear_attn.* short-key dict needed by the CPU reference."""
    pfx = f"model.language_model.layers.{layer_idx}.linear_attn."
    return {k[len(pfx) :]: v.float() for k, v in state_dict.items() if k.startswith(pfx)}


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _cpu_reference_deltanet(sd_short: dict, x: torch.Tensor) -> torch.Tensor:
    """Run the validated GatedDeltaNet CPU reference at full float32."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    model = GatedDeltaNet(config).eval()
    with torch.no_grad():
        for key in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "out_proj"]:
            getattr(model, key).weight.data.copy_(sd_short[f"{key}.weight"])
        model.A_log.data.copy_(sd_short["A_log"])
        model.dt_bias.data.copy_(sd_short["dt_bias"])
        model.norm.weight.data.copy_(sd_short["norm.weight"])
        out, _, _ = model(x.float())
    return out  # [B, T, H]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------------
# 1-layer TtTransformer builder
# ---------------------------------------------------------------------------


def _build_tt_model(mesh, state_dict):
    """Construct a 1-layer TtTransformer with layer 0 = linear_attention."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = ["linear_attention"]

    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


# ---------------------------------------------------------------------------
# Block-level forward + gather helpers
# ---------------------------------------------------------------------------


def _send_replicated(t: torch.Tensor, mesh, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _gather_replicated_first_dev(tt_tensor, mesh, T: int = None):
    out = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    out = out[0:1]
    if T is not None:
        # squeeze leading 4D into [B, T, H]
        while out.dim() > 3 and out.shape[0] == 1:
            out = out.squeeze(0)
        if out.dim() == 3:
            out = out[:, :T, :]
        else:
            out = out[..., :T, :]
    return out


# ---------------------------------------------------------------------------
# Test 1: Layer 0 DeltaNet — prefill PCC vs reference
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_qwen36_layer0_deltanet_prefill_pcc(bh_glx_mesh):
    """Layer-0 DeltaNet prefill: PCC > 0.99 vs CPU reference (block-level)."""
    layer_idx = 0
    state_dict = _load_state_dict_for_layer(_SNAPSHOT, layer_idx)
    print(f"[Layer0] loaded {len(state_dict)} weights")

    # --- Build TT model (covers TtTransformer construction surface) ---
    model, args = _build_tt_model(bh_glx_mesh, state_dict)
    assert getattr(model.layers[0], "is_linear_attention_layer", False) is True
    print(f"[Layer0] TT 1-layer DeltaNet built — H={args.dim}")

    # --- Random hidden state (mirroring v1 pattern) ---
    torch.manual_seed(42)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
    print(f"[Layer0] input shape: {x_cpu.shape}")

    # --- CPU reference (DeltaNet block alone, no norm/MLP) ---
    sd_short = _deltanet_short_weights(state_dict, layer_idx)
    out_ref = _cpu_reference_deltanet(sd_short, x_cpu)  # [B, T, H]
    print(f"[Layer0] CPU ref shape: {out_ref.shape}")

    # --- TT DeltaNet block forward (full-H replicated input → block output) ---
    x_tt = _send_replicated(x_cpu, bh_glx_mesh, dtype=ttnn.bfloat16)
    tt_out = model.layers[0].attention.forward(
        x_tt,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    )
    tt_out_cpu = _gather_replicated_first_dev(tt_out, bh_glx_mesh, T=_T_PREFILL)
    tt_out_cpu = tt_out_cpu.reshape(_B, _T_PREFILL, _H).float()
    print(f"[Layer0] TT out shape: {tt_out_cpu.shape}")

    pcc = _pcc(tt_out_cpu, out_ref[:, :_T_PREFILL, :])
    p99 = torch.quantile((tt_out_cpu.float() - out_ref[:, :_T_PREFILL, :].float()).abs().flatten(), 0.99).item()
    print(f"[Layer0] PCC = {pcc:.6f} (thresh={_PCC_THRESH})  |  p99 abs-diff = {p99:.4f}")
    assert pcc > _PCC_THRESH, f"Layer0 DeltaNet PCC {pcc:.4f} < {_PCC_THRESH} (p99={p99:.4f})"
    print("[Layer0] PASSED")
