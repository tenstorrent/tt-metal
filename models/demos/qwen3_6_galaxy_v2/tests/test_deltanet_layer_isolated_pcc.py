# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-DEC — Isolated single-layer DeltaNet PCC across layer_idx × mode.

Extends ``test_layer0_deltanet_forward_pcc.py`` (which only tested layer 0
prefill) to:
  - Two DeltaNet layer indices (0 and 4) — both are ``linear_attention`` per
    the qwen3.6 hybrid pattern (full_attention is at indices 3, 7, 11, ...).
  - Two modes (prefill, decode) — decode runs prefill T=128 first to populate
    the recurrent / conv state buffers, then a single decode step at
    position 128. Compares decode output to CPU reference's position-128 output
    (CPU runs forward on T=129).

Diagnostic question: at 4L hooked decode, L0 DeltaNet gives PCC 0.9999 while
L1+ DeltaNet gives PCC 0.9962. This test isolates whether the bug is:
  (a) DeltaNet's standalone math (would manifest at all layers including L0)
  (b) Layer-specific (manifests at non-zero layer indices in isolation)
  (c) Cross-layer state propagation (only shows up in multi-layer runs)

Target PCC > 0.999 on each combination.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && pytest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_deltanet_layer_isolated_pcc.py \\
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
_T_PREFILL = 128
_H = 5120
_PCC_THRESH = 0.999  # tighter: user-specified target


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


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
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
    # If layer_idx != 0, relabel weights to layers.0 so the 1-layer model picks
    # them up at slot 0 (the model has only n_layers=1).
    if layer_idx != 0:
        relabeled: dict[str, torch.Tensor] = {}
        old_prefix = f"model.language_model.layers.{layer_idx}."
        new_prefix = "model.language_model.layers.0."
        for k, v in sd.items():
            if k.startswith(old_prefix):
                relabeled[new_prefix + k[len(old_prefix) :]] = v
            else:
                relabeled[k] = v
        sd = relabeled
    return sd


def _cpu_reference_layer_forward(
    state_dict_full_hf: dict,
    layer_idx_in_sd: int,
    layer_type: str,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run HybridDecoderLayer on the FULL x (length T). Returns [B, T, H]."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    layer = HybridDecoderLayer(config, 0).eval()  # we relabeled weights to layer 0

    pfx = f"model.language_model.layers.{layer_idx_in_sd}."
    layer_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict_full_hf.items():
        if k.startswith(pfx):
            short = k[len(pfx) :]
            if short.startswith("self_attn."):
                layer_sd["attention." + short[len("self_attn.") :]] = v.float()
            elif short.startswith("linear_attn."):
                layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
            else:
                layer_sd[short] = v.float()
    layer.load_state_dict(layer_sd, strict=False)

    # linear_attention: no RoPE/mask.
    with torch.no_grad():
        out, _, _, _ = layer(x.float(), cos=None, sin=None, attention_mask=None)
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model(mesh, state_dict):
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


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _gather_col_sharded_to_full(tt_tensor, mesh, args, T: int):
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


def _build_partial_rope_cos_sin_tt(mesh, T: int):
    """Minimal cos/sin tensor (rank-3 [1, T, 64])."""
    cos_ref = torch.zeros(T, 64, dtype=torch.bfloat16)
    sin_ref = torch.zeros(T, 64, dtype=torch.bfloat16)
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
@pytest.mark.parametrize(
    "layer_idx,mode",
    [
        (0, "prefill"),
        (4, "prefill"),
        (0, "decode"),
        (4, "decode"),
    ],
)
def test_deltanet_isolated_pcc(bh_glx_mesh, layer_idx, mode):
    """Single-layer DeltaNet block PCC at (layer_idx, mode).

    Hybrid pattern in qwen3.6: indices 0, 1, 2, 4, 5, 6, ... are
    linear_attention. Index 3 is full_attention. We test 0 and 4 (both
    DeltaNet) to detect any layer-specific bug.
    """
    tag = f"L{layer_idx}/{mode}"
    state_dict = _load_state_dict_for_layer(_SNAPSHOT, layer_idx)
    print(f"[{tag}] loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict)
    assert getattr(model.layers[0], "is_linear_attention_layer", False) is True

    torch.manual_seed(42)
    # After _load_state_dict_for_layer, the state_dict already has layer-0 keys
    # (we relabeled). So the CPU reference must also load from layer 0.
    cpu_ref_layer_idx = 0
    if mode == "prefill":
        # Single forward of T_PREFILL tokens.
        x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
        ref_full = _cpu_reference_layer_forward(state_dict, cpu_ref_layer_idx, "linear_attention", x_cpu)
        out_ref = ref_full

        x_tt = _send_col_sharded_hidden(x_cpu, bh_glx_mesh, args)
        cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
        chunk_start_idx_tt = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=bh_glx_mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
        )
        tt_out = model.forward(
            x_tt,
            current_pos=None,
            rot_mats=(cos_tt, sin_tt),
            user_id=0,
            mode="prefill",
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=chunk_start_idx_tt,
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        tt_out_cpu = _gather_col_sharded_to_full(tt_out, bh_glx_mesh, args, T=_T_PREFILL)
        tt_out_cpu = tt_out_cpu.reshape(_B, _T_PREFILL, _H).float()
        T_compare = _T_PREFILL

    else:  # decode
        # Step 1: prefill T_PREFILL on TT (populates state buffers).
        x_cpu_full = torch.randn(_B, _T_PREFILL + 1, _H, dtype=torch.bfloat16)
        x_cpu_prefill = x_cpu_full[:, :_T_PREFILL, :]
        x_cpu_decode = x_cpu_full[:, _T_PREFILL : _T_PREFILL + 1, :]  # [B, 1, H]

        # CPU reference: full T=129 forward, take position-128 output.
        ref_full = _cpu_reference_layer_forward(state_dict, cpu_ref_layer_idx, "linear_attention", x_cpu_full)
        out_ref = ref_full[:, _T_PREFILL : _T_PREFILL + 1, :]  # [B, 1, H]

        # TT prefill T=128.
        x_tt_pre = _send_col_sharded_hidden(x_cpu_prefill, bh_glx_mesh, args)
        cos_tt_pre, sin_tt_pre = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
        chunk_start_idx_tt = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=bh_glx_mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
        )
        _ = model.forward(
            x_tt_pre,
            current_pos=None,
            rot_mats=(cos_tt_pre, sin_tt_pre),
            user_id=0,
            mode="prefill",
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=chunk_start_idx_tt,
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        x_tt_pre.deallocate(True)
        print(f"[{tag}] prefill T={_T_PREFILL} complete; state buffers populated")

        # TT decode step at position 128.
        x_tt_dec = _send_col_sharded_hidden(x_cpu_decode, bh_glx_mesh, args)
        cos_tt_dec, sin_tt_dec = _build_partial_rope_cos_sin_tt(bh_glx_mesh, 1)
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([_T_PREFILL] * _B, dtype=torch.int32),
            device=bh_glx_mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
        )
        tt_out = model.forward(
            x_tt_dec,
            current_pos=cur_pos_tt,
            rot_mats=(cos_tt_dec, sin_tt_dec),
            user_id=0,
            mode="decode",
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=None,
            start_pos=_T_PREFILL,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        # decode forward returns a list-of-batches; first (only) element is the tensor.
        if isinstance(tt_out, list):
            tt_out = tt_out[0]
        tt_out_cpu = _gather_col_sharded_to_full(tt_out, bh_glx_mesh, args, T=1)
        tt_out_cpu = tt_out_cpu.reshape(_B, 1, _H).float()
        T_compare = 1

    pcc = _pcc(tt_out_cpu, out_ref[:, :T_compare, :])
    p99 = torch.quantile((tt_out_cpu.float() - out_ref[:, :T_compare, :].float()).abs().flatten(), 0.99).item()
    print(f"[{tag}] PCC = {pcc:.6f} (thresh={_PCC_THRESH})  |  p99 abs-diff = {p99:.4f}")
    assert pcc > _PCC_THRESH, f"[{tag}] PCC {pcc:.6f} < {_PCC_THRESH} (p99={p99:.4f})"
    print(f"[{tag}] PASSED")
