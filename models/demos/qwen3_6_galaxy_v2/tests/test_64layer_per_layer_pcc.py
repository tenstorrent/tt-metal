# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-7b — Per-layer PCC sweep for 64-layer qwen3.6-27B prefill.

Instruments ``TtTransformer.forward(mode="prefill")`` to capture hidden state
after every layer, runs the CPU reference also capturing per-layer hidden
states, and prints a PCC table showing exactly which layer the dtype leak
begins to compound at.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_64layer_per_layer_pcc.py \\
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

_T_PREFILL = 128
_N_LAYERS = 64


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


def _load_state_dict_all_layers(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        "model.language_model.layers.",
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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _cpu_reference_per_layer(state_dict_hf, x):
    """Returns list of per-layer hidden states (64 tensors), each [1, T, H]."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config, build_mrope_cos_sin

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    T = x.shape[1]
    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    causal_mask = torch.zeros(1, 1, T, T)
    causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))

    hidden = x.float()
    per_layer_hidden: list[torch.Tensor] = []
    for layer_idx in range(_N_LAYERS):
        layer = HybridDecoderLayer(config, layer_idx).eval()
        pfx = f"model.language_model.layers.{layer_idx}."
        layer_sd = {}
        for k, v in state_dict_hf.items():
            if k.startswith(pfx):
                short = k[len(pfx) :]
                if short.startswith("self_attn."):
                    layer_sd["attention." + short[len("self_attn.") :]] = v.float()
                elif short.startswith("linear_attn."):
                    layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
                else:
                    layer_sd[short] = v.float()
        layer.load_state_dict(layer_sd, strict=False)
        with torch.no_grad():
            hidden, _, _, _ = layer(hidden, cos, sin, attention_mask=causal_mask)
        per_layer_hidden.append(hidden.clone())
        del layer
    return per_layer_hidden, config


def _build_tt_model(mesh, state_dict, pattern, n_layers):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    args.linear_attention_pattern = pattern
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


def _build_partial_rope_cos_sin_tt(mesh, T):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
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


def _send_col_sharded_hidden(t, mesh, args):
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


def _gather_col_sharded_to_full(tt_tensor, mesh, args, T):
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


@pytest.mark.hardware
def test_qwen36_64_layer_per_layer_pcc(bh_glx_mesh):
    """Per-layer hidden-state PCC sweep — find where compounding error begins."""
    print("[per-layer] loading HF state_dict ...")
    state_dict = _load_state_dict_all_layers(_SNAPSHOT)

    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    pattern = list(config.layer_types)
    assert len(pattern) == _N_LAYERS

    print("[per-layer] building TT 64-layer model ...")
    model, args = _build_tt_model(bh_glx_mesh, state_dict, pattern, _N_LAYERS)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    input_ids_padded = torch.zeros(1, _T_PREFILL, dtype=input_ids.dtype)
    input_ids_padded[0, :T_prompt] = input_ids[0]

    embed_w = state_dict["model.language_model.embed_tokens.weight"].float()
    x_cpu_torch = embed_w[input_ids_padded[0]].unsqueeze(0)
    print(f"[per-layer] CPU reference: 64 layers ...")
    per_layer_ref, _ = _cpu_reference_per_layer(state_dict, x_cpu_torch)
    print(f"[per-layer] CPU ref done, captured {len(per_layer_ref)} layers")

    x_tt = _send_col_sharded_hidden(x_cpu_torch.to(torch.bfloat16), bh_glx_mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    # Instrument: replace TtTransformer.forward layer loop to capture per-layer outputs.
    print("[per-layer] running TT 64-layer prefill (instrumented) ...")

    # Manually replicate the forward loop, snapping x at each step.
    rot_mats = (cos_tt, sin_tt)
    x = x_tt
    h = None
    per_layer_pcc: list[float] = []
    per_layer_dtype: list[str] = []
    for i, layer in enumerate(model.layers):
        x, h = layer(
            x,
            h,
            None,
            rot_mats,
            0,
            "prefill",
            None,
            chunk_page_table=None,
            chunk_start_idx=0,
            chunk_start_idx_tensor=chunk_start_idx_tt,
            kv_cache=None,
            batch_size=1,
        )
        # Don't deallocate x — it's the layer output going into next layer.
        # Clone via to_torch (gathers data; non-destructive).
        tt_hidden_cpu = _gather_col_sharded_to_full(x, bh_glx_mesh, args, T=_T_PREFILL)
        tt_hidden_cpu = tt_hidden_cpu.reshape(1, _T_PREFILL, -1).float()
        ref_hidden = per_layer_ref[i][:, :_T_PREFILL, :].float()
        pcc_full = _pcc(tt_hidden_cpu, ref_hidden)
        # PCC over only the real prompt tokens (positions 0..T_prompt-1) and
        # over only the last prompt position (the one that matters for logits).
        pcc_prompt = _pcc(tt_hidden_cpu[:, :T_prompt, :], ref_hidden[:, :T_prompt, :])
        pcc_last = _pcc(tt_hidden_cpu[:, T_prompt - 1 : T_prompt, :], ref_hidden[:, T_prompt - 1 : T_prompt, :])
        per_layer_pcc.append(pcc_last)
        per_layer_dtype.append(str(x.dtype))
        print(
            f"[per-layer] L{i:02d} ({pattern[i][:3]}): "
            f"PCC_full={pcc_full:.4f} PCC_prompt={pcc_prompt:.4f} PCC_last={pcc_last:.4f} | "
            f"tt_std={tt_hidden_cpu.std().item():.3f} ref_std={ref_hidden.std().item():.3f} | "
            f"x.dtype={str(x.dtype).split('.')[-1]}"
        )

    print("\n[per-layer] SUMMARY")
    print(f"  First layer with PCC<0.99: ", end="")
    fail_idx = next((i for i, p in enumerate(per_layer_pcc) if p < 0.99), None)
    print(fail_idx)
    print(f"  Final layer PCC: {per_layer_pcc[-1]:.6f}")
    # Don't assert — this is diagnostic.
