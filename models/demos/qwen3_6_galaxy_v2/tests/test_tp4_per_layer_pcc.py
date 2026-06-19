# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 per-layer PCC sweep for Qwen3.6-27B prefill on P150x4 (chips 8-11).

Loads the Qwen35Model (TP=4, 1D tensor-parallel) from the qwen9b-wt tree,
runs a per-layer hidden-state PCC sweep at T=4096, and compares against the
SAME fp32 CPU golden used by the TP=32 test (test_64layer_per_layer_pcc.py).

This establishes whether the TP=32 model's ~0.84 final-layer PCC is at the
TP=4 ceiling (same hardware precision) or still has a fixable gap above it.

Environment:
    cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    TF5_SP=/tmp/tf5env/lib/python3.10/site-packages
    export PYTHONPATH=$TF5_SP:/home/tt-admin/ssinghal/qwen36/qwen9b-wt:$(pwd)
    export TT_VISIBLE_DEVICES=8,9,10,11 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B

Run:
    python -m pytest --noconftest \\
        models/demos/qwen3_6_galaxy_v2/tests/test_tp4_per_layer_pcc.py \\
        -v -s
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

# The qwen9b-wt tree must be first in sys.path for its models/experimental/
# gated_attention_gated_deltanet to resolve correctly — the editable-install
# .pth in python_env adds new/tt-metal's models/ to sys.path unconditionally
# (bypassing PYTHONPATH order), so we explicitly prepend qwen9b-wt here.
_QWEN9B_WT = pathlib.Path("/home/tt-admin/ssinghal/qwen36/qwen9b-wt")
if str(_QWEN9B_WT) not in sys.path:
    sys.path.insert(0, str(_QWEN9B_WT))

import pytest
import torch

import ttnn

# ---------------------------------------------------------------------------
# Shared constants (same as test_64layer_per_layer_pcc.py)
# ---------------------------------------------------------------------------
_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_PROMPT_DIR = pathlib.Path("models/demos/llama3_70b_galaxy/demo/sample_prompts")
_CONTEXT_CACHE_DIR = pathlib.Path("models/tt_transformers/demo/context_cache")

# Number of layers in Qwen3.6-27B (same as TP=32 test)
_N_LAYERS = 64

# Override via QWEN36_PCC_T_PREFILL=4096 (default 128 for fast smoke)
_T_PREFILL = int(os.environ.get("QWEN36_PCC_T_PREFILL", "128"))

# TP=4 mesh shape for MESH_DEVICE=P150x4
_TP4_MESH_SHAPE = (1, 4)


# ---------------------------------------------------------------------------
# Mesh fixture for P150x4 (TT_VISIBLE_DEVICES=8,9,10,11)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tp4_mesh():
    """Open a (1,4) mesh on chips 8-11 for the TP=4 reference run."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*_TP4_MESH_SHAPE),
        l1_small_size=24576,
        num_command_queues=2,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Helpers — copied from test_64layer_per_layer_pcc.py so this test is
# self-contained and runnable without importing the TP=32 file.
# ---------------------------------------------------------------------------
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _load_prompt_for_isl(t_prefill: int) -> str:
    """Load the same prompt as test_64layer_per_layer_pcc.py for identical input."""
    import hashlib

    import requests

    if t_prefill < 1024:
        return "The capital of France is"
    k = t_prefill // 1024
    prompt_file = _PROMPT_DIR / f"input_data_long_{k}k.json"
    with open(prompt_file) as f:
        data = json.load(f)
    entry = data[0]
    prompt = entry["prompt"]
    context_url = entry.get("context")
    if context_url:
        max_length = max(entry.get("max_length") or 0, t_prefill * 6)
        cache_dir = _CONTEXT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()
        if cache_file.exists():
            context_text = cache_file.read_text()
        else:
            resp = requests.get(context_url, timeout=30)
            resp.raise_for_status()
            context_text = resp.text
            cache_file.write_text(context_text)
        if max_length:
            context_text = context_text[:max_length]
        prompt = "```" + context_text + "```\n\n" + prompt
    return prompt


def _cpu_reference_per_layer(state_dict_hf, x):
    """Return list of per-layer hidden states (N_LAYERS tensors), each [1, T, H].

    This is the same fp32 CPU golden as test_64layer_per_layer_pcc.py (minus the
    optional sub-block capture which isn't needed here).
    """
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
    return per_layer_hidden


def _load_state_dict_hf(snapshot_dir: pathlib.Path) -> dict:
    """Load the HF state dict (all language-model layers + embeddings)."""
    from safetensors.torch import load_file as load_st

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


def _gather_tp4_hidden(tt_tensor, mesh) -> torch.Tensor:
    """Gather a (1,4)-mesh hidden state [1,1,T,dim/4] → CPU float [1,T,dim].

    The TP=4 prefill produces residual stream sharded along the hidden dim
    (dim=3 in 4D: [batch=1, 1, T, dim_frac]).  ConcatMeshToTensor on dim=3
    concatenates the 4 shards back to the full hidden dim.
    """
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3),
    )
    # out: [4, 1, T, dim/4] after concat on dim=3 across the 4 mesh devices is
    # actually [1, 1, T, dim] (ConcatMeshToTensor stacks then concatenates along
    # the given dim of the per-device tensors).  Drop leading unit dims.
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    # out should now be [T, dim] or [1, T, dim] or [1, 1, T, dim]
    if out.dim() == 2:
        out = out.unsqueeze(0)  # [1, T, dim]
    elif out.dim() == 4:
        # [1, 1, T, dim] → [1, T, dim]
        out = out.reshape(1, out.shape[2], out.shape[3])
    return out.float()


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
@pytest.mark.hardware
def test_tp4_64_layer_per_layer_pcc(tp4_mesh):
    """Per-layer PCC sweep for the TP=4 reference at T=_T_PREFILL.

    Establishes whether TP=32 has already matched the TP=4 precision ceiling
    (~0.84) or whether there is still a fixable gap above it.
    """
    print("\n[tp4-per-layer] Loading HF state dict...")
    state_dict_hf = _load_state_dict_hf(_SNAPSHOT)

    # -----------------------------------------------------------------------
    # Tokenize the same prompt as the TP=32 test
    # -----------------------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = _load_prompt_for_isl(_T_PREFILL)
    preview = prompt if len(prompt) <= 200 else f"{prompt[:120]!r}...{prompt[-80:]!r}"
    print(f"[tp4-per-layer] prompt ({len(prompt)} chars): {preview}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = int(input_ids.shape[-1])
    if T_prompt > _T_PREFILL:
        input_ids = input_ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    input_ids_padded = torch.zeros(1, _T_PREFILL, dtype=input_ids.dtype)
    input_ids_padded[0, :T_prompt] = input_ids[0]
    print(f"[tp4-per-layer] T_prompt={T_prompt}  T_PREFILL={_T_PREFILL}")

    # -----------------------------------------------------------------------
    # CPU reference (fp32, same golden as TP=32 test)
    # -----------------------------------------------------------------------
    embed_w = state_dict_hf["model.language_model.embed_tokens.weight"].float()
    x_cpu_torch = embed_w[input_ids_padded[0]].unsqueeze(0)  # [1, T_PREFILL, dim]
    print(f"[tp4-per-layer] Computing CPU fp32 reference ({_N_LAYERS} layers)...")
    per_layer_ref = _cpu_reference_per_layer(state_dict_hf, x_cpu_torch)
    print(f"[tp4-per-layer] CPU reference done ({len(per_layer_ref)} layers)")

    # -----------------------------------------------------------------------
    # Load the TP=4 Qwen35Model from the qwen9b-wt tree.
    # Qwen35Model.from_pretrained reads HF_MODEL from the environment
    # (set to Qwen/Qwen3.6-27B in the environment setup above).
    # We cap n_layers so the test can also run on a partial model.
    # -----------------------------------------------------------------------
    print("[tp4-per-layer] Loading TP=4 Qwen35Model...")
    from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

    model = Qwen35Model.from_pretrained(
        tp4_mesh,
        max_batch_size=1,
        max_seq_len=_T_PREFILL + 128,  # a little headroom
        n_layers=_N_LAYERS,
    )
    print(f"[tp4-per-layer] TP=4 model loaded ({len(model.layers)} layers)")

    # -----------------------------------------------------------------------
    # Build cos/sin in the rope_tp format used by prefill_tp
    # -----------------------------------------------------------------------
    from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_prefill

    cos_tt, sin_tt = rot_mats_prefill(tp4_mesh, model.args.rope_head_dim, _T_PREFILL, model.args.rope_theta)

    # -----------------------------------------------------------------------
    # Send the embedded input to the mesh (replicated, then reshape to 4D)
    # -----------------------------------------------------------------------
    x_bf16 = x_cpu_torch.to(torch.bfloat16)  # [1, T_PREFILL, dim]
    x_tt = ttnn.from_torch(
        x_bf16.reshape(1, 1, _T_PREFILL, x_bf16.shape[-1]),  # [1,1,T,dim]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=tp4_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            tp4_mesh,
            dims=(None, 3),  # shard hidden dim across cols
            mesh_shape=model.args.cluster_shape,
        ),
    )

    # -----------------------------------------------------------------------
    # Per-layer instrumented forward loop
    # -----------------------------------------------------------------------
    print("[tp4-per-layer] Running TP=4 per-layer prefill (instrumented)...")
    x = x_tt
    per_layer_pcc: list[float] = []

    for i, layer in enumerate(model.layers):
        # Forward one layer.  The TP path uses prefill_tp's call signature.
        x = layer.forward(
            x,
            cos=cos_tt,
            sin=sin_tt,
            mode="prefill",
            chunk_size=model.args.gdn_chunk_size,
            valid_len=T_prompt,
        )

        # Gather the TP-sharded hidden state back to CPU [1, T_PREFILL, dim].
        tt_hidden_cpu = _gather_tp4_hidden(x, tp4_mesh)
        # Trim or pad to exactly T_PREFILL tokens.
        T_out = tt_hidden_cpu.shape[1]
        if T_out < _T_PREFILL:
            pad = torch.zeros(1, _T_PREFILL - T_out, tt_hidden_cpu.shape[2])
            tt_hidden_cpu = torch.cat([tt_hidden_cpu, pad], dim=1)
        elif T_out > _T_PREFILL:
            tt_hidden_cpu = tt_hidden_cpu[:, :_T_PREFILL, :]

        ref_hidden = per_layer_ref[i][:, :_T_PREFILL, :].float()

        pcc_full = _pcc(tt_hidden_cpu, ref_hidden)
        pcc_prompt = _pcc(tt_hidden_cpu[:, :T_prompt, :], ref_hidden[:, :T_prompt, :])
        pcc_last = _pcc(
            tt_hidden_cpu[:, T_prompt - 1 : T_prompt, :],
            ref_hidden[:, T_prompt - 1 : T_prompt, :],
        )
        per_layer_pcc.append(pcc_last)

        print(
            f"[tp4-per-layer] L{i:02d}: "
            f"PCC_full={pcc_full:.4f}  PCC_prompt={pcc_prompt:.4f}  PCC_last={pcc_last:.4f} | "
            f"tt_std={tt_hidden_cpu.std().item():.3f}  ref_std={ref_hidden.std().item():.3f}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n[tp4-per-layer] SUMMARY")
    fail_idx = next((i for i, p in enumerate(per_layer_pcc) if p < 0.99), None)
    print(f"  First layer with PCC<0.99: {fail_idx}")
    print(f"  Final layer PCC: {per_layer_pcc[-1]:.6f}")
    # Don't assert — this is diagnostic.
