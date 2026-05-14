# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-device-smoke — first real-mesh TtTransformer construction test.

Confirms the v2 tree is *constructible* on a live BH GLX 8×4 mesh. NO
forward pass — this just verifies the entire construction graph (config
+ embedding + decoder layer 0 (DeltaNet) + norm + LM head) builds without
KeyError, AssertionError, or device-side crash.

Failures here surface missing ``model_config`` keys, wrong tensor shapes
for HF weights, prefetcher/dispatcher mis-configurations, etc. — the
blockers that CPU mocks hide.

Use ``--noconftest`` invocation: this test opens its own mesh; the v2
``conftest.py``'s autouse ``ensure_devices`` fixture would otherwise
double-open.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_device_construct_smoke.py \\
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


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    """Load only the HF weights needed for 1-layer construction.

    Slim: embedding + norm + lm_head + one decoder layer.
    """
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
    state_dict: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                state_dict[k] = shard[k]
    return state_dict


@pytest.fixture(scope="module")
def bh_glx_mesh():
    """Open one BH GLX 8x4 mesh for the entire test module — sequential.

    Mirrors models/demos/qwen3_6_galaxy/demo/demo.py:120-127.
    """
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


@pytest.mark.hardware
def test_qwen36_v2_transformer_constructs_1_layer_on_device(bh_glx_mesh):
    """Build a 1-layer TtTransformer on real BH GLX 8x4. No forward."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    print("[smoke] loading HF state_dict (embedding + norm + lm_head + layer 0)...")
    state_dict = _load_state_dict_for_layer(_SNAPSHOT, layer_idx=0)
    print(f"[smoke]   loaded {len(state_dict)} weight tensors")

    print("[smoke] building TtQwen36ModelArgs on mesh...")
    args = TtQwen36ModelArgs(bh_glx_mesh)
    # Override n_layers so TtTransformer only builds 1 decoder block —
    # exercises layer 0 (DeltaNet) given the [lin,lin,lin,full]x16 pattern.
    args.n_layers = 1
    # Keep linear_attention_pattern aligned so layer 0 is still 'linear_attention'.
    args.linear_attention_pattern = (args.linear_attention_pattern or ["linear_attention"])[:1]
    print(f"[smoke]   args.is_qwen36={args.is_qwen36}, n_layers={args.n_layers}")
    print(f"[smoke]   layer 0 pattern={args.linear_attention_pattern[0]!r}")
    print(f"[smoke]   sub_core_grids={args.sub_core_grids}")

    print("[smoke] constructing TtTransformer...")
    # Upstream TtLlamaMLP / TtLlamaAttention build their tensor-cache file paths
    # via `weight_cache_path / "<name>"` (no None-guard), so we must pass a real
    # path. Use the same cache the model_args reports for bf8 weights.
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=bh_glx_mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )

    print(f"[smoke] SUCCESS — model has {len(model.layers)} decoder layer(s)")
    print(f"[smoke]   layer 0 is_linear_attention_layer={getattr(model.layers[0], 'is_linear_attention_layer', None)}")
    # Pacify the linter — assert SOMETHING about the constructed model.
    assert len(model.layers) == 1
    assert getattr(model.layers[0], "is_linear_attention_layer", False) is True
