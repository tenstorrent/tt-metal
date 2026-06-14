# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generation bringup test for NemotronH-30B on QB device 0.

Runs the first NUM_TEST_LAYERS layers (MEMEM*) using our TTNN components and
compares hidden-state PCC against the pure-PyTorch reference implementation.

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
    pytest models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_generation.py -v -s
"""

import os
import sys

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch
import torch.nn.functional as F

import ttnn

DEMO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SNAP = (
    "/home/ttuser/.cache/huggingface/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/"
    "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# Test first 6 layers (MEMEM*): covers all 3 layer types
NUM_TEST_LAYERS = 6
PCC_THRESHOLD = 0.99


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    denom = torch.sqrt((a**2).sum() * (b**2).sum())
    return ((a * b).sum() / denom).item() if denom.item() != 0.0 else 1.0


@pytest.fixture(scope="module")
def mesh_device():
    from skills.orchestrator.lib.device import prepare_device

    prepare_device("qb")
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def weight_cache():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache

    return WeightCache()


def _ref_forward_n_layers(input_ids: torch.Tensor, num_layers: int, wc) -> torch.Tensor:
    """Pure-PyTorch reference forward for the first num_layers layers."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.reference.functional import (
        dense_attention,
        layer_norm,
        mamba2_layer,
        moe_experts,
        moe_gate,
        shared_expert,
    )

    B, S = input_ids.shape
    emb_w = wc["backbone.embeddings.weight"]
    h = F.embedding(input_ids, emb_w)  # [B, S, 2688]

    for li in range(num_layers):
        lt = PATTERN[li]
        p = f"backbone.layers.{li}"

        if lt == "M":
            h = mamba2_layer(
                hidden_states=h,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )
        elif lt == "E":
            residual = h
            normed = layer_norm(h, wc[f"{p}.norm.weight"])
            flat = normed.reshape(B * S, -1)
            topk_idx, topk_wts = moe_gate(
                flat,
                wc[f"{p}.mixer.gate.weight"],
                wc[f"{p}.mixer.gate.e_score_correction_bias"],
            )
            eu = [wc[f"{p}.mixer.experts.{e}.up_proj.weight"] for e in range(128)]
            ed = [wc[f"{p}.mixer.experts.{e}.down_proj.weight"] for e in range(128)]
            ex_out = moe_experts(flat, topk_idx, topk_wts, eu, ed).reshape(B, S, -1)
            sh_out = shared_expert(
                normed, wc[f"{p}.mixer.shared_experts.up_proj.weight"], wc[f"{p}.mixer.shared_experts.down_proj.weight"]
            )
            h = (residual + ex_out + sh_out).bfloat16()
        else:  # '*'
            h = dense_attention(
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )
    return h


def test_generation_first_6_layers_pcc(mesh_device, weight_cache):
    """Run first 6 layers (MEMEM*) on TTNN and compare hidden-state PCC to reference."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.dense_attention import dense_attention_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.embedding import embedding_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_layer import mamba2_layer_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import _moe_layer_forward

    torch.manual_seed(0)
    input_ids = torch.randint(0, 131072, (1, 8), dtype=torch.long)

    # Reference forward
    print(f"\nRunning reference forward for first {NUM_TEST_LAYERS} layers ({PATTERN[:NUM_TEST_LAYERS]})...")
    ref_h = _ref_forward_n_layers(input_ids, NUM_TEST_LAYERS, weight_cache)

    # TTNN partial forward (hidden state only — no lm_head)
    print(f"Running TTNN forward for first {NUM_TEST_LAYERS} layers...")
    wc = weight_cache
    h = embedding_forward(mesh_device, input_ids, wc["backbone.embeddings.weight"])
    for li in range(NUM_TEST_LAYERS):
        lt = PATTERN[li]
        p = f"backbone.layers.{li}"
        if lt == "M":
            h = mamba2_layer_forward(
                mesh_device,
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )
        elif lt == "E":
            h = _moe_layer_forward(mesh_device, h, li, wc)
        else:
            h = dense_attention_forward(
                mesh_device,
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )

    score = pcc(h, ref_h)
    print(f"\nHidden-state PCC after {NUM_TEST_LAYERS} layers ({PATTERN[:NUM_TEST_LAYERS]}): {score:.6f}")
    assert score >= PCC_THRESHOLD, f"PCC {score:.6f} < {PCC_THRESHOLD}"
