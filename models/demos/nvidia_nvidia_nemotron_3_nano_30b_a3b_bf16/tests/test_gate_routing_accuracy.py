# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify MoE gate routing accuracy: device bfloat16 vs CPU float32 reference.

Compares two gate implementations on the same hidden state:
  - current device gate  : bfloat16 linear + sigmoid + topk (what we run)
  - reference CPU gate   : float32, exactly matching HF NemotronHTopkRouter

With n_group=1, topk_group=1 the HF group-mask step is trivial (all experts
in one group, always selected), so the HF algorithm reduces to:

    logits  = hidden @ weight.T            # float32
    scores  = sigmoid(logits)
    indices = topk(scores + bias, k=6)
    weights = gather(scores, indices)
    weights = weights / sum(weights) * 2.5

If device and CPU pick the same 6 experts, match=100%.  A match rate near
random (6/128 ≈ 5%) confirms the bfloat16 saturation bug.

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
    pytest models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_gate_routing_accuracy.py -v -s
"""

import os
import sys

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
for p in (
    f"{os.environ['TT_METAL_HOME']}/ttnn",
    f"{os.environ['TT_METAL_HOME']}/tools",
    os.environ["TT_METAL_HOME"],
):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch
import torch.nn.functional as F

import ttnn

N_EXPERTS = 128
N_TOP_K = 6
HIDDEN_SIZE = 2688
ROUTED_SCALING_FACTOR = 2.5


def _ref_gate_f32(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """Exact HF NemotronHTopkRouter.forward with n_group=1, topk_group=1.

    Returns (indices [T, 6], weights [T, 6]) in float32.
    """
    T = hidden.shape[0]
    h = hidden.reshape(T, HIDDEN_SIZE).float()
    logits = F.linear(h, weight.float())  # [T, 128]
    scores = torch.sigmoid(logits)  # [T, 128]
    scores_biased = scores + bias.float()  # [T, 128]
    indices = torch.topk(scores_biased, k=N_TOP_K, dim=-1, sorted=False)[1]  # [T, 6]
    weights = scores.gather(1, indices)  # [T, 6]
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * ROUTED_SCALING_FACTOR
    return indices, weights


@pytest.fixture(scope="module")
def mesh_device():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    dev = open_device_tp4()
    yield dev
    close_device_tp4(dev)


@pytest.fixture(scope="module")
def weight_cache():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache

    return WeightCache()


@pytest.mark.timeout(0)
def test_gate_routing_accuracy(mesh_device, weight_cache):
    """Run the gate for 3 E-layers on synthetic hidden states; compare device vs CPU."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.moe_gate import moe_gate_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _R

    wc = weight_cache
    # E-layers: indices 3, 6, 10, ...  First three are enough to establish the pattern.
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import PATTERN

    e_layer_indices = [i for i, t in enumerate(PATTERN) if t == "E"][:3]

    print(f"\n{'='*65}")
    print(f"Gate routing accuracy: device bfloat16 vs CPU float32")
    print(f"{'='*65}")

    all_match = []
    for li in e_layer_indices:
        p = f"backbone.layers.{li}"
        gate_w = wc[f"{p}.mixer.gate.weight"]  # [128, 2688] float32
        gate_b = wc[f"{p}.mixer.gate.e_score_correction_bias"]  # [128] float32

        # Synthetic hidden state: sample from N(0,1) scaled to realistic hidden-state magnitude
        # Real hidden states at these layers are ~O(1-5) after norm.
        torch.manual_seed(li)
        hidden_cpu = torch.randn(1, HIDDEN_SIZE) * 2.0

        # --- CPU reference (float32) ---
        ref_idx, ref_wts = _ref_gate_f32(hidden_cpu, gate_w, gate_b)  # [1,6], [1,6]

        # --- Device gate (bfloat16) ---
        hidden_tt = ttnn.from_torch(
            hidden_cpu.bfloat16().unsqueeze(0),  # [1,1,2688]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=_R(mesh_device),
        )
        dense_tt = moe_gate_forward(mesh_device, ttnn.reshape(hidden_tt, [1, HIDDEN_SIZE]), gate_w, gate_b)
        # dense_tt: [1,1,1,128]  bfloat16, nonzero at active experts
        dense_cpu = ttnn.to_torch(dense_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
            0, 0, 0, :
        ]  # first shard [128]
        dev_idx = set(dense_cpu.nonzero(as_tuple=True)[0].tolist())

        ref_set = set(ref_idx[0].tolist())
        match = len(ref_set & dev_idx)
        all_match.append(match)

        print(f"\nLayer {li:2d}: CPU f32 = {sorted(ref_set)}")
        print(f"          dev bf16 = {sorted(dev_idx)}")
        print(f"          match    = {match}/{N_TOP_K}  ({100*match/N_TOP_K:.0f}%)")

        # Also show why bf16 saturates: logit range and unique sigmoid values
        logits_f32 = F.linear(hidden_cpu.float(), gate_w.float())
        logits_bf16 = F.linear(hidden_cpu.bfloat16().float(), gate_w.bfloat16().float())
        scores_f32 = torch.sigmoid(logits_f32)
        scores_bf16 = torch.sigmoid(logits_bf16.bfloat16())
        print(
            f"          logit range (f32): [{logits_f32.min():.2f}, {logits_f32.max():.2f}]  "
            f"std={logits_f32.std():.2f}"
        )
        n_unique = scores_bf16.unique().numel()
        print(
            f"          sigmoid unique vals (bf16): {n_unique}/128  "
            f"(all-same: {(scores_bf16 == scores_bf16[0,0]).all().item()})"
        )

    avg = sum(all_match) / (len(all_match) * N_TOP_K)
    print(f"\n{'='*65}")
    print(f"Average index-match rate: {avg:.1%}")
    if avg < 0.3:
        print("CONFIRMED: bfloat16 gate routing is severely degraded (< 30% match).")
    elif avg > 0.9:
        print("Gate routing is acceptable (> 90% match).")
    else:
        print(f"Partial degradation: {avg:.1%} match rate.")
    print(f"{'='*65}")
