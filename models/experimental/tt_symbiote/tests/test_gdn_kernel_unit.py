# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test isolating the GDN kernel from the full Qwen pipeline.

Compares `gdn_recurrent_step` (one decode step on the mesh kernel) against
`torch_recurrent_gated_delta_rule(...seq_len=1)` from transformers on the same
random inputs. If the kernel is correct, PCC should be very close to 1.0 and
max-abs-error small relative to the output magnitude.

Run with:
  export TTNN_GDN_KERNEL=1
  export MESH_DEVICE=QB2
  pytest models/experimental/tt_symbiote/tests/test_gdn_kernel_unit.py -s
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_recurrent_gated_delta_rule,
)


MESH_DEVICE_MAP = {
    "T3K": (1, 8),
    "QB2": (1, 4),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_gdn_kernel_matches_torch_reference(mesh_device):
    """One decode step: kernel output vs torch reference."""
    from models.experimental.tt_symbiote.modules.gdn_kernel import (
        gdn_recurrent_step,
        NUM_V_HEADS,
        D_K,
        D_V_DIM,
        BF16,
    )

    n_devices = mesh_device.get_num_devices()
    if n_devices not in (4, 8):
        pytest.skip(f"Kernel requires 4- or 8-device mesh; got {n_devices}")

    torch.manual_seed(0)
    H = NUM_V_HEADS  # 32

    # Inputs as the Qwen pipeline produces them: [batch, seq, num_heads, head_dim].
    query = torch.randn(1, 1, H, D_K, dtype=BF16)
    key = torch.randn(1, 1, H, D_K, dtype=BF16)
    value = torch.randn(1, 1, H, D_V_DIM, dtype=BF16)
    g = torch.randn(1, 1, H, dtype=torch.float32) * 0.1  # small log-decay
    beta = torch.rand(1, 1, H, dtype=BF16)  # in [0,1]
    initial_state = torch.randn(1, H, D_K, D_V_DIM, dtype=BF16) * 0.01

    # --- Torch reference: l2norm + scale + recurrent rule (matches transformers).
    ref_out, ref_state = torch_recurrent_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # ref_out: [1, 1, H, D_V], ref_state: [1, H, D_K, D_V]

    # --- Kernel path: emulate exactly what qwen_attention.py does before calling the kernel.
    # L2-norm Q/K and apply the `query * 1/sqrt(D_K)` scaling that
    # torch_recurrent_gated_delta_rule applies internally.
    q_scale = 1.0 / (query.shape[-1] ** 0.5)
    q_norm = (F.normalize(query.float(), p=2, dim=-1, eps=1e-6) * q_scale).to(query.dtype)
    k_norm = F.normalize(key.float(), p=2, dim=-1, eps=1e-6).to(key.dtype)

    kernel_out, kernel_state = gdn_recurrent_step(
        mesh_device=mesh_device,
        query=q_norm,
        key=k_norm,
        value=value,
        g=g,
        beta=beta,
        recurrent_state=initial_state.clone(),
        return_state_as_ttnn=False,
    )

    # --- Compare.
    print(f"\nref_out shape={ref_out.shape}, kernel_out shape={kernel_out.shape}")
    print(f"ref_out  abs mean={ref_out.float().abs().mean():.4e}  max={ref_out.float().abs().max():.4e}")
    print(f"kernel_out abs mean={kernel_out.float().abs().mean():.4e}  max={kernel_out.float().abs().max():.4e}")

    out_pcc = _pcc(kernel_out, ref_out)
    out_max_abs_err = (kernel_out.float() - ref_out.float()).abs().max().item()
    out_rel_err = out_max_abs_err / max(ref_out.float().abs().max().item(), 1e-6)
    print(f"\nOUTPUT  pcc={out_pcc:.6f}  max_abs_err={out_max_abs_err:.4e}  rel_err={out_rel_err:.4e}")

    state_pcc = _pcc(kernel_state, ref_state)
    state_max_abs_err = (kernel_state.float() - ref_state.float()).abs().max().item()
    state_rel_err = state_max_abs_err / max(ref_state.float().abs().max().item(), 1e-6)
    print(f"STATE   pcc={state_pcc:.6f}  max_abs_err={state_max_abs_err:.4e}  rel_err={state_rel_err:.4e}")

    # Show a sliver so we can eyeball whether the kernel got the magnitude/sign right.
    print(f"\nref_out[0,0,0,:6]    = {ref_out[0,0,0,:6].float().tolist()}")
    print(f"kernel_out[0,0,0,:6] = {kernel_out[0,0,0,:6].float().tolist()}")
    print(f"\nref_state[0,0,0,:6]    = {ref_state[0,0,0,:6].float().tolist()}")
    print(f"kernel_state[0,0,0,:6] = {kernel_state[0,0,0,:6].float().tolist()}")

    # Loose bf16 thresholds; tighten once correctness is established.
    assert out_pcc > 0.99, f"output PCC too low: {out_pcc}"
    assert state_pcc > 0.99, f"state PCC too low: {state_pcc}"


@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_gdn_kernel_multistep_matches_torch_reference(mesh_device):
    """N decode steps with state carryover. Mirrors what end-to-end decode does:
    state ping-pongs across iterations, with the prev kernel output state fed
    back as the next step's input state."""
    from models.experimental.tt_symbiote.modules.gdn_kernel import (
        gdn_recurrent_step,
        NUM_V_HEADS,
        D_K,
        D_V_DIM,
        BF16,
    )

    n_devices = mesh_device.get_num_devices()
    if n_devices not in (4, 8):
        pytest.skip(f"Kernel requires 4- or 8-device mesh; got {n_devices}")

    torch.manual_seed(0)
    H = NUM_V_HEADS
    NUM_STEPS = 8

    # Pre-generate per-step inputs so torch and kernel see identical sequences.
    queries = [torch.randn(1, 1, H, D_K, dtype=BF16) for _ in range(NUM_STEPS)]
    keys = [torch.randn(1, 1, H, D_K, dtype=BF16) for _ in range(NUM_STEPS)]
    values = [torch.randn(1, 1, H, D_V_DIM, dtype=BF16) for _ in range(NUM_STEPS)]
    gs = [torch.randn(1, 1, H, dtype=torch.float32) * 0.1 for _ in range(NUM_STEPS)]
    betas = [torch.rand(1, 1, H, dtype=BF16) for _ in range(NUM_STEPS)]
    init_state = torch.randn(1, H, D_K, D_V_DIM, dtype=BF16) * 0.01

    # --- Torch reference, one step at a time, carrying state.
    ref_state = init_state.clone()
    ref_outs = []
    for i in range(NUM_STEPS):
        ref_out, ref_state = torch_recurrent_gated_delta_rule(
            query=queries[i],
            key=keys[i],
            value=values[i],
            g=gs[i],
            beta=betas[i],
            initial_state=ref_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        ref_outs.append(ref_out)

    # --- Kernel path, one step at a time, with my qwen_attention.py-equivalent preprocessing.
    kernel_state = init_state.clone()
    kernel_outs = []
    for i in range(NUM_STEPS):
        q_scale = 1.0 / (queries[i].shape[-1] ** 0.5)
        q_norm = (F.normalize(queries[i].float(), p=2, dim=-1, eps=1e-6) * q_scale).to(queries[i].dtype)
        k_norm = F.normalize(keys[i].float(), p=2, dim=-1, eps=1e-6).to(keys[i].dtype)
        kernel_out, kernel_state = gdn_recurrent_step(
            mesh_device=mesh_device,
            query=q_norm,
            key=k_norm,
            value=values[i],
            g=gs[i],
            beta=betas[i],
            recurrent_state=kernel_state,
            return_state_as_ttnn=False,
        )
        kernel_outs.append(kernel_out)

    # Per-step divergence
    print(f"\nPer-step PCC and max-abs-err:")
    print(f"{'step':>4}  {'out_pcc':>10}  {'out_maxerr':>12}  {'state_pcc':>10}  {'state_maxerr':>12}")
    for i in range(NUM_STEPS):
        ko, ro = kernel_outs[i], ref_outs[i]
        op = _pcc(ko, ro)
        oe = (ko.float() - ro.float()).abs().max().item()
        # State is only checked at the end, but show running state pcc too via the running kernel_state if needed
        print(f"{i:>4}  {op:>10.6f}  {oe:>12.4e}")

    final_state_pcc = _pcc(kernel_state, ref_state)
    final_state_maxerr = (kernel_state.float() - ref_state.float()).abs().max().item()
    print(f"\nFINAL STATE  pcc={final_state_pcc:.6f}  max_abs_err={final_state_maxerr:.4e}")

    # PCC should stay close to 1 across all steps if the integration is correct.
    for i, (ko, ro) in enumerate(zip(kernel_outs, ref_outs)):
        op = _pcc(ko, ro)
        assert op > 0.99, f"step {i}: output PCC degraded to {op}"
    assert final_state_pcc > 0.99, f"final state PCC degraded to {final_state_pcc}"
