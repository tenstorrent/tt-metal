# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Correctness test for gdn_prefill_fused kernel.

Verifies that a single prefill kernel dispatch over N tokens produces
identical output and final state to N sequential gdn_full_fused_inplace
calls (the proven decode kernel).

Run:
    export TT_METAL_HOME=$(pwd)
    export HF_MODEL=/local/ttuser/atupe/Qwen27bFP8
    pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_fused.py -v -s
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _unshard(t):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def _to_mesh(t, mesh_device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _compute_pcc(ref, test):
    """Pearson correlation between two tensors."""
    r = ref.float().flatten()
    t = test.float().flatten()
    if r.numel() == 0:
        return 1.0
    vr = r - r.mean()
    vt = t - t.mean()
    num = (vr * vt).sum()
    den = (vr.norm() * vt.norm()) + 1e-12
    return (num / den).item()


def _run_per_token_reference(
    conv_out_all,
    a_all,
    b_all,
    gdn,
    tw,
    mesh_device,
    num_tokens,
    num_pairs,
    qkv_dim_tp,
    Nv_TP,
    Nk_TP,
    repeat_factor,
    key_dim_tp,
):
    """Run per-token decode kernel as correctness reference.

    Returns (all_outputs [num_pairs, N, Dv], final_state [num_pairs, Dk, Dv]).
    """
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace

    Dk = gdn.Dk
    Dv = gdn.Dv
    B = 1

    rec_states = _to_mesh(torch.zeros(num_pairs, Dk, Dv, dtype=torch.bfloat16), mesh_device)
    fused_output = _to_mesh(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16), mesh_device)

    ref_outputs = []
    for t in range(num_tokens):
        # Slice single token
        conv_t = ttnn.slice(conv_out_all, (0, 0, t, 0), (1, 1, t + 1, qkv_dim_tp))
        conv_t = ttnn.reshape(conv_t, (1, B, qkv_dim_tp))
        conv_t = _unshard(conv_t)

        a_t = ttnn.slice(a_all, (0, 0, t, 0), (1, 1, t + 1, Nv_TP))
        a_t = ttnn.reshape(a_t, (1, B, Nv_TP))
        a_t = _unshard(a_t)

        b_t = ttnn.slice(b_all, (0, 0, t, 0), (1, 1, t + 1, Nv_TP))
        b_t = ttnn.reshape(b_t, (1, B, Nv_TP))
        b_t = _unshard(b_t)

        gdn_full_fused_inplace(
            conv_t,
            a_t,
            b_t,
            gdn.neg_exp_A,
            tw["dt_bias"],
            tw["norm_w"],
            gdn.scale_tt,
            gdn.rms_scale_tt,
            gdn.rms_eps_tt,
            rec_states,
            fused_output,
            num_pairs=num_pairs,
            num_cores=min(96, num_pairs),
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )

        # Read output for this token
        out_t = ttnn.to_torch(fused_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        # Take first device's data: [num_pairs, 1, Dv]
        ref_outputs.append(out_t[:num_pairs].clone())

        ttnn.deallocate(conv_t)
        ttnn.deallocate(a_t)
        ttnn.deallocate(b_t)

    # Stack all token outputs: [num_pairs, N, Dv]
    all_outputs = torch.cat(ref_outputs, dim=1).float()

    # Read final state
    final_state = ttnn.to_torch(rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :num_pairs
    ].float()

    ttnn.deallocate(rec_states)
    ttnn.deallocate(fused_output)

    return all_outputs, final_state


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("num_tokens", [32, 64])
def test_gdn_prefill_fused_correctness(mesh_device, reset_seeds, ensure_gc, num_tokens):
    """gdn_prefill_fused must match per-token decode kernel output and state."""

    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    # Load model with just enough layers to get a GDN layer
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
        n_layers=3,
    )
    args = model.args

    # Find first GDN layer
    gdn_layer_idx = None
    for i in range(args.n_layers):
        if args.layer_types[i] == "linear_attention":
            gdn_layer_idx = i
            break
    assert gdn_layer_idx is not None, "No GDN layer found in first 3 layers"
    gdn = model.layers[gdn_layer_idx].attention
    tw = gdn.tw

    B = 1
    N = num_tokens
    Nv_TP = gdn.Nv_TP
    Nk_TP = gdn.Nk_TP
    Dk = gdn.Dk
    Dv = gdn.Dv
    qkv_dim_tp = gdn.qkv_dim_tp
    key_dim_tp = gdn.key_dim_tp
    num_pairs = B * Nv_TP
    repeat_factor = Nv_TP // Nk_TP
    dim = args.dim

    logger.info(f"Testing gdn_prefill_fused: N={N}, Nv_TP={Nv_TP}, Nk_TP={Nk_TP}, Dk={Dk}, Dv={Dv}")
    logger.info(f"  qkv_dim_tp={qkv_dim_tp}, key_dim_tp={key_dim_tp}, num_pairs={num_pairs}")

    # ---- Create random input tensors (post-conv1d + SiLU) ----
    # Use small random values to avoid numerical issues
    torch.manual_seed(42)
    conv_out_all = _to_mesh(
        torch.randn(1, 1, N, qkv_dim_tp, dtype=torch.bfloat16) * 0.1,
        mesh_device,
    )
    a_all = _to_mesh(
        torch.randn(1, 1, N, Nv_TP, dtype=torch.bfloat16) * 0.1,
        mesh_device,
    )
    b_all = _to_mesh(
        torch.randn(1, 1, N, Nv_TP, dtype=torch.bfloat16) * 0.1,
        mesh_device,
    )

    # ================================================================
    # REFERENCE: per-token decode kernel (N dispatches)
    # ================================================================
    logger.info(f"Running reference: {N} x gdn_full_fused_inplace...")
    ref_outputs, ref_final_state = _run_per_token_reference(
        conv_out_all,
        a_all,
        b_all,
        gdn,
        tw,
        mesh_device,
        num_tokens=N,
        num_pairs=num_pairs,
        qkv_dim_tp=qkv_dim_tp,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    logger.info(f"  Reference output shape: {ref_outputs.shape}")
    logger.info(f"  Reference final state norm: {ref_final_state.norm():.4f}")

    # ================================================================
    # TEST: gdn_prefill_fused (1 dispatch for all N tokens)
    # ================================================================
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused

    logger.info(f"Running test: gdn_prefill_fused with N={N} tokens...")

    # Prepare inputs: reshape to 3D [1, N, dim]
    conv_out_3d = ttnn.reshape(conv_out_all, (1, N, qkv_dim_tp))
    conv_out_3d = _unshard(conv_out_3d)
    a_3d = ttnn.reshape(a_all, (1, N, Nv_TP))
    a_3d = _unshard(a_3d)
    b_3d = ttnn.reshape(b_all, (1, N, Nv_TP))
    b_3d = _unshard(b_3d)

    # Fresh state and output buffer (flat layout: [num_pairs * N, 1, Dv])
    test_rec_states = _to_mesh(torch.zeros(num_pairs, Dk, Dv, dtype=torch.bfloat16), mesh_device)
    test_output = _to_mesh(torch.zeros(num_pairs * N, 1, Dv, dtype=torch.bfloat16), mesh_device)

    gdn_prefill_fused(
        conv_out_3d,
        a_3d,
        b_3d,
        gdn.neg_exp_A,
        tw["dt_bias"],
        tw["norm_w"],
        gdn.scale_tt,
        gdn.rms_scale_tt,
        gdn.rms_eps_tt,
        test_rec_states,
        test_output,
        num_pairs=num_pairs,
        num_tokens=N,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )

    # Read results — output is flat [num_pairs * N, 1, Dv], reshape to [num_pairs, N, Dv]
    test_out_flat = ttnn.to_torch(test_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        : num_pairs * N
    ].float()
    test_outputs = test_out_flat.reshape(num_pairs, N, Dv)
    test_final_state = ttnn.to_torch(test_rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :num_pairs
    ].float()

    # ================================================================
    # COMPARE: output and state
    # ================================================================
    output_pcc = _compute_pcc(ref_outputs, test_outputs)
    state_pcc = _compute_pcc(ref_final_state, test_final_state)

    logger.info(f"  Output PCC: {output_pcc:.6f}")
    logger.info(f"  State PCC:  {state_pcc:.6f}")
    logger.info(f"  Test output shape: {test_outputs.shape}")
    logger.info(f"  Test final state norm: {test_final_state.norm():.4f}")

    # Cleanup
    ttnn.deallocate(conv_out_all)
    ttnn.deallocate(a_all)
    ttnn.deallocate(b_all)
    ttnn.deallocate(conv_out_3d)
    ttnn.deallocate(a_3d)
    ttnn.deallocate(b_3d)
    ttnn.deallocate(test_rec_states)
    ttnn.deallocate(test_output)

    # Assert correctness
    assert output_pcc > 0.99, f"Output PCC {output_pcc:.6f} < 0.99 — prefill kernel output diverges from reference"
    assert state_pcc > 0.99, f"State PCC {state_pcc:.6f} < 0.99 — prefill kernel final state diverges from reference"

    logger.info(
        f"PASS: gdn_prefill_fused matches decode kernel (output PCC={output_pcc:.6f}, state PCC={state_pcc:.6f})"
    )
