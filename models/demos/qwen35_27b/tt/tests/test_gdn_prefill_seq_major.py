# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Correctness test for gdn_prefill_fused_seq_major.

Compares the new kernel pair (fused-norm compute + seq-major writer) against
the existing gdn_prefill_fused output post-applied with ttnn.rms_norm and the
reshape -> permute -> reshape chain that gdn.py:forward_prefill currently does
between the kernel and the silu*z multiply.

Pass criteria: PCC >= 0.999 on output, PCC >= 0.999 on final state.

Run:
    export TT_METAL_HOME=$(pwd)
    export HF_MODEL=/local/ttuser/atupe/Qwen27bFP8
    pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py -v -s
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
    r = ref.float().flatten()
    t = test.float().flatten()
    if r.numel() == 0:
        return 1.0
    vr = r - r.mean()
    vt = t - t.mean()
    num = (vr * vt).sum()
    den = (vr.norm() * vt.norm()) + 1e-12
    return (num / den).item()


def _load_gdn_layer(mesh_device):
    model_path = _get_model_path()
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=32,
        max_seq_len=2048,
        dtype=ttnn.bfloat8_b,
        n_layers=3,
    )
    args = model.args
    gdn_layer_idx = None
    for i in range(args.n_layers):
        if args.layer_types[i] == "linear_attention":
            gdn_layer_idx = i
            break
    assert gdn_layer_idx is not None, "No GDN layer found in first 3 layers"
    gdn = model.layers[gdn_layer_idx].attention
    tw = gdn.tw
    return model, args, gdn, tw


def _reference_old_path_with_norm_and_permute(
    conv_out_3d, a_3d, b_3d, gdn, tw, mesh_device, num_pairs, num_tokens, Nv_TP, Nk_TP, repeat_factor, key_dim_tp
):
    """Drives the existing gdn_prefill_fused, then applies the post-kernel
    rms_norm + reshape + permute + reshape that gdn.py currently performs.
    Returns the final dense tensor in shape [1, 1, num_tokens, Nv_TP*Dv]."""
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused

    Dv = gdn.Dv
    rec_states = _to_mesh(torch.zeros(num_pairs, gdn.Dk, Dv, dtype=torch.bfloat16), mesh_device)
    flat_output = _to_mesh(torch.zeros(num_pairs * num_tokens, 1, Dv, dtype=torch.bfloat16), mesh_device)

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
        rec_states,
        flat_output,
        num_pairs=num_pairs,
        num_tokens=num_tokens,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )

    out_n = ttnn.rms_norm(flat_output, weight=tw["norm_w"], epsilon=1e-6)
    ttnn.deallocate(flat_output)
    out_4d = ttnn.reshape(out_n, (1, num_pairs, num_tokens, Dv))
    ttnn.deallocate(out_n)
    out_4d = ttnn.permute(out_4d, (0, 2, 1, 3))
    out_dense = ttnn.reshape(out_4d, (1, 1, num_tokens, num_pairs * Dv))
    ttnn.deallocate(out_4d)

    ref_dense = ttnn.to_torch(out_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    ref_state = ttnn.to_torch(rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:num_pairs].float()

    ttnn.deallocate(out_dense)
    ttnn.deallocate(rec_states)
    return ref_dense, ref_state


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
@pytest.mark.parametrize("num_tokens", [64, 96, 2048])
def test_gdn_prefill_seq_major_correctness(mesh_device, reset_seeds, ensure_gc, num_tokens):
    """gdn_prefill_fused_seq_major must match (old kernel + ttnn.rms_norm + reshape/permute/reshape).

    num_tokens parametrization covers:
      - 64: small tile-aligned baseline.
      - 96: another tile-aligned case (catches off-by-one in the writer's L1 repacking).
      - 2048: production chunked-prefill chunk size.
    """
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    # Lazy import — at Task 1, this function does not exist yet, so the test
    # fails at import. That is the "red" stage. Tasks 2+ make this import
    # succeed and the assertion pass.
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused_seq_major

    _, _args, gdn, tw = _load_gdn_layer(mesh_device)
    Nv_TP = gdn.Nv_TP
    Nk_TP = gdn.Nk_TP
    Dv = gdn.Dv
    qkv_dim_tp = gdn.qkv_dim_tp
    key_dim_tp = gdn.key_dim_tp
    num_pairs = 1 * Nv_TP
    repeat_factor = Nv_TP // Nk_TP

    logger.info(f"test_gdn_prefill_seq_major: N={num_tokens}, Nv_TP={Nv_TP}, Dv={Dv}")

    torch.manual_seed(42)
    conv_out_all = _to_mesh(torch.randn(1, 1, num_tokens, qkv_dim_tp, dtype=torch.bfloat16) * 0.1, mesh_device)
    a_all = _to_mesh(torch.randn(1, 1, num_tokens, Nv_TP, dtype=torch.bfloat16) * 0.1, mesh_device)
    b_all = _to_mesh(torch.randn(1, 1, num_tokens, Nv_TP, dtype=torch.bfloat16) * 0.1, mesh_device)

    conv_out_3d_a = _unshard(ttnn.reshape(conv_out_all, (1, num_tokens, qkv_dim_tp)))
    a_3d_a = _unshard(ttnn.reshape(a_all, (1, num_tokens, Nv_TP)))
    b_3d_a = _unshard(ttnn.reshape(b_all, (1, num_tokens, Nv_TP)))

    ref_dense, ref_state = _reference_old_path_with_norm_and_permute(
        conv_out_3d_a,
        a_3d_a,
        b_3d_a,
        gdn,
        tw,
        mesh_device,
        num_pairs=num_pairs,
        num_tokens=num_tokens,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )

    ttnn.deallocate(conv_out_3d_a)
    ttnn.deallocate(a_3d_a)
    ttnn.deallocate(b_3d_a)

    # --- New kernel path ---
    conv_out_3d_b = _unshard(ttnn.reshape(conv_out_all, (1, num_tokens, qkv_dim_tp)))
    a_3d_b = _unshard(ttnn.reshape(a_all, (1, num_tokens, Nv_TP)))
    b_3d_b = _unshard(ttnn.reshape(b_all, (1, num_tokens, Nv_TP)))

    test_rec_states = _to_mesh(torch.zeros(num_pairs, gdn.Dk, Dv, dtype=torch.bfloat16), mesh_device)
    test_output = _to_mesh(torch.zeros(1, 1, num_tokens, num_pairs * Dv, dtype=torch.bfloat16), mesh_device)

    gdn_prefill_fused_seq_major(
        conv_out_3d_b,
        a_3d_b,
        b_3d_b,
        gdn.neg_exp_A,
        tw["dt_bias"],
        tw["norm_w"],
        gdn.scale_tt,
        gdn.rms_scale_tt,
        gdn.rms_eps_tt,
        test_rec_states,
        test_output,
        num_pairs=num_pairs,
        num_tokens=num_tokens,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )

    test_dense = ttnn.to_torch(test_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    test_state = ttnn.to_torch(test_rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :num_pairs
    ].float()

    ttnn.deallocate(conv_out_all)
    ttnn.deallocate(a_all)
    ttnn.deallocate(b_all)
    ttnn.deallocate(conv_out_3d_b)
    ttnn.deallocate(a_3d_b)
    ttnn.deallocate(b_3d_b)
    ttnn.deallocate(test_rec_states)
    ttnn.deallocate(test_output)

    output_pcc = _compute_pcc(ref_dense, test_dense)
    state_pcc = _compute_pcc(ref_state, test_state)

    logger.info(f"  num_tokens={num_tokens} Output PCC: {output_pcc:.6f}")
    logger.info(f"  num_tokens={num_tokens} State  PCC: {state_pcc:.6f}")

    assert output_pcc > 0.999, f"Output PCC {output_pcc:.6f} < 0.999 (num_tokens={num_tokens})"
    assert state_pcc > 0.999, f"State PCC {state_pcc:.6f} < 0.999 (num_tokens={num_tokens})"
