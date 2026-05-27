# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic test: compare intermediate values between forward_prefill (parallel
chunk path) and the per-token sequential reference to find where divergence occurs.

Compares at 4 checkpoints:
  A. conv1d output (post-SiLU)
  B. Q/K/V tensors fed into the scan (post L2-norm, post head expansion)
  C. beta and g tensors (decay/gate values)
  D. scan output (before norm/gate/output-proj)

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=P150x4 HF_MODEL=<path> \\
        pytest models/demos/qwen35_27b/tt/tests/test_gdn_forward_prefill_diag.py -v -s
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model

_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    va = a - a.mean()
    vb = b - b.mean()
    return ((va * vb).sum() / (va.norm() * vb.norm() + 1e-12)).item()


def _pull(t, mesh_device, n_dev=None):
    """Pull tensor to CPU, taking first device's replica."""
    cpu = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    if n_dev is None:
        n_dev = mesh_device.get_num_devices()
    return cpu[: cpu.shape[0] // n_dev].float()


def _to_mesh(t, mesh_device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ---------------------------------------------------------------------------
# Build reference Q/K/V/beta/g from per-token sequential decode path
# ---------------------------------------------------------------------------


def _ref_qkvbg_from_sequential(gdn, x_cpu, mesh_device, seq_len):
    """
    Run _forward_prefill_sequential on x, intercept and return the per-token
    Q/K/V/beta/g values (stacked across T) so they can be compared against
    what forward_prefill computes.

    Returns dict of CPU tensors:
      conv_out  [T, qkv_dim_tp]    — post-SiLU conv output
      q         [Nv_TP, T, Dk]     — L2-normed and scaled Q, head-first
      k         [Nv_TP, T, Dk]     — L2-normed K, head-first
      v         [Nv_TP, T, Dv]     — V, head-first
      beta      [Nv_TP, T]         — sigmoid gate
      g         [Nv_TP, T]         — decay (neg_exp_A * softplus(a+dt_bias))
    """
    import torch.nn.functional as F

    Nk_TP = gdn.Nk_TP
    Nv_TP = gdn.Nv_TP
    Dk = gdn.Dk
    Dv = gdn.Dv
    key_dim_tp = gdn.key_dim_tp
    qkv_dim_tp = gdn.qkv_dim_tp
    repeat_factor = Nv_TP // Nk_TP
    tw = gdn.tw
    n_dev = mesh_device.get_num_devices()

    neg_exp_A = _pull(gdn.neg_exp_A, mesh_device, n_dev)[0, 0, :]  # [Nv_TP]
    dt_bias = _pull(tw["dt_bias"], mesh_device, n_dev)[0, 0, :]  # [Nv_TP]

    # Run projections + conv1d for full sequence (batched, same as forward_prefill)
    # — use device ops to match forward_prefill's matmul precision
    x_dev = _to_mesh(x_cpu, mesh_device)
    x_dram = ttnn.to_memory_config(x_dev, ttnn.DRAM_MEMORY_CONFIG)

    from models.demos.qwen35_27b.tt.model_config import create_prefill_matmul_program_config

    dim = gdn.args.dim
    qkvz_dim_tp = gdn.qkvz_dim_tp
    pc_qkvz = create_prefill_matmul_program_config(seq_len, dim, qkvz_dim_tp)
    pc_ab = create_prefill_matmul_program_config(seq_len, dim, Nv_TP * 2)
    qkvz_all = ttnn.linear(x_dram, tw["qkvz"], memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc_qkvz)
    ab_all = ttnn.linear(x_dram, tw["ab"], memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc_ab)
    ttnn.deallocate(x_dram)
    ttnn.deallocate(x_dev)

    qkv_all = ttnn.slice(qkvz_all, (0, 0, 0, 0), (1, 1, seq_len, qkv_dim_tp))
    ttnn.deallocate(qkvz_all)
    a_all = ttnn.slice(ab_all, (0, 0, 0, 0), (1, 1, seq_len, Nv_TP))
    b_all = ttnn.slice(ab_all, (0, 0, 0, Nv_TP), (1, 1, seq_len, Nv_TP * 2))
    ttnn.deallocate(ab_all)

    # Conv1d (same as forward_prefill — batched over full sequence)
    K = gdn.conv_kernel_size
    pad = _to_mesh(torch.zeros(1, 1, K - 1, qkv_dim_tp, dtype=torch.bfloat16), mesh_device)
    padded = ttnn.concat([pad, qkv_all], dim=2)
    ttnn.deallocate(pad)
    padded = ttnn.to_layout(padded, ttnn.ROW_MAJOR_LAYOUT)
    conv_acc = None
    for j in range(K):
        sl = ttnn.slice(padded, (0, 0, j, 0), (1, 1, j + seq_len, qkv_dim_tp))
        sl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT)
        tap = ttnn.reshape(tw["conv_taps"][j], (1, 1, 1, qkv_dim_tp))
        if conv_acc is None:
            conv_acc = ttnn.multiply(sl, tap)
        else:
            conv_acc = ttnn.mac(sl, tap, conv_acc)
        ttnn.deallocate(sl)
    ttnn.deallocate(padded)
    ttnn.deallocate(qkv_all)
    conv_all = ttnn.silu(conv_acc)  # [1, 1, T, qkv_dim_tp]
    ttnn.deallocate(conv_acc)

    # Pull conv output to CPU
    conv_cpu = _pull(conv_all, mesh_device, n_dev)[0, 0]  # [T, qkv_dim_tp]
    ttnn.deallocate(conv_all)

    # Pull a/b to CPU for beta/g
    a_cpu = _pull(a_all, mesh_device, n_dev)[0, 0]  # [T, Nv_TP]
    b_cpu = _pull(b_all, mesh_device, n_dev)[0, 0]  # [T, Nv_TP]
    ttnn.deallocate(a_all)
    ttnn.deallocate(b_all)

    # Compute Q/K/V/beta/g on CPU (float32 precision) to match decode reference
    q_raw = conv_cpu[:, :key_dim_tp].view(seq_len, Nk_TP, Dk)  # [T, Nk, Dk]
    k_raw = conv_cpu[:, key_dim_tp : 2 * key_dim_tp].view(seq_len, Nk_TP, Dk)
    v_raw = conv_cpu[:, 2 * key_dim_tp :].view(seq_len, Nv_TP, Dv)

    import torch.nn.functional as F

    scale = Dk**-0.5
    q_n = F.normalize(q_raw, p=2, dim=-1)  # [T, Nk, Dk]
    k_n = F.normalize(k_raw, p=2, dim=-1)

    # Expand Nk → Nv and apply scale
    q_exp = q_n.repeat_interleave(repeat_factor, dim=1) * scale  # [T, Nv, Dk]
    k_exp = k_n.repeat_interleave(repeat_factor, dim=1)  # [T, Nv, Dk]

    # Permute to [Nv, T, D] (head-first)
    q_hf = q_exp.permute(1, 0, 2)  # [Nv, T, Dk]
    k_hf = k_exp.permute(1, 0, 2)
    v_hf = v_raw.permute(1, 0, 2)  # [Nv, T, Dv]

    # Beta and g
    beta_cpu = torch.sigmoid(b_cpu).permute(1, 0)  # [Nv, T]
    g_cpu = (neg_exp_A.unsqueeze(0) * torch.nn.functional.softplus(a_cpu + dt_bias.unsqueeze(0))).permute(
        1, 0
    )  # [Nv, T]

    return {
        "conv_out": conv_cpu,
        "q": q_hf,
        "k": k_hf,
        "v": v_hf,
        "beta": beta_cpu,
        "g": g_cpu,
    }


# ---------------------------------------------------------------------------
# Monkey-patch forward_prefill to capture intermediates
# ---------------------------------------------------------------------------

_captured_par = {}


def _patched_chunk_gated_delta_rule(q, k, v, beta, g, **kwargs):
    """Wrapper that captures Q/K/V/beta/g before calling the real function."""
    from models.demos.qwen35_27b.tt.gdn_chunk_ops import chunk_gated_delta_rule as _real

    mesh = kwargs.get("mesh_device")
    n_dev = mesh.get_num_devices() if mesh else 4

    def _p(t):
        cpu = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        return cpu[: cpu.shape[0] // n_dev].float()

    # scale is applied inside chunk_gated_delta_rule — pull before
    _captured_par["q_before_scale"] = _p(q)
    _captured_par["k"] = _p(k)
    _captured_par["v"] = _p(v)
    _captured_par["beta"] = _p(beta)
    _captured_par["g"] = _p(g)

    out, final_state = _real(q, k, v, beta, g, **kwargs)
    _captured_par["scan_out"] = _p(out)  # [Nv_TP, T, Dv] float32
    _captured_par["scan_state"] = _p(final_state)  # [Nv_TP, Dk, Dv] float32
    return out, final_state


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("seq_len", [128], ids=["seq128"])
def test_forward_prefill_intermediates(mesh_device, reset_seeds, ensure_gc, seq_len):
    """
    Compare Q/K/V/beta/g computed by forward_prefill against the reference
    (same projections + CPU float32 math) to find the divergence point.

    Checkpoints:
      A: conv output
      B: Q/K/V (L2-normed, head-expanded, head-first)
      C: beta and g
    """
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = max(2048, seq_len * 2)

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
        n_layers=3,
    )
    args = model.args
    gdn_idx = next(i for i in range(args.n_layers) if args.layer_types[i] == "linear_attention")
    gdn = model.layers[gdn_idx].attention
    n_dev = mesh_device.get_num_devices()

    dim = args.dim
    torch.manual_seed(42)
    x_cpu = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16) * 0.1

    # ---- Build reference intermediates ----
    logger.info("Building reference Q/K/V/beta/g from projections + CPU math...")
    ref = _ref_qkvbg_from_sequential(gdn, x_cpu, mesh_device, seq_len)
    logger.info(f"  ref q shape={ref['q'].shape}, k={ref['k'].shape}, v={ref['v'].shape}")

    # ---- Run forward_prefill with monkey-patch to intercept intermediates ----
    import models.demos.qwen35_27b.tt.gdn as _gdn_mod

    _orig = _gdn_mod._chunk_gated_delta_rule
    _gdn_mod._chunk_gated_delta_rule = _patched_chunk_gated_delta_rule
    _captured_par.clear()

    gdn._init_prefill_states()
    x_dev = _to_mesh(x_cpu, mesh_device)
    try:
        out_par = gdn.forward_prefill(x_dev, current_pos=None)
        ttnn.deallocate(x_dev)
        ttnn.deallocate(out_par)
    finally:
        _gdn_mod._chunk_gated_delta_rule = _orig

    if not _captured_par:
        logger.error("  INTERCEPTOR did not fire — forward_prefill did not call _chunk_gated_delta_rule!")
        assert False, "forward_prefill did not call _chunk_gated_delta_rule"

    # ---- Compare checkpoint B: Q/K/V ----
    # _captured_par["q_before_scale"] is UNSCALED Q from forward_prefill.
    # ref["q"] is already scaled (ref applies scale in the CPU math above).
    # Apply scale to match:
    Dk = gdn.Dk
    scale = Dk**-0.5
    par_q = _captured_par["q_before_scale"] * scale  # now scaled [Nv_TP, T, Dk]
    par_k = _captured_par["k"]  # [Nv_TP, T, Dk]
    par_v = _captured_par["v"]  # [Nv_TP, T, Dv]
    par_beta_3d = _captured_par["beta"]  # [Nv_TP, T, 1]
    par_g_2d = _captured_par["g"]  # [Nv_TP, T]

    # beta from parallel is [Nv_TP, T, 1] — squeeze last dim for comparison
    par_beta = par_beta_3d.squeeze(-1)  # [Nv_TP, T]

    pcc_q = _pcc(ref["q"], par_q)
    pcc_k = _pcc(ref["k"], par_k)
    pcc_v = _pcc(ref["v"], par_v)
    pcc_beta = _pcc(ref["beta"], par_beta)
    pcc_g = _pcc(ref["g"], par_g_2d)

    logger.info("")
    logger.info("=== Intermediate PCC (reference vs forward_prefill) ===")
    logger.info(f"  Q   PCC: {pcc_q:.6f}   shape ref={ref['q'].shape} par={par_q.shape}")
    logger.info(f"  K   PCC: {pcc_k:.6f}   shape ref={ref['k'].shape} par={par_k.shape}")
    logger.info(f"  V   PCC: {pcc_v:.6f}   shape ref={ref['v'].shape} par={par_v.shape}")
    logger.info(f"  beta PCC:{pcc_beta:.6f}  shape ref={ref['beta'].shape} par={par_beta.shape}")
    logger.info(f"  g   PCC: {pcc_g:.6f}   shape ref={ref['g'].shape} par={par_g_2d.shape}")

    # Report max abs diff for beta and g (they're scalars so easier to interpret)
    beta_diff = (ref["beta"] - par_beta).abs()
    g_diff = (ref["g"] - par_g_2d).abs()
    logger.info(f"  beta abs-err: max={beta_diff.max():.4f} mean={beta_diff.mean():.6f}")
    logger.info(f"  g    abs-err: max={g_diff.max():.4f} mean={g_diff.mean():.6f}")

    # Element-wise: compare ref["q"][h=0, :4, :4] vs par_q[h=0, :4, :4]
    logger.info(f"  Q[h=0, :2, :4]: ref={ref['q'][0,:2,:4].tolist()}  par={par_q[0,:2,:4].tolist()}")
    logger.info(f"  K[h=0, :2, :4]: ref={ref['k'][0,:2,:4].tolist()}  par={par_k[0,:2,:4].tolist()}")
    logger.info(f"  beta[h=0, :4]:  ref={ref['beta'][0,:4].tolist()}  par={par_beta[0,:4].tolist()}")
    logger.info(f"  g[h=0, :4]:     ref={ref['g'][0,:4].tolist()}     par={par_g_2d[0,:4].tolist()}")

    assert pcc_q > 0.95, f"Q PCC {pcc_q:.6f} < 0.95 — Q diverges in forward_prefill"
    assert pcc_k > 0.95, f"K PCC {pcc_k:.6f} < 0.95 — K diverges in forward_prefill"
    assert pcc_v > 0.95, f"V PCC {pcc_v:.6f} < 0.95 — V diverges in forward_prefill"
    assert pcc_beta > 0.95, f"beta PCC {pcc_beta:.6f} < 0.95 — beta diverges in forward_prefill"
    assert pcc_g > 0.95, f"g PCC {pcc_g:.6f} < 0.95 — g diverges in forward_prefill"

    logger.info("PASS: all scan inputs match reference (PCC > 0.95)")

    # ---- Compare checkpoint D: scan output vs sequential reference ----
    # If scan inputs are correct but forward_prefill output PCC is 0.829, the bug
    # is either in chunk_gated_delta_rule itself or in post-processing (norm/permute/
    # retile/z-gate/output-proj).  Run a sequential reference scan using the
    # *captured* Q/K/V/beta/g (same values the parallel scan received) and compare.

    if "scan_out" not in _captured_par:
        logger.warning("  scan_out not in captured dict — skipping checkpoint D")
    else:
        from models.demos.qwen35_27b.reference.functional import gdn_recurrence_step

        Nv_TP_val = gdn.Nv_TP
        Dk_val = gdn.Dk
        Dv_val = gdn.Dv

        ref_state_scan = torch.zeros(Nv_TP_val, Dk_val, Dv_val)
        ref_scan_list = []
        for t_idx in range(seq_len):
            out_t = gdn_recurrence_step(
                par_q[:, t_idx, :],  # [Nv_TP, Dk] — L2-normed, scaled
                par_k[:, t_idx, :],  # [Nv_TP, Dk] — L2-normed
                par_v[:, t_idx, :],  # [Nv_TP, Dv]
                par_g_2d[:, t_idx],  # [Nv_TP] log-space decay
                par_beta[:, t_idx],  # [Nv_TP]
                ref_state_scan,  # modified in-place
            )
            ref_scan_list.append(out_t)

        ref_scan_out = torch.stack(ref_scan_list, dim=1)  # [Nv_TP, T, Dv]
        par_scan_out = _captured_par["scan_out"]  # [Nv_TP, T, Dv] float32
        par_scan_state = _captured_par["scan_state"]  # [Nv_TP, Dk, Dv] float32

        pcc_scan_out = _pcc(ref_scan_out, par_scan_out)
        pcc_scan_state = _pcc(ref_state_scan, par_scan_state)

        scan_abs = (ref_scan_out - par_scan_out).abs()

        logger.info("")
        logger.info("=== Checkpoint D: scan output vs sequential reference recurrence ===")
        logger.info(f"  scan_out   PCC: {pcc_scan_out:.6f}   shape={par_scan_out.shape}")
        logger.info(f"  scan_state PCC: {pcc_scan_state:.6f}  shape={par_scan_state.shape}")
        logger.info(f"  scan_out abs-err: max={scan_abs.max():.4f} mean={scan_abs.mean():.6f}")
        logger.info(f"  ref_scan_out[h=0,:3,:4] = {ref_scan_out[0,:3,:4].tolist()}")
        logger.info(f"  par_scan_out[h=0,:3,:4] = {par_scan_out[0,:3,:4].tolist()}")

        if pcc_scan_out > 0.99:
            logger.info("  VERDICT: scan output CORRECT (PCC > 0.99)")
            logger.info("           => bug is in POST-PROCESSING after the scan")
            logger.info("              (typecast bf16, rms_norm, permute, _retile_reshape, z-gate, output-proj)")
        elif pcc_scan_out > 0.95:
            logger.info("  VERDICT: scan output marginal (0.95 < PCC < 0.99)")
            logger.info("           => both scan and post-processing may contribute")
        else:
            logger.info("  VERDICT: scan output WRONG (PCC < 0.95)")
            logger.info("           => bug is inside chunk_gated_delta_rule")

        # Per-token max error to find WHERE errors concentrate.
        per_tok_err = (ref_scan_out - par_scan_out).abs().max(dim=0).values.max(dim=-1).values  # [T]
        per_tok_ref_max = ref_scan_out.abs().max(dim=0).values.max(dim=-1).values  # [T]
        per_tok_ratio = (par_scan_out.abs() / (ref_scan_out.abs() + 1e-30)).mean(dim=-1).mean(dim=0)  # [T]
        logger.info("  Per-token max abs error (top 10):")
        top_errs, top_toks = per_tok_err.topk(min(10, seq_len))
        for err_val, tok_idx in zip(top_errs.tolist(), top_toks.tolist()):
            logger.info(
                f"    t={tok_idx:4d}  max_err={err_val:.6f}  ref_max={per_tok_ref_max[tok_idx]:.6f}  par/ref_ratio={per_tok_ratio[tok_idx]:.4f}"
            )

        # Overall par/ref ratio (mean over all elements)
        ratio_all = (par_scan_out.abs() / (ref_scan_out.abs() + 1e-30)).flatten()
        logger.info(
            f"  Global par/ref ratio: median={ratio_all.median():.4f} mean={ratio_all.mean():.4f} std={ratio_all.std():.4f}"
        )

        # Check if par values are consistently scaled relative to ref
        flat_ref = ref_scan_out.flatten()
        flat_par = par_scan_out.flatten()
        top_ref_idx = flat_ref.abs().topk(20).indices
        logger.info(f"  Top-20 ref values: {flat_ref[top_ref_idx][:5].tolist()} ...")
        logger.info(f"  Top-20 par values: {flat_par[top_ref_idx][:5].tolist()} ...")

        # Non-fatal: log verdict without asserting so the test always runs to completion.
        # The key information is in the VERDICT line above.
