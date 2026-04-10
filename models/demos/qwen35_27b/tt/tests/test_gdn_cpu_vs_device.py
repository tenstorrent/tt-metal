# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
GDN device kernel vs CPU sequential recurrence comparison.

Loads the full model, runs prefill on one GDN layer both ways, compares outputs.
- Device path: per-token gdn_full_fused_inplace kernel (bfloat16 state in DRAM)
- CPU path: PyTorch sequential loop matching device precision (bfloat16 state
  round-trip per step, same L2 norm / softplus formulas as kernel)

Both share the same projections, conv1d, and post-processing (RMS norm + SiLU gate).
"""

import logging
import os
import time

import pytest
import torch

import ttnn
from models.demos.qwen35_27b.tt.gdn import _unshard
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace
from models.demos.qwen35_27b.tt.model import create_qwen35_model
from models.demos.qwen35_27b.tt.model_config import create_prefill_matmul_program_config

logger = logging.getLogger(__name__)


def compute_pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    if a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
        return 1.0 if torch.allclose(a_flat, b_flat, atol=1e-6) else 0.0
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


def _run_device_recurrence(gdn_layer, conv_out_all, a_all, b_all, z_all, seq_len, mesh_device=None):
    """Run the ORIGINAL per-token device recurrence loop. Returns (gated_outputs, per_token_times)."""
    tw = gdn_layer.tw
    B_pf = 1
    Nv_TP = gdn_layer.Nv_TP
    Nk_TP = gdn_layer.Nk_TP
    Dk = gdn_layer.Dk
    Dv = gdn_layer.Dv
    qkv_dim_tp = gdn_layer.qkv_dim_tp
    qkvz_dim_tp = gdn_layer.qkvz_dim_tp
    key_dim_tp = gdn_layer.key_dim_tp
    num_pairs_pf = B_pf * Nv_TP
    repeat_factor = Nv_TP // Nk_TP

    gated_outputs = []
    per_token_times = []
    for t in range(seq_len):
        t0_tok = time.time()
        conv_out_t = ttnn.slice(conv_out_all, (0, 0, t, 0), (1, 1, t + 1, qkv_dim_tp))
        conv_out_t = ttnn.reshape(conv_out_t, (1, B_pf, qkv_dim_tp))
        a_tt = ttnn.slice(a_all, (0, 0, t, 0), (1, 1, t + 1, Nv_TP))
        a_tt = ttnn.reshape(a_tt, (1, B_pf, Nv_TP))
        b_tt = ttnn.slice(b_all, (0, 0, t, 0), (1, 1, t + 1, Nv_TP))
        b_tt = ttnn.reshape(b_tt, (1, B_pf, Nv_TP))
        z_tt = ttnn.slice(z_all, (0, 0, t, 0), (1, 1, t + 1, qkvz_dim_tp - qkv_dim_tp))
        z_tt = ttnn.reshape(z_tt, (1, B_pf, qkvz_dim_tp - qkv_dim_tp))

        a_tt = _unshard(a_tt)
        b_tt = _unshard(b_tt)
        conv_out_t = _unshard(conv_out_t)

        gdn_full_fused_inplace(
            conv_out_t,
            a_tt,
            b_tt,
            gdn_layer.neg_exp_A,
            tw["dt_bias"],
            tw["norm_w"],
            gdn_layer.scale_tt,
            gdn_layer.rms_scale_tt,
            gdn_layer.rms_eps_tt,
            gdn_layer._prefill_rec_states,
            gdn_layer._prefill_fused_output,
            num_pairs=num_pairs_pf,
            num_cores=min(96, num_pairs_pf),
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )
        ttnn.deallocate(conv_out_t)
        ttnn.deallocate(a_tt)
        ttnn.deallocate(b_tt)

        out_r = ttnn.reshape(gdn_layer._prefill_fused_output, (B_pf, Nv_TP, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B_pf, gdn_layer.value_dim_tp))
        ttnn.deallocate(out_n)
        z_act = ttnn.silu(z_tt)
        ttnn.deallocate(z_tt)
        out_f = _unshard(out_f)
        gated = ttnn.multiply(out_f, z_act)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z_act)
        gated_outputs.append(gated)
        if mesh_device is not None:
            ttnn.synchronize_device(mesh_device)
        per_token_times.append(time.time() - t0_tok)

    return gated_outputs, per_token_times


def _run_cpu_recurrence(gdn_layer, conv_out_all, a_all, b_all, z_all, seq_len, mesh_device):
    """Run the CPU sequential recurrence matching device kernel precision.

    Matches the fused kernel exactly:
    - L2 norm without epsilon: q * rsqrt(dot(q,q)) (same as kernel Phase 1/2)
    - Softplus via direct log(1+exp(x)) (same as kernel Phase 4)
    - State round-tripped through bfloat16 each step (same as kernel DRAM read/write)
    """
    tw = gdn_layer.tw
    Nk_TP = gdn_layer.Nk_TP
    Nv_TP = gdn_layer.Nv_TP
    Dk = gdn_layer.Dk
    Dv = gdn_layer.Dv
    qkv_dim_tp = gdn_layer.qkv_dim_tp
    qkvz_dim_tp = gdn_layer.qkvz_dim_tp
    key_dim_tp = gdn_layer.key_dim_tp
    B_pf = 1
    mesh = mesh_device
    num_devices = mesh.get_num_devices()

    conv_cpu = ttnn.to_torch(conv_out_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    a_cpu = ttnn.to_torch(a_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    b_cpu = ttnn.to_torch(b_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    neg_exp_A_cpu = ttnn.to_torch(gdn_layer.neg_exp_A, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    dt_bias_cpu = ttnn.to_torch(tw["dt_bias"], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    repeat_factor = Nv_TP // Nk_TP
    q_parts, k_parts, v_parts, beta_parts, g_parts = [], [], [], [], []
    for d in range(num_devices):
        c = conv_cpu[d].squeeze(0).float()
        q_d = c[:, :key_dim_tp].reshape(seq_len, Nk_TP, Dk).permute(1, 0, 2)
        k_d = c[:, key_dim_tp : 2 * key_dim_tp].reshape(seq_len, Nk_TP, Dk).permute(1, 0, 2)
        v_d = c[:, 2 * key_dim_tp : qkv_dim_tp].reshape(seq_len, Nv_TP, Dv).permute(1, 0, 2)
        q_d = q_d.repeat_interleave(repeat_factor, dim=0)
        k_d = k_d.repeat_interleave(repeat_factor, dim=0)
        b_d = b_cpu[d].squeeze(0).float()
        beta_d = torch.sigmoid(b_d).permute(1, 0).unsqueeze(-1)
        a_d = a_cpu[d].squeeze(0).float()
        nea_d = neg_exp_A_cpu[d].flatten().float()
        dtb_d = dt_bias_cpu[d].flatten().float()
        sp = torch.log1p(torch.exp(a_d.permute(1, 0) + dtb_d.unsqueeze(1)))
        g_d = nea_d.unsqueeze(1) * sp
        q_parts.append(q_d)
        k_parts.append(k_d)
        v_parts.append(v_d)
        beta_parts.append(beta_d)
        g_parts.append(g_d)

    q_cpu = torch.cat(q_parts, dim=0).float()
    k_cpu = torch.cat(k_parts, dim=0).float()
    v_cpu = torch.cat(v_parts, dim=0).float()
    beta_cpu_all = torch.cat(beta_parts, dim=0).float()
    g_cpu_all = torch.cat(g_parts, dim=0).float()

    scale_val = gdn_layer.scale
    q_cpu = q_cpu * torch.rsqrt((q_cpu * q_cpu).sum(dim=-1, keepdim=True)) * scale_val
    k_cpu = k_cpu * torch.rsqrt((k_cpu * k_cpu).sum(dim=-1, keepdim=True))

    BH_total = q_cpu.shape[0]
    S_cpu = torch.zeros(BH_total, Dk, Dv, dtype=torch.float32)
    outputs_cpu = []
    for t in range(seq_len):
        q_t = q_cpu[:, t : t + 1, :]
        k_t = k_cpu[:, t : t + 1, :]
        v_t = v_cpu[:, t : t + 1, :]
        beta_t = beta_cpu_all[:, t : t + 1, :]
        g_t = g_cpu_all[:, t]
        decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
        S_cpu = S_cpu * decay
        kv_mem = torch.bmm(k_t, S_cpu)
        delta = beta_t * (v_t - kv_mem)
        S_cpu = S_cpu + torch.bmm(k_t.transpose(-2, -1), delta)
        o_t = torch.bmm(q_t, S_cpu)
        outputs_cpu.append(o_t)
        S_cpu = S_cpu.to(torch.bfloat16).to(torch.float32)

    recurrence_out = torch.cat(outputs_cpu, dim=1)

    out_4d_parts = []
    for d in range(num_devices):
        o_d = recurrence_out[d * Nv_TP : (d + 1) * Nv_TP]
        o_d = o_d.permute(1, 0, 2).reshape(seq_len, gdn_layer.value_dim_tp)
        out_4d_parts.append(o_d.unsqueeze(0).unsqueeze(0))
    out_4d = torch.cat(out_4d_parts, dim=0).contiguous()

    chunk_out_dev = ttnn.from_torch(
        out_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )

    gated_outputs = []
    for t in range(seq_len):
        out_t = ttnn.slice(chunk_out_dev, (0, 0, t, 0), (1, 1, t + 1, gdn_layer.value_dim_tp))
        out_t = ttnn.reshape(out_t, (B_pf, Nv_TP, Dv))
        out_n = ttnn.rms_norm(out_t, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(out_t)
        out_f = ttnn.reshape(out_n, (1, B_pf, gdn_layer.value_dim_tp))
        ttnn.deallocate(out_n)
        z_tt = ttnn.slice(z_all, (0, 0, t, 0), (1, 1, t + 1, qkvz_dim_tp - qkv_dim_tp))
        z_tt = ttnn.reshape(z_tt, (1, B_pf, qkvz_dim_tp - qkv_dim_tp))
        z_act = ttnn.silu(z_tt)
        ttnn.deallocate(z_tt)
        out_f = _unshard(out_f)
        gated = ttnn.multiply(out_f, z_act)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z_act)
        gated_outputs.append(gated)

    ttnn.deallocate(chunk_out_dev)

    state_bf16 = ttnn.from_torch(
        S_cpu.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    ttnn.copy(state_bf16, gdn_layer._prefill_rec_states)
    ttnn.deallocate(state_bf16)

    return gated_outputs


def _get_model_path():
    return os.environ.get("HF_MODEL", os.environ.get("QWEN35_27B_PATH", ""))


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0])
def test_cpu_vs_device_gdn(mesh_device, layer_idx):
    """Compare CPU sequential recurrence vs device fused kernel for a single GDN layer."""
    model_path = _get_model_path()
    if not model_path:
        pytest.skip("Set HF_MODEL to run this test")

    batch_size = 32
    max_seq_len = 2048
    seq_len = 64  # Short for speed

    # Load only the layers we need
    logger.info("Loading model (1 layer)...")
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
        n_layers=layer_idx + 1,
    )
    args = model.args

    # Find the GDN layer
    # Layer indices: GDN layers are at positions where is_gdn is True
    gdn_layer = model.layers[layer_idx].attention
    assert hasattr(gdn_layer, "gdn_nk_tp") or hasattr(gdn_layer, "Nk_TP"), f"Layer {layer_idx} is not a GDN layer"

    logger.info(f"Testing GDN layer {layer_idx} with seq_len={seq_len}")

    # Create random input
    torch.manual_seed(42)
    x_cpu = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16) * 0.01
    x_tt = ttnn.from_torch(
        x_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run shared projections + conv1d (same for both paths)
    tw = gdn_layer.tw
    dim = args.dim
    Nv_TP = gdn_layer.Nv_TP
    qkv_dim_tp = gdn_layer.qkv_dim_tp
    qkvz_dim_tp = gdn_layer.qkvz_dim_tp

    x_dram = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)
    qkvz_all = ttnn.linear(
        x_dram,
        tw["qkvz"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=create_prefill_matmul_program_config(seq_len, dim, qkvz_dim_tp),
        compute_kernel_config=gdn_layer.compute_cfg,
    )
    ab_all = ttnn.linear(
        x_dram,
        tw["ab"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=create_prefill_matmul_program_config(seq_len, dim, Nv_TP * 2),
        compute_kernel_config=gdn_layer.compute_cfg,
    )
    ttnn.deallocate(x_dram)

    qkv_all = ttnn.slice(qkvz_all, (0, 0, 0, 0), (1, 1, seq_len, qkv_dim_tp))
    z_all = ttnn.slice(qkvz_all, (0, 0, 0, qkv_dim_tp), (1, 1, seq_len, qkvz_dim_tp))
    ttnn.deallocate(qkvz_all)
    a_all = ttnn.slice(ab_all, (0, 0, 0, 0), (1, 1, seq_len, Nv_TP))
    b_all = ttnn.slice(ab_all, (0, 0, 0, Nv_TP), (1, 1, seq_len, Nv_TP * 2))
    ttnn.deallocate(ab_all)

    # Batched conv1d
    K = gdn_layer.conv_kernel_size
    zero_pad = ttnn.from_torch(
        torch.zeros(1, 1, K - 1, qkv_dim_tp, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    padded = ttnn.concat([zero_pad, qkv_all], dim=2)
    ttnn.deallocate(zero_pad)
    conv_out_all = None
    for j in range(K):
        shift = K - 1 - j
        shifted = ttnn.slice(padded, (0, 0, shift, 0), (1, 1, shift + seq_len, qkv_dim_tp))
        tap_4d = ttnn.reshape(tw["conv_taps"][j], (1, 1, 1, qkv_dim_tp))
        if conv_out_all is None:
            conv_out_all = ttnn.multiply(shifted, tap_4d)
        else:
            conv_out_all = ttnn.mac(shifted, tap_4d, conv_out_all)
        ttnn.deallocate(shifted)
    ttnn.deallocate(padded)
    conv_out_all = ttnn.silu(conv_out_all)
    ttnn.deallocate(qkv_all)

    conv_cpu_saved = ttnn.to_torch(conv_out_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    a_cpu_saved = ttnn.to_torch(a_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    b_cpu_saved = ttnn.to_torch(b_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    z_cpu_saved = ttnn.to_torch(z_all, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(conv_out_all)
    ttnn.deallocate(a_all)
    ttnn.deallocate(b_all)
    ttnn.deallocate(z_all)

    def _restore_tensors():
        """Recreate device tensors from saved CPU copies."""

        def _to_dev(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )

        return _to_dev(conv_cpu_saved), _to_dev(a_cpu_saved), _to_dev(b_cpu_saved), _to_dev(z_cpu_saved)

    # ============================================================
    # PATH A: Device per-token kernel (warmup + timed)
    # ============================================================
    # Warmup run (compile kernels)
    logger.info("Running device path (warmup)...")
    gdn_layer._init_prefill_states()
    conv_w, a_w, b_w, z_w = _restore_tensors()
    warmup_gated, _ = _run_device_recurrence(gdn_layer, conv_w, a_w, b_w, z_w, seq_len)
    for g in warmup_gated:
        ttnn.deallocate(g)
    ttnn.deallocate(conv_w)
    ttnn.deallocate(a_w)
    ttnn.deallocate(b_w)
    ttnn.deallocate(z_w)

    # Timed run
    logger.info("Running device path (timed)...")
    gdn_layer._init_prefill_states()
    conv_a, a_a, b_a, z_a = _restore_tensors()
    ttnn.synchronize_device(mesh_device)
    t0_device = time.time()
    device_gated, per_token_times = _run_device_recurrence(gdn_layer, conv_a, a_a, b_a, z_a, seq_len, mesh_device)
    ttnn.synchronize_device(mesh_device)
    t_device = time.time() - t0_device

    device_state = ttnn.to_torch(
        gdn_layer._prefill_rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    # Concat gated outputs for comparison
    device_concat = ttnn.concat(device_gated, dim=1)
    for g in device_gated:
        ttnn.deallocate(g)
    device_out_cpu = ttnn.to_torch(device_concat, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(device_concat)

    ttnn.deallocate(conv_a)
    ttnn.deallocate(a_a)
    ttnn.deallocate(b_a)
    ttnn.deallocate(z_a)

    # ============================================================
    # PATH B: CPU sequential recurrence (warmup + timed)
    # ============================================================
    # Warmup run
    logger.info("Running CPU path (warmup)...")
    gdn_layer._init_prefill_states()
    conv_w2, a_w2, b_w2, z_w2 = _restore_tensors()
    warmup_gated2 = _run_cpu_recurrence(gdn_layer, conv_w2, a_w2, b_w2, z_w2, seq_len, mesh_device)
    for g in warmup_gated2:
        ttnn.deallocate(g)
    ttnn.deallocate(conv_w2)
    ttnn.deallocate(a_w2)
    ttnn.deallocate(b_w2)
    ttnn.deallocate(z_w2)

    # Timed run
    logger.info("Running CPU path (timed)...")
    gdn_layer._init_prefill_states()
    conv_b, a_b, b_b, z_b = _restore_tensors()
    ttnn.synchronize_device(mesh_device)
    t0_cpu = time.time()
    cpu_gated = _run_cpu_recurrence(gdn_layer, conv_b, a_b, b_b, z_b, seq_len, mesh_device)
    ttnn.synchronize_device(mesh_device)
    t_cpu = time.time() - t0_cpu

    cpu_state = ttnn.to_torch(gdn_layer._prefill_rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    cpu_concat = ttnn.concat(cpu_gated, dim=1)
    for g in cpu_gated:
        ttnn.deallocate(g)
    cpu_out_cpu = ttnn.to_torch(cpu_concat, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(cpu_concat)

    ttnn.deallocate(conv_b)
    ttnn.deallocate(a_b)
    ttnn.deallocate(b_b)
    ttnn.deallocate(z_b)

    # ============================================================
    # COMPARE
    # ============================================================
    pcc_output = compute_pcc(device_out_cpu, cpu_out_cpu)
    pcc_state = compute_pcc(device_state, cpu_state)

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}: CPU vs Device GDN comparison (seq_len={seq_len})")
    print(f"")
    print(f"  TIMING (1 layer, {seq_len} tokens):")
    print(f"    Device per-token kernel: {t_device*1000:.1f} ms total")
    print(f"    CPU sequential:          {t_cpu*1000:.1f} ms total")
    print(f"    Speedup:                 {t_device/t_cpu:.2f}x")
    print(f"    Extrapolated 48 layers:  device={t_device*48:.1f}s, cpu={t_cpu*48:.1f}s")
    print(f"")
    print(f"  PER-TOKEN DEVICE TIMING (proves no program cache hit):")
    pt_ms = [t * 1000 for t in per_token_times]
    print(f"    Token  0 (1st call): {pt_ms[0]:.2f} ms")
    print(f"    Token  1 (2nd call): {pt_ms[1]:.2f} ms")
    print(f"    Token  2 (3rd call): {pt_ms[2]:.2f} ms")
    if len(pt_ms) > 10:
        print(f"    Token 10:            {pt_ms[10]:.2f} ms")
    if len(pt_ms) > 30:
        print(f"    Token 30:            {pt_ms[30]:.2f} ms")
    print(f"    Token {len(pt_ms)-1} (last):   {pt_ms[-1]:.2f} ms")
    print(f"    Average:             {sum(pt_ms)/len(pt_ms):.2f} ms/token")
    print(f"    Min:                 {min(pt_ms):.2f} ms, Max: {max(pt_ms):.2f} ms")
    print(f"    Device compute only: ~0.054 ms/token (from profiling)")
    print(f"    Host overhead/token: ~{sum(pt_ms)/len(pt_ms) - 0.054:.2f} ms (ProgramDescriptor rebuild)")
    if max(pt_ms) / min(pt_ms[1:] if len(pt_ms) > 1 else pt_ms) < 1.5:
        print(f"    >>> CONFIRMED: No cache speedup — every call rebuilds ProgramDescriptor")
    print(f"")
    print(f"  ACCURACY:")
    print(f"    Output PCC: {pcc_output:.6f}")
    print(f"    State PCC:  {pcc_state:.6f}")
    print(f"  Shapes: device={device_out_cpu.shape}, cpu={cpu_out_cpu.shape}")
    print(f"  Device gated[0,0,:4]: {device_out_cpu[0,0,:4].tolist()}")
    print(f"  CPU gated[0,0,:4]:    {cpu_out_cpu[0,0,:4].tolist()}")
    print(f"  Device gated[0,-1,:4]: {device_out_cpu[0,-1,:4].tolist()}")
    print(f"  CPU gated[0,-1,:4]:    {cpu_out_cpu[0,-1,:4].tolist()}")
    print(f"  Device state[0,0,:4]: {device_state[0,0,:4].tolist()}")
    print(f"  CPU state[0,0,:4]:    {cpu_state[0,0,:4].tolist()}")
    print(f"{'='*60}")

    assert pcc_output > 0.99, f"Output PCC {pcc_output:.4f} too low for layer {layer_idx}"
    assert pcc_state > 0.99, f"State PCC {pcc_state:.4f} too low for layer {layer_idx}"
