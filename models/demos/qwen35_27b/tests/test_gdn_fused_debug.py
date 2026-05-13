# Quick diagnostic script for fused GDN kernel
import math

import torch
from loguru import logger

import ttnn


def _l2_norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def main():
    device = ttnn.open_device(device_id=0)

    B = 1
    Nk_TP = 4
    Nv_TP = 12
    Dk = 128
    Dv = 128
    repeat_factor = Nv_TP // Nk_TP
    key_dim_tp = Nk_TP * Dk
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv
    z_dim_tp = Nv_TP * Dv
    value_dim_tp = Nv_TP * Dv
    num_pairs = B * Nv_TP
    scale = Dk**-0.5

    torch.manual_seed(42)
    conv_out_ref = torch.randn(1, B, qkv_dim_tp, dtype=torch.float32) * 0.1
    a_ref = torch.randn(1, B, Nv_TP, dtype=torch.float32) * 0.5
    b_ref = torch.randn(1, B, Nv_TP, dtype=torch.float32) * 0.5
    z_ref = torch.randn(1, B, z_dim_tp, dtype=torch.float32) * 0.1
    neg_exp_A_ref = -torch.exp(torch.randn(1, 1, Nv_TP, dtype=torch.float32) * 0.5)
    dt_bias_ref = torch.randn(1, 1, Nv_TP, dtype=torch.float32) * 0.1
    norm_w_ref = torch.ones(1, 1, Dv, dtype=torch.float32) + torch.randn(1, 1, Dv, dtype=torch.float32) * 0.01
    state_ref = torch.randn(num_pairs, Dk, Dv, dtype=torch.float32) * 0.01

    # Reference step-by-step
    q_raw = conv_out_ref[:, :, :key_dim_tp].reshape(B, Nk_TP, Dk)
    k_raw = conv_out_ref[:, :, key_dim_tp : 2 * key_dim_tp].reshape(B, Nk_TP, Dk)
    v_raw = conv_out_ref[:, :, 2 * key_dim_tp :].reshape(B, Nv_TP, Dv)

    q_normed = _l2_norm(q_raw) * scale
    k_normed = _l2_norm(k_raw)

    q_exp = q_normed.repeat_interleave(repeat_factor, dim=1)
    k_exp = k_normed.repeat_interleave(repeat_factor, dim=1)

    q = q_exp.reshape(num_pairs, 1, Dk)
    k = k_exp.reshape(num_pairs, 1, Dk)
    v = v_raw.reshape(num_pairs, 1, Dv)

    logger.info(f"Ref Q normed [pair 0]: min={q[0].min():.6f}, max={q[0].max():.6f}, norm={q[0].norm():.6f}")
    logger.info(f"Ref K normed [pair 0]: min={k[0].min():.6f}, max={k[0].max():.6f}, norm={k[0].norm():.6f}")

    beta = torch.sigmoid(b_ref.reshape(num_pairs, 1, 1))
    softplus_val = torch.nn.functional.softplus(
        a_ref.reshape(num_pairs, 1, 1) + dt_bias_ref.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1)
    )
    g = neg_exp_A_ref.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1) * softplus_val

    logger.info(f"Ref gates [pair 0]: beta={beta[0].item():.6f}, g={g[0].item():.6f}")

    g_exp_val = g.squeeze(-1).exp()
    new_state = state_ref.clone() * g_exp_val.unsqueeze(-1)
    kv_mem = torch.bmm(k, new_state)
    delta = beta * (v - kv_mem)
    new_state = new_state + torch.bmm(k.transpose(-2, -1), delta)
    rec_out = torch.bmm(q, new_state)

    logger.info(f"Ref rec_out [pair 0]: min={rec_out[0].min():.6f}, max={rec_out[0].max():.6f}")

    rec_out_r = rec_out.reshape(B, Nv_TP, Dv)
    rms = rec_out_r.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    normed = rec_out_r * rms * norm_w_ref

    z_r = z_ref.reshape(B, Nv_TP, Dv)
    z_act = z_r * torch.sigmoid(z_r)
    output_ref = (normed * z_act).reshape(1, B, value_dim_tp)

    logger.info(f"Ref output: min={output_ref.min():.6f}, max={output_ref.max():.6f}, mean={output_ref.mean():.6f}")

    # Also compute raw rec_out reference (before RMS norm + SiLU) for bypass mode
    rec_out_ref_flat = rec_out.reshape(1, B, value_dim_tp)
    logger.info(f"Ref raw rec_out: min={rec_out_ref_flat.min():.6f}, max={rec_out_ref_flat.max():.6f}")

    # Run fused kernel
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace

    def to_tt(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    conv_out_tt = to_tt(conv_out_ref)
    z_tt = to_tt(z_ref)
    norm_w_tt = to_tt(norm_w_ref)
    state_tt = to_tt(state_ref)

    a_fused = to_tt(a_ref.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1))
    b_fused = to_tt(b_ref.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1))
    neg_exp_A_fused = to_tt(neg_exp_A_ref.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1))
    dt_bias_fused = to_tt(dt_bias_ref.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1))

    scale_tt = to_tt(torch.full((1, 1, 1), scale, dtype=torch.float32))
    rms_scale_tt = to_tt(torch.full((1, 1, 1), math.sqrt(Dv), dtype=torch.float32))
    rms_eps_tt = to_tt(torch.full((1, 1, 1), Dv * 1e-6, dtype=torch.float32))
    output_tt = to_tt(torch.zeros(1, B, value_dim_tp, dtype=torch.float32))

    gdn_full_fused_inplace(
        conv_out_tt,
        a_fused,
        b_fused,
        z_tt,
        neg_exp_A_fused,
        dt_bias_fused,
        norm_w_tt,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        state_tt,
        output_tt,
        num_cores=min(40, num_pairs),
        Nk_TP=Nk_TP,
        Nv_TP=Nv_TP,
        Dk=Dk,
        Dv=Dv,
        qkv_dim_tp=qkv_dim_tp,
        key_dim_tp=key_dim_tp,
        z_dim_tp=z_dim_tp,
    )

    # Force DRAM coherence: ttnn.add reads output via standard pipeline, flushing NOC writes
    output_flushed = ttnn.add(output_tt, 0.0)
    state_flushed = ttnn.add(state_tt, 0.0)
    out_tt_cpu = ttnn.to_torch(output_flushed).float()
    state_tt_cpu = ttnn.to_torch(state_flushed).float()
    ttnn.deallocate(output_flushed)
    ttnn.deallocate(state_flushed)

    logger.info(f"Kernel output: min={out_tt_cpu.min():.6f}, max={out_tt_cpu.max():.6f}, mean={out_tt_cpu.mean():.6f}")
    logger.info(f"Kernel state: min={state_tt_cpu.min():.6f}, max={state_tt_cpu.max():.6f}")

    # Element-wise comparison
    out_ref_flat = output_ref.flatten()
    out_tt_flat = out_tt_cpu.flatten()[: out_ref_flat.shape[0]]

    # Check shapes
    logger.info(f"Ref shape: {output_ref.shape}, Kernel shape: {out_tt_cpu.shape}")

    # Print first 10 values
    logger.info(f"Ref first 10:    {out_ref_flat[:10].tolist()}")
    logger.info(f"Kernel first 10: {out_tt_flat[:10].tolist()}")

    # Per-head PCC
    for v_head in range(Nv_TP):
        ref_head = output_ref[0, 0, v_head * Dv : (v_head + 1) * Dv]
        tt_head = out_tt_cpu.flatten()[v_head * Dv : (v_head + 1) * Dv]
        if ref_head.numel() == tt_head.numel() and ref_head.std() > 0:
            head_pcc = torch.corrcoef(torch.stack([ref_head, tt_head]))[0, 1].item()
            logger.info(
                f"Head {v_head} PCC: {head_pcc:.6f}, ref_range=[{ref_head.min():.4f},{ref_head.max():.4f}], tt_range=[{tt_head.min():.4f},{tt_head.max():.4f}]"
            )

    # State PCC
    state_ref_flat = new_state.flatten()
    state_tt_flat = state_tt_cpu.flatten()[: state_ref_flat.shape[0]]
    state_pcc = torch.corrcoef(torch.stack([state_ref_flat, state_tt_flat]))[0, 1].item()
    logger.info(f"State PCC: {state_pcc:.6f}")

    # Overall output PCC (use raw rec_out ref if in bypass mode)
    # Check if kernel output looks like raw rec_out (small values) or post-processed (larger values)
    if out_tt_cpu.abs().max() < 0.01:
        logger.info("BYPASS MODE: comparing raw rec_out")
        rec_ref_flat = rec_out_ref_flat.flatten()
        pcc = torch.corrcoef(torch.stack([rec_ref_flat, out_tt_flat[: rec_ref_flat.shape[0]]]))[0, 1].item()
        logger.info(f"Raw rec_out PCC: {pcc:.6f}")
        scale_fit = (rec_ref_flat * out_tt_flat[: rec_ref_flat.shape[0]]).sum() / (
            out_tt_flat[: rec_ref_flat.shape[0]] ** 2
        ).sum()
        logger.info(f"Raw rec_out scale factor: {scale_fit:.6f}")
    else:
        pcc = torch.corrcoef(torch.stack([out_ref_flat, out_tt_flat]))[0, 1].item()
        logger.info(f"Output PCC: {pcc:.6f}")

    # Check if there's a systematic scaling factor
    # Compute best-fit linear scale: scale = sum(ref * tt) / sum(tt * tt)
    scale_fit = (out_ref_flat * out_tt_flat).sum() / (out_tt_flat * out_tt_flat).sum()
    logger.info(f"Best-fit scale factor: {scale_fit:.6f} (kernel_output * {scale_fit:.4f} ≈ reference)")
    scaled_out = out_tt_flat * scale_fit
    pcc_scaled = torch.corrcoef(torch.stack([out_ref_flat, scaled_out]))[0, 1].item()
    max_diff_scaled = (out_ref_flat - scaled_out).abs().max().item()
    logger.info(f"After scaling: PCC={pcc_scaled:.6f}, max_diff={max_diff_scaled:.6f}")

    # Also check ref state range
    logger.info(f"Ref state: min={new_state.min():.6f}, max={new_state.max():.6f}")

    # Compare state magnitude ratio
    state_scale_fit = (state_ref_flat * state_tt_flat).sum() / (state_tt_flat * state_tt_flat).sum()
    logger.info(f"State scale factor: {state_scale_fit:.6f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
