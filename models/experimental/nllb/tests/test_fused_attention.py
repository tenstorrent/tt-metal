# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Tiny independent BF16 SDPA checks; no model weights or CHIA fixtures required."""

import json, torch, ttnn


def check(device):
    torch.manual_seed(9127)
    kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    def up(x):
        return ttnn.from_torch(
            x.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def cpu(x):
        return ttnn.to_torch(x).float()

    for name, L, S, valid, causal in [
        ("causal-prefix", 32, 32, 7, True),
        ("causal-tile33", 64, 64, 33, True),
        ("cross-padding", 32, 64, 33, False),
        ("encoder-padding", 64, 64, 33, False),
    ]:
        q = torch.randn(1, 16, L, 64).bfloat16().float()
        k = torch.randn(1, 16, S, 64).bfloat16().float()
        v = torch.randn(1, 16, S, 64).bfloat16().float()
        allow = (torch.arange(S)[None, :] < valid).expand(L, S).clone()
        if causal:
            allow &= torch.arange(S)[None, :] <= torch.arange(L)[:, None]
        mask = torch.where(allow, 0.0, -1e9)[None, None].bfloat16().float()
        qs = (q * 0.125).bfloat16().float()
        ref = torch.softmax(qs @ k.transpose(-2, -1) + mask, -1) @ v
        tq, tk, tv, tm = map(up, [qs, k, v, mask])
        manual = ttnn.matmul(
            ttnn.softmax(
                ttnn.add(ttnn.matmul(tq, tk, transpose_b=True, compute_kernel_config=kernel), tm),
                dim=-1,
                compute_kernel_config=kernel,
                numeric_stable=True,
            ),
            tv,
            compute_kernel_config=kernel,
        )
        fused = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, attn_mask=tm, is_causal=False, scale=1.0, program_config=pc, compute_kernel_config=kernel
        )
        a, b = cpu(manual), cpu(fused)
        nrmse = lambda x: float((((x - ref) ** 2).mean(-1).sqrt() / ref.square().mean(-1).sqrt().clamp_min(1e-8)).max())
        assert torch.isfinite(b).all() and nrmse(b) < 0.04, (name, nrmse(b))
        # Future/padded-key poisoning must not change real query outputs.
        poisoned = v.clone()
        poisoned[:, :, valid:] = 1000.0
        c = cpu(
            ttnn.transformer.scaled_dot_product_attention(
                tq,
                tk,
                up(poisoned),
                attn_mask=tm,
                is_causal=False,
                scale=1.0,
                program_config=pc,
                compute_kernel_config=kernel,
            )
        )
        assert torch.equal(b, c), name
        if causal:
            v2 = v.clone()
            v2[:, :, 1:] = 1000.0
            c = cpu(
                ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    up(v2),
                    attn_mask=tm,
                    is_causal=False,
                    scale=1.0,
                    program_config=pc,
                    compute_kernel_config=kernel,
                )
            )
            assert torch.equal(b[:, :, 0], c[:, :, 0]), name
        # Reuse exact cross K/V tensors with changed Q, compare a fresh upload.
        if name.startswith("cross"):
            tq2 = up(qs * 0.5)
            c = cpu(
                ttnn.transformer.scaled_dot_product_attention(
                    tq2,
                    tk,
                    tv,
                    attn_mask=tm,
                    is_causal=False,
                    scale=1.0,
                    program_config=pc,
                    compute_kernel_config=kernel,
                )
            )
            d = cpu(
                ttnn.transformer.scaled_dot_product_attention(
                    tq2,
                    up(k),
                    up(v),
                    attn_mask=tm,
                    is_causal=False,
                    scale=1.0,
                    program_config=pc,
                    compute_kernel_config=kernel,
                )
            )
            assert torch.equal(c, d), "cached cross KV"
        print(
            "CHECK "
            + json.dumps(
                dict(
                    name=name,
                    manual_nrmse=nrmse(a),
                    fused_nrmse=nrmse(b),
                    max_manual_fused_abs=float((a - b).abs().max()),
                    mask_invariance=True,
                )
            ),
            flush=True,
        )


def test_fused_attention(nllb_component_runner):
    nllb_component_runner("fused_attention")


if __name__ == "__main__":
    torch.set_num_threads(1)
    device = ttnn.open_device(device_id=0)
    try:
        check(device)
    finally:
        ttnn.close_device(device)
