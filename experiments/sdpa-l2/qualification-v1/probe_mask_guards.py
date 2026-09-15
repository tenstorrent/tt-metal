# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit-mask structural cases are unsupported by both specializations."""

import json
import math

import torch
import ttnn

from qualify import repro

torch.set_num_threads(4)
device = ttnn.open_device(device_id=0)
try:
    q, k, v = repro.make_inputs(5, 128, 32768, 128, 1234, "normal")
    for mode in ("fast", "accurate"):
        fp32 = mode == "accurate"
        dq = repro.preprocess_query(q, 6, 1.0027, True) if fp32 else q
        tensors = [
            ttnn.from_torch(
                x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for x in (dq, k, v)
        ]
        try:
            for case in ("single_unmasked_key", "fully_masked_rows", "finite_masked_value_perturbation"):
                mask = torch.full((1, 1, 128, 32768), -math.inf, dtype=torch.bfloat16)
                if case != "fully_masked_rows":
                    mask[..., 0] = 0
                tm = ttnn.from_torch(
                    mask,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                try:
                    result = ttnn.transformer.scaled_dot_product_attention(
                        *tensors,
                        attn_mask=tm,
                        is_causal=False,
                        program_config=ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            q_chunk_size=128,
                            k_chunk_size=1024 if fp32 else 512,
                            exp_approx_mode=True,
                        ),
                        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi2,
                            math_approx_mode=True,
                            fp32_dest_acc_en=fp32,
                            packer_l1_acc=False,
                        ),
                        scale=1 / (math.sqrt(128) * (1.0027 if fp32 else 1)),
                    )
                except RuntimeError as exc:
                    expected = (
                        "QUALIFICATION_UNSUPPORTED_FP32_STREAMING"
                        if fp32
                        else "QUALIFICATION_UNSUPPORTED_BF16_COMPENSATION"
                    )
                    if expected not in str(exc):
                        raise
                    print(
                        json.dumps(
                            dict(
                                case=case,
                                mode=mode,
                                status="UNSUPPORTED",
                                fallback_executed=False,
                                reason="explicit mask excluded by specialization",
                                guard_rejection=True,
                            )
                        ),
                        flush=True,
                    )
                else:
                    ttnn.deallocate(result)
                    raise AssertionError("Unexpected acceptance of excluded mask")
                finally:
                    ttnn.deallocate(tm)
        finally:
            for t in tensors:
                ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
