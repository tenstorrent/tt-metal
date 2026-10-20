# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Diagnostic, not a gate: ttnn SDPA on recorded structure-DiT q/k/v (bf16) against torch fp64 SDPA on the same values,
per program/compute config: relative error and scale bias <e, ref> / <ref, ref>. The structure DiT's last-block
self-attention output was ~0.4% too large (S08). Runs only with TRELLIS2_SDPA_PROBE=<qkv.pt>."""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

PATH = os.environ.get("TRELLIS2_SDPA_PROBE")
_H4 = ttnn.MathFidelity.HiFi4
CONFIGS = {  # name: (q_chunk, k_chunk, fidelity, fp32 dest acc)
    "q128_k128_hifi4_fp32": (128, 128, _H4, True),
    "q128_k256_hifi4_fp32": (128, 256, _H4, True),
    "q128_k512_hifi4_fp32": (128, 512, _H4, True),
    "q64_k64_hifi4_fp32": (64, 64, _H4, True),
    "q256_k256_hifi4_fp32": (256, 256, _H4, True),
    "q128_k128_hifi2_bf16": (128, 128, ttnn.MathFidelity.HiFi2, False),
    "q128_k1024_hifi4_fp32": (128, 1024, _H4, True),
    "q64_k1024_hifi4_fp32": (64, 1024, _H4, True),
    "q64_k2048_hifi4_fp32": (64, 2048, _H4, True),
    "q32_k2048_hifi4_fp32": (32, 2048, _H4, True),
    "q32_k4096_hifi4_fp32": (32, 4096, _H4, True),
    "q128_k128_hifi4_bf16": (128, 128, _H4, False),
}
if os.environ.get("TRELLIS2_SDPA_CONFIGS"):
    CONFIGS = {k: CONFIGS[k] for k in os.environ["TRELLIS2_SDPA_CONFIGS"].split(",")}

pytestmark = pytest.mark.skipif(not PATH, reason="set TRELLIS2_SDPA_PROBE=<qkv.pt>")


def _stats(a: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    e, r = (a - ref).double().flatten(), ref.double().flatten()
    return (e.norm() / r.norm()).item(), (e @ r / (r @ r)).item()


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sdpa_probe(mesh_device, config):
    d = torch.load(PATH, weights_only=False)
    qc, kc, fidelity, fp32_acc = CONFIGS[config]
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    q, k, v = to_dev(d["q"]), to_dev(d["k"]), to_dev(d["v"])
    run = lambda: ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc
        ),
    )
    out = run()
    ttnn.synchronize_device(mesh_device)
    start = time.time()
    for _ in range(10):
        run()
    ttnn.synchronize_device(mesh_device)
    ms = (time.time() - start) * 100
    out = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    rel, bias = _stats(out, d["ref"])
    # the same output rounded to bf16 on the host: what a perfect kernel with a bf16 output would score
    rel_b, bias_b = _stats(d["ref"].bfloat16().float(), d["ref"])
    logger.info(
        f"SDPA block {d['block']} {config}: rel {rel:.3e} bias {bias:+.3e} {ms:.2f} ms (bf16-rounded exact: "
        f"rel {rel_b:.3e} bias {bias_b:+.3e})"
    )
