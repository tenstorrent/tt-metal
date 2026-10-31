# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_FUSED=final_mixer against the stock chain (Qwen38TTNNFinalMixer.__call__, 22 programs at TP=8), BIT FOR BIT
on the TP=8 mesh with the real checkpoint's mixer weights (BFP8 default and BF16), many random residuals; plus the
traced slope per call of both forms.  Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest this file -s."""

import functools
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, RESIDUAL_BRANCHES
from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, TP_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer, Qwen38TTNNFinalMixerWeights

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP mesh")
CALLS = int(os.environ.get("FM_CALLS", "12"))
PARAMS = {**DEVICE_PARAMS, "trace_region_size": 64 * 1024 * 1024}


def per_die(tensor):
    return [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("dtype_name", ["bf8", "bf16"])
@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
@pytest.mark.parametrize("namespace", ["backbone", "mtp"])
def test_final_mixer_fused_bitwise_and_timing(tp_harness, dtype_name, namespace, merged):
    h = tp_harness
    dtype = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[dtype_name]
    weights = Qwen38TTNNFinalMixerWeights.from_checkpoint(
        h.checkpoint,
        h.placement,
        h.mesh_device,
        h.contract,
        h.component_cache_root,
        namespace=namespace,
        tt_metal_sha=h.tt_metal_sha,
        weight_dtype=dtype,
    )
    mixer = Qwen38TTNNFinalMixer(h.mesh_device, h.contract, weights, collective_topology=h.topology)
    assert mixer._fused_forward is None or fused_kernels.enabled("final_mixer")
    chain_forward = mixer._fused_forward
    mixer._fused_forward = None
    fused_forward = functools.partial(fused_kernels.kernel("final_mixer").fused, mixer, merged=merged)

    gen = torch.Generator().manual_seed(77)
    hidden = 2560
    bad = 0
    for call in range(CALLS):
        scale = 0.25 + 0.7 * call
        host = torch.randn(1, RESIDUAL_BRANCHES, 1, hidden, generator=gen) * scale
        if call % 4 == 3:  # heavy tails and a dead branch
            host = host * torch.exp2(torch.randint(-6, 7, host.shape, generator=gen).float())
            host[:, call % RESIDUAL_BRANCHES] = 0.0
        host = host.to(torch.bfloat16)
        residual = h.upload_sharded(host)
        base = mixer(residual)
        base_bits = per_die(base)
        ttnn.deallocate(base)
        out = fused_forward(residual)
        out_bits = per_die(out)
        ttnn.deallocate(out)
        base2 = mixer(residual)  # the chain is deterministic call to call
        assert all(torch.equal(a, b) for a, b in zip(base_bits, per_die(base2)))
        ttnn.deallocate(base2)
        for a, b in zip(base_bits, out_bits):
            bad += int((a != b).sum())
        ttnn.deallocate(residual)
    logger.info(
        f"final_mixer {namespace} {dtype_name} merged={merged} tp={TP_SIZE}: {CALLS} calls, differing bf16 words {bad}"
    )
    assert bad == 0

    residual = h.upload_sharded(torch.randn(1, RESIDUAL_BRANCHES, 1, hidden, generator=gen).to(torch.bfloat16))

    def run(forward):
        def call():
            ttnn.deallocate(forward(residual))

        return call

    stock_us, _ = traced_wall_us(h.mesh_device, run(mixer.__call__))
    fused_us, _ = traced_wall_us(h.mesh_device, run(fused_forward))
    logger.info(
        f"final_mixer {namespace} {dtype_name} merged={merged} traced slope per call: stock {stock_us:.1f} us, fused {fused_us:.1f} us"
    )
    mixer._fused_forward = chain_forward
