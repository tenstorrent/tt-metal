# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Isolation: Ideogram4 hi-res (1024/2048px) washed-out latent.
#
# Hypothesis: of all the block's ops, only attention has sequence-length-dependent
# accumulation (RMSNorm + qkv/o/FFN matmuls are per-token). The flat latent is a
# systematic bias toward the mean = a too-flat softmax. Flash attention with
# k_chunk=256 runs 16 K-chunks at 4096 tokens, 64 at 16384; if the online-softmax
# stats lose precision across many chunks the softmax flattens.
#
# This standalone test reproduces the block's SDPA exactly (head_dim=256, RMS-normed
# q/k, the block's HiFi4 + fp32-acc compute config) and sweeps seq x k_chunk against
# an fp32 torch golden — cheaply, with no 9.3B model.
# =============================================================================

import pytest
import torch
from loguru import logger

import ttnn

from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

HEAD_DIM = 256
NUM_HEADS = 4  # a few heads is enough to measure softmax behavior; head_dim drives the L1/precision behavior


def _rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq", [1024, 4096, 16384], ids=["1024tok", "4096tok", "16384tok"])
@pytest.mark.parametrize("k_chunk", [256, 512, 1024], ids=["k256", "k512", "k1024"])
def test_sdpa_seq_kchunk(*, mesh_device: ttnn.MeshDevice, seq: int, k_chunk: int) -> None:
    torch.manual_seed(0)
    b, h, d = 1, NUM_HEADS, HEAD_DIM

    # Realistic distribution into SDPA: q,k are RMS-normed over head_dim (norm_q/norm_k),
    # v is a projection output (unit-ish gaussian).
    q = _rms_norm(torch.randn(b, h, seq, d, dtype=torch.float32))
    k = _rms_norm(torch.randn(b, h, seq, d, dtype=torch.float32))
    v = torch.randn(b, h, seq, d, dtype=torch.float32)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # fp32 golden

    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=128,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    tt_q = bf16_tensor(q, device=mesh_device)
    tt_k = bf16_tensor(k, device=mesh_device)
    tt_v = bf16_tensor(v, device=mesh_device)
    try:
        out = ttnn.transformer.scaled_dot_product_attention(
            tt_q, tt_k, tt_v, is_causal=False, program_config=prog, compute_kernel_config=compute
        )
    except RuntimeError as e:
        if "Out of Memory" in str(e) or "L1" in str(e):
            pytest.skip(f"seq={seq} k_chunk={k_chunk} OOM: {str(e)[:80]}")
        raise
    tt_out = tensor.to_torch(out, mesh_axes=[None, None, None, None])

    # Report std ratio too: a flattened softmax shrinks the output magnitude.
    std_ratio = tt_out.float().std().item() / ref.std().item()
    logger.info(f"SDPA seq={seq} k_chunk={k_chunk}: out/ref std ratio={std_ratio:.4f}")
    assert_quality(ref, tt_out.float(), pcc=0.99)
