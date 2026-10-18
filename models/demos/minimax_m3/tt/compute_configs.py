# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute kernel configs for the non-expert matmuls and the RMSNorms.

Matmuls (QKV, index_q/k, o_proj, dense MLP, shared expert): a matmul's weight (in1) feeds SrcA and its activation
(in0) SrcB. Below HiFi3 SrcB keeps 7 of a bf16 activation's 8 significant bits. HiFi3 skips only A_lo x B_lo, which
is zero for MXFP8-origin weights (at most 4 significant bits, also after bfp8 packing, so they fit SrcA's high
slice): exact at 3 of HiFi4's 4 passes. For bf16-origin weights the skipped term is below 2^-12 relative.
ttnn's default for a bf16 output accumulates in a bf16 DEST (1.4-2% error on the K=6144 projections); fp32 DEST and
fp32 L1 partials remove that.

RMSNorm: the default (bf16 DEST) sums x^2 in bf16, a 1-2% (up to 12%) per-token scale error at deep layers.
fp32 DEST keeps the statistics fp32; exact rsqrt.
"""

import functools

import ttnn


@functools.lru_cache(maxsize=None)
def _matmul_config(arch):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@functools.lru_cache(maxsize=None)
def _rms_norm_config(arch):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def matmul_compute_config(device):
    """HiFi3, fp32 accumulation: bf16 activation x MXFP8-origin weight, exact products."""
    return _matmul_config(device.arch())


def rms_norm_compute_config(device):
    """fp32 statistics, exact rsqrt."""
    return _rms_norm_config(device.arch())
