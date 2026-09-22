# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The package's compute-kernel default for every projection matmul.

The recipe (§2.3) fixes this: *"The bring-up default for every projection matmul and the plain
SDPA is HiFi4 with ``fp32_dest_acc_en=True``. A narrower setting is a measurement, not an
inheritance."* This module is that default, in one place, so a matmul cannot quietly acquire a
different one by being written without a ``compute_kernel_config`` argument.

**Why it is a module and not a keyword default.** ttnn's own default is
``math_fidelity=LoFi, math_approx_mode=True, fp32_dest_acc_en=False``
(``init_device_compute_kernel_config``), so *omitting* the argument is not neutral — it selects the
lowest fidelity the hardware has. LoFi multiplies only the high ~5 mantissa bits of each operand;
with ``bfloat8_b`` weights that is most of the mantissa the weight had. A single matmul at LoFi
still correlates ~0.999 with the reference, which is why every block-level ``*_vs_ref`` test passed
with it, but the residual stream accumulates the error across all 88 layers: measured at full depth
against the golden trace, LoFi gave worst-layer ``k 0.810 / v 0.552`` — below the spec's
``pcc_lower_bound`` — and HiFi4 with fp32 accumulation gives the numbers in ``README.md``.

That is the general lesson this file exists to pin: **depth is the test for fidelity settings.** A
per-block PCC cannot see a 0.1%-per-layer bias, and this model has 88 layers of it.

``fp32_dest_acc_en`` is *not* universal, though — it is only the default here. The ring cache-read
op (``ring_joint_scaled_dot_product_attention``) genuinely does not support it, so
``ProgramConfig.get_ring_compute_kernel_config`` forces it off. That constraint is local to that
op, and a source that applies it globally (this package did, until the full-depth run) is
over-broad in exactly the way §2.3 warns about.
"""

import ttnn

#: HiFi4 — all four mantissa passes. ``math_approx_mode=False`` because the approximate
#: transcendental path is a second, independent accuracy loss and nothing here needs its speed.
MATH_FIDELITY = ttnn.MathFidelity.HiFi4
MATH_APPROX_MODE = False
#: fp32 accumulation in the destination register. The bf16 accumulator loses low bits over the
#: 12288-long (attention/MLP-down) and 28672-long (MLP-up) reduction dimensions this model has.
FP32_DEST_ACC_EN = True
#: fp32 accumulation in the packer for the matmul's inner-dim partials, for the same reason.
PACKER_L1_ACC = True


def matmul_compute_kernel_config() -> "ttnn.WormholeComputeKernelConfig":
    """The compute-kernel config every projection matmul in this package passes.

    Used by ``tt/mlp.py`` (gate/up/down), ``tt/attention/operations.py`` (fused QKV, o_proj) and
    ``tt/lm_head.py``. ``WormholeComputeKernelConfig`` is the struct name on Blackhole too — it is
    the non-Grayskull config, not a Wormhole-only one.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=MATH_APPROX_MODE,
        fp32_dest_acc_en=FP32_DEST_ACC_EN,
        packer_l1_acc=PACKER_L1_ACC,
    )
