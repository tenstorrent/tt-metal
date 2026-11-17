# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein precisions (host only): the default keeps every layer's own matmul config, the fast mode names LoFi."""

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import PRECISION_BF16, PRECISION_FAST


def test_precision() -> None:
    arch = ttnn.device.Arch.BLACKHOLE
    assert PRECISION_BF16.weight_dtype == ttnn.bfloat16 and PRECISION_BF16.matmul_config(arch) is None
    config = PRECISION_FAST.matmul_config(arch)
    assert PRECISION_FAST.weight_dtype == ttnn.bfloat8_b and config.math_fidelity == ttnn.MathFidelity.LoFi
    assert config.fp32_dest_acc_en and config.packer_l1_acc and not config.math_approx_mode
    # the weight cache keys: existing bf16 and fast-mode caches stay valid
    assert (PRECISION_BF16.name, PRECISION_FAST.name) == ("bf16", "bf8_lofi")
