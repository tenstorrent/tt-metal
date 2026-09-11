# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise Sentence-BERT's formerly legacy-rsqrt configuration without model downloads."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skipif(not is_blackhole(), reason="Uses the Blackhole Sentence-BERT program configuration")
@pytest.mark.parametrize("math_approx_mode", [False, True])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_sentence_bert_norm_config(device, math_approx_mode, fp32_dest_acc_en):
    from models.demos.blackhole.sentence_bert.ttnn.common import layernorm_program_config

    torch.manual_seed(56277)
    values = torch.randn(8, 1, 384, 768, dtype=torch.bfloat16)
    residual = torch.randn_like(values)
    memory_config = ttnn.create_sharded_memory_config(
        values.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    inputs = [
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
        for x in (values, residual)
    ]
    output = ttnn.layer_norm(
        inputs[0],
        residual_input_tensor=inputs[1],
        epsilon=1e-5,
        program_config=layernorm_program_config,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
    )
    actual = ttnn.to_torch(output).float()
    expected = torch.nn.functional.layer_norm(values.float() + residual.float(), (768,), eps=1e-5)
    assert torch.isfinite(actual).all()
    assert_with_pcc(expected, actual, 0.999)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
