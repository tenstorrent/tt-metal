# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import VARIANTS, digest, metrics, reference
from .test_sdpa_joint_recipes import joint, joined, options, upload


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batch,q_heads,kv_heads", [(1, 4, 1), (1, 6, 2), (2, 4, 2), (2, 3, 3)])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_recipe_batch_gqa(device, variant, batch, q_heads, kv_heads, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    generator = torch.Generator().manual_seed(20260923)
    host = [
        torch.randn((batch, heads, length, 128), generator=generator).bfloat16()
        for heads, length in ((q_heads, 768), (kv_heads, 1536), (kv_heads, 1536))
    ]
    expanded = [host[0], *(x.repeat_interleave(q_heads // kv_heads, dim=1) for x in host[1:])]
    grid = (8, 2)
    inputs = upload(device, host, variant)
    actual = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, **options(variant, grid))
    )
    dense = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            *upload(device, expanded, variant), is_causal=False, **options(variant, grid)
        )
    )
    assert digest(actual) == digest(dense)
    observed = metrics(actual, reference(*expanded))
    limits = {"A": 8, "B": 8, "C": 2, "D": 0.4, "E_bf16": 8, "E_bfp8": 8, "E_bfp4": 35}
    assert observed["l2_pct"] < limits[variant]
    for key, value in observed.items():
        record_property(key, value)
    segments = [
        [x[..., :split, :].contiguous() for x, split in zip(host, (512, 1024, 1024))],
        [x[..., split:, :].contiguous() for x, split in zip(host, (512, 1024, 1024))],
    ]
    joint_inputs = [upload(device, segment, variant) for segment in segments]
    hashes = [digest(ttnn.to_torch(x)) for segment in joint_inputs for x in segment]
    assert digest(joined(joint(joint_inputs, variant, grid))) == digest(actual)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = joint(joint_inputs, variant, grid)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert digest(joined(traced)) == digest(actual)
    finally:
        ttnn.release_trace(device, trace)
    assert hashes == [digest(ttnn.to_torch(x)) for segment in joint_inputs for x in segment]


@pytest.mark.parametrize("invalid", ["head_ratio", "batch", "joint_kv_heads"])
def test_recipe_gqa_rejects_mismatch(device, invalid):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = [
        torch.zeros((1, 4, 256, 128), dtype=torch.bfloat16),
        torch.zeros(
            (2 if invalid == "batch" else 1, 3 if invalid == "head_ratio" else 2, 512, 128), dtype=torch.bfloat16
        ),
    ]
    host.append(host[-1].clone())
    inputs = upload(device, host, "D")
    if invalid == "joint_kv_heads":
        other = upload(device, [host[0], *(x[:, :1].contiguous() for x in host[1:])], "D")
    with pytest.raises(RuntimeError, match="SDPA recipes require matching K/V"):
        if invalid == "joint_kv_heads":
            joint([inputs, other], "D", (4, 1))
        else:
            ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, **options("D", (4, 1)))
