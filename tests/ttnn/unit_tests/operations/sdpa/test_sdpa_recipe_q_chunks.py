# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Q-chunk generalization must not change any row's arithmetic.

Each query row sees the same K chunks in the same order with the same
reduction width regardless of how rows are grouped into Q chunks, so every
supported Q chunk must reproduce the Q256 output bit-for-bit within a build.
The Q256 path itself remains gated by the frozen digests elsewhere.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, prepare

Q_CHUNKS = (128, 192, 320)
# Odd tile counts need single-row groups in the compensated/FP32 state updates.
UNSUPPORTED_Q_CHUNKS = (96, 160, 224, 288, 352)


def options(variant, grid, q_chunk_size):
    return dict(
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk_size, k_chunk_size=512
        ),
    )


def upload(device, host, variant):
    return prepare([ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host], variant)


def invoke(segments, variant, grid, q_chunk_size):
    kwargs = options(variant, grid, q_chunk_size)
    if len(segments) == 1:
        return [ttnn.transformer.scaled_dot_product_attention(*segments[0], is_causal=False, **kwargs)]
    return list(
        ttnn.transformer.joint_scaled_dot_product_attention(
            *segments[0], *segments[1], joint_strategy="rear", **kwargs
        )
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("q_chunk_size", Q_CHUNKS)
@pytest.mark.parametrize(
    "q_length,k_length,heads,grid,distribution,joint_rows",
    [
        (1024, 1536, 2, (4, 1), "normal", None),
        (1000, 1025, 1, (3, 1), "changed_max", None),
        (2304, 2048, 1, (8, 1), "uniform", None),
        (1033, 1033, 2, (4, 1), "common_k", (333, 333)),
    ],
    ids=["aligned", "tails", "long_uniform", "joint_tails"],
)
def test_recipe_q_chunk_matches_q256(
    device, variant, q_chunk_size, q_length, k_length, heads, grid, distribution, joint_rows, record_property
):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(k_length, distribution, q_length=q_length, heads=heads)
    if joint_rows is None:
        segments = [upload(device, host, variant)]
    else:
        # Split on the host so each segment keeps its own minimal tile padding.
        segments = [
            upload(device, [x[..., :-rows, :].contiguous() for x, rows in zip(host, joint_rows)], variant),
            upload(device, [x[..., -rows:, :].contiguous() for x, rows in zip(host, joint_rows)], variant),
        ]
    expected = torch.cat([ttnn.to_torch(x) for x in invoke(segments, variant, grid, 256)], dim=2)
    try:
        outputs = invoke(segments, variant, grid, q_chunk_size)
    except RuntimeError as error:
        # The host rejects layouts that exceed unreserved L1 before dispatch.
        assert "bytes of L1" in str(error), error
        record_property("rejected_l1", True)
        pytest.skip(f"{variant} Q{q_chunk_size}/K512 exceeds Blackhole L1")
    actual = torch.cat([ttnn.to_torch(x) for x in outputs], dim=2)
    assert torch.isfinite(actual.float()).all()
    equal = digest(actual) == digest(expected)
    record_property("bitwise_equal_q256", equal)
    assert equal, f"Q{q_chunk_size} changed the {variant} output relative to Q256"


@pytest.mark.parametrize("q_chunk_size", UNSUPPORTED_Q_CHUNKS)
def test_recipe_rejects_unsupported_q_chunk(device, q_chunk_size):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(512, "normal", q_length=512)
    inputs = upload(device, host, "D")
    with pytest.raises(RuntimeError, match="Q chunks of 128, 192, 256 or 320"):
        invoke([inputs], "D", (2, 1), q_chunk_size)
