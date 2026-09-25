# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Q-chunk generalization keeps each recipe's arithmetic and accuracy.

Outputs are not bit-identical to Q256. Phase 2 overlaps the final row's
exponential with the first PV row group of every Q chunk, accumulating that
group's product in L1 over 4-tile K partials, while later groups accumulate
all K tiles in dest. Which rows sit in a chunk's first group therefore changes
their rounding (about one BF16 ulp). Q256 itself stays gated by the frozen
digests. Other Q chunks are gated on determinism, a tight bound on the
difference from Q256, and FP64 accuracy no worse than Q256.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare, reference

Q_CHUNKS = (128, 192, 224, 288, 320)
UNSUPPORTED_Q_CHUNKS = (100, 1056)


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
def test_recipe_q_chunk_preserves_accuracy(
    device, variant, q_chunk_size, q_length, k_length, heads, grid, distribution, joint_rows, record_property
):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(k_length, distribution, q_length=q_length, heads=heads)
    if joint_rows is None:
        segments = [upload(device, host, variant)]
    else:
        # Split on the host so each segment keeps its own minimal tile padding.
        q_rows, kv_rows = joint_rows
        split = list(zip(host, (q_rows, kv_rows, kv_rows)))
        segments = [
            upload(device, [x[..., :-rows, :].contiguous() for x, rows in split], variant),
            upload(device, [x[..., -rows:, :].contiguous() for x, rows in split], variant),
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
    rerun = torch.cat([ttnn.to_torch(x) for x in invoke(segments, variant, grid, q_chunk_size)], dim=2)
    assert digest(rerun) == digest(actual), "Recipe output must be deterministic for a fixed Q chunk"
    assert torch.isfinite(actual.float()).all()
    exact = reference(*host)
    observed, baseline = metrics(actual, exact), metrics(expected, exact)
    delta = (actual.float() - expected.float()).abs().max().item()
    for key, value in observed.items():
        record_property(key, value)
    record_property("q256_l2_pct", baseline["l2_pct"])
    record_property("max_abs_delta_vs_q256", delta)
    record_property("bitwise_equal_q256", digest(actual) == digest(expected))
    if distribution == "zero_v":
        assert observed["max_abs"] <= 1e-6
    else:
        assert observed["l2_pct"] <= baseline["l2_pct"] * 1.05 + 0.01, (observed["l2_pct"], baseline["l2_pct"])
    # Only rounding order moves; a larger change means a real indexing or state bug.
    assert delta <= 0.02, delta


@pytest.mark.parametrize("q_chunk_size", UNSUPPORTED_Q_CHUNKS)
def test_recipe_rejects_unsupported_q_chunk(device, q_chunk_size):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(512, "normal", q_length=512)
    inputs = upload(device, host, "D")
    with pytest.raises(RuntimeError, match="tile-aligned Q chunks from 32 to 1024 rows"):
        invoke([inputs], "D", (2, 1), q_chunk_size)
