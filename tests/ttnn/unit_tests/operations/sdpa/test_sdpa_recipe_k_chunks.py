# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K256/K384 recipe blocking keeps each recipe's arithmetic choices.

A K chunk sets the online-softmax update cadence, the PV partial-sum grouping
and which K chunks the compensated recipes pair, so outputs legitimately differ
from K512. Gate on determinism and FP64 accuracy relative to K512 on the same
inputs; K512 itself stays gated by the frozen digests.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare, reference

PAIRED_VARIANTS = ("B", "E_bf16", "E_bfp8", "E_bfp4")


def options(variant, grid, q_chunk, k_chunk):
    return dict(
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
    )


def upload(device, host, variant):
    return prepare([ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host], variant)


def invoke(segments, variant, grid, q_chunk, k_chunk):
    kwargs = options(variant, grid, q_chunk, k_chunk)
    if len(segments) == 1:
        return [ttnn.transformer.scaled_dot_product_attention(*segments[0], is_causal=False, **kwargs)]
    return list(
        ttnn.transformer.joint_scaled_dot_product_attention(
            *segments[0], *segments[1], joint_strategy="rear", **kwargs
        )
    )


def run(segments, variant, grid, q_chunk, k_chunk):
    return torch.cat([ttnn.to_torch(x) for x in invoke(segments, variant, grid, q_chunk, k_chunk)], dim=2)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "q_chunk,k_chunk", [(256, 256), (256, 384), (128, 256), (320, 256), (224, 384)], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "q_length,k_length,heads,grid,distribution,joint_rows",
    [
        (1024, 3072, 2, (4, 1), "normal", None),
        (1000, 2000, 1, (3, 1), "changed_max", None),
        (512, 8192, 1, (2, 1), "uniform", None),
        (768, 1536, 1, (2, 1), "constant_v", None),
        (1033, 1033, 2, (4, 1), "common_k", (333, 333)),
    ],
    ids=["aligned", "tails", "long_uniform", "constant_v", "joint_tails"],
)
def test_recipe_k_chunk_preserves_accuracy(
    device,
    variant,
    q_chunk,
    k_chunk,
    q_length,
    k_length,
    heads,
    grid,
    distribution,
    joint_rows,
    record_property,
):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(k_length, distribution, q_length=q_length, heads=heads)
    if joint_rows is None:
        segments = [upload(device, host, variant)]
    else:
        q_rows, kv_rows = joint_rows
        split = list(zip(host, (q_rows, kv_rows, kv_rows)))
        segments = [
            upload(device, [x[..., :-rows, :].contiguous() for x, rows in split], variant),
            upload(device, [x[..., -rows:, :].contiguous() for x, rows in split], variant),
        ]
    if variant in PAIRED_VARIANTS and (q_chunk // 32) % 2:
        with pytest.raises(RuntimeError, match="multiple of 64"):
            invoke(segments, variant, grid, q_chunk, k_chunk)
        return
    try:
        actual = run(segments, variant, grid, q_chunk, k_chunk)
    except RuntimeError as error:
        assert "bytes of L1" in str(error), error
        record_property("rejected_l1", True)
        pytest.skip(f"{variant} Q{q_chunk}/K{k_chunk} exceeds Blackhole L1")
    rerun = run(segments, variant, grid, q_chunk, k_chunk)
    assert digest(rerun) == digest(actual), "Recipe output must be deterministic for a fixed geometry"
    assert torch.isfinite(actual.float()).all()
    exact = reference(*host)
    baseline = metrics(run(segments, variant, grid, 256, 512), exact)
    observed = metrics(actual, exact)
    for key, value in observed.items():
        record_property(key, value)
    record_property("k512_l2_pct", baseline["l2_pct"])
    # A smaller K chunk adds online-softmax updates; allow a modest error increase over K512.
    assert observed["l2_pct"] <= baseline["l2_pct"] * 1.25 + 0.02, (observed["l2_pct"], baseline["l2_pct"])
    if distribution == "constant_v":
        # Softmax weights sum to one: constant V must come back (nearly) exactly, independent of K blocking.
        assert observed["max_abs"] <= max(2 * baseline["max_abs"], 1 / 64), (observed["max_abs"], baseline["max_abs"])


@pytest.mark.parametrize("k_chunk", [128, 320, 640, 1024])
def test_recipe_rejects_unsupported_k_chunk(device, k_chunk):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(1024, "normal", q_length=256)
    with pytest.raises(RuntimeError, match="K chunks of 256, 384 or 512"):
        invoke([upload(device, host, "D")], "D", (1, 1), 256, k_chunk)
