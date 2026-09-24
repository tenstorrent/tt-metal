# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic recipe geometry: any tile-aligned Q chunk, K chunk and head dim that fits L1.

The numerical implementation of each recipe is the contract; geometry only changes blocking,
so every result is gated on determinism and on FP64 accuracy relative to the same recipe at the
qualified Q256/K512 blocking (and, for new head dims, the qualified D128 recipe) on the same draw."""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare


def reference(q, k, v):
    q, k, v = q.double(), k.double(), v.double()
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    return torch.softmax(scores, dim=-1) @ v


def inputs_with_head_dim(head_dim, k_length, distribution, q_length, heads=1):
    """Head dims up to 128 slice one D128 draw; larger ones concatenate a second draw."""
    first = make_inputs(k_length, distribution, q_length=q_length, heads=heads)
    if head_dim <= 128:
        return [x[..., :head_dim].contiguous() for x in first]
    second = make_inputs(k_length, distribution, q_length=q_length, heads=heads, seed=20261123)
    return [torch.cat([a, b], dim=-1)[..., :head_dim].contiguous() for a, b in zip(first, second)]


def run(device, host, variant, grid, q_chunk, k_chunk, joint_rows=None):
    kwargs = dict(
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
    )

    def upload(values):
        return prepare([ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in values], variant)

    if joint_rows is None:
        outputs = [ttnn.transformer.scaled_dot_product_attention(*upload(host), is_causal=False, **kwargs)]
    else:
        q_rows, kv_rows = joint_rows
        split = list(zip(host, (q_rows, kv_rows, kv_rows)))
        outputs = ttnn.transformer.joint_scaled_dot_product_attention(
            *upload([x[..., :-rows, :].contiguous() for x, rows in split]),
            *upload([x[..., -rows:, :].contiguous() for x, rows in split]),
            joint_strategy="rear",
            **kwargs,
        )
    return torch.cat([ttnn.to_torch(x) for x in outputs], dim=2)


def run_or_skip(*args, **kwargs):
    try:
        return run(*args, **kwargs)
    except RuntimeError as error:
        assert "bytes of L1" in str(error), error
        pytest.skip(f"geometry exceeds Blackhole L1: {error}")


# (q_chunk, k_chunk, head_dim): Q, K and D swept one at a time around the qualified geometry,
# then combined odd geometries.
GEOMETRIES = [
    *[(q, 512, 128) for q in (32, 64, 96, 160, 384, 512, 1024)],
    *[(256, k, 128) for k in (32, 64, 96, 128, 160, 640, 1024)],
    *[(256, 256, d) for d in (32, 96, 160, 192)],
    (32, 32, 32),
    (96, 96, 96),
    (160, 160, 160),
    (64, 1024, 64),
    (512, 128, 96),
    (1024, 128, 64),  # Q512/Q1024 do not fit L1 at K512/D128
]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("q_chunk,k_chunk,head_dim", GEOMETRIES, ids=lambda x: str(x))
@pytest.mark.parametrize(
    "q_length,k_length,grid,distribution,joint_rows",
    [
        (1000, 1500, (4, 1), "normal", None),
        (1024, 2048, (4, 1), "changed_max", None),
        (1033, 1033, (4, 1), "common_k", (333, 333)),
    ],
    ids=["tails", "changed_max", "joint_tails"],
)
def test_recipe_geometry(
    device,
    variant,
    q_chunk,
    k_chunk,
    head_dim,
    q_length,
    k_length,
    grid,
    distribution,
    joint_rows,
    record_property,
):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = inputs_with_head_dim(head_dim, k_length, distribution, q_length)
    actual = run_or_skip(device, host, variant, grid, q_chunk, k_chunk, joint_rows)
    assert digest(run(device, host, variant, grid, q_chunk, k_chunk, joint_rows)) == digest(actual)
    assert actual.shape[-1] == head_dim and torch.isfinite(actual.float()).all()
    observed = metrics(actual, reference(*host))
    for key, value in observed.items():
        record_property(key, value)

    # Calibrate against the qualified blocking. A new head dim is compared with the qualified D128
    # recipe on the same generator (its error grows with sqrt(D) for the concatenated draws).
    if head_dim in (64, 128, 256):
        qualified_host = host
    else:
        qualified_host = inputs_with_head_dim(128, k_length, distribution, q_length)
    qualified = metrics(
        run_or_skip(device, qualified_host, variant, grid, 256, 512 if head_dim <= 128 else 256, joint_rows),
        reference(*qualified_host),
    )
    record_property("qualified_l2_pct", qualified["l2_pct"])
    record_property("qualified_max_abs", qualified["max_abs"])
    slack = 1.5 if head_dim <= 128 else 2.5
    assert observed["l2_pct"] <= qualified["l2_pct"] * slack + 0.05, (observed["l2_pct"], qualified["l2_pct"])


@pytest.mark.parametrize("variant", ["A", "B", "D", "E_bfp8"])
@pytest.mark.parametrize("q_chunk,k_chunk,head_dim", [(96, 160, 96), (160, 96, 160)], ids=lambda x: str(x))
def test_recipe_geometry_constant_v(device, variant, q_chunk, k_chunk, head_dim):
    """Softmax weights sum to one: constant V returns 1.0 within a few BF16 ulps at any geometry."""
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = inputs_with_head_dim(head_dim, 1100, "constant_v", 700)
    actual = run_or_skip(device, host, variant, (3, 1), q_chunk, k_chunk)
    assert (actual.float() - 1).abs().max().item() <= 1 / 64
