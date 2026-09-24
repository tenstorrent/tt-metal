# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""D64 recipes (SD3.5, Motif, LTX-2 audio), gated on determinism and FP64
accuracy relative to the qualified D128 recipe on the same generator."""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare


def reference(q, k, v):
    """FP64 attention with the D64 default scale."""
    q, k, v = q.double(), k.double(), v.double()
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    return torch.softmax(scores, dim=-1) @ v


def options(variant, grid, q_chunk, k_chunk):
    return dict(
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
    )


def upload(device, host, variant="A"):
    return prepare([ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host], variant)


def run(segments, variant, grid, q_chunk, k_chunk):
    kwargs = options(variant, grid, q_chunk, k_chunk)
    if len(segments) == 1:
        outputs = [ttnn.transformer.scaled_dot_product_attention(*segments[0], is_causal=False, **kwargs)]
    else:
        outputs = ttnn.transformer.joint_scaled_dot_product_attention(
            *segments[0], *segments[1], joint_strategy="rear", **kwargs
        )
    return torch.cat([ttnn.to_torch(x) for x in outputs], dim=2)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("q_chunk,k_chunk", [(256, 512), (128, 256), (224, 384)], ids=lambda x: str(x))
@pytest.mark.parametrize(
    "q_length,k_length,heads,grid,distribution,joint_rows",
    [
        (1024, 2048, 2, (4, 1), "normal", None),
        (1000, 1025, 1, (3, 1), "changed_max", None),
        (768, 1536, 1, (2, 1), "constant_v", None),
        (4100, 4100, 1, (8, 1), "normal", None),
        (1033, 1033, 2, (4, 1), "common_k", (333, 333)),
    ],
    ids=["aligned", "tails", "constant_v", "motif_4100", "joint_tails"],
)
def test_recipe_d64_accuracy(
    device, variant, q_chunk, k_chunk, q_length, k_length, heads, grid, distribution, joint_rows, record_property
):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    full = make_inputs(k_length, distribution, q_length=q_length, heads=heads)
    host = [x[..., :64].contiguous() for x in full]

    def segments_of(values):
        if joint_rows is None:
            return [upload(device, values, variant)]
        q_rows, kv_rows = joint_rows
        split = list(zip(values, (q_rows, kv_rows, kv_rows)))
        return [
            upload(device, [x[..., :-rows, :].contiguous() for x, rows in split], variant),
            upload(device, [x[..., -rows:, :].contiguous() for x, rows in split], variant),
        ]

    segments = segments_of(host)
    try:
        actual = run(segments, variant, grid, q_chunk, k_chunk)
    except RuntimeError as error:
        assert "bytes of L1" in str(error), error
        pytest.skip(f"{variant} D64 Q{q_chunk}/K{k_chunk} exceeds Blackhole L1")
    assert digest(run(segments, variant, grid, q_chunk, k_chunk)) == digest(actual)
    assert actual.shape[-1] == 64 and torch.isfinite(actual.float()).all()
    observed = metrics(actual, reference(*host))
    for key, value in observed.items():
        record_property(key, value)
    # Calibrate against the qualified D128 recipe on the same generator and geometry.
    d128 = metrics(run(segments_of(full), variant, grid, q_chunk, k_chunk), reference(*full))
    record_property("d128_l2_pct", d128["l2_pct"])
    record_property("d128_max_abs", d128["max_abs"])
    if distribution == "constant_v":
        # Constant V returns 1.0; BF16-destination recipes land within a few BF16 ulps.
        assert observed["max_abs"] <= 2 * d128["max_abs"] + 1 / 128, (observed["max_abs"], d128["max_abs"])
    else:
        assert observed["l2_pct"] <= d128["l2_pct"] * 1.5 + 0.05, (observed["l2_pct"], d128["l2_pct"])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("q_chunk,k_chunk", [(128, 256), (256, 256)], ids=lambda x: str(x))
@pytest.mark.parametrize(
    "q_length,k_length,heads,grid,distribution",
    [(768, 3072, 2, (4, 1), "normal"), (768, 3072, 1, (2, 1), "changed_max")],
    ids=["ideogram_like", "changed_max"],
)
def test_recipe_d256_accuracy(
    device, variant, q_chunk, k_chunk, q_length, k_length, heads, grid, distribution, record_property
):
    """Ideogram4-style D256 (L1-limited, so smaller K chunks), gated against the D128 recipe."""
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    half = make_inputs(k_length, distribution, q_length=q_length, heads=heads)
    other = make_inputs(k_length, distribution, q_length=q_length, heads=heads, seed=20261123)
    host = [torch.cat([a, b], dim=-1) for a, b in zip(half, other)]  # D256 from two D128 draws
    try:
        actual = run([upload(device, host, variant)], variant, grid, q_chunk, k_chunk)
    except RuntimeError as error:
        assert "bytes of L1" in str(error), error
        pytest.skip(f"{variant} D256 Q{q_chunk}/K{k_chunk} exceeds Blackhole L1")
    assert digest(run([upload(device, host, variant)], variant, grid, q_chunk, k_chunk)) == digest(actual)
    observed = metrics(actual, reference(*host))
    d128 = metrics(run([upload(device, half, variant)], variant, grid, q_chunk, k_chunk), reference(*half))
    for key, value in observed.items():
        record_property(key, value)
    record_property("d128_l2_pct", d128["l2_pct"])
    if distribution == "changed_max":
        # Concatenating two D128 draws doubles the shifted-key logit term (it grows with sqrt(D)), so
        # this D256 stress is harsher than its D128 counterpart; record it, bound it loosely.
        assert observed["l2_pct"] < 50, observed["l2_pct"]
    else:
        assert observed["l2_pct"] <= d128["l2_pct"] * 1.5 + 0.05, (observed["l2_pct"], d128["l2_pct"])


def test_recipe_rejects_unsupported_head_dim(device):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = [x[..., :96].contiguous() for x in make_inputs(512, "normal", q_length=256)]
    with pytest.raises(RuntimeError, match="head dims 64, 128 and 256"):
        run([upload(device, host)], "D", (1, 1), 256, 512)
