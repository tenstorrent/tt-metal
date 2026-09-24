# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""D64 recipes (SD3.5, Motif, LTX-2 audio): FAST, BALANCED and ACCURATE.

The compensated BF16 state (COMPENSATED, LOW_PRECISION) is laid out for D128
and rejects D64 on the host. Gated on determinism and FP64 accuracy.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, digest, make_inputs, metrics

D64_VARIANTS = ("A", "C", "D")


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


def upload(device, host):
    return [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]


def run(segments, variant, grid, q_chunk, k_chunk):
    kwargs = options(variant, grid, q_chunk, k_chunk)
    if len(segments) == 1:
        outputs = [ttnn.transformer.scaled_dot_product_attention(*segments[0], is_causal=False, **kwargs)]
    else:
        outputs = ttnn.transformer.joint_scaled_dot_product_attention(
            *segments[0], *segments[1], joint_strategy="rear", **kwargs
        )
    return torch.cat([ttnn.to_torch(x) for x in outputs], dim=2)


@pytest.mark.parametrize("variant", D64_VARIANTS)
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
            return [upload(device, values)]
        q_rows, kv_rows = joint_rows
        split = list(zip(values, (q_rows, kv_rows, kv_rows)))
        return [
            upload(device, [x[..., :-rows, :].contiguous() for x, rows in split]),
            upload(device, [x[..., -rows:, :].contiguous() for x, rows in split]),
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
    if distribution == "constant_v":
        assert observed["max_abs"] <= 1 / 64, observed["max_abs"]
    else:
        # Calibrate against the qualified D128 recipe on the same generator and geometry.
        d128 = metrics(run(segments_of(full), variant, grid, q_chunk, k_chunk), reference(*full))
        record_property("d128_l2_pct", d128["l2_pct"])
        assert observed["l2_pct"] <= d128["l2_pct"] * 1.5 + 0.05, (observed["l2_pct"], d128["l2_pct"])


@pytest.mark.parametrize("variant", ["B", "E_bf16"])
def test_recipe_d64_rejects_compensated(device, variant):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = [x[..., :64].contiguous() for x in make_inputs(512, "normal", q_length=256)]
    inputs = upload(device, host)
    if variant.startswith("E_"):
        # Preparation itself is D128-only; the attention call must reject D64 either way.
        with pytest.raises(RuntimeError):
            ttnn.transformer.prepare_sdpa_input(inputs[0], is_query=True)
        return
    with pytest.raises(RuntimeError, match="require head dim 128"):
        run([inputs], variant, (1, 1), 256, 512)
