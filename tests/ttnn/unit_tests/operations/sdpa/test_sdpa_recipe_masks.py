# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Additive attn_mask on the dense named SDPA recipes (A-E).

Contract (same as legacy SDPA): softmax(Q Kᵀ / sqrt(D) + mask) V, the mask [1|B, 1|H, Sq, Sk] added to
the scores before the row max/exp. Gates:
- key padding (0 / -inf or large negative) == the same recipe on K/V truncated to the valid keys
  (near-exact: the masked build only L1-adds 0 or -inf onto the packed scores);
- dense random additive masks, broadcast shapes, segment masks and sub-tile tails vs an FP64 masked
  reference, bounded by the same recipe's error on the unmasked problem;
- determinism (repeat runs are bit-identical);
- fully masked rows behave like legacy SDPA (reported).
"""

import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, metrics, prepare

# Error bounds shared with test_sdpa_recipe_tails.py (L2 %, vs FP64).
LIMITS = {"A": 8, "B": 8, "C": 2, "D": 0.4, "E_bf16": 8, "E_bfp8": 8, "E_bfp4": 35}


@pytest.fixture(autouse=True)
def _blackhole_only():
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")


def options(variant, grid=(4, 1), q_chunk=256, k_chunk=512):
    return dict(
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
    )


def qkv(batch, heads, q_len, k_len, head_dim, seed=20260924):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn((batch, heads, length, head_dim), generator=generator).bfloat16()
        for length in (q_len, k_len, k_len)
    ]


def masked_reference(q, k, v, mask=None):
    """FP64 softmax(QKᵀ/sqrt(D) + mask)V; fully masked rows come out NaN (0/0)."""
    scores = q.double() @ k.double().transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores + mask.double()
    return torch.softmax(scores, dim=-1) @ v.double()


def upload(device, tensors, variant):
    return prepare([ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in tensors], variant)


def upload_mask(device, mask, dtype=ttnn.bfloat16):
    return ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)


def sdpa(inputs, variant, mask=None, **kwargs):
    return ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            *inputs, attn_mask=mask, is_causal=False, **options(variant, **kwargs)
        )
    )


def key_padding(batch, q_len, k_len, valid, fill):
    """[B, 1, Sq, Sk] (the LTX / Ideogram layout): column j >= valid[b] gets `fill`."""
    mask = torch.zeros(batch, 1, q_len, k_len)
    for b, n in enumerate(valid):
        mask[b, :, :, n:] = fill
    return mask


# ---------------------------------------------------------------------------------------------
# Key padding == truncated K (per batch, the recipe on K/V[:valid])
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "q_len,k_len,valid,fill,mask_dtype",
    [
        (256, 1024, (700,), float("-inf"), ttnn.bfloat16),  # aligned K, partial last chunk
        (256, 1024, (300,), float("-inf"), ttnn.bfloat16),  # whole second K chunk masked
        (300, 600, (555,), float("-inf"), ttnn.bfloat16),  # Q/K sub-tile tails
        (256, 256, (200,), float("-inf"), ttnn.bfloat16),  # LTX audio self-attn (N256, 200 real)
        (256, 1024, (700,), -10000.0, ttnn.bfloat16),  # large finite negative
        (256, 1024, (640,), float("-inf"), ttnn.bfloat8_b),  # block-float mask
        (160, 800, (800, 333), float("-inf"), ttnn.bfloat16),  # per-batch masks, B=2
    ],
    ids=["k1024v700", "k1024v300", "tails300x600v555", "ltx256v200", "neg1e4", "bfp8", "batch2"],
)
def test_key_padding_matches_truncated_k(device, variant, q_len, k_len, valid, fill, mask_dtype, record_property):
    batch, heads = len(valid), 2
    q, k, v = qkv(batch, heads, q_len, k_len, 128)
    mask = key_padding(batch, q_len, k_len, valid, fill)
    masked = sdpa(upload(device, (q, k, v), variant), variant, upload_mask(device, mask, mask_dtype))
    worst = 0.0
    exact = True
    for b, n in enumerate(valid):
        truncated = sdpa(
            upload(device, (q[b : b + 1], k[b : b + 1, :, :n], v[b : b + 1, :, :n]), variant), variant
        )
        actual = masked[b : b + 1]
        exact &= digest(actual) == digest(truncated)
        worst = max(worst, metrics(actual, truncated)["l2_pct"])
        reference = masked_reference(q[b : b + 1], k[b : b + 1, :, :n], v[b : b + 1, :, :n])
        observed = metrics(actual, reference)
        record_property(f"b{b}_l2_vs_fp64", observed["l2_pct"])
        assert observed["l2_pct"] < LIMITS[variant]
    record_property("bit_identical_to_truncated", exact)
    record_property("l2_pct_vs_truncated", worst)
    # Near-exact: only the online-softmax chunking of the masked tail can differ.
    assert worst <= 0.05, f"masked vs truncated-K L2 {worst:.4f}%"


# ---------------------------------------------------------------------------------------------
# Dense random additive masks and broadcast shapes vs FP64
# ---------------------------------------------------------------------------------------------
def random_mask(shape, seed, neg_inf_fraction=0.1):
    generator = torch.Generator().manual_seed(seed)
    mask = 2.0 * torch.randn(shape, generator=generator)
    blocked = torch.rand(shape, generator=generator) < neg_inf_fraction
    mask = mask.masked_fill(blocked, float("-inf"))
    mask[..., 0] = 0.0  # every row keeps at least one key
    return mask.bfloat16().float()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "batch,heads,q_len,k_len,mask_batch,mask_heads,mask_dtype",
    [
        (1, 2, 512, 1024, 1, 1, ttnn.bfloat16),
        (2, 2, 256, 512, 1, 1, ttnn.bfloat16),  # broadcast over batch and heads
        (2, 2, 256, 512, 2, 1, ttnn.bfloat16),  # per batch, broadcast over heads
        (2, 2, 256, 512, 1, 2, ttnn.bfloat16),  # per head, broadcast over batch
        (2, 2, 256, 512, 2, 2, ttnn.bfloat16),  # full
        (1, 3, 333, 777, 1, 3, ttnn.bfloat16),  # sub-tile tails, per head
        (1, 2, 512, 1024, 1, 1, ttnn.bfloat8_b),
    ],
    ids=["b1h2", "bcast_bh", "per_b", "per_h", "full", "tails_per_h", "bfp8"],
)
def test_dense_random_mask(
    device, variant, batch, heads, q_len, k_len, mask_batch, mask_heads, mask_dtype, record_property
):
    q, k, v = qkv(batch, heads, q_len, k_len, 128, seed=7)
    mask = random_mask((mask_batch, mask_heads, q_len, k_len), seed=11)
    tt_mask = upload_mask(device, mask, mask_dtype)
    mask = ttnn.to_torch(tt_mask).float()  # the values the device actually sees (bfp8 rounding)
    inputs = upload(device, (q, k, v), variant)
    runs = [sdpa(inputs, variant, tt_mask) for _ in range(3)]
    assert len({digest(x) for x in runs}) == 1, "masked recipe is nondeterministic"
    observed = metrics(runs[0], masked_reference(q, k, v, mask))
    unmasked = metrics(sdpa(inputs, variant), masked_reference(q, k, v))
    for key, value in observed.items():
        record_property(f"masked_{key}", value)
    record_property("unmasked_l2_pct", unmasked["l2_pct"])
    assert observed["l2_pct"] < LIMITS[variant]
    # Same recipe arithmetic: the masked error stays on the unmasked problem's scale.
    assert observed["l2_pct"] <= 1.5 * unmasked["l2_pct"] + 0.05


# ---------------------------------------------------------------------------------------------
# DiT caller shapes: Ideogram4 segment mask (D256, Q128/K256) and LTX audio (D64)
# ---------------------------------------------------------------------------------------------
def segment_mask(length, splits):
    ids = torch.zeros(length, dtype=torch.long)
    for split in splits:
        ids[split:] += 1
    same = ids[:, None] == ids[None, :]
    return torch.zeros(1, 1, length, length).masked_fill(~same, float("-inf"))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "heads,length,head_dim,splits,q_chunk,k_chunk",
    [
        (4, 512, 256, (170,), 128, 256),  # Ideogram4 dense: two packed samples
        (4, 1088, 256, (300, 777), 128, 256),  # three segments, several K chunks fully masked per row
        (2, 256, 64, (200,), 256, 256),  # LTX audio D64 (segment == padded tail attends to itself)
    ],
    ids=["ideogram_n512", "ideogram_n1088", "ltx_d64"],
)
def test_dit_segment_masks(device, variant, heads, length, head_dim, splits, q_chunk, k_chunk, record_property):
    q, k, v = qkv(1, heads, length, length, head_dim, seed=3)
    mask = segment_mask(length, splits)
    inputs = upload(device, (q, k, v), variant)
    tt_mask = upload_mask(device, mask)
    actual = sdpa(inputs, variant, tt_mask, grid=(8, 1), q_chunk=q_chunk, k_chunk=k_chunk)
    assert digest(actual) == digest(sdpa(inputs, variant, tt_mask, grid=(8, 1), q_chunk=q_chunk, k_chunk=k_chunk))
    observed = metrics(actual, masked_reference(q, k, v, mask))
    for key, value in observed.items():
        record_property(key, value)
    assert observed["l2_pct"] < LIMITS[variant]
    # Each segment equals the unmasked recipe on that segment alone (block-diagonal attention).
    bounds = (0, *splits, length)
    worst = 0.0
    for start, end in zip(bounds[:-1], bounds[1:]):
        segment = [x[:, :, start:end].contiguous() for x in (q, k, v)]
        alone = sdpa(upload(device, segment, variant), variant, grid=(8, 1), q_chunk=q_chunk, k_chunk=k_chunk)
        worst = max(worst, metrics(actual[:, :, start:end], alone)["l2_pct"])
    record_property("l2_pct_vs_segment_alone", worst)
    assert worst < LIMITS[variant]


# ---------------------------------------------------------------------------------------------
# Fully masked rows: report and compare with legacy SDPA
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
def test_fully_masked_rows(device, variant, record_property):
    q, k, v = qkv(1, 1, 256, 512, 128, seed=5)
    mask = torch.zeros(1, 1, 256, 512)
    mask[:, :, 40:72, :] = float("-inf")  # rows 40..71 see no key
    tt_mask = upload_mask(device, mask)
    actual = sdpa(upload(device, (q, k, v), variant), variant, tt_mask)
    legacy = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            *[ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)],
            attn_mask=tt_mask,
            is_causal=False,
        )
    )
    rows = torch.zeros(256, dtype=torch.bool)
    rows[40:72] = True
    record_property("recipe_masked_rows_nan", bool(torch.isnan(actual[:, :, rows]).all()))
    record_property("recipe_masked_rows_finite", bool(torch.isfinite(actual[:, :, rows]).all()))
    record_property("legacy_masked_rows_nan", bool(torch.isnan(legacy[:, :, rows]).all()))
    record_property("legacy_masked_rows_finite", bool(torch.isfinite(legacy[:, :, rows]).all()))
    # Other rows are unaffected by the fully masked ones.
    observed = metrics(actual[:, :, ~rows], masked_reference(q, k, v, mask)[:, :, ~rows])
    assert observed["l2_pct"] < LIMITS[variant]


# ---------------------------------------------------------------------------------------------
# Timing (reported, not gated): mask overhead at a DiT-like geometry
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["A", "B", "C", "D", "E_bfp8"])
def test_mask_timing(device, variant, record_property):
    device.enable_program_cache()
    q, k, v = qkv(1, 10, 4096, 4096, 128, seed=1)
    inputs = upload(device, (q, k, v), variant)
    tt_mask = upload_mask(device, key_padding(1, 4096, 4096, (4000,), float("-inf")))
    size = device.compute_with_storage_grid_size()
    grid = (size.x, size.y)

    def run(mask):
        return ttnn.transformer.scaled_dot_product_attention(
            *inputs, attn_mask=mask, is_causal=False, **options(variant, grid=grid)
        )

    timings = {}
    for label, mask in (("unmasked", None), ("masked", tt_mask)):
        run(mask)
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(5):
            run(mask)
        ttnn.synchronize_device(device)
        timings[label] = (time.perf_counter() - start) / 5 * 1e3
        record_property(f"{label}_ms", timings[label])
    record_property("mask_overhead_pct", 100 * (timings["masked"] / timings["unmasked"] - 1))
