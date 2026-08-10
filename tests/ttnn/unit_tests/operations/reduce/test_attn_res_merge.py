# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# See test_attn_res_stats.py — same reason: swept per-PR on architectures the op
# has never been built for.
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="attn_res_merge has only been validated on Blackhole")

# The op exists to remove six passes over a full-width tensor, not to change the
# arithmetic, so the gate is against torch in fp32 and it is strict. bfloat16 in
# and out means one rounding at the pack.
PCC = 0.9999

# The model's full hidden size, not the per-rank shard: the statistics the op
# folds are already summed across tensor-parallel ranks by the time it scores.
HIDDEN_SIZE = 7168
INV_HIDDEN_SIZE = 1.0 / HIDDEN_SIZE
EPS = 1e-6


def _place(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _scalars(tokens, torch_dtype):
    """Shift, mass and live score in the ranges an online softmax produces.

    `mass` is a sum of exponentials taken against the running maximum, so it is
    at least one; a mass drawn around zero would put the denominator near zero
    and make the gate measure catastrophic cancellation instead of the op.
    """
    shape = [1, 1, tokens, 1]
    shift = (torch.randn(shape) * 2.0).to(torch_dtype)
    mass = (torch.rand(shape) * 7.0 + 1.0).to(torch_dtype)
    live_scores = (torch.randn(shape) * 2.0).to(torch_dtype)
    return shift, mass, live_scores


def _rank_stacked_stats(tokens, num_partials):
    """Per-rank statistics and the live score they normalize to.

    Each rank carries its share of the whole, so the summed sum of squares lands
    at the scale `inv_hidden_size` normalizes against however many ranks there
    are, and the score comes out in the same range `_scalars` draws. A sum of
    squares must stay positive or the reciprocal square root is not defined.
    """
    shape = [1, 1, tokens, 1]
    pairs = [
        (
            (torch.rand(shape) + 0.5) * HIDDEN_SIZE / num_partials,
            torch.randn(shape) * 2.0 / num_partials,
        )
        for _ in range(num_partials)
    ]

    stacked = torch.cat([half for pair in pairs for half in pair], dim=1)
    squares = sum(pair[0] for pair in pairs)
    dots = sum(pair[1] for pair in pairs)
    return stacked, dots * torch.rsqrt(squares * INV_HIDDEN_SIZE + EPS)


def _reference(partial, prefix_sum, shift, mass, live_scores):
    partial, prefix_sum = partial.float(), prefix_sum.float()
    shift, mass, live_scores = shift.float(), mass.float(), live_scores.float()

    merged_shift = torch.maximum(shift, live_scores)
    rescale = torch.exp(shift - merged_shift)
    live_weight = torch.exp(live_scores - merged_shift)
    return (partial * rescale + prefix_sum * live_weight) / (mass * rescale + live_weight)


@pytest.mark.parametrize(
    "shape",
    (
        [1, 1, 256, 1792],
        [1, 1, 640, 1792],
        [1, 1, 128, 3584],
        [1, 1, 64, 32],
        [1, 1, 32, 512],
        [1, 1, 2560, 128],
    ),
    ids=[
        # One read site's fold on a (2,4) mesh — d/4 wide — with the token count
        # cut to keep the test cheap, then at the production 640 per chip.
        "attnres_short",
        "attnres_full",
        # Twice the production width.
        "wide_d",
        # Wt=1: every output tile starts a new token row, so the row scalars turn
        # over on every single tile.
        "wt1_refetch_every_tile",
        # Ht=1: one tile row for the whole tensor, so they never do.
        "ht1_single_row",
        # Many rows against few columns, which is the opposite work split.
        "tall",
    ],
)
def test_attn_res_merge(shape, device):
    torch.manual_seed(2026)
    tokens = shape[2]

    partial = torch.randn(shape, dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    shift, mass, live_scores = _scalars(tokens, torch.bfloat16)

    output = ttnn.experimental.attn_res_merge(
        _place(partial, device),
        _place(prefix_sum, device),
        _place(shift, device),
        _place(mass, device),
        _place(live_scores, device),
    )

    assert list(output.shape) == shape
    assert output.dtype == ttnn.bfloat16
    assert_with_pcc(_reference(partial, prefix_sum, shift, mass, live_scores), ttnn.to_torch(output).float(), PCC)


def test_attn_res_merge_fp32_scalars(device):
    """bfloat16 full-width operands against fp32 row scalars, which is how
    AttnRes calls it when its score chain runs in fp32.

    Requiring bfloat16 scalars would make the caller spend a typecast to throw
    away accuracy it deliberately kept. The full-width operands stay bfloat16 —
    that asymmetry is the point."""
    torch.manual_seed(2026)
    shape = [1, 1, 256, 1792]

    partial = torch.randn(shape, dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    shift, mass, live_scores = _scalars(shape[2], torch.float32)

    output = ttnn.experimental.attn_res_merge(
        _place(partial, device),
        _place(prefix_sum, device),
        _place(shift, device, ttnn.float32),
        _place(mass, device, ttnn.float32),
        _place(live_scores, device, ttnn.float32),
    )

    assert output.dtype == ttnn.bfloat16
    assert_with_pcc(_reference(partial, prefix_sum, shift, mass, live_scores), ttnn.to_torch(output).float(), PCC)


@pytest.mark.parametrize("num_partials", (1, 2, 4, 8), ids=["tp1", "tp2", "tp4", "tp8"])
def test_attn_res_merge_folds_stats(num_partials, device):
    """The live score derived here, from statistics still stacked per rank.

    This is what lets a read drop its scoring program: the gate is the same fold
    against a score computed outside, so the op has to reproduce both the
    cross-rank sum and the normalization it absorbed."""
    torch.manual_seed(2026)
    shape = [1, 1, 256, 1792]

    partial = torch.randn(shape, dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    shift, mass, _ = _scalars(shape[2], torch.float32)
    stats, live_scores = _rank_stacked_stats(shape[2], num_partials)

    output = ttnn.experimental.attn_res_merge(
        _place(partial, device),
        _place(prefix_sum, device),
        _place(shift, device, ttnn.float32),
        _place(mass, device, ttnn.float32),
        _place(stats, device, ttnn.float32),
        num_partials=num_partials,
        inv_hidden_size=INV_HIDDEN_SIZE,
        eps=EPS,
    )

    assert list(output.shape) == shape
    assert_with_pcc(_reference(partial, prefix_sum, shift, mass, live_scores), ttnn.to_torch(output).float(), PCC)


def test_attn_res_merge_folds_stats_at_site(device):
    """The folding path against batched shift and mass.

    The statistics are the live stream's and never batch, so their page walk and
    the scalars' site offset are independent; an offset applied to both would
    pass every unbatched test."""
    torch.manual_seed(2026)
    shape = [1, 1, 128, 256]
    sites, num_partials = 3, 4

    partial = torch.randn([sites, 1, shape[2], shape[3]], dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    shift = (torch.randn([sites, 1, shape[2], 1]) * 2.0).float()
    mass = (torch.rand([sites, 1, shape[2], 1]) * 7.0 + 1.0).float()
    stats, live_scores = _rank_stacked_stats(shape[2], num_partials)

    tt_partial, tt_prefix_sum = _place(partial, device), _place(prefix_sum, device)
    tt_shift, tt_mass, tt_stats = (_place(t, device, ttnn.float32) for t in (shift, mass, stats))

    for site in range(sites):
        output = ttnn.experimental.attn_res_merge(
            tt_partial,
            tt_prefix_sum,
            tt_shift,
            tt_mass,
            tt_stats,
            site=site,
            num_partials=num_partials,
            inv_hidden_size=INV_HIDDEN_SIZE,
            eps=EPS,
        )
        want = _reference(
            partial[site : site + 1], prefix_sum, shift[site : site + 1], mass[site : site + 1], live_scores
        )
        assert list(output.shape) == shape
        assert_with_pcc(want, ttnn.to_torch(output).float(), PCC)


def test_attn_res_merge_folds_stats_matches_scoring_first(device):
    """Against `attn_res_scores` followed by the unfolded merge.

    That is exactly the pair of programs the folding path replaces, so this is
    the gate on it being a drop-in rather than a second implementation."""
    torch.manual_seed(2026)
    shape = [1, 1, 256, 1792]
    num_partials = 4

    tt_partial = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_prefix_sum = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    shift, mass, _ = _scalars(shape[2], torch.float32)
    stats, _ = _rank_stacked_stats(shape[2], num_partials)
    tt_shift, tt_mass, tt_stats = (_place(t, device, ttnn.float32) for t in (shift, mass, stats))

    folded = ttnn.experimental.attn_res_merge(
        tt_partial,
        tt_prefix_sum,
        tt_shift,
        tt_mass,
        tt_stats,
        num_partials=num_partials,
        inv_hidden_size=INV_HIDDEN_SIZE,
        eps=EPS,
    )

    scored = ttnn.experimental.attn_res_scores(
        tt_stats, INV_HIDDEN_SIZE, EPS, num_partials=num_partials, dtype=ttnn.float32
    )
    separate = ttnn.experimental.attn_res_merge(tt_partial, tt_prefix_sum, tt_shift, tt_mass, scored)

    assert_with_pcc(ttnn.to_torch(separate).float(), ttnn.to_torch(folded).float(), PCC)


def test_attn_res_merge_unaligned_rows(device):
    """A token count that is not a multiple of the tile height.

    `from_torch` zero-pads, and a padded row's mass is zero, so its denominator
    is whatever the exponentials give rather than a division by zero — only the
    logical region is gated."""
    torch.manual_seed(2026)
    shape = [1, 1, 100, 128]

    partial = torch.randn(shape, dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    shift, mass, live_scores = _scalars(shape[2], torch.bfloat16)

    output = ttnn.experimental.attn_res_merge(
        _place(partial, device),
        _place(prefix_sum, device),
        _place(shift, device),
        _place(mass, device),
        _place(live_scores, device),
    )

    assert_with_pcc(_reference(partial, prefix_sum, shift, mass, live_scores), ttnn.to_torch(output).float(), PCC)


def test_attn_res_merge_site(device):
    """A batched partial, shift and mass against a per-site live_scores — how
    AttnRes calls it.

    Its partials, shifts and masses come out of one `inter_block` for the whole
    block, while the live score is computed per read, so the two arrive at
    different dim 0. Each site must land on its own plane, and the partial's
    stride is the full `Ht * Wt` block against the scalars' `Ht`; an offset that
    was right only for plane 0 would pass a single-site test."""
    torch.manual_seed(2026)
    shape = [1, 1, 128, 256]
    sites = 3

    partial = torch.randn([sites, 1, shape[2], shape[3]], dtype=torch.bfloat16)
    prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
    shift = (torch.randn([sites, 1, shape[2], 1]) * 2.0).to(torch.bfloat16)
    mass = (torch.rand([sites, 1, shape[2], 1]) * 7.0 + 1.0).to(torch.bfloat16)
    _, _, live_scores = _scalars(shape[2], torch.bfloat16)

    tt_partial, tt_prefix_sum = _place(partial, device), _place(prefix_sum, device)
    tt_shift, tt_mass, tt_live = (_place(t, device) for t in (shift, mass, live_scores))

    for site in range(sites):
        output = ttnn.experimental.attn_res_merge(tt_partial, tt_prefix_sum, tt_shift, tt_mass, tt_live, site=site)
        want = _reference(
            partial[site : site + 1], prefix_sum, shift[site : site + 1], mass[site : site + 1], live_scores
        )
        assert list(output.shape) == shape
        assert_with_pcc(want, ttnn.to_torch(output).float(), PCC)


def test_attn_res_merge_matches_composed(device):
    """Against the eleven-op form it replaces, at the same precision.

    The torch gate above says the op is correct; this one says it is a drop-in
    for the chain, which is what the caller is giving up."""
    torch.manual_seed(2026)
    shape = [1, 1, 256, 1792]

    tt_partial = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_prefix_sum = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    shift, mass, live_scores = _scalars(shape[2], torch.bfloat16)
    tt_shift, tt_mass, tt_live = (_place(t, device) for t in (shift, mass, live_scores))

    fused = ttnn.experimental.attn_res_merge(tt_partial, tt_prefix_sum, tt_shift, tt_mass, tt_live)

    merged_shift = ttnn.maximum(tt_shift, tt_live)
    rescale = ttnn.exp(ttnn.sub(tt_shift, merged_shift))
    live_weight = ttnn.exp(ttnn.sub(tt_live, merged_shift))
    numerator = ttnn.add(ttnn.mul(tt_partial, rescale), ttnn.mul(tt_prefix_sum, live_weight))
    denominator = ttnn.add(ttnn.mul(tt_mass, rescale), live_weight)
    composed = ttnn.div(numerator, denominator)

    assert_with_pcc(ttnn.to_torch(composed).float(), ttnn.to_torch(fused).float(), PCC)


def test_attn_res_merge_program_cache(device):
    """Second call must hit the program cache and still read the right buffers.

    The descriptor declares all five input addresses as buffer bindings; a cache
    hit patches them in place rather than rebuilding. If any binding were wrong
    the second call would silently fold the first call's tensors, which a
    single-shot test cannot see."""
    torch.manual_seed(2026)
    shape = [1, 1, 128, 256]

    entries_before = device.num_program_cache_entries()
    for _ in range(2):
        partial = torch.randn(shape, dtype=torch.bfloat16)
        prefix_sum = torch.randn(shape, dtype=torch.bfloat16)
        shift, mass, live_scores = _scalars(shape[2], torch.bfloat16)
        # Held until after the check so the next iteration cannot reuse the
        # addresses, which would make a stale binding look correct.
        operands = [
            _place(partial, device),
            _place(prefix_sum, device),
            _place(shift, device),
            _place(mass, device),
            _place(live_scores, device),
        ]
        output = ttnn.experimental.attn_res_merge(*operands)
        assert_with_pcc(_reference(partial, prefix_sum, shift, mass, live_scores), ttnn.to_torch(output).float(), PCC)

    assert device.num_program_cache_entries() - entries_before == 1


@pytest.mark.parametrize(
    "bad, message",
    (
        ("partial_dtype", "only supports specific data types"),
        ("mixed_scalar_dtypes", "requires one dtype across shift, mass and live_scores"),
        ("prefix_sum_shape", "requires an unbatched prefix_sum matching partial's plane"),
        ("batched_prefix_sum", "requires an unbatched prefix_sum matching partial's plane"),
        ("candidates", "requires a candidate dim of 1"),
        ("scalar_width", "must carry one scalar per row"),
        ("scalar_rows", "the candidate and row dims must match"),
        ("site", "is past shift's dim 0"),
        ("partial_site", "is past partial's dim 0"),
        ("rank", "requires rank-4 operands"),
        ("stats_planes", "partials requires live_scores shaped"),
        ("stats_inv_hidden", "needs a positive inv_hidden_size"),
    ),
)
def test_attn_res_merge_rejects(bad, message, device, expect_error):
    """The narrow contract is enforced, not documented and hoped for.

    Each case pins its own message, so a rejection for the wrong reason fails
    here rather than counting as coverage."""
    torch.manual_seed(2026)
    shape = [1, 1, 128, 128]
    scalar_shape = [1, 1, 128, 1]

    partial = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    prefix_sum = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    shift = _place(torch.randn(scalar_shape, dtype=torch.bfloat16), device)
    mass = _place(torch.rand(scalar_shape, dtype=torch.bfloat16) + 1.0, device)
    live_scores = _place(torch.randn(scalar_shape, dtype=torch.bfloat16), device)
    site = 0
    num_partials = 0
    inv_hidden_size = INV_HIDDEN_SIZE

    if bad == "partial_dtype":
        # The MAC path is the one numeric configuration gated against a
        # reference, so the full-width operands take bfloat16 only.
        partial = _place(torch.randn(shape, dtype=torch.float32), device, ttnn.float32)
    elif bad == "mixed_scalar_dtypes":
        # All three share one circular buffer and one unpack configuration.
        mass = _place(torch.rand(scalar_shape, dtype=torch.float32) + 1.0, device, ttnn.float32)
    elif bad == "prefix_sum_shape":
        prefix_sum = _place(torch.randn([1, 1, 128, 256], dtype=torch.bfloat16), device)
    elif bad == "batched_prefix_sum":
        # The partial batches over read sites; the live stream is one plane behind
        # all of them, so a batched prefix_sum means the caller has confused the two.
        prefix_sum = _place(torch.randn([2, 1, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "candidates":
        partial = _place(torch.randn([1, 2, 128, 128], dtype=torch.bfloat16), device)
        prefix_sum = _place(torch.randn([1, 2, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "scalar_width":
        shift = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    elif bad == "scalar_rows":
        shift = _place(torch.randn([1, 1, 256, 1], dtype=torch.bfloat16), device)
    elif bad == "site":
        # A scalar carrying two sites, asked for a third.
        shift = _place(torch.randn([2, 1, 128, 1], dtype=torch.bfloat16), device)
        site = 2
    elif bad == "partial_site":
        partial = _place(torch.randn([2, 1, 128, 128], dtype=torch.bfloat16), device)
        site = 2
    elif bad == "rank":
        partial = _place(torch.randn([1, 128, 128], dtype=torch.bfloat16), device)
        prefix_sum = _place(torch.randn([1, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "stats_planes":
        # A score where two ranks' statistics were promised. Nothing about the
        # tensor says which it is, so the plane count is the only check there is.
        num_partials = 2
    elif bad == "stats_inv_hidden":
        num_partials = 1
        live_scores = _place(torch.rand([1, 2, 128, 1], dtype=torch.bfloat16) + 1.0, device)
        inv_hidden_size = 0.0

    with expect_error(RuntimeError, message):
        ttnn.experimental.attn_res_merge(
            partial,
            prefix_sum,
            shift,
            mass,
            live_scores,
            site=site,
            num_partials=num_partials,
            inv_hidden_size=inv_hidden_size,
            eps=EPS,
        )
