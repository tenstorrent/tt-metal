# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# The op exists to remove an intermediate, not to change the arithmetic, so the
# gate is against torch in fp32 and it is strict. bfloat16 in and out means one
# rounding at the pack; anything looser than this would hide a real defect.
PCC = 0.9999


def _place(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _reference(torch_input, torch_weight):
    return (torch_input.float() * torch_weight.float()).sum(dim=1, keepdim=True)


@pytest.mark.parametrize(
    "shape",
    (
        [1, 9, 256, 1792],
        [1, 9, 2560, 1792],
        [1, 8, 128, 128],
        [1, 12, 128, 256],
        [1, 5, 64, 64],
        [1, 13, 64, 64],
        [1, 1, 64, 64],
        [2, 9, 128, 128],
        [1, 9, 128, 32],
        [1, 9, 32, 512],
    ),
    ids=[
        # The shape this op was written for: one AttnRes read on a (2,4) mesh at
        # S=8, with the token count cut to keep the test cheap, then at full size.
        "attnres_short",
        "attnres_full",
        # C=8 takes the granularity cap; C=12 takes 6; C=5 and C=13 are prime to
        # 8 and fall back to a granularity of 1, which is the path where the
        # candidate loop runs one tile at a time.
        "c8_granularity_8",
        "c12_granularity_6",
        "c5_granularity_5",
        "c13_granularity_1",
        # C=1 makes the reduction a plain scale.
        "c1_degenerate",
        # B=2 exercises the batch stride in both the input and the weight index.
        "batched",
        # Wt=1: every output tile starts a new token row, so the weight set turns
        # over on every single tile. Ht=1: one row per batch, so it never does.
        "wt1_refetch_every_tile",
        "ht1_single_row",
    ],
)
def test_fast_weighted_reduce_nc(shape, device):
    torch.manual_seed(2026)
    batch, candidates, height, width = shape

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([batch, candidates, height, 1], dtype=torch.bfloat16)

    output = ttnn.experimental.fast_weighted_reduce_nc(_place(torch_input, device), _place(torch_weight, device), dim=1)

    assert list(output.shape) == [batch, 1, height, width]
    assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)


def test_fast_weighted_reduce_nc_unaligned_rows(device):
    """A token count that is not a multiple of the tile height.

    `from_torch` zero-pads, and zero times zero contributes nothing, so the
    logical region is unaffected — this asserts that rather than assuming it."""
    torch.manual_seed(2026)
    shape = [1, 9, 100, 128]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([1, 9, 100, 1], dtype=torch.bfloat16)

    output = ttnn.experimental.fast_weighted_reduce_nc(_place(torch_input, device), _place(torch_weight, device), dim=1)

    assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)


def test_fast_weighted_reduce_nc_fp32_weight(device):
    """A bfloat16 input against an fp32 weight, which is how AttnRes calls it.

    Its score chain runs in fp32 on purpose, so requiring a bf16 weight would
    make the caller pay a typecast to throw away accuracy it deliberately kept.
    The reference is the same, in fp32, and the gate is the same."""
    torch.manual_seed(2026)
    shape = [1, 9, 256, 1792]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([1, 9, 256, 1], dtype=torch.float32)

    output = ttnn.experimental.fast_weighted_reduce_nc(
        _place(torch_input, device), _place(torch_weight, device, ttnn.float32), dim=1
    )

    assert output.dtype == ttnn.bfloat16
    assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)


def test_fast_weighted_reduce_nc_program_cache(device):
    """Second call must hit the program cache and still read the right buffers.

    The descriptor declares both input addresses as buffer bindings; a cache hit
    patches them in place rather than rebuilding. If that binding were wrong the
    second call would silently reduce the first call's tensors, which a
    single-shot test cannot see."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 256]

    entries_before = device.num_program_cache_entries()
    for _ in range(2):
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        torch_weight = torch.randn([1, 9, 128, 1], dtype=torch.bfloat16)
        # Held until after the check so the next iteration cannot reuse the
        # addresses, which would make a stale binding look correct.
        tt_input, tt_weight = _place(torch_input, device), _place(torch_weight, device)
        output = ttnn.experimental.fast_weighted_reduce_nc(tt_input, tt_weight, dim=1)
        assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)

    assert device.num_program_cache_entries() - entries_before == 1


def test_fast_weighted_reduce_nc_matches_composed(device):
    """Against the two-op form it replaces, at the same precision.

    The torch gate above says the op is correct; this one says it is a drop-in
    for `mul` + `sum`, which is what the caller is giving up."""
    torch.manual_seed(2026)
    shape = [1, 9, 256, 1792]

    tt_input = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_weight = _place(torch.randn([1, 9, 256, 1], dtype=torch.bfloat16), device)

    fused = ttnn.experimental.fast_weighted_reduce_nc(tt_input, tt_weight, dim=1)
    composed = ttnn.sum(ttnn.mul(tt_input, tt_weight), dim=1, keepdim=True)

    assert_with_pcc(ttnn.to_torch(composed).float(), ttnn.to_torch(fused).float(), PCC)


@pytest.mark.parametrize(
    "bad, message",
    (
        ("dim", "supports dim == 1 only"),
        ("weight_width", "weight must carry one scalar per row"),
        ("leading_dims", "the leading three dims must match"),
        ("rank", "requires rank-4 operands"),
        ("input_dtype", "input only supports specific data types"),
    ),
)
def test_fast_weighted_reduce_nc_rejects(bad, message, device, expect_error):
    """The narrow contract is enforced, not documented and hoped for.

    Each case pins its own message, so a rejection for the wrong reason fails
    here rather than counting as coverage."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 128]
    tt_input = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_weight = _place(torch.randn([1, 9, 128, 1], dtype=torch.bfloat16), device)
    dim = 1

    if bad == "dim":
        dim = 0
    elif bad == "weight_width":
        tt_weight = _place(torch.randn([1, 9, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "leading_dims":
        tt_weight = _place(torch.randn([1, 8, 128, 1], dtype=torch.bfloat16), device)
    elif bad == "rank":
        tt_input = _place(torch.randn([9, 128, 128], dtype=torch.bfloat16), device)
        tt_weight = _place(torch.randn([9, 128, 1], dtype=torch.bfloat16), device)
    elif bad == "input_dtype":
        # The weight takes fp32; the input does not, and that asymmetry is
        # deliberate rather than an oversight.
        tt_input = _place(torch.randn(shape, dtype=torch.float32), device, ttnn.float32)

    with expect_error(RuntimeError, message):
        ttnn.experimental.fast_weighted_reduce_nc(tt_input, tt_weight, dim=dim)
