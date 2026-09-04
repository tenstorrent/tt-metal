# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# This directory is swept wholesale per-PR, so landing the file here also enlists
# it on Wormhole and on both simulators — architectures this op has never been
# built for. The L2 nightly entry that scopes it to bh_p150b_civ2 does not
# prevent that sweep; only this does.
pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="attn_res_weighted_reduce_nc has only been validated on Blackhole"
)


# The op exists to remove an intermediate, not to change the arithmetic, so the
# gate is against torch in fp32 and it is strict. bfloat16 in and out means one
# rounding at the pack; anything looser than this would hide a real defect.
PCC = 0.9999


def _place(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _reference(torch_input, torch_weight):
    """`[1, C, H, W]` against `[R, C, H, 1]` -> `[R, 1, H, W]`, in fp32."""
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
        [1, 9, 128, 32],
        [1, 9, 32, 512],
    ),
    ids=[
        # The shape this op was written for: one AttnRes read on a (2,4) mesh at
        # S=8, with the token count cut to keep the test cheap, then at full size.
        "attnres_short",
        "attnres_full",
        # C is not blocked — the candidate loop always runs the whole axis one
        # tile at a time, and C sets both CB depths along with it. These are a
        # spread of counts against that sizing, not distinct branches.
        "c8",
        "c12",
        "c5",
        "c13",
        # C=1 makes the reduction a plain scale.
        "c1_degenerate",
        # Wt=1: every output tile starts a new token row, so the weight set turns
        # over on every single tile. Ht=1: one row, so it never does.
        "wt1_refetch_every_tile",
        "ht1_single_row",
    ],
)
def test_attn_res_weighted_reduce_nc(shape, device):
    torch.manual_seed(2026)
    _, candidates, height, width = shape

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([1, candidates, height, 1], dtype=torch.bfloat16)

    output = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
        _place(torch_input, device), _place(torch_weight, device), dim=1
    )

    assert list(output.shape) == [1, 1, height, width]
    assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)


@pytest.mark.parametrize("num_sites", (1, 3, 5, 8, 24))
def test_attn_res_weighted_reduce_nc_weight_batch(num_sites, device):
    """R weight sets against one input, every plane checked.

    The sets are reduced in groups sized by what DEST holds, so R decides both
    how many accumulators run at once and whether the last group is short. A
    single-plane test would pass with the group loop broken in either direction:
    an accumulator that is not reset between groups, or a tail group that runs
    at full width and writes past the output. R=24 is what AttnRes asks for."""
    torch.manual_seed(2026)
    torch_input = torch.randn([1, 9, 128, 256], dtype=torch.bfloat16)
    torch_weight = torch.randn([num_sites, 9, 128, 1], dtype=torch.bfloat16)

    output = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
        _place(torch_input, device), _place(torch_weight, device), dim=1
    )

    assert list(output.shape) == [num_sites, 1, 128, 256]
    got = ttnn.to_torch(output).float()
    want = _reference(torch_input, torch_weight)
    for site in range(num_sites):
        assert_with_pcc(want[site : site + 1], got[site : site + 1], PCC)


def test_attn_res_weighted_reduce_nc_weight_batch_matches_per_site(device):
    """A batch of R sets against R calls of one set each.

    The batching exists to read the input once for a whole group instead of once
    per set; that is only a valid trade if the two give the same answer, and
    torch alone cannot say so — both could be within PCC of it while differing
    from each other."""
    torch.manual_seed(2026)
    tt_input = _place(torch.randn([1, 9, 128, 256], dtype=torch.bfloat16), device)
    torch_weight = torch.randn([6, 9, 128, 1], dtype=torch.bfloat16)

    batched = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
        tt_input, _place(torch_weight, device), dim=1
    )
    got = ttnn.to_torch(batched).float()

    for site in range(torch_weight.shape[0]):
        alone = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
            tt_input, _place(torch_weight[site : site + 1], device), dim=1
        )
        assert_with_pcc(ttnn.to_torch(alone).float(), got[site : site + 1], PCC)


def test_attn_res_weighted_reduce_nc_negative_dim(device):
    """`dim=-3` is the rank-4 alias for `dim=1` and must reach the same op.

    Normalization happens in the host wrapper, above the validation that accepts
    only dim 1, so a broken alias surfaces as a rejection rather than a wrong
    answer — but only if something calls it."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 256]

    tt_input = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_weight = _place(torch.randn([1, 9, 128, 1], dtype=torch.bfloat16), device)

    aliased = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(tt_input, tt_weight, dim=-3)
    direct = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(tt_input, tt_weight, dim=1)

    assert list(aliased.shape) == list(direct.shape)
    assert torch.equal(ttnn.to_torch(aliased), ttnn.to_torch(direct))


def test_attn_res_weighted_reduce_nc_unaligned_rows(device):
    """A token count that is not a multiple of the tile height.

    `from_torch` zero-pads, and zero times zero contributes nothing, so the
    logical region is unaffected — this asserts that rather than assuming it."""
    torch.manual_seed(2026)
    shape = [1, 9, 100, 128]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([1, 9, 100, 1], dtype=torch.bfloat16)

    output = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
        _place(torch_input, device), _place(torch_weight, device), dim=1
    )

    assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)


def test_attn_res_weighted_reduce_nc_fp32_weight(device):
    """A bfloat16 input against an fp32 weight, which is how AttnRes calls it.

    Its score chain runs in fp32 on purpose, so requiring a bf16 weight would
    make the caller pay a typecast to throw away accuracy it deliberately kept.
    The reference is the same, in fp32, and the gate is the same."""
    torch.manual_seed(2026)
    shape = [1, 9, 256, 1792]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn([1, 9, 256, 1], dtype=torch.float32)

    output = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
        _place(torch_input, device), _place(torch_weight, device, ttnn.float32), dim=1
    )

    assert output.dtype == ttnn.bfloat16
    assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)


def test_attn_res_weighted_reduce_nc_program_cache(device):
    """Second call must hit the program cache and still read the right buffers.

    The descriptor declares both input addresses as buffer bindings; a cache hit
    patches them in place rather than rebuilding. If that binding were wrong the
    second call would silently reduce the first call's tensors, which a
    single-shot test cannot see."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 256]

    entries_before = device.num_program_cache_entries()
    # Every call's tensors outlive the loop, so the second call cannot land on the
    # first call's addresses — a stale binding would look correct if it did.
    live = []
    for _ in range(2):
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        torch_weight = torch.randn([1, 9, 128, 1], dtype=torch.bfloat16)
        tt_input, tt_weight = _place(torch_input, device), _place(torch_weight, device)
        output = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(tt_input, tt_weight, dim=1)
        live += [tt_input, tt_weight, output]
        assert_with_pcc(_reference(torch_input, torch_weight), ttnn.to_torch(output).float(), PCC)

    assert device.num_program_cache_entries() - entries_before == 1


def test_attn_res_weighted_reduce_nc_matches_composed(device):
    """Against the two-op form it replaces, at the same precision.

    The torch gate above says the op is correct; this one says it is a drop-in
    for `mul` + `sum`, which is what the caller is giving up."""
    torch.manual_seed(2026)
    shape = [1, 9, 256, 1792]

    tt_input = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_weight = _place(torch.randn([1, 9, 256, 1], dtype=torch.bfloat16), device)

    fused = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(tt_input, tt_weight, dim=1)
    composed = ttnn.sum(ttnn.mul(tt_input, tt_weight), dim=1, keepdim=True)

    assert_with_pcc(ttnn.to_torch(composed).float(), ttnn.to_torch(fused).float(), PCC)


@pytest.mark.parametrize(
    "bad, message",
    (
        ("dim", "supports dim == 1 only"),
        ("weight_width", "weight must carry one scalar per row"),
        ("leading_dims", "the candidate and row dims must match"),
        ("rows_same_tile_bucket", "the candidate and row dims must match"),
        ("input_batch", "takes an unbatched input"),
        ("rank", "requires rank-4 operands"),
        ("input_dtype", "input only supports specific data types"),
        ("weight_dtype", "weight only supports specific data types"),
        ("host_weight", "Device Operations expect device tensors as inputs"),
        ("row_major", "input only supports TILE layout"),
        ("sharded", "supports interleaved operands only"),
    ),
)
def test_attn_res_weighted_reduce_nc_rejects(bad, message, device, expect_error):
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
    elif bad == "rows_same_tile_bucket":
        # 100 and 120 rows both pad to 128, so a padded-only check lets this
        # through and the shorter operand's padding becomes output data.
        tt_input = _place(torch.randn([1, 9, 120, 128], dtype=torch.bfloat16), device)
        tt_weight = _place(torch.randn([1, 9, 100, 1], dtype=torch.bfloat16), device)
    elif bad == "input_batch":
        # Dim 0 belongs to the weight. A batched input would need its own read per
        # plane, which is the reuse the op is built on.
        tt_input = _place(torch.randn([2, 9, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "rank":
        tt_input = _place(torch.randn([9, 128, 128], dtype=torch.bfloat16), device)
        tt_weight = _place(torch.randn([9, 128, 1], dtype=torch.bfloat16), device)
    elif bad == "input_dtype":
        # The weight takes fp32; the input does not, and that asymmetry is
        # deliberate rather than an oversight.
        tt_input = _place(torch.randn(shape, dtype=torch.float32), device, ttnn.float32)
    elif bad == "weight_dtype":
        # bfloat16 and float32 are the two the CBs are sized for; a block float
        # weight would be read as neither.
        tt_weight = _place(torch.randn([1, 9, 128, 1], dtype=torch.float32), device, ttnn.bfloat8_b)
    elif bad == "host_weight":
        # Caught by the device-operation framework ahead of this op's own
        # validation, so the message pinned here is that one, not ours.
        tt_weight = ttnn.from_torch(torch.randn([1, 9, 128, 1], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    elif bad == "row_major":
        tt_input = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
    elif bad == "sharded":
        # The reader addresses both operands as interleaved pages; a sharded one
        # would be read at the wrong addresses rather than rejected.
        tt_input = ttnn.to_memory_config(
            tt_input,
            ttnn.create_sharded_memory_config(
                shape, core_grid=ttnn.CoreGrid(y=3, x=3), strategy=ttnn.ShardStrategy.HEIGHT
            ),
        )

    with expect_error(RuntimeError, message):
        ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(tt_input, tt_weight, dim=dim)
