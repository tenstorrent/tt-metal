# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-operator PCC validation for the router, and negative controls for the expert FFN.

The router's softmax feeds a top-2 selection, so an error large enough to reorder two
near-tied experts is not a small error: it sends the token to a different pair of
4.7M-parameter experts. Hence the explicit kernel config on the softmax and fp32 held through
the top-k selection.

The router's matmul is tested through TtNomicRouter, which runs it under the program config of
tt/matmul_config.py, and the other operators on their own. The experts run upstream's ragged loop
as this port's own ops, which have their own tests under
tests/ttnn/nightly/unit_tests/operations/experimental/nomic_embed_text_v2_moe/; the pass itself is
tested in test_ttnn_experts.py.

Measured results are tabulated in docs/OPERATOR_MAPPING.md.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from models.experimental.nomic_embed_text_v2_moe.tt.router import PADDING_LOGIT, SCORED_COLUMNS, TtNomicRouter
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

ROUTER_PREFIX = "encoder.layers.1.mlp.router."
EXPERTS_PREFIX = "encoder.layers.1.mlp.experts."

OPERATOR_PCC = 0.999

# (batch, seqlen). 37 is deliberately off-tile: padding bugs only surface when S is not a
# multiple of 32, and S is the batch's longest tokenized sequence, so that is the common case.
TOKEN_SHAPES = [(1, 128), (2, 512), (2, 37)]

# The router softmax's budget. On identical fp32 logits the port's config measures 1.4e-3 to
# 1.9e-3 across seeds, HiFi4 without fp32 accumulation 5.4e-3 to 6.8e-3, the stock config
# 2.7e-2 to 3.0e-2. HiFi4 alone clears the budget by as little as 7%, so the control below is
# seeded rather than left to chance. Gating on max-abs is deliberate: all three exceed 0.9996 PCC.
SOFTMAX_MAX_ABS = 5e-3


def router_probabilities(tokens: int, experts: int) -> torch.Tensor:
    """A plausible (1, 1, T, E) softmax distribution over experts."""
    return torch.randn(1, 1, tokens, experts).softmax(dim=-1)


# Router.


@pytest.mark.needs_weights
@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_router_linear(device, tt_config, config, state_dict, batch, seqlen):
    """aten.mm -> ttnn.linear at the real router weights, as TtNomicRouter.logits runs it.

    768 -> 8 is the narrowest matmul in the model, the only one without a checkpoint bias, and the
    only one with an fp32 weight and output. It scores SCORED_COLUMNS columns, the experts first and
    PADDING_LOGIT in the rest. Its input is the bfloat16 activation, so the reference takes the same
    rounded values.
    """
    router = TtNomicRouter(device, config, tt_config, state_dict, ROUTER_PREFIX)
    x = torch.randn(1, 1, batch * seqlen, config.hidden_size).bfloat16().float()

    out = router.logits(to_device(x, device))

    assert out.dtype == tt_config.router_dtype
    assert out.shape[-1] == SCORED_COLUMNS
    got = ttnn.to_torch(out).float()
    assert_with_pcc(
        torch.nn.functional.linear(x, state_dict[ROUTER_PREFIX + "layer.weight"]),
        got[..., : config.num_experts],
        OPERATOR_PCC,
    )
    assert torch.equal(got[..., config.num_experts :], torch.full_like(got[..., config.num_experts :], PADDING_LOGIT))


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_softmax(device, tt_config, config, batch, seqlen):
    """aten._softmax -> ttnn.softmax with the HiFi4 kernel config, gated on max-abs."""
    logits = torch.randn(1, 1, batch * seqlen, config.num_experts)

    out = ttnn.softmax(
        to_device(logits, device, dtype=tt_config.router_dtype),
        dim=-1,
        compute_kernel_config=tt_config.compute_kernel_config(OpGroup.SOFTMAX),
    )

    ref = logits.softmax(dim=-1)
    assert_with_pcc(ref, out, OPERATOR_PCC)
    assert compute_max_abs_error(ttnn.to_torch(out).float(), ref) < SOFTMAX_MAX_ABS


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_softmax_zeroes_the_padding_scores(device, tt_config, config, batch, seqlen):
    """The router's softmax over SCORED_COLUMNS gives the experts' softmax bit for bit, then zeros.

    TtNomicRouter scores 64 columns so that ttnn.topk has nothing to pad; that is only free if the
    PADDING_LOGIT columns add nothing to the denominator and leave the experts' probabilities as
    they were.
    """
    tokens = batch * seqlen
    logits = torch.randn(1, 1, tokens, config.num_experts)
    scored = torch.full((1, 1, tokens, SCORED_COLUMNS), PADDING_LOGIT)
    scored[..., : config.num_experts] = logits
    kernel = tt_config.compute_kernel_config(OpGroup.SOFTMAX)

    narrow = ttnn.to_torch(
        ttnn.softmax(to_device(logits, device, dtype=tt_config.router_dtype), dim=-1, compute_kernel_config=kernel)
    )
    wide = ttnn.to_torch(
        ttnn.softmax(to_device(scored, device, dtype=tt_config.router_dtype), dim=-1, compute_kernel_config=kernel)
    )

    assert torch.equal(wide[..., : config.num_experts], narrow)
    assert torch.equal(wide[..., config.num_experts :], torch.zeros_like(wide[..., config.num_experts :]))


def test_softmax_needs_both_hifi4_and_fp32_accumulation(device, tt_config, config):
    """Negative control: neither half of the kernel config is optional here.

    HiFi4 without the fp32 accumulator misses the budget, and so does the fp32 accumulator at
    HiFi2. All three variants score above 0.9996 PCC, so only max-abs separates them.
    """
    logits = torch.randn(1, 1, 1024, config.num_experts)
    ref = logits.softmax(dim=-1)
    logits_tt = to_device(logits, device, dtype=tt_config.router_dtype)

    def kernel_config(math_fidelity, fp32_dest_acc_en):
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
        )

    def max_abs(compute_kernel_config):
        out = ttnn.softmax(logits_tt, dim=-1, compute_kernel_config=compute_kernel_config)
        return compute_max_abs_error(ttnn.to_torch(out).float(), ref)

    ported = max_abs(tt_config.compute_kernel_config(OpGroup.SOFTMAX))
    without_fp32_acc = max_abs(kernel_config(ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))
    without_hifi4 = max_abs(kernel_config(ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True))

    assert ported < SOFTMAX_MAX_ABS, f"the port's own config missed its budget at {ported:.3e}"
    assert without_fp32_acc > SOFTMAX_MAX_ABS, f"HiFi4 alone should miss it, got {without_fp32_acc:.3e}"
    assert without_hifi4 > SOFTMAX_MAX_ABS, f"fp32 accumulation at HiFi2 should miss it, got {without_hifi4:.3e}"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_topk(device, config, dtype, batch, seqlen):
    """aten.topk -> ttnn.topk over the 8-wide expert axis.

    Indices, not values, are what matter: a flipped index reroutes the token to different
    experts entirely. Fed the same numbers as torch, the operator picks the same experts with
    bit-identical weights, so it introduces no routing error of its own.

    The contract asserted is that the chosen experts carry the top-k probabilities, not that
    the index tensors are equal. In bfloat16 an 8-wide probability row lands on a coarse enough
    grid that two experts sometimes hold exactly the same value, and torch and ttnn order a tie
    differently. That is a tie, not a disagreement; fp32 has no ties at these shapes and is
    checked index for index.
    """
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    probabilities = router_probabilities(batch * seqlen, config.num_experts).to(torch_dtype).float()

    values, indices = ttnn.topk(to_device(probabilities, device, dtype=dtype), k=config.moe_top_k, dim=-1)

    ref_values, ref_indices = torch.topk(probabilities, config.moe_top_k, dim=-1)
    selected = ttnn.to_torch(indices).long()
    assert int(selected.max()) < config.num_experts
    assert torch.equal(ttnn.to_torch(values).float(), ref_values)
    assert torch.equal(torch.gather(probabilities, -1, selected), ref_values)
    if dtype == ttnn.float32:
        assert torch.equal(selected, ref_indices)


def test_rounding_probabilities_to_bfloat16_before_topk_flips_routing(device, config):
    """Why the router stays fp32 through the topk, not just up to it.

    ttnn.scatter rejects fp32, so the chain has to reach bfloat16 somewhere. Casting the
    probabilities first, then selecting, reroutes 0.34% to 0.59% of tokens: bfloat16 quantizes an
    8-wide probability row coarsely enough that near-ties collapse into exact ties and the
    boundary lands on a different expert. Selecting in fp32 and casting only the two chosen
    weights reroutes none. ttnn.topk accepts fp32 and its uint32 index feeds ttnn.scatter
    directly, so the later cast costs nothing.
    """
    tokens = 4096
    probabilities = router_probabilities(tokens, config.num_experts)
    reference = torch.topk(probabilities, config.moe_top_k, dim=-1).indices

    def selection(dtype):
        _, indices = ttnn.topk(to_device(probabilities, device, dtype=dtype), k=config.moe_top_k, dim=-1)
        return ttnn.to_torch(indices).long()

    assert torch.equal(selection(ttnn.float32), reference)
    flipped = int((~(selection(ttnn.bfloat16) == reference).all(dim=-1)).sum())
    assert flipped > 0, "bfloat16 no longer costs routing decisions; re-check whether fp32 topk still pays"


def test_topk_on_bfloat8_b_is_silently_wrong(device, config):
    """Negative control: bfloat8_b is accepted and returns a wrong selection.

    A tile-wide shared exponent flattens an 8-wide probability row, so the block format that
    suits weights does not suit this. It does not raise, which is the whole problem.
    """
    probabilities = router_probabilities(512, config.num_experts)

    _, indices = ttnn.topk(to_device(probabilities, device, dtype=ttnn.bfloat8_b), k=config.moe_top_k, dim=-1)

    ref_indices = torch.topk(probabilities, config.moe_top_k, dim=-1).indices
    disagreeing = int((~(ttnn.to_torch(indices).long() == ref_indices).all(dim=-1)).sum())
    assert disagreeing > 0, "bfloat8_b topk agreed with torch; re-check whether this is still a trap"


def test_bfloat8_b_cast_reads_the_tile_padding(device, config):
    """Negative control: a bfloat8_b cast mixes a tile's padding into its shared exponents.

    The transposed dense expert pass the routed one replaced cast its permuted (1, E, 1, t) gate to
    bfloat8_b. At t=300 the last tile holds 12 real columns, and 16 columns share one exponent, so
    padding left stale by the permute crushed the real gates beside it: module PCC 0.984. Zeroing
    the padding first fixes it, and the fill writes in place.
    """
    tokens, edge = 300, slice(288, 300)
    gate_host = torch.rand(1, config.num_experts, 1, tokens)
    gate = ttnn.from_torch(gate_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    stale = ttnn.fill_implicit_tile_padding(gate, 1e4)
    assert stale.buffer_address() == gate.buffer_address(), "the fill no longer writes in place"
    crushed = ttnn.to_torch(ttnn.typecast(stale, ttnn.bfloat8_b)).float()
    assert (
        compute_max_abs_error(gate_host[..., edge], crushed[..., edge]) > 0.1
    ), "the padding no longer reaches the real gates"

    zeroed = ttnn.to_torch(ttnn.typecast(ttnn.fill_implicit_tile_padding(gate, 0.0), ttnn.bfloat8_b)).float()
    assert compute_max_abs_error(gate_host, zeroed) < 1e-2


# Experts.


@pytest.mark.needs_weights
def test_transposed_expert_weights_are_a_shape_error(device, config, state_dict, expect_error):
    """Negative control: on the packed operand, the w2 misorientation is loud.

    Viewing w2 as (E, H, F) instead of (E, F, H) succeeds in torch, because E*F*H is symmetric
    in F and H, and the reference computes uncorrelated noise from it without raising. Packed
    to a 4D operand, the misoriented view fails the matmul's inner-dimension check instead. The
    port stores w2 transposed per expert, the mistake's own shape, so its guard is
    test_ttnn_experts.py::test_misoriented_w2_decorrelates.
    """
    w2 = state_dict[EXPERTS_PREFIX + "mlp.w2"]
    transposed = w2.view(config.num_experts, config.hidden_size, config.intermediate_size).unsqueeze(0)
    activated = to_device(torch.randn(1, config.num_experts, 128, config.intermediate_size), device)

    with expect_error(RuntimeError, "width of the first tensor must be equal to the height"):
        ttnn.matmul(activated, to_device(transposed.contiguous(), device))


@pytest.mark.needs_weights
def test_shared_bias_must_be_added_after_the_weighted_sum(device, config, state_dict):
    """aten.add -> ttnn.add, once, outside the reduce. PCC cannot police this.

    The eight experts share one (H,) bias. Adding it inside the per-expert loop scales it by
    the routed-weight sum, leaving an offset of (sum(w) - 1) * bias. Since the top-2 weights
    are deliberately not renormalized, that sum is below 1, so the offset is real. It is also
    nearly constant across tokens, and PCC mean-centres, so the wrong result still correlates
    above 0.9999. Only max-abs sees it.
    """
    tokens, experts, hidden = 512, config.num_experts, config.hidden_size
    bias = state_dict[EXPERTS_PREFIX + "bias"]
    weighted_sum = torch.randn(1, 1, tokens, hidden)
    probabilities = router_probabilities(tokens, experts)
    routed_weight_sum = torch.topk(probabilities, config.moe_top_k, dim=-1).values.sum(-1, keepdim=True)
    assert (routed_weight_sum < 1.0).all(), "top-k weights are not renormalized, so they must sum below 1"

    sum_tt = to_device(weighted_sum, device)
    bias_tt = to_device(bias.reshape(1, 1, 1, hidden), device)
    correct = ttnn.to_torch(ttnn.add(sum_tt, bias_tt)).float()
    inside_the_loop = ttnn.to_torch(
        ttnn.add(sum_tt, ttnn.multiply(bias_tt, to_device(routed_weight_sum, device)))
    ).float()

    assert_with_pcc(weighted_sum + bias, correct, OPERATOR_PCC)
    assert compute_pcc(inside_the_loop, correct) > 0.9999, "if PCC caught this, the comment above is stale"
    assert compute_max_abs_error(inside_the_loop, correct) > 1e-3
