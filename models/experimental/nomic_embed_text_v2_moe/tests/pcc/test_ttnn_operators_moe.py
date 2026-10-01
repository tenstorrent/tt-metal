# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-operator PCC validation for the router and the expert FFN.

The router's softmax feeds a top-2 selection, so an error large enough to reorder two
near-tied experts is not a small error: it sends the token to a different pair of
4.7M-parameter experts. Hence the explicit kernel config on the softmax and fp32 logits held
until the last moment ttnn.scatter allows.

The expert chain is the dense-all-experts formulation from ARCHITECTURE.md section 4, which
replaces upstream's data-dependent ragged loop with two matmuls over every expert, a GELU, a
multiply and a reduce. The matmuls are tested through TtNomicExperts and TtNomicRouter, which
run them under the program configs of tt/matmul_config.py, and the other operators on their own.

Measured results are tabulated in docs/OPERATOR_MAPPING.md.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.tt.common import pack_expert_weights, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import TOKEN_MAJOR_MAX_TOKENS, TtNomicExperts
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from models.experimental.nomic_embed_text_v2_moe.tt.router import TtNomicRouter
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

    768 -> 8 is the narrowest matmul in the model, the only bias-free one, and the only one with
    an fp32 weight and output. Its input is the bfloat16 activation, so the reference takes the
    same rounded values.
    """
    router = TtNomicRouter(device, config, tt_config, state_dict, ROUTER_PREFIX)
    x = torch.randn(1, 1, batch * seqlen, config.hidden_size).bfloat16().float()

    out = router.logits(to_device(x, device))

    assert out.dtype == tt_config.router_dtype
    assert_with_pcc(torch.nn.functional.linear(x, state_dict[ROUTER_PREFIX + "layer.weight"]), out, OPERATOR_PCC)


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


def test_softmax_needs_both_hifi4_and_fp32_accumulation(device, tt_config, config):
    """Negative control: neither half of the kernel config is optional here.

    HiFi4 alone still misses the budget, so the fp32 accumulator is not a refinement on top of
    it. numeric_stable changes the reduction order rather than the fidelity and measures like
    the stock config. All four variants score above 0.9996 PCC, so only max-abs separates them.
    """
    logits = torch.randn(1, 1, 1024, config.num_experts)
    ref = logits.softmax(dim=-1)
    logits_tt = to_device(logits, device, dtype=tt_config.router_dtype)
    hifi4_only = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def max_abs(**kwargs):
        return compute_max_abs_error(ttnn.to_torch(ttnn.softmax(logits_tt, dim=-1, **kwargs)).float(), ref)

    ported = max_abs(compute_kernel_config=tt_config.compute_kernel_config(OpGroup.SOFTMAX))
    without_fp32_acc = max_abs(compute_kernel_config=hifi4_only)
    stock = max_abs()
    stable = max_abs(numeric_stable=True)

    assert ported < SOFTMAX_MAX_ABS, f"the port's own config missed its budget at {ported:.3e}"
    assert without_fp32_acc > SOFTMAX_MAX_ABS, f"HiFi4 alone should miss it, got {without_fp32_acc:.3e}"
    assert stock > without_fp32_acc, f"HiFi4 should beat the stock config, got {stock:.3e}"
    assert stable > SOFTMAX_MAX_ABS, f"numeric_stable is not a substitute, got {stable:.3e}"


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

    A transposed expert pass casts its (1, E, 1, t) gate to bfloat8_b, so that the gate multiply
    runs two bfloat8_b operands. At t=300 the last tile holds 12 real columns, and 16 columns share
    one exponent, so padding left stale by the permute crushed the real gates beside it: module
    PCC 0.984. TtNomicExperts zeroes the padding first, and the fill writes in place.
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


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_scatter_builds_the_dense_routing_weights(device, tt_config, config, batch, seqlen):
    """aten.zeros_like + aten.scatter_ -> ttnn.zeros_like + ttnn.scatter, on the real chain.

    The index comes straight off the fp32 topk as uint32 and needs no cast; only the values and
    the destination are bfloat16, which is the narrowest cast that satisfies scatter. The dense
    (T, E) result is what the expert gate multiply consumes: exactly moe_top_k non-zero entries
    per row, carrying the softmax weights unrenormalized.
    """
    probabilities = router_probabilities(batch * seqlen, config.num_experts)
    values, indices = ttnn.topk(
        to_device(probabilities, device, dtype=tt_config.router_dtype), k=config.moe_top_k, dim=-1
    )

    dense = ttnn.scatter(
        to_device(torch.zeros_like(probabilities), device),
        dim=-1,
        index=indices,
        src=ttnn.typecast(values, tt_config.activation_dtype),
    )

    ref_values, ref_indices = torch.topk(probabilities, config.moe_top_k, dim=-1)
    ref = torch.zeros_like(probabilities).scatter_(-1, ref_indices, ref_values)
    got = ttnn.to_torch(dense).float()
    assert torch.equal((got != 0).sum(-1), torch.full(got.shape[:-1], config.moe_top_k))
    assert (got.sum(-1) < 1.0).all(), "the top-k weights must reach the gate unrenormalized"
    assert_with_pcc(ref, dense, OPERATOR_PCC)


def test_scatter_rejects_float32(device, config, expect_error):
    """Negative control: the router cannot stay fp32 all the way through.

    This restriction is what forces a cast somewhere after the softmax. It does not dictate
    where: ttnn.topk takes fp32, so the cast lands after the selection rather than before it,
    which is what test_rounding_probabilities_to_bfloat16_before_topk_flips_routing measures.
    """
    probabilities = to_device(router_probabilities(512, config.num_experts), device, dtype=ttnn.float32)
    values, indices = ttnn.topk(ttnn.typecast(probabilities, ttnn.bfloat16), k=config.moe_top_k, dim=-1)

    with expect_error(RuntimeError, "input_dtype == DataType::FLOAT32"):
        ttnn.scatter(ttnn.zeros_like(probabilities), dim=-1, index=indices, src=values)


# Experts.


@pytest.mark.needs_weights
@pytest.mark.parametrize("batch, seqlen", [*TOKEN_SHAPES, (3, 100)])
def test_expert_matmuls(device, tt_config, config, state_dict, batch, seqlen):
    """aten.matmul -> both expert projections, as a pass of this many tokens runs them.

    Every token runs through every expert, which is what removes upstream's data-dependent
    gather. Up to TOKEN_MAJOR_MAX_TOKENS the tokens stay on the rows: w1 is a sparse_matmul with
    every expert enabled and w2 reads its weight through transpose_b. Above, they are the
    columns: w1 is one minimal_matmul of the stacked weight and x^T, and w2 a batched matmul
    writing bfloat8_b. 1x128 and 2x37 take the first, 2x512 and 3x100 the second, 3x100 off the
    tile grid. The chain runs as a pass runs it, each operator on the previous one's output.
    """
    experts = TtNomicExperts(device, config, tt_config, state_dict, EXPERTS_PREFIX)
    w1, w2 = pack_expert_weights(state_dict[EXPERTS_PREFIX + "mlp.w1"], state_dict[EXPERTS_PREFIX + "mlp.w2"], config)
    tokens = batch * seqlen
    token_major = tokens <= TOKEN_MAJOR_MAX_TOKENS
    x = torch.randn(1, 1, tokens, config.hidden_size)

    x_tt = to_device(x, device)
    hidden = experts.token_major_w1(x_tt) if token_major else experts.transposed_w1(x_tt)
    activated = ttnn.gelu(hidden, variant=tt_config.expert_gelu)
    out = experts.token_major_w2(activated) if token_major else experts.transposed_w2(activated)

    def tokens_on_rows(tensor: ttnn.Tensor, width: int) -> torch.Tensor:
        """(1, E, t, width), from either layout."""
        got = ttnn.to_torch(tensor).float()
        return got if token_major else got.reshape(1, config.num_experts, width, tokens).transpose(-2, -1)

    ref_hidden = torch.matmul(x, w1)
    ref_activated = torch.nn.functional.gelu(ref_hidden, approximate="none")
    assert_with_pcc(ref_hidden, tokens_on_rows(hidden, config.intermediate_size), OPERATOR_PCC)
    assert_with_pcc(ref_activated, tokens_on_rows(activated, config.intermediate_size), OPERATOR_PCC)
    assert_with_pcc(torch.matmul(ref_activated, w2), tokens_on_rows(out, config.hidden_size), OPERATOR_PCC)


# One pass per block shape of matmul_config.expert_w1_gelu_config, by token tiles a core holds: 1
# (200 tokens, off the tile grid), 2, 3, 5 and 7 raised to 6 and 8, 9, 11 raised to 12, and 12.
FUSED_W1_TOKENS = [200, 500, 1024, 1600, 2300, 3072, 3584, 4096]

# GELU's minimum is about -0.17, near x = -0.75. bfloat8_b rounds it to -0.25 where a 16-value
# block's largest magnitude is 16 to 32, a step of 0.25, and to less in any other block, so no
# GELU output reads below this. The w1 product reaches -25.
GELU_FLOOR = -0.25


@pytest.mark.needs_weights
@pytest.mark.parametrize("tokens", FUSED_W1_TOKENS)
def test_transposed_w1_fuses_the_gelu_at_every_block_shape(device, tt_config, config, state_dict, tokens):
    """The transposed w1 with its GELU fused, against the same product with the GELU as its own op.

    A 2D multicast ttnn.matmul applies the fused GELU from the packer (matmul_config.gelu_on_packer),
    in a block shape set by the token tiles each core holds; each pass here lands on another one.
    A program that dropped the activation would still return the finite product, as sparse_matmul
    does, so the floor below is asserted as well as the agreement.
    """
    experts = TtNomicExperts(device, config, tt_config, state_dict, EXPERTS_PREFIX)
    x_tt = to_device(torch.randn(1, 1, tokens, config.hidden_size), device)

    fused = ttnn.to_torch(experts.transposed_w1(x_tt, tt_config.expert_gelu)).float()
    product = experts.transposed_w1(x_tt)
    separate = ttnn.to_torch(ttnn.gelu(product, variant=tt_config.expert_gelu)).float()
    raw_min = float(ttnn.to_torch(product).float().min())

    assert raw_min < 4 * GELU_FLOOR, f"the product only reaches {raw_min:.3f}, too close to GELU's floor to test it"
    assert float(fused.min()) >= GELU_FLOOR, f"fused output reaches {float(fused.min()):.3f}: the GELU was not applied"
    assert_with_pcc(separate, fused, OPERATOR_PCC)


@pytest.mark.needs_weights
def test_transposed_expert_weights_are_a_shape_error(device, tt_config, config, state_dict, expect_error):
    """Negative control: on the packed operand, the w2 misorientation is loud.

    Viewing w2 as (E, H, F) instead of (E, F, H) succeeds in torch, because E*F*H is symmetric
    in F and H, and the reference computes uncorrelated noise from it without raising. Packing
    to a 4D operand moves the mistake into the matmul's inner-dimension check, so
    pack_expert_weights is what turns a silent failure into a loud one.
    """
    w2 = state_dict[EXPERTS_PREFIX + "mlp.w2"]
    transposed = w2.view(config.num_experts, config.hidden_size, config.intermediate_size).unsqueeze(0)
    activated = to_device(torch.randn(1, config.num_experts, 128, config.intermediate_size), device)

    with expect_error(RuntimeError, "width of the first tensor must be equal to the height"):
        ttnn.matmul(
            activated,
            to_device(transposed.contiguous(), device),
            compute_kernel_config=tt_config.compute_kernel_config(OpGroup.EXPERT_W2),
        )


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_gate_multiply_and_expert_reduce(device, tt_config, config, batch, seqlen):
    """aten.mul + aten.sum over the expert axis -> ttnn.mul + fast_reduce_nc(dims=[1]).

    The gate arrives as the dense (1, 1, T, E) routing weights permuted to (1, E, T, 1), whose
    trailing singleton broadcasts over the hidden axis. Only moe_top_k of the E slots are
    non-zero, so the reduce is a weighted sum over two experts even though eight were computed.

    Left to allocate its output, fast_reduce_nc returns the tile-padded row count, not T: at T=74
    the output is 96 rows, the trailing 22 zero. The data is right, the logical shape is not, so
    TtNomicExperts passes it an output of the logical shape. ttnn.sum(dim=1, keepdim=True)
    reaches the same PCC without the quirk.
    """
    tokens = batch * seqlen
    experts, hidden = config.num_experts, config.hidden_size
    per_expert = torch.randn(1, experts, tokens, hidden)
    probabilities = router_probabilities(tokens, experts)
    values, indices = torch.topk(probabilities, config.moe_top_k, dim=-1)
    dense = torch.zeros_like(probabilities).scatter_(-1, indices, values)

    gate = ttnn.permute(to_device(dense, device), (0, 3, 2, 1))
    gated = ttnn.multiply(to_device(per_expert, device), gate)
    summed = ttnn.experimental.fast_reduce_nc(
        gated, dims=[1], compute_kernel_config=tt_config.compute_kernel_config(OpGroup.REDUCE)
    )

    ref_gated = per_expert * dense.permute(0, 3, 2, 1)
    padded = -(-tokens // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    reduced = ttnn.to_torch(summed).float()
    assert tuple(gate.shape) == (1, experts, tokens, 1)
    assert tuple(summed.shape) == (1, 1, padded, hidden)
    if padded > tokens:
        assert reduced[:, :, tokens:].abs().max() == 0, "fast_reduce_nc padding is expected to be zero"
    assert_with_pcc(ref_gated, gated, OPERATOR_PCC)
    assert_with_pcc(ref_gated.sum(dim=1, keepdim=True), reduced[:, :, :tokens], OPERATOR_PCC)


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
