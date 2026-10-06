# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The expert FFN bank, the TTNN form of reference.NomicExperts.dense_forward.

Upstream gathers each expert's tokens by value, runs them, and scatter-adds the results back.
That loop is data-dependent, so it has no device form. dense_forward runs every token through
every expert instead and zeroes the unrouted contributions with the gate, which is
arithmetically identical.

A pass of t tokens runs in one of two layouts. Up to STACKED_MAX_TOKENS the experts sit side by
side on one axis, so each projection is one unbatched matmul over all eight:

    x            (1, 1, t, H)
      w1, gelu   (1, 1, t, E*F)  x times every expert's (H, F) slab side by side, the GELU fused
      gate       (1, 1, t, E*F)  the router's (t, E) weights, each spread over its expert's F
      w2         (1, 1, t, H)    the gated product times every expert's (F, H) slab stacked: one
                                 K = E*F reduction, which sums the experts in its accumulator

Weighting before w2 rather than after is the same sum by linearity, sum_e g_e (h_e W2_e) =
sum_e (g_e h_e) W2_e. It is what lets w2 be one unbatched matmul: a batched (1, E, t, F) w2 has
to multicast its whole activation from one core, while an unbatched one takes it width-sharded,
each core sending its own K range. That took w2 from 141 to 72 us at 128 tokens. The pass
replaced a token-major one of sparse_matmul, GELU, batched w2, multiply and reduce up to 128
tokens and the transposed pass above: 138 -> 127 us at 32 tokens, 277 -> 198 at 128, 380 -> 313
at 256, 386 -> 373 at 320, and 391 against 412 at 352, where the w1 output, the gate and their
product have grown in L1 while the transposed pass holds them in DRAM.

Above it the pass runs transposed, the tokens on the columns:

    x^T          (1, 1, H, t)
      w1, gelu   (1, 1, E*F, t) the checkpoint's own (E*F, H) w1 times x^T, one unbatched product,
                                the GELU fused into it
      w2         (1, E, H, t)
      gate       (1, E, 1, t)
      reduce E   (1, 1, H, t), transposed back to (1, 1, t, H)

There w2 splits H over the grid instead of the tokens. The one shared (H,) bias is added once to
each token's sum, by the stacked w2 itself on a stacked pass.

On a transposed pass w2 writes bfloat8_b, the gate is cast to match and their product stays
bfloat8_b, which halves the multiply's and the reduce's reads; the sum is written in bfloat16.
The gate has to match because ttnn.multiply runs a bfloat8_b and bfloat16 pair on a slower
program, 1338 us against 241 for two bfloat8_b operands at 8x512. There the tokens are the
columns, and a bfloat8_b tile shares one exponent across 16 of them, so the tile padding of x^T
and of the gate is zeroed first: stale values there crush the real tokens beside them. A stacked
pass keeps the tokens on the rows, where a padding row shares no exponent with a real one.

The w1 intermediate is the port's peak transient, 27 MB at T=1024 in the bfloat8_b it is kept in,
and it scales with batch times sequence length rather than sequence length alone. T is the one
axis here that grows without bound, so the pipeline runs in passes over the token axis, capping
that transient at one pass.
"""

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    block_spread,
    pack_expert_weights,
    stack_expert_columns,
    to_device,
    transpose_linear_weight,
)
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import (
    expert_w1_gelu_config,
    expert_w1_transposed_config,
    expert_w2_transposed_program_config,
    gelu_activation,
    gelu_on_packer,
    stacked_columns,
    stacked_columns_config,
    stacked_memory_config,
    stacked_w2_config,
    stacked_w2_shard_tiles,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# The largest pass that runs stacked. See the module docstring.
STACKED_MAX_TOKENS = 320

# The most tile rows of M, as the blocks count them, a stacked pass holds its buffers for. See
# StackedBuffers.
HELD_MAX_TILES = 4

# The token count of one expert pass. It caps the w1 intermediate and the GELU's copy of it, 107 MB
# each in bfloat8_b at this size. One 4096-token pass measured 1.38 ms faster at 8x512 than the
# 3520 + 576 split it replaced, which dated from a deadlock of the broadcast-batch ttnn.matmul that
# no program here runs any more.
MAX_TOKENS_PER_PASS = 4096


class StackedBuffers:
    """A stacked pass's three width-sharded L1 tensors, written by every MoE layer of one forward.

    Allocating a width-sharded L1 tensor takes about 17 us of host time, against about 1 for an
    interleaved one, and a stacked pass writes three: w1's output, the gate and w2's input. Below
    512 tokens the host, not the device, sets the latency, and holding the three from the first MoE
    layer to the end of the forward saves the other five layers their allocations: 0.3 ms at 1x128.

    Only for a layer input of at most HELD_MAX_TILES tile rows as the blocks count them, B times
    S rounded up to the tile, in one pass. The buffers stay in L1 through the layers in between,
    whose plans count L1 as free: SDPA's circular buffers, which grow with S, clashed with them at
    1x320, and at 6x704, whose 4224 tokens run as passes of 4096 and 128; a dense fc1 at 120x1,
    120 tile rows from 120 tokens, overflowed beside them. The owner releases them after the last
    layer; nothing outlives a forward.
    """

    def __init__(self):
        self._tokens = None
        self._tensors = None

    def get(self, tokens: int) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None:
        return self._tensors if tokens == self._tokens else None

    def hold(self, tokens: int, tensors: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]) -> None:
        self.release()
        self._tokens, self._tensors = tokens, tensors

    def release(self) -> None:
        for tensor in self._tensors or ():
            ttnn.deallocate(tensor)
        self._tokens = self._tensors = None


class TtNomicExperts(LightweightModule):
    """All eight experts, each weight in the layout of each form, plus one bias shared across them.

    The bias is added once after the weighted sum. Adding it inside the per-expert loop scales
    it by the routed-weight sum, leaving an offset of (sum(w) - 1) * bias, which is real because
    the weights are not renormalized. That offset is nearly constant across tokens and PCC
    mean-centres, so the wrong version still scores 0.9999998; a per-token projection onto the
    bias sees it (test_shared_bias_is_added_after_the_weighted_sum).

    w2 is kept transposed per expert, (E, H, F), which is also the shape of the one silent
    mistake: viewing the checkpoint's (E*F, H) block as (E, H, F) rather than (E, F, H) is an
    equally legal reshape, since E*F*H is symmetric in F and H. No inner dimension can catch it,
    so the module PCC tests are the guard, and test_misoriented_w2_decorrelates pins that they do.
    set_w2 builds both device forms of w2 from one host tensor, so that test misorients both.

    Each step of a pass is a method of its own, so that the operator tests run the programs a pass
    runs.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.num_experts = config.num_experts
        self.max_tokens_per_pass = MAX_TOKENS_PER_PASS

        w1_stacked, w2 = state_dict[f"{state_dict_prefix}mlp.w1"], state_dict[f"{state_dict_prefix}mlp.w2"]
        _, w2 = pack_expert_weights(w1_stacked, w2, config)
        w1_dtype = tt_config.matmul_weight_dtype(OpGroup.EXPERT_W1)
        # w1 once per form, each converted from the host copy: a device transpose of a bfloat8_b
        # tensor would regroup its shared exponents. The checkpoint's (E*F, H) block, expert-major,
        # serves the transposed pass as is; transposed, it is every expert's (H, F) slab side by
        # side, the stacked pass's (H, E*F).
        self.w1_columns = to_device(
            transpose_linear_weight(w1_stacked).reshape(1, 1, config.hidden_size, -1), device, dtype=w1_dtype
        )
        self.w1_stacked = to_device(w1_stacked.reshape(1, 1, -1, config.hidden_size), device, dtype=w1_dtype)
        self.set_w2(w2.transpose(-2, -1).contiguous(), device)
        # The 0/1 matrix spreading a token's E routing weights over its experts' F columns. Exact
        # in bfloat8_b; in DRAM, where the spread costs the same as from L1 and leaves L1 to SDPA.
        spread = block_spread(config.num_experts, config.intermediate_size)
        self.gate_spread = to_device(spread.reshape(1, 1, *spread.shape), device, dtype=ttnn.bfloat8_b)
        self.bias = to_device(
            state_dict[f"{state_dict_prefix}bias"].reshape(1, 1, 1, config.hidden_size),
            device,
            dtype=tt_config.weight_dtype,
        )

    def set_w2(self, w2_per_expert: torch.Tensor, device) -> None:
        """w2 on device from its (1, E, H, F) host form: as is for a transposed pass, side by side for a stacked."""
        dtype = self.tt_config.matmul_weight_dtype(OpGroup.EXPERT_W2)
        self.w2 = to_device(w2_per_expert, device, dtype=dtype)
        self.w2_columns = to_device(stack_expert_columns(w2_per_expert), device, dtype=dtype)

    def stacked_w1(self, x: ttnn.Tensor, output: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, t, E*F), the GELU fused, width-sharded stacked_columns tiles a core."""
        tokens, hidden = x.shape[-2], x.shape[-1]
        n_tiles = self.w1_columns.shape[-1] // ttnn.TILE_SIZE
        grid = self.tt_config.core_grid
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.STACKED_W1)
        return ttnn.matmul(
            x,
            self.w1_columns,
            program_config=stacked_columns_config(
                ttnn.core.divup(tokens, ttnn.TILE_SIZE),
                hidden // ttnn.TILE_SIZE,
                n_tiles,
                grid,
                compute_kernel_config,
                self.tt_config.expert_gelu,
            ),
            compute_kernel_config=compute_kernel_config,
            dtype=self.tt_config.expert_intermediate_dtype,
            memory_config=stacked_memory_config(tokens, n_tiles, stacked_columns(n_tiles, grid), grid),
            optional_output_tensor=output,
        )

    def stacked_gate(self, dense_weights: ttnn.Tensor, output: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """(1, 1, t, E) -> (1, 1, t, E*F), each routing weight repeated over its expert's F columns.

        A matmul against the 0/1 gate_spread, one non-zero term per sum, laid out as stacked_w1's
        output so the two multiply shard by shard. It writes the activation dtype, in which the
        repeat is exact: bfloat8_b would move a weight by up to a step, 7.8e-3, and the bfloat16
        operand costs the multiply 1 us at 128 tokens.
        """
        tokens = dense_weights.shape[-2]
        n_tiles = self.gate_spread.shape[-1] // ttnn.TILE_SIZE
        grid = self.tt_config.core_grid
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.EXPERT_GATE)
        return ttnn.matmul(
            dense_weights,
            self.gate_spread,
            program_config=stacked_columns_config(
                ttnn.core.divup(tokens, ttnn.TILE_SIZE), 1, n_tiles, grid, compute_kernel_config
            ),
            compute_kernel_config=compute_kernel_config,
            dtype=self.tt_config.activation_dtype,
            memory_config=stacked_memory_config(tokens, n_tiles, stacked_columns(n_tiles, grid), grid),
            optional_output_tensor=output,
        )

    def stacked_w2_input(self, gated: ttnn.Tensor, output: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """stacked_w1's layout -> the wider shards stacked_w2 reads its K blocks in, 6 to 12 us at 128 tokens."""
        tokens, n_tiles = gated.shape[-2], gated.shape[-1] // ttnn.TILE_SIZE
        width = stacked_w2_shard_tiles(ttnn.core.divup(tokens, ttnn.TILE_SIZE))
        return ttnn.reshard(
            gated, stacked_memory_config(tokens, n_tiles, width, self.tt_config.core_grid), output_tensor=output
        )

    def stacked_w2(self, gated: ttnn.Tensor, bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """(1, 1, t, E*F) width-sharded stacked_w2_shard_tiles a core -> (1, 1, t, H), summed over the experts.

        The shared bias, when given, is added by the same program once the sum over the experts is
        complete in its accumulator: still once per token, and an add fewer, 2.4 us on the device and
        25 to 30 us of host time a pass at 128 tokens.
        """
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.STACKED_W2)
        return ttnn.linear(
            gated,
            self.w2_columns,
            bias=bias,
            transpose_b=True,
            program_config=stacked_w2_config(
                ttnn.core.divup(gated.shape[-2], ttnn.TILE_SIZE), self.tt_config.core_grid, compute_kernel_config
            ),
            compute_kernel_config=compute_kernel_config,
            dtype=self.tt_config.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def transposed_w1(self, x: ttnn.Tensor, gelu: ttnn.GeluVariant | None = None) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, E*F, t) for a pass above STACKED_MAX_TOKENS, with the GELU if given.

        The GELU is fused into the matmul: a 2D multicast ttnn.matmul, which applies it from the
        packer, when gelu_on_packer; minimal_matmul, the bare product's program, otherwise.
        """
        tokens, hidden = x.shape[-2], x.shape[-1]
        x_transposed = ttnn.transpose(x, -2, -1)
        if tokens % ttnn.TILE_SIZE:
            # In place: the fill returns a tensor on the same buffer.
            x_transposed = ttnn.fill_implicit_tile_padding(x_transposed, 0.0)
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.EXPERT_W1)
        if gelu is not None and gelu_on_packer(gelu):
            hidden_states = ttnn.matmul(
                self.w1_stacked,
                x_transposed,
                program_config=expert_w1_gelu_config(
                    self.w1_stacked.shape[-2] // ttnn.TILE_SIZE,
                    hidden // ttnn.TILE_SIZE,
                    tokens,
                    self.tt_config.core_grid,
                    gelu,
                    compute_kernel_config,
                ),
                dtype=self.tt_config.expert_intermediate_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            hidden_states = ttnn.experimental.minimal_matmul(
                self.w1_stacked,
                x_transposed,
                config=expert_w1_transposed_config(
                    hidden // ttnn.TILE_SIZE,
                    tokens,
                    self.w1_stacked.dtype,
                    x_transposed.dtype,
                    self.tt_config.expert_intermediate_dtype,
                    self.tt_config.core_grid,
                    self.tt_config.l1_cb_bytes,
                    compute_kernel_config,
                ),
                dtype=self.tt_config.expert_intermediate_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel_config,
                fused_activation=None if gelu is None else gelu_activation(gelu),
            )
        ttnn.deallocate(x_transposed)
        return hidden_states

    def transposed_w2(self, activated: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, E*F, t) -> (1, E, H, t), the GELU of transposed_w1's output."""
        tokens = activated.shape[-1]
        hidden, intermediate = self.w2.shape[-2], self.w2.shape[-1]
        # (1, 1, E*F, t) and (1, E, F, t) are the same tiles, so this is a view.
        activated = ttnn.reshape(activated, (1, self.num_experts, intermediate, tokens))
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.EXPERT_W2)
        return ttnn.matmul(
            self.w2,
            activated,
            program_config=expert_w2_transposed_program_config(
                hidden // ttnn.TILE_SIZE,
                intermediate // ttnn.TILE_SIZE,
                tokens,
                self.w2.dtype,
                activated.dtype,
                self.tt_config.expert_output_dtype,
                self.tt_config.core_grid,
                self.tt_config.l1_cb_bytes,
                compute_kernel_config,
            ),
            compute_kernel_config=compute_kernel_config,
            dtype=self.tt_config.expert_output_dtype,
        )

    def _stacked_sum(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor, buffers: StackedBuffers | None) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, t, H) for a pass of at most STACKED_MAX_TOKENS, the bias included."""
        tokens = x.shape[-2]
        m_tiles = ttnn.core.divup(tokens, ttnn.TILE_SIZE)
        keep = buffers is not None and m_tiles <= HELD_MAX_TILES
        held = buffers.get(tokens) if keep else None
        w1_out, gate_out, w2_in = held or (None, None, None)
        gated = self.stacked_w1(x, w1_out)
        gate = self.stacked_gate(dense_weights, gate_out)
        # In place: allocating a width-sharded output costs about 17 us of host time, and in place
        # the product takes none, 20 us less a pass, bit-identical and as fast on the device.
        ttnn.multiply_(gated, gate)
        if not keep:
            ttnn.deallocate(gate)
        resharded = self.stacked_w2_input(gated, w2_in)
        if not keep:
            ttnn.deallocate(gated)
        elif held is None:
            buffers.hold(tokens, (gated, gate, resharded))
        summed = self.stacked_w2(resharded, self.bias)
        if not keep:
            ttnn.deallocate(resharded)
        return summed

    def _transposed_sum(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, H, t) for a pass above STACKED_MAX_TOKENS."""
        activated = self.transposed_w1(x, self.tt_config.expert_gelu)
        per_expert = self.transposed_w2(activated)
        ttnn.deallocate(activated)

        # (1, 1, t, E) -> (1, E, 1, t), whose singleton row broadcasts over the hidden axis.
        gate = ttnn.permute(dense_weights, (0, 3, 1, 2))
        return self._gated_sum(per_expert, gate)

    def _gated_sum(self, per_expert: ttnn.Tensor, gate: ttnn.Tensor) -> ttnn.Tensor:
        """Weight each expert's output by its gate and sum over the expert axis.

        The gate and the product take the dtype of the per-expert outputs. The sum is the MoE
        output, which joins the residual stream, so it is written in the activation dtype and at
        the logical token count: left to allocate its output, fast_reduce_nc reports the
        tile-padded one, 352 columns at t=330.
        """
        if gate.dtype != per_expert.dtype:
            if gate.shape[-1] % ttnn.TILE_SIZE:
                # The permute leaves the gate's tile padding unset. Only the padding columns share
                # exponents with real gates: garbage there crushed them, PCC 0.984 at 300 tokens.
                # In place: the fill returns a tensor on the same buffer.
                gate = ttnn.fill_implicit_tile_padding(gate, 0.0)
            cast = ttnn.typecast(gate, per_expert.dtype)
            ttnn.deallocate(gate)
            gate = cast
        gated = ttnn.multiply(per_expert, gate, dtype=per_expert.dtype)
        ttnn.deallocate(per_expert)
        ttnn.deallocate(gate)
        summed = ttnn.experimental.fast_reduce_nc(
            gated,
            dims=[1],
            output=ttnn.empty(
                (1, 1, gated.shape[-2], gated.shape[-1]),
                dtype=self.tt_config.activation_dtype,
                layout=self.tt_config.layout,
                device=gated.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.REDUCE),
        )
        ttnn.deallocate(gated)
        return summed

    def _weighted_expert_sum(
        self, x: ttnn.Tensor, dense_weights: ttnn.Tensor, buffers: StackedBuffers | None = None
    ) -> ttnn.Tensor:
        """Run every expert over one pass of tokens, sum the routed contributions and add the bias.

        The bias is shared across the experts, so it is added once to each token's sum and never
        inside an expert's term. Passes split the tokens, so a pass adding it to its own adds it once
        per token too. A stacked pass adds it inside its w2.

        Args:
            x: (1, 1, t, H) flat token activations, t at most max_tokens_per_pass.
            dense_weights: (1, 1, t, E) from TtNomicRouter, zero off the top-k.

        Returns:
            ttnn.Tensor: (1, 1, t, H).
        """
        if x.shape[-2] <= STACKED_MAX_TOKENS:
            return self._stacked_sum(x, dense_weights, buffers)
        transposed = self._transposed_sum(x, dense_weights)
        summed = ttnn.transpose(transposed, -2, -1)
        ttnn.deallocate(transposed)
        out = ttnn.add(summed, self.bias)
        ttnn.deallocate(summed)
        return out

    def forward(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor, buffers: StackedBuffers | None = None) -> ttnn.Tensor:
        """Run every expert, weight the outputs by the routing, sum them and add the shared bias.

        Tokens are processed in passes of at most max_tokens_per_pass (see MAX_TOKENS_PER_PASS).
        One pass covers B*S up to 4096; the chunking shapes in the bring-up tests exceed that and
        take the multi-pass branch, which is the point of them.

        Args:
            x: (1, 1, T, H) flat token activations.
            dense_weights: (1, 1, T, E) from TtNomicRouter, zero off the top-k.
            buffers: held across the MoE layers of a forward, by a caller whose whole input is at
                most HELD_MAX_TILES tile rows (see StackedBuffers); None allocates per pass.

        Returns:
            ttnn.Tensor: (1, 1, T, H).
        """
        tokens, hidden = x.shape[-2], x.shape[-1]
        if tokens <= self.max_tokens_per_pass:
            return self._weighted_expert_sum(x, dense_weights, buffers)

        passes = []
        for begin in range(0, tokens, self.max_tokens_per_pass):
            end = min(begin + self.max_tokens_per_pass, tokens)
            token_slice = ttnn.slice(x, [0, 0, begin, 0], [1, 1, end, hidden])
            weight_slice = ttnn.slice(dense_weights, [0, 0, begin, 0], [1, 1, end, self.num_experts])
            passes.append(self._weighted_expert_sum(token_slice, weight_slice))
            ttnn.deallocate(token_slice)
            ttnn.deallocate(weight_slice)
        out = ttnn.concat(passes, dim=-2)
        for piece in passes:
            ttnn.deallocate(piece)
        return out
