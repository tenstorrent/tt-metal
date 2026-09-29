# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The expert FFN bank, the TTNN form of reference.NomicExperts.dense_forward.

Upstream gathers each expert's tokens by value, runs them, and scatter-adds the results back.
That loop is data-dependent, so it has no device form. dense_forward runs every token through
every expert instead and zeroes the unrouted contributions with the gate, which is
arithmetically identical and reduces to two batched matmuls, a multiply and a reduce.

A pass of t tokens runs in one of two layouts. Up to TOKEN_MAJOR_MAX_TOKENS the tokens stay on
the rows, and the pass keeps its tensors in L1:

    x            (1, 1, t, H)
      w1         (1, E, t, F)   sparse_matmul, every expert enabled
      gelu, w2   (1, E, t, H)
      gate       (1, E, t, 1)   from the router, zero off the top-k
      reduce E   (1, 1, t, H)

Above it the pass runs transposed, the tokens on the columns:

    x^T          (1, 1, H, t)
      w1         (1, 1, E*F, t) the checkpoint's own (E*F, H) w1 times x^T, one unbatched product
      gelu, w2   (1, E, H, t)
      gate       (1, E, 1, t)
      reduce E   (1, 1, H, t), transposed back to (1, 1, t, H)

x is shared by all eight experts. ttnn.matmul broadcasts a shared in1 on every program, but a
shared in0 only on one that streams all of the experts' weights through a single core, so the
token-major form needs sparse_matmul, whose one K block bounds its rows, or a copy of x per
expert. The transposed form needs neither. Its w2 splits H over the grid instead of the tokens,
which loses to the token-major w2 at 4 tile rows and below. The one shared (H,) bias is added
once, after the sum of every pass.

On a transposed pass w2 writes bfloat8_b, the gate is cast to match and their product stays
bfloat8_b, which halves the multiply's and the reduce's reads; the sum is written in bfloat16.
The gate has to match because ttnn.multiply runs a bfloat8_b and bfloat16 pair on a slower
program, 1338 us against 241 for two bfloat8_b operands at 8x512. There the tokens are the
columns, and a bfloat8_b tile shares one exponent across 16 of them, so the tile padding of x^T
and of the gate is zeroed first: stale values there crush the real tokens beside them. A
token-major pass keeps the three in bfloat16: its gate would need the same fill before a cast,
8 to 13 us a pass, against the 2 us a bfloat8_b multiply saves there.

The w1 intermediate and the GELU's copy of it are the port's peak transient, 2 x 27 MB at T=1024
in the bfloat8_b they are kept in, and they scale with batch times sequence length rather than
sequence length alone. T is the one axis here that grows without bound, so the pipeline runs in
passes over the token axis, capping that transient at one pass.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import pack_expert_weights, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import (
    expert_w1_program_config,
    expert_w1_transposed_config,
    expert_w2_program_config,
    expert_w2_transposed_program_config,
    l1_bank_bytes,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# The largest pass that runs token-major, 4 tile rows. There the 1D w2 beats the transposed one,
# 158 against 221 us at 128 tokens; at 256 the transposed pass is 487 against 512 us, its w1 112
# against sparse_matmul's 160. Pass sizes in between were not measured.
TOKEN_MAJOR_MAX_TOKENS = 128

# The token count of one expert pass. It caps the w1 intermediate and the GELU's copy of it, 107 MB
# each in bfloat8_b at this size. One 4096-token pass measured 1.38 ms faster at 8x512 than the
# 3520 + 576 split it replaced, which dated from a deadlock of the broadcast-batch ttnn.matmul that
# no program here runs any more.
MAX_TOKENS_PER_PASS = 4096


class TtNomicExperts(LightweightModule):
    """All eight experts, each weight in the layout of each form, plus one bias shared across them.

    The bias is added once after the weighted sum. Adding it inside the per-expert loop scales
    it by the routed-weight sum, leaving an offset of (sum(w) - 1) * bias, which is real because
    the weights are not renormalized. That offset is nearly constant across tokens and PCC
    mean-centres, so the wrong version still scores 0.9999998; only max-abs sees it.

    w2 is kept transposed per expert, (E, H, F), which is also the shape of the one silent
    mistake: viewing the checkpoint's (E*F, H) block as (E, H, F) rather than (E, F, H) is an
    equally legal reshape, since E*F*H is symmetric in F and H. No inner dimension can catch it,
    so the module PCC tests are the guard, and test_misoriented_w2_decorrelates pins that they do.

    Each matmul is a method of its own, so that the operator tests run the programs a pass runs.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.num_experts = config.num_experts
        self.max_tokens_per_pass = MAX_TOKENS_PER_PASS

        w1_stacked, w2 = state_dict[f"{state_dict_prefix}mlp.w1"], state_dict[f"{state_dict_prefix}mlp.w2"]
        w1, w2 = pack_expert_weights(w1_stacked, w2, config)
        w1_dtype = tt_config.matmul_weight_dtype(OpGroup.EXPERT_W1)
        # w1 twice, one per form, since sparse_matmul cannot transpose an operand: (E, H, F) for
        # it, and as the checkpoint stores it, every expert's (F, H) slab stacked, for the
        # transposed pass. Both are converted from the host copy: a device transpose of a
        # bfloat8_b tensor would regroup its shared exponents.
        self.w1 = to_device(w1, device, dtype=w1_dtype)
        self.w1_stacked = to_device(w1_stacked.reshape(1, 1, -1, config.hidden_size), device, dtype=w1_dtype)
        # w2 once, transposed per expert: the transposed pass takes it as is, and the token-major
        # pass through transpose_b, where it makes each core's weight column one contiguous row.
        self.w2 = to_device(
            w2.transpose(-2, -1).contiguous(), device, dtype=tt_config.matmul_weight_dtype(OpGroup.EXPERT_W2)
        )
        # The w1 sparse_matmul mask. All ones: every token still runs through every expert.
        self.every_expert = ttnn.ones(
            (1, 1, 1, config.num_experts), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.bias = to_device(
            state_dict[f"{state_dict_prefix}bias"].reshape(1, 1, 1, config.hidden_size),
            device,
            dtype=tt_config.weight_dtype,
        )

    def token_major_w1(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, E, t, F) in L1, for a pass of at most TOKEN_MAJOR_MAX_TOKENS.

        At most TOKEN_MAJOR_MAX_TOKENS tokens, so the output takes at most 30 KB of each L1 bank.
        Writing it to L1 made w1 about 10% faster at 2x37; w2 reads it no faster than from DRAM.
        """
        tokens, hidden = x.shape[-2], x.shape[-1]
        intermediate = self.w1.shape[-1]
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.EXPERT_W1)
        output_bank = l1_bank_bytes(
            self.num_experts * ttnn.core.divup(tokens, ttnn.TILE_SIZE) * (intermediate // ttnn.TILE_SIZE),
            self.tt_config,
            self.tt_config.expert_intermediate_dtype,
        )
        # Left to allocate its output, sparse_matmul zero-fills it for the pairs a mask skips:
        # a full write of the (1, E, t, F) intermediate that every expert being active then
        # overwrites. An output of the compact (1, nnz, t, F) shape is written once, and with
        # nnz = E it is already the layout w2 takes.
        return ttnn.sparse_matmul(
            x,
            self.w1,
            sparsity=self.every_expert,
            nnz=self.num_experts,
            program_config=expert_w1_program_config(
                tokens,
                hidden // ttnn.TILE_SIZE,
                intermediate // ttnn.TILE_SIZE,
                x.dtype,
                self.w1.dtype,
                self.tt_config.expert_intermediate_dtype,
                self.tt_config.core_grid,
                self.tt_config.l1_cb_bytes - output_bank,
                compute_kernel_config,
            ),
            compute_kernel_config=compute_kernel_config,
            optional_output_tensor=ttnn.empty(
                (1, self.num_experts, tokens, intermediate),
                dtype=self.tt_config.expert_intermediate_dtype,
                layout=self.tt_config.layout,
                device=x.device(),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
        )

    def token_major_w2(self, activated: ttnn.Tensor) -> ttnn.Tensor:
        """(1, E, t, F) in L1 -> (1, E, t, H) in L1, the GELU of token_major_w1's output."""
        tokens, intermediate = activated.shape[-2], activated.shape[-1]
        hidden = self.w2.shape[-2]
        tile_rows = ttnn.core.divup(tokens, ttnn.TILE_SIZE)
        input_bank = l1_bank_bytes(
            self.num_experts * tile_rows * (intermediate // ttnn.TILE_SIZE), self.tt_config, activated.dtype
        )
        output_bank = l1_bank_bytes(self.num_experts * tile_rows * (hidden // ttnn.TILE_SIZE), self.tt_config)
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.EXPERT_W2)
        # The w1 output the GELU freed sits above the GELU's, where no circular buffer can reach.
        return ttnn.matmul(
            activated,
            self.w2,
            transpose_b=True,
            program_config=expert_w2_program_config(
                tokens,
                intermediate // ttnn.TILE_SIZE,
                activated.dtype,
                self.w2.dtype,
                self.tt_config.activation_dtype,
                self.tt_config.core_grid,
                self.tt_config.l1_cb_bytes - 2 * input_bank - output_bank,
                compute_kernel_config,
            ),
            compute_kernel_config=compute_kernel_config,
            dtype=self.tt_config.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def transposed_w1(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, E*F, t), for a pass above TOKEN_MAJOR_MAX_TOKENS."""
        tokens, hidden = x.shape[-2], x.shape[-1]
        x_transposed = ttnn.transpose(x, -2, -1)
        if tokens % ttnn.TILE_SIZE:
            # In place: the fill returns a tensor on the same buffer.
            x_transposed = ttnn.fill_implicit_tile_padding(x_transposed, 0.0)
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.EXPERT_W1)
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

    def _token_major_sum(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, t, H) for a pass of at most TOKEN_MAJOR_MAX_TOKENS."""
        hidden_states = self.token_major_w1(x)
        activated = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states)
        per_expert = self.token_major_w2(activated)
        ttnn.deallocate(activated)

        # (1, 1, t, E) -> (1, E, t, 1), whose trailing singleton broadcasts over the hidden axis.
        gate = ttnn.permute(dense_weights, (0, 3, 2, 1))
        return self._gated_sum(per_expert, gate)

    def _transposed_sum(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """(1, 1, t, H) -> (1, 1, H, t) for a pass above TOKEN_MAJOR_MAX_TOKENS."""
        hidden_states = self.transposed_w1(x)
        activated = ttnn.gelu(hidden_states)
        ttnn.deallocate(hidden_states)
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
        tile-padded one, 96 rows at t=74.
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

    def _weighted_expert_sum(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run every expert over one pass of tokens and sum the routed contributions.

        The bias is not added here: it is shared across experts and across passes, so it is
        added once to the assembled result rather than once per pass.

        Args:
            x: (1, 1, t, H) flat token activations, t at most max_tokens_per_pass.
            dense_weights: (1, 1, t, E) from TtNomicRouter, zero off the top-k.

        Returns:
            ttnn.Tensor: (1, 1, t, H).
        """
        if x.shape[-2] <= TOKEN_MAJOR_MAX_TOKENS:
            return self._token_major_sum(x, dense_weights)
        transposed = self._transposed_sum(x, dense_weights)
        summed = ttnn.transpose(transposed, -2, -1)
        ttnn.deallocate(transposed)
        return summed

    def forward(self, x: ttnn.Tensor, dense_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run every expert, weight the outputs by the routing, and sum them.

        Tokens are processed in passes of at most max_tokens_per_pass (see MAX_TOKENS_PER_PASS).
        One pass covers B*S up to 4096; the chunking shapes in the bring-up tests exceed that and
        take the multi-pass branch, which is the point of them.

        Args:
            x: (1, 1, T, H) flat token activations.
            dense_weights: (1, 1, T, E) from TtNomicRouter, zero off the top-k.

        Returns:
            ttnn.Tensor: (1, 1, T, H).
        """
        tokens, hidden = x.shape[-2], x.shape[-1]

        if tokens <= self.max_tokens_per_pass:
            summed = self._weighted_expert_sum(x, dense_weights)
        else:
            passes = []
            for begin in range(0, tokens, self.max_tokens_per_pass):
                end = min(begin + self.max_tokens_per_pass, tokens)
                token_slice = ttnn.slice(x, [0, 0, begin, 0], [1, 1, end, hidden])
                weight_slice = ttnn.slice(dense_weights, [0, 0, begin, 0], [1, 1, end, self.num_experts])
                passes.append(self._weighted_expert_sum(token_slice, weight_slice))
                ttnn.deallocate(token_slice)
                ttnn.deallocate(weight_slice)

            summed = ttnn.concat(passes, dim=-2)
            for piece in passes:
                ttnn.deallocate(piece)

        out = ttnn.add(summed, self.bias)
        ttnn.deallocate(summed)
        return out
