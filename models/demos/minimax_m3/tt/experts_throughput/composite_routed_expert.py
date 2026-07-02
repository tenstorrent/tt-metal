# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Composite (non-fused) routed expert for MiniMax-M3 expert-parallel (EP) MoE.

The inherited deepseek routed expert runs the fused C++ op
``ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`` on Blackhole, which bakes in a
**plain SiLU SwiGLU** activation. M3 needs the clamped "swigluoai" activation
(clamp gate to max=limit, up to [-limit, limit], then ``(up + 1) * (gate * sigmoid(alpha*gate))``),
which the fused kernel can't express.

This class reuses deepseek's EP plumbing verbatim — dispatch/combine/reduce, plus the per-expert
``extract`` / ``insert`` ops — but replaces the fused FFN with a per-local-expert loop of plain
ttnn matmuls + our clamped ``apply_swiglu``. It is the FUNCTIONAL bring-up path.

OPTIMIZATION TODO (Blackhole, optimal path): copy+modify the fused
``unified_routed_expert_moe`` compute kernel to apply clamped swigluoai directly (gate/up clamp,
alpha-scaled sigmoid, (up+1) term) and switch back to the single-op path. This composite loop
(one matmul set per expert, host-side loop) is correct but slower than the fused kernel.
"""

from types import SimpleNamespace

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from .activation import apply_swiglu


class CompositeRoutedExpert(TtRoutedExpert):
    """TtRoutedExpert with the fused FFN replaced by a composite clamped-swigluoai FFN."""

    def __init__(self, *args, swiglu_limit: float = 7.0, alpha: float = 1.702, **kwargs):
        super().__init__(*args, **kwargs)
        # apply_swiglu only reads .swiglu_limit and .alpha.
        self._swiglu_cfg = SimpleNamespace(swiglu_limit=swiglu_limit, alpha=alpha)

    def forward(self, dispatched_buffer, expert_token_counts, expert_region_offsets):
        """Per-local-expert: extract -> (gate/up matmul -> clamped swigluoai -> down matmul) -> insert.

        Weight layout (from TtRoutedExpert): gate_proj/up_proj are (emb, hidden), down_proj is
        (hidden, emb), so plain ttnn.linear(tokens, proj) matches the fused op's matmul semantics.
        Forced on all archs (incl. Blackhole) — see module docstring for the fused-kernel optimization.
        """
        if dispatched_buffer.dtype != self.activations_dtype:
            dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)

        expert_outputs = dispatched_buffer
        for e in range(self.experts_per_chip):
            tokens = ttnn.experimental.deepseek_prefill.extract(
                dispatched_buffer,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=e,
                max_dispatched_tokens_per_expert=self.max_tokens,
            )

            gate_out = ttnn.linear(tokens, self.gate_projs[e], compute_kernel_config=self.compute_kernel_config)
            up_out = ttnn.linear(tokens, self.up_projs[e], compute_kernel_config=self.compute_kernel_config)
            tokens.deallocate(True)

            # clamped swigluoai (consumes gate_out/up_out in place, returns the activated buffer)
            activated = apply_swiglu(gate_out, up_out, self._swiglu_cfg)
            output = ttnn.linear(activated, self.down_projs[e], compute_kernel_config=self.compute_kernel_config)
            activated.deallocate(True)

            expert_outputs = ttnn.experimental.deepseek_prefill.insert(
                expert_outputs,
                output,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=e,
            )
            output.deallocate(True)

        return expert_outputs
