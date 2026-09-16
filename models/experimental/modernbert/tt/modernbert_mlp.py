# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN ModernBERT GeGLU feed-forward.

    out = Wo(gelu(x @ Wi_act) * (x @ Wi_gate))

HF expresses this as a single `Wi` of width 2 * intermediate_size followed by
`chunk(2, dim=-1)`. Here `Wi` is split into two weights at load time and run as
two matmuls, which avoids two runtime slices and measured faster at every
validated shape (seq 256: 189.7 -> 151.1 us).

The GELU is folded into the Wi_act matmul where a program config exists; other
shapes fall back to a plain matmul plus ttnn.gelu. The fusion is worth little on
its own - 0.9 us of 224.9 on the b8s256 sharded geometry - and is kept only
because it is marginally ahead.

The activation lands on `Wi_act`, the FIRST half of HF's `Wi`. This is inverted
relative to the usual SwiGLU convention where the gate receives the activation;
getting it backwards produces plausible-looking but wrong output, so it has a
negative control in the tests.

Both paths take exact-vs-approximate from `_GELU_APPROX` so they cannot drift
apart. See model_config for why that is currently the approximation.
"""

import ttnn
from models.experimental.modernbert.tt import model_config as _cfg
from models.experimental.modernbert.tt.model_config import compute_kernel_config


class TtnnModernBertMLP:
    def __init__(
        self,
        parameters,
        down_core_grid=None,
        act_program_config=None,
        gate_program_config=None,
        shard=None,
    ):
        self.Wi_act = parameters["Wi_act"]
        self.Wi_gate = parameters["Wi_gate"]
        self.Wo = parameters["Wo"]
        self.compute_kernel_config = compute_kernel_config()
        self.down_core_grid = down_core_grid
        # act_program_config carries the fused GELU; gate_program_config must not,
        # since only the first half is activated. Either may be None on an
        # unmeasured shape, in which case ttnn chooses and the GELU runs separately.
        self.act_program_config = act_program_config
        self.gate_program_config = gate_program_config
        # MlpShardPlan when the block runs sharded, else None. When set, the
        # layer hands this module an already-sharded, already-normed tensor.
        self.shard = shard
        # Built once; see the note in modernbert_attention.
        self._act_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if act_program_config is not None:
            self._act_kwargs["program_config"] = act_program_config
        self._gate_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if gate_program_config is not None:
            self._gate_kwargs["program_config"] = gate_program_config
        self._down_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if down_core_grid is not None:
            self._down_kwargs["core_grid"] = down_core_grid

    def __call__(self, hidden_states):
        if self.shard is not None:
            return self._sharded(hidden_states)

        activated = ttnn.linear(hidden_states, self.Wi_act, **self._act_kwargs)
        if self.act_program_config is None:
            unfused = activated
            activated = ttnn.gelu(unfused, fast_and_approximate_mode=_cfg._GELU_APPROX)
            ttnn.deallocate(unfused)

        gate = ttnn.linear(hidden_states, self.Wi_gate, **self._gate_kwargs)

        gated = ttnn.mul(activated, gate)
        ttnn.deallocate(activated)
        ttnn.deallocate(gate)

        out = ttnn.linear(gated, self.Wo, **self._down_kwargs)
        ttnn.deallocate(gated)
        return out

    def _sharded(self, normed):
        """Block-sharded GeGLU: the three intermediates stay in L1 on one 6x8 grid
        and nothing reshards inside the block. Sharding pays across a sequence of
        ops rather than per op - see mlp_shard_plan for the threshold."""
        plan = self.shard

        activated = ttnn.linear(
            normed,
            self.Wi_act,
            program_config=plan.act_matmul,
            memory_config=plan.intermediate_memory,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate = ttnn.linear(
            normed,
            self.Wi_gate,
            program_config=plan.gate_matmul,
            memory_config=plan.intermediate_memory,
            compute_kernel_config=self.compute_kernel_config,
        )

        gated = ttnn.mul(activated, gate, memory_config=plan.intermediate_memory)
        ttnn.deallocate(activated)
        ttnn.deallocate(gate)

        out = ttnn.linear(
            gated,
            self.Wo,
            program_config=plan.down_matmul,
            memory_config=plan.hidden_memory,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(gated)

        if plan.norm is not None:
            # The layer owns the residual add and the trip back to interleaved, so
            # the whole mlp half stays in L1.
            return out

        interleaved = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return interleaved
