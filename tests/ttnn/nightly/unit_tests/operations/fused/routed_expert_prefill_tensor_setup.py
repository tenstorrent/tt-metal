# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device placement of the routed-expert op's weight tensors, owned by the op implementor.

test_routed_expert_prefill.py prescribes WHAT the op is fed -- one expert, its shape, how many
tokens reach it, the weight values and the reference it is graded against -- and nothing about
HOW those tensors sit on the device. The layout / memory-config choices live here instead, in
one function, so an op implementor can retune placement without touching the test.

What is NOT here, deliberately:

* dtype. Precision is the model's, not the op's: each model config names its own
  ROUTED_EXPERT_WEIGHTS_DTYPE / ROUTED_EXPERT_BIAS_DTYPE, which the test reads. An op
  implementation does not get to grade itself at a precision the model never serves.
* the token buffer, whose placement is the format the upstream dispatch hands the op.

These are offers, not guarantees: the op validates its own inputs (TILE weights, either
DRAM-interleaved or DRAM ND-sharded on tile-row boundaries; TILE DRAM-interleaved biases), and
a placement it does not accept fails that validation with its own message rather than being
silently mis-run.
"""

from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config


@dataclass(frozen=True)
class TensorSetup:
    """How one op input is placed on the device."""

    layout: ttnn.Layout
    memory_config: ttnn.MemoryConfig


@dataclass(frozen=True)
class RoutedExpertTensorSetup:
    """One TensorSetup per op input, named after the op parameter it belongs to.

    The bias entries are read only for models whose experts carry biases.
    """

    gate_proj: TensorSetup
    up_proj: TensorSetup
    down_proj: TensorSetup
    gate_bias: TensorSetup
    up_bias: TensorSetup
    down_bias: TensorSetup


def routed_expert_prefill_tensor_setup(mesh_device, emb_dim: int, hidden_dim: int) -> RoutedExpertTensorSetup:
    """Placement for the expert's weights and biases.

    ``emb_dim`` / ``hidden_dim`` are the expert's K and N: gate and up are (emb_dim, hidden_dim),
    down is (hidden_dim, emb_dim), so a placement that depends on a tensor's free dim (an
    N-sharded weight, say) gets it from here rather than guessing.
    """
    bias = TensorSetup(ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)

    def weight(n_dim: int) -> TensorSetup:
        # DRAM ND-sharded on the free dim: the Blackhole default, which lets the FFN fetch a whole
        # per-core N slice in one NoC request. See routed_expert_weight_memory_config.
        return TensorSetup(
            ttnn.TILE_LAYOUT,
            routed_expert_weight_memory_config(mesh_device, n_dim, dram_nd_sharded=True),
        )

    return RoutedExpertTensorSetup(
        gate_proj=weight(hidden_dim),
        up_proj=weight(hidden_dim),
        down_proj=weight(emb_dim),
        gate_bias=bias,
        up_bias=bias,
        down_bias=bias,
    )
