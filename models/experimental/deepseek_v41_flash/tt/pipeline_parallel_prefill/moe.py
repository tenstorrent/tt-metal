# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill MoE, on top of the V4-Flash prefill block.

V4.1 keeps V4's MoE (see :mod:`..decode.moe` for the differences): the router, the routed experts in one fused op
that reads the decode experts' weights in place, and the shared expert added on top. The prefill block is V4's
with V4.1's router (``gate_temp`` folded into the gate, the bias shifted by its max) and shared expert (clamped
like the routed experts) swapped in; every layer is learned-routed, so ``token_ids`` is never needed.

``experts`` is the layer's :class:`~..decode.moe.DeepSeekV41PreloadedExperts`, built for the same ``tp_size``.

The routed op, ``fused_experts_prefill``, handles V4.1's shapes (``D = 5120``, ``I`` per rank 576, 1152 or 2304 at
TP 4, 2 or 1): it runs 3x2 groups of 6 cores instead of 4x2 groups of 8 when that reads fewer weight tiles on the
busiest core (at TP4 the 18 I tiles split 3 per core), and an expert too big for L1 has its weights streamed through
the ring one shard at a time.
"""

from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41MLP, DeepSeekV41TopKRouter
from models.experimental.deepseek_v4_flash.tt.prefill.moe import DeepSeekV4PrefillMoE


class DeepSeekV41PrefillMoE(DeepSeekV4PrefillMoE):
    """V4.1 prefill MoE block: V4's with the V4.1 router and shared expert."""

    router_cls = DeepSeekV41TopKRouter
    shared_expert_cls = DeepSeekV41MLP
