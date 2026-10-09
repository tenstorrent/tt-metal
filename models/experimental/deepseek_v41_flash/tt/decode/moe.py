# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode MoE, on top of the V4-Flash block.

V4.1 keeps V4's MoE -- a ``sqrtsoftplus`` router whose bias picks the experts but not their
weights, the routed experts in one ``fused_experts`` op that takes the top-k itself, and a
shared expert added on top -- with these differences:

* ``D = 5120``, ``I = 2304`` and 384 routed experts (V4: 4096, 2048, 256).
* No hash-routed layers: every layer uses the learned router.
* The shared expert takes the routed experts' ``swiglu_limit`` clamp.
* The router bias is ~10 to ~57 depending on the layer, and the op ranks on a bf16 row.
"""

import dataclasses
from types import SimpleNamespace

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.decode.moe import (
    _FUSED_GRID_Y,
    _FUSED_PARALLEL_EXPERTS,
    _FUSED_PARALLEL_GRID_X,
    _FUSED_TILE,
    DeepSeekV4MLP,
    DeepSeekV4PreloadedExperts,
    DeepSeekV4SparseMoeBlock,
    DeepSeekV4TopKRouter,
    _fused_hidden,
)
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import _memo

DECODE_LAYOUTS = {
    # One 32-expert tile per core; 12x1 sits inside the 12x8 fused-experts replica.
    "router_gate": {"K": 5120, "N": 384, "n_blocks": 12},
    # 64 columns per core; 12x3 sits inside the same replica.
    "shared_gate_proj": {"K": 5120, "N": 2304, "n_blocks": 36},
    "shared_up_proj": {"K": 5120, "N": 2304, "n_blocks": 36},
    "shared_down_proj": {"K": 2304, "N": 5120},
}
# The 6-expert path of ``fused_experts`` splits ``I`` tile-evenly over each expert's 16 cores.
_I_ALIGN = _FUSED_TILE * _FUSED_PARALLEL_GRID_X * _FUSED_GRID_Y // _FUSED_PARALLEL_EXPERTS


class DeepSeekV41TopKRouter(DeepSeekV4TopKRouter):
    """V4's router with ``gate_temp`` folded into the gate weight and the bias shifted by its max.

    The op ranks on ``scores + bias`` in bf16, whose step at V4.1's bias (up to ~57) is as large
    as the gaps between the scores, so it would mostly pick different experts than the fp32
    reference. A constant shift does not change the ranking, and after it the bias is in
    ``[-0.5, 0]``, where bf16 steps are ~100x finer. The gate stays bf16, the checkpoint's own
    dtype, whatever ``weight_dtype`` the block uses for the shared expert.
    """

    decode_layouts = DECODE_LAYOUTS

    def __init__(self, config, weights: dict, device, cache=None, weight_dtype=None, **kwargs):
        gate = _memo(weights["gate.weight"])
        bias = _memo(weights["gate.e_score_correction_bias"])
        temp = getattr(config, "gate_temp", 1.0)
        weights = {
            "gate.weight": lambda: gate() / temp,
            "gate.e_score_correction_bias": lambda: bias() - bias().max(),
        }
        super().__init__(config, weights, device, cache=cache, weight_dtype=ttnn.bfloat16, **kwargs)


class DeepSeekV41MLP(DeepSeekV4MLP):
    """V4's shared expert at V4.1's shapes, clamped like the routed experts."""

    decode_layouts = DECODE_LAYOUTS
    clamp_swiglu = True


class DeepSeekV41PreloadedExperts(DeepSeekV4PreloadedExperts):
    """V4's resident ``fused_experts`` experts at V4.1's shapes.

    The op cuts the hidden row into 64-column DRAM shards, 80 at ``D = 5120`` (V4: 64), and
    its 6-expert path needs ``I`` to split evenly over 16 cores in whole tiles. ``I = 2304``
    (72 tiles) does not, so it is zero-padded to 2560: zero gate/up rows give
    ``silu(0) * 0 = 0`` and zero down columns add nothing, so the padding is exact. It costs
    ~11% more expert bytes (~8.5 GB per layer at bfloat4_b).

    ``provider(e) -> (gate_up [2I, D], down [D, I])`` is V4's contract, unpadded (see
    :func:`~...config.expert_provider`).
    """

    def __init__(self, config, provider, device, dtype=None, cache=None, system_config=None):
        intermediate = config.moe_intermediate_size
        padded = -(-intermediate // _I_ALIGN) * _I_ALIGN
        pad = padded - intermediate

        def padded_provider(e: int):
            gate_up, down = provider(e)
            zeros = gate_up.new_zeros(pad, gate_up.shape[1])
            gate, up = gate_up.split(intermediate)
            return torch.cat([gate, zeros, up, zeros]), torch.nn.functional.pad(down, (0, pad))

        sys_cfg = system_config or active_system_config()
        sys_cfg = dataclasses.replace(
            sys_cfg, moe=dataclasses.replace(sys_cfg.moe, fused_num_cores=config.hidden_size // _fused_hidden(1))
        )
        super().__init__(
            SimpleNamespace(**{**vars(config), "moe_intermediate_size": padded}),
            padded_provider,
            device,
            dtype=dtype,
            cache=cache,
            system_config=sys_cfg,
        )


class DeepSeekV41SparseMoeBlock(DeepSeekV4SparseMoeBlock):
    """V4's MoE block with the V4.1 router and shared expert, decoded through ``matmul_decode``.

    ``weights`` comes from :func:`~...config.moe_weights` and ``experts`` is a
    :class:`DeepSeekV41PreloadedExperts`. ``weight_dtype`` is the shared expert's: bfloat8_b by
    default, since at bfloat4_b its error is larger than the routed experts'.
    """

    router_cls = DeepSeekV41TopKRouter
    shared_expert_cls = DeepSeekV41MLP

    def __init__(self, config, weights: dict, device, experts, cache=None, weight_dtype=ttnn.bfloat8_b):
        super().__init__(config, weights, device, experts, cache=cache, weight_dtype=weight_dtype, matmul_decode=True)
