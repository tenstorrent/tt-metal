# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview feed-forward sublayers on the SC1 layout (tenstorrent/tt-metal#65890): the dense
layer 0 FFN and the shared expert as 2D tensor parallelism, the router, and the 256 routed experts
expert-parallel over the 32 chips (EP32).

Mirrors the pure-torch ground truth in ../reference/moe.py method for method.

Layout (tt/common.py MeshPlan; the 8x4 mesh: hidden axis = the 4-chip axis, the other axis = 8 chips):

    weight            in -> out           placement                              then
    FFN gate | up     6144 -> 2 x I       in over the hidden axis, I over cols   reduce(hidden)
    FFN down          I -> 6144           I over cols, out over the hidden axis  reduce(cols)
    router            6144 -> 256         whole on every chip
    expert gate | up  6144 -> 2 x 2048    whole, 8 experts per chip
    expert down       2048 -> 6144        whole, 8 experts per chip              reduce(cols), reduce(hidden)

* TtHy4FFN (the dense layer 0, I = 18432, and the shared expert, I = 2048) is sharded once, 1/32 of
  each weight per chip. gate and up are stacked on their output dim, so one matmul and one
  all-reduce give a chip both halves for its slice of the intermediate dim; the activation is
  chip-local; down contracts over the intermediate dim and lands split over the hidden axis, the
  layout the residual streams have. Two collectives.
* The routed experts need the token whole, so the MoE gathers it over the hidden axis once. With
  the token whole on every chip the router is chip-local: every chip holds the 256 x 6144 router
  and computes the same scores, top-8 and weights.
* The router has no scatter: the 8th largest biased score is the threshold, the picks are the
  scores at or above it, and a token's weight for every expert is ``score * picked``, normalised.
  A chip takes the 8 columns that are its own experts from that [T, 256] row.
* A chip's 8 experts sit side by side on the width of one weight, so gate | up is one 2D matmul
  (6144 -> 8 x 4096) and down is another (8 x 2048 -> 6144) whose contraction is also the sum over
  the chip's experts. A batched matmul with the experts on a batch dim overflows L1 with BFP8
  weights at Hy4's widths.
* The routed experts are BFP4 by default and the shared expert BFP8 (EXPERT_WEIGHT_DTYPE below).
* Every chip runs its 8 experts on every token and multiplies by the token's weight for that
  expert, which is zero for the 248 it did not pick. That is exact and has no data-dependent
  control flow, and it costs 32x the arithmetic sparse dispatch would. It is the functional
  version; dispatching only the picked (token, expert) pairs is the performance work.
* The chips' partial sums are reduced over both axes and each chip keeps its hidden slice.

A MoE sublayer costs five collectives: gather(hidden), reduce(cols) + reduce(hidden) for the routed
experts, reduce(hidden) + reduce(cols) for the shared expert.

Tensors are [1, 1, T, width] with the T tokens down the tile rows. The caller applies the layer's
pre-norm before and iHC's hc_post after.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference.moe import MoEConfig
from models.experimental.hy4_preview.tt.common import MeshPlan, compute_config


def _cols(x, start, end):
    """Columns [start, end) of the last dim."""
    shape = list(x.shape)
    return ttnn.slice(x, [0, 0, 0, start], [shape[0], shape[1], shape[2], end])


# Precision policy (tenstorrent/tt-metal#65884, "Experts BFP4, rest BFP8"): the 256 routed experts are
# 80% of a MoE layer's weights, and at BFP4 they leave ~17.6 GB of every chip to the KV cache where
# BFP8 leaves ~5.8 GB. BFP8 experts stay loadable (``expert_dtype=ttnn.bfloat8_b``, #65890).
EXPERT_WEIGHT_DTYPE = ttnn.bfloat4_b


def _fidelity(weight_dtype):
    # BFP8 carries 7 mantissa bits and BFP4 3, which HiFi2 covers; BF16 weights need HiFi4. HiFi2 has a
    # cost PCC does not show: the FFN comes out 0.9% too small on one chip, with BFP8 and with BF16
    # weights alike, where HiFi4 gives 0.9996 and 0.9999 (tests/test_moe.py test_ffn_gain_by_fidelity).
    block_float = weight_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
    return ttnn.MathFidelity.HiFi2 if block_float else ttnn.MathFidelity.HiFi4


class TtHy4FFN(LightweightModule):
    """down(silu(gate(x)) * up(x)) as 2D tensor parallelism. x and the result are [1, 1, T, hidden/tp],
    split over the hidden axis."""

    def __init__(self, plan: MeshPlan, hidden_size, intermediate_size, weights: dict, weight_dtype=ttnn.bfloat8_b):
        super().__init__()
        self.plan = plan
        htp, ctp = plan.hidden_tp, plan.head_tp
        assert hidden_size % htp == 0 and intermediate_size % ctp == 0, (hidden_size, intermediate_size, htp, ctp)
        self.intermediate_per_chip = intermediate_size // ctp
        self.ckc = compute_config(_fidelity(weight_dtype))

        gate = weights["gate_proj.weight"].float().T  # [hidden, I]
        up = weights["up_proj.weight"].float().T
        down = weights["down_proj.weight"].float().T  # [I, hidden]
        assert gate.shape == up.shape == (hidden_size, intermediate_size), (gate.shape, up.shape)
        assert down.shape == (intermediate_size, hidden_size), down.shape
        # (gate | up) per column of the mesh: sharding the last dim hands a chip both halves of its slice.
        gate_up = torch.cat(
            [torch.cat((g, u), dim=-1) for g, u in zip(gate.chunk(ctp, dim=-1), up.chunk(ctp, dim=-1))], dim=-1
        )
        self.w_gate_up = self._weight(gate_up, weight_dtype, plan.shard(hidden_dim=2, head_dim=3))
        self.w_down = self._weight(down, weight_dtype, plan.shard(hidden_dim=3, head_dim=2))

    def _weight(self, w, dtype, mesh_mapper):
        return ttnn.from_torch(
            w.reshape(1, 1, *w.shape).contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            dtype=dtype,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def activation(self, x):
        """x: [1, 1, T, hidden/tp] -> silu(gate) * up for this chip's slice of the intermediate dim."""
        gate_up = ttnn.matmul(x, self.w_gate_up, compute_kernel_config=self.ckc)
        gate_up = self.plan.all_reduce_hidden(gate_up)
        n = self.intermediate_per_chip
        return ttnn.multiply(ttnn.silu(_cols(gate_up, 0, n)), _cols(gate_up, n, 2 * n))

    def down(self, act):
        """act: [1, 1, T, I/ctp] -> [1, 1, T, hidden/tp]."""
        return self.plan.all_reduce(ttnn.matmul(act, self.w_down, compute_kernel_config=self.ckc), self.plan.head_axis)

    def forward(self, x):
        return self.down(self.activation(x))


class TtHy4MoE(LightweightModule):
    """Router, routed experts (expert parallel) and the shared expert. x and the result are
    [1, 1, T, hidden/tp], split over the hidden axis.

    ``weight_dtype`` is the shared expert's dtype. ``expert_dtype`` is the routed experts': left as
    None it is BFP4 (EXPERT_WEIGHT_DTYPE) beside a BFP8 shared expert, the deployment policy, and
    ``weight_dtype`` otherwise, so a BF16 build is BF16 throughout. The router is always BF16.
    """

    def __init__(self, plan: MeshPlan, cfg: MoEConfig, weights: dict, weight_dtype=ttnn.bfloat8_b, expert_dtype=None):
        super().__init__()
        if expert_dtype is None:
            expert_dtype = EXPERT_WEIGHT_DTYPE if weight_dtype == ttnn.bfloat8_b else weight_dtype
        self.expert_dtype = expert_dtype
        self.plan, self.cfg = plan, cfg
        c = cfg
        htp, ctp = plan.hidden_tp, plan.head_tp
        assert c.num_experts % (htp * ctp) == 0, f"{c.num_experts} experts do not split over {htp}x{ctp} chips"
        assert c.hidden_size % htp == 0, (c.hidden_size, htp)
        self.experts_per_chip = per = c.num_experts // (htp * ctp)
        self.ckc = compute_config(_fidelity(expert_dtype))
        # The router decides which experts run, so it keeps full fidelity whatever the experts use.
        self.router_ckc = compute_config(ttnn.MathFidelity.HiFi4)

        E, H, I = c.num_experts, c.hidden_size, c.intermediate_size
        router = weights["gate.weight"].float()
        bias = weights["gate.e_score_correction_bias"].float()
        gate_up = weights["experts.gate_up_proj"]
        down = weights["experts.down_proj"]
        assert router.shape == (E, H) and bias.shape == (E,), (router.shape, bias.shape)
        assert gate_up.shape == (E, 2 * I, H) and down.shape == (E, H, I), (gate_up.shape, down.shape)

        mesh_device = plan.mesh_device
        # The checkpoint is BF16, so BF16 holds the router exactly.
        self.w_router = ttnn.from_torch(
            router.T.reshape(1, 1, H, E).contiguous(), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
        )
        self.router_bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, E), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.float32
        )

        # Expert e lives on chip e // per, counting chips hidden-axis-major: the experts are cut into
        # hidden_tp runs on one tensor dim and each run into head_tp chips on another.
        def placed(t, dtype, head_dim):
            return ttnn.from_torch(
                t.contiguous(),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                dtype=dtype,
                mesh_mapper=plan.shard(hidden_dim=0, head_dim=head_dim),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # A chip's experts side by side on the width, so each projection is one plain 2D matmul and the
        # sum over the chip's experts is the down matmul's own contraction.
        #   gate_up per chip: [1, 1, H, 2 * per * I], columns (gate of expert 0..per-1 | up of expert 0..per-1)
        #   down per chip:    [1, 1, per * I, H]
        gate, up = gate_up.float().reshape(htp, ctp, per, 2, I, H).unbind(dim=3)
        sides = torch.stack((gate.reshape(htp, ctp, per * I, H), up.reshape(htp, ctp, per * I, H)), dim=2)
        del gate, up
        self.w_gate_up = placed(sides.reshape(htp, ctp * 2 * per * I, H).transpose(1, 2).unsqueeze(1), expert_dtype, 3)
        del sides
        self.w_down = placed(down.float().transpose(1, 2).reshape(htp, 1, ctp * per * I, H), expert_dtype, 2)
        # Per chip [1, 1, E, per * I], zeros and ones: takes the chip's own experts out of a [T, E] row
        # of weights and spreads each over that expert's I activations.
        select = torch.eye(E).reshape(htp, ctp * per, E).transpose(1, 2).repeat_interleave(I, dim=2)
        self.select = placed(select.unsqueeze(1), ttnn.bfloat16, 3)

        self.shared = TtHy4FFN(
            plan,
            H,
            c.shared_intermediate_size,
            {
                name: weights[f"shared_experts.{name}"]
                for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
            },
            weight_dtype=weight_dtype,
        )

    # ------------------------------------------------------------------------------------------
    def whole(self, x):
        """[1, 1, T, hidden/tp] split over the hidden axis -> [1, 1, T, hidden] on every chip."""
        return self.plan.all_gather(x, dim=3, axis=self.plan.hidden_axis)

    def scores(self, x_whole):
        """Sigmoid router scores [1, 1, T, E], fp32, the same on every chip."""
        logits = ttnn.matmul(x_whole, self.w_router, compute_kernel_config=self.router_ckc, dtype=ttnn.float32)
        return ttnn.sigmoid(logits)

    def route(self, x_whole):
        """A token's weight for every expert, zero where not picked: [1, 1, T, E], fp32."""
        c = self.cfg
        T = x_whole.shape[-2]
        scores = self.scores(x_whole)
        biased = ttnn.add(scores, self.router_bias)
        top, _ = ttnn.topk(biased, k=c.top_k, dim=-1, largest=True, sorted=True)
        threshold = ttnn.slice(top, [0, 0, 0, c.top_k - 1], [1, 1, T, c.top_k])
        picked = ttnn.multiply(scores, ttnn.ge(biased, threshold))
        total = ttnn.sum(picked, dim=-1, keepdim=True)
        return ttnn.multiply(ttnn.multiply(picked, ttnn.reciprocal(total)), c.routed_scaling_factor)

    def chip_weights(self, dense_weights):
        """[1, 1, T, E] -> this chip's experts' weights, one per activation: [1, 1, T, per * I]."""
        return ttnn.matmul(dense_weights, self.select, compute_kernel_config=self.router_ckc)

    def experts(self, x_whole, chip_weights):
        """This chip's experts on every token, weighted and summed: [1, 1, T, hidden], a partial sum."""
        n, limit = self.experts_per_chip * self.cfg.intermediate_size, self.cfg.swiglu_limit
        gate_up = ttnn.matmul(x_whole, self.w_gate_up, compute_kernel_config=self.ckc)  # [1, 1, T, 2 * per * I]
        gate = ttnn.clamp(_cols(gate_up, 0, n), max=limit)
        up = ttnn.clamp(_cols(gate_up, n, 2 * n), min=-limit, max=limit)
        act = ttnn.multiply(ttnn.multiply(ttnn.silu(gate), up), chip_weights)
        return ttnn.matmul(act, self.w_down, compute_kernel_config=self.ckc)

    def routed(self, x_whole, dense_weights):
        """The weighted sum over all experts: [1, 1, T, hidden/tp], split over the hidden axis."""
        plan = self.plan
        partial = self.experts(x_whole, self.chip_weights(dense_weights))
        total = plan.all_reduce(plan.all_reduce(partial, plan.head_axis), plan.hidden_axis)
        return plan.partition(total, dim=3, axis=plan.hidden_axis)

    def forward(self, x):
        x_whole = self.whole(x)
        return ttnn.add(self.routed(x_whole, self.route(x_whole)), self.shared(x))
