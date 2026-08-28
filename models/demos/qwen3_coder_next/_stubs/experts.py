# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextExperts`, EXPERT-parallel over TP chips.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextExperts`.

    y[t] = sum over the experts token t selected of  w[t,e] * down_e( silu(gate_e(x_t)) * up_e(x_t) )

HOW THE SPARSE PYTHON LOOP BECOMES TWO DENSE MATMULS
----------------------------------------------------
The reference walks the hit experts, gathers each one's tokens, and `index_add_`s them back.
Routing is data-dependent, so that loop has no fixed shape -- but the *result* does: an expert a
token did not select contributes with weight zero. Running every local expert densely over every
token and folding the routing weight in before the down projection therefore gives the identical
value, in a shape fixed at build time.

The trick that makes it two matmuls rather than E of them is the FEATURE ORDER. The per-expert
weights are re-laid on the host in FEATURE-MAJOR order -- column `j * E_local + e` belongs to expert
e, intermediate feature j -- so:

  * one (hidden -> 2 * E_local * inter) matmul produces every local expert's gate and up at once;
  * `ttnn.repeat` of the (tokens x E_local) routing weights by `inter` TILES them, and tiling in
    feature-major order lands weight[t, e] on exactly the columns belonging to expert e -- so the
    routing scale is one broadcast multiply, with no gather and no scatter;
  * one (E_local * inter -> hidden) matmul does every local expert's down projection AND the sum
    over experts, because summing over the contracted axis is what a matmul already does.

Tensor-parallel scheme (expert-parallel, per the MoE principle -- never split a single expert):

  * Each chip owns a disjoint block of experts; no expert's gate/up/down is divided, so nothing has
    to be exchanged inside an expert.
  * The routing arrives (or is computed) over ALL experts and is identical on every chip. A chip
    needs only its own experts' columns of it -- and since a device-uniform program cannot slice at
    a chip-dependent offset, that slice is expressed as a matmul against a SHARDED IDENTITY, which
    hands each chip exactly its own block.
  * Each chip's dense pass is a partial sum over the full model dim; one all_reduce combines them.
"""
from __future__ import annotations

import torch
import ttnn

from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    num_devices,
    replicate_mapper,
    shard_mapper,
    to_device,
)


class TtQwen3NextExperts:
    """Native ttnn Qwen3-Next expert bank, expert-parallel over the TP mesh."""

    def __init__(self, device, cfg) -> None:
        self.device = device
        self.__dict__.update(cfg)
        self.num_devices = num_devices(device)
        self._replicate = replicate_mapper(device, self.num_devices)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._arange = None

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("experts stub needs the torch reference module for its weights")

        gate_up = torch_module.gate_up_proj.detach()  # (E, 2*inter, hidden)
        down = torch_module.down_proj.detach()  # (E, hidden, inter)
        n_exp = int(gate_up.shape[0])
        hidden = int(gate_up.shape[-1])
        inter = int(down.shape[-1])

        n = num_devices(device)
        tp = n if n_exp % n == 0 else 1
        e_loc = n_exp // tp

        # Chip-major assembly straight into ONE preallocated buffer per weight: chip c owns the
        # column block [c*2*blk, (c+1)*2*blk) of `w_gate_up` and the row block [c*blk, (c+1)*blk)
        # of `w_down`, which is exactly what `shard_mapper` then splits along.
        blk = inter * e_loc
        w_gate_up = torch.empty(hidden, tp * 2 * blk, dtype=torch.bfloat16)
        w_down = torch.empty(tp * blk, hidden, dtype=torch.bfloat16)
        for c in range(tp):
            gu_c = gate_up[c * e_loc : (c + 1) * e_loc].to(torch.float32)
            # permute(2, 1, 0) puts (hidden, feature, expert) in memory, so the flattened feature
            # axis runs feature-major: column j * e_loc + e.
            base = c * 2 * blk
            w_gate_up[:, base : base + blk] = gu_c[:, :inter, :].permute(2, 1, 0).reshape(hidden, blk).to(torch.bfloat16)
            w_gate_up[:, base + blk : base + 2 * blk] = (
                gu_c[:, inter:, :].permute(2, 1, 0).reshape(hidden, blk).to(torch.bfloat16)
            )
            d_c = down[c * e_loc : (c + 1) * e_loc].to(torch.float32)
            w_down[c * blk : (c + 1) * blk] = d_c.permute(2, 0, 1).reshape(blk, hidden).to(torch.bfloat16)

        cfg = dict(
            hidden_size=hidden,
            num_experts=n_exp,
            experts_local=e_loc,
            intermediate=inter,
            w_gate_up=to_device(
                w_gate_up.unsqueeze(0).unsqueeze(0), device, mesh_mapper=shard_mapper(device, tp, -1)
            ),
            w_down=to_device(
                w_down.unsqueeze(0).unsqueeze(0), device, mesh_mapper=shard_mapper(device, tp, -2)
            ),
            # Sharded identity: `routing @ selector` is the chip-dependent column slice that a
            # device-uniform slice op cannot express.
            selector=to_device(
                torch.eye(n_exp).unsqueeze(0).unsqueeze(0), device, mesh_mapper=shard_mapper(device, tp, -1)
            ),
        )
        return cls(device, cfg)

    # -------------------------------------------------------------- helpers

    def dense_routing(self, top_k_index, top_k_weights, seq):
        """Scatter (index, weight) pairs into a dense (1, 1, tokens, num_experts) routing matrix.

        Done on device as `sum_k (index[:, k] == arange) * weight[:, k]`, in FLOAT32: expert ids run
        past 256, where bfloat16 stops representing consecutive integers exactly and the equality
        test would start matching the wrong expert.
        """
        n_exp = self.num_experts
        top_k = int(top_k_index.shape[-1])
        if self._arange is None:
            self._arange = to_device(
                torch.arange(n_exp, dtype=torch.float32).view(1, 1, 1, n_exp),
                self.device,
                mesh_mapper=self._replicate,
                dtype=ttnn.float32,
            )
        # The graduated `top_k_router` hands these over as DEVICE tensors, so the real pipeline
        # never leaves the chip here.  The host branch below is the per-component PCC harness,
        # which replays the golden's torch (index, weight) pair.
        def _stage(t):
            if isinstance(t, ttnn.Tensor):
                return ttnn.typecast(ttnn.reshape(t, (1, 1, seq, top_k)), ttnn.float32)
            return to_device(
                t.reshape(1, 1, seq, top_k).float(),
                self.device,
                mesh_mapper=self._replicate,
                dtype=ttnn.float32,
            )

        index = _stage(top_k_index)
        weight = _stage(top_k_weights)
        dense = None
        for k in range(top_k):
            col_i = ttnn.slice(index, [0, 0, 0, k], [1, 1, seq, k + 1])
            col_w = ttnn.slice(weight, [0, 0, 0, k], [1, 1, seq, k + 1])
            hot = ttnn.eq(ttnn.repeat(col_i, ttnn.Shape([1, 1, 1, n_exp])), self._arange)
            term = ttnn.multiply(hot, col_w)
            dense = term if dense is None else ttnn.add(dense, term)
        return ttnn.typecast(dense, ttnn.bfloat16)

    def local_routing(self, dense_routing):
        """This chip's columns of a routing matrix that spans all experts."""
        return ttnn.matmul(dense_routing, self.selector, compute_kernel_config=self.compute_config)

    def partial(self, x, local_routing, seq):
        """Every local expert, densely, scaled by its routing weight. Returns a PARTIAL sum."""
        width = self.experts_local * self.intermediate
        gate_up = ttnn.linear(x, self.w_gate_up, compute_kernel_config=self.compute_config)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, seq, width])
        up = ttnn.slice(gate_up, [0, 0, 0, width], [1, 1, seq, 2 * width])
        h = ttnn.multiply(ttnn.silu(gate), up)
        # Tiling the (tokens x E_local) weights `inter` times lands weight[t, e] on every column of
        # expert e, because the feature axis is laid out feature-major.
        h = ttnn.multiply(h, ttnn.repeat(local_routing, ttnn.Shape([1, 1, 1, self.intermediate])))
        return ttnn.linear(h, self.w_down, compute_kernel_config=self.compute_config)

    # -------------------------------------------------------------- forward

    def __call__(self, hidden_states, top_k_index=None, top_k_weights=None, *args, **kwargs):
        rank = len(hidden_states.shape)
        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))

        routing = self.dense_routing(top_k_index, top_k_weights, seq)
        partial = self.partial(x, self.local_routing(routing), seq)
        if self.num_devices > 1:
            partial = ttnn.all_reduce(partial)
        shape = (seq, self.hidden_size) if rank == 2 else (1, seq, self.hidden_size)
        return ttnn.reshape(partial, shape)


def build(device, torch_module=None):
    return TtQwen3NextExperts.build(device, torch_module)


def experts(device, torch_module=None):
    return TtQwen3NextExperts.build(device, torch_module)
