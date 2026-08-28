# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextSparseMoeBlock`, EXPERT-parallel over TP chips.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextSparseMoeBlock`
(+ `Qwen3NextTopKRouter`, `Qwen3NextMLP`).

    router: logits = x @ W_g -> softmax -> top-k -> renormalise over the k selected
    out    = sum_e w[t,e] * expert_e(x_t) + sigmoid(x @ w_sg) * shared_expert(x)

The expert bank itself is the sibling `_stubs/experts.py` port (dense feature-major evaluation of
every local expert); this module adds the router, the shared expert, and the collective.

Tensor-parallel scheme:

  * The expert bank is EXPERT-parallel -- see `_stubs/experts.py`. A single expert is never split.
  * The ROUTER is REPLICATED: every chip computes the same logits and the same softmax over ALL
    experts, so the top-k selection and the renormalising denominator agree everywhere. Each chip
    then takes only its own experts' columns, via the expert bank's sharded-identity slice.
  * The SHARED expert is column/row-parallel (gate/up split on output features, down split on the
    matching input features), and it all_reduces its own split.
    Replicating it instead would double-count it once the partials are summed.
    `shared_expert_gate` produces a per-token scalar and stays replicated; scaling each chip's
    partial by it is exact because (a + b) * s == a*s + b*s.
  * One all_reduce over the model dim turns the per-chip partials into the golden output.

The router, the expert bank and the shared expert are NOT re-implemented here -- this block holds
the graduated `top_k_router`, `experts` and `m_l_p` stubs (the ports of `mlp.gate`, `mlp.experts`
and `mlp.shared_expert`) and chains them, so each keeps its own proven TP placement.
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.experts import TtQwen3NextExperts
from models.demos.qwen3_coder_next._stubs.m_l_p import TtQwen3NextMLP
from models.demos.qwen3_coder_next._stubs.top_k_router import TtQwen3NextTopKRouter
from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    matmul_weight,
    num_devices,
    replicate_mapper,
    shard_mapper,
    to_device,
)


class TtQwen3NextSparseMoeBlock:
    """Native ttnn Qwen3-Next sparse MoE block, expert-parallel over the TP mesh."""

    def __init__(self, device, cfg) -> None:
        self.device = device
        self.__dict__.update(cfg)
        self.num_devices = num_devices(device)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("sparse_moe_block stub needs the torch reference module for its weights")

        router = torch_module.gate
        shared = torch_module.shared_expert
        hidden = int(router.hidden_dim)

        n = num_devices(device)
        replicate = replicate_mapper(device, n)

        # The three sub-blocks are the graduated stubs for exactly these HF submodules
        # (`layers.*.mlp.gate`, `layers.*.mlp.experts`, `layers.*.mlp.shared_expert`), each
        # carrying its own proven TP placement:
        #   * top_k_router -- expert-axis column-parallel, all_gather before the softmax
        #   * experts      -- EXPERT-parallel (disjoint expert blocks), partial + all_reduce
        #   * m_l_p        -- column/row-parallel over the shared expert's intermediate features
        cfg = dict(
            hidden_size=hidden,
            top_k=int(router.top_k),
            router=TtQwen3NextTopKRouter.build(device, router),
            experts=TtQwen3NextExperts.build(device, torch_module.experts),
            shared_expert=TtQwen3NextMLP.build(device, shared),
            # The shared-expert SIGMOID gate is the one piece of this block no graduated stub
            # covers, so it stays inline. It is a (hidden -> 1) projection: replicated.
            w_sh_gate_scalar=to_device(
                matmul_weight(torch_module.shared_expert_gate.weight.detach().float()),
                device,
                mesh_mapper=replicate,
            ),
        )
        return cls(device, cfg)

    # -------------------------------------------------------------- forward

    def __call__(self, hidden_states, *args, **kwargs):
        rank = len(hidden_states.shape)
        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))

        # --- routing: the graduated top_k_router (softmax -> top-k -> renormalise) ---------------
        # Its (scores, indices) stay ON DEVICE and are scattered straight back into the dense
        # routing matrix the expert bank contracts against.
        _logits, scores, indices = self.router(x)
        routing = self.experts.dense_routing(indices, scores, seq)

        # --- routed experts: each chip runs its OWN expert block, then one all_reduce ------------
        partial = self.experts.partial(x, self.experts.local_routing(routing), seq)
        if self.num_devices > 1:
            partial = ttnn.all_reduce(partial)

        # --- shared expert: the graduated m_l_p, which all_reduces its own column/row split -----
        shared = ttnn.reshape(self.shared_expert(x), (1, 1, seq, self.hidden_size))
        shared_gate = ttnn.sigmoid(
            ttnn.linear(x, self.w_sh_gate_scalar, compute_kernel_config=self.compute_config)
        )
        out = ttnn.add(partial, ttnn.multiply(shared, shared_gate))

        shape = (seq, self.hidden_size) if rank == 2 else (1, seq, self.hidden_size)
        return ttnn.reshape(out, shape)


def build(device, torch_module=None):
    return TtQwen3NextSparseMoeBlock.build(device, torch_module)


def sparse_moe_block(device, torch_module=None):
    return TtQwen3NextSparseMoeBlock.build(device, torch_module)
