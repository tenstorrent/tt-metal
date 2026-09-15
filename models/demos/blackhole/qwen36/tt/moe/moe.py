# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MoE MLP block. Drop-in replacement for Qwen36MLP on MoE layers:
forward(x) takes a single (ffn-normed, full-hidden) tensor and returns the same
fractured-hidden layout the dense MLP produces.
"""

import ttnn
from models.demos.blackhole.qwen36.tt.moe.experts import Qwen36Experts
from models.demos.blackhole.qwen36.tt.moe.router import Qwen36Router
from models.demos.blackhole.qwen36.tt.moe.shared import Qwen36SharedExpert
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.ccl import tt_all_reduce


class Qwen36MoE:
    def __init__(self, mesh_device, config, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.config = config
        num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        topology = args.ccl_topology() if (args is not None and num_devices > 1) else None
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.topology = topology
        self.num_devices = num_devices

        self.router = Qwen36Router(mesh_device, config, substate(state_dict, "gate"), tensor_cache_path)
        self.experts = Qwen36Experts(
            mesh_device,
            config,
            substate(state_dict, "experts"),
            tensor_cache_path,
            tt_ccl=tt_ccl,
            topology=topology,
        )
        self.shared = None
        if config.shared_intermediate_size:
            self.shared = Qwen36SharedExpert(mesh_device, state_dict, tensor_cache_path, args=args, tt_ccl=tt_ccl)

    def forward(self, x, mode="decode"):
        # The router matmul, the gate_up sparse_matmul and the shared expert all read x; in decode
        # it is one 32-row tile, so stage it in L1 once here rather than letting each op stream it
        # from DRAM (the shared MLP already made its own L1 copy). Prefill x is left in DRAM.
        if mode == "decode" and x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        # Both branches produce a row-parallel partial over the SAME full hidden dim, and
        # reduce-scatter is linear, so RS(routed + shared) == RS(routed) + RS(shared). Summing
        # first costs one collective instead of two. Single-device has no collective to fold.
        fold = self.shared is not None and self.num_devices > 1

        dense_routing = self.router(x)
        out = self.experts(x, dense_routing, mode=mode, reduce=not fold)
        if self.shared is not None:
            shared_out = self.shared.forward(x, reduce=not fold)
            out = ttnn.add(out, shared_out)
            ttnn.deallocate(shared_out)
        if fold:
            out = tt_all_reduce(
                out,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=3,
                topology=self.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return out
