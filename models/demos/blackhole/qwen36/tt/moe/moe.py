# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE sparse MLP block. Drop-in replacement for Qwen36MLP on MoE layers:
forward(x) takes a single (ffn-normed, full-hidden) tensor and returns the same
fractured-hidden layout the dense MLP produces.
"""

import ttnn
from models.demos.blackhole.qwen36.tt.moe.experts import Qwen36Experts
from models.demos.blackhole.qwen36.tt.moe.router import Qwen36Router
from models.demos.blackhole.qwen36.tt.moe.shared import Qwen36SharedExpert
from models.demos.blackhole.qwen36.utils.substate import substate


class Qwen36MoE:
    def __init__(self, mesh_device, config, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.config = config
        num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        topology = args.ccl_topology() if (args is not None and num_devices > 1) else None

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

    def forward(self, x):
        dense_routing = self.router(x)
        out = self.experts(x, dense_routing)
        if self.shared is not None:
            shared_out = self.shared.forward(x)
            out = ttnn.add(out, shared_out)
            ttnn.deallocate(shared_out)
        return out
