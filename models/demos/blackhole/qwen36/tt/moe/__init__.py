# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse Mixture-of-Experts block for Qwen3.5-MoE (35B-A3B) on Blackhole.

Replaces the dense SwiGLU ``Qwen36MLP`` on MoE layers. Gemma4-style: a dense-routing
router feeds ``sparse_matmul`` experts (expert-parallel: the experts are sharded across
the mesh, each device holding its experts at the full intermediate width), reduce-scattered
after down_proj so the output matches the fractured hidden layout the dense MLP produces
(see ``tt/mlp.py``). An optional gated shared expert is added when the checkpoint has one.
"""

from models.demos.blackhole.qwen36.tt.moe.config import MoEConfig
from models.demos.blackhole.qwen36.tt.moe.moe import Qwen36MoE

__all__ = ["MoEConfig", "Qwen36MoE"]
