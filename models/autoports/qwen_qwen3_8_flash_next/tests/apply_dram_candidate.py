# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Install setup-only DRAM candidate policies without changing decoder defaults.

Python: apply("shared", {"decode": {...}, "rows": {...}}).
A flat config configures Q1 only; omitted/None phases retain the original path.
CLI: --role shared --config-json '{"cores":8,"readers":1,"block":10}'
followed by ordinary probe_multichip_decoder arguments. --policies-json accepts
multiple role policies in one invocation. Every policy is fixed before model
construction, with no device work during this module's import.
"""

import argparse
import json
import sys
from types import MethodType


def apply(role, config):
    if role not in ("shared", "gdn", "gdn-compact", "qsa", "qsa-compact", "wo", "gr"):
        raise ValueError("role must be shared/gdn/qsa/wo/gr")
    import ttnn as t
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as m
    from models.autoports.qwen_qwen3_8_flash_next.tt.dram_projection import (
        PhaseDramProjection,
        PhaseDramProjectionGroup,
        phase_configs,
    )

    # Validate before constructor/device admission; neither import nor policy
    # parsing opens the mesh. Allocation happens only during model construction.
    phase_configs(config)

    def wrap(loader, parts, original, compute):
        group = PhaseDramProjectionGroup(loader, parts, config)
        return PhaseDramProjection(group, lambda x: m.linear(x, original, compute))

    if role in ("gdn", "gdn-compact"):
        original_init = m.MeshGDN.__init__

        def initialize(self, loader, *args, **kwargs):
            original_init(self, loader, *args, **kwargs)
            weights = [
                loader.state_dict["linear_attn.in_proj_" + name + ".weight"].chunk(4, dim=0) for name in ("z", "a", "b")
            ]
            parts = [[w[r] for w in weights] for r in range(4)]
            if role == "gdn-compact":
                import torch

                parts = [[torch.cat(rank, dim=0)] for rank in parts]
            self.zab = wrap(loader, parts, self.zab, self.compute)

        m.MeshGDN.__init__ = initialize
    elif role == "shared":
        original_init = m.MeshMoe.__init__

        def initialize(self, loader, *args, **kwargs):
            original_init(self, loader, *args, **kwargs)
            weights = [
                loader.state_dict["mlp.shared_expert." + name + "_proj.weight"].chunk(4, dim=0)
                for name in ("gate", "up")
            ]
            original_weights = list(self.shared)
            if not original_weights and any(policy is None for policy in phase_configs(config)):
                original_weights = [
                    loader.shards([part.T for part in weight], dim=1, dtype=t.bfloat8_b) for weight in weights
                ]
            self.dram_shared = PhaseDramProjectionGroup(
                loader,
                [[w[r] for w in weights] for r in range(4)],
                config,
                native_forward=lambda x: [m.linear(x, w, self.compute) for w in original_weights],
            )
            if self.dram_shared.fully_selected:
                # The instance forward below owns gate/up for every shape.
                # Drop the old list as well as the native callback's references.
                self.shared = []

            def shared_forward(this, x):
                gate, up = this.dram_shared(x)
                middle = t.multiply(t.silu(gate), up)
                shared = m.linear(middle, this.shared_down, this.compute)
                return t.multiply(shared, t.sigmoid(m.linear(x, this.shared_route, this.compute)))

            self.shared_forward = MethodType(shared_forward, self)

        m.MeshMoe.__init__ = initialize
    elif role in ("qsa", "qsa-compact", "wo"):
        original_build = m.MeshQSA.build.__func__

        def build(cls, loader, *args, **kwargs):
            result = original_build(cls, loader, *args, **kwargs)
            w = loader.state_dict
            if role in ("qsa", "qsa-compact"):
                q = w["self_attn.q_proj.weight"].chunk(4, dim=0)
                k = w["self_attn.k_proj.weight"].chunk(2, dim=0)
                v = w["self_attn.v_proj.weight"].chunk(2, dim=0)
                parts = [[q[r], k[r // 2], v[r // 2]] for r in range(4)]
                if role == "qsa-compact":
                    import torch

                    parts = [[torch.cat(rank, dim=0)] for rank in parts]
                result.packed_qkv = wrap(
                    loader,
                    parts,
                    result.packed_qkv,
                    result.attention.compute,
                )
            else:
                result.attention.out = wrap(
                    loader,
                    [[w] for w in w["self_attn.o_proj.weight"].chunk(4, dim=1)],
                    result.attention.out,
                    result.attention.compute,
                )
            return result

        m.MeshQSA.build = classmethod(build)
    else:
        original_init = m.MeshResidual.__init__

        def initialize(self, loader, prefix, *args, **kwargs):
            original_init(self, loader, prefix, *args, **kwargs)
            parts = [[p] for p in loader.state_dict[prefix + "block_inject_weight.weight"].chunk(4, dim=0)]
            self.inject = wrap(loader, parts, self.inject, self.compute)

        m.MeshResidual.__init__ = initialize


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", choices=("shared", "gdn", "gdn-compact", "qsa", "qsa-compact", "wo", "gr"))
    parser.add_argument("--config-json")
    parser.add_argument("--policies-json")
    parser.add_argument("--base-candidate", help="Optional existing candidate composed before DRAM policies")
    args, remaining = parser.parse_known_args()
    if args.policies_json is not None:
        if args.role is not None or args.config_json is not None:
            parser.error("Use either policies-json or role plus config-json")
        policies = json.loads(args.policies_json)
    else:
        if args.role is None or args.config_json is None:
            parser.error("Need role/config-json or policies-json")
        policies = {args.role: json.loads(args.config_json)}
    if args.base_candidate:
        from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_optimization_candidates import apply as apply_base

        apply_base(args.base_candidate)
    for role, config in policies.items():
        apply(role, config)
    sys.argv = [sys.argv[0], *remaining]
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder import main as probe_main

    probe_main()


if __name__ == "__main__":
    main()
