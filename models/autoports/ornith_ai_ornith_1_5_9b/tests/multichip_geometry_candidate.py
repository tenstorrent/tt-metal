# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-stack test of isolated DRAM geometry winners, including input reshards."""

import argparse
import json
import math
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_create_candidate import QkvCreateCandidate


class GeometryCandidate(QkvCreateCandidate):
    selected_role = "all"
    preserve_inputs = False

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        keys = {
            "packed_input": (8, 48, 4, 2) if self.kind == "linear_attention" else (64, 16, 2, 2),
            "mlp.gate_proj": (8, 32, 2, 2),
            "mlp.up_proj": (8, 32, 2, 2),
            "mlp.down_proj": (8, 80, 4, 2),
            "linear_attn.out_proj": (32, 64, 4, 1),
            "self_attn.o_proj": (32, 64, 4, 1),
        }
        self.candidate_geometry = {}
        for name, (inc, outc, readers, block) in keys.items():
            if name not in self.dram_layouts:
                continue
            family = (
                "input"
                if name == "packed_input"
                else (
                    "gate"
                    if name in ("mlp.gate_proj", "mlp.up_proj")
                    else "down" if name == "mlp.down_proj" else "output"
                )
            )
            if self.selected_role not in ("all", family):
                continue
            k, n = self.dram_weights[name].shape
            ow = math.ceil(n / (32 * outc))
            im = ttnn.create_sharded_memory_config(
                (32, k // inc),
                ttnn.CoreGrid(x=8, y=inc // 8),
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            if self.preserve_inputs:
                im = self.dram_layouts[name][0]
                if name == "mlp.down_proj":
                    im = self.dram_layouts["mlp.gate_proj"][1]
            grid = ttnn.num_cores_to_corerangeset(
                math.ceil(n / (32 * ow)), self.device.compute_with_storage_grid_size(), True
            )
            om = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (32, ow * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=ow,
                num_workers_per_dram_bank=readers,
                use_multicore_pipeline=True,
            )
            self.dram_layouts[name] = im, om, cfg
            self.candidate_geometry[name] = dict(input=str(im), output=str(om), config=str(cfg))
        return self


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--geometry-role", choices=["all", "input", "gate", "down", "output"], required=True)
    ap.add_argument("--preserve-inputs", action="store_true")
    args, rest = ap.parse_known_args()
    GeometryCandidate.selected_role = args.geometry_role
    GeometryCandidate.preserve_inputs = args.preserve_inputs
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack

    stack.MultichipDecoder = GeometryCandidate
    sys.argv = [sys.argv[0], *rest]
    stack.main()
    output = Path(rest[rest.index("--output") + 1])
    data = json.loads(output.read_text())
    data["geometry_candidate"] = vars(args)
    output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
