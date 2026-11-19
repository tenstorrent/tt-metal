# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-precision DRAM-sharded realization of the selected packed projection graph."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import validate_fused
from models.autoports.ornith_ai_ornith_1_5_9b.tt.fused_decoder import FusedDecoder


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--readers", type=int, default=1)
    ap.add_argument("--cores", type=int, default=32)
    ap.add_argument("--block", type=int, default=0)
    args, rest = ap.parse_known_args()

    class Candidate(FusedDecoder):
        use_dram_decode = False

        @classmethod
        def from_state_dict(cls, sd, **kwargs):
            self = super().from_state_dict(sd, **kwargs)
            banks = self.device.dram_grid_size().x
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            self.dram_weights = {}
            self.dram_layouts = {}
            for name in [
                "packed_input",
                "gate_up",
                "mlp.down_proj",
                "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj",
            ]:
                w = self.weights[name + ".weight"]
                k, n = w.shape[-2], w.shape[-1]
                mc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.dram_weights[name] = ttnn.from_torch(
                    ttnn.to_torch(w), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mc
                )

                def layout(width):
                    sw = math.ceil(width / (32 * args.cores))
                    nc = math.ceil(width / (32 * sw))
                    cg = ttnn.num_cores_to_corerangeset(nc, self.device.compute_with_storage_grid_size(), True)
                    return (
                        ttnn.MemoryConfig(
                            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                            ttnn.BufferType.L1,
                            ttnn.ShardSpec(cg, (32, sw * 32), ttnn.ShardOrientation.ROW_MAJOR),
                        ),
                        sw,
                    )

                im, iw = layout(k)
                om, ow = layout(n)
                self.dram_layouts[name] = im, om, iw, ow
            return self

        def linear(self, x, name):
            if x.shape[2] != 1 or name not in self.dram_weights:
                return super().linear(x, name)
            im, om, iw, ow = self.dram_layouts[name]
            h = ttnn.to_memory_config(x, im)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=args.block or iw,
                per_core_M=1,
                per_core_N=ow,
                num_workers_per_dram_bank=args.readers,
                use_multicore_pipeline=args.readers > 1,
            )
            out = ttnn.linear(
                h, self.dram_weights[name], program_config=cfg, compute_kernel_config=self.compute, memory_config=om
            )
            return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

    validate_fused.FusedDecoder = Candidate
    sys.argv = [sys.argv[0], *rest]
    validate_fused.main()
    output = Path(rest[rest.index("--output") + 1])
    j = json.loads(output.read_text())
    j.update(dram_candidate=vars(args), candidate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(j, indent=2) + "\n")


if __name__ == "__main__":
    main()
