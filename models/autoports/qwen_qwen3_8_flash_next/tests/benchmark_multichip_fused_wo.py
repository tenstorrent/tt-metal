"""Qualified attention WO helper: payload/output precision and block experiments.

Earlier integration probes are preserved verbatim in their artifact directories.
This runner targets the production AttentionOutputProjection ownership contract.
The agmm option reproduces a known native eight-slice invariant failure on TP4;
it is historical diagnostic coverage and is not a qualified implementation.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", choices=["agmm", "minimal", "ag", "row"], default="ag")
    parser.add_argument("--payload", choices=["bf16", "bf8"], default="bf8")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--cores", type=int)
    parser.add_argument("--linear-fabric", action="store_true")
    parser.add_argument("--block", type=int, default=48)
    parser.add_argument("--output", choices=["bf8", "bf16"], default="bf8")
    parser.add_argument("--public-bf16", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.linear_fabric:
        original_fabric = ttnn.set_fabric_config
        ttnn.set_fabric_config = lambda _: original_fabric(ttnn.FabricConfig.FABRIC_1D)
    original_init = target.AttentionOutputProjection.__init__
    original_call = target.AttentionOutputProjection.__call__
    payload_dtype = ttnn.bfloat8_b if args.payload == "bf8" else ttnn.bfloat16
    output_dtype = ttnn.bfloat8_b if args.output == "bf8" else ttnn.bfloat16

    def initialize(self, *a, **kw):
        original_init(self, *a, **kw)
        self.gathered = ttnn.empty((1, 1, 1, 6144), dtype=payload_dtype, layout=ttnn.TILE_LAYOUT, device=self.ops.mesh)
        self.program.in0_block_w = args.block
        if args.cores:
            assert args.cores in (5, 10, 20)
            width, height = min(10, args.cores), math.ceil(args.cores / 10)
            self.program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(width, height),
                in0_block_w=args.block,
                out_subblock_h=1,
                out_subblock_w=20 // args.cores,
                per_core_M=1,
                per_core_N=20 // args.cores,
                fuse_batch=True,
                mcast_in0=True,
                allowed_worker_cores=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))]
                ),
            )

    def project(self, x, *, concatenated=False):
        shape = tuple(x.shape)
        if shape[-2:] != (1, 1536) or math.prod(shape[:-2]) != 1:
            return original_call(self, x, concatenated=concatenated)
        bench.CANDIDATE["applied_calls"] += 1
        x = ttnn.typecast(ttnn.reshape(x, (1, 1, 1, 1536)), payload_dtype)
        if args.family == "row":
            out = self.ops.scatter_reduce(
                self.ops.linear(x, self.row_weight, compute_kernel_config=self.compute, dtype=output_dtype)
            )
        elif args.family == "ag":
            out = ttnn.linear(
                self.ops.gather(x),
                self.column_weight,
                program_config=self.program,
                compute_kernel_config=self.compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=output_dtype,
            )
        elif args.family == "minimal":
            config = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=args.block,
                N_block_size=4,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
            out = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=self.column_weight,
                config=config,
                compute_kernel_config=self.compute,
                multi_device_global_semaphore=self.ops.ccl.get_and_cycle_ag_semaphore_handles(1),
                num_links=2,
                topology=ttnn.Topology.Linear,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=output_dtype,
                force_transpose=True,
                num_workers_per_link=4,
                num_buffers_per_channel=8,
                persistent_output_buffer=self.gathered,
            )[0]
        else:
            _, out = ttnn.experimental.all_gather_matmul_async(
                x,
                self.column_weight,
                persistent_output_buffer=self.gathered,
                dim=3,
                multi_device_global_semaphore=self.ops.ccl.get_and_cycle_ag_semaphore_handles(1),
                all_gather_core_grid_offset=(0, 4),
                barrier_semaphore=self.ops.ccl.get_and_cycle_barrier_semaphore_handle(1),
                num_links=2,
                topology=ttnn.Topology.Ring,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_mm=ttnn.L1_MEMORY_CONFIG,
                program_config=self.program,
                compute_kernel_config=self.compute,
                dtype=output_dtype,
            )
        if (args.public_bf16 or concatenated) and out.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, ttnn.bfloat16)
        return ttnn.reshape(out, (*shape[:-1], 640))

    target.AttentionOutputProjection.__init__ = initialize
    target.AttentionOutputProjection.__call__ = project
    bench.CANDIDATE = dict(**vars(args), applied_calls=0)
    tag = sys.argv[sys.argv.index("--tag") + 1]
    dest = Path(__file__).parents[1] / "doc/multichip_decoder" / tag
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "candidate_runner.py.txt").write_bytes(Path(__file__).read_bytes())
    try:
        if args.validate:
            from models.autoports.qwen_qwen3_8_flash_next.tests.validate_multichip_decoder import main as validate

            validate()
        else:
            bench.main()
        assert bench.CANDIDATE["applied_calls"] > 0, "Fused candidate must execute"
    finally:
        (dest / "candidate.json").write_text(json.dumps(bench.CANDIDATE, indent=2))


if __name__ == "__main__":
    main()
