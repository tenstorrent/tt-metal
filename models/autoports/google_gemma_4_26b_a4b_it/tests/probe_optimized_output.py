# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare decode output-projection configs with the audited whole-layer harness.

Prefill delegates to the selected decoder unchanged. Decode retains the fused
head concat, FP32 output, and optional BF16 input conversion only when native
SDPA guarantees BF16-valued attention. DRAM weight readback/repacking is setup
only and preserves the selected device weight values before changing layout.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import NativePagedAttention, OptimizedDecoder


def tensor_description(value):
    return {
        "shape": list(value.shape),
        "padded_shape": list(value.padded_shape),
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "memory_config": str(value.memory_config()),
    }


class OutputProjectionProbe:
    def __init__(self, attention, mesh, args, runtime):
        self.original = attention.project
        self.runtime = runtime
        self.args = args
        self.config = attention.source.config
        self.k = self.config.num_attention_heads * self.config.head_dim
        self.width = attention.source.weights.o_proj.shape[-1]
        if args.output_memory == "sharded" and args.output_mode != "dram":
            raise ValueError("Retaining sharded output requires the DRAM-sharded matmul mode")
        self.final_memory = {
            "dram": ttnn.DRAM_MEMORY_CONFIG,
            "l1": ttnn.L1_MEMORY_CONFIG,
            "sharded": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        }[args.output_memory]
        self.input_dtype = getattr(ttnn, args.output_input_dtype)
        if not attention.project_sharded:
            raise ValueError("Output probe requires the fused sharded head-concat path")
        if self.input_dtype == ttnn.bfloat16 and not isinstance(attention.decode_sdpa, NativePagedAttention):
            raise ValueError("Lossless BF16 input conversion requires the native BF16-valued attention backend")
        if self.k % 32 or self.width % 32:
            raise ValueError("Output projection dimensions must be tile aligned")
        if args.output_k_block <= 0 or self.k // 32 % args.output_k_block:
            raise ValueError("Output K block must divide the tiled attention width")

        self.head_memory = ttnn.create_sharded_memory_config(
            shape=(32, self.config.head_dim),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.compute = attention.compute
        if args.output_fidelity:
            self.compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, args.output_fidelity),
                math_approx_mode=self.compute.math_approx_mode,
                fp32_dest_acc_en=True,
                packer_l1_acc=self.compute.packer_l1_acc,
                dst_full_sync_en=self.compute.dst_full_sync_en,
            )
        if not self.compute.fp32_dest_acc_en:
            raise ValueError("Output probe requires FP32 destination accumulation")

        weight = attention.source.weights.o_proj
        runtime["source_weight"] = tensor_description(weight)
        if args.output_weight_dtype:
            weight = ttnn.typecast(weight, getattr(ttnn, args.output_weight_dtype))
        self.weight = weight
        self.program = None
        self.input_memory = None
        self.matmul_memory = self.final_memory
        self.physical_n = self.width
        available = mesh.compute_with_storage_grid_size()

        if args.output_mode == "interleaved":
            gx, gy = args.output_grid
            if not (0 < gx <= available.x and 0 < gy <= available.y):
                raise ValueError("Output compute grid must fit the device")
            per_n = math.ceil(self.width / 32 / (gx * gy))
            subblock = args.output_subblock or next(v for v in (4, 3, 2, 1) if per_n % v == 0)
            if subblock not in (1, 2, 3, 4) or per_n % subblock:
                raise ValueError("FP32 output subblock must be 1..4 tiles and divide per-core N")
            self.input_memory = ttnn.L1_MEMORY_CONFIG
            self.program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=args.output_k_block,
                out_subblock_h=1,
                out_subblock_w=subblock,
                out_block_h=1,
                out_block_w=per_n,
                per_core_M=1,
                per_core_N=per_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            runtime["geometry"] = dict(grid=[gx, gy], per_core_N=per_n, output_subblock=[1, subblock])
        elif args.output_mode == "dram":
            readers = args.output_readers
            if readers not in (1, 2, 3) or (readers > 1 and mesh.arch() != ttnn.Arch.BLACKHOLE):
                raise ValueError("Multiple output DRAM readers require Blackhole; supported counts are 1, 2, 3")
            legal_cores = [
                c
                for c in (8, 6, 4, 3, 2, 1)
                if c <= available.x and self.k // 32 % c == 0 and self.k // 32 // c % args.output_k_block == 0
            ]
            cores = args.output_input_cores or (legal_cores[0] if legal_cores else 0)
            if cores not in legal_cores:
                raise ValueError(f"Input storage cores must divide K and leave a whole K block; legal: {legal_cores}")
            bank_grid = mesh.dram_grid_size()
            banks = bank_grid.x * bank_grid.y
            alignment = 32 * math.lcm(banks * readers, cores)
            self.physical_n = math.ceil(self.width / alignment) * alignment
            if args.output_memory == "sharded" and self.physical_n != self.width:
                raise ValueError(
                    f"Retaining sharded output requires unpadded N: logical {self.width}, padded {self.physical_n}"
                )
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(bank_grid.x - 1, bank_grid.y - 1))}
                    ),
                    (self.k, self.physical_n // banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            # Read the selected device values, including any BFP quantization,
            # before repacking. Padding whole N tiles preserves existing BFP
            # exponent groups. There are no host transfers inside __call__.
            matrix = ttnn.to_torch(weight, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            matrix = torch.nn.functional.pad(matrix, (0, self.physical_n - self.width))
            self.weight = ttnn.from_torch(
                matrix, device=mesh, dtype=weight.dtype, layout=ttnn.TILE_LAYOUT, memory_config=weight_memory
            )
            self.input_memory = ttnn.create_sharded_memory_config(
                (32, self.k // cores),
                ttnn.CoreGrid(x=cores, y=1),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=args.output_k_block,
                per_core_M=1,
                per_core_N=self.physical_n // 32 // cores,
                num_workers_per_dram_bank=readers,
                fused_activation=None,
            )
            self.matmul_memory = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            tile_bytes = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[self.weight.dtype]
            runtime["geometry"] = dict(
                input_storage_cores=cores,
                input_shard_shape=[32, self.k // cores],
                output_shard_shape=[32, self.physical_n // cores],
                weight_banks=banks,
                readers_per_bank=readers,
                weight_shard_shape=[self.k, self.physical_n // banks],
                logical_N=self.width,
                padded_N=self.physical_n,
                in1_triple_buffer_bytes_per_reader=(
                    self.physical_n // 32 // banks // readers * args.output_k_block * 3 * tile_bytes
                ),
                buffer_estimate_scope="Weight buffer only; excludes activation, FP32 output/intermediate and live L1",
                source_weight_values_preserved_before_repack=True,
            )

        runtime.update(
            mode=args.output_mode,
            weight=tensor_description(self.weight),
            program=str(self.program),
            compute={
                field: str(getattr(self.compute, field))
                for field in (
                    "math_fidelity",
                    "math_approx_mode",
                    "fp32_dest_acc_en",
                    "packer_l1_acc",
                    "dst_full_sync_en",
                )
            },
            input_dtype=str(self.input_dtype),
            matmul_output_dtype="float32",
            final_memory=str(self.final_memory),
            prefill="delegate unchanged",
        )

    def __call__(self, attention, decode):
        if not decode:
            return self.original(attention, decode)
        combined = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(attention, self.head_memory), num_heads=self.config.num_attention_heads
        )
        combined = ttnn.reshape(combined, (1, 1, 1, self.k), (1, 1, 32, self.k))
        if combined.dtype != self.input_dtype:
            combined = ttnn.typecast(combined, self.input_dtype)
        if self.input_memory is not None:
            combined = ttnn.to_memory_config(combined, self.input_memory)
        result = ttnn.linear(
            combined,
            self.weight,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute,
            memory_config=self.matmul_memory,
            program_config=self.program,
        )
        if self.args.output_mode == "dram" and self.args.output_memory != "sharded":
            result = ttnn.to_memory_config(result, self.final_memory)
            if self.physical_n != self.width:
                result = result[..., : self.width]
        if "observed_decode" not in self.runtime:
            # Descriptor inspection only; the parity harness still forbids
            # host tensor transfers/allocation during forward and capture.
            self.runtime["observed_decode"] = {
                "attention": tensor_description(attention),
                "matmul_input": tensor_description(combined),
                "output": tensor_description(result),
            }
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-mode", choices=("auto", "interleaved", "dram"), default="interleaved")
    parser.add_argument("--output-grid", nargs=2, type=int, default=(8, 8))
    parser.add_argument("--output-k-block", type=int, default=8)
    parser.add_argument("--output-subblock", type=int, default=0, help="0 chooses the largest legal FP32 N subblock")
    parser.add_argument("--output-readers", type=int, default=1)
    parser.add_argument("--output-input-cores", type=int, default=0, help="0 chooses the largest legal storage grid")
    parser.add_argument("--output-memory", choices=("dram", "l1", "sharded"), default="dram")
    parser.add_argument("--output-input-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output-weight-dtype", choices=("bfloat16", "bfloat8_b", "bfloat4_b"))
    parser.add_argument("--output-fidelity", choices=("HiFi4", "HiFi2", "LoFi"))
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to mix probe evidence with an existing artifact: {output}")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        decoder.layer.self_attn.project = OutputProjectionProbe(
            decoder.layer.self_attn, kw["mesh_device"], args, runtime
        )
        return decoder

    original_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)):
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = original_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["output_projection_probe"] = vars(args)
        report["output_projection_probe_runtime"] = runtime
        report["output_projection_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if failure is not None:
            report["output_projection_probe_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
