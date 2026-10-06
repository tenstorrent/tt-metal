# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Actual-layer native QKV matmul control without lane/decomposition overhead.

Decode consumes the selected packed projection weight and preserves FP32 output,
compute fidelity, decode memory and tied-K duplication. Prefill delegates to the
existing lane projection unchanged. This is a precision experiment: FP32 input
storage does not establish full FP32 product precision on the hardware.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import LanePartitionQKV, OptimizedDecoder


def tensor_description(value):
    return {
        "shape": list(value.shape),
        "padded_shape": list(value.padded_shape),
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "memory_config": str(value.memory_config()),
    }


class DirectQKVProbe:
    def __init__(self, original, mesh, args, runtime):
        if not isinstance(original, LanePartitionQKV) or original.dram or len(original.weights) != 1:
            raise ValueError("Direct QKV control requires packed interleaved LanePartitionQKV")
        self.original = original
        self.source = original.source
        self.weight = original.weights[0]
        self.compute = original.compute
        self.input_dtype = getattr(ttnn, args.direct_qkv_input_dtype)
        self.memory = self.source.decode_memory
        self.runtime = runtime
        prior = original.programs[0]
        if prior is None and args.direct_qkv_grid is None:
            raise ValueError("Direct QKV requires an existing explicit lane program or --direct-qkv-grid")
        if args.direct_qkv_grid is None:
            prior_grid = prior.compute_with_storage_grid_size
            grid = (prior_grid.x, prior_grid.y)
        else:
            grid = tuple(args.direct_qkv_grid)
        block = args.direct_qkv_block or (prior.in0_block_w if prior is not None else 1)
        subblock = args.direct_qkv_subblock or (prior.out_subblock_w if prior is not None else 1)
        available = mesh.compute_with_storage_grid_size()
        if not (0 < grid[0] <= available.x and 0 < grid[1] <= available.y):
            raise ValueError("Direct QKV compute grid must fit the device")
        k, width = self.weight.shape[-2], self.weight.shape[-1]
        if k % 32 or width % 32 or block <= 0 or k // 32 % block:
            raise ValueError("Direct QKV dimensions must be tiled and K block must divide K tiles")
        per_n = math.ceil(width / 32 / (grid[0] * grid[1]))
        if subblock not in (1, 2, 3, 4) or per_n % subblock:
            raise ValueError("Direct FP32 output subblock must be 1..4 tiles and divide per-core N")
        self.program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            in0_block_w=block,
            out_subblock_h=1,
            out_subblock_w=subblock,
            per_core_M=1,
            per_core_N=per_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        runtime.update(
            backend="native_packed_linear",
            prefill="delegate unchanged",
            input_dtype=str(self.input_dtype),
            output_dtype="float32",
            weight=tensor_description(self.weight),
            output_memory=str(self.memory),
            activation_decomposition=False,
            lane_mask=False,
            partial_row_reduce=False,
            tied_k_duplicate=hasattr(self.source, "kv_width"),
            original_lane_terms=original.terms,
            original_program=str(prior),
            program=dict(
                grid=list(grid),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=subblock,
                per_core_M=1,
                per_core_N=per_n,
                fuse_batch=True,
                mcast_in0=True,
            ),
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
        )

    def __call__(self, hidden_states, compute_kernel_config=None, out_memory_config=None):
        if hidden_states.shape[-2] != 1:
            return self.original(hidden_states, compute_kernel_config, out_memory_config)
        operand = hidden_states
        if operand.dtype != self.input_dtype:
            operand = ttnn.typecast(operand, self.input_dtype)
        result = ttnn.linear(
            operand,
            self.weight,
            dtype=ttnn.float32,
            program_config=self.program,
            compute_kernel_config=self.compute,
            memory_config=self.memory,
        )
        if hasattr(self.source, "kv_width"):
            result = ttnn.concat(
                (result, result[..., self.source.width - self.source.kv_width :]),
                dim=-1,
                memory_config=self.memory,
            )
        if "observed_decode" not in self.runtime:
            # Descriptor inspection only; no host transfer or upload in forward.
            self.runtime["observed_decode"] = {
                "hidden_input": tensor_description(hidden_states),
                "matmul_input": tensor_description(operand),
                "output": tensor_description(result),
            }
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--direct-qkv-input-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--direct-qkv-grid", nargs=2, type=int)
    parser.add_argument("--direct-qkv-block", type=int)
    parser.add_argument("--direct-qkv-subblock", type=int)
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to mix direct QKV evidence with an existing artifact: {output}")
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        source = decoder.layer.self_attn.source
        projection = DirectQKVProbe(source.weights.wqkv, kw["mesh_device"], args, runtime)
        source.weights = replace(source.weights, wqkv=projection)
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
        report["direct_qkv_probe"] = vars(args)
        report["direct_qkv_probe_runtime"] = runtime
        report["direct_qkv_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if failure is not None:
            report["direct_qkv_probe_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
