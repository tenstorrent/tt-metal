# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct FP32 router-score GEMM; normalization, centering and native gate unchanged."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import GeneralizedRouter, OptimizedDecoder


def describe(value):
    return dict(
        shape=list(value.shape),
        padded_shape=list(value.padded_shape),
        dtype=str(value.dtype),
        layout=str(value.layout),
        memory=str(value.memory_config()),
    )


class DirectRouterProbe(GeneralizedRouter):
    def __init__(self, original, mesh, args, runtime):
        if type(original) is not GeneralizedRouter:
            raise ValueError("Direct router probe requires the configured GeneralizedRouter")
        self.__dict__.update(original.__dict__)
        self.probe_runtime = runtime
        self.probe_backend = args.direct_router_backend
        self.probe_input_memory = args.direct_router_input_memory
        self.probe_memory = (
            ttnn.L1_MEMORY_CONFIG if args.direct_router_output_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
        )
        weight = self.original.source.source.proj_weight
        dtype = getattr(ttnn, args.direct_router_weight_dtype)
        self.probe_weight = weight if weight.dtype == dtype else ttnn.typecast(weight, dtype)
        if tuple(weight.shape) != (1, 1, 2816, 128) or weight.layout != ttnn.TILE_LAYOUT or weight.is_sharded():
            raise ValueError("Direct router expects the original tiled interleaved [1,1,2816,128] weight")
        gx, gy = args.direct_router_grid
        available = mesh.compute_with_storage_grid_size()
        if not (0 < gx <= available.x and 0 < gy <= available.y) or 4 % (gx * gy):
            raise ValueError("Direct router grid must fit the device and divide four output tiles")
        block = args.direct_router_block
        if block <= 0 or 88 % block:
            raise ValueError("Direct router K block must divide 88 tiles")
        per_n = 4 // (gx * gy)
        self.probe_program = None
        if args.direct_router_program == "explicit":
            self.probe_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=per_n,
                out_block_h=1,
                out_block_w=per_n,
                per_core_M=1,
                per_core_N=per_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        self.probe_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, args.direct_router_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=args.direct_router_full_sync,
        )
        runtime.update(
            backend=self.probe_backend,
            prefill="unchanged original BroadcastRouter",
            center_logits=self.center_logits,
            native_gate="unchanged",
            original_weight=describe(weight),
            actual_weight=describe(self.probe_weight),
            weight_values="original BF16 values; optional FP32 storage cast only",
            input_dtype="float32",
            output_dtype="float32",
            input_memory=args.direct_router_input_memory,
            output_memory=str(self.probe_memory),
            program=str(self.probe_program),
            grid=[gx, gy] if self.probe_program is not None else None,
            k_block=block if self.probe_program is not None else None,
            per_core_n=per_n if self.probe_program is not None else None,
            fidelity=str(self.probe_compute.math_fidelity),
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=args.direct_router_full_sync,
            preserved_scale=True,
            preserved_per_expert_scale=True,
        )

    def project_scores(self, scaled):
        if scaled.dtype != ttnn.float32:
            raise ValueError("Direct router control requires the existing FP32 scaled activation")
        if self.probe_backend == "broadcast":
            products = ttnn.mul(self.original.source.projection_rows, scaled)
            return ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1)
        operand = scaled
        if self.probe_input_memory != "inherit" or operand.is_sharded():
            memory = ttnn.DRAM_MEMORY_CONFIG if self.probe_input_memory == "dram" else ttnn.L1_MEMORY_CONFIG
            operand = ttnn.to_memory_config(operand, memory)
        scores = ttnn.linear(
            operand,
            self.probe_weight,
            dtype=ttnn.float32,
            program_config=self.probe_program,
            compute_kernel_config=self.probe_compute,
            memory_config=self.probe_memory,
        )
        if "observed_decode" not in self.probe_runtime:
            self.probe_runtime["observed_decode"] = dict(
                scaled_input=describe(scaled), matmul_input=describe(operand), scores=describe(scores)
            )
        return scores

    def __call__(self, x, normalized=None):
        if x.shape[-2] != 1:
            return self.original(x, normalized=normalized)

        source = self.original.source
        router = source.source
        if normalized is None:
            normalized = self.original.normalize(x, source.epsilon)
        scaled = ttnn.mul(
            normalized,
            source.scale,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, router.scalar_root_size)],
        )
        scores = self.project_scores(scaled)
        if tuple(scores.shape) != (1, 1, 1, 128):
            raise ValueError("Generalized router supports one decode token with 128 logits")
        if self.center_logits:
            if scores.dtype != ttnn.float32:
                raise ValueError("Score-centering control requires FP32 scores before subtraction")
            scores = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
        rounded = ttnn.typecast(scores, ttnn.bfloat16)

        padded = ttnn.pad(rounded, [(0, 0), (0, 0), (0, 0), (0, 128)], float("-inf"))
        face = ttnn.to_memory_config(ttnn.reshape(padded, (1, 16, 16)), self.memory)
        values, indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            face,
            bias_tensor=self.bias,
            input_indices_tensor=self.indices,
            output_tensor=self.output,
            output_indices_tensor=self.output_indices,
            eps=1e-20,
            scaling_factor=1.0,
            enable_sigmoid=False,
            topk=8,
            output_softmax=True,
            grouped=False,
        )
        values = ttnn.to_memory_config(values, ttnn.L1_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.L1_MEMORY_CONFIG)
        values = ttnn.view(values[:, 0, :8], (1, 1, 1, 8))
        indices = ttnn.view(indices[:, 0, :8], (1, 1, 1, 8))
        routing = ttnn.scatter(ttnn.zeros_like(rounded), dim=-1, index=indices, src=values)
        return ttnn.mul(routing, router.per_expert_scale)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--direct-router-backend", choices=("linear", "broadcast"), default="linear")
    parser.add_argument("--direct-router-grid", type=int, nargs=2, default=(4, 1))
    parser.add_argument("--direct-router-block", type=int, default=11)
    parser.add_argument("--direct-router-fidelity", choices=("HiFi4", "HiFi3", "HiFi2", "LoFi"), default="HiFi4")
    parser.add_argument("--direct-router-full-sync", action="store_true")
    parser.add_argument("--direct-router-program", choices=("explicit", "auto"), default="explicit")
    parser.add_argument("--direct-router-input-memory", choices=("inherit", "l1", "dram"), default="inherit")
    parser.add_argument("--direct-router-output-memory", choices=("l1", "dram"), default="l1")
    parser.add_argument("--direct-router-weight-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to mix direct-router evidence with an existing artifact: {output}")
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        decoder.layer.moe.router = DirectRouterProbe(decoder.layer.moe.router, kw["mesh_device"], args, runtime)
        decoder.precision_policy["router_projection"] = runtime
        return decoder

    previous_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)):
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = previous_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["direct_router_probe"] = vars(args)
        report["direct_router_runtime"] = runtime
        report["direct_router_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["runtime_sha256"] = hashlib.sha256(
            Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
        ).hexdigest()
        if failure is not None:
            report["direct_router_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
