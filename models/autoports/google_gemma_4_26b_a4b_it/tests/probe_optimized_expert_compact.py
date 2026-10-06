# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure indexed expert projections and fused weighted reduction on real inputs.

The native gate's existing device output supplies the IDs. Gather reads the
already scaled routing tensor, preserving its BF16 rounding and learned scale.
Prefill and all weights/configs remain those of the selected production policy.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    GeneralizedRouter,
    OptimizedDecoder,
    OptimizedExperts,
)


def describe(value):
    return dict(
        shape=list(value.shape),
        padded_shape=list(value.padded_shape),
        dtype=str(value.dtype),
        layout=str(value.layout),
        memory=str(value.memory_config()),
    )


class ExpertTopologyProbe(OptimizedExperts):
    def __init__(self, original, router, mesh, args, runtime):
        self.__dict__.update(original.__dict__)
        self.probe_original = original
        self.probe_router = router
        self.probe_args = args
        self.probe_runtime = runtime
        self.probe_indexed = args.expert_backend == "indexed"
        if self.probe_indexed and type(router) is not GeneralizedRouter:
            raise ValueError("Indexed expert probe requires the production GeneralizedRouter")
        if self.config.top_k != 8 or self.config.num_experts != 128:
            raise ValueError("Expert topology probe requires top-8 routing over 128 experts")
        if args.expert_merge == "weighted" and not hasattr(
            ttnn.experimental.deepseek_prefill, "attn_res_weighted_reduce_nc"
        ):
            raise ValueError("This build does not expose attn_res_weighted_reduce_nc")
        self.probe_weighted_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.probe_mix_program = self.mix_program
        if self.probe_indexed:
            # Eight compact experts occupy one K tile instead of four. Keep
            # the original output geometry and FP32 accumulation policy.
            cfg = self.mix_program
            self.probe_mix_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
                in0_block_w=1,
                out_subblock_h=cfg.out_subblock_h,
                out_subblock_w=cfg.out_subblock_w,
                out_block_h=cfg.out_block_h,
                out_block_w=cfg.out_block_w,
                per_core_M=cfg.per_core_M,
                per_core_N=cfg.per_core_N,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        runtime.update(
            backend=args.expert_backend,
            merge=args.expert_merge,
            packed=not self.expert_split,
            prefill="unchanged production path",
            gate_weight=describe(self.gate_up),
            down_weight=describe(self.down),
            gate_program=str(self.separate_program if self.expert_split else self.gate_config),
            down_program=str(self.down_config),
            mix_program=str(self.probe_mix_program),
            expert_compute=str(self.decode_compute),
            mix_compute=str(self.mix_compute),
            weighted_compute=dict(fidelity="HiFi4", fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=False),
            activation_dtype=str(self.decode_activation_dtype),
            fused_gelu=self.expert_split or self.decode_gelu_activations is not None,
            indices="GeneralizedRouter output_indices on device; native top-8 order",
            routing_weights="gather after original scatter and learned per_expert_scale multiply",
            nnz=None if self.probe_indexed else 8,
            expert_slots=8 if self.probe_indexed else 128,
            host_work_in_forward=False,
        )

    def _chunk(self, x, routing, decode):
        if not decode or (not self.probe_indexed and self.probe_args.expert_merge == "matmul"):
            return self.probe_original._chunk(x, routing, decode)
        if self.decode_activation_dtype is not None:
            x = ttnn.typecast(x, self.decode_activation_dtype)
        sparsity = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
        slots = self.config.num_experts
        mix_weights = routing
        common = dict(
            sparsity=sparsity,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.decode_compute,
        )
        indices = None
        if self.probe_indexed:
            ids = ttnn.to_memory_config(self.probe_router.output_indices, ttnn.L1_MEMORY_CONFIG)
            indices = ttnn.view(ids[:, 0, :8], (1, 1, 1, 8))
            if indices.dtype != ttnn.uint16 or indices.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError("Indexed sparse matmul requires device UINT16 row-major IDs")
            common["indices"] = indices
            slots = self.config.top_k
            selected = ttnn.gather(sparsity, dim=-1, index=indices, memory_config=ttnn.L1_MEMORY_CONFIG)
            mix_weights = ttnn.to_layout(selected, ttnn.TILE_LAYOUT)
        else:
            common["nnz"] = self.config.top_k
        if self.expert_split:
            gate = ttnn.sparse_matmul(x, self.gate, program_config=self.separate_program, **common)
            up = ttnn.sparse_matmul(x, self.up, program_config=self.separate_program, **common)
            gate = ttnn.reshape(gate, (1, slots, 1, self.width))
            up = ttnn.reshape(up, (1, slots, 1, self.width))
            hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)])
        else:
            gu = ttnn.sparse_matmul(x, self.gate_up, program_config=self.gate_config, **common)
            gu = ttnn.reshape(gu, (1, slots, 1, 2 * self.width))
            gate, up = gu[..., : self.width], gu[..., self.width :]
            if self.decode_gelu_activations is not None:
                hidden = ttnn.mul(gate, up, input_tensor_a_activations=self.decode_gelu_activations)
            else:
                hidden = ttnn.mul(ttnn.gelu(gate, variant=ttnn.GeluVariant.Accurate), up)
        down = ttnn.sparse_matmul(hidden, self.down, program_config=self.down_config, is_input_a_sparse=True, **common)
        down = ttnn.reshape(down, (1, slots, 1, self.config.hidden_size))
        if self.probe_args.expert_merge == "weighted":
            weight_columns = ttnn.permute(mix_weights, (0, 3, 2, 1))
            result = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
                down,
                weight_columns,
                dim=1,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.probe_weighted_compute,
            )
            result = ttnn.to_memory_config(result, self.mix_memory)
        else:
            result = ttnn.matmul(
                mix_weights,
                ttnn.permute(down, (0, 2, 1, 3)),
                dtype=ttnn.bfloat16,
                memory_config=self.mix_memory,
                program_config=self.probe_mix_program,
                compute_kernel_config=self.mix_compute,
            )
        if "observed_decode" not in self.probe_runtime:
            self.probe_runtime["observed_decode"] = dict(
                input=describe(x),
                indices=describe(indices) if indices is not None else None,
                mix_weights=describe(mix_weights),
                hidden=describe(hidden),
                down=describe(down),
                output=describe(result),
            )
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--expert-backend", choices=("indexed", "expanded"), default="indexed")
    parser.add_argument("--expert-merge", choices=("matmul", "weighted"), default="matmul")
    parser.add_argument("--expert-layout", choices=("inherit", "packed", "separate"), default="inherit")
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to mix expert topology evidence with an existing artifact: {output}")
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        if args.expert_layout != "inherit":
            kw["expert_split"] = args.expert_layout == "separate"
        decoder = factory(cls, *a, **kw)
        decoder.layer.moe.experts = ExpertTopologyProbe(
            decoder.layer.moe.experts, decoder.layer.moe.router, kw["mesh_device"], args, runtime
        )
        decoder.precision_policy["expert_topology_probe"] = runtime
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
        report["expert_topology_probe"] = vars(args)
        report["expert_topology_runtime"] = runtime
        report["expert_topology_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["runtime_sha256"] = hashlib.sha256(
            Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
        ).hexdigest()
        if failure is not None:
            report["expert_topology_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
