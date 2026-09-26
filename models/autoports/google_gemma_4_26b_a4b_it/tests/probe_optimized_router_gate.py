# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-only BF16 router control and generalized gate composite candidate.

The generalized gate validator requires BF16 logits/bias/output and UInt16
indices (generalized_moe_gate_device_operation.cpp:65-69). FP32 logits cannot
reach this composite through its current API. Both modes preserve the original
BroadcastRouter score computation; prefill calls the original router unchanged.
Optional FP32 row-maximum subtraction precedes BF16 conversion to test whether
score rounding near top-k boundaries explains the composite's output changes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import BroadcastRouter
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


class RouterGateProbe:
    def __init__(self, original, mesh, generalized, runtime, center_logits=False):
        if type(original) is not BroadcastRouter:
            raise ValueError("Router gate probe requires the unmodified BroadcastRouter backend")
        if original.source.source.num_experts != 128 or original.source.source.top_k != 8:
            raise ValueError("Router gate probe requires 128 experts and top-8 routing")
        self.original = original
        self.generalized = generalized
        self.runtime = runtime
        self.center_logits = center_logits
        runtime["score_transform"] = {
            "center_logits": center_logits,
            "order": "FP32 scores -> FP32 subtract row maximum -> BF16" if center_logits else "FP32 scores -> BF16",
            "scope": "decode only; extra maximum/subtract operations included in whole-layer timing",
        }
        if not generalized:
            return

        core = ttnn.CoreCoord(0, 0)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
        shard = ttnn.ShardSpec(grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
        self.memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)

        def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh, memory_config=self.memory)

        # Match test_generalized_moe_gate.py: logits stay untransposed, whereas
        # bias and global expert ids are transposed within the 16x16 face.
        self.bias = upload(torch.zeros((1, 16, 16), dtype=torch.bfloat16).transpose(-2, -1), ttnn.bfloat16)
        self.indices = upload(
            torch.arange(256, dtype=torch.int32).reshape(1, 16, 16).transpose(-2, -1).to(torch.uint16), ttnn.uint16
        )
        # Match TTMoEGate's row-major output buffers so UInt16 ids need only
        # metadata views after slicing, without a device reshape or dtype cast.
        self.output = upload(torch.zeros((1, 32, 32), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.output_indices = upload(torch.zeros((1, 32, 32), dtype=torch.uint16), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        runtime["generalized_gate"] = {
            "logical_experts": 128,
            "padded_experts": 256,
            "padding_value": "-inf",
            "input_logical_shape": [1, 16, 16],
            "input_layout": "TILE",
            "input_dtype": "bfloat16",
            "bias_and_indices_transposed": True,
            "output_layout": "ROW_MAJOR",
            "output_shape": [1, 32, 32],
            "output_indices_dtype": "uint16",
            "memory_config": str(self.memory),
            "eps": 1e-20,
            "scaling_factor": 1.0,
            "enable_sigmoid": False,
            "topk": 8,
            "output_softmax": True,
            "grouped": False,
        }

    def __getattr__(self, name):
        return getattr(self.original, name)

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
        products = ttnn.mul(source.projection_rows, scaled)
        scores = ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1)
        if tuple(scores.shape) != (1, 1, 1, 128):
            raise ValueError("Router gate probe supports one decode token with 128 logits")
        if self.center_logits:
            if scores.dtype != ttnn.float32:
                raise ValueError("Score-centering control requires FP32 scores before subtraction")
            scores = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
        rounded = ttnn.typecast(scores, ttnn.bfloat16)
        self.runtime.setdefault("decode_logits", {"shape": list(scores.shape), "dtype": str(scores.dtype)})

        if self.generalized:
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
        else:
            selected, indices = ttnn.topk(ttnn.typecast(rounded, ttnn.float32), k=router.top_k, dim=-1)
            values = ttnn.typecast(ttnn.softmax(selected, dim=-1), ttnn.bfloat16)
        routing = ttnn.scatter(ttnn.zeros_like(rounded), dim=-1, index=indices, src=values)
        return ttnn.mul(routing, router.per_expert_scale)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--round-logits-bf16", action="store_true")
    modes.add_argument("--generalized-gate", action="store_true")
    parser.add_argument("--center-logits", action="store_true")
    args, rest = parser.parse_known_args()
    if args.center_logits and not (args.round_logits_bf16 or args.generalized_gate):
        parser.error("--center-logits requires --round-logits-bf16 or --generalized-gate")
    output = Path(rest[rest.index("--output") + 1])
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        if args.round_logits_bf16 or args.generalized_gate:
            decoder.layer.moe.router = RouterGateProbe(
                decoder.layer.moe.router, kw["mesh_device"], args.generalized_gate, runtime, args.center_logits
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
        report["router_gate_probe"] = vars(args)
        report["router_gate_probe_runtime"] = runtime
        report["router_gate_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if failure is not None:
            report["router_gate_probe_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
