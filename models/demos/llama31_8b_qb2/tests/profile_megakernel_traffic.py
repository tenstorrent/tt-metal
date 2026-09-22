# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused real-weight MLP traffic capture, separate from latency/serving runs.

Run under tracy -r --collect-noc-traces. Setup, correctness and warmup are
outside the QB2_TRAFFIC signposts. The selected path replays one captured
program sequence per window, with explicit profiler drains between windows.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from transformers import AutoConfig
import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tests.test_decoder import to_host
from models.demos.llama31_8b_qb2.tests.test_megakernel import compare
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.generator_vllm import LlamaForCausalLM
from models.demos.llama31_8b_qb2.tt.megakernel.mlp import FusedMLP
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path, REVISION
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config
from models.demos.utils.trace_region_sizes import build_trace_device_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "mlp"), required=True)
    parser.add_argument("--gu-workers", type=int, choices=(8, 16), default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TT_METAL_DEVICE_PROFILER_NOC_EVENTS") != "1" or os.environ.get("TT_METAL_WATCHER"):
        parser.error("Use tracy --collect-noc-traces in a separate run without Watcher")
    if args.repeats < 1:
        parser.error("At least one complete traffic window is required")
    from tracy import signpost

    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    ttnn.set_fabric_config(**LlamaForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), l1_small_size=16384, **build_trace_device_params("llama3.1-8b-qb2-decoder")
    )
    trace = None
    try:
        folder = checkpoint_path()
        checkpoint = Checkpoint(folder)
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
        layer = LlamaDecoder.from_state_dict(
            checkpoint.load([name for name in checkpoint.index if name.startswith("model.layers.0.")]),
            hf_config=config,
            layer_idx=0,
            mesh_device=mesh,
            ccl=TT_CCL(mesh),
            precision_policy=load_precision_config(),
        )
        layer.prepare_decode(1)
        body = FusedMLP([layer], gu_workers=args.gu_workers)
        hidden = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"][12345].float()
        value = (hidden * torch.rsqrt(hidden.square().mean() + config.rms_norm_eps)).bfloat16()
        normalized = ttnn.from_torch(
            value.reshape(1, 1, 1, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=layer.decode_inputs["gate_up"],
        )

        def baseline():
            packed = ttnn.to_memory_config(layer._decode_linear(normalized, "gate_up"), ttnn.L1_MEMORY_CONFIG)
            gate, up = packed[:, :, :, :3584], packed[:, :, :, 3584:]
            product = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=gate.memory_config(),
                dtype=ttnn.bfloat16,
            )
            return layer._decode_linear(product, "down")

        expected = to_host(baseline())
        actual = to_host(body(normalized, 0))
        try:
            correctness = compare(actual, expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        except AssertionError:
            torch.save({"actual": actual, "expected": expected}, args.output / "traffic-bringup-failure.pt")
            raise
        selected = baseline if args.mode == "baseline" else lambda: body(normalized, 0)
        selected()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = selected()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh)
        for repeat in range(args.repeats):
            signpost(f"QB2_TRAFFIC_BEGIN_{repeat}", f"mode={args.mode},B=1,layer=0,gu={args.gu_workers}")
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            signpost(f"QB2_TRAFFIC_END_{repeat}")
            ttnn.ReadDeviceProfiler(mesh)
        torch.testing.assert_close(to_host(output), expected, rtol=0, atol=0)
        result = {
            "mode": args.mode,
            "gu_workers": args.gu_workers,
            "repeats": args.repeats,
            "checkpoint_revision": REVISION,
            "precision": layer.precision_policy,
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "correctness": correctness,
            "scope": "Real layer0 weights/embedding token12345; local MLP only; no KV or model-level claim",
            "measurement": "NoC issue-event payloads; reject incomplete coverage; profiler overhead is not model latency",
        }
        (args.output / "traffic-run.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
