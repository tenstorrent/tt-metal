# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized hardware component comparison against original recurrence.

Requires bit equality of FP32 output and live FP32 state on every rank for
both BF16-on-read and unrounded-state branches. No device is opened at import.
"""

import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

import ttnn as t
from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_exact_recurrence_probe import install
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import LinearTP4Mixin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must include changing-input recurrence")
    torch.manual_seed(31)
    t.set_fabric_config(t.FabricConfig.FABRIC_1D)
    mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=0)
    records = []
    try:
        config = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        dec = SimpleNamespace(kind="linear_attention", mesh_device=mesh, compute=config, precision_policy={})
        dec.round_widen_chain = [
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
        ]
        dec._recurrent_sum = MethodType(LinearTP4Mixin._recurrent_sum, dec)
        dec._recurrent_step = MethodType(LinearTP4Mixin._recurrent_step, dec)
        original = dec._recurrent_step
        install(dec)

        def upload(value, dtype=t.float32):
            return t.from_torch(
                value,
                device=mesh,
                dtype=dtype,
                layout=t.TILE_LAYOUT,
                memory_config=t.DRAM_MEMORY_CONFIG,
                mesh_mapper=t.ReplicateTensorToMesh(mesh),
            )

        def host(value):
            return [t.to_torch(rank) for rank in t.get_device_tensors(value)]

        for rounded in (False, True):
            state = upload(torch.randn(1, 8, 128, 128) * 0.125)
            candidate_state = upload(host(state)[0].clone())
            for step in range(args.steps):
                # BF16-widened q/k match the model's L2-normalization boundary.
                q = upload((torch.randn(1, 1, 4, 128) * 0.1).bfloat16().float())
                k = upload((torch.randn(1, 1, 4, 128) * 0.1).bfloat16().float())
                v = upload(torch.randn(1, 1, 8, 128), t.bfloat16)
                beta = upload(torch.rand(1, 1, 8).bfloat16().float())
                decay = upload(torch.rand(1, 1, 8) * 0.2 + 0.79)
                reference, state = original(
                    q, k, v, None, beta, state, decay_value=decay, round_state=rounded, output_state=state
                )
                candidate, candidate_state = dec._recurrent_step(
                    q,
                    k,
                    v,
                    None,
                    beta,
                    candidate_state,
                    decay_value=decay,
                    round_state=rounded,
                    output_state=candidate_state,
                )
                for name, actual, expected in (("output", candidate, reference), ("state", candidate_state, state)):
                    for rank, (a, b) in enumerate(zip(host(actual), host(expected))):
                        records.append(
                            dict(
                                rounded=rounded,
                                step=step,
                                tensor=name,
                                rank=rank,
                                bit_equal=torch.equal(a.view(torch.int32), b.view(torch.int32)),
                                max_abs=float((a - b).abs().max()),
                            )
                        )
        result = dict(
            passed=all(r["bit_equal"] for r in records),
            records=records,
            scope="component only; real-weight layer34/PCC/trace/watcher remain required",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"passed": result["passed"], "comparisons": len(records)}))
        if not result["passed"]:
            raise AssertionError("exact recurrence component differs from original FP32 sequence")
    finally:
        t.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
