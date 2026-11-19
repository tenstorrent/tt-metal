# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 precision comparisons against recorded, teacher-forced HF activations."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, weights, pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, clear, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_phase_candidate import PhaseCandidate, PrefillLoFiCandidate


def policy(name):
    if name == "control":
        return MultichipDecoder
    if name == "phase":
        return PhaseCandidate
    if name == "phase_prefill_input_lofi_full":
        PrefillLoFiCandidate.selected_kind = "full_attention"
        return PrefillLoFiCandidate
    if name == "phase_prefill_input_lofi":
        PrefillLoFiCandidate.selected_kind = None
        return PrefillLoFiCandidate
    if name in ("phase_attention", "phase_mlp", "phase_ccl"):
        selected = {"phase_attention": {"input_layernorm"}, "phase_mlp": {"post_attention_layernorm"}, "phase_ccl": set()}[name]

        class IsolatedPhase(PhaseCandidate):
            def norm(self, x, role):
                y = MultichipDecoder.norm(self, x, role)
                return ttnn.typecast(y, ttnn.bfloat8_b) if x.shape[2] >= 512 and role in selected else y

            def reduce(self, x):
                return PhaseCandidate.reduce(self, x) if name == "phase_ccl" else MultichipDecoder.reduce(self, x)

        return IsolatedPhase
    if name == "kv_bfp4":
        return type("RecordedKV4", (MultichipDecoder,), {"kv_dtype": ttnn.bfloat4_b})
    roles, dtype, fidelity = {
        "input_lofi": (("packed_input",), None, ttnn.MathFidelity.LoFi),
        "output_lofi": (("linear_attn.out_proj", "self_attn.o_proj"), None, ttnn.MathFidelity.LoFi),
        "input_bfp4": (("packed_input",), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        "output_bfp4": (("linear_attn.out_proj", "self_attn.o_proj"), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        "gate_up_bfp4": (("mlp.gate_proj", "mlp.up_proj", "packed_mlp"), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        "down_bfp4": (("mlp.down_proj",), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    }[name]

    class RecordedRolePolicy(MultichipDecoder):
        def weight_dtype(self, role):
            return dtype if role in roles and dtype is not None else super().weight_dtype(role)

        def matmul_compute(self, role):
            if role in roles:
                return ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=False,
                    fp32_dest_acc_en=True, packer_l1_acc=True)
            return super().matmul_compute(role)

    return RecordedRolePolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(991)
    data = torch.load(args.inputs, weights_only=True)
    cases = [case for case in data["cases"] if case["layer"] == args.layer]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
    records, failed = [], []
    observed_calls, patched = {}, []

    def upload(x, integer=False, device=True):
        return ttnn.from_torch(x, dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            device=mesh if device else None, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def check(label, actual, expected, residual):
        ranks = [ttnn.to_torch(t).reshape_as(expected) for t in ttnn.get_device_tensors(actual)]
        output_scores = [pcc(expected, rank) for rank in ranks]
        contribution_scores = [pcc(expected - residual, rank - residual) for rank in ranks]
        agreement = [pcc(ranks[0], rank) for rank in ranks[1:]]
        row = dict(label=label, pcc=min(output_scores), contribution_pcc=min(contribution_scores),
                   rank_agreement=min(agreement), rank_pcc=output_scores)
        if min(output_scores + contribution_scores) < 0.995 or min(agreement) < 0.99999:
            failed.append(row)
        records.append(row)

    try:
        cls = policy(args.policy)
        layer = cls.from_state_dict(weights(args.layer), hf_config=config(), layer_idx=args.layer,
                                   mesh_device=mesh, residual_layout="replicated")
        weight_roles = {id(value): name for name, value in layer.weights.items()}
        weight_roles.update({id(value): name + ".dram_weight" for name, value in layer.dram_weights.items()})

        def observe(module, name):
            original = getattr(module, name)
            def wrapped(*operands, **kwargs):
                result = original(*operands, **kwargs)
                row = dict(op=name, input_shape=list(operands[0].shape), input_dtype=str(operands[0].dtype),
                           input_memory=str(operands[0].memory_config()),
                           compute_config=str(kwargs.get("compute_kernel_config")),
                           requested_output_dtype=str(kwargs.get("dtype")), output_dtype=str(result.dtype))
                if name in ("linear", "minimal_matmul"):
                    row.update(weight_role=weight_roles.get(id(operands[1]), "unknown"),
                               weight_dtype=str(operands[1].dtype), weight_shape=list(operands[1].shape))
                observed_calls[json.dumps(row, sort_keys=True)] = row
                return result
            patched.append((module, name, original))
            setattr(module, name, wrapped)

        observe(ttnn, "linear")
        for name in ("minimal_matmul", "reduce_scatter_minimal_async", "all_gather_async", "all_reduce_async"):
            observe(ttnn.experimental, name)
        capacity = ((max(c["length"] + c["steps"] for c in cases) + 31) // 32) * 32
        state = layer.allocate_state(context=capacity)
        table = upload(torch.randperm(capacity // 32, dtype=torch.int32)[None], True)
        for case in cases:
            label = f"corpus{case['corpus']}_n{case['length']}"
            x = case["prefill_input"]
            tx = upload(x[None])
            prefill = []
            for repeat in range(18):
                clear(state)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                with device_only():
                    out = layer.prefill_forward(tx, state=state, page_table=table)
                ttnn.synchronize_device(mesh)
                elapsed = (time.perf_counter() - begin) * 1000
                if repeat >= 3:
                    prefill.append(elapsed)
            check(label + "_prefill", out, case["prefill_output"], x)
            del out
            snapshots = [(b, ttnn.clone(b)) for b in buffers(state)]

            def restore():
                for dst, src in snapshots:
                    ttnn.copy(src, dst)

            td = upload(case["decode_input"][0][None])
            pos = upload(torch.tensor([case["length"]], dtype=torch.int32), True)

            def forward():
                with device_only():
                    return layer.decode_forward(td, state=state, page_table=table, current_position=pos)

            for _ in range(2):
                tmp = forward()
                del tmp
                restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            from ttnn.tools import trace_allocation_tracker
            trace_allocation_tracker.acknowledge_corruptible(out)
            restore()
            for step, (value, expected) in enumerate(zip(case["decode_input"], case["decode_output"])):
                ttnn.copy_host_to_device_tensor(upload(value[None], device=False), td)
                ttnn.copy_host_to_device_tensor(upload(torch.tensor([case["length"] + step], dtype=torch.int32), True, False), pos)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                check(label + f"_decode{step}", out, expected, value)
            ttnn.copy_host_to_device_tensor(upload(case["decode_input"][0][None], device=False), td)
            ttnn.copy_host_to_device_tensor(upload(torch.tensor([case["length"]], dtype=torch.int32), True, False), pos)
            samples = []
            for _ in range(5):
                restore()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(50):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - begin) * 1000 / 50)
            records.append(dict(label=label + "_timing", prefill_ms=statistics.median(prefill),
                                prefill_samples_ms=prefill, decode_ms=statistics.median(samples), decode_samples_ms=samples))
            ttnn.release_trace(mesh, trace)
            del snapshots, out, td, pos, tx
            print("RECORDED_CASE", label, "failures", len(failed), flush=True)
        result = dict(policy=args.policy, layer=args.layer, devices=[0, 1, 2, 3], mesh=[1, 4], regime="serial",
                      source_sha256=hashlib.sha256((Path(__file__).parents[1] / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                      inputs_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                      input_semantics="recorded HF front-four-layer teacher-forced activations", records=records,
                      actual_weight_dtypes={name: str(value.dtype) for name, value in layer.weights.items()},
                      actual_dram_weight_dtypes={name: str(value.dtype) for name, value in layer.dram_weights.items()},
                      matmul_compute_policy_configs={name: str(layer.matmul_compute(name.removesuffix(".weight")))
                                                     for name in layer.weights},
                      observed_op_arguments=list(observed_calls.values()),
                      timing_note="Prefill includes op-metadata instrumentation; diagnostic only. Final ranking uses uninstrumented probe/stack harnesses.",
                      actual_kv_dtype=str(layer.kv_dtype),
                      failed=failed, passed=not failed, threshold=0.995, traced=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        assert not failed, (args.policy, len(failed), failed[:2])
        print("RECORDED_PRECISION_PASS", args.policy, args.layer)
    finally:
        for module, name, original in reversed(patched):
            setattr(module, name, original)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
