# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Non-serving SDPA geometry sweep from recorded real target-model activations."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import NativePagedAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=4096, choices=(128, 4096))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    report = {
        "scope": "Isolated traced native SDPA, actual recorded layer0/5 inputs; not full-model or serving TSU",
        "length": args.length,
        "rows": [],
        "fixtures": {},
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sorted((root / "tt").glob("*.py"))]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1000000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=262144, layer_indices=(0, 5))
        for layer_number, decoder in enumerate(gen.model.layers):
            index = gen.model.layer_indices[layer_number]
            path = root / f"doc/optimized_decoder/actual_text_layer{index}_4096_128.pt"
            report["fixtures"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            fixture = torch.load(path, weights_only=True)
            caches, host_table = gen.model.allocate_cache(slots=32, context=8192)
            cache = caches[layer_number]
            host_table = torch.nn.functional.pad(host_table[:1], (0, 8192 - host_table.shape[1]))
            table = gen.model.upload(host_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            prefill = gen.model.upload(fixture["prefill"][:, : args.length].unsqueeze(0))
            result = decoder.prefill_forward(
                prefill,
                rope_mats=gen.model.rope_prefill[gen.model.config.layer_types[index]],
                page_table=table,
                kv_cache=cache,
                user_id=0,
            )
            del prefill, result
            value = (
                fixture["decode"][:, :1]
                if args.length == 4096
                else fixture["prefill"][:, args.length : args.length + 1]
            )
            hidden = gen.model.upload(value.unsqueeze(0))
            positions = gen.model.upload(torch.tensor([[args.length]]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            cache_positions = gen.model.upload(torch.tensor([args.length]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            native = decoder.layer.self_attn.decode_sdpa
            captured = {}
            original = NativePagedAttention.__call__

            def capture(instance, q, k, v, **kwargs):
                assert instance is native
                captured.update(q=ttnn.clone(q), k=k, v=v, kwargs=kwargs)
                return original(instance, q, k, v, **kwargs)

            with patch.object(NativePagedAttention, "__call__", capture):
                output = decoder.decode_forward(
                    hidden,
                    rope_mats=gen.model.rope_decode[gen.model.config.layer_types[index]],
                    current_pos=positions,
                    cache_pos=cache_positions,
                    page_table=table,
                    kv_cache=cache,
                )
            del output
            baseline_program = native.program

            def forward():
                return native(captured["q"], captured["k"], captured["v"], **captured["kwargs"])

            reference_output = forward()
            references = [ttnn.to_torch(t).float().clone() for t in ttnn.get_device_tensors(reference_output)]
            del reference_output
            configs = [
                entry
                for cores in (4, 8, 16, 32, 64)
                for chunk in (0, 32, 64, 128)
                for entry in (("baseline", 16, 0), ("candidate", cores, chunk))
            ] + [("baseline_repeat", 16, 0)]
            for label, cores, chunk in configs:
                native.program = (
                    baseline_program
                    if label.startswith("baseline")
                    else ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=(8, 8),
                        q_chunk_size=0,
                        k_chunk_size=chunk,
                        exp_approx_mode=False,
                        max_cores_per_head_batch=cores,
                    )
                )
                try:
                    warmed = forward()
                except RuntimeError as error:
                    if "Statically allocated circular buffers" not in str(error):
                        raise
                    report["rows"].append(
                        {
                            "layer": index,
                            "label": label,
                            "max_cores_per_head_batch": cores,
                            "k_chunk_size": chunk,
                            "blocked_op_contract": str(error).split("backtrace:")[0],
                        }
                    )
                    save()
                    continue
                del warmed
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                try:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    ranks = [ttnn.to_torch(t).float().clone() for t in ttnn.get_device_tensors(output)]
                    row = {
                        "layer": index,
                        "label": label,
                        "max_cores_per_head_batch": cores,
                        "k_chunk_size": chunk,
                        "compute": {
                            "math_fidelity": str(native.compute.math_fidelity),
                            "math_approx_mode": native.compute.math_approx_mode,
                            "fp32_dest_acc_en": native.compute.fp32_dest_acc_en,
                            "packer_l1_acc": native.compute.packer_l1_acc,
                            "dst_full_sync_en": native.compute.dst_full_sync_en,
                        },
                        "query_shape": list(captured["q"].shape),
                        "cache_shape": list(captured["k"].shape),
                        "exact": all(torch.equal(reference, rank) for reference, rank in zip(references, ranks)),
                        "pcc": min(
                            float(torch.corrcoef(torch.stack((reference.flatten(), rank.flatten())))[0, 1])
                            for reference, rank in zip(references, ranks)
                        ),
                        "max_abs": max(
                            float((reference - rank).abs().max()) for reference, rank in zip(references, ranks)
                        ),
                        "queued_trace_us": [],
                    }
                    for repeat in range(5):
                        started = time.perf_counter()
                        for _ in range(50):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        row["queued_trace_us"].append((time.perf_counter() - started) * 1e6 / 50)
                    report["rows"].append(row)
                    save()
                    print("SDPA", json.dumps(row), flush=True)
                finally:
                    ttnn.release_trace(mesh, trace)
                    del output
            native.program = baseline_program
            del hidden, positions, cache_positions, table, captured, caches, cache
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
