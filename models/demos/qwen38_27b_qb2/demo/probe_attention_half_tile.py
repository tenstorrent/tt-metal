# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU simulator screen of accurate partial-tile SDPA against full-tile control."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.attention_half_tile import CONTEXTS, compilation_evidence, sha, verify_overlay


def save(path, report):
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    assert not args.output.exists(), "Preserve attempts"
    simulator = Path(os.environ["TT_METAL_SIMULATOR"])
    assert sha(simulator) == "d01be2a094f9f0f6a00e771defeae2311cb82f051b8af89c2df509e78b8432b7"
    assert os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") == "1"
    assert os.environ.get("TT_METAL_DISABLE_SFPLOADMACRO") == "1"
    assert os.environ.get("TT_METAL_INSPECTOR_RPC") == "0"
    for key, value in os.environ.items():
        if (
            ("TT" in key or "SIM" in key)
            and ("SKIP" in key or "DISABLE" in key)
            and key != "TT_METAL_DISABLE_SFPLOADMACRO"
        ):
            assert value in ("", "0", "false", "False"), f"Refusing disabled checks: {key}"
    manifest = json.loads(args.manifest.read_text())
    verify_overlay(manifest, os.environ["TT_METAL_KERNEL_PATH"], Path.cwd())
    report = dict(
        state="importing",
        variant=manifest["variant"],
        overlay=manifest,
        cases=[],
        cleanup_completed=False,
        physical_devices_accessed=False,
        model_accuracy_qualified=False,
        promoted_to_model=False,
        simulator_sha256=sha(simulator),
        probe_sha256=sha(__file__),
        compiler_fallback="Native TT_METAL_DISABLE_SFPLOADMACRO=1; production instruction parity unqualified",
        scope="BFP8 KV, BF16 Q/output, HiFi4 FP32 accumulation, accurate exp; synthetic CPU simulator",
        hardware_timing_claim=False,
    )
    save(args.output, report)
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.tests.attention_tuning import accuracy, reference

    def digest(tensor):
        return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    torch.set_num_threads(1)
    assert ttnn.GetNumAvailableDevices() == 1, "Expected one virtual device"
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]

        def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def download(value):
            parts = ttnn.get_device_tensors(value)
            assert len(parts) == 1
            return ttnn.to_torch(parts[0]).float()

        for length in CONTEXTS:
            rng = torch.Generator().manual_seed(20261009 + length)
            pages = ((length + 127 + 511) // 512 * 512) // 32
            table = torch.randperm(pages, generator=rng, dtype=torch.int32).reshape(1, pages)
            query = torch.randn((1, 1, 6, 256), generator=rng).bfloat16()
            key_host = torch.randn((pages, 1, 32, 256), generator=rng).bfloat16()
            value_host = torch.randn((pages, 1, 32, 256), generator=rng).bfloat16()
            key = upload(key_host, ttnn.bfloat8_b)
            key_quantized = download(key)
            page_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for active in (length, length - 37):
                values = value_host.clone()
                for page in range(active // 32, pages):
                    values[int(table[0, page]), 0, max(0, active - page * 32) :, :] = 32
                value = upload(values, ttnn.bfloat8_b)
                value_quantized = download(value)
                expected = reference(query, key_quantized, value_quantized, table, [active - 1])
                operand_hashes = dict(
                    query=digest(query), key=digest(key_quantized), value=digest(value_quantized), pages=digest(table)
                )
                del values, value_quantized
                positions = upload(torch.tensor([active - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                for heads in (32, 6):
                    host_q = (
                        torch.cat([query, torch.zeros((1, 1, 26, 256), dtype=query.dtype)], dim=2)
                        if heads == 32
                        else query
                    )
                    q = upload(host_q, ttnn.bfloat16)
                    row = dict(
                        context=length,
                        active_tokens=active,
                        query_heads=heads,
                        operand_sha256=operand_hashes,
                        reference_sha256=digest(expected),
                        state="executing",
                    )
                    report["cases"].append(row)
                    save(args.output, report)
                    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q,
                        key,
                        value,
                        cur_pos_tensor=positions,
                        page_table_tensor=page_table,
                        scale=256**-0.5,
                        program_config=ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=[grid.x, grid.y],
                            q_chunk_size=32,
                            k_chunk_size=256,
                            exp_approx_mode=False,
                            max_cores_per_head_batch=16,
                        ),
                        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            fp32_dest_acc_en=True,
                            math_approx_mode=False,
                            packer_l1_acc=True,
                        ),
                    )
                    actual = download(output)[:, :, :6, :]
                    check = accuracy(actual, expected)
                    row.update(state="completed", accuracy=check, output_sha256=digest(actual))
                    report["compilation_evidence"] = compilation_evidence(Path(os.environ["TT_METAL_CACHE"]), manifest)
                    save(args.output, report)
                    print(
                        json.dumps(
                            dict(
                                variant=manifest["variant"], context=length, active=active, heads=heads, accuracy=check
                            )
                        ),
                        flush=True,
                    )
                    # A failed partial-tile case is diagnostic evidence. Preserve
                    # it and continue; it must never qualify the candidate.
                    if heads == 32:
                        assert check["passed"], "Full-tile numerical control failed"
                    ttnn.deallocate(output)
                    ttnn.deallocate(q)
                ttnn.deallocate(value)
                ttnn.deallocate(positions)
            ttnn.deallocate(key)
            ttnn.deallocate(page_table)
            del key_host, value_host, key_quantized
        verify_overlay(manifest, os.environ["TT_METAL_KERNEL_PATH"], Path.cwd())
        report["state"] = "completed"
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                report["cleanup_completed"] = True
        finally:
            save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
