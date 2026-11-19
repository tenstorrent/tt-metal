# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Disposable pre-capture census of preparation and native program storage."""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import ccl_inventory, source_identity
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uniform-length", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    result = dict(source_sha256=source_identity(), snapshots=[], scope="reduced [0,1,3], no traces")
    generator = None
    try:
        model = Model(mesh, layer_indices=[0, 1, 3])

        def snapshot(label):
            ttnn.synchronize_device(mesh)
            buffers = ttnn._ttnn.reports.get_buffers(mesh)
            rank = min(b.device_id for b in buffers)
            counts = Counter(
                (str(b.buffer_type), str(b.buffer_layout), int(b.max_size_per_bank))
                for b in buffers
                if b.device_id == rank
            )
            row = dict(
                label=label,
                memory=model.memory_view(),
                programs=mesh.num_program_cache_entries(),
                ccl=ccl_inventory(model),
                buffers=[dict(type=k[0], layout=k[1], per_bank=k[2], count=v) for k, v in sorted(counts.items())],
            )
            result["snapshots"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("MEMORY " + json.dumps({k: v for k, v in row.items() if k not in ("buffers", "ccl")}), flush=True)

        snapshot("loaded")
        physical = torch.randperm(32 * 257, generator=torch.Generator().manual_seed(42))
        table = physical.reshape(32, 257)[:, torch.arange(8192) % 257].to(torch.int32)
        generator = QwenGenerator(model, batch_size=32, max_seq_len=262144, page_table=table)
        snapshot("allocated")
        terminal = model.terminal

        def measured_terminal(hidden, **kwargs):
            selected = hidden.shape[0] == 32 and hidden.shape[1] > 1 and hidden.shape[1] % 32 == 0
            if selected:
                snapshot(f"before_terminal_{hidden.shape[1]}")
            output = terminal(hidden, **kwargs)
            if selected:
                snapshot(f"with_terminal_{hidden.shape[1]}")
            return output

        model.terminal = measured_terminal
        generator.prefill_single.prepare()
        snapshot("single_prepared")
        generator.prefill_uniform.prepare(_diagnostic_lengths=range(1, args.uniform_length + 1))
        snapshot("uniform_prepared")
        # Diagnostic only: no trace exists, and this process never serves requests.
        mesh.clear_program_cache()
        gc.collect()
        snapshot("program_cache_cleared")
        result["passed"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
