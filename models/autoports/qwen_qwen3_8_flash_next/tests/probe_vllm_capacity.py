# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic capacity-shaped adapter probe; does not change production admission.

Run under timeout, serialized with all device jobs. This tests an explicit
pool plan before the conservative production planner admits that plan.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
from models.common.sampling.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=8195)
    args = parser.parse_args()
    from vllm.transformers_utils.config import get_config

    checkpoint = os.environ.get(
        "QWEN_CHECKPOINT",
        "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
    )
    report = dict(
        passed=False,
        scope="diagnostic_explicit_pool_not_production_admission",
        context=args.context,
        batch=args.batch,
        pool_blocks=args.blocks,
        reduced_layers=os.environ.get("QWEN_VLLM_LAYER_INDICES"),
        tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        snapshots={},
    )
    mesh = adapter = None
    start = time.monotonic()

    def snapshot(name):
        ttnn.synchronize_device(mesh)
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        report["snapshots"][name] = dict(
            seconds=time.monotonic() - start,
            allocated_bytes_per_rank=view.total_bytes_allocated_per_bank * view.num_banks,
            free_bytes_per_rank=view.total_bytes_free_per_bank * view.num_banks,
            largest_contiguous_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank,
            program_cache_entries=mesh.num_program_cache_entries(),
            counters=dict(adapter.generator.counters) if adapter else None,
        )

    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        snapshot("empty")
        config = get_config(checkpoint, trust_remote_code=False)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(config, mesh, args.batch, args.context)
        snapshot("loaded")
        specs = [((args.blocks, 1, 32, 256), torch.bfloat16, i) for i in range(12)]
        cache = adapter.allocate_kv_cache_per_layer(specs)
        snapshot("pool_allocated")
        adapter.warmup_model_prefill(kv_cache=cache)
        snapshot("prepared")
        columns = (args.context + 31) // 32
        table = torch.zeros(args.batch, columns, dtype=torch.int32)
        for row in range(args.batch):
            table[row, :4] = torch.arange(1 + row * 4, 5 + row * 4)
        sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        prompt = torch.tensor(([785, 13458, 13] * 21), dtype=torch.int64).repeat(args.batch, 1)
        adapter.prefill_forward(
            tokens=prompt,
            prompt_lens=[63] * args.batch,
            start_pos=[0] * args.batch,
            page_table=table,
            kv_cache=cache,
            sampling_params=sampling,
        )
        snapshot("all_prefill_buckets")
        for rows in (1, args.batch, 1, args.batch):
            output = adapter.decode_forward(
                tokens=torch.full((rows, 1), 785, dtype=torch.int64),
                start_pos=torch.tensor([adapter.generator.caches[i].position for i in range(rows)]),
                page_table=table[:rows],
                kv_cache=cache,
                sampling_params=sampling,
                reload_inputs=True,
                reload_sampling_params=True,
                reset_sampling_state=True,
                read_from_device=False,
            )
            host, events = adapter.read_decode_output(output, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            tokens = adapter.process_decode_output_host(host)
            if not bool(((tokens >= 0) & (tokens < adapter.generator.model.config.vocab_size)).all()):
                raise AssertionError("Sampled token outside vocabulary")
        snapshot("replayed")
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["seconds"] = time.monotonic() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
