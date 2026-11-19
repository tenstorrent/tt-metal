# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One advertised-context acceptance: nonaligned prefill, final legal decode."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import ccl_inventory, source_identity
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {"passed": False, "scope": "all64 TP4, one maximum-context acceptance"}
    report["source_sha256"] = source_identity()

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        model = Model(mesh)
        capacity = model.config.max_position_embeddings
        assert capacity == 262144
        report.update(capacity=capacity, weights_loaded_memory=model.loaded_memory)
        # Match the pinned plugin's B1 Q=262144 budget plus one reserved block.
        # Page zero remains an unused/null page; active history uses pages1..8192.
        page_table = torch.arange(1, capacity // 32 + 1, dtype=torch.int32)[None]
        gen = QwenGenerator(model, max_seq_len=capacity, page_table=page_table)
        report["cache_and_setup_memory"] = model.memory_view()
        report["physical_pages"] = capacity // 32 + 1
        report["reserved_null_page"] = 0
        report["precision_config"] = model.precision
        tile_bytes = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}[model.precision["kv_cache_dtype"]]
        report["kv_payload_bytes_per_device"] = report["physical_pages"] * 16 * 2 * 8 * tile_bytes
        report["logical_context_kv_payload_bytes_per_device"] = capacity * 16 * 2 * 256 * tile_bytes // 1024
        report["page_size"] = 32
        gen.prepare(prefill_lengths=(2,))
        report["prepared_memory"] = model.memory_view()
        report["prepared_ccl_inventory"] = ccl_inventory(model)
        programs = int(mesh.num_program_cache_entries())
        traces = dict(gen.traces)
        chunk_count = 0
        original = gen.prefill_single._layer_chunk

        def observed(layer, *a, **kw):
            nonlocal chunk_count
            result = original(layer, *a, **kw)
            if layer is model.layers[-1]:
                chunk_count += 1
                if chunk_count % 32 == 0:
                    print(f"CONTEXT_PREFILL chunks={chunk_count}/512", flush=True)
            return result

        gen.prefill_single._layer_chunk = observed
        seed_text = "The city library keeps records of books, dates, authors, and scientific discoveries. "
        pattern = gen.tokenizer.encode(seed_text, add_special_tokens=False)
        prompt = (pattern * ((capacity - 2) // len(pattern) + 1))[: capacity - 1]
        gen.reset()
        gen.configure_sampling(temperature=0.0)
        start = time.perf_counter()
        print(f"CONTEXT_PREFILL_BEGIN length={len(prompt)}", flush=True)
        first = int(
            gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=gen.page_host,
                kv_cache=gen.cache,
                prompt_lens=[len(prompt)],
                sampling_params=SimpleNamespace(temperature=0.0, top_k=1, top_p=0.0, seed=0),
            )[0, 0]
        )
        report["prefill_seconds"] = time.perf_counter() - start
        report["prefill_length"] = len(prompt)
        report["after_prefill_memory"] = model.memory_view()
        save()
        gen._write(torch.tensor([capacity - 1], dtype=torch.int32), gen.positions, "position_refreshes")
        gen._write(torch.tensor([capacity - 1], dtype=torch.int32), gen.rope_positions, "rope_refreshes")
        before = dict(gen.counters)
        start = time.perf_counter()
        second = int(
            gen.decode_forward(
                None, None, page_table=gen.page_host, kv_cache=gen.cache, reload_inputs=False, sample_on_device=True
            )[0, 0]
        )
        report["decode_seconds"] = time.perf_counter() - start
        position = int(ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).item())
        report.update(
            tokens=[first, second],
            decode_position=capacity - 1,
            next_position=position,
            after_decode_memory=model.memory_view(),
            program_cache_before=programs,
            program_cache_after=int(mesh.num_program_cache_entries()),
            traces_unchanged=gen.traces == traces,
            decode_counter_delta={key: value - before.get(key, 0) for key, value in gen.counters.items()},
            bounded_prefill=dict(gen.prefill_single.counters),
        )
        assert position == capacity and all(0 <= token < model.config.vocab_size for token in (first, second))
        assert gen.traces == traces and int(mesh.num_program_cache_entries()) == programs
        assert gen.counters["full_logits_readbacks"] == 0
        for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
            assert gen.counters[name] == before.get(name, 0)
        report["passed"] = True
        print("CONTEXT_ACCEPTANCE passed=True context=262144", flush=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
