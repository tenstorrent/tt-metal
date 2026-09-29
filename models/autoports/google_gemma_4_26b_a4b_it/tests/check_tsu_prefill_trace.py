# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer controls for chunked prefill and decode trace reuse.

This is a correctness/transition probe, not a full-model performance result.
"""

import argparse
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path
from types import MethodType

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 4097])
    parser.add_argument("--trace-max-length", type=int, default=8192)
    parser.add_argument("--cache-context", type=int, default=262144)
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument(
        "--full-model", action="store_true", help="Use all 30 layers for final token-equivalence checks"
    )
    parser.add_argument("--reuse-eager", action="store_true", help="Retain decode only; do not capture long prefill")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layers = tuple(range(30)) if args.full_model else (0, 5)
    model_root = Path(__file__).resolve().parent.parent
    report = {
        "layers": list(layers),
        "scope": "token-equivalence correctness probe",
        "config": vars(args) | {"output": str(args.output)},
        "source_sha256": {
            str(path.relative_to(model_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), *sorted((model_root / "tt").glob("*.py"))]
        },
        "rows": [],
        "passed": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1000000000)
    gen = None
    try:
        context = max(args.cache_context, ((max(args.lengths) + args.decode_steps + 31) // 32) * 32)
        gen = Gemma4Generator(mesh, max_seq_len=context, layer_indices=layers)
        if not args.reuse_eager:
            # Probe-only reproduction of the rejected long-prefill trace policy.
            # The production limit remains1024; full8K exceeds the1GB region.
            def experimental_prefill_key(self, tokens, page_table, kv_cache, prompt_lens):
                if not self.prefill_trace_enabled or len(prompt_lens) != 1 or tokens.shape[0] != 1:
                    return None
                length = int(prompt_lens[0])
                if not 1 <= length <= min(args.trace_max_length, tokens.shape[1], self.model.max_seq_len):
                    return None
                if any(
                    not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[0] != 1 for t in self._tables(page_table)
                ):
                    return None
                return (
                    length,
                    id(kv_cache),
                    tuple(id(t) for pair in kv_cache for t in pair),
                    self._table_shapes(page_table),
                )

            gen._serving_prefill_key = MethodType(experimental_prefill_key, gen)
        adapter = AutoportGemma4ForCausalLM(gen, 32)
        cache, table = gen.model.allocate_cache(slots=1, context=context)
        compact = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        padded = SamplingParams(**{f.name: [getattr(compact, f.name)] * 32 for f in fields(SamplingParams)})

        def sampler_traces():
            return {
                str(key): str(slot["id"]) for key, slot in gen.sampler._trace_states.items() if slot["id"] is not None
            }

        def request(length, token, reverse_pages):
            ids = torch.full((1, length), token, dtype=torch.long)
            ids[0, 0] = 2
            host_table = table.flip(1) if reverse_pages else table.clone()
            padded_table = torch.nn.functional.pad(host_table, (0, 0, 0, 31))
            tables = [padded_table] * 30
            before = dict(gen.counters)
            trace_before = gen.trace_id
            sampling_before = sampler_traces()
            programs_before = mesh.num_program_cache_entries()
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            output = adapter.prefill_forward(
                ids, padded_table, cache, [length], sampling_params=compact, page_tables_per_layer=tables
            )
            prefill_ms = (time.perf_counter() - start) * 1000
            tokens = [int(output.flatten()[0])]
            decode_ms = []
            for step in range(args.decode_steps):
                start = time.perf_counter()
                output = adapter.decode_forward(
                    output,
                    torch.tensor([length + step], dtype=torch.int32),
                    padded_table,
                    cache,
                    sampling_params=padded,
                    page_tables_per_layer=tables,
                    reset_batch=step == 0,
                )
                decode_ms.append((time.perf_counter() - start) * 1000)
                tokens.append(int(output.flatten()[0]))
            return {
                "length": length,
                "input_token": token,
                "reversed_pages": reverse_pages,
                "tokens": tokens,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "counters": {k: v - before.get(k, 0) for k, v in gen.counters.items()},
                "trace_before": None if trace_before is None else str(trace_before),
                "trace_after": None if gen.trace_id is None else str(gen.trace_id),
                "sampling_traces_before": sampling_before,
                "sampling_traces_after": sampler_traces(),
                "program_cache_delta": mesh.num_program_cache_entries() - programs_before,
            }

        controls_by_length = {}
        for length in args.lengths:
            controls = []
            gen.prefill_trace_enabled = False
            adapter.eager_prefill_decode_reuse = False
            for token, reverse in ((100, False), (113, False), (113, True)):
                controls.append(request(length, token, reverse))
            controls_by_length[length] = controls
            gen._release_trace()
            gen.prefill_trace_enabled = True
            adapter.eager_prefill_decode_reuse = args.reuse_eager
            for repeat, (token, reverse, control) in enumerate(
                (
                    (100, False, controls[0]),
                    (100, False, controls[0]),
                    (113, False, controls[1]),
                    (113, True, controls[2]),
                )
            ):
                mesh.set_program_cache_misses_allowed(not (args.reuse_eager and repeat > 0))
                try:
                    row = request(length, token, reverse)
                finally:
                    mesh.set_program_cache_misses_allowed(True)
                row.update(repeat=repeat, control=control, matches_control=row["tokens"] == control["tokens"])
                report["rows"].append(row)
                save()
                assert row["matches_control"], row
                if repeat and (args.reuse_eager or length <= args.trace_max_length):
                    assert row["counters"].get("decode_captures", 0) == 0, row
                    assert row["counters"].get("prefill_replays", 0) == (0 if args.reuse_eager else 1), row
                    assert row["trace_before"] == row["trace_after"], row
                    assert row["sampling_traces_before"], row
                    assert row["sampling_traces_before"] == row["sampling_traces_after"], row
                    assert row["program_cache_delta"] == 0, row
                print("TSU_TRACE_CASE", json.dumps(row), flush=True)
        if args.reuse_eager and 4096 in controls_by_length and 4097 in controls_by_length:
            for length in (4096, 4097, 4096):
                row = request(length, 100, False)
                row["transition"] = True
                row["matches_control"] = row["tokens"] == controls_by_length[length][0]["tokens"]
                report["rows"].append(row)
                save()
                assert row["matches_control"], row
                assert row["counters"].get("decode_captures", 0) == 1, row
        report["passed"] = True
        save()
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
