"""Prepared full-path prefill trace alongside split decode, with real shapes."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    trace = None
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers, head_k=2)
        prompt = gen.tokenizer.encode(
            "The sky appears blue because sunlight scatters in the atmosphere. " * args.length
        )[: args.length]
        reference = gen.generate(prompt, 8)
        gen._release_traces(drop_state=False)
        tokens = gen.model.upload(torch.tensor([prompt]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        indices = gen.model.upload(
            torch.arange(args.length).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        table = gen.model.upload(gen.page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        plan = gen.model.layers[0].prepare_prefill(seq_len=args.length)

        def prefill():
            return gen.model.prefill_from_device(
                tokens, indices, plan=plan, page_table=table, kv_cache=gen.kv_cache, last_only=True
            )

        warm = prefill()
        expected = gen._read_logits(warm)
        logits = ttnn.empty_like(warm)
        ttnn.copy(warm, logits)
        del warm
        eager_ms = []
        for _ in range(5):
            begin = time.perf_counter()
            out = prefill()
            ttnn.synchronize_device(mesh)
            eager_ms.append((time.perf_counter() - begin) * 1000)
            del out
        gen._prepare_state(
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([args.length]),
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
        )
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        transient = prefill()
        ttnn.copy(transient, logits)
        del transient
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traced_ms = []
        for _ in range(5):
            begin = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            traced_ms.append((time.perf_counter() - begin) * 1000)
        actual = gen._read_logits(logits)
        assert torch.equal(actual, expected)
        gen.configure_sampling(seed=123)
        gen._sample(gen.model.sampler_logits(logits), strategy="split")
        observed = [int(gen._read_tokens(batch=1)[0])]
        for _ in range(7):
            gen.replay()
            observed.append(int(gen._read_tokens(batch=1)[0]))
        assert observed == reference, (observed, reference)
        result = {
            "layers": args.layers,
            "logical_length": args.length,
            "eager_ms": eager_ms,
            "traced_ms": traced_ms,
            "eager_median_ms": statistics.median(eager_ms),
            "traced_median_ms": statistics.median(traced_ms),
            "logits_exact": True,
            "tokens": observed,
            "split_decode_after_prefill_trace": True,
            "pass": True,
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
