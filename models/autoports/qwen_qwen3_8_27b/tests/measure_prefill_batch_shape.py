# SPDX-License-Identifier: Apache-2.0
"""Time one live prefill row at several resident batch shapes (local, all64 layers).

Measures how padded-batch prefill cost scales with the configured batch while only
one row carries tokens. Timing only; exactness is checked by probe_unpadded_prefill.py.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import ttnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/data/mvasiljev/weights/Qwen3.8-27B")
    p.add_argument("--batches", type=int, nargs="+", default=[32, 8, 4, 2, 1])
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 4096])
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--all-live-batches", type=int, nargs="*", default=[8])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault("QWEN38_GROUPED_RECURRENCE", "1")
    os.environ.setdefault("QWEN38_MASKED_RECURRENCE", "1")
    os.environ.setdefault("QWEN38_FLAT_PREFILL", "1")
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
    from models.autoports.qwen_qwen3_8_27b.tt.model import Model
    from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator

    out = dict(flags={k: os.environ.get(k) for k in ("QWEN38_GROUPED_RECURRENCE", "QWEN38_MASKED_RECURRENCE",
                                                      "QWEN38_FLAT_PREFILL", "QWEN38_UNPADDED_PREFILL")},
               context=a.context, chunk=a.chunk, records=[])
    save = lambda: a.output.write_text(json.dumps(out, indent=1))
    try:
        t = time.perf_counter()
        model = Model(a.checkpoint, mesh)
        out["load_s"] = time.perf_counter() - t
        torch.manual_seed(0)
        for batch in a.batches:
            gen = Generator(model, a.checkpoint, batch_size=batch, context_length=a.context,
                            prefill_chunk_size=a.chunk, serving_state=True)

            def run(lengths, start=0):
                tokens = torch.randint(1000, 100000, (batch, max(lengths)), dtype=torch.int64)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                if start:
                    gen.prefill_continuation(tokens, page_table=gen.page_table, kv_cache=gen.kv_cache,
                                             prompt_lens=lengths, start_position=start)
                else:
                    gen.prefill_forward(tokens, page_table=gen.page_table, kv_cache=gen.kv_cache,
                                        prompt_lens=lengths, return_device=True)
                ttnn.synchronize_device(mesh)
                return time.perf_counter() - t0

            cases = [(n, 0, 1) for n in a.lengths] + [(512, 2048, 1)]
            if batch in a.all_live_batches:
                cases.append((512, 0, batch))
            for length, start, live in cases:
                lens = [length if i < live else 0 for i in range(batch)]
                warm = run(lens, start)
                times = [run(lens, start) for _ in range(a.repeats)]
                rec = dict(batch=batch, live_rows=live, length=length, start=start, warm_s=warm,
                           times_s=times, median_s=sorted(times)[len(times) // 2],
                           tok_per_s=live * length / sorted(times)[len(times) // 2])
                out["records"].append(rec)
                print("MEASURE", json.dumps(rec), flush=True)
                save()
            mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            out.setdefault("free_bytes_after", {})[batch] = mv.total_bytes_free_per_bank * mv.num_banks
            for state in gen.kv_cache:
                for v in vars(state).values():
                    ttnn.deallocate(v)
            del gen
        out["status"] = "pass"
    except BaseException as e:
        out.update(status="fail", error=repr(e))
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
