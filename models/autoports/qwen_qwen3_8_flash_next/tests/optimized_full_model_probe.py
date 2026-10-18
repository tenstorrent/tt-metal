"""Same-session prefill controls with all real layer kinds and live trace reuse."""

import argparse
import json
import time
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_trace_contract import host, guard_runtime
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="prefill_probe")
    ap.add_argument(
        "--lengths", default="1,2,3,7,8,15,16,31,32,33,63,64,65,127,128,129,203,255,256,257,511,512,513,2053"
    )
    ap.add_argument("--trace-max", type=int, default=512)
    ap.add_argument("--trace-bytes", type=int, default=768000000)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=a.trace_bytes)
    gen = None
    out = Path("models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model")
    report = dict(args=vars(a), rows=[])
    start = time.perf_counter()
    try:
        gen = build_generator(
            out.parents[1], mesh, checkpoint=a.checkpoint, layer_indices=[0, 1, 3], prefill_trace_max=a.trace_max
        )
        gen.model.forward = guard_runtime(gen.model.forward)
        gen.sampler.decode_forward = guard_runtime(gen.sampler.decode_forward)
        gen.prepare()
        report["trace_memory"] = view(mesh, ttnn.BufferType.TRACE)
        report["reset_ms"] = {}
        for traced in (False, True):
            gen.trace_cache_reset = traced
            times = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                tick = time.perf_counter()
                gen.reset()
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - tick) * 1000)
            report["reset_ms"][str(traced)] = times
        ids = dict(gen.traces)
        buckets = gen.prefill_trace_buckets.copy()
        for n in map(int, a.lengths.split(",")):
            prompt = [100 + i % 300 for i in range(n)]
            row = dict(length=n, variants={})
            expected = None
            for name, last, traces in [
                ("baseline", False, []),
                ("last_only", True, []),
                ("traced", True, buckets),
                ("traced_repeat", True, buckets),
            ]:
                gen.last_token_only, gen.prefill_trace_buckets = last, traces
                gen.trace_cache_reset = name.startswith("traced")
                tokens = gen.generate(prompt, 4)
                if expected is None:
                    expected = tokens
                assert tokens == expected, (n, name, tokens, expected)
                row["variants"][name] = dict(perf=dict(gen.last_perf), tokens=tokens)
            assert gen.traces == ids and gen.counters["releases"] == 0
            report["rows"].append(row)
            print("PREFILL_CONTROL_PASS", json.dumps(row), flush=True)
            (out / f"{a.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
        # Changed page mappings and continuation offsets select the same traces.
        prompt = [100 + i % 300 for i in range(203)]
        expected = gen.generate(prompt, 4)
        pages = gen.page_table.flip(1).contiguous()
        gen.reset()
        for begin, end in ((0, 73), (73, 203)):
            first = gen.prefill_forward(
                torch.tensor([prompt[begin:end]]),
                page_table=pages,
                kv_cache=gen.kv_cache,
                prompt_lens=[end - begin],
                start_pos=[begin],
            )
        actual = [int(first[0])]
        before_pages = gen.counters["page_table_refreshes"]
        for _ in range(3):
            actual.append(int(gen.decode_forward(None, None, page_table=pages, kv_cache=gen.kv_cache)[0]))
        assert gen.counters["page_table_refreshes"] == before_pages
        # Different chunk boundaries can change quantization; compare each path
        # against the same eager continuation, independently of whole-prompt math.
        gen.prefill_trace_buckets = []
        gen.reset()
        for begin, end in ((0, 73), (73, 203)):
            first = gen.prefill_forward(
                torch.tensor([prompt[begin:end]]),
                page_table=pages,
                kv_cache=gen.kv_cache,
                prompt_lens=[end - begin],
                start_pos=[begin],
            )
        control = [int(first[0])]
        for _ in range(3):
            control.append(int(gen.decode_forward(None, None, page_table=pages, kv_cache=gen.kv_cache)[0]))
        assert actual == control, (actual, control)
        gen.prefill_trace_buckets = buckets
        report["continuation"] = dict(
            starts=[0, 73],
            lengths=[73, 130],
            traced=actual,
            eager=control,
            whole_prompt=expected,
            changed_page_mapping=True,
            unchanged_decode_page_copies=0,
        )
        assert ids == gen.traces
        before = dict(gen.counters)
        prior = host(gen.tt_token).reshape(-1).tolist()
        pos = host(gen.tt_pos)[:, 0].tolist()
        feedback = []
        for _ in range(3):
            gen.decode_forward(None, None, page_table=None, kv_cache=gen.kv_cache, read_tokens=False)
            current = host(gen.tt_token).reshape(-1).tolist()
            nextpos = host(gen.tt_pos)[:, 0].tolist()
            assert nextpos == [p + 1 for p in pos]
            feedback.append(dict(consumed=prior, output=current, positions=pos, next_positions=nextpos))
            prior, pos = current, nextpos
        delta = {k: gen.counters[k] - before.get(k, 0) for k in gen.counters}
        for key in (
            "token_refreshes",
            "position_refreshes",
            "page_table_refreshes",
            "synchronizations",
            "token_readbacks",
            "history_readbacks",
        ):
            assert delta.get(key, 0) == 0, delta
        for mode in ("greedy", "sample", "argmax", "greedy"):
            gen._replay(mode)
        report.update(
            status="pass",
            trace_ids={k: str(v) for k, v in ids.items()},
            counters=dict(gen.counters),
            feedback=feedback,
            steady_state_counter_delta=delta,
            memory=view(mesh),
            elapsed_seconds=time.perf_counter() - start,
        )
        (out / f"{a.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
        print("OPTIMIZED_PREFILL_PASS", flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
