# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Public resident decode, TTFT decomposition and nonaligned prefill evidence."""
import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import source_identity
from models.autoports.qwen_qwen3_6_27b.tests.profile_full_terminal import delta, snapshot
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def first_token(gen, prompt, sampling):
    start = time.perf_counter()
    gen.reset()
    gen.configure_sampling(**sampling)
    gen._prefill_last_into_logits(prompt)
    gen._reset_rng(torch.tensor([len(prompt) - 1]))
    gen._execute("sample")
    first = gen._read_tokens()[0]
    ttft = time.perf_counter() - start
    position = torch.tensor([len(prompt)], dtype=torch.int32)
    gen._write(position, gen.positions, "position_refreshes")
    gen._write(position, gen.rope_positions, "rope_refreshes")
    return first, ttft


def resident(gen, steps):
    before = dict(gen.counters)
    start = time.perf_counter()
    for _ in range(steps):
        output = gen.decode_forward(
            None, None, page_table=gen.page_host, kv_cache=gen.cache, reload_inputs=False, read_from_device=False
        )
        assert output is gen.tokens
    ttnn.synchronize_device(gen.mesh)
    seconds = time.perf_counter() - start
    counters = delta(before, dict(gen.counters))
    assert counters.get("decode_replays") == counters.get("sample_replays") == steps
    assert all(
        value == 0 for key, value in counters.items() if key not in ("decode_replays", "sample_replays")
    ), counters
    return dict(
        steps=steps,
        seconds=seconds,
        ms_per_step=seconds * 1000 / steps,
        tokens_per_second_per_user=steps / seconds,
        counters=counters,
        synchronizations=1,
        synchronization_scope="one at window end, none per token",
        blocking_replay=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--lengths", default="33,127,128,129,511,512,513,1025")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--steps", type=int, default=127)
    a = p.parse_args()
    torch.set_num_threads(8)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=False,
        source_sha256=source_identity(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={k: v for k, v in os.environ.items() if k.startswith("TT_METAL_")},
        context=a.context,
        batch=1,
        prompt_length=128,
        generated=a.steps + 1,
        regime="serial TP4 chips0..3",
        diagnostic=a.diagnostic,
    )

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        model = Model(mesh, layer_indices=None if a.all_layers else [0, 3])
        gen = QwenGenerator(model, max_seq_len=a.context)
        report["precision_config"] = model.precision
        lengths = list(map(int, a.lengths.split(","))) if a.lengths else []
        if a.diagnostic:
            buckets = {128}
            for n in lengths:
                buckets.update([min(n, 512), n % 512 or 512])
            gen.prefill_single.prepare(_diagnostic_lengths=sorted(buckets))
        gen.prepare(prefill_lengths=(128,))
        report["runtime_precision"] = model.precision_summary()
        report["runtime_buffers"] = gen.precision_runtime_summary()
        report["before"] = snapshot(gen)
        prompt = ((torch.arange(128) * 37 + 101) % model.config.vocab_size).tolist()
        settings = [
            dict(temperature=0.0, top_k=1, top_p=0.0, seed=1700),
            dict(temperature=1.0, top_k=20, top_p=0.95, seed=1700),
        ]
        report["feedback_checks"] = []
        for sampling in settings:
            helper = gen.prefill_trace
            gen.prefill_trace = None
            try:
                eager = gen.generate(prompt, a.steps + 1, ignore_eos=True, **sampling)
                eager_timings = dict(gen.last_timings)
            finally:
                gen.prefill_trace = helper
            expected = gen.generate(prompt, a.steps + 1, ignore_eos=True, **sampling)
            assert expected == eager, (expected, eager)
            first, _ = first_token(gen, prompt, sampling)
            assert first == expected[0]
            previous = 0
            checks = []
            for endpoint in sorted({min(1, a.steps), min(2, a.steps), min(31, a.steps), a.steps}):
                result = resident(gen, endpoint - previous)
                observed = gen._read_tokens()[0]
                assert observed == expected[endpoint], (endpoint, observed, expected[endpoint])
                checks.append(dict(endpoint=endpoint, token=observed, window=result))
                previous = endpoint
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).flatten().tolist()
            assert position == [128 + a.steps], position
            report["feedback_checks"].append(
                dict(
                    sampling=sampling,
                    checks=checks,
                    position=position,
                    eager_trace_tokens_equal=True,
                    eager_timings=eager_timings,
                )
            )
            save()
        report["runs"] = []
        for _ in range(a.repeats):
            first, ttft = first_token(gen, prompt, settings[0])
            ttnn.synchronize_device(mesh)
            result = resident(gen, a.steps)
            result.update(ttft_ms=ttft * 1000, first=first, final=gen._read_tokens()[0])
            report["runs"].append(result)
            save()
        report["median"] = {
            key: statistics.median(r[key] for r in report["runs"])
            for key in ("ttft_ms", "ms_per_step", "tokens_per_second_per_user")
        }
        report["sampling_param_ab"] = []
        for _ in range(a.repeats):
            pair = {}
            for cached in (False, True):
                gen.cache_sampling_params = cached
                before = dict(gen.counters)
                token, elapsed = first_token(gen, prompt, settings[0])
                pair[str(cached)] = dict(
                    ttft_ms=elapsed * 1000, token=token, counters=delta(before, dict(gen.counters))
                )
            assert pair["False"]["token"] == pair["True"]["token"]
            assert pair["False"]["counters"]["sampling_param_refreshes"] == 1
            assert pair["True"]["counters"]["sampling_param_refreshes"] == 0
            report["sampling_param_ab"].append(pair)
        gen.cache_sampling_params = True
        report["prefill_rows"] = []
        # Change physical page ownership and prompt data before returning to the
        # retained P128 trace after nonaligned eager requests below.
        gen._bind_pages(torch.roll(gen.page_host, 16, dims=1), gen.cache)
        for length in lengths:
            prompt = ((torch.arange(length) * 37 + 101) % model.config.vocab_size).tolist()
            helper = gen.prefill_trace
            gen.prefill_trace = None
            eager_times = []
            try:
                for _ in range(a.repeats):
                    _, ttft = first_token(gen, prompt, settings[0])
                    eager_times.append(ttft * 1000)
            finally:
                gen.prefill_trace = helper
            times = []
            for _ in range(a.repeats):
                _, ttft = first_token(gen, prompt, settings[0])
                times.append(ttft * 1000)
            # End with decode to check allocations created by every eager length.
            resident(gen, 1)
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).flatten().tolist()
            assert position == [length + 1], (length, position)
            report["prefill_rows"].append(
                dict(
                    length=length,
                    chunk_size=512,
                    chunks=(length + 511) // 512,
                    ttft_ms=times,
                    eager_ttft_ms=eager_times,
                    median_ttft_ms=statistics.median(times),
                    post_decode_position=position,
                )
            )
            save()
        changed_prompt = ((torch.arange(128) * 19 + 303) % model.config.vocab_size).tolist()
        helper = gen.prefill_trace
        gen.prefill_trace = None
        try:
            expected = gen.generate(changed_prompt, 4, ignore_eos=True, **settings[0])
        finally:
            gen.prefill_trace = helper
        actual = gen.generate(changed_prompt, 4, ignore_eos=True, **settings[0])
        assert actual == expected
        report["changed_pages_and_tokens"] = dict(passed=True, tokens=actual, page_rotation=16)
        report["after"] = snapshot(gen)
        assert report["before"] == report["after"], (report["before"], report["after"])
        report["memory"] = model.memory_view()
        report["passed"] = True
    finally:
        save()
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)
    print(json.dumps(dict(passed=True, median=report["median"])), flush=True)


if __name__ == "__main__":
    main()
