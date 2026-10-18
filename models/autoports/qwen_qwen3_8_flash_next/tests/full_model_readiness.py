"""All-layer standard readiness gates sharing one loaded model and trace set."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import ttnn
from readiness_check import run_prefill_check, run_teacher_forcing, run_autoregressive
from readiness_check.schema import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view


class SessionView:
    """Standard runners borrow the session; only main owns final teardown."""

    def __init__(self, generator):
        self.generator = generator

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def teardown(self):
        pass

    def generate(self, *args, enable_trace=True, **kwargs):
        self.last_tokens = self.generator.generate(*args, enable_trace=enable_trace, **kwargs)
        return self.last_tokens

    def prefill_logits(self, *args, **kwargs):
        logits = self.generator.prefill_logits(*args, **kwargs)
        self.prefill_predictions = logits.argmax(-1)[0].tolist()
        return logits


def prediction_details(predictions, entry):
    rows = []
    for i, token in enumerate(predictions):
        expected = entry.topk_tokens[i].tolist()
        pos = entry.tf_prompt_len + i - 1
        rows.append(
            dict(
                index=i,
                prediction=token,
                hf_top1=expected[0],
                top1=token == expected[0],
                top5=token in expected[:5],
                top100=token in expected,
                absolute_input_position=pos,
                page_id=pos // 32,
                attention_chunk=pos // 128,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--compare-prefill", action="store_true")
    ap.add_argument("--long-context", action="store_true")
    ap.add_argument("--wait-reference", action="store_true")
    ap.add_argument("--tag", default="readiness")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--prefill-trace-max", type=int, default=None)
    ap.add_argument("--trace-region-size", type=int, default=768000000)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    root = Path("models/autoports/qwen_qwen3_8_flash_next").resolve()
    output = Path(a.output_dir) if a.output_dir else root / "doc/full_model"
    output.mkdir(parents=True, exist_ok=True)
    reference = root / "readiness_aime24_chat.refpt"
    if not a.smoke_only and not a.wait_reference:
        assert reference.exists(), reference
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=a.trace_region_size)
    gen = None
    report = dict(checkpoint=a.checkpoint, context=262144, layers=48, initial_memory=view(mesh))

    report["args"] = vars(a)
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "tt").glob("*.py")}
    native_reader = Path(
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/"
        "device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp"
    )
    report["source_sha256"][str(native_reader)] = hashlib.sha256(native_reader.read_bytes()).hexdigest()

    def save():
        (output / f"{a.tag}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    def phase(name):
        report["phase"] = name
        save()
        print("READINESS_PHASE", name, flush=True)

    try:
        tick = time.perf_counter()
        factory = run_teacher_forcing._import_build_generator(root)
        gen = factory(root, mesh, checkpoint=a.checkpoint, prefill_trace_max=a.prefill_trace_max)
        report["loaded_memory"] = view(mesh)
        report["load_seconds"] = time.perf_counter() - tick
        save()
        gen.prepare()
        report["prepared_memory"] = view(mesh)
        report["trace_memory"] = view(mesh, ttnn.BufferType.TRACE)
        report["trace_ids"] = {k: str(v) for k, v in gen.traces.items()}
        report["smoke_tokens"] = gen.generate([100, 101, 102, 103, 104, 105, 106], 4)
        report["smoke_perf"] = dict(gen.last_perf)
        save()
        if a.smoke_only:
            return
        while not reference.exists() or not (root / "doc/full_model/reference_metadata.json").exists():
            assert a.wait_reference, "Fresh reference not ready"
            print("WAITING_FOR_FRESH_REFERENCE", time.strftime("%H:%M:%S"), flush=True)
            time.sleep(30)
        borrowed = SessionView(gen)
        for runner in (run_prefill_check, run_teacher_forcing, run_autoregressive):
            runner._import_build_generator = lambda _: lambda **kwargs: borrowed
        entry = load_reference(reference).entries[0]
        report["prefill"] = run_prefill_check.run_prefill_check(
            model_dir=root, reference_path=reference, mesh_device=mesh
        )
        report["prefill_predictions"] = prediction_details(
            borrowed.prefill_predictions[entry.tf_prompt_len - 1 : entry.tf_prompt_len + entry.num_generated - 1], entry
        )
        save()
        report["teacher_forcing"] = run_teacher_forcing.run_teacher_forcing(
            model_dir=root, reference_path=reference, mesh_device=mesh
        )
        report["decode_predictions"] = prediction_details(borrowed.last_tokens, entry)
        save()
        # The freshly generated pinned reference supplies the identical greedy
        # HF completion; validate exact input IDs and requested generation size.
        prompt = entry.prompt_tokens[0].tolist()
        rendered = gen.tokenizer.decode(prompt, skip_special_tokens=False)
        assert gen.tokenizer.encode(rendered, add_special_tokens=False) == prompt
        prompt_file = output / "aime24_rendered_prompt.txt"
        prompt_file.write_text(rendered)

        class RenderedTokenizer:
            # The generic runner strips prompt files. Restore the exact pinned
            # chat-template IDs, including its trailing generation whitespace.
            def encode(self, text, **kwargs):
                assert text == rendered.strip()
                return prompt

            def decode(self, *args, **kwargs):
                return gen.tokenizer.decode(*args, **kwargs)

        run_autoregressive.AutoTokenizer.from_pretrained = lambda *args, **kwargs: RenderedTokenizer()

        def reference_completion(**kwargs):
            assert kwargs["prompt_token_ids"] == prompt
            assert kwargs["max_new_tokens"] == entry.num_generated
            return entry.generated_tokens[0].tolist()

        run_autoregressive._hf_generate_greedy = reference_completion
        paths = run_autoregressive.run_autoregressive(
            model_dir=root,
            hf_model_id="Qwen/Qwen3.8-Flash-Next",
            prompt_file=prompt_file,
            mesh_device=mesh,
            output_dir=output / "autoregressive_aime24",
            max_new_tokens=entry.num_generated,
        )
        report["autoregressive"] = {k: str(v) for k, v in paths.items()}
        report["hf_control_source"] = str(reference)
        phase("primary_warmup")
        # Warmed primary workload, including caller-visible token/history reads.
        primary = (prompt * 2)[:128]
        gen.generate(primary, 128)
        perf = []
        report["primary_counter_deltas"] = []
        report["primary_tokens"] = []
        report["primary_128_128"] = perf
        for repetition in range(3):
            phase(f"primary_{repetition}")
            before = dict(gen.counters)
            tokens = gen.generate(primary, 128)
            report["primary_tokens"].append(tokens)
            report["primary_counter_deltas"].append({k: gen.counters[k] - before.get(k, 0) for k in gen.counters})
            perf.append(dict(gen.last_perf))
            save()
        assert report["primary_tokens"][0] == report["primary_tokens"][1] == report["primary_tokens"][2]
        report["primary_128_128"] = perf
        # A distinct model-only trace timing excludes sampler and host feedback.
        model_only = []
        report["traced_logits_only"] = model_only
        for repetition in range(3):
            phase(f"logits_only_{repetition}")
            gen.generate(primary, 1)
            ttnn.synchronize_device(mesh)
            tick = time.perf_counter()
            for _ in range(127):
                gen._replay("decode")
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - tick
            model_only.append(
                dict(
                    tokens=127,
                    seconds=elapsed,
                    tps=127 / elapsed,
                    boundary="model trace to logits; fixed input token, advancing device positions; no sampling",
                )
            )
            save()
        report["traced_logits_only"] = model_only
        if a.compare_prefill:
            buckets = gen.prefill_trace_buckets.copy()
            report["prefill_comparison"] = []
            for n in (33, 128, 203, 513, 2053):
                tokens_in = (prompt * (n // len(prompt) + 1))[:n]
                row = dict(prompt_len=n, gen_len=4, variants={})
                expected = None
                for name, last, traces in (
                    ("baseline", False, []),
                    ("last_only", True, []),
                    ("selected", True, buckets),
                ):
                    phase(f"prefill_comparison_{n}_{name}")
                    gen.last_token_only, gen.prefill_trace_buckets = last, traces
                    gen.trace_cache_reset = name == "selected"
                    samples = []
                    for _ in range(3):
                        tokens = gen.generate(tokens_in, 4)
                        samples.append(dict(gen.last_perf))
                        if expected is None:
                            expected = tokens
                        assert tokens == expected, (n, name, tokens, expected)
                    row["variants"][name] = dict(samples=samples, tokens=tokens)
                report["prefill_comparison"].append(row)
                save()
            gen.last_token_only, gen.prefill_trace_buckets = True, buckets
            gen.trace_cache_reset = True
        if a.long_context:
            phase("full_context")
            length = gen.model.context - 3
            tick = time.perf_counter()
            tokens = gen.generate([100 + i % 300 for i in range(length)], 4)
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.tt_pos)[0])[:, 0].tolist()
            assert len(tokens) == 4 and position == [gen.model.context]
            report["long_context"] = dict(
                prompt_len=length, tokens=tokens, final_position=position,
                seconds=time.perf_counter() - tick, perf=dict(gen.last_perf),
                scope="All48 execution/capacity; synthetic token IDs, no language-quality claim",
            )
            save()
        if (root / "doc/full_model/qualitative_hf.json").exists():
            from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_qualitative import run_tt

            controls = json.loads((root / "doc/full_model/qualitative_hf.json").read_text())
            if len(controls["prompts"]) == 6:
                report["qualitative"] = run_tt(gen, root, output_dir=output)
        assert report["trace_ids"] == {k: str(v) for k, v in gen.traces.items()}
        assert gen.counters["captures"] == len(gen.traces) and gen.counters["releases"] == 0
        report["counters"] = dict(gen.counters)
        report["final_memory"] = view(mesh)
        for stage in ("prefill", "teacher_forcing"):
            for row in report[stage]:
                assert row["top5"] >= 0.98 and row["top100"] == 1.0, (stage, row)
        report["phase"] = "complete"
        save()
        print("FULL_MODEL_READINESS_PASS", json.dumps(report, default=str), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
