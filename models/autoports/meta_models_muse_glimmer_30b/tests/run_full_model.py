# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the unmodified packaged readiness checks on one prepared full model.

The build adapter shares construction, and defers teardown until all requested
checks finish. HF controls are produced in the separate official-reference
environment; their exact prompt/token metadata is verified before reuse.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_autoregressive as autoregressive
from readiness_check import run_prefill_check as prefill_check
from readiness_check import run_teacher_forcing as teacher_check

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator
from models.autoports.meta_models_muse_glimmer_30b.tt.model import MODEL_ID, REVISION

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/full_model"


class SharedGenerator:
    def __init__(self, generator):
        self.generator = generator

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def teardown(self):
        pass


def main(args):
    evidence_dir = args.evidence_dir or DOC
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    result = dict(model_id=MODEL_ID, revision=REVISION, layers=52, checks=args.checks)
    result.update(
        command=[sys.executable, *sys.argv],
        precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
        hardware="QB2/P300x2: 4 Blackhole chips, 2 p300c boards",
        mesh=[1, 4],
    )
    result["implementation_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "tt").glob("*.py")
    }
    generator = None

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        construction_started = time.perf_counter()
        generator = build_generator(
            ROOT,
            mesh,
            terminal_policy=json.loads(args.terminal_policy),
            trace_reset=not args.eager_reset,
            clear_cache_on_generate=args.clear_cache_on_generate,
        )
        result["clear_cache_on_generate"] = generator.clear_cache_on_generate
        result["trace_reset"] = generator.trace_reset
        result["terminal_policy"] = generator.model.terminal_policy
        if hasattr(generator.model, "precision_config"):
            from models.autoports.meta_models_muse_glimmer_30b.tests.run_datatype_candidate import runtime_summary

            result["precision_config"] = generator.model.precision_config
            result["runtime_summary"] = runtime_summary(generator)
        result["construction_and_preparation_seconds"] = time.perf_counter() - construction_started
        result["preparation_seconds"] = generator.prepare_seconds
        result["allocator"] = {}
        for kind in ("DRAM", "L1", "TRACE"):
            view = ttnn.get_memory_view(mesh, getattr(ttnn.BufferType, kind))
            result["allocator"][kind] = {
                key: getattr(view, key)
                for key in (
                    "num_banks",
                    "total_bytes_per_bank",
                    "total_bytes_allocated_per_bank",
                    "total_bytes_free_per_bank",
                    "largest_contiguous_bytes_free_per_bank",
                )
            }
        shared = SharedGenerator(generator)
        reference = ROOT / "readiness_aime24_chat.refpt"
        for name, module, function in (
            ("prefill", prefill_check, prefill_check.run_prefill_check),
            ("teacher", teacher_check, teacher_check.run_teacher_forcing),
        ):
            if name in args.checks:
                print("FULL_CHECK_START", name, flush=True)
                with patch.object(module, "_import_build_generator", return_value=lambda **kw: shared):
                    stats = function(model_dir=ROOT, reference_path=reference, mesh_device=mesh)
                result[name] = stats
                save()
                print("FULL_CHECK_DONE", name, stats, flush=True)
        if "qualitative" in args.checks:
            controls = json.loads((DOC / "qualitative_hf_outputs.json").read_text())
            metadata = json.loads((DOC / "qualitative_prompt_format.json").read_text())
            assert metadata["model_id"] == MODEL_ID and metadata["revision"] == REVISION
            assert len(controls) == 6 and metadata["max_new_tokens"] == 128
            qualitative = []
            for control in controls:
                output = evidence_dir / "autoregressive" / str(control["id"])
                output.mkdir(parents=True, exist_ok=True)
                prompt = output / "rendered_prompt.txt"
                prompt.write_text(control["rendered_prompt"])
                ids = generator.tokenizer.encode(control["rendered_prompt"], add_special_tokens=False)
                assert ids == control["prompt_token_ids"]

                class ExactTokenizer:
                    def encode(self, text, **kwargs):
                        assert text == control["rendered_prompt"].strip()
                        return ids

                    def decode(self, values, **kwargs):
                        return generator.tokenizer.decode(values, **kwargs)

                def hf_control(**kwargs):
                    assert kwargs["hf_model_id"] == MODEL_ID
                    assert kwargs["prompt_token_ids"] == ids and kwargs["max_new_tokens"] == 128
                    return control["generated_token_ids"]

                with (
                    patch.object(autoregressive, "_import_build_generator", return_value=lambda **kw: shared),
                    patch.object(autoregressive.AutoTokenizer, "from_pretrained", return_value=ExactTokenizer()),
                    patch.object(autoregressive, "_hf_generate_greedy", side_effect=hf_control),
                ):
                    paths = autoregressive.run_autoregressive(
                        model_dir=ROOT,
                        hf_model_id=MODEL_ID,
                        prompt_file=prompt,
                        mesh_device=mesh,
                        output_dir=output,
                        max_new_tokens=128,
                    )
                qualitative.append({key: str(value) for key, value in paths.items()})
                result["qualitative"] = qualitative
                save()
                print("QUALITATIVE_DONE", control["id"], paths["tt_completion"].read_text(), flush=True)
        if "perf" in args.checks:
            prompt = [42] * 128
            measurements = []
            for mode in ("sample", "argmax"):
                for repeat in range(4):
                    before = dict(generator.counters)
                    started = time.perf_counter()
                    tokens = generator.generate(prompt, 128, sampler_mode=mode, stop_on_eos=False)
                    elapsed = time.perf_counter() - started
                    measurements.append(
                        dict(
                            mode=mode,
                            repeat=repeat,
                            generated=len(tokens),
                            ttft_ms=generator.last_ttft_seconds * 1000,
                            elapsed_seconds=elapsed,
                            decode_ms=generator.last_decode_seconds / 127 * 1000,
                            decode_tps=127 / generator.last_decode_seconds,
                            counters={key: value - before.get(key, 0) for key, value in generator.counters.items()},
                            tokens=tokens,
                        )
                    )
                    print(
                        "PERF", {key: value for key, value in measurements[-1].items() if key != "tokens"}, flush=True
                    )
                    result["perf"] = measurements
                    save()
            # No-sampling comparison: device tokens and positions stay bound;
            # this is a logits-only repeated-decode throughput measurement.
            generator.generate(prompt, 1)
            started = time.perf_counter()
            for _ in range(127):
                generator._execute("decode")
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - started
            result["logits_only"] = dict(
                steps=127,
                elapsed_seconds=elapsed,
                decode_tps=127 / elapsed,
                decode_ms=elapsed / 127 * 1000,
                boundary="model decode through softcapped vocab-sharded logits; no sampling/readback",
            )
            result["token_out_no_readback"] = []
            for mode in ("sample", "argmax"):
                for repeat in range(4):
                    generator.generate(prompt, 1, stop_on_eos=False)
                    before = dict(generator.counters)
                    started = time.perf_counter()
                    generator.decode_steps(127, sampler_mode=mode)
                    ttnn.synchronize_device(mesh)
                    elapsed = time.perf_counter() - started
                    counters = {key: value - before.get(key, 0) for key, value in generator.counters.items()}
                    assert all(
                        counters.get(key, 0) == 0
                        for key in (
                            "token_refreshes",
                            "position_refreshes",
                            "rope_refreshes",
                            "page_table_refreshes",
                            "token_readbacks",
                            "async_token_readbacks",
                            "full_logits_readbacks",
                            "captures",
                            "synchronizations",
                        )
                    ), counters
                    final_token = generator._read_tokens()[0]
                    expected = next(row["tokens"][-1] for row in measurements if row["mode"] == mode)
                    assert final_token == expected, (final_token, expected)
                    result["token_out_no_readback"].append(
                        dict(
                            mode=mode,
                            repeat=repeat,
                            steps=127,
                            ttft_ms=generator.last_ttft_seconds * 1000,
                            decode_ms=elapsed / 127 * 1000,
                            decode_tps=127 / elapsed,
                            final_token=final_token,
                            equals_output_returning_path=True,
                            counters=counters,
                            boundary_synchronizations=1,
                            validation_readbacks_outside_timing=1,
                        )
                    )
                    save()
            result["prefill_lengths"] = []
            for length in (33, 127, 128, 129, 4095, 4096, 4097):
                for repeat in range(3):
                    token = generator.generate([42] * length, 1, stop_on_eos=False)
                    result["prefill_lengths"].append(
                        dict(
                            logical_length=length,
                            repeat=repeat,
                            ttft_ms=generator.last_ttft_seconds * 1000,
                            first_token=token[0],
                        )
                    )
            save()
            if generator.trace_reset:
                result["reset_dispatch_control"] = []
                original_clear = generator.clear_cache_on_generate
                generator.clear_cache_on_generate = True
                for traced in (True, False, True):
                    generator.trace_reset = traced
                    for repeat in range(3):
                        generator.generate(prompt, 1, stop_on_eos=False)
                        result["reset_dispatch_control"].append(
                            dict(
                                traced=traced,
                                repeat=repeat,
                                ttft_ms=generator.last_ttft_seconds * 1000,
                            )
                        )
                generator.trace_reset = True
                generator.clear_cache_on_generate = original_clear
                save()
            from models.autoports.meta_models_muse_glimmer_30b.tests.check_generate_cache_reuse import (
                check_cache_reset_equivalence,
            )

            result["cache_reset_equivalence"] = check_cache_reset_equivalence(generator)
            save()
            by_mode = {
                mode: [row["tokens"] for row in measurements if row["mode"] == mode] for mode in ("sample", "argmax")
            }
            result["greedy_repeat_determinism"] = {
                mode: all(row == rows[0] for row in rows) for mode, rows in by_mode.items()
            }
            assert all(result["greedy_repeat_determinism"].values())
            different = next(
                (i for i, (a, b) in enumerate(zip(by_mode["sample"][0], by_mode["argmax"][0])) if a != b), None
            )
            result["greedy_strategy_first_divergence"] = different
            if different is not None:
                assert different > 0
                prefix = generator.generate(prompt, different, stop_on_eos=False)
                assert prefix == by_mode["sample"][0][:different]
                generator._execute("decode")
                generator._execute("sample")
                sampled = generator._read_tokens()[0]
                generator._execute("argmax")
                argmax = generator._read_tokens()[0]
                logits = generator._read_logits(generator.logits).flatten()
                maximum = logits.max().item()
                result["greedy_tie_control"] = dict(
                    index=different,
                    sampled=sampled,
                    argmax=argmax,
                    sampled_logit=logits[sampled].item(),
                    argmax_logit=logits[argmax].item(),
                    maximum=maximum,
                    tied_maximum_count=int((logits == maximum).sum()),
                )
                save()
                assert logits[sampled].item() == logits[argmax].item() == maximum, result["greedy_tie_control"]
        result["counters"] = dict(generator.counters)
        result["traces"] = {str(key): str(value) for key, value in generator.traces.items()}
        accuracy_stats = [stat for name in ("prefill", "teacher") for stat in result.get(name, [])]
        result["accuracy_gate_pass"] = (
            all(
                stat["top1"] >= 0.90
                and stat["top5"] >= 0.98
                and stat["top100"] == 1.0
                and stat["total"] == 100
                and stat["k"] == 100
                for stat in accuracy_stats
            )
            if accuracy_stats
            else None
        )
        save()
        assert result["accuracy_gate_pass"] is not False, "Full-model top-k gate failed; see saved statistics"
    except BaseException:
        traceback.print_exc()
        save()
        raise
    finally:
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checks", nargs="+", choices=("prefill", "teacher", "qualitative", "perf"), default=["prefill", "teacher"]
    )
    parser.add_argument("--output", type=Path, default=DOC / "full_model_checks.json")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--terminal-policy", default="{}")
    parser.add_argument("--eager-reset", action="store_true")
    parser.add_argument("--clear-cache-on-generate", action="store_true")
    main(parser.parse_args())
