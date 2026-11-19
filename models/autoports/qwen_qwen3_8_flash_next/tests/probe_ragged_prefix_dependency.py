# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct generator counterpart of pinned chunked-prefill prefix dependence.

Device owner only, accepted mesh environment, serialized::

  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    timeout 1800 python_env/bin/python -m \
    models.autoports.qwen_qwen3_8_flash_next.tests.probe_ragged_prefix_dependency \
    --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/ragged_prefix_dependency.json

Full48, B2, context2048. Two pinned needle prompts (128/384 filler words)
exercise a needle retained in the prefix and one written after a nonzero start.
One public prefill submission carries distinct absolute continuation starts.
Compare uninterrupted versus resumed terminal/decode logits at identical B2
geometry and forced token history, using the installed resume thresholds.
Sampling stays traced on device; reference tokens are explicit teacher forcing.

This is numerical cache/position evidence, not a semantic recall verdict or
a server test. No answer text is scored. Any later passphrase-quality verdict
requires the qualitative-check matched HF controls (exact saved prompts/chat
template/settings; ten matched seeds and blind scoring before debugging).
"""

import argparse
import ast
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import torch

PIN = "042a94863ad27d88207bcf71abdf67122c6f4c27"


def pinned_prompt_builder(path):
    """Read only the pinned pure prompt constructor; never import server tests."""
    tree = ast.parse(path.read_text())
    wanted = {"NEEDLE_POSITION", "FILLER_WORDS"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_needle_prompt")
        or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))
    ]
    assert len(nodes) == 3, "Pinned prompt constructor changed"
    namespace = {"random": random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["_needle_prompt"]


def continuation_starts(lengths):
    # Leave the first request's needle in its retained prefix, and write the
    # second request's much later needle in a continuation from nonzero start.
    first = lengths[0] - 17
    first -= int(first % 32 == 0)
    starts = [first, 68]
    assert all(0 < start < end and start % 32 for start, end in zip(starts, lengths, strict=True))
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, default=Path("/home/ttuser/qwen38-flash/vllm-tt-plugin"))
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "passed": False,
        "scope": "all48 numerical ragged prefix dependency; no semantic verdict",
        "plugin_commit": PIN,
        "semantic_recall_verdict": "unassessed: matched HF controls required",
    }
    mesh = gen = None
    started = time.monotonic()
    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore accepted mesh descriptor"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
        import subprocess

        assert subprocess.check_output(["git", "-C", str(args.plugin), "rev-parse", "HEAD"], text=True).strip() == PIN
        from readiness_check.run_prefill_resume_check import DEFAULT_THRESHOLDS, compare_logits, evaluate_case

        import ttnn
        from models.common.sampling.sampling_params import SamplingParams

        from ..tt.generator import DecodeSubmission, build_generator
        from .probe_full_model_serving import packed

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        options = dict(max_seq_len=2048, max_batch_size=2)
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator(Path(__file__).resolve().parents[1], mesh, **options)
        source = args.plugin / "tests/tt/test_chunked_prefill.py"
        builder = pinned_prompt_builder(source)
        raw = [builder(128, "cobalt-heron-42", 0xC401, "Solo.\n"), builder(384, "cobalt-heron-256", 0xC402, "Long.\n")]
        chat = bool(getattr(gen.tokenizer, "chat_template", None))
        rendered = [
            (
                gen.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
                )
                if chat
                else text
            )
            for text in raw
        ]
        prompts = [gen.tokenizer.encode(text, add_special_tokens=False) for text in rendered]
        lengths = list(map(len, prompts))
        assert max(lengths) + 24 <= gen.capacity, "Pinned prompts exceed this test's allocation; do not truncate"
        starts = continuation_starts(lengths)
        report.update(
            prompt_source=str(source),
            prompt_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            prompt_mode="chat" if chat else "completion",
            tokenizer_class=type(gen.tokenizer).__name__,
            rendered_prompts=rendered,
            prompt_token_ids=prompts,
            lengths=lengths,
            absolute_starts=starts,
            thresholds=DEFAULT_THRESHOLDS,
            token_budget=24,
            sampling={"temperature": 0, "top_k": 1, "top_p": 1},
        )
        gen.prepare()
        program_count = mesh.num_program_cache_entries()
        tables = torch.arange(gen.caches[0].capacity // 32, dtype=torch.int32).reshape(1, -1).repeat(2, 1)
        slots = (0, 1)
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

        def logits_snapshot():
            submission = DecodeSubmission(gen.sample_logits, slots, (0, 1), False)
            host = gen.read_decode_output(submission)
            return gen.process_decode_output_host(host)[:, 0].clone()

        def run(resumed, teacher=None):
            gen.reset()
            if resumed:
                gen.prefill_forward(
                    packed(prompts),
                    page_table=tables,
                    kv_cache=gen.caches,
                    prompt_lens=starts,
                    start_pos=[0, 0],
                    slots=slots,
                )
            first = gen.prefill_forward(
                packed(prompts),
                page_table=tables,
                kv_cache=gen.caches,
                prompt_lens=lengths,
                start_pos=starts if resumed else [0, 0],
                slots=slots,
                sampling_params=params,
            )
            tokens, logits = [first.flatten().clone()], [logits_snapshot()]
            for step in range(1, 24):
                previous = teacher[step - 1] if teacher is not None else tokens[-1]
                prediction = gen.decode_forward(
                    previous,
                    [length + step - 1 for length in lengths],
                    slots=slots,
                    page_table=tables,
                    kv_cache=gen.caches,
                    sample_on_device=True,
                )
                tokens.append(prediction.clone())
                logits.append(logits_snapshot())
            return torch.stack(tokens), torch.stack(logits)

        control_tokens, control_logits = run(False)
        resumed_tokens, resumed_logits = run(True, teacher=control_tokens)
        report["control_tokens"], report["resumed_teacher_forced_tokens"] = (
            control_tokens.T.tolist(),
            resumed_tokens.T.tolist(),
        )
        report["cases"] = []
        for row in range(2):
            stats = compare_logits(control_logits[:, row].unsqueeze(0), resumed_logits[:, row].unsqueeze(0))
            reasons = evaluate_case(stats, DEFAULT_THRESHOLDS)
            report["cases"].append(
                {
                    "slot": row,
                    "start": starts[row],
                    "length": lengths[row],
                    **stats,
                    "pass": not reasons,
                    "reasons": reasons,
                }
            )
        report["program_cache_entries"] = {"prepared": program_count, "final": mesh.num_program_cache_entries()}
        report["counters"] = dict(gen.counters)
        assert report["program_cache_entries"]["final"] == program_count, "Continuation compiled after preparation"
        assert all(case["pass"] for case in report["cases"]), report["cases"]
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
