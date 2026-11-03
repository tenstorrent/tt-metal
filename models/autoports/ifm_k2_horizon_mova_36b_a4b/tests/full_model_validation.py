# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-layer readiness, shared prompts and physical context gate on one owner.

Standard readiness runners borrow the prepared generator. Autoregressive HF
controls are fresh pinned artifacts, checked for exact prompt/generation identity.
No serving adapter or vLLM runtime is imported.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_autoregressive as autoregressive
from readiness_check import run_prefill_check as prefill
from readiness_check import run_teacher_forcing as teacher
from transformers import AutoTokenizer

import ttnn

from ..tt.generator import build_generator
from ..tt.model import HF_MODEL, HF_REVISION
from .full_model_performance import memory

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/full_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", action="store_true")
    ap.add_argument("--artifact-dir", type=Path, default=DOC)
    ap.add_argument("--context", type=int, default=524288)
    ap.add_argument("--output", type=Path, default=DOC / "full_validation.json")
    args = ap.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    report = dict(
        context=args.context,
        layers=48,
        source_sha256={
            name: hashlib.sha256((ROOT / "tt" / name).read_bytes()).hexdigest()
            for name in ("model.py", "generator.py", "multichip_decoder.py")
        },
    )
    output = args.output

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        gen = build_generator(ROOT, mesh, max_seq_len=args.context)
        report["constructed_memory"] = memory(mesh)
        print("CONSTRUCTED", report["constructed_memory"], flush=True)
        gen.capture()
        report["captured_memory"] = memory(mesh)
        print("CAPTURED", report["captured_memory"], flush=True)
        save()
        factory = lambda **_: gen
        with patch.object(gen, "teardown", lambda: None):
            with patch.object(prefill, "_import_build_generator", lambda _: factory):
                report["prefill"] = prefill.run_prefill_check(
                    model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                )
            gen.reset()
            with patch.object(teacher, "_import_build_generator", lambda _: factory):
                report["teacher_forcing"] = teacher.run_teacher_forcing(
                    model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                )
            save()
            for mode in ("prefill", "teacher_forcing"):
                assert all(
                    row["top1"] >= 0.90 and row["top5"] >= 0.98 and row["top100"] == 1 for row in report[mode]
                ), report[mode]
            original_tokenizer = AutoTokenizer.from_pretrained

            def tokenizer(*a, **kw):
                kw.setdefault("revision", HF_REVISION)
                value = original_tokenizer(*a, **kw)
                original_encode = value.encode

                # run_autoregressive receives an already-rendered chat prompt.
                def encode(text, **opts):
                    # The stock runner strips its file input. The chat template
                    # ends with a significant newline after the thinking marker.
                    assert text == rendered_prompt.strip()
                    return original_encode(rendered_prompt, **{**opts, "add_special_tokens": False})

                value.encode = encode
                return value

            controls = [(DOC / "qualitative" / f"prompt_{i}", 128) for i in range(6)]
            controls.append((DOC / "qualitative_extended/prompt_3", 512))
            report["autoregressive"] = []
            for source, length in controls:
                control = json.loads((source / "hf.json").read_text())
                rendered_prompt = (source / "prompt.txt").read_text()
                assert control["model"] == HF_MODEL and control["revision"] == HF_REVISION
                assert control["max_new_tokens"] == length and control["prompt_mode"] == "chat"

                def hf_control(*, hf_model_id, prompt_token_ids, max_new_tokens, device):
                    assert hf_model_id == HF_MODEL and prompt_token_ids == control["prompt_token_ids"]
                    assert max_new_tokens == control["max_new_tokens"]
                    return control["token_ids"]

                target = args.artifact_dir / "autoregressive" / (source.parent.name + "_" + source.name)
                gen.reset()
                with (
                    patch.object(autoregressive, "_import_build_generator", lambda _: factory),
                    patch.object(autoregressive, "_hf_generate_greedy", hf_control),
                    patch.object(AutoTokenizer, "from_pretrained", tokenizer),
                ):
                    paths = autoregressive.run_autoregressive(
                        model_dir=ROOT,
                        hf_model_id=HF_MODEL,
                        prompt_file=source / "prompt.txt",
                        mesh_device=mesh,
                        output_dir=target,
                        max_new_tokens=length,
                    )
                provenance = dict(
                    source=str(source / "hf.json"),
                    source_sha256=hashlib.sha256((source / "hf.json").read_bytes()).hexdigest(),
                    reused_fresh_pinned_control=True,
                    runner="readiness_check.run_autoregressive",
                    prompt_reencoding="restore template trailing newline stripped by runner; add_special_tokens=False; exact token IDs asserted",
                )
                (target / "control_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
                report["autoregressive"].append({key: str(value) for key, value in paths.items()})
                save()
        if args.capacity:
            assert args.context == 524288
            base = json.loads((DOC / "reference_metadata.json").read_text())["prompt_tokens"]
            length = 524275
            prompt = (base * ((length + len(base) - 1) // len(base)))[:length]
            original = gen._replay_prefill
            progress = [0]

            def observed(*a, **kw):
                value = original(*a, **kw)
                progress[0] += 1
                if progress[0] % 4 == 0:
                    ttnn.synchronize_device(mesh)
                    print("CAPACITY_PREFILL_CHUNKS", progress[0], flush=True)
                return value

            gen._replay_prefill = observed
            gen.reset()
            began = time.perf_counter()
            tokens = gen.generate(prompt, 13, stop_on_eos=False)
            report["nonaligned_capacity"] = dict(
                prompt_length=length,
                generated_tokens=tokens,
                seconds=time.perf_counter() - began,
                counters=dict(gen.counters),
            )
            save()
            final = gen.prefill_forward(
                torch.tensor([prompt + [100] * 13]),
                page_table=gen.page_table_host,
                kv_cache=gen.kv_cache,
                start_pos=[length],
                prompt_lens=[524288],
                sampling_mode="device",
            )
            report["full_extent_continuation"] = dict(start=length, end=524288, token=int(final[0, 0]))
            report["final_memory"] = memory(mesh)
            save()
        if args.context == 524288:
            token = gen.decode_forward(
                torch.tensor([[100]]),
                torch.tensor([524287]),
                page_table=gen.page_table_host,
                kv_cache=gen.kv_cache,
                sampling_mode="device",
            )
            report["last_position_trace_smoke"] = dict(
                position=524287,
                token=int(token[0, 0]),
                scope="allocation/execution boundary; cache outside previously filled prefixes is zero in non-capacity runs",
            )
        report["prefill_trace_ids"] = {str(k): str(v) for k, v in gen.prefill_traces.items()}
        report["trace_counters"] = dict(gen.counters)
        report["status"] = "passed"
        save()
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
