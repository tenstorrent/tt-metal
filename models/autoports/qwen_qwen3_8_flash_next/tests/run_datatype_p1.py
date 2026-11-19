# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Installed P1 runner with the pinned original-HF mmap loader and TP4 mesh.

The package's scoring, noise gates and result schema remain unchanged. This
adapter supplies construction and checkpoint loading, resets persistent sampling
for teacher forcing, and excludes sampled output after the first EOS.
"""

import json
import sys
from functools import wraps
from pathlib import Path


def trim_after_eos(tokens, stop_ids):
    """Exclude post-completion output from fixed-token-count generator calls."""
    for index, token in enumerate(tokens):
        if int(token) in stop_ids:
            return tokens[: index + 1]
    return tokens


def main():
    from readiness_check import run_precision_reasoning as p1
    from readiness_check.build_kwargs import add_build_kwargs_args, parse_build_kwargs
    from readiness_check.mesh_device import add_mesh_device_args

    parser = p1.build_parser()
    add_build_kwargs_args(parser)
    add_mesh_device_args(parser)
    args = p1.finalize_args(parser.parse_args())
    args.model_dir = args.model_dir.resolve()
    common = parse_build_kwargs(args)
    assert args.mesh_device == "P300x2" and args.fabric_config == "FABRIC_1D_RING"

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GenerationConfig

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests import full_model_hf_reference as exact_hf
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator

    torch.set_num_threads(8)
    exact_hf.EVIDENCE_DIR = args.model_dir / "doc/datatype_sweep/p1_hf_loader"
    snapshot = Path(snapshot_download(exact_hf.MODEL_ID, revision=exact_hf.REVISION, local_files_only=True))
    if args.hf_model != exact_hf.MODEL_ID and Path(args.hf_model).resolve() != snapshot.resolve():
        raise ValueError("P1 tokenizer and original-HF control must use the exact pinned checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    eos = GenerationConfig.from_pretrained(snapshot, local_files_only=True).eos_token_id
    stop_ids = {int(token) for token in (eos if isinstance(eos, list) else [eos])}
    text_stop_events = []
    teacher_forcing_traces = []

    def build(*, model_dir, mesh_device, **kwargs):
        generator = build_generator(model_dir, mesh_device, **{**common, **kwargs})
        generate = generator.generate

        @wraps(generate)
        def p1_generate(*args, **options):
            # Sampling is persistent in the public generator. P1 first samples
            # reasoning text, then asks for argmax teacher-forcing predictions.
            # Reset only that latter call, preserving the runner's text sampling.
            if options.get("next_input") is not None:
                generator.set_sampling_params()
                before = dict(generator.counters)
                output = generate(*args, **options)
                delta = {key: value - before[key] for key, value in generator.counters.items()}
                teacher_forcing_traces.append(
                    {"config_id": generator.model.precision["config_id"], "counter_delta": delta}
                )
                assert delta["model_replays"] > 0, "P1 teacher forcing did not replay model traces"
                return output
            output = generate(*args, **options)
            # The public generator intentionally executes a fixed token count.
            # P1 needs one completed answer, matching HF's EOS stopping. Tokens
            # after EOS can otherwise make valid answers look like long loops.
            answer = trim_after_eos(output, stop_ids)
            text_stop_events.append(
                {"raw_tokens": len(output), "answer_tokens": len(answer), "eos_trimmed": len(answer) < len(output)}
            )
            return answer

        generator.generate = p1_generate
        return generator

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
    try:
        code = p1.run(
            args,
            build_generator=build,
            mesh_device=mesh,
            tokenizer=tokenizer,
            hf_loader=lambda _: exact_hf.load_exact_text_model(),
            # This loader demand-pages the original tensors, including exact
            # PLE row reads; it does not require the dense checkpoint in RAM.
            hf_fit=lambda _: None,
        )
        return code
    finally:
        ttnn.close_mesh_device(mesh)
        record = args.model_dir / "doc/datatype_sweep/p1_last_command.json"
        record.write_text(
            json.dumps(
                {
                    "command": [sys.executable, *sys.argv],
                    "common_build_kwargs": common,
                    "runner": str(Path(p1.__file__).resolve()),
                    "hf_loader": str(Path(exact_hf.__file__).resolve()),
                    "hf_snapshot": str(snapshot),
                    "teacher_forcing_sampling": "explicit greedy reset before next_input calls",
                    "generated_text_stop_policy": "retain first EOS; exclude subsequent fixed-count output",
                    "generated_text_stop_events": text_stop_events,
                    "teacher_forcing_trace_evidence": teacher_forcing_traces,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
