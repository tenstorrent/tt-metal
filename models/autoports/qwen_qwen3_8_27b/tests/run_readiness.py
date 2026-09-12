# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the standard readiness entrypoints on the target TP4 mesh.

A fresh pinned HF reference is also the exact-prompt autoregressive HF control;
reuse is checked against token IDs rather than launching a second CPU model.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.common.readiness_check import run_autoregressive as autoregressive
from models.common.readiness_check import run_prefill_check as prefill
from models.common.readiness_check import run_teacher_forcing as teacher
from models.common.readiness_check.schema import load_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reference", type=Path, default=Path("models/autoports/qwen_qwen3_8_27b/readiness_aime24_chat.refpt")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", choices=["prefill", "decode", "autoregressive", "all"], default="all")
    p.add_argument("--qualitative", action="store_true")
    p.add_argument("--qualitative-reference", default="hf_qualitative.json")
    p.add_argument("--qualitative-output", default="tt_qualitative.json")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--context", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(8)
    root = Path("models/autoports/qwen_qwen3_8_27b")
    ref = load_reference(a.reference)
    metadata = json.loads(a.reference.with_suffix(".metadata.json").read_text())
    assert metadata["hf_model_id"] == "Qwen/Qwen3.8-27B" and metadata["chat_template"]
    assert metadata["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert metadata["generation_length"] == 100 and metadata["top_k"] == 100
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(root, mesh)
        factory = lambda *_args, **_kwargs: gen
        report = dict(reference=str(a.reference), hf_metadata=metadata)
        if a.only in ("prefill", "all"):
            with patch.object(prefill, "_import_build_generator", return_value=factory):
                report["prefill"] = prefill.run_prefill_check(
                    model_dir=root, reference_path=a.reference, mesh_device=mesh
                )
            a.output.write_text(json.dumps(report, indent=2) + "\n")
        if a.only in ("decode", "all"):
            with patch.object(teacher, "_import_build_generator", return_value=factory):
                report["decode"] = teacher.run_teacher_forcing(
                    model_dir=root, reference_path=a.reference, mesh_device=mesh
                )
            a.output.write_text(json.dumps(report, indent=2) + "\n")
        if a.only in ("autoregressive", "all"):
            entry = ref.entries[0]
            prompt_file = root / "doc/full_model/aime24_rendered.txt"
            prompt_file.write_text(entry.prompt_text)

            def exact_control(*, hf_model_id, prompt_token_ids, max_new_tokens, device):
                assert prompt_token_ids == entry.prompt_tokens[0].tolist()
                assert max_new_tokens == entry.num_generated
                return entry.generated_tokens[0].tolist()

            with (
                patch.object(autoregressive, "_import_build_generator", return_value=factory),
                patch.object(autoregressive, "_hf_generate_greedy", side_effect=exact_control),
            ):
                artifacts = autoregressive.run_autoregressive(
                    model_dir=root,
                    hf_model_id=str(gen.model.snapshot),
                    prompt_file=prompt_file,
                    mesh_device=mesh,
                    output_dir=root / "doc/full_model/autoregressive",
                    max_new_tokens=entry.num_generated,
                )
            report["autoregressive"] = {k: str(v) for k, v in artifacts.items()}
            a.output.write_text(json.dumps(report, indent=2) + "\n")
        for key in ("prefill", "decode"):
            for row in report.get(key, []):
                assert row["top5"] >= 0.98 and row["top100"] == 1.0, (key, row)
        if a.qualitative:
            from models.autoports.qwen_qwen3_8_27b.tests.tt_qualitative import run

            report["qualitative"] = str(run(gen, root, a.qualitative_reference, output_name=a.qualitative_output))
            a.output.write_text(json.dumps(report, indent=2) + "\n")
        if a.benchmark:
            entry = ref.entries[0]
            prompt = entry.prompt_tokens[0].tolist()
            forced = entry.generated_tokens[0].tolist()
            for _ in range(2):
                gen.generate(prompt, len(forced), next_input=lambda step, predicted: forced[step])
            report["warmed_teacher_forcing_perf"] = gen.last_perf
            # The fixed-length fixture is used only for latency; quality above
            # uses intact chat templates and exact HF prompt IDs.
            for _ in range(2):
                gen.generate(prompt[:128], 128)
            report["warmed_token_out_perf_s128_g128"] = gen.last_perf
            assert not gen.last_perf["steady_state_counters"].get("full_logits_readbacks", 0)
            for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
                assert not gen.last_perf["steady_state_counters"].get(key, 0)
            a.output.write_text(json.dumps(report, indent=2) + "\n")
        if a.context:
            import time

            gen._ensure_cache(1, gen.model.context)
            report["final_context"] = []
            for length in (gen.model.context - 1, gen.model.context):
                begin = time.perf_counter()
                count = gen.model.context - length + 1
                tokens = gen.generate([1596] * length, count)
                assert len(tokens) == count
                position = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).item()
                assert position == gen.model.context
                report["final_context"].append(
                    dict(length=length, generated=count, position=position, seconds=time.perf_counter() - begin)
                )
                a.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
