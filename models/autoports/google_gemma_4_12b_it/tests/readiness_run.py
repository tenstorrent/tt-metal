# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the plugin's readiness runners (``readiness_check.run_prefill_check`` / ``run_teacher_forcing`` /
``run_autoregressive``) on one generator, in one process.

The runners are used unchanged through their programmatic entry points. Two things differ from their CLIs:
* the mesh is opened with ``open_qb2_mesh`` (ring fabric with the 8192-byte router packets the decoder stages were
  measured with) instead of ``open_readiness_mesh_device`` (fabric default packets);
* ``tt/generator.py::build_generator`` is called once and the same generator serves every runner (its traces are
  released once at the end), instead of reloading the 48-layer model per runner.
The stock CLI path (``python -m readiness_check.run_teacher_forcing --mesh-device P300x2 --fabric-config
FABRIC_1D_RING``) is exercised separately (doc/full_model/README.md).

  python models/autoports/google_gemma_4_12b_it/tests/readiness_run.py --checks prefill,tf,ar \
      --reference bringup/references/gemma4_12b_it_aime24_chat.refpt
"""

import argparse
import json
import time
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="prefill,tf,ar")
    ap.add_argument("--reference", default="bringup/references/gemma4_12b_it_aime24_chat.refpt")
    ap.add_argument("--hf-model", default="google/gemma-4-12B-it")
    ap.add_argument(
        "--ar-prompts",
        default="raw_story=default",
        help="comma-separated name=prompt_file for run_autoregressive ('default' = the plugin's raw prompt); "
        "each writes <ar-output>/<name>/",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ar-output", default=str(MODEL_DIR / "doc/full_model/readiness_autoregressive"))
    ap.add_argument("--build-kwargs", default="{}", help="JSON kwargs for build_generator")
    ap.add_argument(
        "--hf-ar-cache",
        default="",
        help="comma-separated dirs of earlier run_autoregressive outputs (<name>/autoregressive_meta.json): an HF greedy "
        "completion with the same model id, prompt token ids and max_new_tokens is reused instead of regenerated (the "
        "runner loads the full HF model on the CPU, which this host's free memory cannot hold); others run HF as usual",
    )
    ap.add_argument("--out", default="", help="JSON summary path")
    a = ap.parse_args()
    torch.set_num_threads(16)

    import readiness_check.run_autoregressive as rar
    import readiness_check.run_prefill_check as rpc
    import readiness_check.run_teacher_forcing as rtf

    from models.autoports.google_gemma_4_12b_it.tt.generator import build_generator
    from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh

    checks = a.checks.split(",")
    summary = {"checks": checks, "build_kwargs": json.loads(a.build_kwargs)}
    if a.hf_ar_cache:
        metas = [
            (str(m), json.loads(m.read_text()))
            for d in a.hf_ar_cache.split(",")
            for m in sorted(Path(d).glob("*/autoregressive_meta.json"))
        ]
        hf_generate = rar._hf_generate_greedy
        summary["hf_ar_cache_hits"] = []

        def cached_hf_generate(*, hf_model_id, prompt_token_ids, max_new_tokens, device):
            for path, m in metas:
                if (
                    m["hf_model_id"] == hf_model_id
                    and m["prompt_token_ids"] == list(prompt_token_ids)
                    and m["max_new_tokens"] == max_new_tokens
                ):
                    summary["hf_ar_cache_hits"].append(path)
                    print(f"HF completion reused from {path}", flush=True)
                    return list(m["hf"]["token_ids"])
            return hf_generate(
                hf_model_id=hf_model_id, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens, device=device
            )

        rar._hf_generate_greedy = cached_hf_generate
    mesh = open_qb2_mesh(trace_region_size=0)
    try:
        t0 = time.time()
        gen = build_generator(MODEL_DIR, mesh, **summary["build_kwargs"])
        summary["build_and_prepare_s"] = round(time.time() - t0, 1)
        print(f"generator ready in {summary['build_and_prepare_s']} s", flush=True)
        real_teardown = gen.teardown
        gen.teardown = lambda: None  # one generator for every runner; released at the end

        def shared_build(_model_dir):
            return lambda **kw: gen

        for mod in (rpc, rtf, rar):
            mod._import_build_generator = shared_build
        ref = Path(a.reference).resolve()
        if "prefill" in checks:
            before = dict(gen.counters)
            per = rpc.run_prefill_check(model_dir=MODEL_DIR, reference_path=ref, mesh_device=mesh)
            summary["prefill"] = per
            summary["prefill_counters"] = {k: gen.counters[k] - before[k] for k in before}
            gen.reset()
        if "tf" in checks:
            before = dict(gen.counters)
            per = rtf.run_teacher_forcing(model_dir=MODEL_DIR, reference_path=ref, mesh_device=mesh)
            summary["teacher_forcing"] = per
            summary["tf_counters"] = {k: gen.counters[k] - before[k] for k in before}
            gen.reset()
        if "ar" in checks:
            before = dict(gen.counters)
            summary["autoregressive"] = {}
            for item in a.ar_prompts.split(","):
                name, path = item.split("=")
                prompt = rar.DEFAULT_PROMPT_FILE if path == "default" else Path(path)
                paths = rar.run_autoregressive(
                    model_dir=MODEL_DIR,
                    hf_model_id=a.hf_model,
                    prompt_file=prompt.resolve(),
                    mesh_device=mesh,
                    output_dir=(Path(a.ar_output) / name).resolve(),
                    max_new_tokens=a.max_new_tokens,
                )
                summary["autoregressive"][name] = {k: str(v) for k, v in paths.items()}
                gen.reset()
            summary["ar_counters"] = {k: gen.counters[k] - before[k] for k in before}
        if "qual" in checks:
            from models.autoports.google_gemma_4_12b_it.tests.qualitative_suite import tt_run

            before = dict(gen.counters)
            summary["qualitative"] = [
                {k: r[k] for k in ("id", "first_divergence_vs_hf", "batch_equals_single", "tokens_tt_hf")}
                for r in tt_run(gen)
            ]
            summary["qual_counters"] = {k: gen.counters[k] - before[k] for k in before}
        summary["counters_total"] = dict(gen.counters)
        real_teardown()
    finally:
        close_qb2_mesh(mesh)
    print("SUMMARY", json.dumps(summary, default=str), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
