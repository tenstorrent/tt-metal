#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the stage-07 datatype-sweep table (markdown + JSON) from doc/optimize/perf_runs/sweep_*.json.

    $MM3_PY scripts/dtype_sweep_table.py            # prints markdown, writes doc/optimize/dtype_sweep.{md,json}
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
RUNS = MODEL_DIR / "doc" / "optimize" / "perf_runs"
OUT = MODEL_DIR / "doc" / "optimize"

BAR_LATENT_PCC = 0.98
BAR_LOGMEL_DB = 2.0
CONTROL_LOGMEL_DB = 1.115  # stage 06: the vendored vocoder in bf16 on the golden latents (torch-vs-torch control)


def main():
    rows = []
    for p in sorted(RUNS.glob("sweep_*.json")) + sorted(RUNS.glob("before.json")) + sorted(RUNS.glob("after.json")):
        r = json.loads(p.read_text())
        g = r.get("golden_replay")
        if not g:
            continue
        lk = r["load_kwargs"]
        t = g["timings"]
        rows.append(
            {
                "label": r["label"],
                "llm_policy": lk.get(
                    "llm_policy", {"functional": "functional", "optimized": "optimized"}.get(lk.get("dtype_policy"))
                ),
                "dit_dtype": lk.get("dit_dtype", "bf16" if lk.get("dtype_policy") == "functional" else "bfp8"),
                "frame_hiddens_pcc": g["frame_hiddens_pcc"],
                "frame_hiddens_pcc_min": g["frame_hiddens_pcc_min"],
                "latent_pcc": g["latent_pcc"],
                "log_mel_rms_db": g["log_mel_rms_db"],
                "wav_pcc": g["wav_pcc"],
                "ar_frames_per_s_teacher_forced": t["ar_frames_per_s"],
                "dit_chunk0_s": t["dit_per_chunk"][0],
                "total_s": t["total"],
                "dram_allocated_gb": (r.get("dram_after") or {}).get("allocated_bytes", 0) / 1e9,
                "passes": min(g["latent_pcc"]) >= BAR_LATENT_PCC and g["log_mel_rms_db"] <= BAR_LOGMEL_DB,
                "within_control": g["log_mel_rms_db"] <= CONTROL_LOGMEL_DB,
                "commit": r.get("commit"),
                "recorded_at": r.get("recorded_at"),
            }
        )
    md = [
        "| run | LLM policy | DiT | frame-hidden PCC (min) | latent PCC w0 / w1 | log-mel RMS dB | wav PCC | AR frames/s (teacher-forced) | DiT chunk 0 s | total 10 s clip s | DRAM GB | bars |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['label']} | {r['llm_policy']} | {r['dit_dtype']} | {r['frame_hiddens_pcc']:.5f} ({r['frame_hiddens_pcc_min']:.5f}) | "
            f"{' / '.join('%.5f' % p for p in r['latent_pcc'])} | {r['log_mel_rms_db']:.3f} | {r['wav_pcc']:.5f} | "
            f"{r['ar_frames_per_s_teacher_forced']:.2f} | {r['dit_chunk0_s']:.2f} | {r['total_s']:.1f} | {r['dram_allocated_gb']:.1f} | "
            f"{'pass' if r['passes'] else 'FAIL'}{' (within bf16 control)' if r['within_control'] else ''} |"
        )
    text = "\n".join(md) + "\n"
    print(text)
    (OUT / "dtype_sweep.md").write_text(text)
    (OUT / "dtype_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
