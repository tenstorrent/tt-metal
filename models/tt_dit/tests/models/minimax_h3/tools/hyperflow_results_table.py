# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tabulate the JSON sidecars ``test_pipeline_hyperflow_tasks_minimax_h3.py`` leaves per run.

Reads the sidecars rather than the job logs so the table carries the numbers the test measured,
not a regex's reading of them. A point that did not run is absent from the table rather than
blank: an empty cell reads as a measurement of zero.

    python -m models.tt_dit.tests.models.minimax_h3.tools.hyperflow_results_table \\
        ~/h3_hyperflow_artifacts -o RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK_ORDER = ("t2va", "fl2va", "ref2va")
# The pipeline's own row labels, in the order a run pays them. Per-mode rows ("Keyframe encode",
# "Reference encode") are columns only when some run has them, and "--" where a mode has none.
STAGE_ORDER = ("Encoder", "Keyframe encode", "Reference encode", "Denoise", "VAE decode", "Audio decode")


def load(directory: Path) -> list[dict]:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    return sorted(records, key=lambda r: (TASK_ORDER.index(r["task"]), r["duration_s"]))


def table(records: list[dict]) -> str:
    stages = [s for s in STAGE_ORDER if any(s in r["timings_s"] for r in records)]
    header = ["Mode", "Clip", "Frames", "Canvas", "Fwd", "Padded", *stages, "Total", "s / video s", "CLIP"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in records:
        row = [
            r["task"],
            f"{r['duration_s']} s",
            str(r["num_frames"]),
            f"{r['width']}x{r['height']}",
            str(r["num_forwards"]),
            str(r["padded_len"]),
            *[f"{r['timings_s'][s]:.1f}" if s in r["timings_s"] else "--" for s in stages],
            f"{r['total_compute_s']:.1f}",
            f"{r['s_per_video_second']:.1f}",
            f"{r['clip']['mean']:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render(records: list[dict]) -> str:
    if not records:
        return "# MiniMax-H3 HyperFlow results\n\nNo runs recorded.\n"
    adapters = sorted({r["adapter"] for r in records})
    meshes = sorted({"x".join(str(d) for d in r["mesh"]) for r in records})
    missing = [
        f"{task} {duration} s"
        for task in TASK_ORDER
        for duration in (5, 10, 15)
        if not any(r["task"] == task and r["duration_s"] == duration for r in records)
    ]
    parts = [
        "# MiniMax-H3 HyperFlow results",
        "",
        f"Adapter: `{', '.join(adapters)}` | mesh: {', '.join(meshes)} Blackhole | "
        "warm window (one full warmup generation at each shape; prepares and export excluded).",
        "",
        "Timings are seconds of compute. CLIP is prompt alignment, **recorded not gated** -- the "
        "calibrated bars elsewhere are set against the 49-forward base model.",
        "",
        table(records),
    ]
    if missing:
        parts += ["", f"**Not measured:** {', '.join(missing)}."]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory holding the per-run JSON sidecars")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args()

    text = render(load(args.directory))
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
