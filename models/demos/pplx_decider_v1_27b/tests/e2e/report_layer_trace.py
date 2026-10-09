# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-wise PCC table and plot (TT vs HF last-token residual after every layer) for the flagged rows.

Reads ``e2e_decisions.json`` written by ``tests/e2e/test_model.py::test_decision_agreement`` and writes
``doc/full_model/layer_trace.md`` and ``doc/full_model/layer_trace.png``::

    python models/demos/pplx_decider_v1_27b/tests/e2e/report_layer_trace.py [<e2e_decisions.json>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLAGGED = ("s01_ticket_routing", "m06_server_500", "x01_log_most_errors")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (dataviz reference palette, light)
DOC = Path(__file__).resolve().parents[2] / "doc" / "full_model"
DEFAULT_IN = Path("/local/ttuser/gtobar/artifacts/pplx_decider/stage6/e2e_decisions.json")


def main() -> None:
    data = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN).read_text())
    rows = {r["id"]: r for r in data["rows"]}
    trace = {rid: data["layer_pcc"][rid] for rid in FLAGGED}
    kinds = ["full" if (i % 4) == 3 else "linear" for i in range(len(trace[FLAGGED[0]]))]

    lines = [
        "| layer | kind | " + " | ".join(f"{rid} (S={rows[rid]['seq_len']})" for rid in FLAGGED) + " |",
        "|---:|---|" + "---:|" * len(FLAGGED),
    ]
    for i, kind in enumerate(kinds):
        lines.append(f"| {i} | {kind} | " + " | ".join(f"{trace[rid][i]:.6f}" for rid in FLAGGED) + " |")
    lines.append("")
    lines.append(
        "PCC of the TT vs HF bf16 last-real-token residual after each layer (before the final norm). "
        f"Source: {data['summary']['time']} run of tests/e2e/test_model.py."
    )
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "layer_trace.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    for rid, color in zip(FLAGGED, COLORS):
        values = trace[rid]
        label = f"{rid} (S={rows[rid]['seq_len']})"
        ax.plot(range(len(values)), values, color=color, linewidth=2, label=label)
        ax.annotate(
            label,
            (len(values) - 1, values[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
            va="center",
        )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("PCC vs HF (last token)")
    ax.set_xlim(0, len(kinds) + 14)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_title("TT vs HF last-token residual PCC after each layer (bf16 golden)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(DOC / "layer_trace.png")
    print(f"wrote {DOC / 'layer_trace.md'} and {DOC / 'layer_trace.png'}")


if __name__ == "__main__":
    main()
