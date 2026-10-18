"""Match current-L1 prefill subblock candidates to final-default trace outputs."""

import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    torch.set_num_threads(8)
    rows = []
    for path in sorted(DOC.glob("s5_subblock_*/timing.json")):
        candidate = json.loads(path.read_text())
        config = (candidate.get("candidate") or {}).get("prefill")
        if config is None:
            continue
        layer = candidate["layer"]
        length = candidate["policy"]["prefill_length"]
        reference = DOC / f"s5_final4_trace_s{length}_l{layer}/timing.json"
        if "repeat" in path.parent.name:
            reference = DOC / path.parent.name.replace("candidate", "control") / "timing.json"
        control = json.loads(reference.read_text())
        assert control["source_sha256"] == candidate["source_sha256"]
        expected = torch.load(control["outputs"], weights_only=False)
        actual = torch.load(candidate["outputs"], weights_only=False)
        metrics = {key: pcc_metrics(expected[key], actual[key]) for key in expected}
        assert min(item["pcc"] for item in metrics.values()) >= 0.995, (path, metrics)
        rows.append(
            dict(
                artifact=str(path.relative_to(DOC)),
                reference=str(reference.relative_to(DOC)),
                source_sha256=candidate["source_sha256"],
                layer=layer,
                length=length,
                config=config,
                control_ms=control["median_prefill_host_ms"],
                candidate_ms=candidate["median_prefill_host_ms"],
                metrics=metrics,
            )
        )
    (DOC / "prefill_subblocks.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Current-L1 output-subblock follow-up",
        "",
        "Same recorded_inputs_long fixture and final-default source; caller-owned traced prefill.",
        "Long candidates retain the outer-chunk scope and leave tiny tails unchanged.",
        "",
        "| Layer | Tokens | Grid x | Requested height | Default ms | Candidate ms | Minimum PCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['layer']} | {row['length']} | {row['config']['grid_x']} | "
            f"{row['config']['subblock_h']} | {row['control_ms']:.4f} | {row['candidate_ms']:.4f} | "
            f"{min(item['pcc'] for item in row['metrics'].values()):.8f} |"
        )
    (DOC / "prefill_subblocks.md").write_text("\n".join(lines) + "\n")
    print("PREFILL_SUBBLOCK_PCC_PASS", len(rows))
    for row in rows:
        print(row["artifact"], row["control_ms"], row["candidate_ms"])


if __name__ == "__main__":
    main()
