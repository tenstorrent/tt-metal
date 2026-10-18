"""Regenerate the stage-5 candidate ledger from saved timings and tensors."""

import json
from pathlib import Path
import torch
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_multichip_decoder"


def main():
    torch.set_num_threads(8)
    rows = []
    for path in sorted(DOC.glob("*/timing.json")):
        d = json.loads(path.read_text())
        if d["profiler"]:
            continue
        fixture = d["policy"]["recorded_tag"]
        length = d["prefill_length"]
        layer = d["layer"]
        ref = (
            DOC
            / (
                f"s5_before_l{layer}"
                if fixture == "recorded_inputs" and length == 128
                else f"s5_before_mesh_s{length}_l{layer}"
            )
            / "timing.json"
        )
        if not ref.exists():
            ref = DOC / f"s5_prefill_control_s{length}_l{layer}" / "timing.json"
        if not ref.exists():
            ref = DOC / (f"s5_qsa_bf16_k32_s{length}" if layer == 3 else f"s5_long_control_l{layer}") / "timing.json"
        metrics = None
        if ref.exists():
            control = json.loads(ref.read_text())
            if all(
                control["policy"][k] == d["policy"][k]
                for k in ("recorded", "recorded_tag", "prefill_length", "decode_position", "advance_position")
            ):
                a, b = torch.load(control["outputs"], weights_only=False), torch.load(d["outputs"], weights_only=False)
                metrics = {k: pcc_metrics(a[k], b[k]) for k in a}
        rows.append(
            dict(
                tag=path.parent.name,
                layer=layer,
                tokens=length,
                fixture=fixture,
                prefill_ms=d["median_prefill_host_ms"],
                prefill_mode="traced" if d["policy"].get("trace_prefill") else "eager",
                decode_ms=d["median_traced_decode_ms"],
                reference=str(ref.relative_to(DOC)) if metrics else None,
                metrics=metrics,
                source_sha256=d["source_sha256"],
                candidate=d.get("candidate"),
                timing=str(path.relative_to(DOC)),
            )
        )
    (DOC / "experiment_index.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Measured stage-5 experiments",
        "",
        "PCC is recomputed from saved outputs against the named matched mesh control. It is diagnostic; final acceptance uses unchanged single controls. Compare latency only within the same fixture/length/source family. Native-fix failure/recovery evidence is in triage_norm1.",
        "",
        "| Tag | Layer | Tokens / fixture | Prefill mode | Prefill ms | Decode ms | Prefill / decode PCC |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for r in rows:
        metrics = r["metrics"]
        pcc = f"{metrics['prefill']['pcc']:.8f} / {metrics['decode']['pcc']:.8f}" if metrics else "unpaired"
        lines.append(
            f"| [{r['tag']}]({r['timing']}) | {r['layer']} | {r['tokens']} / {r['fixture']} | {r['prefill_mode']} | {r['prefill_ms']:.6f} | {r['decode_ms']:.6f} | {pcc} |"
        )
    (DOC / "experiments.md").write_text("\n".join(lines) + "\n")
    print("EXPERIMENT_INDEX", len(rows))


if __name__ == "__main__":
    main()
