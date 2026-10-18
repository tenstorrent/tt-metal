"""Index prefill memory/tracing experiments and exact saved-output comparisons."""

import json
from pathlib import Path
import torch
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    torch.set_num_threads(8)
    rows = []
    for layer in (0, 1, 3):
        for length in (128, 2053):
            control_path = DOC / f"s5_prefill_control_s{length}_l{layer}/timing.json"
            if not control_path.exists():
                continue
            control = json.loads(control_path.read_text())
            expected = torch.load(control["outputs"], weights_only=False)
            for variant in ("control", "trace", "l1", "l1_trace"):
                path = DOC / f"s5_prefill_{variant}_s{length}_l{layer}/timing.json"
                if not path.exists():
                    continue
                run = json.loads(path.read_text())
                outputs = torch.load(run["outputs"], weights_only=False)
                assert run["source_sha256"] == control["source_sha256"]
                metrics = {key: pcc_metrics(expected[key], outputs[key]) for key in expected}
                assert min(v["pcc"] for v in metrics.values()) >= 0.995, (path, metrics)
                rows.append(
                    dict(
                        layer=layer,
                        length=length,
                        variant=variant,
                        prefill_ms=run["median_prefill_host_ms"],
                        decode_ms=run["median_traced_decode_ms"],
                        capture_ms=run.get("prefill_capture_ms"),
                        metrics=metrics,
                        exact={key: torch.equal(expected[key], outputs[key]) for key in expected},
                        changed_trace_checks=len(run.get("prefill_trace_checks", [])),
                        tracking=run["tracking"],
                        source_sha256=run["source_sha256"],
                        artifact=str(path.relative_to(DOC)),
                    )
                )
    (DOC / "prefill_advice.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Prefill-specific advice experiments",
        "",
        "Same default 1×4 decoder and recorded_inputs_long fixture. Times are warmed milliseconds; prefill capture cost is separate. Traced rows use caller-owned fixed-shape capture. Changed-request checks compare prefill and appended decode bitwise with eager, refreshing residuals/tokens/slots/pages/positions without recapture.",
        "",
        "| Layer | Tokens | Mode | Prefill ms | Capture ms | PCC prefill / decode | Exact outputs | Changed checks |",
        "|---|---:|---|---:|---:|---|---|---:|",
    ]
    for r in rows:
        cap = f"{r['capture_ms']:.4f}" if r["capture_ms"] is not None else "—"
        lines.append(
            f"| {r['layer']} | {r['length']} | [{r['variant']}]({r['artifact']}) | {r['prefill_ms']:.4f} | {cap} | {r['metrics']['prefill']['pcc']:.8f} / {r['metrics']['decode']['pcc']:.8f} | {all(r['exact'].values())} | {r['changed_trace_checks']} |"
        )
    (DOC / "prefill_advice.md").write_text("\n".join(lines) + "\n")
    print("PREFILL_ADVICE_PCC_PASS", len(rows))


if __name__ == "__main__":
    main()
