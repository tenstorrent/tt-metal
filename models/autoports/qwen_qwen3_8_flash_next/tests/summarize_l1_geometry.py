"""Compare short-prefill geometry under the final L1 working contract."""

import json
from pathlib import Path
import torch
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    torch.set_num_threads(8)
    rows = []
    for layer in (0, 1, 3):
        reference = DOC / f"s5_final3_trace_s128_l{layer}/timing.json"
        if not reference.exists():
            continue
        control = json.loads(reference.read_text())
        expected = torch.load(control["outputs"], weights_only=False)
        paths = list(DOC.glob(f"s5_l1_geometry_g*_l{layer}/timing.json"))
        paths += list(DOC.glob(f"s5_l1_geometry_k*_packed20_l{layer}/timing.json"))
        paths += list(DOC.glob(f"s5_l1_geometry_repeat_candidate_l{layer}/timing.json"))
        for path in sorted(paths):
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            if "repeat_candidate" in str(path):
                control = json.loads((DOC / f"s5_l1_geometry_repeat_control_l{layer}/timing.json").read_text())
                expected = torch.load(control["outputs"], weights_only=False)
            args = run["candidate"]["prefill"]
            grid, block = args["grid_x"], args["k_block"]
            actual = torch.load(run["outputs"], weights_only=False)
            metrics = {k: pcc_metrics(expected[k], actual[k]) for k in expected}
            assert min(v["pcc"] for v in metrics.values()) >= 0.995, (path, metrics)
            assert run["candidate"]["prefill"]["preserve_memory"]
            rows.append(
                dict(
                    layer=layer,
                    grid_x=grid,
                    requested_k_block=block,
                    packed_k_cap=args.get("packed_k_cap"),
                    control_ms=control["median_prefill_host_ms"],
                    candidate_ms=run["median_prefill_host_ms"],
                    metrics=metrics,
                    exact={k: torch.equal(expected[k], actual[k]) for k in expected},
                    source_sha256=run["source_sha256"],
                    artifact=str(path.relative_to(DOC)),
                )
            )
    (DOC / "l1_geometry.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Short-prefill geometry with L1 preserved",
        "",
        "B1/S128 recorded_inputs_long, caller-owned traced prefill, final-default memory placement. This repeats the earlier DRAM geometry family under the selected L1 contract.",
        "",
        "| Layer | Grid x | K block / packed cap | Default ms | Candidate ms | PCC prefill / decode |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['grid_x']} | {r['requested_k_block']} / {r['packed_k_cap']} | {r['control_ms']:.4f} | {r['candidate_ms']:.4f} | {r['metrics']['prefill']['pcc']:.8f} / {r['metrics']['decode']['pcc']:.8f} |"
        )
    (DOC / "l1_geometry.md").write_text("\n".join(lines) + "\n")
    print("L1_GEOMETRY_PCC_PASS", len(rows))
    for layer in (0, 1, 3):
        candidates = [r for r in rows if r["layer"] == layer]
        if candidates:
            print("BEST", min(candidates, key=lambda r: r["candidate_ms"]))


if __name__ == "__main__":
    main()
