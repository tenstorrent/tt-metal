# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: identify widening code differences and adversarial conv propagation.

No timing or acceptance waiver. CPU setup does not import TTNN/open hardware.
"""

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_conv_adapter_probe import (
    DOC,
    REVISION,
    conv_function,
    digest,
    load_weights,
)


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = load_weights(args.checkpoint)["linear_attn.conv1d.weight"][:, 0].float()
    paths = {
        "baseline": DOC / "v4_serial_l1_compact.py.snapshot",
        "candidate": DOC / "v4_conv_tilize_widen.py.snapshot",
    }
    selected_codes = [0x8000, 0x8001, 0x807F, 0x8080, 0x80BF, 0x80FF, 0x8100, 0x817F]
    tiny = torch.finfo(torch.float32).tiny
    cases = []
    for tap in (1, 2, 3):
        for scale in (0, 1, 2, 4, 8, 16):
            for code in selected_codes:
                history = torch.zeros((1, 4, 8192), dtype=torch.bfloat16)
                # Real first-tap weights approximately anchor the FP32 accumulator near
                # the underflow region; BF16 quantization is intentional and retained.
                anchor = torch.where(weights[:, 0] != 0, (scale * tiny) / weights[:, 0], 0.0)
                history[0, 0] = anchor.bfloat16()
                value = torch.tensor(code, dtype=torch.uint16).view(torch.bfloat16)
                history[0, tap] = value
                cases.append((dict(tap=tap, anchor_scale=scale, bf16_code=f"0x{code:04x}"), history))
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=digest(Path(__file__)),
        dependencies={
            name: digest(Path(__file__).with_name(name))
            for name in ("v4_conv_adapter_probe.py", "conv_first_tap_mixed_probe.py")
        },
        snapshots={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        planned_cases=len(cases),
        input_description="All finite BF16 codes in isolated conversion; actual checkpoint taps with one tiny negative BF16 tap and first-tap accumulation anchors near underflow. No oracle runtime injection.",
        tap_max_abs=[float(weights[:, j].abs().max()) for j in range(4)],
        cases=[],
    )
    for k, p in paths.items():
        _, h = conv_function(p, None)
        result["snapshots"][k]["conv_extracted_ast_sha256"] = h

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        assert all(torch.isfinite(h).all() for _, h in cases)
        save()
        print("CPU_BOUNDARY_SETUP_PASS", len(cases))
        return
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        narrow = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        obj = SimpleNamespace(
            taps_fp32=[upload(weights[:, j].reshape(1, 1, 8192), ttnn.float32) for j in range(4)],
            conv_post_chain=[narrow, widen, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), narrow],
        )
        fns = {k: conv_function(p, ttnn)[0] for k, p in paths.items()}
        codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        finite = codes[torch.isfinite(codes)]
        host = torch.cat([finite, torch.zeros(256, dtype=torch.bfloat16)]).reshape(1, 8, 8192)
        raw = upload(host, layout=ttnn.ROW_MAJOR_LAYOUT)
        original = ttnn.typecast(ttnn.to_layout(raw, ttnn.TILE_LAYOUT), ttnn.float32)
        folded = ttnn.to_layout(raw, ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        old, new = ttnn.to_torch(original), ttnn.to_torch(folded)
        bad = (old.contiguous().view(torch.int32) != new.contiguous().view(torch.int32)).flatten()
        ix = torch.nonzero(bad).flatten()
        entries = []
        for i in ix.tolist():
            entries.append(
                dict(
                    index=i,
                    input_code=f"0x{int(host.view(torch.uint16).flatten()[i]):04x}",
                    baseline_code=f"0x{int(old.view(torch.int32).flatten()[i]) & 0xffffffff:08x}",
                    candidate_code=f"0x{int(new.view(torch.int32).flatten()[i]) & 0xffffffff:08x}",
                )
            )
        result["conversion"] = compare(old, new)
        result["conversion"]["all_differing_input_codes"] = entries
        result["conversion"]["differing_unique_input_codes"] = len(set(e["input_code"] for e in entries))
        result["conversion"]["input_vs_baseline"] = compare(host.float(), old)
        result["conversion"]["input_vs_candidate"] = compare(host.float(), new)
        # Preserve compact raw conversion tensors for CPU diagnosis after the device run.
        capture = args.output.with_suffix(".conversion.pt")
        torch.save(dict(input=host, baseline=old, candidate=new), capture)
        result["conversion_capture"] = dict(path=str(capture), sha256=digest(capture))
        save()
        for metadata, history in cases:
            qkv = upload(history[:, 3:])
            outs, preacts, carries = {}, {}, {}
            chain = ttnn.unary_chain
            for variant in paths:
                state = {"conv": upload(history[:, :3])}
                seen = []

                def capture_chain(x, ops, **kw):
                    if ops is obj.conv_post_chain:
                        seen.append(x)
                    return chain(x, ops, **kw)

                try:
                    ttnn.unary_chain = capture_chain
                    with device_only():
                        out = fns[variant](obj, qkv, state)
                finally:
                    ttnn.unary_chain = chain
                assert len(seen) == 1
                outs[variant] = ttnn.to_torch(out)
                preacts[variant] = ttnn.to_torch(seen[0])
                carries[variant] = ttnn.to_torch(state["conv"])
            row = dict(
                **metadata,
                preactivation=compare(preacts["baseline"], preacts["candidate"]),
                output=compare(outs["baseline"], outs["candidate"]),
                carry=compare(carries["baseline"], carries["candidate"]),
            )
            result["cases"].append(row)
            save()
        result["conversion_exact"] = result["conversion"]["exact"]
        result["conv_exact"] = all(
            all(c[k]["exact"] for k in ("preactivation", "output", "carry")) for c in result["cases"]
        )
        result["status"] = "diagnostic_complete"
        assert all(digest(paths[k]) == v["sha256"] for k, v in result["snapshots"].items())
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(result["status"], args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, default=DOC / "v4_conv_widen_boundary_probe.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
