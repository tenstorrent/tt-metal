# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight policy/geometry experiments; never changes production defaults."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    ("bfp8_lofi_block16", dict(expert_fidelity="LoFi", sparse_block_w=[16, 16])),
    ("bfp8_lofi_block64", dict(expert_fidelity="LoFi", sparse_block_w=[64, 16])),
    (
        "bfp8_lofi_cores16_32",
        dict(expert_fidelity="LoFi", sparse_block_w=[16, 16], sparse_cores=[16, 32], sparse_subblock_w=[2, 2]),
    ),
    (
        "bfp8_lofi_cores8_16",
        dict(expert_fidelity="LoFi", sparse_block_w=[16, 16], sparse_cores=[8, 16], sparse_subblock_w=[4, 4]),
    ),
    (
        "bfp4_gate_bfp8_down_lofi",
        dict(expert_dtype="bfp4", expert_down_dtype="bfp8", expert_fidelity="LoFi", sparse_block_w=[32, 16]),
    ),
    (
        "bfp8_gate_bfp4_down_lofi",
        dict(expert_dtype="bfp8", expert_down_dtype="bfp4", expert_fidelity="LoFi", sparse_block_w=[32, 16]),
    ),
    ("bf16_hifi4_l1", dict(expert_dtype="bf16", expert_fidelity="HiFi4", shared_dtype="bf16", shared_fidelity="HiFi4")),
    ("bfp8_hifi4_l1", dict(expert_fidelity="HiFi4", shared_dtype="bf16", shared_fidelity="HiFi4")),
    ("bfp8_hifi2_dram", dict(expert_memory="dram", shared_dtype="bf16", shared_fidelity="HiFi4")),
    ("bfp8_hifi2_l1", dict(shared_dtype="bf16", shared_fidelity="HiFi4")),
    ("bfp8_lofi_l1", dict(expert_fidelity="LoFi", shared_dtype="bf16", shared_fidelity="HiFi4")),
    ("bfp4_lofi_l1", dict(expert_dtype="bfp4", expert_fidelity="LoFi", shared_dtype="bf16", shared_fidelity="HiFi4")),
    ("bfp8_hifi2_shared_bfp8_hifi2", {}),
    ("bfp8_hifi2_shared_bfp8_lofi", dict(shared_fidelity="LoFi")),
    ("bfp8_hifi2_block16", dict(sparse_block_w=[16, 16])),
    ("bfp8_hifi2_block32", dict(sparse_block_w=[32, 16])),
    ("bfp8_hifi2_block64", dict(sparse_block_w=[64, 16])),
    ("bfp8_hifi2_cores16_32", dict(sparse_block_w=[16, 16], sparse_cores=[16, 32], sparse_subblock_w=[2, 2])),
    ("bfp8_hifi2_cores8_16", dict(sparse_block_w=[16, 16], sparse_cores=[8, 16], sparse_subblock_w=[4, 4])),
    ("bfp8_lofi_block32", dict(expert_fidelity="LoFi", sparse_block_w=[32, 16])),
    ("bfp4_lofi_block32", dict(expert_dtype="bfp4", expert_fidelity="LoFi", sparse_block_w=[32, 16])),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--only", nargs="*")
    p.add_argument("--matrix", type=Path)
    p.add_argument("--module", default="models.autoports.qwen_qwen3_6_35b_a3b.tests.run_decoder")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QWEN_DECODER="optimized")
    for key in ("TT_VISIBLE_DEVICES", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        env.pop(key, None)
    results = []
    candidates = json.loads(args.matrix.read_text()) if args.matrix else CANDIDATES
    for name, policy in candidates:
        if args.only and name not in args.only:
            continue
        if (args.output_dir / "STOP_QUEUE").exists():
            break
        if subprocess.run(["fuser", "/dev/tenstorrent/0"], capture_output=True).returncode == 0:
            raise RuntimeError("Chip 0 is owned; refusing concurrent device work")
        output = args.output_dir / f"{name}_l{args.layer}.json"
        if output.exists():
            raise FileExistsError(output)
        cmd = [
            "timeout",
            "600",
            sys.executable,
            "-m",
            args.module,
            "--layer",
            str(args.layer),
            "--length",
            "256",
            "--context",
            "512",
            "--prefill-repeats",
            "5",
            "--decode-repeats",
            "60",
            "--build-kwargs",
            json.dumps(policy),
            "--output",
            str(output),
        ]
        with output.with_suffix(".log").open("w") as log:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        row = dict(
            candidate=name,
            policy=policy,
            layer=args.layer,
            exit_code=code,
            command=cmd,
            regime="serial physical chip 0",
            activation_source="seed42 random BF16; real checkpoint weights",
        )
        if output.exists():
            d = json.loads(output.read_text())
            row.update(
                {
                    k: d[k]
                    for k in (
                        "prefill_pcc",
                        "decode_traced_pcc",
                        "prefill_warmed_ms",
                        "decode_traced_warmed_ms",
                        "precision_policy",
                    )
                }
            )
        results.append(row)
        (args.output_dir / "search.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, code, row.get("decode_traced_warmed_ms"), flush=True)
        if code in (124, 137):
            raise RuntimeError("Bounded device job timed out; preserve state and recover before continuing")


if __name__ == "__main__":
    main()
