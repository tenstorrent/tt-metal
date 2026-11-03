# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Commit compact raw profiler contracts and verify actual measured precision."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"
for layer, kind in ((0, "dense"), (3, "mova_moe")):
    source = DOC / "tracy/final" / kind / "decode_ops.csv"
    phase = None
    rows = {"prefill": [], "decode": []}
    with source.open() as f:
        for row in csv.DictReader(f):
            if row["OP TYPE"] == "signpost":
                if row["OP CODE"] in ("PERF_PREFILL", "PERF_DECODE"):
                    phase = row["OP CODE"].removeprefix("PERF_").lower()
                elif row["OP CODE"].endswith("_END"):
                    phase = None
            elif phase and "matmul" in row["OP CODE"].lower() and row.get("DEVICE KERNEL DURATION [ns]"):
                fields = {
                    k: v
                    for k, v in row.items()
                    if v
                    and (
                        k in ("OP CODE", "ATTRIBUTES", "CORE COUNT", "MATH FIDELITY", "GLOBAL CALL COUNT")
                        or k.startswith(("INPUT_", "OUTPUT_"))
                    )
                }
                fields["device_us"] = float(row["DEVICE KERNEL DURATION [ns]"]) / 1000
                rows[phase].append(fields)
    expected = ["BFLOAT8_B", "BFLOAT4_B", "BFLOAT4_B", "BFLOAT4_B"] if layer == 0 else ["BFLOAT4_B"] * 7
    for phase, ops in rows.items():
        assert len(ops) == len(expected), (phase, len(ops))
        for op, dtype in zip(ops, expected):
            assert op["INPUT_1_DATATYPE"] == dtype, op
            assert op["MATH FIDELITY"] == "LoFi", op
    payload = dict(
        layer=layer,
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        decoder_sha256=hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        precision_assertions="pass",
        rows=rows,
        sparse_accounting="Indexed decode uses4 value /8 MLP experts; prefill is the dynamic union per32-row tile. No fixed nnz is asserted.",
    )
    (DOC / "tracy/final" / kind / "matmul_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("MEASURED_PRECISION_PASS", kind)
