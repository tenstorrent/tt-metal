# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU controls for native triangular recurrence underflow and signed zero.

Compiles the portable exact-order recurrence with independent input/output
subnormal controls. Saved device matrices are observations, not runtime inputs
to any decoder. No TTNN import or device access.
"""
import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_triangular_recurrence_fusion_cpu as base


def source(flush_inputs, flush_outputs, signed):
    code = base.CPP.replace(
        "#include <stdint.h>",
        """#include <stdint.h>
#include <string.h>
static inline float flush(float v) {
 uint32_t x;memcpy(&x,&v,4);
 if((x&0x7f800000)==0) {x=ZERO_BITS;memcpy(&v,&x,4);}return v;
}""".replace(
            "ZERO_BITS", "x&0x80000000" if signed else "0"
        ),
    )
    for op in ("*", "+"):
        code = code.replace(
            f"volatile float x=a{op}b;return x;",
            ("a=flush(a);b=flush(b);" if flush_inputs else "")
            + f"volatile float x=a{op}b;return "
            + ("flush(x)" if flush_outputs else "x")
            + ";",
        )
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=base.E / "v6_triangular_recurrence_ftz_cpu.json")
    args = p.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    fixtures, provenance = [], {}
    for tag in ("zero", "nonzero"):
        metadata = base.E / f"v6_chunk64_fused_corrected_output_{tag}_device.json"
        meta = json.loads(metadata.read_text())
        path = Path(meta["device_matrix_snapshot"])
        assert base.sha(path) == meta["device_matrix_snapshot_sha256"]
        assert meta["device_closed"] and meta["status"] == "completed_diagnostic"
        provenance[str(path)] = base.sha(path)
        for length, case in torch.load(path, weights_only=False, map_location="cpu").items():
            saved = case["intermediates"]
            for prefix in sorted(k.removesuffix("A_initial") for k in saved if k.endswith("/A_initial")):
                fixtures.append((f'{tag}_S{length}_{prefix.rstrip("/")}', saved, prefix))
    result = dict(
        status="running",
        source_sha256=base.sha(__file__),
        cpp_reference_source_sha256=base.sha(base.__file__),
        provenance=provenance,
        classification="CPU arithmetic-boundary controls against saved device matrices; no hardware",
        modes=[],
    )
    pointer = ctypes.POINTER(ctypes.c_float)
    for flush_inputs, flush_outputs, signed in (
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ):
        code = source(flush_inputs, flush_outputs, signed)
        mode = dict(
            flush_inputs=bool(flush_inputs),
            flush_outputs=bool(flush_outputs),
            preserve_zero_sign=bool(signed),
            cpp_sha256=hashlib.sha256(code.encode()).hexdigest(),
            cases=[],
        )
        with tempfile.TemporaryDirectory(prefix="triangular_ftz_") as directory:
            cpp, so = Path(directory) / "probe.cpp", Path(directory) / "probe.so"
            cpp.write_text(code)
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-fno-fast-math",
                    "-shared",
                    "-fPIC",
                    str(cpp),
                    "-o",
                    str(so),
                ],
                check=True,
                capture_output=True,
            )
            function = ctypes.CDLL(str(so)).triangular
            function.argtypes = [pointer, pointer, pointer, ctypes.c_uint32]
            for name, saved, prefix in fixtures:
                initial = saved[prefix + "A_initial"].contiguous()
                final = torch.empty_like(initial)
                snapshots = torch.empty((4, *initial.shape))
                ptr = lambda t: ctypes.cast(t.data_ptr(), pointer)
                function(ptr(initial), ptr(final), ptr(snapshots), 32)
                rows = {i: base.bits(saved[prefix + f"A_row{i}"], snapshots[slot]) for slot, i in enumerate(base.ROWS)}
                expected = saved[prefix + "A_row63"]
                mode["cases"].append(
                    dict(
                        name=name,
                        row_bits=rows,
                        final_bits=base.bits(expected, final),
                        max_abs=float((expected - final).abs().max()),
                    )
                )
        mode["total_final_bits"] = sum(c["final_bits"] for c in mode["cases"])
        mode["all_row_boundaries_exact"] = all(v == 0 for c in mode["cases"] for v in c["row_bits"].values())
        result["modes"].append(mode)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    chosen = next(
        m for m in result["modes"] if not m["flush_inputs"] and m["flush_outputs"] and m["preserve_zero_sign"]
    )
    assert chosen["all_row_boundaries_exact"]
    result["status"] = "signed_output_ftz_matches_all_saved_native_rows"
    result[
        "limits"
    ] = "These actual inputs do not discriminate signed output FTZ from signed input+output flush; generalized native DAZ behavior is not inferred. Device candidate must still pass native same-input tests including signed zeros."
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert "ttnn" not in sys.modules
    print(result["status"], "fixtures", len(fixtures))


if __name__ == "__main__":
    main()
