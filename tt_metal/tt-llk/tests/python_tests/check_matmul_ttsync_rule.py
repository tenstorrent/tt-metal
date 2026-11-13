#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Checks matmul_prod_keys.matmul_unpack_ttsync_cell equals matmul_unpack_ttsync_cell of llk_matmul_forms.h over every fidelity, format triple, accumulate mode, sync and small block."""
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_matmul_grid_rule import HEADER, STUB  # noqa: E402
from helpers.llk_params import DataFormat, MathFidelity  # noqa: E402
from helpers.matmul_prod_keys import matmul_unpack_ttsync_cell  # noqa: E402

FMTS = ["Float16_b", "Float32", "Bfp8_b", "Bfp4_b", "Bfp2_b", "Bfp8", "Float16"]
FIDS = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]


def main() -> int:
    rows = list(
        itertools.product(
            FIDS,
            FMTS,
            FMTS,
            FMTS,
            (False, True),
            (False, True),
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4, 5, 8),
        )
    )
    body = ",".join(
        "{MathFidelity::%s,%s,%s,DataFormat::%s,DataFormat::%s,DataFormat::%s,%d,%d,%d}"
        % (f, str(h).lower(), str(a).lower(), i, b, o, rt, ct, kt)
        for f, i, b, o, a, h, rt, ct, kt in rows
    )
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "llk_defs.h").write_text(STUB)
        (Path(d) / "llk_matmul_forms.h").write_text(HEADER.read_text())
        (Path(d) / "t.cpp").write_text(
            '#include <cstdio>\n#include "llk_matmul_forms.h"\nusing namespace ckernel;\n'
            "struct R{MathFidelity f;bool h,a;DataFormat i,b,o;int rt,ct,kt;};\n"
            f"static const R rows[]={{{body}}};\n"
            "int main(){for(const R& r:rows)putchar(matmul_unpack_ttsync_cell(r.f,r.h,r.a,r.i,r.b,r.o,r.rt,r.ct,r.kt)?'1':'0');}\n"
        )
        subprocess.run(
            [
                "g++",
                "-std=c++20",
                "-O1",
                f"-I{d}",
                str(Path(d) / "t.cpp"),
                "-o",
                str(Path(d) / "t"),
            ],
            check=True,
        )
        cpp = subprocess.run(
            [str(Path(d) / "t")], check=True, capture_output=True, text=True
        ).stdout
    py = "".join(
        (
            "1"
            if matmul_unpack_ttsync_cell(
                getattr(MathFidelity, f),
                h,
                a,
                getattr(DataFormat, i),
                getattr(DataFormat, b),
                getattr(DataFormat, o),
                rt,
                ct,
                kt,
            )
            else "0"
        )
        for f, i, b, o, a, h, rt, ct, kt in rows
    )
    bad = sum(x != y for x, y in zip(cpp, py))
    print(
        f"{len(rows)} cells, ttsync kept: C++ {cpp.count('1')} python {py.count('1')}; {bad} differ"
    )
    return 1 if bad or len(cpp) != len(py) else 0


if __name__ == "__main__":
    sys.exit(main())
