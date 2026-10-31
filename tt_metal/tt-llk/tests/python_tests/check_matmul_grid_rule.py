# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Checks that matmul_prod_keys.grid_main_cell equals matmul_grid_main_form of llk_matmul_forms.h over every row of the perf_matmul_grid sweep.

Hardware free: the header is compiled on the host against a stub of llk_defs.h.   usage: CHIP_ARCH=blackhole PYTHONPATH=<tt_llk_perf parent> python3 check_matmul_grid_rule.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers.format_config import DataFormat
from helpers.llk_params import DestSync, MathFidelity
from helpers.matmul_sweep import matmul_pack_form
from helpers.matmul_prod_keys import grid_main_cell, grid_main_flip_cell, grid_unfused_post_cell, tiny_main_cell
import perf_matmul_grid as grid
import perf_matmul_grid2 as grid2

HEADER = Path(__file__).resolve().parents[2] / "tt_llk_blackhole/llk_lib/llk_matmul_forms.h"
STUB = """#pragma once
#include <cstdint>
namespace ckernel {
enum class DataFormat { Invalid, Float32, Tf32, Float16, Float16_b, Bfp8, Bfp8_b, Bfp4_b, Bfp2, Bfp2_b, Bfp4 };
enum class MathFidelity { LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4 };
enum class DstSync { SyncHalf, SyncFull };
}
"""
NAME = {"F16b": "Float16_b", "B8": "Bfp8_b", "B4": "Bfp4_b", "F32": "Float32", "F16": "Float16", "B8x": "Bfp8", "B2": "Bfp2_b"}


def check_tiny() -> int:
    """The tiny tile rows (perf_matmul_grid2_tiny): python tiny_main_cell against matmul_tiny_main_form."""
    rows = grid2._tiny_rows()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "llk_defs.h").write_text(STUB)
        (Path(d) / "llk_matmul_forms.h").write_text(HEADER.read_text())
        body = ",\n".join(
            "{MathFidelity::%s,%s,%s,DataFormat::%s,DataFormat::%s,DataFormat::%s,%d,%d,%d,%d,%d}"
            % (fi, "true" if sy == "H" else "false", "true" if acc == "Y" else "false", NAME[a], NAME[b], NAME[o], r0, c1, rt, ct, kt)
            for a, b, o, acc, sy, fi, r0, c1, rt, ct, kt in rows
        )
        (Path(d) / "t.cpp").write_text(
            '#include <cstdio>\n#include "llk_matmul_forms.h"\nusing namespace ckernel;\nstruct R{MathFidelity f;bool h,a;DataFormat i1,i0,o;int r0,c1,rt,ct,kt;};\n'
            f"static const R rows[]={{{body}}};\nint main(){{for(const R& r:rows)putchar(matmul_tiny_main_form(r.f,r.h,r.a,r.i1,r.i0,r.o,r.r0,r.c1,r.rt,r.ct,r.kt)?'1':'0');}}\n"
        )
        subprocess.run(["g++", "-std=c++20", "-O1", f"-I{d}", str(Path(d) / "t.cpp"), "-o", str(Path(d) / "t")], check=True)
        cpp = subprocess.run([str(Path(d) / "t")], check=True, capture_output=True, text=True).stdout
    py = "".join(
        "1" if tiny_main_cell(getattr(MathFidelity, fi.name if hasattr(fi, "name") else fi), sy == "H", acc == "Y", getattr(DataFormat, NAME[a]), getattr(DataFormat, NAME[b]), getattr(DataFormat, NAME[o]), r0, c1, rt, ct, kt) else "0"
        for a, b, o, acc, sy, fi, r0, c1, rt, ct, kt in rows
    )
    bad = sum(x != y for x, y in zip(cpp, py))
    print(f"{len(rows)} tiny rows, C++ keeps main's form on {cpp.count('1')}, python on {py.count('1')}, {bad} differ")
    return bad or len(cpp) != len(py)


def _pack_py(a, b, o, sy, fi, rt, ct, kt, thr, r0, c1, dense, l1) -> str:
    from types import SimpleNamespace

    fm = SimpleNamespace(input_format=getattr(DataFormat, NAME[a]), input_format_B=getattr(DataFormat, NAME[b]), output_format=getattr(DataFormat, NAME[o]))
    return str(int(matmul_pack_form(fm, DestSync.Half if sy == "H" else DestSync.Full, getattr(MathFidelity, fi), rt, ct, kt, r0, c1, thr, dense, l1)))


def check_pack() -> int:
    """The pack form decision: python matmul_pack_form against the C++ matmul_pack_form_k over the grid, grid2 and tiny rows and an exhaustive cross product of
    the formats, fidelity, sync, block shape, k, throttle, tile shape, dense pages and packer L1 accumulate."""
    import itertools

    rows = [(a, b, o, sy, fi, h, w, kt, thr, 32, 32, False, bool(l1)) for a, b, o, acc, sy, fi, h, w, kt, thr, l1 in grid._grid()]
    rows += [(a, b, o, sy, fi, h, w, kt, thr, 32, 32, False, bool(l1)) for a, b, o, acc, sy, fi, h, w, kt, thr, l1, nb in grid2._grid2()]
    rows += [(a, b, o, sy, fi, rt, ct, kt, 0, r0, c1, True, False) for a, b, o, acc, sy, fi, r0, c1, rt, ct, kt in grid2._tiny_rows()]
    fm = ["F16b", "B8", "B4", "F32", "F16", "B8x", "B2"]
    fis = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
    dims = (1, 2, 3, 4, 5, 8)
    shapes = ((32, 32), (4, 32), (16, 32), (32, 16))
    cross = list(itertools.product(fm, fm, fm, "HF", fis, dims, dims, (1, 2, 8), (0, 2, 3, 4), shapes, (False, True), (False, True)))
    cross = [(a, b, o, sy, fi, rt, ct, kt, thr, sh[0], sh[1], d, l1) for a, b, o, sy, fi, rt, ct, kt, thr, sh, d, l1 in cross[::7]]
    allrows = rows + cross
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "llk_defs.h").write_text(STUB)
        (Path(d) / "llk_matmul_forms.h").write_text(HEADER.read_text())
        body = ",\n".join(
            "{MathFidelity::%s,%s,%d,DataFormat::%s,DataFormat::%s,DataFormat::%s,%d,%d,%d,%d,%d,%s,%s}"
            % (fi, "true" if sy == "H" else "false", thr, NAME[a], NAME[b], NAME[o], rt, ct, kt, r0, c1, "true" if dn else "false", "true" if l1 else "false")
            for a, b, o, sy, fi, rt, ct, kt, thr, r0, c1, dn, l1 in allrows
        )
        (Path(d) / "t.cpp").write_text(
            '#include <cstdio>\n#include "llk_matmul_forms.h"\nusing namespace ckernel;\nstruct R{MathFidelity f;bool h;int th;DataFormat i1,i0,o;int rt,ct,kt,r0,c1;bool dn,l1;};\n'
            f"static const R rows[]={{{body}}};\nint main(){{for(const R& r:rows)putchar('0'+matmul_pack_form_k(r.f,r.h,r.th,r.kt,r.rt,r.ct,r.i1,r.i0,r.o,r.r0,r.c1,r.dn,r.l1));}}\n"
        )
        subprocess.run(["g++", "-std=c++20", "-O0", f"-I{d}", str(Path(d) / "t.cpp"), "-o", str(Path(d) / "t")], check=True)
        cpp = subprocess.run([str(Path(d) / "t")], check=True, capture_output=True, text=True).stdout
    py = "".join(_pack_py(*r) for r in allrows)
    bad = sum(x != y for x, y in zip(cpp, py))
    forms = {k: cpp.count(k) for k in "0124"}
    print(f"{len(rows)} grid, grid2 and tiny rows + {len(cross)} cross product rows, pack form: C++ {forms}, {bad} differ")
    return bad or len(cpp) != len(py)


def check_cell_table(label: str, rows_name: str, func: str, table: str) -> int:
    """A cell table (python rows against the C++ table and its lookup function): every python row is a C++ cell and the two tables have the same size."""
    import helpers.matmul_prod_keys as pk

    rows = getattr(pk, rows_name)

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "llk_defs.h").write_text(STUB)
        (Path(d) / "llk_matmul_forms.h").write_text(HEADER.read_text())
        body = ",\n".join(
            "{MathFidelity::%s,%s,%s,DataFormat::%s,DataFormat::%s,DataFormat::%s,%d,%d,%d,%d,%s,%d}"
            % (fi, "true" if h else "false", "true" if a else "false", NAME[i1], NAME[i0], NAME[o], rt, ct, kt, th, "true" if l1 else "false", nb)
            for fi, h, a, i1, i0, o, rt, ct, kt, th, l1, nb in rows
        )
        (Path(d) / "t.cpp").write_text(
            '#include <cstdio>\n#include "llk_matmul_forms.h"\nusing namespace ckernel;\nstruct R{MathFidelity f;bool h,a;DataFormat i1,i0,o;int rt,ct,kt,th;bool l1;int nb;};\n'
            f"static const R rows[]={{{body}}};\nint main(){{for(const R& r:rows)putchar({func}(r.f,r.h,r.a,r.i1,r.i0,r.o,r.rt,r.ct,r.kt,r.th,r.l1,r.nb)?'1':'0');printf(\" %zu\",sizeof({table})/sizeof({table}[0]));}}\n"
        )
        subprocess.run(["g++", "-std=c++20", "-O1", f"-I{d}", str(Path(d) / "t.cpp"), "-o", str(Path(d) / "t")], check=True)
        out = subprocess.run([str(Path(d) / "t")], check=True, capture_output=True, text=True).stdout
    flags, size = out.split()
    bad = flags.count("0") + (int(size) != len(rows)) + (len(set(rows)) != len(rows))
    print(f"{label}: {len(rows)} python cells, {size} C++ cells, {bad} differ")
    return bad


def check_forced() -> int:
    """matmul_forced_form of the header against matmul_prod_keys.forced_form over a cross product of formats, block shapes, k steps and syncs (every cell of the table is hit)."""
    from itertools import product

    from helpers.matmul_prod_keys import forced_form

    cases = [
        (fi, sy, acc, i0, i1, o, rt, ct, kb, 1)
        for fi, sy, acc, i0, i1, o, (rt, ct), kb in product(
            ("LoFi", "HiFi2", "HiFi4"),
            (0, 1),
            (False, True),
            ("B8x", "B8", "B4", "F32"),
            ("F16", "F16b", "B8", "F32"),
            ("B8x", "B8", "B4", "F16b", "F32"),
            ((1, 7), (1, 6), (2, 4)),
            (1, 3, 8),
        )
    ]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "llk_defs.h").write_text(STUB)
        (Path(d) / "llk_matmul_forms.h").write_text(HEADER.read_text())
        calls = "\n".join(
            "putchar(matmul_forced_form<MathFidelity::%s,%d,%s,DataFormat::%s,DataFormat::%s,DataFormat::%s,DataFormat::%s,%d,%d,%d,false,%d>() ?: '-');"
            % (fi, kb, "true" if acc else "false", NAME[i0], NAME[i1], NAME[o], NAME[o], sy, rt, ct, sec)
            for fi, sy, acc, i0, i1, o, rt, ct, kb, sec in cases
        )
        (Path(d) / "t.cpp").write_text('#include <cstdio>\n#include "llk_matmul_forms.h"\nusing namespace ckernel;\nint main(){\n' + calls + "\n}\n")
        subprocess.run(["g++", "-std=c++20", "-O0", "-fconstexpr-ops-limit=1000000000", f"-I{d}", str(Path(d) / "t.cpp"), "-o", str(Path(d) / "t")], check=True)
        cpp = subprocess.run([str(Path(d) / "t")], check=True, capture_output=True, text=True).stdout
    py = "".join(
        forced_form(getattr(MathFidelity, fi), kb, acc, getattr(DataFormat, NAME[i1]), getattr(DataFormat, NAME[o]), getattr(DataFormat, NAME[o]), sy == 1, rt, ct, False, sec, in_format_b=getattr(DataFormat, NAME[i0])) or "-"
        for fi, sy, acc, i0, i1, o, rt, ct, kb, sec in cases
    )
    bad = sum(x != y for x, y in zip(cpp, py)) + (len(cpp) != len(py))
    print(f"forced form: {len(cases)} cases, C++ forces {len(cpp) - cpp.count('-')}, python {len(py) - py.count('-')}, {bad} differ")
    return bad


def main() -> int:
    rows = [(*r, 1) for r in grid._grid()] + list(grid2._grid2())
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "llk_defs.h").write_text(STUB)
        (Path(d) / "llk_matmul_forms.h").write_text(HEADER.read_text())
        body = ",\n".join(
            "{MathFidelity::%s,%s,%s,DataFormat::%s,DataFormat::%s,DataFormat::%s,%d,%d,%d,%d,%s,%d}"
            % (fi, "true" if sy == "H" else "false", "true" if acc == "Y" else "false", NAME[a], NAME[b], NAME[o], h, w, kt, thr, "true" if l1 else "false", nb)
            for a, b, o, acc, sy, fi, h, w, kt, thr, l1, nb in rows
        )
        (Path(d) / "t.cpp").write_text(
            '#include <cstdio>\n#include "llk_matmul_forms.h"\nusing namespace ckernel;\nstruct R{MathFidelity f;bool h,a;DataFormat i1,i0,o;int rt,ct,kt,th;bool l1;int nb;};\n'
            f"static const R rows[]={{{body}}};\nint main(){{for(const R& r:rows){{putchar(matmul_grid_main_form(r.f,r.h,r.a,r.i1,r.i0,r.o,r.rt,r.ct,r.kt,r.th,r.l1,r.nb)?'1':'0');putchar(matmul_unfused_post(r.f,r.h,r.a,r.i1,r.i0,r.o,r.rt,r.ct,r.kt,r.th,r.l1,r.nb)?'1':'0');putchar(matmul_main_flip(r.f,r.h,r.a,r.i1,r.i0,r.o,r.rt,r.ct,r.kt,r.th,r.l1,r.nb)?'1':'0');}}}}\n"
        )
        subprocess.run(["g++", "-std=c++20", "-O1", f"-I{d}", str(Path(d) / "t.cpp"), "-o", str(Path(d) / "t")], check=True)
        cpp = subprocess.run([str(Path(d) / "t")], check=True, capture_output=True, text=True).stdout
    py = ""
    for a, b, o, acc, sy, fi, h, w, kt, thr, l1, nb in rows:
        args = (getattr(MathFidelity, fi), sy == "H", acc == "Y", getattr(DataFormat, NAME[a]), getattr(DataFormat, NAME[b]), getattr(DataFormat, NAME[o]), h, w, kt, thr, l1, nb)
        py += ("1" if grid_main_cell(*args) else "0") + ("1" if grid_unfused_post_cell(*args) else "0") + ("1" if grid_main_flip_cell(*args) else "0")
    bad = sum(x != y for x, y in zip(cpp, py))
    print(f"{len(rows)} grid rows, main's form: C++ {cpp[0::3].count('1')} python {py[0::3].count('1')}; separate post: C++ {cpp[1::3].count('1')} python {py[1::3].count('1')}; main flip: C++ {cpp[2::3].count('1')} python {py[2::3].count('1')}; {bad} differ")
    return 1 if bad or len(cpp) != len(py) or check_tiny() or check_pack() or check_cell_table("main form table", "_GRID_MAIN_ROWS", "matmul_grid_main_form", "matmul_grid_main_cells") or check_cell_table("end fold off table", "_END_FOLD_OFF_ROWS", "matmul_end_fold_off", "matmul_end_fold_off_cells") or check_forced() else 0


if __name__ == "__main__":
    sys.exit(main())
