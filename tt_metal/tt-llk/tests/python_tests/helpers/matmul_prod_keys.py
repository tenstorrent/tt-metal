# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Form keys of the Blackhole matmul kernels that reach ttnn, as the kernels select them at compile time.

Metal2 (bmm_large_block_zm_fused_bias_activation_metal2.cpp), the legacy kernel and minimal_matmul key the forms table by the format
each DEST section is packed to (the partials until the last inner block) and the k steps of one inner block, and M2 and minimal_matmul
force the row MOP by their own rule; the harness keys by the output format and the whole k. This module is that selection, so a sweep
can run a cell with the forms production would take."""
import re
from pathlib import Path

from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity

_HEADER = (Path(__file__).resolve().parents[3] / "tt_llk_blackhole/llk_lib/llk_matmul_forms.h").read_text()
_TABLE = re.findall(r'"([.rRh]{5})"', _HEADER)
# matmul_main_form_cells of the header: cells whose keyed forms read slower than main's, which keep main's form
_NUM = r"(0x[0-9A-Fa-f]+|\d+)"
_CELL = (
    r"\{MathFidelity::(\w+), (true|false), (true|false), DataFormat::(\w+), DataFormat::(\w+), DataFormat::(\w+), (true|false)"
    + (r"(?:, " + _NUM + r")?") * 6
    + r"(?:, DataFormat::(\w+))?\},"
)


def _cells(text: str) -> dict:
    out = {}
    for fid, k1, acc, i, o, p, several_ok, *rest in re.findall(_CELL, text):
        if rest[6]:  # a mixed pair: the harness sweeps equal in0 and in1 formats, so the cell never applies there
            continue
        n = [int(x, 0) if x else d for x, d in zip(rest[:6], (0, 0xFFFF, -1, 0, 0xFFFF, 0))]
        out[(fid, k1 == "true", acc == "true", DataFormat[i], DataFormat[o], DataFormat[p], n[2])] = (several_ok == "true", n[0], n[1], n[3], n[4], n[5])
    return out


_MAIN_PART, _, _FORCED_PART = _HEADER.partition("matmul_forced_form_cells[] = {")
# matmul_main_form_cells of the header: cells whose keyed forms read slower than main's, which keep main's form
_MAIN_CELLS = _cells(_MAIN_PART)
# matmul_forced_form_cells: cells that take the form 'R', 'h' or 'r' whatever the table says
_FORCED_CELLS = [
    (fid, int(sync), int(mink), int(maxk, 0), acc == "true", DataFormat[i], DataFormat[o], DataFormat[p], int(minw), int(maxw, 0), int(minh), wide == "true", form, several == "true", int(sections or 0), int(maxh, 0) if maxh else 0xFFFF, in0_only == "true")
    for fid, sync, mink, maxk, acc, i, o, p, minw, maxw, minh, wide, form, several, sections, maxh, in0_only in re.findall(
        r"\{MathFidelity::(\w+), (\d), (\d+), (0x[0-9A-Fa-f]+|\d+), (true|false), DataFormat::(\w+), DataFormat::(\w+), DataFormat::(\w+), "
        r"(\d+), (0x[0-9A-Fa-f]+|\d+), (\d+), (true|false), '(\w)'(?:, (true|false))?(?:, (\d))?(?:, (0x[0-9A-Fa-f]+|\d+))?(?:, (true|false))?\},",
        _FORCED_PART.split("};")[0],
    )
]
_FMT = {
    DataFormat.Float16_b: 0,
    DataFormat.Float16: 1,
    DataFormat.Float32: 2,
    DataFormat.Bfp8_b: 3,
    DataFormat.Bfp4_b: 4,
}
_FID = {MathFidelity.LoFi: 0, MathFidelity.HiFi2: 1, MathFidelity.HiFi3: 2, MathFidelity.HiFi4: 3}


def form_char(kt: int, fidelity, half_sync: bool, fp32_dest: bool, fmt) -> str:
    """The forms table cell (llk_matmul_form.h matmul_default_form); '.' for an unknown format."""
    if fmt not in _FMT:
        return "."
    k = 0 if kt == 1 else (1 if kt <= 4 else 2)
    return _TABLE[((k * 4 + _FID[fidelity]) * 2 + (0 if half_sync else 1)) * 2 + (1 if fp32_dest else 0)][_FMT[fmt]]


def form(row_mop: bool, char: str) -> tuple:
    """(math MOP kind, hoisted unpack): the MOP the init programs and whether the unpack call hoists its row invariants."""
    return ("row" if row_mop else ("rows" if char in "rR" else "tile"), char in "hR")


def partials_format(fp32_dest: bool, packer_l1_acc: bool, out_format):
    """matmul_multicore_reuse_mcast_2d_program_factory interm0_data_format when more than one inner block keeps partials."""
    if packer_l1_acc:
        return DataFormat.Float32 if fp32_dest else DataFormat.Float16_b
    return DataFormat.Float32 if fp32_dest else out_format


def m2_row_mop(fidelity, h, w, kb, fp32_dest, out_format, pack_format) -> bool:
    return (fidelity != MathFidelity.LoFi or w >= 6 or (h * w >= 8 and kb > 1)) and (
        fp32_dest or (out_format != DataFormat.Float32 and pack_format != DataFormat.Float32)
    )


def lg_row_mop(fidelity, h, w, kb, fp32_dest, out_format, pack_format, in1_format) -> bool:
    return (
        m2_row_mop(fidelity, h, w, kb, fp32_dest, out_format, pack_format)
        and h * w >= 4
        and in1_format in (DataFormat.Bfp8_b, DataFormat.Bfp4_b)
    )


def _cell_hit(cells, fidelity, kb, fp32_dest, in_format, out_format, pack_format, several_blocks, row_tiles, full_sync, min_side):
    for sync in ((1,) if full_sync else (-1, 0)):
        cell = cells.get((fidelity.name, kb == 1, fp32_dest, in_format, out_format, pack_format, sync))
        if cell is not None and not (cell[0] and several_blocks) and cell[1] <= row_tiles <= cell[2] and cell[3] <= kb <= cell[4] and min_side >= cell[5]:
            return True
    return False


def mixed_main_form(fidelity, kb, fp32_dest, in_format, in_format_b, out_format, full_sync) -> bool:
    """matmul_mixed_main_form of llk_matmul_forms.h; in_format is in1 (SrcA), in_format_b is in0 (SrcB)."""
    if in_format_b is None or fidelity != MathFidelity.HiFi2:
        return False
    block = (DataFormat.Bfp8_b, DataFormat.Bfp4_b)
    return (
        (in_format == DataFormat.Float32 and in_format_b in block)
        or (not fp32_dest and kb == 1 and not full_sync and in_format == DataFormat.Bfp8_b and in_format_b == DataFormat.Float16_b and out_format == DataFormat.Float32)
    )


def main_form_cell(fidelity, kb, fp32_dest, in_format, out_format, pack_format, several_blocks=False, row_tiles=0, full_sync=False, min_side=99, in_format_b=None) -> bool:
    """matmul_main_form_cell_selected of llk_matmul_forms.h; the table cells for equal in0 and in1 formats, in_format_b (in0) given for a mixed pair."""
    if mixed_main_form(fidelity, kb, fp32_dest, in_format, in_format_b, out_format, full_sync):
        return True
    return _cell_hit(_MAIN_CELLS, fidelity, kb, fp32_dest, in_format, out_format, pack_format, several_blocks, row_tiles, full_sync, min_side)


def forced_form(fidelity, kb, fp32_dest, in_format, out_format, pack_format, full_sync, rt, ct, several_blocks=False, sections=1, in_format_b=None) -> str:
    """matmul_forced_form of llk_matmul_forms.h: 'R', 'h', 'r' or '' (the table's form), for a block of rt x ct tiles."""
    for fid, sync, mink, maxk, acc, i, o, p, minw, maxw, minh, wide, form, several, sec, maxh, in0_only in _FORCED_CELLS:
        if (
            fid == fidelity.name
            and sync == (1 if full_sync else 0)
            and mink <= kb <= maxk
            and acc == fp32_dest
            and (i in (in_format, in_format_b) if in_format_b is not None else in_format == i)
            and (not in0_only or (in_format if in_format_b is None else in_format_b) == i)
            and out_format == o
            and pack_format == p
            and minw <= ct <= maxw
            and rt >= minh
            and rt <= maxh
            and (not wide or ct >= rt)
            and (not several or several_blocks)
            and (sec == 0 or (sections == 1 if sec == 1 else sections > 1))
        ):
            return form
    return ""


MAIN_FORM = (("tile", False), None)  # main's form: tile MOP, per-row unpack, unknown format key


def production_forms(in1_format, out_format, fp32_dest, fidelity, h, w, kb) -> dict:
    """{(kernel, variant): (form, form_format)} for one sub block of kb inner k steps, Half sync (every ttnn matmul kernel but the
    gather_in0 ring). Variants: nb1 = one inner block (packed to the output format), nb4 = several inner blocks, nb4_l1acc = the
    same with packer_l1_acc. minimal_matmul packs every section to its intermediate buffer."""
    out = {}
    for nb in (1, 4):
        for l1 in (False, True) if nb > 1 else (False,):
            pf = partials_format(fp32_dest, l1, out_format) if nb > 1 else out_format
            tag = f"nb{nb}" + ("_l1acc" if l1 else "")
            if main_form_cell(fidelity, kb, fp32_dest, in1_format, out_format, pf, nb > 1, max(h, w)):
                out[("M2", tag)] = out[("LG", tag)] = MAIN_FORM
                continue
            ch = form_char(kb, fidelity, True, fp32_dest, pf)
            out[("M2", tag)] = (form(m2_row_mop(fidelity, h, w, kb, fp32_dest, out_format, pf), ch), pf)
            out[("LG", tag)] = (form(lg_row_mop(fidelity, h, w, kb, fp32_dest, out_format, pf, in1_format), ch), pf)
    pf = DataFormat.Float32 if fp32_dest else DataFormat.Float16_b
    if main_form_cell(fidelity, kb, fp32_dest, in1_format, out_format, pf, False, max(h, w)):
        out[("MM", "interm")] = MAIN_FORM
    else:
        out[("MM", "interm")] = (form(h * w >= 8 and kb > 1, form_char(kb, fidelity, True, fp32_dest, pf)), pf)
    return out


def harness_forms(fidelity, fp32_dest, out_format, h, w, kt) -> set:
    """The (math, hoist) pairs the sweep measures for a Half cell: the forms table keyed by the output format, and the row MOP
    module (perf_matmul_row_mop: at least 8 tiles, kt > 1, not Float32 from a 16-bit DEST)."""
    ch = form_char(kt, fidelity, True, fp32_dest, out_format)
    s = {form(False, ch)}
    if h * w >= 8 and kt > 1 and not (out_format == DataFormat.Float32 and not fp32_dest):
        s.add(form(True, ch))
    return s


def unmeasured_production_forms(in1_format, out_format, fp32_dest, fidelity, h, w, kt) -> dict:
    """{form: form_format} production selects for the cell that the sweep does not measure (one representative format per form)."""
    seen = harness_forms(fidelity, fp32_dest, out_format, h, w, kt)
    res = {}
    for (kernel, tag), (f, fmt) in production_forms(in1_format, out_format, fp32_dest, fidelity, h, w, kt).items():
        if f not in seen:
            res.setdefault(f, fmt)
    return res
