# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.4 (OWNER 2026-11-27): render results/WA.4_ladder.md from the A/B runs of WA.4 (CPU only, no device).

Reads the orchestrator's A/B directories ``<run>/ab/WA.4/{new,old}/<unit>/WA.4.json`` (new = DS41_WOA_BANK=on, old =
off; the same gate commands, the switch as each side sets it) and writes one table per path (single 8192, 4x2k,
4x1792s, single 5120) with steady wall ms, tok/s, resumed wall ms, device ms and exposed host ms off vs on at every
rung, the attention section's device ms, and the test_woa_offon numbers (wo_a device ms, load delta). A side that has
not run yet shows "-".

    python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_ladder_report [--ab DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import pathlib

AB = pathlib.Path("/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/WA.4")
OUT = pathlib.Path("models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/results/WA.4_ladder.md")
RUNGS = (51200, 131072, 139264, 262144, 524288, 786432, 1024000)
SIDES = ("old", "new")  # off, on
TAG = "_tagentic_cc"


def _metrics(side_dir: pathlib.Path, pick) -> dict:
    """Merged metrics of the unit directories of one side whose name ``pick`` accepts."""
    out = {}
    if not side_dir.is_dir():
        return out
    for d in sorted(side_dir.iterdir()):
        if d.is_dir() and pick(d.name):
            for f in d.glob("*.json"):
                try:
                    m = json.loads(f.read_text()).get("metrics", {})
                except (OSError, ValueError, AttributeError):
                    continue
                out.update({k: (v.get("value") if isinstance(v, dict) else v) for k, v in m.items()})
    return out


def _f(v, nd=1) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def _d(a, b, nd=1) -> str:
    return "-" if a is None or b is None else f"{b - a:+.{nd}f}"


def _table(ms: dict, key) -> list[str]:
    """key(field, rung) -> metric name. Rows: one a rung; columns off / on / on - off per field."""
    fields = [
        ("steady wall ms", "steady_wall_ms", 1),
        ("steady tok/s", "steady_tok_s", 0),
        ("resumed wall ms", "resumed_wall_ms", 1),
        ("steady device ms", "steady_device_ms", 2),
        ("steady exposed host ms", "host", 1),
        ("attention device ms", "steady_device_ms_attention", 2),
    ]
    head = "| history | " + " | ".join(f"{n} off / on (on - off)" for n, _, _ in fields) + " |"
    rows = [head, "|---|" + "---|" * len(fields)]
    for r in RUNGS:
        cells = []
        for _, fld, nd in fields:
            a, b = (ms[s].get(key(fld, r)) for s in SIDES)
            cells.append(f"{_f(a, nd)} / {_f(b, nd)} ({_d(a, b, nd)})")
        rows.append(f"| {r} | " + " | ".join(cells) + " |")
    return rows


def _single_key(fld: str, r: int) -> str:
    return f"steady_host_ms_p{r}{TAG}" if fld == "host" else f"{fld}_p{r}{TAG}"


def _lane_key(case: str):
    def k(fld: str, r: int) -> str:
        if fld == "host":
            return f"ladder_{case}_host_exposed_ms_p{r}{TAG}"
        return f"ladder_{case}_{fld}_p{r}{TAG}"

    return k


def switch_as_run(ab: pathlib.Path) -> dict:
    """{side: {unit: DS41_WOA_BANK as the runner got it}} from the sides' batch plans (WA.5): a segment's own prefix
    wins over the side's env (orchestrator.ab_batches: "as the shell"), so a gate command that names the switch runs
    that value on BOTH sides. None = the unit sets the switch itself (test_woa_offon) or no plan."""
    out = {}
    for s in SIDES:
        got = {}
        for f in sorted((ab / s).glob("batch*.json")):
            if not f.name[len("batch") :].split(".")[0].isdigit() or f.name.count(".") != 1:
                continue
            try:
                entries = json.loads(f.read_text()).get("entries", [])
            except (OSError, ValueError, AttributeError):
                continue
            for e in entries:
                got[e.get("group", e.get("name"))] = e.get("env", {}).get("DS41_WOA_BANK")
        out[s] = got
    return out


def render(ab: pathlib.Path) -> str:
    sk8 = {s: _metrics(ab / s, lambda n: n.startswith("test_skip_positions") and "5120" not in n) for s in SIDES}
    sk5 = {s: _metrics(ab / s, lambda n: n.startswith("test_skip_positions") and "5120" in n) for s in SIDES}
    pw = {s: _metrics(ab / s, lambda n: n.startswith("test_packed_walls")) for s in SIDES}
    wo = {s: _metrics(ab / s, lambda n: n.startswith("test_woa_offon")) for s in SIDES}
    nr = {s: _metrics(ab / s, lambda n: n.startswith("test_no_rebuild")) for s in SIDES}
    out = [
        "# WA.4: wo_a bank-local reads, the accept measurement (DS41_WOA_BANK off vs on)",
        "",
        "Rendered by tests/woa/woa_ladder_report.py from the A/B runs of WA.4 (" + str(ab) + "; old = off, new = on,",
        "the gate's commands, one resident load a side's chunk size, the switch call-time). Ladder method as PB.LAD",
        "(tests/packed/ladder_common.py): steady = mid-request chunk, resumed = first chunk after a prefix hit; 10 timed",
        "runs, median, program-cache misses refused; device ms = one profiled run (slowest chip); exposed host = wall -",
        "device. Layers 0-20 (DECODER_SKIP=decode). `-` = that side has not run (yet).",
        "",
    ]
    sw = switch_as_run(ab)
    same = sorted(u for u, v in sw.get("old", {}).items() if v is not None and v == sw.get("new", {}).get(u, v))
    if same:
        out += [
            "**Switch as run (the sides' batch plans):** the off side ran "
            + ", ".join(f"{u} with DS41_WOA_BANK={sw['old'][u]}" for u in same)
            + ": the gate command's own `DS41_WOA_BANK=on` prefix wins over the side's env (orchestrator.ab_batches,"
            " a segment's prefix wins). Those tables are on vs on: their deltas are session noise, not the change."
            " The one real off vs on is test_woa_offon (it switches per call itself).",
            "",
        ]
    for title, ms, key in (
        ("Single request, 8192 tokens (test_skip_positions, BRINGUP_CHUNK=8192)", sk8, _single_key),
        ("Chip-aligned 4 x 2048 (lanes4x2k, test_packed_walls)", pw, _lane_key("lanes4x2k")),
        ("4 x 1792 + 1024 split (lanes4x1792s, test_packed_walls)", pw, _lane_key("lanes4x1792s")),
        ("Single request, 5120 tokens (test_skip_positions, BRINGUP_CHUNK=5120: 160 rows a chip)", sk5, _single_key),
    ):
        out += [f"## {title}", ""] + _table(ms, key) + [""]
    out[-1:] = []
    out += ["", "## test_woa_offon (one load, all 40 layers, DECODER_SKIP=off; H = 139264)", ""]
    w = wo["new"] or wo["old"]
    out.append(
        "| case | off == on | on repeat | device ms off / on | wo_a ms off / on | host us a wo_a call off / on |"
    )
    out.append("|---|---|---|---|---|---|")
    for case in ("single8192", "4x2k", "single5120"):
        g = lambda k: w.get(f"{k}{TAG}")  # noqa: E731
        out.append(
            f"| {case} | {_f(g(f'woa_offon_max_abs_diff_{case}'), 1)} | {_f(g(f'woa_repeat_max_abs_diff_{case}'), 1)} | "
            f"{_f(g(f'woa_device_ms_{case}_off'), 3)} / {_f(g(f'woa_device_ms_{case}_on'), 3)} | "
            f"{_f(g(f'woa_sec_ms_{case}_off'), 3)} / {_f(g(f'woa_sec_ms_{case}_on'), 3)} | "
            f"{_f(g(f'woa_host_us_off_{case}'), 1)} / {_f(g(f'woa_host_us_on_{case}'), 1)} |"
        )
    out += [
        "",
        f"real160 (op level, real inputs, 160-row knobs): off vs on {_f(w.get('woa_offon_max_abs_diff_real160' + TAG), 1)}.",
        f"Load: bank copy - interleaved copy {_f(w.get('woa_load_delta_s' + TAG), 2)} s for the layers above "
        f"(interleaved {_f(w.get('woa_load_interleaved_s' + TAG), 2)} s, bank {_f(w.get('woa_load_bank_s' + TAG), 2)} s, "
        f"host prep common {_f(w.get('woa_load_host_s' + TAG), 2)} s).",
        "",
        "## Cold no-rebuild (test_no_rebuild, strict)",
        "",
    ]
    for s, name in (("old", "off"), ("new", "on")):
        p = {k: v for k, v in nr[s].items() if k.startswith("no_rebuild_pass_")}
        b = {k: v for k, v in nr[s].items() if k.startswith("program_builds_after_warmup_")}
        out.append(
            f"- {name}: no_rebuild_pass {sorted(set(p.values())) or '-'} over {len(p)} paths, program builds after "
            f"warm-up {sorted(set(b.values())) or '-'}."
        )
    return "\n".join(out) + "\n"


def keep_verdict(old: str) -> str:
    """The hand-written verdict section of an existing WA.4_ladder.md (kept across re-renders)."""
    i = old.find("## Verdict data")
    j = old.find("Rendered by tests/woa/woa_ladder_report.py")
    return old[i:j].rstrip() + "\n\n" if 0 <= i < j else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", default=str(AB))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    text = render(pathlib.Path(a.ab))
    if a.print:
        print(text)
    else:
        out = pathlib.Path(a.out)
        keep = keep_verdict(out.read_text()) if out.exists() else ""
        if keep:
            head, _, body = text.partition("\n\n")
            text = head + "\n\n" + keep + body
        out.write_text(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
