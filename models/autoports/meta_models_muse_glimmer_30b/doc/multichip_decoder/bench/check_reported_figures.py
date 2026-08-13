# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the mechanically-sourced figures in ``README.md`` and the contract.

The single-chip stage of this model had four consecutive review rounds find the
same defect class and nothing else: a number in a report that no longer matched
the CSV it came from.  ``refresh_context_contract.py`` regenerates the PCC and
test-count blocks but not the performance block or any prose, so this closes the
rest of the gap.

It checks, against the committed artifacts rather than against itself:

* every per-window device time, from ``tracy/*/*_perf_report.csv``;
* the decode op-group breakdown in "Performance accounting";
* the warmed end-to-end latencies and the single-chip baseline, from the
  committed ``logs/layer_ab_*.log`` lines;
* the speedups, as baseline / multichip;
* the per-device weight and KV-cache byte budget, from the model config;
* the multichip-vs-single-chip PCC, from ``logs/vs_single_chip_run.log``.

    python check_reported_figures.py [--check]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/multichip_decoder/
README = ROOT / "README.md"
WORK_LOG = ROOT / "work_log.md"
CONTRACT = ROOT.parent / "context_contract.json"
SOURCE = ROOT.parent.parent / "tt/multichip_decoder.py"

TOLERANCE = 0.005  # 0.5 % on measured quantities, exact on byte counts


class Failures(list):
    def check(self, label: str, reported: float, derived: float, *, exact: bool = False) -> None:
        if exact:
            ok = reported == derived
        else:
            ok = abs(reported - derived) <= TOLERANCE * max(abs(derived), 1e-9)
        status = "ok " if ok else "STALE"
        print(f"{status} {label:52s} reported={reported!r:>14} derived={derived!r:>14}")
        if not ok:
            self.append(label)


def device_time_us(window: str, iters: int) -> float:
    """Sum of the Device Time column of a committed tt-perf-report CSV, per iteration."""
    import pandas as pd

    frame = pd.read_csv(ROOT / f"tracy/{window}_perf_report.csv")
    return float(frame["Device Time"].sum()) / iters


def op_groups(window: str, iters: int) -> dict[str, float]:
    import pandas as pd

    frame = pd.read_csv(ROOT / f"tracy/{window}_perf_report.csv")
    codes = frame["OP Code"].astype(str)
    groups = {
        "matmul": codes.str.startswith(("MatmulDeviceOperation", "MinimalMatmulDeviceOperation")),
        "collective": codes.str.contains("ReduceScatter|AllGather|AllReduce"),
        "norm": codes.str.contains("LayerNorm"),
        "binary": codes.str.contains("BinaryNg"),
        "sdpa": codes.str.contains("Sdpa|SDPA"),
    }
    return {name: float(frame.loc[mask, "Device Time"].sum()) / iters for name, mask in groups.items()}


def ab_row(log: str, candidate: str, kind: str) -> tuple[float, float]:
    """``(prefill ms, traced decode ms/token)`` from a committed layer_ab line."""
    pattern = re.compile(
        rf"^AB\S*\s+{re.escape(candidate)}\s+kind={kind}\s+.*?prefill\d+=\s*([0-9.]+) ms.*?"
        rf"traced_decode@\d+=\s*([0-9.]+) ms/token"
    )
    for line in (ROOT / "logs" / log).read_text(errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1)), float(match.group(2))
    raise SystemExit(f"no AB line for {candidate}/{kind} in {log}")


def readme_numbers(pattern: str) -> list[float]:
    text = README.read_text()
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise SystemExit(f"README pattern not found: {pattern}")
    return [float(value.replace(",", "")) for value in match.groups()]


def matmul_dram_rows(window: str) -> dict[str, tuple[float, float]]:
    """Per-role mean ``(GB/s, % of peak)`` over every replay in a decode capture.

    The report emits one row per op *instance*, so a role appears once per replay
    and the README quotes the mean.  Roles are keyed on what the CSV actually
    carries -- the weight dtype, the inner-dim block size, and (for the two BFP8
    rows that share both) the FLOP count, since ``wqkv``'s per-device N is 1280
    against ``attn_gate``'s 1024.
    """
    import pandas as pd

    frame = pd.read_csv(ROOT / f"tracy/{window}_perf_report.csv")
    frame = frame[frame["OP Code"].astype(str).str.contains("Matmul", na=False)].copy()
    for column in ("DRAM", "DRAM %", "FLOPs", "Inner Dim Block Size"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    def role(row) -> str:
        dtype, block = row["Input 1 Datatype"], row["Inner Dim Block Size"]
        if dtype == "BFLOAT8_B":
            return "o_proj" if block == 2 else ("wqkv" if row["FLOPs"] > 23.5 else "attn_gate")
        return "mlp_gate_up" if block == 13 else "mlp_down"

    frame["role"] = frame.apply(role, axis=1)
    return {
        name: (round(float(group["DRAM"].mean()), 2), round(float(group["DRAM %"].mean()), 2))
        for name, group in frame.groupby("role")
    }


def cited_figure_violations(doc: pathlib.Path) -> list[str]:
    """Every measurement a doc attributes to a log, that is in none of them.

    This is the gate the first four review rounds needed and did not have. Three
    rounds running, a correction landed in one document while another kept
    quoting the superseded number, and twice a regenerated log left a table
    behind still citing it. Nothing mechanical could see either.

    The unit is a **block** -- a run of non-blank lines, i.e. a table or a
    paragraph. Every block that names one or more ``logs/...`` artifacts is
    checked: each measurement in it must appear literally in at least one of the
    artifacts that block cites.

    "Measurement" is three decimals or more (``0.4573``), or two decimals at a
    magnitude of 5 or more (``8598.18``, ``27.01``, ``5.33``). That rule selects
    what a probe prints and skips what a document is supposed to compute --
    ratios (``1.16x``), percentages (``0.24 %``) and small differences, none of
    which are in any log by construction.

    A block that is *deliberately* historical opts out with a
    ``<!-- superseded -->`` comment, which forces that choice to be written down
    rather than merely being true on the day.
    """
    root = doc.parent
    violations = []
    blocks = re.split(r"\n\s*\n", doc.read_text())
    for index, block in enumerate(blocks):
        if "<!-- superseded" in block:
            continue
        # A table's citation is usually in the sentence that introduces it, one
        # block up, so a table inherits the previous block's artifacts.  Without
        # this the round-4 findings -- a whole table gone stale under a
        # regenerated log -- are invisible to the gate.
        scope = block
        if block.lstrip().startswith("|") and index:
            scope = blocks[index - 1] + "\n" + block
        cited = sorted(set(re.findall(r"`+(logs/[\w./-]+)`+", scope)))
        haystack = "".join((root / name).read_text(errors="ignore") for name in cited if (root / name).exists())
        if not haystack:
            continue  # nothing cited, or the cited-artifact-exists check owns it
        measured = None
        for value in sorted(set(re.findall(r"\d+\.\d{2,}", block))):
            if len(value.split(".")[1]) < 3 and float(value) < 5:
                continue
            if value in haystack:
                continue
            # A doc may quote a rounding of a full-precision log value on
            # purpose -- PCC prints 17 digits and the tables show 6.
            if measured is None:
                measured = [float(v) for v in re.findall(r"\d+\.\d+", haystack)]
            places = len(value.split(".")[1])
            target = float(value)
            if any(round(number, places) == target for number in measured):
                continue
            head = " ".join(block.split())[:70]
            violations.append(f"{doc.name}: {value} in none of {', '.join(cited)} -- {head}...")
    return violations


def doc_anchors(text: str) -> set[str]:
    """Every in-document anchor GitHub would generate for a markdown body."""
    anchors = set()
    for line in text.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip().lower()
            anchors.add(re.sub(r"[^a-z0-9 _-]", "", title).replace(" ", "-"))
    return anchors


def source_constant(name: str) -> int:
    """An int constant's value, read out of the shipped module's text.

    Text, not an import: this checker must run without a device, and importing
    ``multichip_decoder`` pulls in ttnn.
    """
    match = re.search(rf"^{name} = (\d+)", SOURCE.read_text(), re.MULTILINE)
    if not match:
        raise SystemExit(f"no constant {name} in {SOURCE}")
    return int(match.group(1))


def section_numbers(text: str) -> set[str]:
    return {match.group(1) for match in re.finditer(r"^#+ (\d+(?:\.\d+[a-z]?)?)\.?\s", text, re.MULTILINE)}


def readme_headings() -> set[str]:
    return doc_anchors(README.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on any stale figure (default behaviour)")
    parser.parse_args()
    failures = Failures()

    # ---- warmed end-to-end, and the baseline it is divided by -----------------
    base_prefill_s, base_decode_s = ab_row("layer_ab_single_baseline.log", "single", "sliding")
    base_prefill_f, base_decode_f = ab_row("layer_ab_single_baseline.log", "single", "full")
    mc_prefill_s, mc_decode_s = ab_row("layer_ab_final.log", "tp4", "sliding")
    mc_prefill_f, mc_decode_f = ab_row("layer_ab_final.log", "tp4", "full")

    reported = readme_numbers(
        r"traced decode, sliding @2048 \| ([0-9.]+) \| \*\*([0-9.]+) ms/token\*\* \| \*\*([0-9.]+)x\*\*"
    )
    failures.check("result: decode sliding baseline", reported[0], base_decode_s)
    failures.check("result: decode sliding multichip", reported[1], mc_decode_s)
    failures.check("result: decode sliding speedup", reported[2], round(base_decode_s / mc_decode_s, 2))

    reported = readme_numbers(
        r"traced decode, full @2048 \| ([0-9.]+) \| \*\*([0-9.]+) ms/token\*\* \| \*\*([0-9.]+)x\*\*"
    )
    failures.check("result: decode full baseline", reported[0], base_decode_f)
    failures.check("result: decode full multichip", reported[1], mc_decode_f)
    failures.check("result: decode full speedup", reported[2], round(base_decode_f / mc_decode_f, 2))

    reported = readme_numbers(r"prefill 8192, sliding \| ([0-9.]+) \| \*\*([0-9.]+) ms\*\* \| \*\*([0-9.]+)x\*\*")
    failures.check("result: prefill sliding baseline", reported[0], base_prefill_s)
    failures.check("result: prefill sliding multichip", reported[1], mc_prefill_s)
    failures.check("result: prefill sliding speedup", reported[2], round(base_prefill_s / mc_prefill_s, 2))

    reported = readme_numbers(r"prefill 8192, full \| ([0-9.]+) \| \*\*([0-9.]+) ms\*\* \| \*\*([0-9.]+)x\*\*")
    failures.check("result: prefill full baseline", reported[0], base_prefill_f)
    failures.check("result: prefill full multichip", reported[1], mc_prefill_f)
    failures.check("result: prefill full speedup", reported[2], round(base_prefill_f / mc_prefill_f, 2))

    # ---- device time per window ----------------------------------------------
    windows = {
        "decode sliding @2048": ("sliding/decode_2048", 8),
        "decode full @2048": ("full/decode_2048", 8),
        "decode sliding @131071": ("sliding/decode_131071", 8),
        "decode full @131071": ("full/decode_131071", 8),
    }
    reported = readme_numbers(
        r"decode sliding @2048 \| 1072 μs \| \*\*([0-9.]+) μs\*\*.*?"
        r"decode full @2048 \| 1049 \| \*\*([0-9.]+)\*\*.*?"
        r"decode sliding @131071 \| 1071 \| \*\*([0-9.]+)\*\*.*?"
        r"decode full @131071 \| 1255 \| \*\*([0-9.]+)\*\*"
    )
    for value, (label, (window, iters)) in zip(reported, windows.items()):
        failures.check(f"device: {label}", value, round(device_time_us(window, iters), 1))

    reported = readme_numbers(
        r"prefill 128, sliding / full \| 2140 / 2096 \| \*\*([0-9.]+) / ([0-9.]+)\*\*.*?"
        r"prefill 8192, sliding / full \| 37762 / 36606 \| \*\*([0-9.]+) / ([0-9.]+)\*\*"
    )
    for value, window in zip(
        reported, ("sliding/prefill_128", "full/prefill_128", "sliding/prefill_8192", "full/prefill_8192")
    ):
        failures.check(f"device: {window}", value, round(device_time_us(window, 1), 1))

    # ---- the decode op-group breakdown ---------------------------------------
    groups = op_groups("sliding/decode_2048", 8)
    reported = readme_numbers(
        r"\| 6 matmuls \| ([0-9.]+) \|.*?"
        r"\| 2 reductions \(`ReduceScatter` \+ `AllGather`\) \| ([0-9.]+) \|.*?"
        r"\| 4 hidden-size RMSNorms \| ([0-9.]+) \|.*?"
        r"\| 4 elementwise \(`BinaryNg`\) \| ([0-9.]+) \|.*?"
        r"\| `SdpaDecode` \| ([0-9.]+) \|"
    )
    failures.check("decode groups: matmul", reported[0], round(groups["matmul"], 1))
    failures.check("decode groups: collective", reported[1], round(groups["collective"], 1))
    # The norm row is the four hidden-size norms only; the two per-head QK norms
    # are the separate 1-core rows, listed on their own line in the same table.
    qk = readme_numbers(r"\| 2 per-head QK norms \| ([0-9.]+) \|")[0]
    failures.check("decode groups: norms (hidden + QK)", reported[2] + qk, round(groups["norm"], 1))
    failures.check("decode groups: binary", reported[3], round(groups["binary"], 1))
    failures.check("decode groups: sdpa", reported[4], round(groups["sdpa"], 1))

    # ---- the byte budget ------------------------------------------------------
    bfp8, bfp4 = 1.0625, 0.5625
    weights = int(6656 * 1280 * bfp8 + 6656 * 1024 * bfp8 + 1024 * 6656 * bfp8 + 3 * 6656 * 5120 * bfp4)
    cache = int(2 * 1 * 128 * 131072 * bfp8)
    contract = json.loads(CONTRACT.read_text())
    failures.check(
        "budget: per-device weight bytes",
        contract["byte_budget_at_full_context"]["per_device_layer_weight_bytes"],
        weights,
        exact=True,
    )
    failures.check(
        "budget: per-device KV bytes",
        contract["byte_budget_at_full_context"]["per_device_kv_cache_bytes_per_layer_batch1"],
        cache,
        exact=True,
    )
    failures.check(
        "budget: weight ratio",
        contract["byte_budget_at_full_context"]["weight_ratio"],
        round(314802176 / weights, 3),
    )
    readme_weights = readme_numbers(r"3 x 19,169,280 \(MLP\) = \*\*([0-9,]+) B\*\*")[0]
    failures.check("README: per-device weight bytes", readme_weights, weights, exact=True)

    # ---- the contract's performance block against the same sources ------------
    performance = contract["performance"]
    failures.check(
        "contract: decode sliding e2e", performance["traced_decode_ms_per_token_e2e"]["sliding@2048"], mc_decode_s
    )
    failures.check("contract: decode full e2e", performance["traced_decode_ms_per_token_e2e"]["full@2048"], mc_decode_f)
    failures.check("contract: prefill sliding e2e", performance["prefill_ms_e2e"]["8192_sliding"], mc_prefill_s)
    failures.check("contract: prefill full e2e", performance["prefill_ms_e2e"]["8192_full"], mc_prefill_f)
    for label, (window, iters) in windows.items():
        key = label.replace("decode ", "").replace(" @", "@").replace(" ", "")
        failures.check(
            f"contract: device {key}",
            performance["traced_decode_us_device"][key],
            round(device_time_us(window, iters), 1),
        )

    # ---- multichip vs single chip --------------------------------------------
    # The comparison labels carry the shape they were measured at, e.g.
    # "decode2[full seq_len=12345 batch=4]", and the README reports one row per
    # (kind, shape) with that row's worst.
    log = (ROOT / "logs/vs_single_chip_run.log").read_text(errors="ignore")
    for kind, seq_len, batch in (("sliding", 2049, 1), ("full", 2049, 1), ("sliding", 12345, 4), ("full", 12345, 4)):
        case = rf"\[{kind} seq_len={seq_len} batch={batch}\]"
        values = [float(v) for v in re.findall(rf"vs single-chip TTNN \w+{case}: ([0-9.]+)", log)]
        assert values, f"no vs-single-chip lines for {kind}/{seq_len}/{batch}"
        row = rf"\| {kind}, {seq_len} tokens, batch {batch} \| ([0-9.]+) \| [0-9.–-]+ \| \*\*([0-9.]+)\*\* \|"
        failures.check(
            f"vs single-chip worst[{kind} {seq_len} b{batch}]", readme_numbers(row)[1], round(min(values), 6)
        )

    # ---- the DRAM/%-of-peak table --------------------------------------------
    # Round-2 review: this table was quoted from eyeballed instances (one value
    # was a minimum, not the mean), and nothing re-derived it.
    dram = matmul_dram_rows("sliding/decode_2048")
    for role, row in (
        ("wqkv", r"\(`wqkv`\) \| BFP8 \| ([0-9.]+) GB/s \| ([0-9.]+) %"),
        ("attn_gate", r"\(`attn_gate`\) \| BFP8 \| ([0-9.]+) GB/s \| ([0-9.]+) %"),
        ("o_proj", r"\(`o_proj`\) \| BFP8 \| ([0-9.]+) \| ([0-9.]+) %"),
        ("mlp_gate_up", r"\(gate, up\) \| BFP4 \| ([0-9.]+) \| ([0-9.]+) %"),
        ("mlp_down", r"\(`mlp_down`\) \| BFP4 \| ([0-9.]+) \| ([0-9.]+) %"),
    ):
        reported_gbs, reported_pct = readme_numbers(row)
        failures.check(f"dram table: {role} GB/s", reported_gbs, dram[role][0])
        failures.check(f"dram table: {role} % of peak", reported_pct, dram[role][1])

    # ---- every artifact the prose cites, and every anchor it links to ---------
    text = README.read_text()
    missing_logs = sorted({name for name in re.findall(r"`(logs/[\w./-]+)`", text) if not (ROOT / name).exists()})
    failures.check("README: cited artifacts that exist", len(missing_logs), 0, exact=True)
    if missing_logs:
        print("   missing: " + ", ".join(missing_logs))
    anchors = readme_headings()
    dangling = sorted({a for a in re.findall(r"\]\(#([\w-]+)\)", text) if a not in anchors})
    failures.check("README: in-document links that resolve", len(dangling), 0, exact=True)
    if dangling:
        print("   dangling: " + ", ".join(dangling))

    # ---- the work log, which no gate read until round 3 ----------------------
    # Every finding that survived round 2 lived in work_log.md.  It is prose, so
    # it cannot be re-derived wholesale -- but the three defect classes that
    # actually occurred can be.
    work_log = WORK_LOG.read_text()
    cited = {name for name in re.findall(r"`+(logs/[\w./-]+)`+", work_log) if "..." not in name}
    missing = sorted({name for name in cited if not (ROOT / name).exists()})
    failures.check("work log: cited artifacts that exist", len(missing), 0, exact=True)
    if missing:
        print("   missing: " + ", ".join(missing))
    sections = section_numbers(work_log)
    referenced = {ref for ref in re.findall(r"§(\d+(?:\.\d+[a-z]?)?)", work_log)}
    unresolved = sorted(referenced - sections)
    failures.check("work log: section references that resolve", len(unresolved), 0, exact=True)
    if unresolved:
        print("   unresolved: " + ", ".join("§" + ref for ref in unresolved))

    # ---- prose against the shipped constants ---------------------------------
    # The round-3 P1 was work_log.md documenting l1_small_size 4096 as shipped
    # while the code shipped 6144.  Every doc that names a shipped value for one
    # of these three knobs must name the value the module actually defines.
    for name, pattern, label in (
        # "opens with"/"ships at" -- not the ladder rows, which quote values on purpose
        ("DEFAULT_L1_SMALL_SIZE", r"(?:opens with|ships at|shipped) `l1_small_size ?= ?(\d+)`", "l1_small_size"),
        ("FABRIC_PACKET_PAYLOAD_BYTES", r"an? \*?\*?(\d+) B\*?\*? fabric packet", "fabric packet"),
    ):
        expected = source_constant(name)
        for doc_name, text in (("README", README.read_text()), ("work log", work_log)):
            quoted = {int(value) for value in re.findall(pattern, text)}
            wrong = sorted(quoted - {expected})
            failures.check(f"{doc_name}: shipped {label} == {name}", len(wrong), 0, exact=True)
            if wrong:
                print(f"   quotes {wrong} but the module defines {expected}")

    # ---- every figure a doc attributes to a log is in that log ---------------
    violations = cited_figure_violations(README) + cited_figure_violations(WORK_LOG)
    failures.check("docs: quoted figures found in the cited artifact", len(violations), 0, exact=True)
    for violation in violations:
        print(f"   {violation}")

    print()
    if failures:
        print(f"{len(failures)} stale figure(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    print("every mechanically-sourced figure matches its artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
