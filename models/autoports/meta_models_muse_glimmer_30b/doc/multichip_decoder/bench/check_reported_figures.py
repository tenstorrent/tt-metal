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
import functools
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
    labels: list[str] = []

    def check(self, label: str, reported: float, derived: float, *, exact: bool = False) -> None:
        self.labels.append(label)
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


def pcc_populations() -> tuple[int, dict[str, int], dict[str, str]]:
    """``(total, {bar: count}, {bar: worst})`` from the generated PCC summary."""
    text = (ROOT / "logs/pcc_summary.txt").read_text()
    total = int(re.search(r"^(\d+) asserted checks", text, re.MULTILINE).group(1))
    rows = re.findall(r"^\s+(\d+) .*?bar (0\.\d+), worst (0\.\d+)", text, re.MULTILINE)
    return total, {bar: int(n) for n, bar, _ in rows}, {bar: worst for _, bar, worst in rows}


def script_to_logs() -> dict[str, list[str]]:
    """``bench/x.py`` -> the ``logs/*.log`` the committed chain scripts redirect it to.

    Round 5's P1 got past the first version of the cited-figure check by citing
    the *script* rather than its log, so a stale table was never compared with
    anything.  The mapping is read out of the chain scripts rather than guessed,
    with a stem fallback for probes no chain runs.
    """
    mapping: dict[str, set[str]] = {}
    for script in sorted((ROOT / "bench").glob("*.sh")):
        for call, log in re.findall(r"bench/(\w+\.py)\"?[^\n]*?>\s*\"?\$D/(logs/[\w./-]+)\"?", script.read_text()):
            mapping.setdefault(f"bench/{call}", set()).add(log)
    for probe in sorted((ROOT / "bench").glob("*.py")):
        guess = f"logs/{probe.stem}.log"
        if (ROOT / guess).exists():
            mapping.setdefault(f"bench/{probe.name}", set()).add(guess)
    return {name: sorted(logs) for name, logs in mapping.items()}


def alone_marker(block: str, kind: str) -> re.Match | None:
    """A marker reaches the *following* block only when it is alone on its own.

    Round 8: a ``verified-by`` line sitting inside a numbered list item muted the
    next item as well, silently dropping the block that carries the real-weight
    margin.  A marker that shares a block with prose applies to that block only.
    """
    if block.strip().startswith("<!--") and block.strip().endswith("-->"):
        return re.search(rf"<!-- {kind}:\s*(.+?)-->", block, re.DOTALL)
    return None


@functools.lru_cache(maxsize=None)
def artifact_numbers(haystack: str) -> frozenset[float]:
    return frozenset(float(v) for v in re.findall(r"\d+\.\d+", haystack))


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
    root = doc.parent if doc.parent.name == "multichip_decoder" else ROOT
    bars = set(pcc_populations()[1])
    scripts = script_to_logs()
    violations = []
    verified_by: list[tuple[str, str, str]] = []
    suppressed = 0
    # Numbered list items are separate claims with separate citations even
    # though no blank line separates them, so they are separate blocks.
    blocks = [part for chunk in re.split(r"\n\s*\n", doc.read_text()) for part in re.split(r"\n(?=\d+\. )", chunk)]
    for index, block in enumerate(blocks):
        # A marker on its own line is its own block, so it has to suppress the
        # block that *follows* it as well as itself -- round 6 found three
        # markers suppressing nothing but themselves.  At line start only:
        # §15.1 describes the marker in prose, inside backticks, and that is not
        # a use of it.
        # Like the superseded marker, a delegation on its own line has to reach
        # the block it labels.
        verified = re.search(r"^<!-- verified-by:\s*(.+?)-->", block, re.MULTILINE) or (
            index and alone_marker(blocks[index - 1], "verified-by")
        )
        if verified:
            verified_by.append((doc.name, verified.group(1).strip(), " ".join(block.split())[:60]))
            continue
        marker_here = re.search(r"^<!-- superseded", block, re.MULTILINE)
        marker_above = index and alone_marker(blocks[index - 1], "superseded")
        if marker_here or marker_above:
            # The marker must say why and point somewhere; an empty one would be
            # a silent mute for any live claim sharing the block.
            reason = re.search(r"<!-- superseded:\s*(.+?)-->", block if marker_here else blocks[index - 1], re.DOTALL)
            if not reason or len(reason.group(1).strip()) < 20:
                violations.append(
                    f"{doc.name}: <!-- superseded --> with no reason -- {' '.join(block.split())[:60]}..."
                )
            suppressed += 1
            continue
        # A block's citation is usually in the sentence that introduces it, but
        # not always the block immediately above: a table can be introduced,
        # then commented on, then printed.  Round 7 found ~25 figure-bearing
        # blocks reaching an empty haystack because inheritance was one block
        # deep.  Walk back to the start of the section instead.
        scope = block
        for previous in reversed(blocks[:index]):
            scope = previous + "\n" + scope
            if re.match(r"\s*#{1,6} ", previous):
                break
        cited = set(re.findall(r"`+((?:\.\./[\w-]+/)?logs/[\w./-]+)`+", scope))
        # A block may cite the probe instead of the log it wrote, or a Tracy CSV.
        for script in re.findall(r"`+(bench/\w+\.py)`+", scope):
            cited.update(scripts.get(script, ()))
        cited.update(re.findall(r"`+(tracy/[\w./-]+\.csv)`+", scope))
        cited = sorted(cited)
        haystack = "".join((root / name).read_text(errors="ignore") for name in cited if (root / name).exists())
        if not haystack:
            continue  # nothing cited, or the cited-artifact-exists check owns it
        measured = None
        for value in sorted(set(re.findall(r"\d+\.\d{2,}", block))):
            if len(value.split(".")[1]) < 3 and float(value) < 5:
                continue
            # A PCC *bar* is a threshold the tests define, not a measurement; the
            # population check verifies the bars themselves.
            if value in bars:
                continue
            # Boundaries, not substring: "9.11" occurs inside "19.11", which
            # silently accepted a planted defect during the round-7 fix.
            if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", haystack):
                continue
            # A doc may quote a rounding of a full-precision log value on
            # purpose -- PCC prints 17 digits and the tables show 6.
            places = len(value.split(".")[1])
            if places >= 4:
                # PCC prints 17 digits and the tables show 6, so a rounding is
                # legitimate.  Only at 4+ places, where a coincidental match in a
                # large log is unlikely -- at 2 places it would accept ~1 value
                # in 9 against logs/full_test_run.log.
                if measured is None:
                    measured = [float(v) for v in re.findall(r"\d+\.\d+", haystack)]
                target = float(value)
                if any(round(number, places) == target for number in measured):
                    continue
            # A doc legitimately *adds* two logged values -- "reduce_scatter +
            # all_gather" is one number in a table and two rows in the log.  Only
            # the addition the block actually writes down is accepted: "a + b =
            # value", with a and b both in the artifact.  Round 6 measured the
            # earlier "any pair in the log" rule at a 29-100 % false-accept rate,
            # which is worse than the rounding shortcut round 5 rejected.
            if any(
                round(float(a) + float(b), places) == round(float(value), places)
                and float(a) in artifact_numbers(haystack)
                and float(b) in artifact_numbers(haystack)
                for a, b in re.findall(rf"(\d+\.\d+)\s*\+\s*(\d+\.\d+)\s*=\s*{re.escape(value)}\b", block)
            ):
                continue
            head = " ".join(block.split())[:70]
            violations.append(f"{doc.name}: {value} in none of {', '.join(cited)} -- {head}...")
    if suppressed:
        print(f"   ({doc.name}: {suppressed} block(s) marked superseded and skipped)")
    return violations, verified_by


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

    # ---- the PCC population counts, which are integers the figure check skips --
    # Round 6: the total was updated in both docs when the K/V test added eight
    # checks and the *row* was not, so the table contradicted its own total.
    total, counts, worsts = pcc_populations()
    for doc_name, path in (("README", README), ("work log", WORK_LOG)):
        text = path.read_text()
        failures.check(
            f"{doc_name}: asserted PCC total", int(re.search(r"(\d+) asserted PCC checks", text).group(1)), total
        )
        rows = {bar: int(n) for n, bar in re.findall(r"\| (\d+) \| (0\.\d+) \|", text)}
        rows = rows or {bar: int(n) for n, bar in re.findall(r"(\d+) [\w\- ]*?\(worst 0\.\d+, bar (0\.\d+)\)", text)}
        # Only the four bars the summary defines -- the docs contain other tables
        # with a "| count | value |" shape.
        for bar, count in sorted(counts.items()):
            if bar in rows:
                failures.check(f"{doc_name}: PCC population at bar {bar}", rows[bar], count)
        quoted = [rows[bar] for bar in counts if bar in rows]
        if len(quoted) == len(counts):
            failures.check(f"{doc_name}: PCC populations sum to the total", sum(quoted), total)
        # ...and each population's worst, which the docs quote next to its bar.
        for bar, worst in sorted(worsts.items()):
            row = re.search(rf"\| \d+ \| {re.escape(bar)} \| \*?\*?(0\.\d+)", text)
            if row:
                failures.check(f"{doc_name}: PCC worst at bar {bar}", float(row.group(1)), float(worst))

    # ---- integer counts, which the figure check skips by design --------------
    # Structural integers (52 layers, 6656 columns) would drown that check, so
    # the counts that have actually gone stale get their own derivation: the
    # watcher artifact's size and dump count (round 5's "50 dumps", round 6's
    # 28,788 lines) and the two suites' pass counts.
    import gzip

    watcher = gzip.open(ROOT / "watcher/watcher.log.gz", "rt", errors="ignore").read()
    watcher_facts = {
        "watcher log lines": watcher.count("\n"),
        "watcher dumps": len(re.findall(r"Dump #\d+ completed", watcher)),
    }
    for doc_name, path in (("README", README), ("work log", WORK_LOG)):
        text = path.read_text().replace(",", "")
        for label, derived in watcher_facts.items():
            pattern = r"(\d+) log lines" if "lines" in label else r"(\d+) dumps"
            quoted = {int(v) for v in re.findall(pattern, text)}
            if quoted:
                failures.check(f"{doc_name}: {label}", sorted(quoted)[0], derived, exact=True)
    # Every bolded "**N passed**" must be one of the runs that actually happened.
    ran = {
        int(re.findall(r"(\d+) passed", (ROOT / log).read_text(errors="ignore"))[-1])
        for log in ("logs/full_test_run.log", "logs/vs_single_chip_run.log", "logs/watcher_run.log")
    }
    for doc_name, path in (("README", README), ("work log", WORK_LOG)):
        quoted = {int(v) for v in re.findall(r"\*\*(\d+) passed\*\*", path.read_text())}
        failures.check(f"{doc_name}: quoted pass counts that ran", len(quoted - ran), 0, exact=True)
        if quoted - ran:
            print(f"   {doc_name} quotes {sorted(quoted - ran)}; the committed runs are {sorted(ran)}")

    # ---- the teardown-fault occurrence count, in three documents -------------
    # Round 8: work_log 9.4 said three, 9.2 said two.  The count is derived from
    # the timestamps 9.4 enumerates and compared against every document that
    # states it, so a future occurrence has one place to land.
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    log_text = WORK_LOG.read_text()
    # The timestamps §9.4 enumerates are the source of truth; the word in front
    # of them is checked against them too, so neither can drift alone.
    listed = re.search(r"Seen (\w+) times \(([^)]+)\)", log_text)
    stamps = re.findall(r"\d{2}:\d{2}", listed.group(2))
    occurrences = len(stamps)
    failures.check(
        "work log: 'Seen N times' matches the timestamps listed", words[listed.group(1)], occurrences, exact=True
    )
    # Round 9: the first version of this check matched only the sentence it
    # derived from, so §9.2's own statement of the count -- the site round 8
    # found stale -- was still hand-maintained.  Every phrasing either document
    # uses is matched now, including the reset total, which is one more than the
    # abort count (the extra reset is the two-modules-in-one-invocation case).
    # ``\s+`` throughout: these documents are hard-wrapped, so any of these
    # phrases can straddle a newline -- which is why round 9's plant of §9.2
    # survived the first version of this check.
    patterns = (
        (r"(?:from|has|Seen)\s+(\w+)(?:\s+more)?\s+(?:occurrences|times)", 0),
        (r"each\s+of\s+the\s+\**(\w+)\**\s+watcher\s+runs'\s+teardown\s+aborts", 0),
        (r"(\w+)\s+more\s+resets\s+were\s+needed", 1),
    )
    for doc_name, text in (
        ("README", README.read_text()),
        ("work log", log_text),
        ("contract", CONTRACT.read_text()),
    ):
        for pattern, offset in patterns:
            for word in re.findall(pattern, text):
                if word.lower() in words:
                    failures.check(
                        f"{doc_name}: teardown count '{word}'", words[word.lower()], occurrences + offset, exact=True
                    )

    # ---- every figure a doc attributes to a log is in that log ---------------
    # The module's own docstring tables cite logs too, and round 5 found a stale
    # figure in one of them that nothing was checking.
    violations, delegated = [], []
    for document in (README, WORK_LOG, SOURCE):
        found, claimed = cited_figure_violations(document)
        violations += found
        delegated += claimed
    failures.check("docs: quoted figures found in the cited artifact", len(violations), 0, exact=True)
    for violation in violations:
        print(f"   {violation}")

    # A block may delegate to a named check that re-derives its figures (a mean
    # over a CSV is not a substring of it).  The delegation is only honoured if
    # that check actually ran and passed -- otherwise it is a mute.
    ran = {label for label in failures.labels}
    # A marker alone on its block registers for that block and the next; report
    # each (document, label) once.
    for doc_name, label, head in {(d, l): (d, l, h) for d, l, h in delegated}.values():
        matched = [name for name in ran if name.startswith(label)]
        ok = bool(matched) and not any(name in failures for name in matched)
        failures.check(f"{doc_name}: verified-by '{label}' ran and passed", int(ok), 1, exact=True)
        if not ok:
            print(f"   no passing check named '{label}' for -- {head}...")

    print()
    if failures:
        print(f"{len(failures)} stale figure(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    print("every mechanically-sourced figure matches its artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
