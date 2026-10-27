#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Consolidate native IWYU output and link both kernel analyzers from one site.

Usage: render_kernel_analysis_site.py <per-leg artifact directory> <site directory>
CodeChecker's HTML must already be in <site>/clang-tidy. No third-party packages
or browser-side data downloads are needed; the IWYU report is standalone HTML.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html import escape
import json
import os
from pathlib import Path
import posixpath
import re

from brand_kernel_tidy_site import FAVICON

HEADING = re.compile(r"^(.+) should (add|remove) these lines:$")
WHEEL_ROOT = re.compile(r"^/opt/venv/lib/python\d+\.\d+/site-packages/ttnn/")
DIAGNOSTIC = re.compile(r"^(.+?):(\d+):(\d+): ((?:fatal )?error:.*)$")
REMOVAL_LINES = re.compile(r"\s+// lines (\d+)-(\d+)\s*$")


def source_path(path: str) -> str:
    """Checkout and installed wheel copies refer to the same repository file."""
    return posixpath.normpath(WHEEL_ROOT.sub("", path.removeprefix("/work/")))


@dataclass
class Report:
    # Key: file, action, suggested source line. Value: reported removal ranges.
    # Locations are retained without making duplicate findings or tracking groups.
    findings: dict[tuple[str, str, str], set[tuple[int, int]]] = field(default_factory=dict)
    errors: set[str] = field(default_factory=set)
    reports: int = 0
    incomplete: bool = False


def read_output(path: Path, report: Report) -> None:
    source = action = None
    with path.open(errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            heading = HEADING.fullmatch(line)
            diagnostic = DIAGNOSTIC.fullmatch(line)
            if diagnostic:
                file, row, col, message = diagnostic.groups()
                report.errors.add(f"{source_path(file)}:{row}:{col}: {message}")
                source = action = None
            elif re.match(r"^(?:[\w./+-]+: )?(?:fatal )?error:", line):
                report.errors.add(line)
                source = action = None
            elif heading:
                source, action = source_path(heading[1]), heading[2]
            elif not line or line == "---" or line.startswith("The full include-list for "):
                source = action = None
            elif source and action:
                if action == "remove":
                    if not line.startswith("- "):
                        source = action = None
                        continue
                    line = line[2:]
                # Deduplicate the suggestion independently of its explanation
                # and location, but retain all reported removal line ranges.
                code = re.split(r"\s+//", line, maxsplit=1)[0].strip()
                if code:
                    locations = report.findings.setdefault((source, action, code), set())
                    location = REMOVAL_LINES.search(line) if action == "remove" else None
                    if location:
                        locations.add((int(location[1]), int(location[2])))


def collect(legs: Path) -> Report:
    report = Report()
    for leg in sorted(legs.glob("*/")):
        output = leg / "iwyu.txt"
        if output.is_file():
            report.reports += 1
            read_output(output, report)
            status = leg / "iwyu-exit-code.txt"
            if not status.is_file() or status.read_text().strip() != "0":
                report.incomplete = True
        else:
            commands = leg / "compile_commands.json"
            if commands.is_file():
                try:
                    report.incomplete |= bool(json.loads(commands.read_text()))
                except (ValueError, OSError):
                    report.incomplete = True
    report.incomplete |= bool(report.errors)
    return report


STYLE = """
:root { color-scheme: light dark; --bg: #f6f5fa; --panel: #fff; --text: #252334;
  --muted: #625e73; --line: #ddd9e8; --accent: #5944c5; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1200px; padding: 40px 24px 80px; margin: auto; }
a { color: var(--accent); }
h1 { font-size: 2rem; letter-spacing: -.035em; margin: 16px 0 8px; }
h2 { font-size: 1.25rem; } h3 { font-size: 1rem; margin: 0 0 12px; }
p { margin: 8px 0 20px; } .muted { color: var(--muted); }
.cards, .changes { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.card, details { border: 1px solid var(--line); background: var(--panel); border-radius: 10px; }
.card { padding: 24px; } .card h2 { margin-top: 0; }
.notice { border-left: 4px solid #c78b19; padding: 14px 18px; background: var(--panel); }
.search { margin: 28px 0 20px; } label { display: block; font-weight: 600; margin-bottom: 8px; }
input { width: 100%; padding: 12px; font: inherit; color: var(--text);
  background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
input:focus-visible, a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
details { margin-bottom: 12px; } summary { padding: 16px; cursor: pointer; overflow-wrap: anywhere; }
summary code { font-size: .88rem; } .changes { padding: 0 20px 20px; }
.add h3 { color: #187344; } .remove h3 { color: #af3442; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: .85rem; margin: 0; }
.diagnostics pre { padding: 0 20px 20px; } [hidden] { display: none !important; }
@media (max-width: 720px) { .cards, .changes { grid-template-columns: 1fr; } main { padding: 24px 16px; } }
@media (prefers-color-scheme: dark) {
  :root { --bg: #17151f; --panel: #211e2c; --text: #eeeaf7; --muted: #b6afc8;
    --line: #423b53; --accent: #b6a8ff; }
  .add h3 { color: #78d7a4; } .remove h3 { color: #ff9da8; }
}
"""

SEARCH = """
<script>
const files = Array.from(document.querySelectorAll('.file')).map(element => ({
  element, text: element.textContent.toLowerCase()
}));
document.querySelector('#search').addEventListener('input', event => {
  const query = event.target.value.trim().toLowerCase();
  let visible = 0;
  for (const {element, text} of files) {
    const matches = text.includes(query);
    element.hidden = !matches;
    element.open = Boolean(query) && matches;
    if (matches) visible++;
  }
  document.querySelector('#visible').textContent = `${visible} ${visible === 1 ? 'file' : 'files'} shown`;
  document.querySelector('#no-matches').hidden = visible !== 0;
});
</script>
"""


def page(title: str, content: str, *, parent: bool = False, script: str = "") -> str:
    icon = "../favicon.svg" if parent else "favicon.svg"
    nav = '<a href="../index.html">All kernel reports</a>' if parent else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · tt-metal</title><link rel="icon" href="{icon}">
<style>{STYLE}</style></head>
<body><main>{nav}<h1>{escape(title)}</h1>{content}</main>{script}</body></html>
"""


def render_iwyu(report: Report, enabled: bool) -> str:
    if not enabled:
        return page("Kernel Include What You Use", "<p>IWYU was disabled for this run.</p>", parent=True)
    sections: dict[str, dict[str, list[str]]] = {}
    for source, action, code in sorted(report.findings):
        suggestion = escape(code)
        locations = sorted(report.findings[(source, action, code)])
        if locations:
            ranges = ", ".join(str(start) if start == end else f"{start}–{end}" for start, end in locations)
            label = "line" if len(locations) == 1 and locations[0][0] == locations[0][1] else "lines"
            suggestion += f' <span class="muted">// {label} {ranges}</span>'
        sections.setdefault(source, {"add": [], "remove": []})[action].append(suggestion)
    content = '<p class="muted">Consolidated include and forward-declaration recommendations.</p>'
    if report.incomplete:
        content += (
            '<p class="notice"><strong>Partial analysis.</strong> Some IWYU invocations failed or '
            "did not produce a report or exit status. Recommendations may be incomplete. "
            "Raw output remains in the per-leg workflow artifacts.</p>"
        )
    if not report.reports:
        content += "<p>No IWYU reports were collected.</p>"
    elif not sections:
        content += "<p>No include changes were reported in the collected output.</p>"
    content += f"""<div class="search"><label for="search">Find a file, include, or declaration</label>
<input id="search" type="search" placeholder="e.g. reader_unary.cpp or &lt;cstdint&gt;">
<p id="visible" class="muted" role="status">{len(sections)} files shown</p></div>
<p id="no-matches" hidden>No matching recommendations.</p>"""
    for source, changes in sections.items():
        content += f'<details class="file"><summary><code>{escape(source)}</code></summary><div class="changes">'
        for action, title in (("add", "Add"), ("remove", "Remove")):
            code = "\n".join(changes[action])
            lines = f"<pre><code>{code}</code></pre>" if code else '<p class="muted">None reported.</p>'
            content += f'<section class="{action}"><h3>{title}</h3>{lines}</section>'
        content += "</div></details>"
    if report.errors:
        errors = escape("\n".join(sorted(report.errors)))
        content += f'<details class="diagnostics"><summary>Parsing errors</summary><pre>{errors}</pre></details>'
    return page("Kernel Include What You Use", content, parent=True, script=SEARCH)


def render_site(legs: Path, site: Path, *, iwyu_enabled: bool = True) -> bool:
    report = collect(legs) if iwyu_enabled else Report()
    (site / "iwyu").mkdir(parents=True, exist_ok=True)
    (site / "iwyu/index.html").write_text(render_iwyu(report, iwyu_enabled))
    (site / "favicon.svg").write_text(FAVICON)
    (site / ".nojekyll").touch()
    tidy = site / "clang-tidy"
    tidy_index = tidy / "index.html"
    tidy_ready = tidy_index.is_file() and tidy_index.stat().st_size > 0 and any(tidy.glob("*.plist.html"))
    tidy_link = (
        '<a href="clang-tidy/index.html">Open clang-tidy report →</a>' if tidy_ready else "HTML report unavailable."
    )
    iwyu_status = "IWYU was disabled for this run." if not iwyu_enabled else "Include and forward-declaration changes."
    if iwyu_enabled and report.incomplete:
        iwyu_status = "Partial analysis; some invocations failed or produced incomplete output."
    elif iwyu_enabled and not report.reports:
        iwyu_status = "No IWYU reports were collected."
    content = f"""<p class="muted">Static analysis of captured JIT kernel compilations.</p>
<div class="cards">
<section class="card"><h2>clang-tidy</h2><p>Code diagnostics and source details.</p>{tidy_link}</section>
<section class="card"><h2>Include What You Use</h2><p>{iwyu_status}</p>
<a href="iwyu/index.html">Open IWYU report →</a></section></div>"""
    if os.environ.get("GITHUB_RUN_ID"):
        url = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
        content += (
            f'<p class="muted">Commit <code>{escape(os.environ.get("GITHUB_SHA", "")[:12])}</code> · '
            f'<a href="{escape(url, quote=True)}">Workflow run</a></p>'
        )
    (site / "index.html").write_text(page("Kernel analysis reports", content))
    summary = f"IWYU: {len(report.findings)} distinct recommendations in {len({f[0] for f in report.findings})} files."
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(
                f"\n## Consolidated kernel reports\n\n{summary}\n\nOpen `index.html` in the report-site artifact.\n"
            )
    return tidy_ready or bool(report.reports)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legs", type=Path)
    parser.add_argument("site", type=Path)
    parser.add_argument("--skip-iwyu", action="store_true", help="show IWYU as disabled on clang-tidy-only runs")
    args = parser.parse_args()
    ready = render_site(args.legs, args.site, iwyu_enabled=not args.skip_iwyu)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"found={str(ready).lower()}\n")


if __name__ == "__main__":
    main()
