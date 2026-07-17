#!/usr/bin/env python3
"""Group release-notes PR lines by code layer (MINFRA-975).

Rewrites the given changelog file in place. Each PR line is identified by
its github.com/<owner>/<repo>/pull/<n> link and placed under an H2 header
for every layer it touches; layer membership is decided by the paths of
the files the PR changed (so a PR spanning layers is listed under each).
"""
import json
import os
import re
import sys
import urllib.request

# (header, path prefixes) in priority order; first matching prefix wins per file.
LAYERS = [
    ("LLK (low-level kernels)", ["tt_metal/tt-llk/", "tt_metal/hw/ckernels/"]),
    ("Metalium (tt-metal core)", ["tt_metal/", "tests/tt_metal/"]),
    ("TT-NN", ["ttnn/", "tests/ttnn/", "tests/tt_eager/"]),
    ("tt-train", ["tt-train/"]),
    ("Models", ["models/", "model_tracer/", "tests/models/"]),
    ("TT-STL", ["tt_stl/"]),
    (
        "Infrastructure & CI",
        [
            ".github/",
            "infra/",
            "scripts/",
            "cmake/",
            "dockerfile/",
            "tests/scripts/",
            "tests/pipeline_reorg/",
            "releases/",
        ],
    ),
    ("Documentation", ["docs/", "tech_reports/", "contributing/"]),
    ("Tooling", ["tools/"]),
]
OTHER = "Other"
ORDER = [h for h, _ in LAYERS] + [OTHER]
PR_LINK = re.compile(r"\]\(https://github\.com/([^/]+)/([^/]+)/pull/(\d+)\)")


def layer_of(path):
    for header, prefixes in LAYERS:
        if any(path.startswith(p) for p in prefixes):
            return header
    return OTHER


def changed_files(owner, repo, number, token):
    files, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}" f"/files?per_page=100&page={page}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        batch = json.load(urllib.request.urlopen(req, timeout=30))
        files += [f["filename"] for f in batch]
        if len(batch) < 100:
            return files
        page += 1


def main(path):
    token = os.environ.get("GITHUB_TOKEN", "")
    groups = {}  # header -> PR lines, changelog order preserved
    found = False
    for line in open(path, encoding="utf-8").read().splitlines():
        m = PR_LINK.search(line)
        if not m:
            continue
        found = True
        layers = {layer_of(f) for f in changed_files(*m.groups(), token)}
        for header in layers or {OTHER}:
            groups.setdefault(header, []).append(line)
    if not found:
        return  # nothing recognisable; leave the changelog untouched
    out = []
    for header in ORDER:
        if groups.get(header):
            out += [f"## {header}", *groups[header], ""]
    open(path, "w", encoding="utf-8").write("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
