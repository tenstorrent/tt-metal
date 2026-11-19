# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in proof of four exact clang-format transitions, never outcome acceptance."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "doc/functional_decoder/plans/format19_transition_v1.json"
MANIFEST_SHA = "6780ad2066643ae8a952c3623a96de4bb14d24b6f050c861c13830ae652605ad"
BINDING = {"kind": "clang_format_19_1_4_v1", "manifest_sha256": MANIFEST_SHA}
SUFFIXES = {".py", ".cpp", ".hpp", ".h"}
BRACE_EDITS = {
    "tt/kernels/gdn_compute_sfpu.cpp": [
        ("if(emit_state) final.reserve_back(16);", "if(emit_state) { final.reserve_back(16); }"),
        (
            "if(emit_state) pack_tile<true>(0,cb_state,ki*4+vi);",
            "if(emit_state) { pack_tile<true>(0,cb_state,ki*4+vi); }",
        ),
        ("if(emit_state) final.push_back(16);", "if(emit_state) { final.push_back(16); }"),
    ],
    "tt/kernels/router_topk_algorithm.hpp": [
        ("if (an || bn) return an && !bn;", "if (an || bn) { return an && !bn; }"),
        ("if (aa == 0 && bb == 0) return false;", "if (aa == 0 && bb == 0) { return false; }"),
        (
            "if ((a.bits ^ b.bits) & 0x80000000u) return !(a.bits & 0x80000000u);",
            "if ((a.bits ^ b.bits) & 0x80000000u) { return !(a.bits & 0x80000000u); }",
        ),
        (
            "if (router_greater(q[first + child], q[first + child - 1])) --child;",
            "if (router_greater(q[first + child], q[first + child - 1])) { --child; }",
        ),
        (
            "for (int parent = (count - 2) / 2; parent >= 0; --parent)\n            router_heap_adjust(q, first, parent, count, q[first + parent]);",
            "for (int parent = (count - 2) / 2; parent >= 0; --parent) { router_heap_adjust(q, first, parent, count, q[first + parent]); }",
        ),
        ("while (router_greater(q[left], q[first])) ++left;", "while (router_greater(q[left], q[first])) { ++left; }"),
        (
            "while (router_greater(q[first], q[right])) --right;",
            "while (router_greater(q[first], q[right])) { --right; }",
        ),
        ("if (left >= right) break;", "if (left >= right) { break; }"),
        (
            "if (left <= 9) first = left;\n        else end = left;",
            "if (left <= 9) { first = left; } else { end = left; }",
        ),
    ],
}
# Limited lexical comparison is supplemental to exact byte hashes and the pinned
# formatter output. It is not an equivalence test for arbitrary C++ programs.
TOKENS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?)(?:[eE][+-]?\d+)?[uUlLfF]*|>>=|<<=|->\*|\.\.\.|::|->|\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s]',
    re.S,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic_tokens(text):
    return [x for x in TOKENS.findall(text) if not x.startswith(("//", "/*"))]


def verify_lexical_transition(name, before, after):
    original = before.decode()
    changed = original
    for old, new in BRACE_EDITS.get(name, []):
        require(changed.count(old) == 1, f"Expected unique reviewed brace site: {name}")
        changed = changed.replace(old, new)
    require(semantic_tokens(changed) == semantic_tokens(after.decode()), f"Unexpected lexical change: {name}")
    directives = lambda s: [line for line in s.splitlines() if line.lstrip().startswith("#")]
    require(directives(original) == directives(after.decode()), f"Preprocessor/include order changed: {name}")
    require(not any(x in semantic_tokens(original) for x in ("__LINE__", "__COUNTER__")), "Location-dependent token")


class FormatSourceClosure:
    """An explicit, immutable binding; source hashes remain historical in reports."""

    def __init__(self, root, binding=None):
        self.root = Path(root)
        self.proof = None
        self.files = {}
        if binding is None:
            return
        require(type(binding) is dict and binding == BINDING, "Unknown/stale format-source binding")
        manifest_path = ROOT / MANIFEST
        raw = manifest_path.read_bytes()
        require(digest(raw) == MANIFEST_SHA, "Pinned formatter manifest changed")
        manifest = json.loads(raw)
        formatter = Path(manifest["formatter"]["path"])
        require(digest(formatter.read_bytes()) == manifest["formatter"]["sha256"], "Formatter binary changed")
        require(
            subprocess.check_output([str(formatter), "--version"], text=True, timeout=10).strip()
            == "clang-format version 19.1.4",
            "Formatter version changed",
        )
        config = Path(manifest["configuration"]["path"])
        require(digest(config.read_bytes()) == manifest["configuration"]["sha256"], "Formatter config changed")
        baseline_raw = (ROOT / "tests/diagnostic_sources/linear_runtime_before_ordered.json").read_bytes()
        require(
            digest(baseline_raw) == "08f69155bc2c1ae32862da5a11f70e17d1e4c74352597ac462a7df14e3417440",
            "Runtime baseline changed",
        )
        baseline = json.loads(baseline_raw)
        added = {
            "tt/ordered_projection.py",
            "tt/kernels/ordered_projection_reader.cpp",
            "tt/kernels/ordered_projection_compute.cpp",
            "tt/kernels/ordered_projection_writer.cpp",
        }
        inventory = {
            str(p.relative_to(self.root)) for p in (self.root / "tt").rglob("*") if p.is_file() and p.suffix in SUFFIXES
        }
        require(inventory == set(baseline) | added, "Runtime inventory differs from exact 26-file format profile")
        expected_runtime = dict(baseline)
        expected_runtime.update(
            {
                "tt/qsa.py": "0c312ab20a33d245dc6956dfa7ad36560d9206ceb8048cb5f9105f8ca71ea15f",
                "tt/ordered_projection.py": "a952d765473f0cb3e68205d9b588a2344726fabae5b30a2107035c13b255e4e4",
                "tt/kernels/ordered_projection_compute.cpp": "c9701ee9c1acb1c209463cf6cc19b6acc4e4fa397a0ba87f1ff62c45f2df9c77",
                "tt/kernels/ordered_projection_reader.cpp": "f04e14870d5a81a6fd1b7b6922fbbe1eee9c8b196b77bcac53291fb0e45f992b",
                "tt/kernels/ordered_projection_writer.cpp": "72c000b5775221692fda5422db0f746d1c9ed1b436d36cd75a9269dc1b1e73a9",
            }
        )
        for name, expected in expected_runtime.items():
            if name not in manifest["files"]:
                require(
                    digest((self.root / name).read_bytes()) == expected,
                    f"Unapproved shared runtime source changed: {name}",
                )
        states = set()
        verified = {}
        for name, item in manifest["files"].items():
            before = (ROOT / item["before_snapshot"]).read_bytes()
            after = (ROOT / item["after_snapshot"]).read_bytes()
            require(
                digest(before) == item["before_sha256"] and digest(after) == item["after_sha256"],
                f"Preserved formatter snapshot changed: {name}",
            )
            canonical = subprocess.check_output(
                [str(formatter), f"--style=file:{config}", f"--assume-filename={self.root / name}"],
                input=before,
                timeout=10,
            )
            require(canonical == after, f"Canonical formatter output changed: {name}")
            verify_lexical_transition(name, before, after)
            current = digest((self.root / name).read_bytes())
            require(
                current in (item["before_sha256"], item["after_sha256"]), f"Unapproved current format source: {name}"
            )
            states.add("before" if current == item["before_sha256"] else "after")
            verified[name] = {
                "recorded_before_sha256": item["before_sha256"],
                "canonical_after_sha256": item["after_sha256"],
                "current_sha256": current,
            }
        require(len(states) == 1, "Partially applied formatter transition")
        self.files = manifest["files"]
        self.proof = {
            "binding": dict(binding),
            "verifier_sha256": digest(Path(__file__).read_bytes()),
            "state": next(iter(states)),
            "files": verified,
            "formatter": manifest["formatter"],
            "configuration": manifest["configuration"],
            "runtime_inventory": sorted(inventory),
            "recorded_hashes_preserved": True,
            "source_semantics_only": True,
            "compiled_binary_identity_asserted": False,
            "stage_acceptance": "not_evaluated_review_required",
        }

    def verify(self, name, recorded_sha, message="Required source identity changed"):
        require(
            isinstance(recorded_sha, str) and len(recorded_sha) == 64, f"Missing historical source identity: {name}"
        )
        current = digest((self.root / name).read_bytes())
        if current == recorded_sha:
            return
        entry = self.files.get(name)
        require(
            entry is not None and recorded_sha == entry["before_sha256"] and current == entry["after_sha256"],
            f"{message}: {name}",
        )
