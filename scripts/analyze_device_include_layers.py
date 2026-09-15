#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Find suspicious includes in the device header stack.

Two questions are answered from one graph:

  layering  a header that should be invariant reaches into content that is
            specific to the kernel instance being compiled -- generated
            headers (chlkc_*, *_generated.h), the kernel source itself, or
            compile_time_args.h, whose kernel_compile_time_args array is
            materialised from the per-kernel KERNEL_COMPILE_TIME_ARGS macro.

  cycles    strongly connected components of the include graph.

The graph is harvested by re-running the pinned SFPI cross-compiler with
-E -H over compile commands captured from a real JIT build
(TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1), so macro-gated and
__has_include-gated edges resolve exactly as they did for the device. Text
scanning cannot do this: the two worst offenders found so far sit behind
`#if __has_include("chlkc_descriptors.h")`, which is a function of -I order
rather than of anything visible in the file.

-H reports the include *tree* a translation unit actually read, so a header
already pulled in under its guard is not re-reported. A single TU therefore
yields a spanning tree, not the full graph: edges can be missed, but none are
invented. Unioning many real TUs recovers the rest, and --list-unobserved
prints the directives the compiler never exercised so the gap stays visible.
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The capture format, its SGR stripping and its split-define repair are already
# written and self-tested in the clang-tidy flow; reuse rather than re-derive.
_SIBLING = Path(__file__).with_name("build_kernel_clang_tidy_commands.py")


def _load_capture_module():
    spec = importlib.util.spec_from_file_location("kernel_tidy_capture", _SIBLING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Layers, ascending in how much the content varies per compile. The rule is that
# nothing at DEVICE_LIB or below may include KERNEL_INSTANCE or above.
VENDOR, PLATFORM, DEVICE_LIB, KERNEL_INSTANCE, KERNEL_SOURCE, TU = range(6)
LAYER_NAMES = {
    VENDOR: "vendor",
    PLATFORM: "platform",
    DEVICE_LIB: "device-lib",
    KERNEL_INSTANCE: "kernel-instance",
    KERNEL_SOURCE: "kernel-source",
    TU: "translation-unit",
}

# Written by tt_metal/jit_build into the per-kernel cache dir.
GENERATED = {
    "chlkc_descriptors.h",
    "defines_generated.h",
    "kernel_args_generated.h",
    "kernel_bindings_generated.h",
    "kernel_includes.hpp",
    "named_args_generated.h",
    "named_ct_arg_map_generated.h",
}
# Static file, per-kernel content: see module docstring.
INSTANCE_BY_CONTENT = {"compile_time_args.h"}

KERNEL_DIR_RE = re.compile(r"/(kernels|kernels_ng|kernels_dfb|test_kernels)/")
H_LINE_RE = re.compile(r"^(\.+)\s+(\S.*)$")
# gcc -H ends with a guard advisory whose payload lines carry no dot prefix.
H_STOP = "Multiple include guards may be useful for:"

DROP_EXACT = {"-c", "-MMD", "-MD", "-MP", "-flto", "-fpch-preprocess", "-Winvalid-pch"}
DROP_WITH_ARG = {"-o", "-MF", "-MT", "-MQ", "-include"}
DROP_PREFIX = ("-flto=", "-Werror")


GENERATED_NODE = "<jit-generated>/"
VENDORED_IN_TREE = ("runtime/sfpi/", "tt_metal/third_party/")


def is_generated(path):
    name = os.path.basename(path)
    return name in GENERATED or name.startswith("chlkc_")


def canon(path):
    """Collapse per-kernel copies of a generated header to one logical node.

    jit_build writes these into the kernel's own cache directory, so the same
    #include yields a different absolute path for every kernel compiled. The
    layering question is about the directive, not the copy.
    """
    return GENERATED_NODE + os.path.basename(path) if is_generated(path) else path


def classify(path, repo_root, is_tu):
    """Assign a volatility layer to a canonical node."""
    if is_tu:
        return TU
    name = os.path.basename(path)
    # Generated files live under a per-kernel ".../kernels/<name>/<hash>/" path,
    # so they must be recognised before the kernel-directory test below.
    if path.startswith(GENERATED_NODE) or is_generated(path) or name in INSTANCE_BY_CONTENT:
        return KERNEL_INSTANCE
    if KERNEL_DIR_RE.search(path):
        # Kernel sources also live outside the checkout (e.g. a downstream
        # repo's ops/*/kernels), so this precedes the repo-root test.
        return KERNEL_SOURCE
    if not path.startswith(repo_root):
        return VENDOR  # /usr/include, an out-of-tree toolchain
    rel = os.path.relpath(path, repo_root)
    if rel.startswith(VENDORED_IN_TREE):
        # The pinned SFPI release ships its own newlib and libstdc++ inside the
        # checkout, so being under the repo root does not make it ours to fix.
        return VENDOR
    if rel.startswith("tt_metal/hostdevcommon/"):
        return PLATFORM
    return DEVICE_LIB


def harvest_cmd(argv, keep_pch):
    """Turn a captured compile command into a preprocess-and-report-includes run."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in DROP_WITH_ARG and not (keep_pch and a == "-include"):
            i += 2
            continue
        if a in DROP_EXACT or a.startswith(DROP_PREFIX):
            i += 1
            continue
        out.append(a)
        i += 1
    # -E stops after preprocessing: the include tree is fully realised and the
    # parse is skipped, which is the whole cost. -H writes the tree to stderr.
    return out + ["-E", "-H", "-o", os.devnull]


def config_key(entry, resolved_includes):
    """Two commands with the same preprocessor inputs yield the same graph."""
    argv = entry["arguments"]
    defines = tuple(sorted(a for a in argv if a.startswith("-D")))
    flags = tuple(a for a in argv if a.startswith(("-mcpu=", "-std=", "-ftt-")))
    return (os.path.basename(entry["file"]), defines, resolved_includes, flags)


def resolved_include_key(entry):
    argv, cwd = entry["arguments"], entry["directory"]
    paths, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "-I" and i + 1 < len(argv):
            paths.append(os.path.realpath(os.path.join(cwd, argv[i + 1])))
            i += 2
            continue
        if a.startswith("-I") and len(a) > 2:
            paths.append(os.path.realpath(os.path.join(cwd, a[2:])))
        elif a == "-isystem" and i + 1 < len(argv):
            paths.append(os.path.realpath(os.path.join(cwd, argv[i + 1])))
            i += 2
            continue
        i += 1
    return tuple(paths)


def parse_h_output(stderr, cwd, tu_abs):
    """gcc -H stderr -> (edges, files). Depth is the count of leading dots."""
    edges, files = set(), set()
    stack = [tu_abs]
    for line in stderr.splitlines():
        if line.startswith(H_STOP):
            break
        m = H_LINE_RE.match(line)
        if not m:
            continue
        depth = len(m.group(1))
        path = os.path.realpath(os.path.join(cwd, m.group(2).strip()))
        del stack[depth:]
        while len(stack) < depth:
            # Depth jumped: a parent line was suppressed. Do not guess an edge.
            stack.append(None)
        parent = stack[depth - 1] if depth - 1 < len(stack) else None
        stack.append(path)
        files.add(path)
        if parent is not None:
            edges.add((parent, path))
    return edges, files


def run_one(entry, keep_pch, timeout):
    argv = harvest_cmd(entry["arguments"], keep_pch)
    tu_abs = os.path.realpath(os.path.join(entry["directory"], entry["file"]))
    try:
        p = subprocess.run(
            argv,
            cwd=entry["directory"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"{tu_abs}: {e}"
    edges, files = parse_h_output(p.stderr, entry["directory"], tu_abs)
    # -H prints as it reads, so a later hard error does not invalidate the tree
    # already reported; only an empty tree means the run told us nothing.
    if not files:
        return None, f"{tu_abs}: no include tree (rc={p.returncode})"
    return (tu_abs, edges, files), None


def tarjan_scc(nodes, succ):
    """Iterative Tarjan; returns components with more than one member."""
    index, low, on_stack, stack, out = {}, {}, set(), [], []
    counter = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ.get(root, ())))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(succ.get(nxt, ()))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1:
                    out.append(comp)
    return out


INCLUDE_DIRECTIVE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)


def text_directives(files):
    """Every #include spelled in the reachable files, conditional or not."""
    spelled = defaultdict(set)
    for f in files:
        try:
            text = Path(f).read_text(errors="replace")
        except OSError:
            continue
        for m in INCLUDE_DIRECTIVE_RE.finditer(text):
            spelled[f].add(m.group(1))
    return spelled


def rel(p, repo_root):
    return os.path.relpath(p, repo_root) if p.startswith(repo_root) else p


def self_test():
    """Assert the classification, canonicalisation and -H parsing invariants.

    Runs without a toolchain, a capture or a device, so CI can gate the logic
    even though producing a real graph needs a JIT build.
    """
    root = "/repo/"
    cache = "/cache/tt-metal-cache/1/kernels/some_op/9/"

    # A generated header lives under a per-kernel ".../kernels/<name>/<hash>/"
    # path, so the kernel-directory test must not claim it first.
    assert classify(cache + "chlkc_descriptors.h", root, False) == KERNEL_INSTANCE
    assert classify(cache + "kernel_includes.hpp", root, False) == KERNEL_INSTANCE
    assert classify(GENERATED_NODE + "chlkc_descriptors.h", root, False) == KERNEL_INSTANCE
    # Static file, per-kernel content: the array comes from KERNEL_COMPILE_TIME_ARGS.
    assert classify(root + "tt_metal/hw/inc/api/compile_time_args.h", root, False) == KERNEL_INSTANCE
    # Kernel sources also live outside the checkout, so that test precedes the
    # repo-root fallback rather than falling through to VENDOR.
    assert classify("/elsewhere/ops/foo/kernels/op.hpp", root, False) == KERNEL_SOURCE
    assert classify(root + "ttnn/cpp/ttnn/operations/x/kernels/reader.cpp", root, False) == KERNEL_SOURCE
    # The pinned SFPI release ships newlib and libstdc++ inside the checkout.
    assert classify(root + "runtime/sfpi/compiler/riscv-tt-elf/include/c++/15.1.0/cstdint", root, False) == VENDOR
    assert classify("/usr/include/stdio.h", root, False) == VENDOR
    assert classify(root + "tt_metal/hostdevcommon/api/x.hpp", root, False) == PLATFORM
    assert classify(root + "tt_metal/hw/inc/internal/firmware_common.h", root, False) == DEVICE_LIB
    assert classify(root + "tt_metal/hw/firmware/src/tt-1xx/brisck.cc", root, True) == TU

    # Per-kernel copies of one generated header collapse to a single node; a
    # tracked file keeps its path.
    assert canon(cache + "chlkc_descriptors.h") == canon("/other/kernels/b/2/chlkc_descriptors.h")
    assert canon(root + "tt_metal/hw/inc/api/dataflow/dataflow_api.h").startswith(root)

    # gcc -H: dot depth gives nesting. The trailing guard advisory lists bare
    # paths and must not be read as depth-0 includes.
    h = "\n".join(
        [
            ". /repo/a.h",
            ".. /repo/b.h",
            "... /repo/c.h",
            ".. /repo/d.h",
            "Multiple include guards may be useful for:",
            "/repo/b.h",
        ]
    )
    edges, files = parse_h_output(h, "/repo", "/repo/tu.cc")
    assert ("/repo/tu.cc", "/repo/a.h") in edges
    assert ("/repo/a.h", "/repo/b.h") in edges
    assert ("/repo/b.h", "/repo/c.h") in edges
    assert ("/repo/a.h", "/repo/d.h") in edges
    assert files == {"/repo/a.h", "/repo/b.h", "/repo/c.h", "/repo/d.h"}, files
    assert not any(e[1] == "/repo/b.h" for e in edges if e[0] == "/repo/tu.cc")

    # A depth jump means a parent line was suppressed; do not invent an edge.
    edges, _ = parse_h_output(". /repo/a.h\n... /repo/deep.h", "/repo", "/repo/tu.cc")
    assert ("/repo/a.h", "/repo/deep.h") not in edges
    assert len(edges) == 1

    # Preprocess-and-report: the object, the dependency file and a forced PCH
    # include all go; -D and -I survive because they decide resolution.
    out = harvest_cmd(
        ["g++", "-c", "-MMD", "-o", "x.o", "-include", "pch.h", "-DFOO=1", "-I.", "-Iinc", "x.cc"],
        keep_pch=False,
    )
    assert "-c" not in out and "-MMD" not in out
    assert "x.o" not in out and "pch.h" not in out and "-include" not in out
    assert out[-4:] == ["-E", "-H", "-o", os.devnull]
    assert "-DFOO=1" in out and "-I." in out and "-Iinc" in out and "x.cc" in out
    assert "-include" in harvest_cmd(["g++", "-include", "pch.h", "x.cc"], keep_pch=True)

    # Tarjan: report the cycle, and only the cycle.
    succ = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}, "e": set()}
    comps = tarjan_scc(sorted(succ), succ)
    assert len(comps) == 1 and set(comps[0]) == {"a", "b", "c"}, comps
    assert tarjan_scc(["x", "y"], {"x": {"y"}, "y": set()}) == []

    print("[layers] self-test PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-log", help="Run log captured with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1")
    src.add_argument("--input", help="compile_commands.json of SFPI compile entries")
    src.add_argument("--self-test", action="store_true", help="Run built-in fixture assertions and exit")
    ap.add_argument("--repo-root", help="Checkout the capture was built from (default: inferred)")
    ap.add_argument("--limit", type=int, default=0, help="Cap deduplicated configs (0 = all)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--keep-pch", action="store_true", help="Keep -include (default: strip, for the canonical graph)")
    ap.add_argument("--list-unobserved", action="store_true", help="Also list directives no config exercised")
    ap.add_argument("--top", type=int, default=40, help="Findings to print per section")
    ap.add_argument("--graph-cache", help="Reuse (or write) the harvested graph so reports can be re-run cheaply")
    ap.add_argument("--reharvest", action="store_true", help="Ignore an existing --graph-cache and re-preprocess")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    cap = _load_capture_module()
    if args.input_log:
        entries = cap.entries_from_log(args.input_log)
    else:
        raw = json.load(open(args.input))
        entries = [
            {
                "directory": e["directory"],
                "file": e["file"],
                "arguments": e.get("arguments") or e["command"].split(),
            }
            for e in raw
        ]
    entries = [e for e in entries if cap.is_sfpi_compile(e["arguments"])]
    print(f"[layers] captured SFPI compiles: {len(entries)}")
    if not entries:
        return 1

    repo_root = args.repo_root
    if not repo_root:
        for e in entries:
            for a in e["arguments"]:
                if a.startswith("-I") and a.endswith("/tt_metal/hw/inc"):
                    repo_root = a[2:][: -len("/tt_metal/hw/inc")]
                    break
            if repo_root:
                break
    if not repo_root or not os.path.isdir(repo_root):
        print("[layers] could not infer --repo-root", file=sys.stderr)
        return 1
    repo_root = os.path.realpath(repo_root).rstrip("/") + "/"
    print(f"[layers] repo root: {repo_root}")

    dedup = {}
    for e in entries:
        dedup.setdefault(config_key(e, resolved_include_key(e)), e)
    configs = list(dedup.values())
    if args.limit:
        configs = configs[: args.limit]
    print(f"[layers] preprocessor-distinct configs: {len(configs)} (from {len(entries)})")

    graph = None
    if args.graph_cache and os.path.exists(args.graph_cache) and not args.reharvest:
        with open(args.graph_cache) as f:
            graph = json.load(f)
        print(f"[layers] loaded harvested graph from {args.graph_cache}")

    if graph is None:
        results, failures = [], []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(run_one, e, args.keep_pch, args.timeout) for e in configs]
            for n, f in enumerate(futs, 1):
                ok, err = f.result()
                (results if ok else failures).append(ok or err)
                if n % 250 == 0:
                    print(f"[layers]   harvested {n}/{len(futs)}", flush=True)
        print(f"[layers] include trees harvested: {len(results)}, failed: {len(failures)}")
        for f in failures[:5]:
            print(f"[layers]   ! {f}")
        if not results:
            return 1

        # Aggregate to counts up front: per-config sets are what make this
        # expensive to keep, and every section below needs only how often a
        # node appeared and how often an edge's endpoints co-occurred.
        file_count = defaultdict(int)
        edge_cooccur = defaultdict(int)
        tus = set()
        for tu, edges, files in results:
            tus.add(canon(tu))
            seen = {canon(p) for p in files} | {canon(tu)}
            for p in seen:
                file_count[p] += 1
            for a, b in {(canon(a), canon(b)) for a, b in edges}:
                edge_cooccur[(a, b)] += 1
        graph = {
            "repo_root": repo_root,
            "configs": len(results),
            "tus": sorted(tus),
            "file_count": dict(file_count),
            "edges": [[a, b, n] for (a, b), n in edge_cooccur.items()],
        }
        if args.graph_cache:
            with open(args.graph_cache, "w") as f:
                json.dump(graph, f)
            print(f"[layers] saved harvested graph to {args.graph_cache}")

    repo_root = graph["repo_root"]
    total = graph["configs"]
    file_count = graph["file_count"]
    tus = set(graph["tus"])
    edges = {(a, b): n for a, b, n in graph["edges"]}
    print(f"[layers] graph: {len(file_count)} nodes, {len(edges)} edges, over {total} configs")

    layer = {p: classify(p, repo_root, p in tus) for p in set(file_count) | {x for e in edges for x in e}}
    succ = defaultdict(set)
    for a, b in edges:
        succ[a].add(b)

    def gating(a, b):
        """Configs in which the parent was read but the child never appeared.

        -H omits a header already read under its guard, so a missing edge is
        not evidence the directive was skipped. Only the child's absence from
        the entire tree shows the include genuinely did not fire.
        """
        parent = file_count.get(a, 0)
        return parent, max(0, parent - edges.get((a, b), 0))

    print(f"\n{'=' * 78}\nLAYERING VIOLATIONS  (device-lib or lower -> kernel-instance or higher)\n{'=' * 78}")
    viol = sorted(
        ((file_count.get(a, 0), a, b) for a, b in edges if layer[a] <= DEVICE_LIB and layer[b] >= KERNEL_INSTANCE),
        reverse=True,
    )
    if not viol:
        print("  none")
    for fanin, a, b in viol[: args.top]:
        pc, supp = gating(a, b)
        gate = "unconditional" if not supp else f"gated: child absent in {supp}/{pc} of the parent's configs"
        print(f"  {rel(a, repo_root)}")
        print(f"      -> {rel(b, repo_root)}  [{LAYER_NAMES[layer[b]]}]")
        print(f"         blast radius: parent read by {fanin}/{total} configs; {gate}")
    if len(viol) > args.top:
        print(f"  ... {len(viol) - args.top} more")

    print(f"\n{'=' * 78}\nINCLUDE CYCLES\n{'=' * 78}")
    comps = tarjan_scc(sorted(succ), succ)
    if not comps:
        print("  none")
    for comp in sorted(comps, key=len, reverse=True):
        vendor = all(layer[p] == VENDOR for p in comp)
        print(f"  component of {len(comp)}{' (vendor, not ours)' if vendor else ''}:")
        for p in sorted(comp):
            print(f"      {rel(p, repo_root)}")

    print(f"\n{'=' * 78}\nCONFIG-GATED EDGES IN THE DEVICE STACK  (parent read, child never reached)\n{'=' * 78}")
    vol = []
    for a, b in edges:
        if layer[a] == VENDOR or layer[b] == VENDOR:
            continue
        pc, supp = gating(a, b)
        if supp and pc:
            vol.append((supp, pc, a, b))
    vol.sort(reverse=True)
    for supp, pc, a, b in vol[: args.top]:
        print(f"  {rel(a, repo_root)}")
        print(f"      -> {rel(b, repo_root)}   absent in {supp}/{pc} of the parent's configs")
    if not vol:
        print("  none")
    elif len(vol) > args.top:
        print(f"  ... {len(vol) - args.top} more")

    # The compiler-observed graph attributes a child to whichever parent read it
    # first, so a second parent of the same header yields no edge. Text-scan the
    # device stack for the layering rule specifically, to recover those.
    print(f"\n{'=' * 78}\nLAYERING VIOLATIONS RECOVERED BY TEXT  (real directive, edge never attributed)\n{'=' * 78}")
    instance_names = GENERATED | INSTANCE_BY_CONTENT
    observed_viol = {(a, os.path.basename(b)) for a, b in edges if layer[b] >= KERNEL_INSTANCE}
    recovered = []
    for p in sorted(file_count):
        if layer[p] > DEVICE_LIB or not os.path.isfile(p):
            continue
        try:
            text = Path(p).read_text(errors="replace")
        except OSError:
            continue
        for m in INCLUDE_DIRECTIVE_RE.finditer(text):
            name = os.path.basename(m.group(1))
            if (name in instance_names or name.startswith("chlkc_")) and (p, name) not in observed_viol:
                recovered.append((file_count.get(p, 0), p, m.group(1)))
    recovered.sort(reverse=True)
    if not recovered:
        print("  none")
    for fanin, p, spelled in recovered[: args.top]:
        print(f"  {rel(p, repo_root)}")
        print(f"      -> {spelled}  [kernel-instance, by directive]")
        print(f"         blast radius: parent read by {fanin}/{total} configs")

    if args.list_unobserved:
        print(f"\n{'=' * 78}\nUNOBSERVED DIRECTIVES  (spelled in a device header, never exercised)\n{'=' * 78}")
        in_tree = [p for p in file_count if layer[p] in (PLATFORM, DEVICE_LIB) and os.path.isfile(p)]
        spelled = text_directives(sorted(in_tree))
        observed = defaultdict(set)
        for a, b in edges:
            observed[a].add(os.path.basename(b))
        n = 0
        for f, names in sorted(spelled.items()):
            for s in sorted(names):
                if os.path.basename(s) in observed[f]:
                    continue
                if n < args.top:
                    print(f"  {rel(f, repo_root)} -> {s}")
                n += 1
        print(f"  ({n} total)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
