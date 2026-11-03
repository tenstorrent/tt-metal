#!/usr/bin/env python3
"""bd_map.py: map PC samples of router ERISCs to the report's cycle-breakdown categories, by source line.

Library for build_breakdown.py, and a CLI for checking a map against a sample set:

  bd_map.py check SET_ID [--sets FILE] [--top N]     coverage, top unmapped lines, debatable rules, per category
  bd_map.py lines SET_ID [--sets FILE] [--top N] [--target C/CH/E]   hottest innermost lines with their inline chains
  bd_map.py resolve MAP [--sources LAYERS]            where each rule lands in a source version (unresolved = re-map)

Method (fabric_exp/report/BREAKDOWN_PLAN.md):
- Samples: pc_sampler.py output (results/<prefix>_samples.{bin,json}). A target's image is found in the run's kernel
  cache by the launch message's text size (= the packed size of the ELF's loadable segments) and the first 256 bytes
  read from L1, against the relocated `<proc>.elf.xip.elf`; optionally restricted to the config hashes the run's own
  compile log lists. Builds sharing size and head but not bytes are all mapped; the share of samples whose category
  differs between them is reported ("ambiguity").
- A kernel PC maps to the ELF address pc - load + text start. The first and last 5% of each target's kernel samples (by
  time) are dropped, as pcs_lines.py does (outside the traffic window).
- addr2line -a -f -i -C on the cache's own `<proc>.elf` (the JIT builds with -g; same loadable code as the xip image)
  gives each address's inline chain, innermost first.
- Sources: the build's files are read from layers (a variant directory over a git revision), never from the live tree,
  which a broker job may have swapped. Rules carry text anchors; each rule is resolved to line ranges in the build's own
  file versions, so a rebuilt variant whose lines moved still maps, and a rule whose anchor is gone is reported.
- Classification: walk the chain from the innermost frame outward; the first frame that a rule matches decides. Rules
  with category "defer" pass the frame on to its caller (generic helpers used for several purposes); frames in files
  no rule names are skipped (library code: NoC register writes, stream registers), so library code takes its caller's
  category.
"""
import argparse
import bisect
import collections
import functools
import glob
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

ROOT = "/home/aagarwal/tt-metal"
RES = ROOT + "/fabric_exp/results"
HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.normpath(os.path.join(HERE, "..", "breakdown"))
A2L = ROOT + "/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
PROCS = {0: "active_erisc", 1: "subordinate_active_erisc"}
TRIM = 0.05


# ------------------------------------------------------------------ samples and builds
def load_samples(prefix):
    meta = json.load(open(prefix + ".json"))
    per = collections.defaultdict(list)
    for t, i, pc in struct.Struct("<dHI").iter_unpack(open(prefix + ".bin", "rb").read()):
        per[i].append((t, pc))
    return meta, per


def segments(path):
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    out = []
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<8I", d, phoff + i * phentsize)
        if p_type == 1:
            out.append((p_vaddr, p_flags, d[p_off:p_off + p_filesz]))
    return out


@functools.lru_cache(maxsize=None)
def image_of(xip):
    """(text vaddr, text bytes, packed size, digest of all loadable bytes) of a relocated image."""
    segs = segments(xip)
    text = next(s for s in segs if s[1] & 1)
    packed = sum((len(s[2]) + 3) // 4 * 4 for s in segs)
    return text[0], text[2], packed, hashlib.md5(b"".join(s[2] for s in segs)).hexdigest()


@functools.lru_cache(maxsize=None)
def cache_images(cache, proc):
    return sorted(glob.glob(f"{cache}/**/kernels/fabric_erisc_router*/*/{proc}/{proc}.elf.xip.elf", recursive=True))


def log_hashes(logs):
    hs = set()
    for log in logs:
        for p in glob.glob(log if os.path.isabs(log) else os.path.join(RES, log)):
            hs |= set(re.findall(r"kernels/fabric_erisc_router(?:_v2)?/(\d+)/", open(p, errors="replace").read()))
    return hs


def find_builds(cache, erisc, layout, hashes=None):
    """Candidate builds for one target: [(elf, xip, hash, digest, text vaddr)], distinct images first."""
    e = layout.get("erisc", {}).get(str(erisc), {}) if isinstance(layout, dict) else {}
    if not e or not e.get("text_size"):
        return []
    head = bytes.fromhex(e["head"])
    out = []
    for x in cache_images(cache, PROCS[erisc]):
        h = x.split("/")[-3]
        if hashes and h not in hashes:
            continue
        base, text, packed, dig = image_of(x)
        if packed == e["text_size"] and text[:len(head)] == head:
            out.append((x[: -len(".xip.elf")], x, h, dig, base))
    seen, uniq = set(), []
    for b in out:
        if b[3] not in seen:
            seen.add(b[3])
            uniq.append(b)
    return uniq


@functools.lru_cache(maxsize=None)
def _a2l_cached(elf):
    return {}


def a2l(elf, addrs):
    """{addr: [(function, repo-relative path, line)]}, innermost first."""
    cache = _a2l_cached(elf)
    need = sorted(set(addrs) - set(cache))
    if need:
        out = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", elf], input="\n".join(hex(a) for a in need),
                             capture_output=True, text=True).stdout.splitlines()
        cur, k = None, 0
        while k < len(out):
            if re.fullmatch(r"0x[0-9a-f]+", out[k]):
                cur = int(out[k], 16)
                cache[cur] = []
                k += 1
                continue
            fn = out[k]
            loc = re.sub(r" \(discriminator \d+\)", "", out[k + 1] if k + 1 < len(out) else "??:0")
            m = re.match(r"(.*):(\d+|\?)$", loc)
            path, line = (m.group(1), int(m.group(2)) if m and m.group(2).isdigit() else 0) if m else (loc, 0)
            cache[cur].append((short_fn(fn), relpath(path), line))
            k += 2
    return {a: cache.get(a, []) for a in addrs}


def short_fn(fn):
    while re.search(r"<[^<>]*>", fn):
        fn = re.sub(r"<[^<>]*>", "", fn)
    fn = re.sub(r"\(.*", "", fn).strip()
    fn = fn.split()[-1] if fn else "?"
    return fn.replace("tt::tt_fabric::", "").replace("fabric_router::", "")


def relpath(p):
    for root in (ROOT + "/", "/home/aagarwal/tt-metal-vc2line/", "/data/aagarwal/frs-glx/"):
        if p.startswith(root):
            p = p[len(root):]
            break
    m = re.search(r"(tt_metal/.*|tests/.*|runtime/.*)$", p)
    return m.group(1) if m else p


def target_window(samples, layout, erisc, base):
    """Kernel samples of one target as ELF addresses, the first and last 5% by time dropped."""
    e = layout["erisc"][str(erisc)]
    load, size = e["load"], e["text_size"]
    kern = sorted((t, pc - load + base) for t, pc in samples if load <= pc < load + size)
    k = int(len(kern) * TRIM) if len(kern) > 20 else 0
    return [a for _, a in kern[k:len(kern) - k]], len(samples), len(kern)


# ------------------------------------------------------------------ sources
class Sources:
    """A build's source files: layers searched in order. 'variant:DIR' (a variant directory, mirrored tree paths or
    bare names as offline_build.resolve takes them), 'git:REV' (git show REV:path), 'dir:DIR' (a tree snapshot)."""

    def __init__(self, layers):
        self.layers = layers
        self._files = {}
        self._names = None

    def _bare_names(self):
        if self._names is None:
            out = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", "HEAD", "tt_metal/fabric"],
                                 capture_output=True, text=True).stdout.split()
            self._names = collections.defaultdict(list)
            for p in out:
                self._names[os.path.basename(p)].append(p)
        return self._names

    def _variant_map(self, d):
        d = d if os.path.isabs(d) else os.path.join(ROOT, d)
        m = {}
        for dp, _, fs in os.walk(d):
            for f in fs:
                src = os.path.join(dp, f)
                rel = os.path.relpath(src, d)
                if f in ("fabric_erisc_router_v3.cpp", "fabric_erisc_router_v4.cpp") and rel == f:
                    m["tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp"] = src
                elif rel.startswith(("tt_metal/", "tests/")):
                    m[rel] = src
                elif f.endswith((".cpp", ".hpp", ".h")) and "/" not in rel:
                    c = self._bare_names().get(f, [])
                    if len(c) == 1:
                        m[c[0]] = src
        return m

    @functools.lru_cache(maxsize=None)
    def _layer(self, layer):
        kind, arg = layer.split(":", 1)
        return self._variant_map(arg) if kind == "variant" else None

    def lines(self, path):
        if path in self._files:
            return self._files[path]
        text = None
        for layer in self.layers:
            kind, arg = layer.split(":", 1)
            if kind == "variant":
                src = self._layer(layer).get(path)
                if src:
                    text = open(src, errors="replace").read()
            elif kind == "git":
                r = subprocess.run(["git", "-C", ROOT, "show", f"{arg}:{path}"], capture_output=True, text=True,
                                   errors="replace")
                if r.returncode == 0:
                    text = r.stdout
            elif kind == "dir":
                p = os.path.join(arg if os.path.isabs(arg) else os.path.join(ROOT, arg), path)
                if os.path.exists(p):
                    text = open(p, errors="replace").read()
            elif kind == "tgz":  # a source snapshot a phase-2 job took while that router was in the tree
                import tarfile
                if not hasattr(self, "_tars"):
                    self._tars = {}
                tp = arg if os.path.isabs(arg) else os.path.join(ROOT, arg)
                tf = self._tars.setdefault(tp, tarfile.open(tp, "r:gz") if os.path.exists(tp) else None)
                if tf is not None:
                    try:
                        text = tf.extractfile(path).read().decode(errors="replace")
                    except (KeyError, AttributeError):
                        text = None
            if text is not None:
                break
        self._files[path] = text.split("\n") if text is not None else None
        return self._files[path]

    def line(self, path, n):
        ls = self.lines(path)
        return ls[n - 1] if ls and 0 < n <= len(ls) else None


# ------------------------------------------------------------------ maps
def load_map(name):
    p = name if os.path.exists(name) else os.path.join(BD, f"map_{name}.json")
    m = json.load(open(p))
    m["_path"] = p
    tax = json.load(open(os.path.join(BD, "taxonomy.json")))
    m["categories"] = tax["categories"]
    ids = {c["id"] for c in tax["categories"]} | {"defer"}
    bad = [r["id"] for r in m["rules"] if r["cat"] not in ids]
    if bad:
        sys.exit(f"{p}: rules with unknown categories: {bad}")
    dup = [k for k, v in collections.Counter(r["id"] for r in m["rules"]).items() if v > 1]
    if dup:
        sys.exit(f"{p}: duplicate rule ids: {dup}")
    return m


def _find(lines, text, hint, lo=1):
    """Line numbers (1-based, >= lo) whose text contains `text` (a /regex/ or a substring), nearest the hint first."""
    if text.startswith("/") and text.endswith("/") and len(text) > 2:
        rx = re.compile(text[1:-1])
        hits = [i + 1 for i, s in enumerate(lines) if i + 1 >= lo and rx.search(s)]
    else:
        hits = [i + 1 for i, s in enumerate(lines) if i + 1 >= lo and text in s]
    return sorted(hits, key=lambda n: (abs(n - (hint or n)), n))


ALL = None  # a rule that covers every line of its file (filtered by its "fn")


def resolve_rule(r, sources):
    """(path, set of line numbers or ALL) in this source version, or None when the rule's anchor is not found (the
    rule needs re-mapping for this build). Rule forms:
      {"match": REGEX}                 every line whose text matches (optionally only within "from"/"to")
      {"from": [hint, TEXT], "to": [hint, TEXT]}   a block: the TEXT nearest the hint, then "to" at or after it
                                       ("to_excl": true ends the block on the line before "to")
      {"from": [hint, TEXT]}           one line
      neither                          the whole file (with "fn": every line of that function, wherever inlined)
    TEXT is a substring, or /regex/."""
    for path in _paths_for(r["file"], sources):
        ls = sources.lines(path)
        if not ls:
            continue
        lo, hi = 1, len(ls)
        if "from" in r:
            a = _find(ls, r["from"][1], r["from"][0])
            if not a:
                continue
            lo = hi = a[0]
            if "to" in r:  # the first match at or after "from" (the hint only documents where it was)
                b = sorted(_find(ls, r["to"][1], None, lo))
                if not b:
                    continue
                hi = b[0] - (1 if r.get("to_excl") else 0)
                if hi < lo:
                    continue
        if "match" in r:
            rx = re.compile(r["match"])
            got = {i for i in range(lo, hi + 1) if rx.search(ls[i - 1])}
            if not got:
                continue
            return path, got
        if "from" in r:
            return path, set(range(lo, hi + 1))
        return path, ALL
    return None


def resolve_rules(mp, sources):
    return {r["id"]: resolve_rule(r, sources) for r in mp["rules"]}


@functools.lru_cache(maxsize=None)
def _git_paths():
    out = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", "HEAD", "tt_metal/fabric",
                          "tt_metal/hw/inc"], capture_output=True, text=True).stdout.split()
    return tuple(out)


def _paths_for(fileref, sources):
    """Repo paths a rule's file reference names (a full path, or a basename: the variant layers' files first, then
    the tree's)."""
    if "/" in fileref:
        return [fileref]
    out = []
    for layer in sources.layers:
        if layer.startswith("variant:"):
            out += [p for p in sources._layer(layer) if os.path.basename(p) == fileref]
        elif layer.startswith("tgz:"):
            sources.lines("tt_metal/README")  # opens the archive
            for tf in getattr(sources, "_tars", {}).values():
                if tf is not None:
                    out += [m for m in tf.getnames() if os.path.basename(m) == fileref]
        elif layer.startswith("dir:"):
            d = layer[4:] if os.path.isabs(layer[4:]) else os.path.join(ROOT, layer[4:])
            for sub in ("tt_metal/fabric", "tt_metal/hw/inc"):
                for dp, _, fs in os.walk(os.path.join(d, sub)):
                    out += [os.path.relpath(os.path.join(dp, f), d) for f in fs if f == fileref]
    out += [p for p in _git_paths() if os.path.basename(p) == fileref]
    return list(dict.fromkeys(out)) or [fileref]


class Classifier:
    def __init__(self, mp, sources):
        self.mp = mp
        self.sources = sources
        self.ranges = resolve_rules(mp, sources)
        self.by_file = collections.defaultdict(list)  # basename -> [(specificity, order, rule, lines)]
        for k, r in enumerate(mp["rules"]):
            got = self.ranges.get(r["id"])
            if got is None:
                continue
            lines = got[1]
            # Most specific first: content matches, then blocks by size, then whole functions, then whole files.
            if "match" in r:
                spec = (0, len(lines))
            elif lines is not ALL:
                spec = (1, len(lines))
            elif r.get("fn"):
                spec = (2, 0)
            else:
                spec = (3, 0)
            self.by_file[os.path.basename(r["file"])].append((spec, k, r, lines))
        for v in self.by_file.values():
            v.sort(key=lambda x: (x[0], x[1]))
        self.fn_rx = {r["id"]: re.compile(r["fn"]) for r in mp["rules"] if r.get("fn")}
        self.router_rx = [re.compile(p) for p in mp.get("router_files", [])]

    def frame_rule(self, fn, path, line):
        for spec, k, r, lines in self.by_file.get(os.path.basename(path), []):
            if "/" in r["file"] and not path.endswith(r["file"]):
                continue
            if lines is not ALL and line not in lines:
                continue
            if r.get("fn") and not self.fn_rx[r["id"]].search(fn):
                continue
            return r
        return None

    def classify(self, chain):
        """(category, rule id or None, frame, how): the first rule from the innermost frame outward decides.
        how: "rule" (a specific rule), "fallback" (a rule marked fallback, reached with the innermost router frame
        unmatched), or "unmapped"."""
        first = None
        for fn, path, line in chain:
            r = self.frame_rule(fn, path, line)
            if r is None or r["cat"] == "defer":
                if first is None and self.is_router_file(path):
                    first = (fn, path, line)
                continue
            how = "fallback" if r.get("fallback") and first is not None else "rule"
            return r["cat"], r["id"], (fn, path, line) if how == "rule" else first, how
        return "unmapped", None, first or (chain[0] if chain else ("?", "?", 0)), "unmapped"

    def is_router_file(self, path):
        return any(re.search(p, path) for p in self.mp.get("router_files", []))


# ------------------------------------------------------------------ one sample set
def analyze_set(s, mp, pick=None, verbose=False):
    """s: a sample-set record (breakdown/sets.json). Returns per chosen target: counts per category and rule, the
    unmapped lines, ambiguity, sample counts."""
    meta, per = load_samples(os.path.join(RES, s["prefix"]))
    hashes = log_hashes(s["logs"]) if s.get("logs") else None
    sources = Sources(s["sources"])
    clf = Classifier(mp, sources)
    results = []
    want = s.get("targets")  # ["chip/chan/E0", ...] or None: pick per ERISC by the work categories
    for i, tgt in enumerate(meta["targets"]):
        key = f"{tgt['chip']}/{tgt['chan']}"
        name = f"{key}/E{tgt['erisc']}"
        if s.get("eriscs") and tgt["erisc"] not in s["eriscs"]:
            continue
        if want and name not in want and pick is None:
            continue
        if pick and name != pick:
            continue
        lay = meta.get("layouts_end", {}).get(key) or meta.get("layouts", {}).get(key, {})
        builds = find_builds(s["cache"], tgt["erisc"], lay, hashes)
        if not builds:
            continue
        cats_by_build = []
        for elf, xip, h, dig, base in builds:
            addrs, n_all, n_kern = target_window(per.get(i, []), lay, tgt["erisc"], base)
            if not addrs:
                break
            chains = a2l(elf, sorted(set(addrs)))
            cnt = collections.Counter(addrs)
            cats, rules, unm, fb, lines = (collections.Counter(), collections.Counter(), collections.Counter(),
                                           collections.Counter(), {})
            for a, c in cnt.items():
                cat, rid, fr, how = clf.classify(chains[a])
                cats[cat] += c
                rules[rid or "(unmapped)"] += c
                if how == "unmapped":
                    unm[(fr[0], fr[1], fr[2])] += c
                elif how == "fallback":
                    fb[(fr[0], fr[1], fr[2], rid)] += c
                lines[a] = (cat, rid, chains[a], how)
            cats_by_build.append(dict(build=h, elf=elf, n=len(addrs), n_all=n_all, n_kern=n_kern, cats=cats,
                                      rules=rules, unmapped=unm, fallback=fb, addr_cnt=cnt, addr_info=lines))
        if not cats_by_build:
            continue
        b0 = cats_by_build[0]
        amb = 0.0
        if len(cats_by_build) > 1:  # share of samples whose category differs between candidate builds
            diff = sum(c for a, c in b0["addr_cnt"].items()
                       if any(b["addr_info"].get(a, (None,))[0] != b0["addr_info"][a][0] for b in cats_by_build[1:]))
            amb = diff / b0["n"]
        results.append(dict(target=name, erisc=tgt["erisc"], candidates=[b["build"] for b in cats_by_build],
                            ambiguity=amb, **b0))
    return results, clf


def work_share(res, mp):
    w = mp["work_categories"][f"E{res['erisc']}"]
    return sum(res["cats"][c] for c in w) / max(res["n"], 1)


def pick_targets(results, mp, s):
    """The target per ERISC kind: explicit in the set, else the most samples in that kind's work categories."""
    out = {}
    if s.get("targets"):
        for r in results:
            out.setdefault(r["erisc"], r) if r["target"] in s["targets"] else None
        return out
    for r in results:
        if r["erisc"] not in out or work_share(r, mp) > work_share(out[r["erisc"]], mp):
            out[r["erisc"]] = r
    return out


def row_rate(s):
    """cycles/packet of the sampled run itself (its CSV), or the set's 'rate' field."""
    if s.get("rate"):
        return float(s["rate"]), "set record"
    import csv
    p = os.path.join(RES, s["rate_csv"])
    vals = [float(r["avg_packets_per_s"]) for r in csv.DictReader(open(p))]
    return 1.35e9 / (sum(vals) / len(vals)), s["rate_csv"]


def load_sets(path=None):
    return {s["id"]: s for s in json.load(open(path or os.path.join(BD, "sets.json")))}


# ------------------------------------------------------------------ CLI
def cmd_check(a):
    s = load_sets(a.sets)[a.set]
    mp = load_map(s["map"])
    results, clf = analyze_set(s, mp)
    unresolved = [k for k, v in clf.ranges.items() if v is None]
    rate, src = row_rate(s)
    print(f"## {a.set}: {s['prefix']} ({s['router']}, {s['build']}); rate {rate:.2f} cycles/packet ({src})")
    if unresolved:
        print(f"   rules not resolved in this source version (re-map): {', '.join(unresolved)}")
    picks = pick_targets(results, mp, s)
    for r in sorted(results, key=lambda r: (r["erisc"], -work_share(r, mp)))[: a.all and 999 or 0]:
        print(f"   {r['target']}: {r['n']} samples, work {100 * work_share(r, mp):.0f}%")
    for e, r in sorted(picks.items()):
        n = r["n"]
        unm = sum(r["unmapped"].values())
        fb = sum(r["fallback"].values())
        print(f"\n   {r['target']} (E{e}): {n} window samples of {r['n_kern']} in the kernel / {r['n_all']} all; "
              f"specific rules {100 * (n - unm - fb) / n:.1f}%, fallback {100 * fb / n:.1f}%, unmapped "
              f"{100 * unm / n:.1f}%; candidate builds {len(r['candidates'])}"
              + (f", categories differ between them on {100 * r['ambiguity']:.1f}% of samples" if len(r['candidates']) > 1 else ""))
        for c in mp["categories"] + [{"id": "unmapped"}]:
            k = c["id"]
            if r["cats"][k]:
                print(f"      {r['cats'][k] / n * rate:6.1f}  {100 * r['cats'][k] / n:5.1f}%  {k}")
        for title, items in (("unmapped", r["unmapped"]), ("fallback", r["fallback"])):
            if not items:
                continue
            print(f"      top {title} lines (innermost router frame):")
            for key, c in items.most_common(a.top):
                fn, path, line = key[:3]
                txt = (clf.sources.line(path, line) or "").strip()[:70]
                extra = f" -> {key[3]}" if len(key) > 3 else ""
                print(f"         {c / n * rate:5.1f} {100 * c / n:5.1f}%  {path.split('/')[-1]}:{line} ({fn}){extra}  {txt}")
        print("      by rule:")
        for rid, c in r["rules"].most_common(a.top):
            print(f"         {c / n * rate:6.1f}  {100 * c / n:5.1f}%  {rid}")


def cmd_lines(a):
    s = load_sets(a.sets)[a.set]
    mp = load_map(s["map"])
    results, clf = analyze_set(s, mp, pick=a.target)
    rate, _ = row_rate(s)
    picks = {r["erisc"]: r for r in results} if a.target else pick_targets(results, mp, s)
    for e, r in sorted(picks.items()):
        n = r["n"]
        print(f"\n## {a.set} {r['target']}: {n} samples, rate {rate:.2f}")
        by_line = collections.Counter()
        ex = {}
        for addr, c in r["addr_cnt"].items():
            cat, rid, chain, how = r["addr_info"][addr]
            k = chain[0][1:] if chain else ("?", 0)
            by_line[k] += c
            ex.setdefault(k, (cat, rid, chain))
        for k, c in by_line.most_common(a.top):
            cat, rid, chain = ex[k]
            txt = (clf.sources.line(k[0], k[1]) or "").strip()[:60]
            ch = " < ".join(f"{fn}:{ln}" for fn, p, ln in chain[:a.depth])
            print(f"  {c / n * rate:6.1f} {100 * c / n:5.1f}%  [{cat}/{rid}] {k[0].split('/')[-1]}:{k[1]}  {txt}\n"
                  f"                 {ch}")


def cmd_resolve(a):
    mp = load_map(a.map)
    src = Sources(a.sources.split(","))
    for rid, v in resolve_rules(mp, src).items():
        r = next(x for x in mp["rules"] if x["id"] == rid)
        if v is None:
            print(f"UNRESOLVED {rid}: {r['file']} {r.get('match') or ''} {r.get('from') or ''} .. {r.get('to') or ''}")
        elif a.verbose and v[1] is not ALL:
            ls = sorted(v[1])
            print(f"{rid}: {v[0].split('/')[-1]}:{ls[0]}-{ls[-1]} ({len(ls)} lines)")


def named_args(build_dir):
    p = os.path.join(build_dir, "named_ct_arg_map_generated.h")
    if not os.path.exists(p):
        return {}
    return {k: int(v) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(-?\d+)\}', open(p).read())}


def instructions(elf):
    out = subprocess.run([A2L.replace("addr2line", "objdump"), "-d", "--no-show-raw-insn", "-j", ".text", elf],
                         capture_output=True, text=True).stdout
    return [int(m.group(1), 16) for m in re.finditer(r"^\s*([0-9a-f]+):\s", out, re.M)]


def cmd_static(a):
    """Instruction-weighted coverage of a map over the main loop of real builds (no samples needed): every instruction
    whose inline chain passes through the map's loop functions is classified. Checks that a map covers code paths no
    sample set reached (main's two-ERISC routers, VC1, rings)."""
    mp = load_map(a.map)
    clf = Classifier(mp, Sources(a.sources.split(",")))
    loop_rx = re.compile(mp.get("loop_fn", "run"))
    where = dict(kv.split("=") for kv in a.where.split(",")) if a.where else {}
    builds = []
    for proc in ("active_erisc", "subordinate_active_erisc"):
        if a.proc and a.proc != proc:
            continue
        for x in cache_images(a.cache, proc):
            bdir = os.path.dirname(os.path.dirname(x))
            na = named_args(bdir)
            if all(str(na.get(k)) == v for k, v in where.items()):
                builds.append((proc, x[: -len(".xip.elf")], os.path.basename(bdir)))
    seen = collections.Counter()
    for proc, elf, h in builds:
        if seen[proc] >= a.max:
            continue
        seen[proc] += 1
        addrs = instructions(elf)
        chains = a2l(elf, addrs)
        cats, unm, fb = collections.Counter(), collections.Counter(), collections.Counter()
        n = 0
        for addr in addrs:
            ch = chains.get(addr, [])
            if not any(loop_rx.search(fn) for fn, _, _ in ch):
                continue
            n += 1
            cat, rid, fr, how = clf.classify(ch)
            cats[cat] += 1
            if how == "unmapped":
                unm[fr] += 1
            elif how == "fallback":
                fb[(fr, rid)] += 1
        print(f"\n## {proc} {h}: {n} loop instructions of {len(addrs)}; specific {100 * (n - sum(unm.values()) - sum(fb.values())) / max(n, 1):.1f}%, "
              f"fallback {100 * sum(fb.values()) / max(n, 1):.1f}%, unmapped {100 * sum(unm.values()) / max(n, 1):.1f}%")
        print("   " + ", ".join(f"{k} {v}" for k, v in cats.most_common()))
        for title, items in (("unmapped", unm), ("fallback", fb)):
            for key, c in items.most_common(a.top):
                fr = key[0] if title == "fallback" else key
                fn, path, line = fr
                txt = (clf.sources.line(path, line) or "").strip()[:70]
                print(f"   {title:8} {c:4}  {path.split('/')[-1]}:{line} ({fn})  {txt}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("static"); st.add_argument("cache"); st.add_argument("--map", required=True)
    st.add_argument("--sources", default="git:HEAD"); st.add_argument("--where", default="")
    st.add_argument("--proc"); st.add_argument("--max", type=int, default=1); st.add_argument("--top", type=int, default=8)
    c = sub.add_parser("check"); c.add_argument("set"); c.add_argument("--sets"); c.add_argument("--top", type=int, default=12)
    c.add_argument("--all", action="store_true")
    l = sub.add_parser("lines"); l.add_argument("set"); l.add_argument("--sets"); l.add_argument("--top", type=int, default=40)
    l.add_argument("--target"); l.add_argument("--depth", type=int, default=4)
    r = sub.add_parser("resolve"); r.add_argument("map"); r.add_argument("--sources", default="git:HEAD")
    r.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    {"check": cmd_check, "lines": cmd_lines, "resolve": cmd_resolve, "static": cmd_static}[a.cmd](a)


if __name__ == "__main__":
    main()
