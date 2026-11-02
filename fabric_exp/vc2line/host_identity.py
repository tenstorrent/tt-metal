#!/usr/bin/env python3
"""host_identity.py OLD NEW [--old-prefix hold_] [--new-prefix hnew_] [--args]: a host change's identity check.

OLD and NEW are coverage builds (COV/builds/<name>) of the same cases captured before and after the host change
(host_captures.py, two prefixes), built with the same router sources. A host change that only adds named args moves
each config's hash, so images cannot be paired by hash: per case and ERISC, the multiset of router images (loadable
bytes, objcopy -O binary; and .text) must be the same in both builds.
Prints per case: images old / new, distinct images old / new, and whether the loadable-byte multisets are equal.

The offline build keeps one config per signature (per-router args such as MY_ETH_CHANNEL left out), choosing the first by
hash, so two captures with different hashes build different representatives. --args checks every captured config
instead: per case and ERISC, the multiset of full config signatures (the compile command with hashes and capture paths
normalized, and the named args without the NEW ones below) must be equal in the two captures (OLD and NEW are then
case prefixes' capture sets, not builds). With images a function of those inputs, the built pairs (same signature on
both sides) show the images agree."""
import argparse
import collections
import concurrent.futures
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc2l_offline as off  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

# The named args the host change adds (LINE_VCS_HOST_DESIGN.md §6).
NEW_ARGS = re.compile(
    r"^(VC\d_SERVICED|VC\d_ROUTES_AS_LINE|RX_CH_\d_FORWARDS_ON_VC|NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL|"
    r"SENDER_CH_\d+_IS_WORKER_CHANNEL)$")


def config_signatures(prefix, case):
    """Counter((erisc, normalized compile command, named args without NEW_ARGS)) over a capture's router configs."""
    d = os.path.join(off.COV, "cases", prefix + case)
    headers = {os.path.basename(os.path.dirname(h)): h for h in
               glob.glob(f"{d}/cache/*/kernels/fabric_erisc_router_v2/*/named_ct_arg_map_generated.h")}
    out = collections.Counter()
    for h, procs in json.load(open(os.path.join(d, "cmds.json"))).items():
        args = dict(re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(headers[h]).read())) if h in headers else {}
        named = tuple(sorted((k, v) for k, v in args.items() if not NEW_ARGS.match(k)))
        for proc, c in procs.items():
            cmd = c["compile"].replace(prefix + case, "CASE").replace(h, "HASH")
            cmd = re.sub(r"tt-metal-cache\d+", "CACHE", cmd)
            cmd = re.sub(r"(\w+k)\.\d+_\d+\.(o|d)\b", r"\1.OBJ.\2", cmd)  # the compiler's temporary object names
            out[(proc, cmd, named)] += 1
    return out


def built_pairs(old_build, new_build, old_prefix, new_prefix):
    """Images of the configs the two builds both built with equal inputs (config_signatures): .text and loadable bytes."""
    def built(build, prefix):
        out = {}
        root = os.path.join(off.COV, "builds", build)
        sig_cache = {}
        for e in glob.glob(f"{root}/{prefix}*/*/*.elf"):
            rel = os.path.relpath(e, root)
            case, h = rel.split("/")[0][len(prefix):].split("__")
            proc = rel.split("/")[1]
            if case not in sig_cache:
                d = os.path.join(off.COV, "cases", prefix + case)
                sig_cache[case] = (d, json.load(open(os.path.join(d, "cmds.json"))))
            d, cmds = sig_cache[case]
            header = glob.glob(f"{d}/cache/*/kernels/fabric_erisc_router_v2/{h}/named_ct_arg_map_generated.h")
            if h not in cmds or proc not in cmds[h] or not header:
                continue
            args = dict(re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(header[0]).read()))
            named = tuple(sorted((k, v) for k, v in args.items() if not NEW_ARGS.match(k)))
            cmd = cmds[h][proc]["compile"].replace(prefix + case, "CASE").replace(h, "HASH")
            cmd = re.sub(r"tt-metal-cache\d+", "CACHE", cmd)
            cmd = re.sub(r"(\w+k)\.\d+_\d+\.(o|d)\b", r"\1.OBJ.\2", cmd)
            out[(case, proc, cmd, named)] = e
        return out
    o, n = built(old_build, old_prefix), built(new_build, new_prefix)
    common = sorted(set(o) & set(n))
    with concurrent.futures.ThreadPoolExecutor(48) as ex:
        res = list(ex.map(lambda k: (k[0], off.section_bytes(o[k]), off.section_bytes(n[k])), common))
    per_case = collections.defaultdict(lambda: [0, 0, 0])
    for case, (ta, la), (tb, lb) in res:
        per_case[case][0] += 1
        per_case[case][1] += ta == tb
        per_case[case][2] += la == lb
    tot = [0, 0, 0]
    for case in sorted(per_case):
        c = per_case[case]
        print(f"{case:28s} pairs {c[0]:4d}  .text same {c[1]:4d}  loadable same {c[2]:4d}")
        tot = [x + y for x, y in zip(tot, c)]
    print(f"{'total':28s} pairs {tot[0]:4d}  .text same {tot[1]:4d}  loadable same {tot[2]:4d} "
          f"(built old {len(o)}, new {len(n)})")
    return 0 if tot[1] == tot[0] and tot[2] == tot[0] else 1


def digests(build, prefix):
    """{case: Counter((erisc, sha of loadable bytes, sha of .text))}"""
    root = os.path.join(off.COV, "builds", build)
    elfs = [e for e in glob.glob(f"{root}/{prefix}*/*/*.elf")]

    def one(e):
        rel = os.path.relpath(e, root)
        case = rel.split("__")[0][len(prefix):]
        text, load = off.section_bytes(e)
        return case, rel.split("/")[1], hashlib.sha256(load).hexdigest(), hashlib.sha256(text).hexdigest()

    out = collections.defaultdict(collections.Counter)
    with concurrent.futures.ThreadPoolExecutor(48) as ex:
        for case, erisc, load, text in ex.map(one, elfs):
            out[case][(erisc, load, text)] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--old-prefix", default="hold_")
    ap.add_argument("--new-prefix", default="hnew_")
    ap.add_argument("--args", action="store_true", help="compare every captured config's inputs (OLD/NEW: case names)")
    ap.add_argument("--pairs", action="store_true", help="compare the images of configs built in both builds with equal inputs")
    a = ap.parse_args()
    if a.args:
        cases = a.old.split(",")
        equal = 0
        for case in cases:
            o, n = config_signatures(a.old_prefix, case), config_signatures(a.new_prefix, case)
            same = set(o) == set(n)
            equal += same
            print(f"{case:28s} configs {sum(o.values()):4d} / {sum(n.values()):4d}  distinct inputs {len(o):3d} / "
                  f"{len(n):3d}  same inputs: {'yes' if same else 'NO'}{'' if o == n or not same else ' (counts differ)'}")
        print(f"cases with the same router config inputs: {equal} of {len(cases)}")
        return 0 if equal == len(cases) else 1
    if a.pairs:
        return built_pairs(a.old, a.new, a.old_prefix, a.new_prefix)
    old, new = digests(a.old, a.old_prefix), digests(a.new, a.new_prefix)
    same_cases = 0
    for case in sorted(set(old) | set(new)):
        o, n = old.get(case, collections.Counter()), new.get(case, collections.Counter())
        equal_sets = set(o) == set(n)
        equal_multisets = o == n
        same_cases += equal_sets
        print(f"{case:28s} images {sum(o.values()):4d} / {sum(n.values()):4d}  distinct {len(o):3d} / {len(n):3d}  "
              f"same images: {'yes' if equal_sets else 'NO'}{'' if equal_multisets or not equal_sets else ' (counts differ)'}")
    total_o = sum(sum(c.values()) for c in old.values())
    total_n = sum(sum(c.values()) for c in new.values())
    print(f"cases with the same images: {same_cases} of {len(set(old) | set(new))}; images {total_o} / {total_n}")
    return 0 if same_cases == len(set(old) | set(new)) else 1


if __name__ == "__main__":
    sys.exit(main())
