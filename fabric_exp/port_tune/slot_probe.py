#!/usr/bin/env python3
"""slot_probe.py CASE_DIR [CASE_DIR ...]: each router build's slot counts, from its own logged compile command (a cov.py
case: cmds.json and the JIT cache). The host's slot log is log_debug, which a Release build compiles out; instead each
router compile is rerun syntax-only with a tail that forces a compile error naming SENDER_NUM_BUFFERS_ARRAY and
RECEIVER_NUM_BUFFERS_ARRAY (SlotsTx<...> / SlotsRx<...>). Prints the distinct rows per case and processor with counts.
Sender channels are listed in channel order: VC0 worker, VC0 forwarding, then VC1's (forwarding), as the router's
arrays hold them."""
import collections, concurrent.futures, json, os, re, subprocess, sys, tempfile

TAIL = r"""
#include <utility>
template <std::size_t... V> struct SlotsTx;
template <std::size_t... V> struct SlotsRx;
template <std::size_t... I> SlotsTx<SENDER_NUM_BUFFERS_ARRAY[I]...> probe_tx(std::index_sequence<I...>);
template <std::size_t... I> SlotsRx<RECEIVER_NUM_BUFFERS_ARRAY[I]...> probe_rx(std::index_sequence<I...>);
static_assert(sizeof(decltype(probe_tx(std::make_index_sequence<SENDER_NUM_BUFFERS_ARRAY.size()>{}))) == 0);
static_assert(sizeof(decltype(probe_rx(std::make_index_sequence<RECEIVER_NUM_BUFFERS_ARRAY.size()>{}))) == 0);
"""


def probe(cmd, cwd, tmp, tail_name="tail.h"):
    src = next(x for x in cmd.split() if x.endswith("erisck.cc"))
    w = os.path.join(tmp, f"w{abs(hash(cmd))}_{tail_name}.cc")
    tail = os.path.join(tmp, tail_name)
    open(w, "w").write(f'#include "{src}"\n#include "{tail}"\n')
    c = re.sub(r" -c -o \S+ \S+erisck\.cc", f" -fsyntax-only {w}", cmd)
    c = re.sub(r" -MF \S+| -MMD| -flto=\S+", "", c)
    r = subprocess.run(c, shell=True, cwd=cwd, capture_output=True, text=True)
    tx = sorted(set(re.findall(r"SlotsTx<([0-9, ]*)>", r.stderr)))
    rx = sorted(set(re.findall(r"SlotsRx<([0-9, ]*)>", r.stderr)))
    if not tx and tail_name == "tail.h":  # the arrays in namespace tt::tt_fabric only (some router builds)
        return probe(cmd, cwd, tmp, "tail_ns.h")
    return (tx[0] if tx else "?"), (rx[0] if rx else "?")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "tail.h"), "w").write(TAIL)
        ns = TAIL.replace("SENDER_NUM_BUFFERS_ARRAY", "tt::tt_fabric::SENDER_NUM_BUFFERS_ARRAY")
        ns = ns.replace("RECEIVER_NUM_BUFFERS_ARRAY", "tt::tt_fabric::RECEIVER_NUM_BUFFERS_ARRAY")
        open(os.path.join(tmp, "tail_ns.h"), "w").write(ns)
        for case in sys.argv[1:]:
            cmds = json.load(open(os.path.join(case, "cmds.json")))
            jobs = []
            for key, v in cmds.items():
                if "fabric_erisc_router" not in key or not v.get("compile"):
                    continue
                proc = key.rsplit("/", 1)[1]
                cache = next(d for d in [os.path.join(case, "cache")] if os.path.isdir(d))
                kdirs = [os.path.join(r, key) for r, ds, fs in os.walk(cache) if os.path.isdir(os.path.join(r, key))]
                if not kdirs:
                    continue
                jobs.append((proc, v["compile"][0], kdirs[0]))
            rows = collections.Counter()
            with concurrent.futures.ThreadPoolExecutor(24) as ex:
                for (proc, _, _), (tx, rx) in zip(jobs, ex.map(lambda j: probe(j[1], j[2], tmp), jobs)):
                    rows[(proc, tx, rx)] += 1
            print(f"== {os.path.basename(case.rstrip('/'))}: {len(jobs)} router builds")
            for (proc, tx, rx), n in sorted(rows.items()):
                print(f"   {proc:26s} x{n:3d}  TX {tx:28s} RX {rx}")


if __name__ == "__main__":
    main()
