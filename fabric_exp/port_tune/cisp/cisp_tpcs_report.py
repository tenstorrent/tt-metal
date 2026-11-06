#!/usr/bin/env python3
"""cisp_tpcs_report.py RESULTS_DIR: the Tensix sender/sync workers' PC samples of a cisp_tpcs.sh run, per arm, split by
the kernel image's inline chain, for the NX 4096 B / 2048 B unicast_write rows, slow iterations against normal ones of
the same row (slow: the test's sender row below 97% of the off arm's median for that test). A within-core split, never
a bound.

Mapping: each sample's core has mailbox-head snapshots every 50 ms (pc_sampler.py --tensix): the current launch slot's
kernel config base + BRISC text offset is the load address, its text size picks the cache's tt_fabric_test_sender /
_sync brisc ELFs of that size (the image whose text starts like the loaded one is not read: same-size images of one
kernel source share a layout here); kernel PCs map to pc - load + text start, addr2line -f -i -C gives the chain. PCs
outside the kernel image are "firmware" (BRISC firmware: the go-signal wait between kernels). NCRISC: the sender kernel
has no NCRISC build, so its samples are the baseline (firmware)."""
import bisect, collections, csv, glob, json, os, re, statistics, struct, subprocess, sys

sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hang_workers as hw  # noqa: E402
import cisp_pcs2_report as P  # noqa: E402

D = sys.argv[1]
A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
CACHE = {"on": "/home/aagarwal/.tt-metal-cache-cisp-on-ci", "off": "/home/aagarwal/.tt-metal-cache-cisp-off-ci"}
ROWS = re.compile(r"size_(4096|2048)_ntype_unicast_write_num_links_\d$")


def elfs(arm):
    out = collections.defaultdict(list)  # text size -> [(elf, text start)]
    for name in ("tt_fabric_test_sender", "tt_fabric_test_sync"):
        for e in glob.glob(f"{CACHE[arm]}/*/kernels/{name}/*/brisc/brisc.elf.xip.elf"):
            base, text = P.text_segment(e)
            out[len(text)].append((e[: -len(".xip.elf")], base, name))
    return out


_ch = {}


def chain(elf, addr):
    if (elf, addr) not in _ch:
        o = subprocess.run([A2L, "-f", "-i", "-C", "-e", elf, hex(addr)], capture_output=True, text=True).stdout.splitlines()
        fr = []
        for k in range(0, len(o) - 1, 2):
            fn = o[k]
            while re.search(r"<[^<>]*>", fn):
                fn = re.sub(r"<[^<>]*>", "", fn)
            fn = re.sub(r"\(.*", "", fn).split()[-1] if fn.strip() else "?"
            fr.append((fn, "/".join(re.sub(r" \(discriminator \d+\)", "", o[k + 1]).split("/")[-2:])))
        _ch[(elf, addr)] = fr
    return _ch[(elf, addr)]


def bucket(fr):
    s = " ".join(f for f, _ in fr)
    lines = " ".join(l for _, l in fr)
    if "kernel_profiler" in s or "profiler" in lines:
        return "profiler"
    if re.search(r"noc_async_write_barrier|noc_async_writes_flushed|barrier|ncrisc_noc_nonposted|writes_flushed", s):
        return "noc write ack / flush wait"
    if re.search(r"noc_cmd_buf_ready|cmd_buf_ready|noc_async_write|noc_fast|noc_inline|ncrisc_noc", s):
        return "noc command (buffer wait / issue)"
    if re.search(r"wait_for_empty_write_slot|EdmToEdmSender|edm_has_space|get_num_free_slots|fabric.*space|free_slots|wait_for_slot", s) \
            or re.search(r"edm_fabric|fabric_connection|tt_fabric", lines) and "wait" in s:
        return "fabric connection credits"
    if re.search(r"fabric|edm|Fabric|Edm", s + lines):
        return "fabric send path"
    return "other kernel code"


def run(arm):
    pre = os.path.join(D, f"tpcs_{arm}_samples")
    meta = json.load(open(pre + ".json"))
    rec = []
    raw = open(pre + ".bin", "rb").read()
    for t, i, pc in struct.iter_unpack("<dHI", raw):
        rec.append((t + meta["t_start_epoch"], i, pc))
    snaps = collections.defaultdict(list)  # core index -> [(t, launch dict)]
    sraw = open(pre + ".snap.bin", "rb").read()
    step = 10 + 1488
    for o in range(0, len(sraw) - step + 1, step):
        t, k = struct.unpack_from("<dH", sraw, o)
        hd = hw.parse_head(sraw[o + 10: o + 10 + 1488])
        if hd:
            # every launch slot: the running kernel is the one whose image holds the sampled PC (the program's kernel
            # config base moves per launch; parse_head's own slot choice is unreliable between launches)
            snaps[k].append((t + meta["t_start_epoch"], [sl for sl in hd["slots"] if sl["text_size"][0]]))
    core_index = {(c["chip"], c["core"]): k for k, c in enumerate(meta["snap_cores"])}
    win = P.windows(os.path.join(D, f"tpcs_{arm}.log"))
    starts = [a for _, _, a, _ in win]
    sender = list(csv.DictReader(open(os.path.join(D, f"tpcs_{arm}_bandwidth_results.csv"))))
    return meta, rec, snaps, core_index, win, starts, sender


def main():
    data = {arm: run(arm) for arm in ("on", "off") if os.path.exists(os.path.join(D, f"tpcs_{arm}_samples.bin"))}
    # slow iterations: sender row below 97% of the off arm's median for the same test
    off_med = {}
    if "off" in data:
        by = collections.defaultdict(list)
        for r in data["off"][6]:
            by[r["test_name"]].append(float(r["bandwidth_GB_s"]))
        off_med = {t: statistics.median(v) for t, v in by.items()}
    for arm, (meta, rec, snaps, core_index, win, starts, sender) in data.items():
        E = elfs(arm)
        stimes = {k: [t for t, _ in v] for k, v in snaps.items()}
        res = collections.defaultdict(collections.Counter)  # (risc, group) -> bucket
        lines = collections.defaultdict(collections.Counter)
        nwin = collections.Counter()
        slow_flags = []
        for i, (test, k, a, b) in enumerate(win):
            bw = float(sender[i]["bandwidth_GB_s"]) if i < len(sender) else None
            slow = bw is not None and test in off_med and bw < 0.97 * off_med[test]
            slow_flags.append(slow)
        unmatched = 0
        for t, idx, pc in rec:
            j = bisect.bisect_right(starts, t) - 1
            if j < 0:
                continue
            test, k, a, b = win[j]
            if t > b or not ROWS.search(test):
                continue
            tg = meta["targets"][idx]
            risc = tg["erisc"]
            grp = "slow" if slow_flags[j] else "normal"
            nwin[(risc, grp)] += 1
            if risc != "brisc":
                res[(risc, grp)]["(NCRISC baseline)"] += 1
                continue
            ci = core_index[(tg["chip"], tg["chan"])]
            ts = stimes.get(ci, [])
            q = bisect.bisect_right(ts, t) - 1
            if q < 0:
                unmatched += 1
                continue
            hit = None
            for qq in (q, q + 1, q - 1):
                if 0 <= qq < len(snaps[ci]):
                    hit = next(((sl["kcb"] + sl["text_off"][0], sl["text_size"][0]) for sl in snaps[ci][qq][1]
                                if sl["kcb"] + sl["text_off"][0] <= pc < sl["kcb"] + sl["text_off"][0] + sl["text_size"][0]), None)
                    if hit:
                        break
            load, size = hit if hit else (0, 0)
            if not hit:
                res[("brisc", grp)]["firmware / outside the kernel image"] += 1
                continue
            # the loaded image's size (its launch-message text size) is a little above the ELF's text segment: take the
            # cache image of the nearest smaller text size (sender ~11.6-15.2 KB, sync ~7.0-10.8 KB; approximate when
            # two configs differ in code, which the line split tolerates)
            cand = [k for k in E if k <= size]
            if not cand:
                res[("brisc", grp)]["kernel image without a cache match"] += 1
                continue
            elf, base, name = E[max(cand)][0]
            fr = chain(elf, pc - load + base)
            bk = bucket(fr) if name == "tt_fabric_test_sender" else f"sync kernel: {bucket(fr)}"
            res[("brisc", grp)][bk] += 1
            if fr:
                lines[grp][(bk, f"{fr[0][0]} {fr[0][1]}")] += 1
        print(f"\n== {arm}: {len(rec)} samples, {sum(slow_flags)} slow of {len(win)} iterations; in NX write-row windows: "
              f"{dict(nwin)}; unmatched (no snapshot yet) {unmatched}")
        for (risc, grp), c in sorted(res.items()):
            n = sum(c.values())
            print(f"   {risc} {grp}: {n} samples: " + ", ".join(f"{b} {100 * v / n:.1f}%" for b, v in c.most_common()))
        for grp in ("slow", "normal"):
            n = sum(lines[grp].values())
            if n:
                print(f"   brisc {grp}: top lines")
                for (bk, l), v in lines[grp].most_common(12):
                    print(f"      {100 * v / n:6.2f}%  {bk:34s} {l}")


if __name__ == "__main__":
    main()
