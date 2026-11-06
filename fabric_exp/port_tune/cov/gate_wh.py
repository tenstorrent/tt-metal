#!/usr/bin/env python3
"""gate_wh.py: the Blackhole identity gate and Wormhole compile check for the PR branch's Wormhole commits
(coordinator 2026-11-05: 3e86f4c80bf..ba486eb70ba on origin/aagarwal/fabric-router-mainline). Offline, mock clusters only.

  gate_wh.py mock TREE_TAG CASE...   cov.py mock runs of the cases with TT_COV_TREE=/home/aagarwal/tt-metal-gate-TAG and
                                     TT_COV_DIR=/home/aagarwal/.tt-metal-gate-TAG-cov (each tree's own host build and
                                     JIT); the Wormhole cases (wh_*) are defined here
  gate_wh.py cmp CASE...             per case, every kernel / firmware build both trees logged (cmds.json key: kernels/
                                     <name>/<hash>/<proc>), compared by loadable bytes (cov.image: every allocated section);
                                     builds only one tree has are listed (a host-side change of compile args or defines
                                     changes the hash)
  gate_wh.py sizes CASE...           Wormhole: every router ELF's PT_LOAD sizes (text, data) against the linker regions
                                     (ERISC_APP_KERNEL_CODE 17,312 B or ERISC_APP_IRAM_CODE 16 KiB with ENABLE_IRAM;
                                     ERISC_APP_KERNEL_DATA 6,144 B)"""
import collections, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREES = {t: f"/home/aagarwal/tt-metal-gate-{t}" for t in ("before", "after")}
COVS = {t: f"/home/aagarwal/.tt-metal-gate-{t}-cov" for t in ("before", "after")}


def load_cov(tag):
    os.environ["TT_COV_TREE"] = TREES.get(tag, f"/home/aagarwal/tt-metal-gate-{tag}")
    os.environ["TT_COV_DIR"] = COVS.get(tag, f"/home/aagarwal/.tt-metal-gate-{tag}-cov")
    sys.argv = ["cov"]
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"cov_{tag}", os.path.join(HERE, "cov.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    wh = m.TREE + "/tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml"
    yamls = m.TREE + "/tests/tt_metal/tt_fabric/test_infra/test_yamls/"
    m.CASES.update({
        "wh_t3k_line": (wh, m.test_yaml("Linear"), {}),
        "wh_t3k_ring": (wh, m.test_yaml("Ring"), {}),
        "wh_t3k_mesh": (wh, m.test_yaml("Mesh"), {}),
        "wh_t3k_ubench_2x2": (wh, "file:" + yamls + "test_fabric_ubench_at_least_2x2_mesh.yaml", {}),
        "wh_t3k_sanity_2x2": (wh, "file:" + yamls + "test_fabric_sanity_at_least_2x2_mesh.yaml", {}),
        "wh_t3k_sanity_common": (wh, "file:" + yamls + "test_fabric_sanity_common.yaml", {}),
    })
    return m


def elfs(m, case):
    path = os.path.join(m.COV, "cases", case, "cmds.json")
    if not os.path.exists(path):
        return None
    out = {}
    for key, cmds in json.load(open(path)).items():
        mm = re.search(r"\s-o\s+(\S+\.elf)", cmds["link"][0])
        if mm:
            elf = os.path.join(os.path.dirname(mm.group(1)), os.path.basename(mm.group(1)).split(".")[0] + ".elf")
            if os.path.exists(elf):
                out[key] = elf
    return out


def main():
    mode, rest = sys.argv[1], sys.argv[2:]  # load_cov resets sys.argv
    if mode == "mock":
        tag, cases = rest[0], rest[1:]
        m = load_cov(tag)
        m.ensure_root()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(int(os.environ.get("GATE_JOBS", "4"))) as ex:
            list(ex.map(m.run_mock, cases))
    elif mode == "cmp":
        cases = rest
        mb, ma = load_cov("before"), load_cov("after")
        bad = 0
        for c in cases:
            eb, ea = elfs(mb, c), elfs(ma, c)
            if eb is None or ea is None:
                print(f"{c:28s} MISSING ({'before' if eb is None else 'after'} not run)")
                bad += 1
                continue
            shared = sorted(set(eb) & set(ea))
            kinds = collections.Counter()
            diffs = []
            for k in shared:
                kind = "router" if "/fabric_erisc_router/" in k else ("firmware" if k.startswith("firmware/") else "other")
                same = mb.image(eb[k]) == ma.image(ea[k])
                kinds[(kind, same)] += 1
                if not same:
                    diffs.append(k)
            only_b, only_a = sorted(set(eb) - set(ea)), sorted(set(ea) - set(eb))
            status = "IDENTICAL" if not diffs and not only_b and not only_a else "DIFFER"
            bad += status != "IDENTICAL"
            print(f"{c:28s} {status:9s} router {kinds[('router', True)]}/{kinds[('router', True)] + kinds[('router', False)]}"
                  f" identical, other {kinds[('other', True)]}/{kinds[('other', True)] + kinds[('other', False)]}, "
                  f"firmware {kinds[('firmware', True)]}/{kinds[('firmware', True)] + kinds[('firmware', False)]}; "
                  f"only before {len(only_b)}, only after {len(only_a)}")
            for k in (diffs + only_b + only_a)[:6]:
                print(f"     {'differs' if k in diffs else 'only before' if k in only_b else 'only after'}: {k}")
        sys.exit(1 if bad else 0)
    elif mode == "sizes":
        ma = load_cov("after")
        rd = ma.OC.replace("objcopy", "readelf")
        for c in rest:
            ea = elfs(ma, c) or {}
            rows = []
            for k, elf in sorted(ea.items()):
                if "/fabric_erisc_router/" not in k:
                    continue
                ph = subprocess.run([rd, "-l", "-W", elf], capture_output=True, text=True).stdout
                loads = [(int(x.group(1), 16), int(x.group(2), 16), x.group(3)) for x in
                         re.finditer(r"LOAD\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+([RWE ]+?)\s+0x", ph)]
                text = sum(sz for va, sz, fl in loads if "E" in fl)
                data = sum(sz for va, sz, fl in loads if "E" not in fl)
                iram = any(va >= 0xFFC00000 for va, sz, fl in loads if "E" in fl)
                rows.append((text, data, iram, k))
            if not rows:
                print(f"{c}: no router builds")
                continue
            t = max(r[0] for r in rows)
            dmax = max(r[1] for r in rows)
            iram = any(r[2] for r in rows)
            lim = 16384 if iram else 17312
            print(f"{c:24s} {len(rows)} router builds; largest text {t} B of {lim} ({'IRAM' if iram else 'L1 kernel code'}, "
                  f"{100 * t / lim:.1f}%), largest data {dmax} B of 6144")


if __name__ == "__main__":
    main()
