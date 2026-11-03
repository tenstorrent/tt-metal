#!/usr/bin/env python3
"""w_check.py BASE NEW: the v4txa11w / v4txa11g gate (V4_SENDER_CONSOLIDATION.md §6.12). For every router ELF the two
coverage builds share (cov_m builds/<name>), whether its loadable contents are identical (bnd_loadcmp.py's comparison),
and its role: E0 (active_erisc), E1 of a two-ERISC router (subordinate_active_erisc with an active_erisc beside it in the
same config), or a one-ERISC router; for E1, the forward-command form the image instantiates (DWARF type names: PartialPresetForwards,
FullPresetForwards, StatelessForwards; none when the image has no forwards).
Pass: every differing image is a two-ERISC E1 with PartialPreset (a multi-downstream receiver)."""
import collections, glob, multiprocessing, os, re, subprocess, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import bnd_loadcmp as LC  # noqa: E402
import loopvars as L  # noqa: E402

CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"


def mode(elf):
    """The forward-command form instantiated in the image (DWARF type names): PartialPreset where a VC's forwards
    pick among several downstreams with presets, FullPreset with one destination, Stateless without presets."""
    out = subprocess.run([L.B + "readelf", "--debug-dump=info", elf], capture_output=True, text=True).stdout
    found = [m for m in ("PartialPreset", "FullPreset", "Stateless") if re.search(r"DW_AT_name\s+:.*\b" + m + r"Forwards<", out)]
    return "+".join(found) if found else "none"


def job(rel):
    a, b = f"{CB}/{sys.argv[1]}/{rel}", f"{CB}/{sys.argv[2]}/{rel}"
    same = LC.loadable(a) == LC.loadable(b)
    cfg = os.path.dirname(os.path.dirname(rel))
    if "/active_erisc/" in rel:
        role = "E0"
    elif os.path.exists(f"{CB}/{sys.argv[2]}/{cfg}/active_erisc") or glob.glob(f"{CB}/{sys.argv[2]}/{cfg.split('__')[0]}__*/active_erisc"):
        role = "E1 (two ERISCs)"
    else:
        role = "one ERISC"
    m = mode(b) if role != "E0" else "-"
    return rel.split("__")[0], role, m, same


if __name__ == "__main__":
    rels = sorted(os.path.relpath(p, f"{CB}/{sys.argv[2]}") for p in glob.glob(f"{CB}/{sys.argv[2]}/*/*/*.elf")
                  if not p.endswith(".xip.elf"))
    rels = [r for r in rels if os.path.exists(f"{CB}/{sys.argv[1]}/{r}")]
    with multiprocessing.Pool(24) as pool:
        res = pool.map(job, rels)
    tab = collections.Counter((role, m, same) for case, role, m, same in res)
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(res)} shared ELFs")
    for (role, m, same), n in sorted(tab.items()):
        print(f"  {role:16s} {m:14s} {'identical' if same else 'DIFFER':9s} {n}")
    bad = [(case, role, m) for case, role, m, same in res if not same and not (role == "E1 (two ERISCs)" and "PartialPreset" in m)]
    cases = collections.Counter(case for case, role, m, same in res if not same)
    print("differing ELFs by case:", dict(sorted(cases.items())))
    print("PASS" if not bad else f"FAIL: {len(bad)} differing images outside two-ERISC PartialPreset E1, e.g. {bad[:5]}")
