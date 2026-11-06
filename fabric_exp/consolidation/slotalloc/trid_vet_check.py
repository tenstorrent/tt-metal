#!/usr/bin/env python3
"""trid_vet_check.py --rule new|uniform [--group G] LOG...: the transaction-ID named args the host emitted, against the rule.

LOG: a mock or device run's log with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 (trid_vet.sh's mock runs). For every
fabric_erisc_router_v2 config the log built, the receivers (slots per channel, first-level acks) and deadlock avoidance
come from the config's CT args and named args (trid_offline_build.py's readers); the expected G / ids / start per
receiver channel from the rule (trid_offline_build.assign, the mirror of fabric_trid_allocator.cpp, checked against it
by the offline table's unit checks); the emitted ones from RX_CH_<c>_COMPLETION_GROUP_SIZE / _NUM_TRIDS / _TRID_START
in the config's named_ct_arg_map_generated.h. Unused channels must be 1 / 0 / 0. Prints each config and a summary;
exit 1 on any mismatch or a config without the args. Works for both ERISCs (the host emits the args for each).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trid_offline_build import assign, named_args, parse_log, receivers  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--rule", required=True, choices=["new", "uniform"])
    ap.add_argument("--group", type=int, default=0, choices=[0, 1, 2, 4],
                    help="TT_FABRIC_EXP_COMPLETION_GROUP the run set (0: unset)")
    a = ap.parse_args()
    configs = {}
    for log in a.logs:
        configs.update(parse_log(log))
    bad = 0
    seen = {}
    for gen, procs in sorted(configs.items()):
        named = named_args(os.path.join(gen, "named_ct_arg_map_generated.h"))
        rx, da = receivers(gen, procs[0][0], named)
        want = assign(rx, a.rule, da, a.group)
        got, missing = [], False
        for c in range(3):
            keys = [f"RX_CH_{c}_COMPLETION_GROUP_SIZE", f"RX_CH_{c}_NUM_TRIDS", f"RX_CH_{c}_TRID_START"]
            if any(k not in named for k in keys):
                missing = True
                break
            got.append(tuple(named[k] for k in keys))
        rx_text = " | ".join(f"{s}{'a' if f else ''}" for s, f in rx)
        want_text = " | ".join(f"{g}/{n}@{s}" for g, n, s in want)
        if missing:
            status = "NO RX_CH_* ARGS (host without the TRID patch?)"
        else:
            unused_ok = all(t == (1, 0, 0) for t in got[len(rx):])
            status = "ok" if got[: len(rx)] == [tuple(w) for w in want] and unused_ok else (
                "MISMATCH: emitted " + " | ".join(f"{g}/{n}@{s}" for g, n, s in got))
        bad += status != "ok"
        procs_text = ",".join(sorted(p for _, _, p in procs))
        key = (rx_text, int(da), want_text, status)
        seen.setdefault(key, []).append(f"{os.path.basename(gen)}({procs_text})")
    for (rx_text, da, want_text, status), gens in sorted(seen.items()):
        print(f"rx {rx_text} da {da} -> {want_text}: {status} [{len(gens)} configs: {', '.join(gens[:3])}"
              f"{', ...' if len(gens) > 3 else ''}]")
    print(f"{len(configs)} configs, {bad} not as the {a.rule} rule")
    print("TRID_VET_OK" if bad == 0 and configs else "TRID_VET_MISMATCH")
    sys.exit(0 if bad == 0 and configs else 1)


if __name__ == "__main__":
    main()
