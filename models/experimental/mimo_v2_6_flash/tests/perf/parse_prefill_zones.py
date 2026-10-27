# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Roll the tracy ops CSV of profile_prefill.py up into MiMo prefill zones (tt/profiling.py).

Runs models/demos/minimax_m3/tests/perf/parse_zone_perf.py (the CSV streaming, zone stack and per-device sums) with
MiMo's signpost names and layer classes: ``layer05_global``, ``layer01_swa``, ``layer00_global_dense``. Same options
(``--json``, ``--top``, ``--per-device``, ``--timeline``).

    python models/experimental/mimo_v2_6_flash/tests/perf/parse_prefill_zones.py <ops_perf_results_*.csv> --json z.json
"""

import importlib.util
import os
import sys

from models.experimental.mimo_v2_6_flash.tt import profiling

_PARSER = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "demos", "minimax_m3", "tests", "perf", "parse_zone_perf.py"
)
_spec = importlib.util.spec_from_file_location("parse_zone_perf", os.path.normpath(_PARSER))
pz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pz)


def layer_class(zone):
    """'profiled_chunk/layer05_global/attn/sdpa' -> ('global', 'attn/sdpa', 5); None for non-layer zones."""
    parts = zone.split("/")
    if len(parts) < 2 or not parts[1].startswith("layer"):
        return None
    return parts[1][8:], "/".join(parts[2:]), int(parts[1][5:7])


def relative_path(zone):
    lc = layer_class(zone)
    return zone if lc is None else f"{lc[0]}:{lc[1] or '(layer total)'}"


def print_report(summary, by_class, acc, top=0):
    total = summary.get(pz.ROOT_ZONE)
    if not total:
        print(f"No `{pz.ROOT_ZONE}` zone: was MIMO_PROFILE_ZONES=1 set, and did the run reach the profiled chunk?")
        return
    print(
        f"\nPROFILED CHUNK: {total['ms_max']:.2f} ms device-kernel time on the worst of {total['num_devices']} devices "
        f"(dev {total['worst_device']}), skew {total['skew_ms']:.2f} ms, {total['ops']} ops"
    )
    if acc.unmatched_ends:
        print(f"WARNING: {acc.unmatched_ends} unmatched zone END marker(s): the CSV may be truncated.")
    for zone, s in sorted(summary.items()):
        if zone.count("/") == 1 and layer_class(zone) is None:
            print(f"  {zone.split('/')[1]:<28} {s['ms_max']:>9.3f} ms")
    for cls, rels in sorted(by_class.items()):
        cls_total = rels.get("(layer total)", {}).get("ms_total", 0.0)
        layers = max(v["layers"] for v in rels.values())
        print(f"\n--- {cls}: {layers} layer(s), {cls_total:.2f} ms ({100 * cls_total / total['ms_max']:.1f}% of chunk)")
        print(f"  {'zone':<28} {'ms/layer':>9} {'ms total':>9} {'% class':>8} {'ops/L':>6} {'GB/s':>7}")
        for rel, v in sorted(rels.items(), key=lambda kv: -kv[1]["ms_total"]):
            pct = 100 * v["ms_total"] / cls_total if cls_total else 0.0
            print(
                f"  {rel:<28} {v['ms_per_layer']:>9.3f} {v['ms_total']:>9.2f} {pct:>7.1f}% "
                f"{v['ops_per_layer']:>6.1f} {v['gbs_mean']:>7.1f}"
            )
    if top:  # the costliest ops per zone, from the M3 report (its per-class tables need M3's classes: skipped)
        _m3_report(summary, {}, acc, top)


_m3_report = pz.print_report
pz.ZONE_START, pz.ZONE_END = profiling.ZONE_START, profiling.ZONE_END
pz.layer_class, pz.relative_path, pz.print_report = layer_class, relative_path, print_report

if __name__ == "__main__":
    sys.exit(pz.main())
