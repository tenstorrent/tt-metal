# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate the isolated AG / MM sweep specs from the AGMM shape spec.

The isolated sweeps break each fused AGMM shape into its two halves:
  - isolated_ag_spec.json : all_gather_async, deduped by (device_config, M, K)
                            (the gather is independent of N and fusion)
  - isolated_mm_spec.json : minimal_matmul, one per AGMM shape (keeps N + fusion
                            so the matmul matches what AGMM fuses)

Run this whenever sweep_shapes.json changes, then re-run run_isolated.sh:

    python agmm/gen_isolated_specs.py
"""

import json
import os

_D = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(_D, "sweep_shapes.json")
AG_OUT = os.path.join(_D, "isolated_ag_spec.json")
MM_OUT = os.path.join(_D, "isolated_mm_spec.json")


def _base(s, op, id_):
    """A spec entry mirroring s but for a single isolated op."""
    return {
        "id": id_,
        "op_type": op,
        "device_config": s["device_config"],
        "M": s["M"],
        "K": s["K"],
        "N": s["N"],
        "grid": s["grid"],
        "dtype": s.get("dtype", "bfloat16"),
        "math_fidelity": s.get("math_fidelity", "HiFi2"),
        # AG ignores fusion/N; MM keeps fusion so it matches AGMM's matmul.
        "fusion": {} if op == "ag" else (s.get("fusion") or {}),
        "tags": s.get("tags", []),
        "notes": ("isolated AG for " if op == "ag" else "isolated MM for ") + s["id"],
    }


def main():
    spec = json.load(open(SPEC))

    ag = {}
    for s in spec:
        key = (s["device_config"], s["M"], s["K"])  # gather is independent of N/fusion
        if key not in ag:
            ag[key] = _base(s, "ag", f"ag_{s['device_config']}_m{s['M']}_k{s['K']}")
    ag_spec = list(ag.values())

    mm_spec = [_base(s, "mm", s["id"] + "_mm") for s in spec]

    json.dump(ag_spec, open(AG_OUT, "w"), indent=1)
    json.dump(mm_spec, open(MM_OUT, "w"), indent=1)
    print(f"wrote {AG_OUT} ({len(ag_spec)} AG shapes)")
    print(f"wrote {MM_OUT} ({len(mm_spec)} MM shapes)")


if __name__ == "__main__":
    main()
