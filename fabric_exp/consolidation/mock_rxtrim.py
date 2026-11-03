#!/usr/bin/env python3
"""mock_rxtrim.py VARIANT CASE...: capture receiver-trimmed shapes on the mock (CPU only), with coverage_txo.py's
mock_pinned (a pinned runtime root; V4_SENDER_OWNERSHIP.md §6.4), VARIANT in the v2 slot. The cases (V4_RECEIVER_OWNERSHIP
§18): VC0's receiver channel trimmed on every router (trimx/vc0_receiver_trimmed.yaml), so the lowest receiver VC run
is VC1 (bh_vc1) or VC2 (bh_2d_vc2)."""
import importlib.util, os, sys
os.environ.setdefault("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m")
os.environ.setdefault("OFFLINE_BASE", "/home/aagarwal/.tt-metal-p1-base-m")
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("coverage_txo", os.path.join(here, "coverage_txo.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
RX_TRIM_CASES = {
    "bh_vc1_trim_rx0": m.trim_case("bh_vc1", "vc0_receiver_trimmed"),
    "bh_vc1_trim_rx0_1erisc": m.trim_case("bh_vc1_1erisc", "vc0_receiver_trimmed"),
    "bh_2d_vc2_trim_rx0": m.trim_case("bh_2d_vc2", "vc0_receiver_trimmed"),
    "bh_2d_vc2_trim_rx0_1erisc": m.trim_case("bh_2d_vc2_1erisc", "vc0_receiver_trimmed"),
}
m.CASES.update(RX_TRIM_CASES)
if __name__ == "__main__":
    variant, cases = sys.argv[1], sys.argv[2:]
    for c in cases:
        if c not in RX_TRIM_CASES:
            sys.exit(f"unknown case {c}")
        if os.path.exists(os.path.join(m.COV, "cases", c)):
            sys.exit(f"case {c} exists")
    for c in cases:
        m.mock_pinned(c, variant)
