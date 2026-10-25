#!/usr/bin/env python3
"""CPU-only mock runs of the rebuilt host WITHOUT the study's keepalive override, to read what the host itself passes
as ETH_TXQ_SEQ_UPDATE_TIMEOUT per config type (coverage.py's mock cases, in a separate COV dir)."""
import glob, os, re, sys
S = "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/ka13"
os.environ["TT_COV_DIR"] = S + "/cov"
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import coverage as cv  # noqa: E402
del cv.BASE_ENV["TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT"]
os.environ.pop("TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT", None)
cv.ensure_root()
for case in sys.argv[1:]:
    cv.run_mock(case)
    vals = {}
    for m in glob.glob(f"{S}/cov/cases/{case}/cache/**/named_ct_arg_map_generated.h", recursive=True):
        t = open(m).read()
        k = tuple(re.search(r'\{"%s",(\d+)\}' % n, t).group(1) if re.search(r'\{"%s",' % n, t) else "-"
                  for n in ("ETH_TXQ_SEQ_UPDATE_TIMEOUT", "EXP_SEQ_UPDATE_TIMEOUT", "NUM_ACTIVE_ERISCS"))
        vals[k] = vals.get(k, 0) + 1
    print(f"  {case}: (ETH_TXQ_SEQ_UPDATE_TIMEOUT, EXP_SEQ_UPDATE_TIMEOUT, NUM_ACTIVE_ERISCS) -> configs: {vals}")
