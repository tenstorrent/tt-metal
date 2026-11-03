"""stage_a9.py: v4txa9 = v4txa4 + Part A9 of ROUTER_HELPERS_REVIEW.md (user-approved 2026-11-03: "we dont want random refs
in the code for the team to see"): every study pointer leaves the comments of every router file (router/*.hpp, the
kernel file): RESULTS sections, LEDGER ids, fabric_exp document names, job numbers, dates, tool names. Each becomes a
one-line why that stands on its own, or goes where the sentence already says why. Paragraphs these edits shortened are
rewrapped (reflow.py). run_handshakes' optimize("Os") gets its one-line why (the user's decision: an allowed primitive
for once-run init code only). Expected: byte-identical to v4txa4."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txa4", "v4txa9", "a9.patch")
