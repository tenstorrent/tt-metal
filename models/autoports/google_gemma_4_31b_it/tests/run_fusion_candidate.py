"""Select an experimental graph explicitly, preserving the default fused entrypoint."""
import argparse

from models.autoports.google_gemma_4_31b_it.tests import run_fused_decoder, time_fusion_candidate
from models.autoports.google_gemma_4_31b_it.tests.fusion_candidates import (
    Candidate,
    ExtendedCandidate,
    SharedRopeCandidate,
)
from models.autoports.google_gemma_4_31b_it.tt.functional_decoder import FunctionalDecoder

p = argparse.ArgumentParser()
p.add_argument("--options", default="")
p.add_argument("--layer", type=int, default=0)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--seq", type=int, default=33)
p.add_argument("--result", required=True)
p.add_argument("--timing", action="store_true")
p.add_argument("--shared-rope", action="store_true")
p.add_argument("--extended-candidate", action="store_true")
p.add_argument("--functional", action="store_true")
a = p.parse_args()
Candidate.options = ExtendedCandidate.options = SharedRopeCandidate.options = frozenset(
    filter(None, a.options.split(","))
)
cls = (
    FunctionalDecoder
    if a.functional
    else (SharedRopeCandidate if a.shared_rope else (ExtendedCandidate if a.extended_candidate else Candidate))
)
if a.timing:
    time_fusion_candidate.FusedDecoder = cls
    time_fusion_candidate.run(a.layer, a.batch, a.result)
else:
    run_fused_decoder.FusedDecoder = cls
    a.synthetic = a.extended = a.new_bucket_case = a.skip_exact_maximum = False
    run_fused_decoder.run(a)
