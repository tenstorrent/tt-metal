# SPDX-License-Identifier: Apache-2.0
import argparse

from . import fused_coverage
from .fused_expert_candidate import ExpertCandidate
from .fusion_candidates import FusedDecoder

for (
    flag
) in "rope_fused,concat_fused,activation_fused,qkv_packed,packed_heads,routing_fused,decode_heads,shared_packed,mixed_weight,outputcast,router_cast,hf_rope,reduce_fused,reduce_all".split(
    ","
):
    setattr(FusedDecoder, flag, True)
fused_coverage.FusedDecoder = ExpertCandidate
p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, required=True)
a = p.parse_args()
fused_coverage.run(a.layer, output=f"expert_coverage_{a.layer}.json")
