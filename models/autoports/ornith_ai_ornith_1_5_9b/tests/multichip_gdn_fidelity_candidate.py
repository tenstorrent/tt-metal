# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate pooled GDN compute fidelity; preserve FP32 recurrent state."""
import argparse
import json
import sys
from pathlib import Path
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_create_candidate import QkvCreateCandidate


class GdnFidelityCandidate(QkvCreateCandidate):
    selected_fidelity = "LoFi"
    approximate = False

    def pooled_linear_attention_decode_batch(self, packed, states):
        original = self.compute
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.selected_fidelity),
            math_approx_mode=self.approximate, fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        try:
            return super().pooled_linear_attention_decode_batch(packed, states)
        finally:
            self.compute = original


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gdn-fidelity", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"], required=True)
    ap.add_argument("--approximate", action="store_true")
    args, rest = ap.parse_known_args()
    GdnFidelityCandidate.selected_fidelity = args.gdn_fidelity
    GdnFidelityCandidate.approximate = args.approximate
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder = GdnFidelityCandidate
    sys.argv = [sys.argv[0], *rest]
    stack.main()
    output = Path(rest[rest.index("--output") + 1]); data = json.loads(output.read_text())
    data["gdn_precision"] = vars(args)
    output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
