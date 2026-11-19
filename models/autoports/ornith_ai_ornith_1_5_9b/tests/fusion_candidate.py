# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit experiment overrides; production FusedDecoder never imports this module."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import validate_fused
from models.autoports.ornith_ai_ornith_1_5_9b.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=["separate_activation", "separate_binary", "interleaved_decode", "packed_prefill", "chunk_decode"],
    )
    args, rest = parser.parse_known_args()

    class Candidate(FusedDecoder):
        use_dram_decode = False

        @classmethod
        def from_state_dict(cls, sd, **kwargs):
            self = super().from_state_dict(sd, **kwargs)
            if args.candidate.startswith("separate"):
                for role in ["gate", "up"]:
                    self.weights[f"{role}.weight"] = ttnn.from_torch(
                        sd[f"mlp.{role}_proj.weight"].T.contiguous(),
                        device=self.device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
            return self

        def linear_attention(self, x, state, *, decode):
            return super().linear_attention(x, state, decode=False if args.candidate == "chunk_decode" else decode)

        def finish(self, x, mixed):
            if args.candidate == "chunk_decode":
                return super().finish(x, mixed)
            if x.shape[2] != 1 and args.candidate != "packed_prefill":
                return super().finish(x, mixed)
            residual = ttnn.add(x, mixed)
            h = self.norm(residual, "post_attention_layernorm")
            if args.candidate == "separate_activation":
                gate = ttnn.linear(
                    h, self.weights["gate.weight"], activation="silu", compute_kernel_config=self.compute
                )
                h = ttnn.multiply(gate, self.linear(h, "up"))
            elif args.candidate == "separate_binary":
                h = ttnn.multiply(
                    self.linear(h, "gate"), self.linear(h, "up"), input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                )
            elif args.candidate == "interleaved_decode":
                gu = self.linear(h, "packed_mlp")
                gu = ttnn.reshape(gu, (1, 1, 384, 64))
                gate = ttnn.reshape(gu[:, :, :, :32], (1, 1, 1, 12288))
                up = ttnn.reshape(gu[:, :, :, 32:], (1, 1, 1, 12288))
                h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            else:
                gu = self.linear(h, "gate_up")
                h = ttnn.multiply(
                    gu[:, :, :, :12288], gu[:, :, :, 12288:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                )
            return ttnn.add(residual, self.linear(h, "mlp.down_proj"))

    validate_fused.FusedDecoder = Candidate
    sys.argv = [sys.argv[0], *rest]
    validate_fused.main()
    output = Path(rest[rest.index("--output") + 1])
    report = json.loads(output.read_text())
    report.update(
        candidate=args.candidate, candidate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
