# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate long-context accuracy changes without changing production defaults."""

import argparse
import sys

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--accuracy-control",
        choices=(
            "chunk1024",
            "sdpa_hifi4_k128",
            "sdpa_hifi2_k128",
            "sdpa_hifi4_k64",
            "sdpa_hifi2_k256",
            "sdpa_hifi2_k512",
        ),
        required=True,
    )
    options, remaining = parser.parse_known_args()

    class Control(MultichipDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            if options.accuracy_control == "chunk1024":
                kwargs["chunk_size"] = 1024
            layer = super().from_state_dict(*args, **kwargs)
            if options.accuracy_control.startswith("sdpa_hifi4"):
                layer.sdpa_decode_compute = layer.compute
            return layer

    if options.accuracy_control.startswith("sdpa_"):
        original = ttnn.transformer.paged_scaled_dot_product_attention_decode

        def sdpa(*args, **kwargs):
            chunk = int(options.accuracy_control.rsplit("k", 1)[1])
            kwargs["program_config"].k_chunk_size = chunk
            extra = (-args[3].shape[-1]) % (chunk // 32)
            if extra:
                table = ttnn.pad(args[3], [(0, 0), (0, extra)], 0)
                args = (*args[:3], table, *args[4:])
            return original(*args, **kwargs)

        ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa
    stack.MultichipDecoder = Control
    sys.argv = [sys.argv[0], *remaining]
    stack.main()


if __name__ == "__main__":
    main()
