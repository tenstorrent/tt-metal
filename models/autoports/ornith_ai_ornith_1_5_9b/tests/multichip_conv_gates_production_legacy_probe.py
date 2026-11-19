# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run unchanged legacy gates with production convolution and separate TTNN gates.

This is a diagnostic control, not a fused candidate. The original probe retains
its RNG sequence, independent histories/states, recurrence, checks and failure
handling. Only its fused.conv_gates callable is replaced process-locally.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import ttnn

from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_conv_gates_probe as probe
from models.autoports.ornith_ai_ornith_1_5_9b.tt import tiled_conv as production_conv

KERNEL_DIR = production_conv.KERNEL_DIR


def conv_gates(packed_tt, history, taps, bias_tt, decay_tt):
    batch = packed_tt.shape[2]
    raw_tt = packed_tt[:, :, :, :2048]
    conv_tt = production_conv.tiled_conv(raw_tt, history, taps)
    # These three assignments are AST-identical to the legacy probe's control.
    beta_tt = ttnn.reshape(ttnn.sigmoid(packed_tt[:, :, :, 3072:3080]), (batch, 1, 8))
    a_tt = ttnn.typecast(ttnn.reshape(packed_tt[:, :, :, 3104:3112], (1, batch, 8)), ttnn.float32)
    g_tt = ttnn.reshape(
        ttnn.multiply(
            ttnn.add(a_tt, bias_tt),
            decay_tt,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        ),
        (batch, 1, 8),
    )
    # The legacy candidate recurrence consumes a single height tile containing
    # all B requests. Its control uses [B,1,8]; own the layout-only conversion.
    return (
        conv_tt,
        ttnn.reshape(g_tt, (1, 1, batch, 8)),
        ttnn.reshape(beta_tt, (1, 1, batch, 8)),
    )


def main():
    metadata_args = argparse.ArgumentParser(add_help=False)
    metadata_args.add_argument("--output", required=True)
    args, _ = metadata_args.parse_known_args()
    probe.fused = sys.modules[__name__]
    try:
        probe.main()
    finally:
        # Preserve the original complete/error/check records, including any
        # uncaught strict-gate failure, and add the production helper provenance.
        output = Path(args.output)
        if output.is_file():
            result = json.loads(output.read_text())
            helper = Path(production_conv.__file__)
            result["sources"][str(helper)] = hashlib.sha256(helper.read_bytes()).hexdigest()
            result["diagnostic_substitution"] = (
                "Production tiled_conv + exact separate TTNN gates replaces only fused.conv_gates; "
                "legacy RNG, packed-gate recurrence, independent control and all thresholds unchanged"
            )
            output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
