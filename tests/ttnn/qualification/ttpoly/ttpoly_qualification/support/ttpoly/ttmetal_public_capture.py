# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Canonical public-operation raw capture, with no compiler or scorer imports.

Invocation and semantic data come from the compiled package. Callers own device
and build locks. Backward inputs are explicitly unit-gradient, not exhaustive
two-input combinations.
"""
import json
import math
from pathlib import Path


def leg(args):
    import numpy as np
    import torch
    import ttnn
    from ttpoly.ttmetal_public_invocation import PublicInvocation, unit_gradient
    from ttpoly.bf16_transport import run_bf16_exhaustive_accuracy

    root = args.root.resolve()
    for module in (ttnn, ttnn._ttnn):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"wrong TTNN import: {module.__file__}")
    requested_architecture = getattr(args, "architecture", None)
    if requested_architecture is not None:
        expected_architecture = {"bh": "blackhole", "wh": "wormhole_b0"}.get(requested_architecture)
        if expected_architecture is None or ttnn.get_arch_name() != expected_architecture:
            raise RuntimeError("actual device architecture differs from requested public capture target")
    invocation = PublicInvocation(**json.loads(args.leg.read_text()))
    gradient_count = 0

    def apply(tensor):
        nonlocal gradient_count
        gradient = None
        if "incoming_gradient" in invocation.tensor_roles:
            gradient = unit_gradient(ttnn, torch, tensor)
            gradient_count += math.prod(tuple(tensor.shape))
        return invocation.invoke(ttnn, tensor, gradient)

    device = ttnn.open_device(device_id=args.chip)
    try:
        coverage = run_bf16_exhaustive_accuracy(np, torch, ttnn, apply, device, args)
        if (
            coverage.get("complete") is not True
            or coverage.get("count") != 65536
            or args.out_bf16_bin.stat().st_size != 131072
        ):
            raise RuntimeError("incomplete exhaustive public output")
    finally:
        ttnn.close_device(device)
    if "incoming_gradient" in invocation.tensor_roles:
        if gradient_count != 65536:
            raise RuntimeError("unit-gradient transport did not cover every activation input")
        semantics = json.loads((args.leg.parent / "semantic.json").read_text())
        inputs = {
            "io_contract": semantics["io_contract"],
            "auxiliary_inputs": [
                {
                    "tensor_index": invocation.tensor_roles.index("incoming_gradient"),
                    "role": "incoming_gradient",
                    "encoding": "constant_bfloat16",
                    "raw_bits": "0x3f80",
                    "value": 1.0,
                    "count": gradient_count,
                }
            ],
        }
        args.out_bf16_bin.with_suffix(".inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print("PUBLIC_EXHAUSTIVE_OUTPUT_COMPLETE 65536")
