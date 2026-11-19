# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing mesh probe with an explicit normal-constructor precision policy.

--host-evidence optionally retains CPU tensors the probe already read for its
accuracy checks. This adds no device readbacks and does not patch KV allocation,
updates, or gathers. Saved decoder residual argmax is not a vocabulary token.
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--host-evidence", type=Path)
    options, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("datatype_probe_precision", root / "tt/precision.py")
    precision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precision)
    policy = precision.load_policy(root, options.precision_config)
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode")
    mode, _ = mode_parser.parse_known_args(remaining)
    if mode.mode != "mesh":
        parser.error("Explicit datatype policy requires --mode mesh")

    # Imports intentionally occur only when the parent launches a bounded device job.
    import torch

    from models.autoports.qwen_qwen3_8_flash_next.tests import probe_multichip_decoder as probe

    original_factory = probe.MultichipDecoder
    original_pcc = probe.pcc
    constructions = []
    evidence = {"schema": 1, "precision": policy, "probe_argv": remaining, "rows": [], "batch": []}

    class ExplicitPrecisionDecoder(original_factory):
        @classmethod
        def from_state_dict(cls, state_dict, **kwargs):
            index = kwargs["layer_idx"]
            resolved = precision.layer_policy(policy, index)
            result = super().from_state_dict(state_dict, precision=resolved, **kwargs)
            constructions.append({"layer": index, "resolved_policy": resolved})
            return result

    def observe_pcc(expected, actual):
        # Observe existing host operands, outside the probe's timed device windows.
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code is probe.main.__code__ and frame.f_locals.get("out") is actual:
                evidence["rows"].append(
                    {
                        "step": frame.f_locals["i"],
                        "input": frame.f_locals["saved"]["inputs"][frame.f_locals["i"]].clone(),
                        "output": actual.clone(),
                    }
                )
            else:
                # Python 3.12 inlines list comprehensions; earlier versions have
                # a separate frame. Support both without tracing device calls.
                parent = frame if frame.f_code is probe.main.__code__ else frame.f_back
                if parent.f_code is probe.main.__code__:
                    local = frame.f_locals
                    outputs = parent.f_locals.get("own_expected", [])
                    if (
                        local.get("expected") is expected
                        and local.get("actual") is actual
                        and any(value is actual for value in outputs)
                    ):
                        index = len(evidence["batch"])
                        evidence["batch"].append(
                            {
                                "user": index,
                                "input": parent.f_locals["saved"]["batch"]["inputs"][index].clone(),
                                "output": actual.clone(),
                            }
                        )
        finally:
            del frame
        return original_pcc(expected, actual)

    probe.MultichipDecoder = ExplicitPrecisionDecoder
    if options.host_evidence:
        probe.pcc = observe_pcc
    old_argv = sys.argv
    sys.argv = [old_argv[0], *remaining]
    try:
        probe.main()
    finally:
        probe.MultichipDecoder = original_factory
        probe.pcc = original_pcc
        sys.argv = old_argv
        if options.host_evidence:
            evidence["constructions"] = constructions
            evidence[
                "scope"
            ] = "Decoder residual outputs, not vocabulary logits; per-user token agreement needs matched full-model logits"
            options.host_evidence.parent.mkdir(parents=True, exist_ok=True)
            torch.save(evidence, options.host_evidence)


if __name__ == "__main__":
    main()
