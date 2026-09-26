# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair the current minimal prefill QKV DRAM/L1 input boundaries."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import describe, digest
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_prefill_pairs import attach_prefill_pairs


class InputPlacement:
    def __init__(self, source, ttnn):
        self.source = source
        self.ttnn = ttnn
        self.enabled = True
        self.observed = {}

    def __call__(self, value, *, memory_config=None):
        before = str(value.memory_config())
        if self.enabled:
            value = self.ttnn.to_memory_config(value, self.ttnn.L1_MEMORY_CONFIG)
        key = "candidate" if self.enabled else "baseline"
        if key not in self.observed:
            self.observed[key] = dict(
                input_shape=list(value.shape),
                input_dtype=str(value.dtype),
                input_before_memory=before,
                input_memory=str(value.memory_config()),
                requested_output_memory=str(memory_config),
            )
        return self.source(value, memory_config=memory_config)

    def __getattr__(self, name):
        return getattr(self.source, name)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--pairs", type=int, default=8)
    args, rest = parser.parse_known_args()
    if not all(flag in rest for flag in ("--defaults", "--real", "--input-fixture", "--verify-program-cache")):
        parser.error("Use current defaults, real input fixture and program-cache verification")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)

    import torch

    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import MinimalPrefillQKV, OptimizedDecoder

    torch.set_num_threads(4)
    paths = [
        Path(__file__).parents[1] / "tt/optimized_decoder.py",
        Path(__file__),
        Path(__file__).with_name("probe_optimized_prefill_pairs.py"),
        Path(__file__).with_name("probe_optimized_minimal_advice.py"),
    ]
    hashes = {str(path): digest(path) for path in paths}
    factory = OptimizedDecoder.from_state_dict.__func__
    metadata, paired = {}, {}
    controls = []

    def build(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        qkv = decoder.layer.self_attn.source.weights.wqkv
        if not isinstance(qkv, MinimalPrefillQKV):
            raise TypeError("Expected selected minimal prefill QKV")
        metadata["projection"] = describe(qkv.projection)
        metadata["decode_policy"] = decoder.precision_policy["qkv_decode"]
        control = InputPlacement(qkv.projection, ttnn)
        qkv.projection = control
        controls.append(control)

        def select(enabled):
            control.enabled = enabled

        attach_prefill_pairs(decoder, kw["mesh_device"], select, paired, pairs=args.pairs, ttnn_module=ttnn)
        return decoder

    failure = None
    try:
        with (
            patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
            patch.object(torch, "set_num_threads", lambda _: None),
            patch.object(sys, "argv", [sys.argv[0], *rest]),
        ):
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report.update(
            qkv_input_l1=dict(
                candidate="L1 interleaved input; all projection configs/dtypes unchanged",
                metadata=metadata,
                observed=[control.observed for control in controls],
                source_hashes=hashes,
            ),
            paired_prefill=paired,
        )
        if failure:
            report["qkv_input_l1_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not all(digest(path) == expected for path, expected in hashes.items()):
            raise RuntimeError("Published source changed during the QKV placement control")


if __name__ == "__main__":
    main()
