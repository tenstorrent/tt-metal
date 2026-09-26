# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair direct L1 normalization production against copied L1 input for full QKV."""

import argparse
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import describe, digest
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_prefill_pairs import attach_prefill_pairs


class ProducerControl:
    def __init__(self, decoder, projection, ttnn, mode):
        self.decoder = decoder
        self.source = projection
        self.ttnn = ttnn
        self.mode = mode
        self.enabled = True
        self.observed = {}
        self.original_normalize = decoder.normalize
        self.baseline_programs = tuple(
            ttnn.MinimalMatmulConfig(
                M_block_size=min(2, program.M_block_size),
                K_block_size=program.K_block_size,
                N_block_size=program.N_block_size,
                subblock_h=program.subblock_h,
                subblock_w=program.subblock_w,
                compute_with_storage_grid_size=program.compute_with_storage_grid_size,
            )
            for program in projection.programs
        )
        self.candidate_programs = (
            tuple(
                ttnn.MinimalMatmulConfig(
                    M_block_size=program.M_block_size,
                    K_block_size=program.K_block_size,
                    N_block_size=program.N_block_size,
                    subblock_h=program.subblock_h,
                    subblock_w=program.subblock_w,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                )
                for program in self.baseline_programs
            )
            if mode == "grid110"
            else self.baseline_programs
        )
        self.policies = []
        for programs in (self.baseline_programs, self.candidate_programs):
            policy = copy.deepcopy(projection.precision_policy)
            grid = programs[-1].compute_with_storage_grid_size
            policy.update(
                m_block="min(2, padded_M_tiles)",
                grid=(grid.x, grid.y),
                programs={str(index + 1): str(program) for index, program in enumerate(programs)},
            )
            self.policies.append(policy)

    @property
    def direct_producer(self):
        return self.mode == "grid110" or self.enabled

    def select(self, enabled):
        self.enabled = enabled
        self.source.programs = self.candidate_programs if enabled else self.baseline_programs
        self.source.precision_policy = self.policies[int(enabled)]
        self.decoder.precision_policy["prefill_qkv_projection"] = self.source.precision_policy

    def normalize(self, value, epsilon, weight=None):
        if self.direct_producer and value.shape[-2] > 1 and weight is self.decoder.input_norm_weight:
            result = self.original_normalize(value, epsilon, weight=None)
            return self.ttnn.mul(result, weight, memory_config=self.ttnn.L1_MEMORY_CONFIG)
        return self.original_normalize(value, epsilon, weight)

    def __call__(self, value, *, memory_config=None):
        key = "candidate" if self.enabled else "baseline"
        before = str(value.memory_config()) if key not in self.observed else None
        if not self.direct_producer:
            value = self.ttnn.to_memory_config(value, self.ttnn.L1_MEMORY_CONFIG)
        if key not in self.observed:
            if value.memory_config() != self.ttnn.L1_MEMORY_CONFIG:
                raise AssertionError("The direct producer did not supply an L1 QKV input")
            self.observed[key] = dict(
                input_shape=list(value.shape),
                input_dtype=str(value.dtype),
                input_before_memory=before,
                input_memory=str(value.memory_config()),
                requested_output_memory=str(memory_config),
                explicit_input_copy=not self.direct_producer,
                direct_input_norm_gamma_output_l1=self.direct_producer,
            )
        return self.source(value, memory_config=memory_config)

    def __getattr__(self, name):
        return getattr(self.source, name)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--producer-control", choices=("producer", "grid110"), default="producer")
    parser.add_argument("--pairs", type=int, default=32)
    args, rest = parser.parse_known_args()
    if not all(flag in rest for flag in ("--defaults", "--real", "--input-fixture", "--verify-program-cache")):
        parser.error("Use current defaults, real input fixture and program-cache verification")
    if "--layer" not in rest or rest[rest.index("--layer") + 1] != "5":
        parser.error("This precision-locked control is only for the full-attention layer 5")
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
        Path(__file__).parents[1] / "tt/fused_decoder.py",
        Path(__file__),
        Path(__file__).with_name("probe_optimized_prefill_pairs.py"),
        Path(__file__).with_name("probe_optimized_minimal_advice.py"),
    ]
    hashes = {str(path): digest(path) for path in paths}
    factory = OptimizedDecoder.from_state_dict.__func__
    metadata, paired, controls = {}, {}, []

    def build(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        qkv = decoder.layer.self_attn.source.weights.wqkv
        if not isinstance(qkv, MinimalPrefillQKV):
            raise TypeError("Expected the selected minimal prefill QKV")
        projection = qkv.projection
        if projection.compute.math_fidelity != ttnn.MathFidelity.HiFi2:
            raise ValueError("This control preserves the selected full-attention HiFi2 policy")
        if any(program.K_block_size != 16 or program.N_block_size != 8 for program in projection.programs):
            raise ValueError("Expected the selected full-attention K16/N8 geometry")
        if args.producer_control == "grid110":
            available = kw["mesh_device"].compute_with_storage_grid_size()
            if available.x < 11 or available.y < 10:
                raise ValueError("The grid control requires 110 available cores")
        control = ProducerControl(decoder, projection, ttnn, args.producer_control)
        for enabled, name in ((False, "baseline"), (True, "candidate")):
            control.select(enabled)
            metadata[name] = describe(projection)
        metadata.update(
            decode_policy=copy.deepcopy(decoder.precision_policy["qkv_decode"]),
            scope=(
                "Full prefill input-normalization site only; original weightless normalization then identical "
                "gamma multiplication with L1 output. Decode and all other norm sites delegate unchanged."
            ),
            baseline=(
                metadata["baseline"]
                | dict(input_boundary="producer L1" if args.producer_control == "grid110" else "DRAM plus L1 copy")
            ),
            candidate=metadata["candidate"] | dict(input_boundary="producer L1"),
        )
        decoder.normalize = control.normalize
        qkv.projection = control
        controls.append(control)
        attach_prefill_pairs(decoder, kw["mesh_device"], control.select, paired, pairs=args.pairs, ttnn_module=ttnn)
        return decoder

    failure = None
    try:
        with (
            patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
            patch.object(torch, "set_num_threads", lambda _: None),
            patch.object(sys, "argv", [sys.argv[0], *rest]),
        ):
            run_optimized_decoder.main()
        if len(paired["samples"]) != 2 * args.pairs:
            raise AssertionError("Incomplete paired samples")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report.update(
            qkv_producer_control=dict(
                arguments=vars(args),
                metadata=metadata,
                observed=[control.observed for control in controls],
                source_hashes=hashes,
            ),
            paired_prefill=paired,
        )
        if failure:
            report["qkv_producer_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not all(digest(path) == expected for path, expected in hashes.items()):
            raise RuntimeError("Published source changed during the producer control")


if __name__ == "__main__":
    main()
