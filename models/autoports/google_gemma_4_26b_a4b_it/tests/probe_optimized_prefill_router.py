# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair one phase-specific router fidelity or producer-placement change."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_prefill_pairs import attach_prefill_pairs


class RouterControl:
    def __init__(self, decoder, mesh, candidate, ttnn, linear, mul):
        self.ttnn, self.candidate = ttnn, candidate
        self.original_linear, self.original_mul = linear, mul
        self.enabled = True
        router = decoder.layer.moe.router
        if not router.direct_projection:
            raise ValueError("Keep the selected independent decode router projection")
        source = router.original.source
        self.scale, self.weight = source.scale, source.source.proj_weight
        self.baseline_compute = source.compute
        if self.baseline_compute.math_fidelity != ttnn.MathFidelity.HiFi4:
            raise ValueError("Expected the selected prefill-router HiFi4 policy")
        self.alternate_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=self.baseline_compute.math_approx_mode,
            fp32_dest_acc_en=self.baseline_compute.fp32_dest_acc_en,
            packer_l1_acc=self.baseline_compute.packer_l1_acc,
            dst_full_sync_en=self.baseline_compute.dst_full_sync_en,
            throttle_level=self.baseline_compute.throttle_level,
        )
        # Lock the observed 1024-row native geometry on both sides so placement
        # cannot silently change the automatic program selection.
        self.program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            in0_block_w=8,
            out_subblock_h=4,
            out_subblock_w=1,
            out_block_h=4,
            out_block_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))}),
        )
        self.metadata = dict(
            candidate=candidate,
            baseline_compute=str(self.baseline_compute),
            candidate_compute=str(self.alternate_compute if candidate == "hifi2" else self.baseline_compute),
            program=str(self.program),
            decode_projection_compute=str(router.projection_compute),
            scale_dtype=str(self.scale.dtype),
            weight_dtype=str(self.weight.dtype),
            weight_shape=list(self.weight.shape),
            observed={},
            scope="Only the 1024-row prefill router projection; topk/softmax/scatter, per-expert scaling, decode and every other projection remain unchanged.",
        )

    def select(self, enabled):
        self.enabled = enabled

    def mul(self, *args, **kwargs):
        if (
            self.enabled
            and self.candidate == "producer_l1"
            and len(args) >= 2
            and args[1] is self.scale
            and args[0].shape[-2] > 1
        ):
            kwargs["memory_config"] = self.ttnn.L1_MEMORY_CONFIG
        return self.original_mul(*args, **kwargs)

    def linear(self, value, weight, *args, **kwargs):
        if weight is not self.weight or value.shape[-2] == 1:
            return self.original_linear(value, weight, *args, **kwargs)
        ttnn = self.ttnn
        if tuple(value.shape) != (1, 1, 1024, 2816):
            raise ValueError("This control requires the selected 1024-row prefill chunks")
        if value.dtype != ttnn.float32 or weight.dtype != ttnn.bfloat16:
            raise ValueError("Preserve actual FP32 router activations and BF16 weights")
        compute = self.alternate_compute if self.enabled and self.candidate == "hifi2" else self.baseline_compute
        kwargs.update(
            program_config=self.program,
            compute_kernel_config=compute,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = "candidate" if self.enabled else "baseline"
        if key not in self.metadata["observed"]:
            expected = (
                ttnn.L1_MEMORY_CONFIG if self.enabled and self.candidate == "producer_l1" else ttnn.DRAM_MEMORY_CONFIG
            )
            if value.memory_config() != expected:
                raise AssertionError("Router producer memory differs from the selected control")
            self.metadata["observed"][key] = dict(
                input_shape=list(value.shape),
                input_dtype=str(value.dtype),
                input_memory=str(value.memory_config()),
                compute=str(compute),
                program=str(self.program),
                output_dtype="float32",
                output_memory="DRAM interleaved",
                explicit_input_copy=False,
            )
        return self.original_linear(value, weight, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--router-prefill", choices=("hifi2", "producer_l1"), required=True)
    parser.add_argument("--pairs", type=int, default=32)
    args, rest = parser.parse_known_args()
    if not all(flag in rest for flag in ("--defaults", "--real", "--input-fixture", "--verify-program-cache")):
        parser.error("Use selected defaults, actual input fixture and program-cache verification")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)

    import torch

    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

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
    original_linear, original_mul = ttnn.linear, ttnn.mul
    controls, paired = [], {}

    def build(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        control = RouterControl(decoder, kw["mesh_device"], args.router_prefill, ttnn, original_linear, original_mul)
        controls.append(control)
        attach_prefill_pairs(decoder, kw["mesh_device"], control.select, paired, pairs=args.pairs, ttnn_module=ttnn)
        return decoder

    def linear(*a, **kw):
        return controls[-1].linear(*a, **kw) if controls else original_linear(*a, **kw)

    def mul(*a, **kw):
        return controls[-1].mul(*a, **kw) if controls else original_mul(*a, **kw)

    failure = None
    try:
        with (
            patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
            patch.object(ttnn, "linear", linear),
            patch.object(ttnn, "mul", mul),
            patch.object(torch, "set_num_threads", lambda _: None),
            patch.object(sys, "argv", [sys.argv[0], *rest]),
        ):
            run_optimized_decoder.main()
        if len(paired["samples"]) != 2 * args.pairs:
            raise AssertionError("Incomplete alternating pair samples")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report.update(
            router_prefill_control=dict(
                arguments=vars(args), metadata=[control.metadata for control in controls], source_hashes=hashes
            ),
            paired_prefill=paired,
        )
        if failure:
            report["router_prefill_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not all(digest(path) == expected for path, expected in hashes.items()):
            raise RuntimeError("Published source changed during the prefill-router control")


if __name__ == "__main__":
    main()
