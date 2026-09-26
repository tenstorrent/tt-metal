# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test L1 input placement at the selected 1024-row prefill boundaries."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest


class PlacementControl:
    """Setup-only programs and a toggle shared by screen and paired timing harnesses."""

    def __init__(self, decoder, mesh, candidate, ttnn):
        self.ttnn = ttnn
        self.candidate = candidate
        self.enabled = True
        self.active_shared = False
        self.original_linear = ttnn.linear
        self.attention = decoder.layer.self_attn
        self.metadata = dict(candidate=candidate, observed={})
        if candidate == "output_l1":
            if self.attention.prefill_minimal_output is None:
                raise ValueError("Output placement control requires the selected full minimal output")
            self.attention.prefill_output_l1 = True
            self.metadata["original_output_policy"] = dict(decoder.precision_policy["prefill_output_projection"])
            decoder.precision_policy["prefill_output_projection"]["input_l1"] = True
        elif candidate == "shared_l1":
            self.compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=False,
            )
            # These are the actual default 1024-row native programs in the v5
            # profiles. Lock them so input placement cannot change autotuning.
            self.programs = {
                (2816, 4224): self._program(8, 12),
                (2112, 2816): self._program(6, 8),
            }
            source = decoder.layer.shared_mlp.source

            def shared_prefill(value):
                if value.shape[-2] == 1:
                    raise ValueError("This wrapper must not intercept decode")
                self.active_shared = True
                try:
                    return source(value)
                finally:
                    self.active_shared = False

            decoder.layer.shared_mlp.source = shared_prefill
            self.metadata["programs"] = {str(key): str(value) for key, value in self.programs.items()}
            self.metadata[
                "compute"
            ] = "HiFi2; math_approx=False; fp32_dest=False; packer_l1_acc=True; dst_full_sync=False"
        else:
            raise ValueError(candidate)

    def _program(self, block, per_n):
        ttnn = self.ttnn
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            in0_block_w=block,
            out_subblock_h=4,
            out_subblock_w=2,
            out_block_h=4,
            out_block_w=per_n,
            per_core_M=4,
            per_core_N=per_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def set_enabled(self, enabled):
        self.enabled = enabled
        if self.candidate == "output_l1":
            self.attention.prefill_output_l1 = enabled

    def linear(self, value, weight, *args, **kwargs):
        if not self.active_shared:
            return self.original_linear(value, weight, *args, **kwargs)
        ttnn = self.ttnn
        key = (weight.shape[-2], weight.shape[-1])
        if value.shape[-2] != 1024 or key not in self.programs:
            raise ValueError("Shared placement probe is bounded to actual 1024-row production chunks")
        assert value.dtype == weight.dtype == ttnn.bfloat16
        before_memory = str(value.memory_config())
        if self.enabled:
            value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)
        kwargs.update(
            program_config=self.programs[key],
            compute_kernel_config=self.compute,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if str(key) not in self.metadata["observed"]:
            self.metadata["observed"][str(key)] = dict(
                input_shape=list(value.shape),
                input_dtype=str(value.dtype),
                weight_shape=list(weight.shape),
                weight_dtype=str(weight.dtype),
                before_memory=before_memory,
                after_memory=str(value.memory_config()),
                output_dtype="bfloat16",
                output_memory="DRAM interleaved",
                program=str(self.programs[key]),
            )
        return self.original_linear(value, weight, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prefill-placement", choices=("output_l1", "shared_l1"), required=True)
    args, rest = parser.parse_known_args()
    if "--defaults" not in rest:
        parser.error("Use selected --defaults")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)

    import torch

    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    torch.set_num_threads(4)
    runtime = Path(__file__).parents[1] / "tt/optimized_decoder.py"
    source_hash, probe_hash = digest(runtime), digest(__file__)
    factory = OptimizedDecoder.from_state_dict.__func__
    controls = []
    original_linear = ttnn.linear

    def build(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        control = PlacementControl(decoder, kw["mesh_device"], args.prefill_placement, ttnn)
        control.original_linear = original_linear
        controls.append(control)
        return decoder

    def linear(*a, **kw):
        return controls[-1].linear(*a, **kw) if controls else original_linear(*a, **kw)

    failure = None
    try:
        with (
            patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
            patch.object(ttnn, "linear", linear),
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
            prefill_placement_candidate=vars(args),
            prefill_placement_runtime=[control.metadata for control in controls],
            prefill_placement_runtime_sha256=source_hash,
            prefill_placement_probe_sha256=probe_hash,
        )
        if failure:
            report["prefill_placement_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if digest(runtime) != source_hash or digest(__file__) != probe_hash:
            raise RuntimeError("Published runtime/probe changed during placement control")


if __name__ == "__main__":
    main()
