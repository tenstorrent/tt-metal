# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Change one setup-time policy on the selected minimal prefill QKV path."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(projection):
    compute = projection.compute
    return dict(
        weight_dtype=str(projection.weight.dtype),
        weight_shape=list(projection.weight.shape),
        weight_memory=str(projection.weight.memory_config()),
        programs=[str(program) for program in projection.programs],
        compute={
            name: str(getattr(compute, name))
            for name in (
                "math_fidelity",
                "math_approx_mode",
                "fp32_dest_acc_en",
                "packer_l1_acc",
                "dst_full_sync_en",
                "throttle_level",
            )
        },
        output_dtype="float32",
        input_dtype="unchanged (selected FP32 prefill activation)",
    )


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--minimal-advice", choices=("baseline", "hifi2", "grid110"), required=True)
    args, rest = parser.parse_known_args()
    if "--defaults" not in rest:
        parser.error("Use the selected --defaults policy")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)

    import torch

    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_contract, run_optimized_decoder
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import MinimalPrefillQKV, OptimizedDecoder

    torch.set_num_threads(4)
    runtime = Path(__file__).parents[1] / "tt/optimized_decoder.py"
    runtime_hash, probe_hash = digest(runtime), digest(__file__)
    factory = OptimizedDecoder.from_state_dict.__func__
    metadata = {}

    def build(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        qkv = decoder.layer.self_attn.source.weights.wqkv
        if not isinstance(qkv, MinimalPrefillQKV):
            raise TypeError("Selected minimal QKV backend is required")
        projection = qkv.projection
        metadata["before"] = describe(projection)
        metadata["decode_policy"] = decoder.precision_policy["qkv_decode"]
        metadata["output_policy"] = decoder.precision_policy["prefill_output_projection"]
        old = projection.compute
        if args.minimal_advice == "hifi2":
            projection.compute = ttnn.init_device_compute_kernel_config(
                kw["mesh_device"].arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=old.math_approx_mode,
                fp32_dest_acc_en=old.fp32_dest_acc_en,
                packer_l1_acc=old.packer_l1_acc,
                dst_full_sync_en=old.dst_full_sync_en,
                throttle_level=old.throttle_level,
            )
        elif args.minimal_advice == "grid110":
            available = kw["mesh_device"].compute_with_storage_grid_size()
            if available.x < 11 or available.y < 10:
                raise ValueError("The 11x10 candidate requires 110 available cores")
            projection.programs = tuple(
                ttnn.MinimalMatmulConfig(
                    M_block_size=program.M_block_size,
                    K_block_size=program.K_block_size,
                    N_block_size=program.N_block_size,
                    subblock_h=program.subblock_h,
                    subblock_w=program.subblock_w,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                )
                for program in projection.programs
            )
        metadata["after"] = describe(projection)
        expected_compute = dict(metadata["before"]["compute"])
        if args.minimal_advice == "hifi2":
            expected_compute["math_fidelity"] = str(ttnn.MathFidelity.HiFi2)
        assert metadata["after"]["compute"] == expected_compute
        projection.precision_policy.update(
            fidelity=str(projection.compute.math_fidelity),
            grid=(11, 10) if args.minimal_advice == "grid110" else (11, 8),
            programs={str(index + 1): str(program) for index, program in enumerate(projection.programs)},
        )
        decoder.precision_policy["prefill_qkv_projection"] = projection.precision_policy
        return decoder

    failure = None
    try:
        with (
            patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
            patch.object(torch, "set_num_threads", lambda _: None),
            patch.object(sys, "argv", [sys.argv[0], *rest]),
        ):
            (run_optimized_contract.main if "--contract" in rest else run_optimized_decoder.main)()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report.update(
            minimal_advice_candidate=vars(args),
            minimal_advice_runtime=metadata,
            minimal_advice_runtime_sha256=runtime_hash,
            minimal_advice_probe_sha256=probe_hash,
        )
        if failure:
            report["minimal_advice_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if digest(runtime) != runtime_hash or digest(__file__) != probe_hash:
            raise RuntimeError("Published runtime/probe changed during this control")


if __name__ == "__main__":
    main()
