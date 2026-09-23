# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real layer0/31 component qualification; timings are not full decode latency."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", choices=("row", "dram"), default="row")
    parser.add_argument("--reader", choices=("original", "coalesced", "pipelined", "pipelined_rows"), default="original")
    parser.add_argument("--buffers", type=int, choices=(2, 3), default=2)
    parser.add_argument("--hoist-pack-config", action="store_true")
    parser.add_argument("--bank-vc", action="store_true")
    parser.add_argument("--wide-subblocks", action="store_true")
    parser.add_argument("--gu-workers", choices=(8, 16), type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import ttnn
    from models.demos.llama31_8b_qb2.tt.megakernel.tuning import ProjectionTuning
    from models.demos.llama31_8b_qb2.tt.generator_vllm import LlamaForCausalLM
    from models.demos.llama31_8b_qb2.tests.test_megakernel_mlp import test_mlp_stages_real_weights
    from models.demos.utils.trace_region_sizes import build_trace_device_params

    tuning = ProjectionTuning(args.reader, args.wide_subblocks, buffer_count=args.buffers, hoist_pack_config=args.hoist_pack_config, bank_vc=args.bank_vc, projection_placement=args.placement)
    result = {"tuning": asdict(tuning), "gu_workers": args.gu_workers,
              "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "scope": "Local gate/up, SiLU/multiply and down; host trace enqueue plus final synchronization, 100 replays per trial"}
    ttnn.set_fabric_config(**LlamaForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384,
                                **build_trace_device_params("llama3.1-8b-qb2-decoder"))
    try:
        result.update(test_mlp_stages_real_weights(mesh, False, args.gu_workers, tuning=tuning, measure=True))
        assert all(check["exact"] for check in result["checks"]), "Tuning changed projection arithmetic"
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
