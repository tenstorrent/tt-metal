# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-shape profiling; never profiles the52-layer model."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).parents[1]


def main(args):
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    result = dict(
        implementation_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "tt").glob("*.py")
        },
        command=[sys.executable, *sys.argv],
        precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
        layers=[0, 3],
        batch=1,
        prompt=128,
        iterations=args.iterations,
        tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        profiler=os.environ.get("TT_METAL_DEVICE_PROFILER"),
        watcher=os.environ.get("TT_METAL_WATCHER"),
    )
    try:
        generator = build_generator(ROOT, mesh, layer_indices=[0, 3])
        from models.autoports.meta_models_muse_glimmer_30b.tests.run_datatype_candidate import runtime_summary

        result["precision_config"] = generator.model.precision_config
        result["runtime_summary"] = runtime_summary(generator)
        result["terminal_policy"] = generator.model.terminal_policy
        generator.generate([42] * 128, 4, stop_on_eos=False)
        for phase in ("prefill", "decode", "sample", "argmax", "token_out", "token_out_no_readback", "reset"):
            generator.generate([42] * 128, 1)
            ttnn.synchronize_device(mesh)
            signpost("PERF_" + phase.upper())
            started = time.perf_counter()
            pending = []
            for _ in range(args.iterations):
                if phase == "prefill":
                    generator._execute(("prefill", 128))
                    generator._execute(("head", 128))
                elif phase == "token_out":
                    generator._execute("decode")
                    generator._execute("sample")
                    pending.append(
                        ttnn.from_device(ttnn.get_device_tensors(generator.tokens)[0], blocking=False, cq_id=0)
                    )
                elif phase == "token_out_no_readback":
                    generator.decode_steps(1)
                else:
                    generator._execute(phase)
            ttnn.synchronize_device(mesh)
            for value in pending:
                ttnn.to_torch(value)
            result[phase + "_wall_ms"] = (time.perf_counter() - started) * 1000 / args.iterations
            signpost("PERF_" + phase.upper() + "_END")
            print("PROFILE_PHASE", phase, result[phase + "_wall_ms"], flush=True)
        result["counters"] = dict(generator.counters)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if generator:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
