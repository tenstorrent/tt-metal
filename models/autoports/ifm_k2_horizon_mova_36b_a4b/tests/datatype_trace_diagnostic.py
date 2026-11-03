# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Detect an unwarmed decode operation outside capture, so errors unwind safely."""

import json
import traceback
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    gen = None
    report = {}
    try:
        gen = build_generator(
            ROOT, mesh, override_num_layers=2, precision_config=DOC / "configs/expert_activations_bfp4.json"
        )
        gen.prepare()
        gen.refresh_decode_inputs(torch.zeros(1, 1), torch.zeros(1))
        mesh.set_program_cache_misses_allowed(False)
        try:
            gen._decode()
            ttnn.synchronize_device(mesh)
            report["status"] = "no_cache_miss"
        except Exception:
            report["status"] = "cache_miss"
            report["error"] = traceback.format_exc()
            print(report["error"], flush=True)
        finally:
            mesh.set_program_cache_misses_allowed(True)
        (DOC / "debug/trace_capture/cache_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
