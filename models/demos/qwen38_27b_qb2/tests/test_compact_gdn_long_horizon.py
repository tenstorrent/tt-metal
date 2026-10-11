# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""4K changing-input compact GDN comparison against the qualified BFP8 path.

This is an exact boundary comparison, not an independent dense reference or
an evaluation score. Both sessions own private state; real per-rank weights,
convolution history, recurrent state and the projected outputs are exercised.
"""

import gc
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.compact_gdn import policy_pair, validate_long_horizon
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue_layer import changing_input_comparison
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


@pytest.mark.skipif(os.getenv("QWEN_COMPACT_LONG_HORIZON") != "1", reason="explicit allocated Galaxy 4K comparison")
def test_compact_gdn_long_horizon():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_DISABLE_SFPLOADMACRO")
    )
    path = Path(os.environ["QWEN_COMPACT_LONG_HORIZON_RECEIPT"])
    assert not path.exists(), "Preserve every long-horizon attempt"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    mode = os.getenv("QWEN_COMPACT_COMBINED", "0")
    assert mode in ("0", "1", "2"), "Unknown combined GDN experiment mode"
    baseline, candidate = policy_pair(combined=mode == "1", padding=mode == "2")
    precision = load_precision(source / f"config/precision_{candidate}_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        baseline=baseline,
        candidate=candidate,
        cases=[],
        source_sha256=model_source_hashes(source),
        precision=precision,
        checkpoint=str(checkpoint),
        independent_dense_reference=False,
        full_model_qualified=False,
        promoted_to_serving=False,
        started_at=time.time(),
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(report["device_ids"]) == 4
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        # Candidate scratch and both geometries predate every captured trace.
        setup = layer.allocate_state(batch_size=32)
        del setup
        gc.collect()
        for batch in (16, 32):
            report.update(state="changing_inputs", active_batch=batch)
            save(path, report)
            started = time.monotonic()
            case = changing_input_comparison(layer, mesh, batch, updates=4096, policies=(baseline, candidate))
            case["test_duration_s"] = time.monotonic() - started
            report["cases"].append(case)
            save(path, report)
        report.update(state="completed")
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                if parent is not None:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = parent is not None
            if report["state"] == "completed":
                report.update(
                    validation=validate_long_horizon(report, baseline=baseline, candidate=candidate), passed=True
                )
        finally:
            report["finished_at"] = time.time()
            save(path, report)
