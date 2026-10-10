# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 padding isolation, matched timing and changing-input layer stress."""

import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_padding import CASES, LAYERS, validate_report
from models.demos.qwen38_27b_qb2.tests.test_compact_gdn_epilogue import run_case
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue_layer import changing_input_comparison
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.gdn_step.workspace import COMBINED_GDN_POLICY
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


@pytest.mark.skipif(os.getenv("QWEN_EPILOGUE_PADDING") != "1", reason="explicit allocated Galaxy experiment")
def test_gdn_epilogue_padding():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_DISABLE_SFPLOADMACRO")
    )
    path = Path(os.environ["QWEN_EPILOGUE_PADDING_RECEIPT"])
    assert not path.exists()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    precision = load_precision(root / f"config/precision_{COMBINED_GDN_POLICY}_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        layers=[],
        source_sha256=model_source_hashes(root),
        precision=precision,
        checkpoint=str(checkpoint),
        scope="Unused-row poisoning and 4096-step real-weight comparison; no full-model/GPQA promotion",
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for case in CASES:
            report.update(state="component", active_case=case)
            save(path, report)
            row = run_case(mesh, *case, padding_experiment=True)
            report["cases"].append(row)
            print("EPILOGUE_PADDING", case, row["comparison"], flush=True)
            save(path, report)
            gc.collect()
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        setup = layer.allocate_state(batch_size=32)
        del setup
        for batch, padding in LAYERS:
            report.update(state="real_weight", active_batch=batch, input_padding=padding)
            save(path, report)
            result = changing_input_comparison(
                layer,
                mesh,
                batch,
                updates=4096,
                policies=(COMBINED_GDN_POLICY, COMBINED_GDN_POLICY),
                epilogue_padding_modes=("zero", padding),
            )
            report["layers"].append(dict(result, input_padding=padding))
            save(path, report)
            gc.collect()
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
                report.update(validation=validate_report(report), passed=True)
        finally:
            save(path, report)
