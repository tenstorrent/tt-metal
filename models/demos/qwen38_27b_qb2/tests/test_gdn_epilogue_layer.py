# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Complete real-weight BFP8 GDN block with the opt-in epilogue policy."""

import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import BATCHES, POLICIES, VARIANTS, compare
from models.demos.qwen38_27b_qb2.tests.test_gdn_layer_integration import run_case
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


@pytest.mark.skipif(os.getenv("QWEN_GDN_EPILOGUE_LAYER") != "1", reason="explicit allocated-Galaxy experiment")
def test_gdn_epilogue_layer():
    path = Path(os.environ["QWEN_GDN_LAYER_RECEIPT"])
    assert not path.exists(), "Preserve each real-weight attempt"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    precision = load_precision(source / "config/precision_single_step_shared_qk_epilogue_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_serving=False,
        checkpoint=str(checkpoint),
        precision=precision,
        source_sha256=model_source_hashes(source),
        reference="64 FP32 recurrence updates on actual per-rank convolution outputs plus bit-identical projected controls",
        cases=[],
        comparisons=[],
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(report["device_ids"]) == 4
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        # Allocate the candidate's persistent scratch before selecting controls.
        # Each case still owns fresh, identical recurrent and convolution state.
        setup_state = layer.allocate_state(batch_size=32)
        del setup_state
        gc.collect()
        addresses = {b: layer.gdn_decode_workspace.epilogue_output(b).buffer_address() for b in (16, 32)}
        for batch in BATCHES:
            group = []
            for variant in VARIANTS:
                report.update(state="real_weight_comparison", active_batch=batch, active_variant=variant)
                save(path, report)
                case = run_case(layer, mesh, batch, recurrence=POLICIES[variant])
                case.update(variant=variant, traced_call_us=case["candidate"]["samples_us"])
                group.append(case)
                report["cases"].append(case)
                save(path, report)
                gc.collect()
            comparison = compare(group)
            assert all(
                layer.gdn_decode_workspace.epilogue_output(b).buffer_address() == a for b, a in addresses.items()
            )
            report["comparisons"].append(comparison)
            print("EPILOGUE_REAL_WEIGHT", comparison, flush=True)
            save(path, report)
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        except BaseException as error:
            report.update(state="failed", passed=False, cleanup_error=str(error)[:2000])
            raise
        finally:
            save(path, report)
