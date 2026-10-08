# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight shared-Q/K experiment at the complete GDN block boundary."""

import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.tests.gdn_shared_qk import compare
from models.demos.qwen38_27b_qb2.tests.test_gdn_layer_integration import run_case
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt import decoder as decoder_module
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def run_shared_case(layer, mesh, batch, shared):
    original = decoder_module.step_from_flat
    if shared:
        scratch = tuple(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([batch * 4, 128]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            for _ in range(2)
        )
        addresses = [tensor.buffer_address() for tensor in scratch]

        def prepared(*args, **kwargs):
            assert args[0].shape[0] == batch
            return original(*args, **kwargs, shared_qk_outputs=scratch)

        # This experiment owns its isolated process. The model's policy and
        # production workspace remain unchanged; preparation is timed inside
        # the real block, with scratch allocated before trace capture.
        decoder_module.step_from_flat = prepared
    try:
        result = run_case(layer, mesh, batch)
        if shared:
            assert addresses == [tensor.buffer_address() for tensor in scratch]
    finally:
        decoder_module.step_from_flat = original
    result.update(shared_qk=shared, traced_call_us=result["candidate"]["samples_us"])
    return result


@pytest.mark.skipif(os.getenv("QWEN_GDN_SHARED_QK_LAYER") != "1", reason="explicit allocated-Galaxy experiment")
def test_gdn_shared_qk_layer():
    path = Path(os.environ["QWEN_GDN_LAYER_RECEIPT"])
    assert not path.exists(), "Preserve each real-weight attempt"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    precision = load_precision(source / "config/precision_single_step_gdn.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        checkpoint=str(checkpoint),
        precision=precision,
        source_sha256=model_source_hashes(source),
        scope="Layer 0 GDN projection, convolution, gates, shared normalization, recurrence, output norm and projection; MLP excluded",
        reference="FP32 recurrence on actual per-rank convolution outputs, 64 updates and bracketed fused-normalization controls",
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
        for batch in (32, 16):
            group = []
            for shared in (False, True, False):
                report.update(state="real_weight_comparison", active_batch=batch, active_shared_qk=shared)
                save(path, report)
                case = run_shared_case(layer, mesh, batch, shared)
                group.append(case)
                report["cases"].append(case)
                save(path, report)
                gc.collect()
            comparison = compare(group)
            hashes = group[0]["projected_output_sha256_per_rank"]
            assert len(hashes) == 4 and all(case["projected_output_sha256_per_rank"] == hashes for case in group)
            comparison.update(
                scope=report["scope"],
                projected_output_bit_identical=True,
                real_weights=True,
            )
            report["comparisons"].append(comparison)
            print("SHARED_QK_REAL_WEIGHT", comparison, flush=True)
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
