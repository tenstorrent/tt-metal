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
from models.demos.qwen38_27b_qb2.tests.compact_gdn import changing_input_checkpoints
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import BATCHES, CANDIDATES, POLICIES, VARIANTS, compare
from models.demos.qwen38_27b_qb2.tests.test_gdn_layer_integration import capture, host_ranks, run_case
from models.demos.qwen38_27b_qb2.tests.test_gdn_model_adapter import tensor_digest
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def changing_input_comparison(
    layer,
    mesh,
    batch,
    *,
    updates=64,
    policies=("single_step_flat_prepare_epilogue", "single_step_compact_gdn"),
):
    """Two independent sessions, changing tokens, same real BFP8 weights.

    Both traces own separate recurrent/history allocations created before
    either trace captures scratch addresses. Compare each user
    on every rank after power-of-two updates, including actual projected
    output. A stationary-input recurrence check alone would miss history and
    stale-input bugs in the new convolution/packed-gate boundary.
    """
    checkpoints = changing_input_checkpoints(updates)
    rng = torch.Generator().manual_seed(341200 + batch)
    sources = [
        ttnn.from_torch(
            torch.randn(1, 1, batch, 5120, generator=rng).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(4)
    ]
    slots, traces = [], []
    original_recurrence = layer.policy["decode_recurrence"]
    try:
        for recurrence in policies:
            layer.policy["decode_recurrence"] = recurrence
            state = layer.allocate_state(batch_size=batch)
            x = ttnn.clone(sources[0])
            zeros = tuple(ttnn.zeros_like(value) for value in (state.recurrent, state.conv))
            slots.append(dict(recurrence=recurrence, x=x, state=state, zeros=zeros))
        # Allocating session B after capturing A lets B's persistent inputs or
        # state reuse A's transient scratch addresses. Warm both complete
        # graphs and their reset copies before either capture, just as serving
        # must warm prefill and decode before reserving trace scratch.
        for slot in slots:
            layer.policy["decode_recurrence"] = slot["recurrence"]
            x, state = slot["x"], slot["state"]
            layer._delta(x, state, decode=True)
            for zero, value in zip(slot["zeros"], (state.recurrent, state.conv)):
                ttnn.copy(zero, value)
        ttnn.synchronize_device(mesh)
        for slot in slots:
            layer.policy["decode_recurrence"] = slot["recurrence"]
            x, state = slot["x"], slot["state"]
            trace, output = capture(mesh, lambda: layer._delta(x, state, decode=True))
            traces.append(trace)
            slot["output"] = output
        for slot in slots:
            state = slot["state"]
            for zero, value in zip(slot["zeros"], (state.recurrent, state.conv)):
                ttnn.copy(zero, value)
        checks = []
        for step in range(updates):
            source = sources[(step * 7 + step // 3) % len(sources)]
            observed = []
            for slot, trace in zip(slots, traces):
                x, state, output = slot["x"], slot["state"], slot["output"]
                ttnn.copy(source, x)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if step + 1 in checkpoints:
                    # Read before the other trace can reuse shared projection
                    # scratch or the all-reduce destination.
                    hashes = []
                    for tensor in (state.recurrent, state.conv, output):
                        ranks = host_ranks(tensor)
                        assert len(ranks) == 4 and all(torch.isfinite(value).all() for value in ranks)
                        hashes.append([tensor_digest(value.reshape(batch, -1)) for value in ranks])
                    observed.append(hashes)
            if observed:
                assert (
                    observed[0] == observed[1]
                ), f"Changing-input compact GDN diverged at batch={batch}, step={step+1}"
                checks.append(dict(step=step + 1, recurrent_conv_projected_sha256=observed[0], all_values_finite=True))
        return dict(
            batch=batch,
            updates=updates,
            checkpoints=checks,
            all_ranks_bit_identical=True,
            persistent_sessions_precede_trace_capture=True,
        )
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        layer.policy["decode_recurrence"] = original_recurrence


@pytest.mark.skipif(os.getenv("QWEN_GDN_EPILOGUE_LAYER") != "1", reason="explicit allocated-Galaxy experiment")
def test_gdn_epilogue_layer():
    path = Path(os.environ["QWEN_GDN_LAYER_RECEIPT"])
    assert not path.exists(), "Preserve each real-weight attempt"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    candidate = CANDIDATES[os.getenv("QWEN_GDN_LAYER_CANDIDATE", "epilogue")]
    policies = dict(POLICIES, fused=candidate)
    if candidate == "single_step_compact_gdn":
        # Attribute incremental compact-path savings against today's fusion,
        # without counting direct preparation/epilogue gains a second time.
        policies["native"] = "single_step_flat_prepare_epilogue"
    precision = load_precision(source / f"config/precision_{candidate}_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_serving=False,
        checkpoint=str(checkpoint),
        precision=precision,
        candidate_recurrence=candidate,
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
        for batch in (16, 32, 8, 1) if candidate == "single_step_compact_gdn" else BATCHES:
            group = []
            for variant in VARIANTS:
                report.update(state="real_weight_comparison", active_batch=batch, active_variant=variant)
                save(path, report)
                case = run_case(layer, mesh, batch, recurrence=policies[variant])
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
        if candidate == "single_step_compact_gdn":
            report["changing_input_checks"] = []
            for batch in (16, 32):
                report.update(state="changing_inputs", active_batch=batch)
                save(path, report)
                report["changing_input_checks"].append(changing_input_comparison(layer, mesh, batch))
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
