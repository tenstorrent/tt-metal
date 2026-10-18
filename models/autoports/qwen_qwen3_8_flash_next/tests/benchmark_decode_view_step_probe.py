# SPDX-License-Identifier: Apache-2.0
"""Forced-token compact-decode tensor contract; exclusive TP4 diagnostic only.

Synthetic finite cache/history fixtures do not represent real long prompts or
model-quality evidence. Eight paired cases cover every requested count/position
boundary, not their Cartesian product. All readbacks are standalone diagnostics;
no device profiler or production phase instrumentation is enabled.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_decode_view_probe as traffic
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM

ROOT = Path(__file__).resolve().parents[1]
BATCH = 32
CASES = [(1, 0), (4, 3), (5, 31), (8, 127), (9, 128), (16, 2047), (-1, 262143), (16, 262143)]


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def public_hashes(model):
    result = traffic.public_state_hashes(model)
    # pooled_cache is RM [1,1,pool_tokens/4,128], unlike raw_key_cache's
    # [physical_pages,1,32,128]. Explicitly exclude the private pooled suffix,
    # including when reusing an older traffic probe's hash helper.
    for index, layer in zip(model.layer_indices, model.layers):
        if layer.kind == "linear_attention":
            continue
        value = ttnn.to_torch(layer.attention.pooled_cache, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh, dim=0))
        result[f"{index}/pooled_cache"] = [
            digest(shard[:, :, : model.public_pool_tokens // 4]) for shard in value.chunk(4, dim=0)
        ]
    return result


def initialize_fixture(g, position):
    """Reset every cache and install request-distinct finite histories/canaries."""
    g.reset()
    for layer in g.model.layers:
        if layer.kind == "linear_attention":
            tensor = layer.attention.conv_bank
            history = torch.zeros(tuple(tensor.shape), dtype=torch.bfloat16)
            for slot in range(g.model.max_requests):
                history[slot].fill_((slot + 1) / 128)
            g._copy(history, tensor)
        if layer.ple is not None:
            cache = layer.ple_cache
            history = torch.zeros(tuple(cache.token_history.shape), dtype=torch.int32)
            for slot in range(g.model.max_requests):
                history[slot, :3] = torch.tensor([400 + slot, 800 + slot, position])
            g._copy(history, cache.token_history)
            convolution = torch.zeros(tuple(cache.conv_history.shape), dtype=torch.bfloat16)
            for slot in range(g.model.max_requests):
                convolution[slot * 9 : (slot + 1) * 9].fill_((slot + 1) / 256)
            g._copy(convolution, cache.conv_history)
    logits = torch.empty(tuple(g.tt_logits.shape), dtype=torch.bfloat16)
    for row in range(BATCH):
        logits[:, :, row, :].fill_(-7 + row / 32)
    g._copy(logits, g.tt_logits)
    return logits.reshape(BATCH, -1)


def inputs(model, count, position):
    active = [i in (0, 3, 7, 15) for i in range(BATCH)] if count < 0 else [i < count for i in range(BATCH)]
    slots = [(row * 7 + 5) % BATCH for row in range(BATCH)]
    logical_pages = model.context // 32
    assert model.public_pool_tokens // 32 >= logical_pages + BATCH
    # Common historical pages are read-only finite zero fixtures. Every row's
    # current write page is distinct, outside the shared historical page set.
    pages = ((torch.arange(logical_pages, dtype=torch.int32) * 37 + 11) % logical_pages).repeat(BATCH, 1)
    pages[:, position // 32] = logical_pages + torch.arange(BATCH, dtype=torch.int32)
    assert len(set(pages[:, position // 32].tolist())) == BATCH
    tokens = torch.tensor([[1000 + row * 17] for row in range(BATCH)], dtype=torch.int32)
    positions = torch.full((BATCH,), position, dtype=torch.int32)
    return active, slots, pages, tokens, positions


def execute(g, count, position, compact):
    before_canary = initialize_fixture(g, position)
    active, slots, pages, tokens, positions = inputs(g.model, count, position)
    g.compact_decode_enabled = compact
    # The same captured decode graph is invoked, followed only by the existing
    # explicit test/host logits boundary. No sampler or token feedback executes.
    g.sampling_mode = "host"
    before = dict(g.counters)
    logits = g.decode_forward(
        tokens,
        positions,
        page_table=pages,
        kv_cache=g.kv_cache,
        slots=slots,
        active_rows=active,
        read_tokens=True,
    )
    assert logits.shape == (BATCH, g.model.cfg.vocab_size)
    assert torch.isfinite(logits).all()
    expected = traffic.bucket(active) if compact else BATCH
    assert g.last_decode_execution_batch == expected
    assert g.counters["decode_replays"] - before.get("decode_replays", 0) == 1
    ttnn.synchronize_device(g.mesh)
    state = public_hashes(g.model)
    inactive = torch.tensor([not flag for flag in active])
    shards = list(logits.chunk(4, dim=-1))
    # B32 fallback deliberately overwrites inactive logits. Preservation is a
    # compact-bridge contract only, and is checked across all four vocab shards.
    canary_checked = compact and expected < BATCH
    if canary_checked:
        for shard in shards:
            assert torch.equal(shard[inactive], before_canary[inactive]), "Inactive owner logits changed"
    device_positions = ttnn.to_torch(ttnn.get_device_tensors(g.tt_pos)[0])[:, 0].tolist()
    assert device_positions == [position + 1 if flag else 0 for flag in active]
    metadata = {
        "compact": compact,
        "execution_batch": expected,
        "active": active,
        "physical_slots": slots,
        "position": position,
        "current_public_pages": pages[:, position // 32].tolist(),
        "forced_tokens": tokens.flatten().tolist(),
        "inactive_logit_canary_checked": canary_checked,
        "active_logits_sha256_per_asic": [digest(shard[active]) for shard in shards],
        "public_state_sha256_per_asic": state,
        "device_positions": device_positions,
        "counters": {key: g.counters[key] - before.get(key, 0) for key in g.counters},
    }
    return logits, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-region-size", type=int, default=400000000)
    args = parser.parse_args()
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    assert ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0")
    assert os.environ.get("TT_METAL_WATCHER", "0") in ("", "0")
    args.output.mkdir(parents=True, exist_ok=False)
    traffic.base.BATCH = BATCH
    os.environ["QWEN_FLASH_SERVING_LAYERS"] = "0,1,3"
    os.environ["QWEN_FLASH_CHECKPOINT"] = args.checkpoint
    os.environ["QWEN_FLASH_TRACE_EVENTS"] = str(args.output / "events.jsonl")
    report = {
        "status": "started",
        "scope": "Synthetic forced-token tensor/state contract; real layers0/1/3; not text quality or serving performance",
        "case_selection": "Eight paired boundaries, not a Cartesian sweep",
        "synthetic_context": "Zero KV/recurrent context; request-distinct GDN conv/PLE histories; PLE next_position consistent",
        "tracking": {"native": True, "python": True, "skip_program_cache": False, "exemptions": []},
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(traffic.__file__),
                ROOT / "tt/generator.py",
                ROOT / "tt/decode_view.py",
                ROOT / "tt/generator_vllm.py",
                ROOT / "tt/multichip_decoder.py",
            )
        },
        "cases": [],
    }
    (args.output / "probe_source.py.txt").write_bytes(Path(__file__).read_bytes())

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    mesh = adapter = None
    save()
    try:
        torch.set_num_threads(8)
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, BATCH, traffic.base.CONTEXT)
        blocks = traffic.base.CONTEXT // 32 + BATCH
        cache = adapter.allocate_kv_cache((blocks, 1, 32, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        g = adapter.generator
        assert g.model.context == 262144 and set(g.decode_views) == {4, 8, 16}
        traces, captures = dict(g.traces), g.counters["captures"]
        report.update(
            device_ids=list(mesh.get_device_ids()), context=g.model.context, precision=g.model.precision_policy
        )
        for case_index, (count, position) in enumerate(CASES):
            control, control_meta = execute(g, count, position, False)
            candidate, candidate_meta = execute(g, count, position, True)
            active = control_meta["active"]
            equal = torch.equal(control[active], candidate[active])
            state_equal = control_meta["public_state_sha256_per_asic"] == candidate_meta["public_state_sha256_per_asic"]
            row = {
                "count": count,
                "position": position,
                "control": control_meta,
                "candidate": candidate_meta,
                "active_logits_exact": equal,
                "public_state_exact": state_equal,
                "max_abs_active_logit_delta": float((control[active].float() - candidate[active].float()).abs().max()),
            }
            report["cases"].append(row)
            # Bound disk use: retain complete active vectors for only first/last
            # boundary, or any failure; every case retains all shard/state hashes.
            if case_index in (0, len(CASES) - 1) or not equal or not state_equal:
                raw = args.output / f"case_{case_index:02d}_active_logits.pt"
                torch.save({"control": control[active], "candidate": candidate[active]}, raw)
                row["raw_logits"] = {"path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
            save()
            assert equal and state_equal, ("Compact decode first-step divergence", case_index)
            assert g.traces == traces and g.counters["captures"] == captures
            del control, candidate
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["closed_mesh"] = mesh is not None
        save()


if __name__ == "__main__":
    main()
