# SPDX-License-Identifier: Apache-2.0
"""Isolated real-weight GDN q/k ROW_MAJOR reader versus current tiled reader.

S128/256/512 and 4K continuation, public-state/owner32 handoff, tracked traces.
"""

import argparse
import gc
import hashlib
import os
import statistics
import subprocess
import time
import traceback
import types
from pathlib import Path

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_qk_rm_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_view_probe as common
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import apply_layer, for_layer, load_policy
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_view import _decoder


def conversion_audit(target, row):
    seen = set()

    def audit(model, q, k, v, g, beta):
        n = q.shape[1]
        if n not in seen:
            seen.add(n)
            record = dict(sequence=n, operands={})
            raw = {}
            for name, value in (("q", q), ("k", k)):
                tiled = common.read_shards(value)
                rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=value.memory_config())
                assert rm.dtype == value.dtype == ttnn.float32
                plain = common.read_shards(rm)
                poisoned = ttnn.fill_implicit_tile_padding(value, float("nan"))
                rm_poison = ttnn.to_layout(poisoned, ttnn.ROW_MAJOR_LAYOUT, memory_config=value.memory_config())
                poison = common.read_shards(rm_poison)
                for values in (plain, poison):
                    assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(tiled, values))
                record["operands"][name] = dict(
                    shape=list(value.shape),
                    tiled_padded=list(value.padded_shape),
                    rm_padded=list(rm.padded_shape),
                    dtype=str(value.dtype),
                    rm_dtype=str(rm.dtype),
                    tiled_memory=str(value.memory_config()),
                    rm_memory=str(rm.memory_config()),
                    exact_bits=True,
                    nan_padding_exact_bits=True,
                )
                raw[name] = dict(tiled=tiled, rm=plain, rm_nan_padding=poison)
            path = target / f"normalized_qk_{n}.pt"
            torch.save(raw, path)
            record["raw"] = dict(path=str(path), sha256=common.sha(path))
            row.setdefault("conversion_audit", []).append(record)
        return candidate.decoder._local_recurrence(model, q, k, v, g, beta)

    return candidate.make_gdn_recurrence(audit)


class Step:
    def __init__(self, layer, size, hidden, tokens):
        self.layer, self.size, self.hidden, self.tokens = layer, size, hidden, tokens
        ops = layer.ops
        self.x = ops.shards([torch.zeros(1, 1, size, 2560, dtype=torch.bfloat16)] * 4)
        self.ids = ops.upload(torch.zeros(1, size, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.pos = ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.slots = ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.page = ops.upload(common.page_table(1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.output = ttnn.empty((1, 1, size, 640), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=layer.mesh)

    def refresh(self, start):
        mesh, n = self.layer.mesh, self.size
        common.host_copy(self.hidden[:, :, start : start + n], self.x, mesh, sharded=True)
        common.host_copy(self.tokens[start : start + n].reshape(1, n), self.ids, mesh)
        pos, slots = torch.zeros(1, 32, dtype=torch.int32), torch.zeros(1, 32, dtype=torch.int32)
        pos[0, 0], slots[0, 0] = start, common.SLOT
        common.host_copy(pos, self.pos, mesh)
        common.host_copy(slots, self.slots, mesh)
        # Admit only pages required by this chunk; future pages remain private.
        pages = common.page_table(1)
        needed = (start + n + 31) // 32
        pages[0, needed:] = common.SCRATCH_BASE + torch.arange(common.CONTEXT // 32 - needed) % 16
        common.host_copy(pages, self.page, mesh)

    def forward(self):
        ops = self.layer.ops
        old = ops.large_prefill, ops.l1_prefill
        ops.large_prefill, ops.l1_prefill = self.size >= 256, True
        try:
            out = self.layer.attention.prefill_forward(self.x, current_positions=self.pos, slots=self.slots)
            ttnn.copy(out, self.output)
        finally:
            ops.large_prefill, ops.l1_prefill = old


def run_layer(args, cfg, policy, index, report, save):
    target = args.output / f"layer{index}"
    target.mkdir()
    row = dict(layer=index, cases=[])
    report["layers"].append(row)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region)
    trace = None
    try:
        owner = MultichipDecoder.from_state_dict(
            CheckpointState(args.checkpoint),
            hf_config=cfg,
            layer_idx=index,
            mesh_device=mesh,
            batch_size=32,
            max_requests=64,
            chunk_size=128,
            attention_pool_tokens=common.PUBLIC_TOKENS + 32 * 512,
            cache_dir=args.weight_cache,
        )
        apply_layer(owner, for_layer(policy, index))
        owner.prepare([1])
        views = {name: _decoder(owner, {}, [1, 128, 256, 512]) for name in ("tiled", "rm")}
        assert owner.kind == "linear_attention" and owner.attention.recurrence_broadcast
        row["actual_precision"] = dict(
            packed_weight=str(owner.attention.packed_weight.dtype),
            output_weight=str(owner.attention.output_projection.row_weight.dtype),
            recurrent_bank=str(owner.attention.recurrent_bank.dtype),
            conv_bank=str(owner.attention.conv_bank.dtype),
            compute={
                key: str(getattr(owner.attention.compute, key))
                for key in (
                    "math_fidelity",
                    "math_approx_mode",
                    "fp32_dest_acc_en",
                    "packer_l1_acc",
                    "dst_full_sync_en",
                )
            },
            shared_weights=all(
                view.attention.packed_weight is owner.attention.packed_weight
                and view.attention.output_projection.row_weight is owner.attention.output_projection.row_weight
                for view in views.values()
            ),
            shared_public_state=all(
                view.attention.recurrent_bank is owner.attention.recurrent_bank
                and view.attention.conv_bank is owner.attention.conv_bank
                for view in views.values()
            ),
        )
        views["rm"].attention._recurrence = types.MethodType(candidate.GDN_RECURRENCE, views["rm"].attention)
        views["tiled"].attention._recurrence = types.MethodType(conversion_audit(target, row), views["tiled"].attention)
        rng = torch.Generator().manual_seed(49256 + index)
        hidden = (torch.randn(4, 1, 4097, 2560, generator=rng) * 0.05).bfloat16()
        tokens = torch.randint(0, cfg.vocab_size, (4097,), generator=rng).int()
        torch.save(dict(hidden_shards=hidden, tokens=tokens), target / "inputs.pt")
        row["input_sha256"] = common.sha(target / "inputs.pt")
        steps = {(group, n): Step(view, n, hidden, tokens) for group, view in views.items() for n in (128, 256, 512)}
        # Every owner-handoff and shape is warmed before any capture.
        decodes = {}
        for length, slot in zip((128, 256, 512, 4096), (7, 19, 23, 31)):
            common.SLOT = slot
            decodes[length] = common.Decode(owner, hidden, tokens, position=length)
            for group, view in views.items():
                common.reset(owner, view)
                step = steps[group, min(length, 512)]
                for start in range(0, length, step.size):
                    step.refresh(start)
                    step.forward()
                decodes[length].forward()
                common.read_shards(decodes[length].output)
        row["memory_after_warmup"] = common.memory_view(mesh)
        # The audit ran only before capture; timed graphs contain no readbacks.
        views["tiled"].attention._recurrence = types.MethodType(
            candidate.decoder.ParallelGatedDeltaNet._recurrence, views["tiled"].attention
        )
        for length, slot in zip((128, 256, 512, 4096), (7, 19, 23, 31)):
            common.SLOT = slot
            case = dict(length=length, slot=slot, outer_chunk=min(length, 512), variants={})
            row["cases"].append(case)
            raw = {}
            for group, view in views.items():
                step = steps[group, min(length, 512)]
                common.reset(owner, view)
                step.refresh(0)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                step.forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                timings, execution_timings, outputs = [], [], []
                for _ in range(args.repeats):
                    common.reset(owner, view)
                    outputs = []
                    execution_seconds = 0.0
                    tick = time.perf_counter()
                    for start in range(0, length, step.size):
                        step.refresh(start)
                        ttnn.synchronize_device(mesh)
                        execute_start = time.perf_counter()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        execution_seconds += time.perf_counter() - execute_start
                        outputs.append(common.read_shards(step.output))
                    timings.append(time.perf_counter() - tick)
                    execution_timings.append(execution_seconds)
                ttnn.release_trace(mesh, trace)
                trace = None
                state, checks = common.public_state(owner)
                decodes[length].forward()
                handoff = common.read_shards(decodes[length].output)
                handoff_state, handoff_checks = common.public_state(owner)
                raw[group] = dict(
                    output=[torch.cat([o[chip] for o in outputs], dim=2) for chip in range(4)],
                    state=state,
                    handoff=handoff,
                    handoff_state=handoff_state,
                )
                case["variants"][str(group)] = dict(
                    seconds=timings,
                    execution_seconds=execution_timings,
                    median_execution_seconds=statistics.median(execution_timings),
                    median_seconds=statistics.median(timings),
                    state_checks=checks,
                    handoff_checks=handoff_checks,
                    trace_released=True,
                )
                save()
            compare = dict(
                output=common.compare(raw["tiled"]["output"], raw["rm"]["output"]),
                handoff=common.compare(raw["tiled"]["handoff"], raw["rm"]["handoff"]),
                state={
                    name: common.compare(value, raw["rm"]["state"][name])
                    for name, value in raw["tiled"]["state"].items()
                },
                handoff_state={
                    name: common.compare(value, raw["rm"]["handoff_state"][name])
                    for name, value in raw["tiled"]["handoff_state"].items()
                },
            )
            path = target / f"length{length}.pt"
            torch.save(raw, path)
            case.update(
                comparisons=compare,
                raw=dict(path=str(path), sha256=common.sha(path)),
                median_speedup=case["variants"]["tiled"]["median_seconds"] / case["variants"]["rm"]["median_seconds"],
            )
            save()

            def require_exact(value):
                if isinstance(value, dict):
                    if "finite" in value:
                        assert value["finite"] and value.get("exact_equal") is True, value
                    else:
                        for child in value.values():
                            require_exact(child)
                elif isinstance(value, list):
                    for child in value:
                        require_exact(child)

            require_exact(compare)
        row["status"] = "measurements_complete_review_numerics"
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        row["mesh_closed"] = True
        save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(f"/home/vkovacevic/qwen38-flash/hf-local/{common.REVISION}")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-region", type=int, default=768000000)
    parser.add_argument(
        "--weight-cache", default="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"
    )
    args = parser.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    assert ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    assert all(os.environ.get(n, "0") in ("", "0") for n in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER"))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = Qwen4ExpConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    policy = load_policy(common.ROOT / "doc/datatype_sweep/selected_precision_config.json")
    assert cfg.max_position_embeddings == common.CONTEXT and policy["config_id"] == "combined_head4_gdn4_lofi"
    common.ACTIVE_PAGES = [256 + (7 * i + 11) % 256 for i in range(129)]
    sources = [
        Path(__file__),
        Path(common.__file__),
        Path(candidate.__file__),
        *sorted((common.ROOT / "tt").glob("*.py")),
    ]
    report = dict(
        status="running",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources={str(p): common.sha(p) for p in sources},
        policy=policy,
        trace_tracking=dict(native=True, python=True, skip_program_cache=False, exemptions=[]),
        candidate_reader_sha256=hashlib.sha256(candidate.READER.encode()).hexdigest(),
        reader_geometry=dict(
            qk_tile_reads_per_worker_token=8,
            qk_rm_reads_per_worker_token=2,
            qk_tile_bytes_per_worker_token=32768,
            qk_rm_bytes_per_worker_token=1024,
            existing_scratch_bytes=4096,
            maximum_rm_read_bytes=512,
        ),
        context=common.CONTEXT,
        active_pages=common.ACTIVE_PAGES,
        sentinel_pages=common.SENTINEL_PAGES,
        layers=[],
        timing="execution_seconds sums warmed whole-GDN trace submission through standalone synchronization per chunk, including q/k conversion; seconds additionally includes host refresh/readback; no serving roofline claim",
        scope="real selected-weight GDN outputs/publicstate and owner32decode; no full-model logits",
    )

    def save():
        common.dump(args.output / "summary.json", report)

    (args.output / "candidate_reader.cpp").write_text(candidate.READER)
    (args.output / "baseline_reader.cpp").write_text(candidate.decoder._BROADCAST_PREFILL_READER)
    for path in sources:
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    save()
    try:
        for index in map(int, args.layers.split(",")):
            assert index == 0
            run_layer(args, cfg, policy, index, report, save)
        report["status"] = "measurements_complete_review_numerics"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
