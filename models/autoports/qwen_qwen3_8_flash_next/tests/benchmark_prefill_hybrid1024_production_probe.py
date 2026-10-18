# SPDX-License-Identifier: Apache-2.0
"""Actual production exact1024 helper versus unchanged512 fallback, layers0/1/3.

Adjacent lengths,4K composition, unchanged public state and original owner32 handoff.
"""

import argparse
import gc
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_view_probe as common
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import apply_layer, for_layer, load_policy
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_view import _decoder

LENGTHS = (127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096)
SLOTS = (7, 19, 23, 31, 5, 12, 28, 3, 17, 29, 6, 11, 21)
PREPARED = (1, 127, 128, 129, 255, 256, 257, 511, 512)
CANDIDATE_PREPARED = (*PREPARED, 513, 1023, 1024)


class Step:
    def __init__(self, layer, size, hidden, tokens):
        self.layer, self.size, self.hidden, self.tokens = layer, size, hidden, tokens
        ops = layer.ops
        self.x = ops.shards([torch.zeros(1, 1, size, 2560, dtype=torch.bfloat16)] * 4)
        self.ids = ops.upload(torch.zeros(1, size, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.pos = ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.slots = ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.page = ops.upload(common.page_table(1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.output = ttnn.empty((1, 1, size, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=layer.mesh)

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
        # Direct non1024 calls prove original fallback;4K explicitly follows
        # serving's four1024 traces versus baseline eight512 traces.
        if self.size != 4096:
            out = self.layer.prefill_forward(
                self.x, current_positions=self.pos, slots=self.slots, tokens=self.ids, page_table=self.page
            )
        else:
            size = 1024 if self.layer.prefill_hybrid_enabled else 512
            outputs = []
            for start in range(0, self.size, size):
                value = self.layer.prefill_forward(
                    self.x[:, :, start : start + size, :],
                    current_positions=self.pos if start == 0 else ttnn.add(self.pos, start),
                    slots=self.slots,
                    tokens=self.ids[:, start : start + size],
                    page_table=self.page,
                )
                outputs.append(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG))
            out = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(out, self.output)


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
        views = {
            "original": _decoder(owner, {}, PREPARED),
            "candidate": _decoder(owner, {}, CANDIDATE_PREPARED),
        }
        assert not getattr(owner.attention, "prefill_qk_rm_enabled", False)
        public = (
            ("recurrent_bank", "conv_bank")
            if owner.kind == "linear_attention"
            else ("packed_cache", "raw_key_cache", "pooled_cache")
        )
        row["shared_public_identity"] = {}
        for name, view in views.items():
            records = {}
            for field in public:
                original, shared = getattr(owner.attention, field), getattr(view.attention, field)
                assert original is shared
                records[field] = dict(
                    owner_address=original.buffer_address(),
                    view_address=shared.buffer_address(),
                    shape=list(shared.shape),
                    dtype=str(shared.dtype),
                    same_object=True,
                )
            assert (
                owner.mlp.gates is view.mlp.gates
                and owner.mlp.ups is view.mlp.ups
                and owner.mlp.downs is view.mlp.downs
            )
            assert owner.mlp.router is view.mlp.router and owner.mlp.routed_compute is view.mlp.routed_compute
            if owner.ple is not None:
                assert owner.ple_cache is view.ple_cache
            row["shared_public_identity"][name] = dict(
                attention=records,
                expert_weights_same_objects=True,
                router_same_object=True,
                routed_compute_same_object=True,
                ple_cache_same_object=owner.ple is None or owner.ple_cache is view.ple_cache,
            )
        rng = torch.Generator().manual_seed(49256 + index)
        hidden = (torch.randn(4, 1, 4097, 2560, generator=rng) * 0.05).bfloat16()
        tokens = torch.randint(0, cfg.vocab_size, (4097,), generator=rng).int()
        torch.save(dict(hidden_shards=hidden, tokens=tokens), target / "inputs.pt")
        row["input_sha256"] = common.sha(target / "inputs.pt")
        steps = {(group, n): Step(view, n, hidden, tokens) for group, view in views.items() for n in LENGTHS}
        # Every owner-handoff and shape is warmed before any capture.
        decodes = {}
        for length, slot in zip(LENGTHS, SLOTS):
            common.SLOT = slot
            decodes[length] = common.Decode(owner, hidden, tokens, position=length)
            for group, view in views.items():
                common.reset(owner, view)
                step = steps[group, length]
                for start in (0,):
                    step.refresh(start)
                    step.forward()
                decodes[length].forward()
                common.read_shards(decodes[length].output)
        row["memory_after_warmup"] = common.memory_view(mesh)
        row["actual_geometry"] = {
            name: dict(
                chunk_size=view.chunk_size,
                hybrid_enabled=view.prefill_hybrid_enabled,
                hybrid_signatures=view.prefill_hybrid_signatures,
                norm_compaction=view.ops.norm_compaction,
                gdn_qk_rm=getattr(view.attention, "prefill_qk_rm_signatures", {}),
                qsa_chunk_offsets=list(view.attention.chunk_offsets.shape)
                if hasattr(view.attention, "chunk_offsets")
                else None,
                qsa_indexer_weights=list(view.attention.indexer_weights.shape)
                if hasattr(view.attention, "indexer_weights")
                else None,
                ple_workspaces=[list(key) for key in view.ple.workspaces] if view.ple is not None else None,
                owner_qk_rm_enabled=getattr(owner.attention, "prefill_qk_rm_enabled", False),
            )
            for name, view in views.items()
        }
        assert all(v.mlp.prefill_group_size == 512 for v in views.values())
        for length, slot in zip(LENGTHS, SLOTS):
            common.SLOT = slot
            case = dict(
                length=length,
                slot=slot,
                execution={
                    name: dict(
                        outer_calls=([1024] * 4 if view.prefill_hybrid_enabled else [512] * 8)
                        if length == 4096
                        else [length],
                        non_moe_chunks=[min(512, length - start) for start in range(0, length, 512)],
                        hybrid_enabled=view.prefill_hybrid_enabled,
                    )
                    for name, view in views.items()
                },
                variants={},
            )
            row["cases"].append(case)
            raw = {}
            for group, view in views.items():
                step = steps[group, length]
                common.reset(owner, view)
                step.refresh(0)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                step.forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                captured_memory = common.memory_view(mesh)
                timings, outputs = [], []
                for _ in range(args.repeats):
                    common.reset(owner, view)
                    outputs = []
                    tick = time.perf_counter()
                    for start in (0,):
                        step.refresh(start)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        outputs.append(common.read_shards(step.output))
                    timings.append(time.perf_counter() - tick)
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
                    median_seconds=statistics.median(timings),
                    state_checks=checks,
                    handoff_checks=handoff_checks,
                    trace_released=True,
                    captured_memory_snapshot_not_peak=captured_memory,
                )
                save()
            # Retain both raw variants before any numerical comparison can fail.
            path = target / f"length{length}.pt"
            torch.save(raw, path)
            case["raw"] = dict(path=str(path), sha256=common.sha(path))
            save()
            compare = dict(
                output=common.compare(raw["original"]["output"], raw["candidate"]["output"]),
                handoff=common.compare(raw["original"]["handoff"], raw["candidate"]["handoff"]),
                state={
                    name: common.compare(value, raw["candidate"]["state"][name])
                    for name, value in raw["original"]["state"].items()
                },
                handoff_state={
                    name: common.compare(value, raw["candidate"]["handoff_state"][name])
                    for name, value in raw["original"]["handoff_state"].items()
                },
            )
            case.update(
                comparisons=compare,
                raw=dict(path=str(path), sha256=common.sha(path)),
                median_speedup=case["variants"]["original"]["median_seconds"]
                / case["variants"]["candidate"]["median_seconds"],
            )
            save()
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
    parser.add_argument("--layers", default="0,1,3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-region", type=int, default=768000000)
    parser.add_argument(
        "--weight-cache", default="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"
    )
    args = parser.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    tracking = dict(
        native=bool(ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()),
        python=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        skip_program_cache=False,
        exemptions=[],
    )
    assert tracking["native"] and tracking["python"]
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
        *sorted((common.ROOT / "tt").glob("*.py")),
    ]
    report = dict(
        status="running",
        trace_tracking=tracking,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources={str(p): common.sha(p) for p in sources},
        policy=policy,
        baseline="current production serial512/group512/compact norm/RMQK",
        candidate="actual production exact1024 hybrid helper; every other direct shape uses original512 fallback",
        memory_limit="Snapshots are not peaks or full48 fit proof; captured intermediates and all48persistent state need later server qualification.",
        context=common.CONTEXT,
        active_pages=common.ACTIVE_PAGES,
        sentinel_pages=common.SENTINEL_PAGES,
        layers=[],
        timing="standalone warmed trace chunkchain including input/page uploads and output readback; no serving roofline claim",
        scope="real selected-weight decoder residual outputs/publicstate and owner32decode; not full-model logits",
    )

    def save():
        common.dump(args.output / "summary.json", report)

    for path in sources:
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    save()
    try:
        for index in map(int, args.layers.split(",")):
            assert index in (0, 1, 3)
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
