# SPDX-License-Identifier: Apache-2.0
"""Compare qualified serial S128 with S256/S512; root owns hardware execution.

This test module changes no production source. It shares the same selected B32
owner weights/public state across three private decoder views and serial traces.
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

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_view_probe as common
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_prefill_view_candidate import make_prefill_decoder_view
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import apply_layer, for_layer, load_policy


class Segment512(common.Segment):
    def __init__(self, layer, native_size, hidden, tokens):
        self.layer, self.mesh = layer, layer.mesh
        self.batch, self.native_size, self.row = 1, native_size, 0
        self.hidden, self.tokens = hidden, tokens
        self.page = layer.ops.upload(common.page_table(1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(1, 32, dtype=torch.int32)
        slots[0, 0] = common.SLOT
        self.slots = layer.ops.upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.inputs = []
        for offset in range(0, 512, native_size):
            h, ids, pos = self.host_inputs(0, offset)
            self.inputs.append(
                dict(
                    offset=offset,
                    hidden=layer.ops.shards(list(h.chunk(4, dim=0))),
                    tokens=layer.ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    positions=layer.ops.upload(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                )
            )
        self.output = ttnn.empty((1, 1, 512, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)


def run_layer(args, cfg, policy, layer_idx, report, save):
    destination = args.output / f"layer{layer_idx}"
    destination.mkdir()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region)
    trace = None
    row = dict(layer=layer_idx, variants={}, comparisons={}, context=common.CONTEXT)
    report["layers"].append(row)
    try:
        row.update(device_ids=list(mesh.get_device_ids()), initial_memory=common.memory_view(mesh))
        owner = MultichipDecoder.from_state_dict(
            CheckpointState(args.checkpoint),
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh,
            batch_size=common.BATCH,
            max_requests=2 * common.BATCH,
            chunk_size=128,
            attention_pool_tokens=common.PUBLIC_TOKENS + common.BATCH * 512,
            cache_dir=args.weight_cache,
        )
        apply_layer(owner, for_layer(policy, layer_idx))
        owner.prepare([1])
        row["policy"] = owner.precision_policy
        views = {n: make_prefill_decoder_view(owner, chunk_size=n, lengths=(n,)) for n in (128, 256, 512)}
        rng = torch.Generator().manual_seed(40512 + layer_idx)
        hidden = (torch.randn(4, 1, 1025, 2560, generator=rng) * 0.05).bfloat16()
        tokens = torch.randint(0, cfg.vocab_size, (1025,), generator=rng).int()
        path = destination / "inputs.pt"
        torch.save(dict(hidden_shards=hidden, tokens=tokens, page_table=common.page_table(32)), path)
        row["inputs"] = dict(path=str(path), sha256=common.sha(path))
        segments = {n: Segment512(view, n, hidden, tokens) for n, view in views.items()}
        decode = common.Decode(owner, hidden, tokens, position=1024)

        def reset():
            # Public tensors are intentionally reset more than once during this
            # reduced diagnostic; all reset work is outside timed replay windows.
            for view in views.values():
                common.reset(owner, view)

        # Warm all variants, continuation, owner decode and readback diagnostics
        # before any capture. No tracker acknowledgments or program-cache skip.
        for segment in segments.values():
            reset()
            for start in (0, 512):
                segment.refresh(start)
                segment.forward()
                common.read_shards(segment.output)
                common.public_state(owner)
            decode.forward()
            common.read_shards(decode.output)
            common.public_state(owner)
        row["prepared_memory"] = common.memory_view(mesh)
        save()
        raw_variants = {}
        for n, segment in segments.items():
            reset()
            segment.refresh(0)
            common.read_shards(segment.inputs[0]["positions"])
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            segment.forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            timings, raw = [], {}
            row["variants"][str(n)] = dict(native_chunk=n, native_calls=512 // n, timings=timings, trace_id=str(trace))
            save()
            for repeat in range(args.repeats):
                reset()
                for start in (0, 512):
                    segment.refresh(start)
                    common.read_shards(segment.inputs[0]["positions"])
                    begin = time.perf_counter_ns()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    output = common.read_shards(segment.output)
                    elapsed = (time.perf_counter_ns() - begin) / 1e9
                    timings.append(dict(repeat=repeat, start=start, tokens=512, seconds=elapsed))
                    raw[f"prefill_{start}"] = dict(output=output, state={})
                    torch.save(raw, destination / f"s{n}_raw.pt")
                    save()
            ttnn.release_trace(mesh, trace)
            trace = None
            state, checks = common.public_state(owner)
            raw["prefill_512"]["state"] = state
            decode.forward()
            handoff = common.read_shards(decode.output)
            handoff_state, handoff_checks = common.public_state(owner)
            raw["owner_decode_1024"] = dict(output=handoff, state=handoff_state)
            path = destination / f"s{n}_raw.pt"
            torch.save(raw, path)
            row["variants"][str(n)].update(
                median_seconds=statistics.median(item["seconds"] for item in timings),
                state_checks=checks,
                handoff_state_checks=handoff_checks,
                trace_released=True,
                raw=dict(path=str(path), sha256=common.sha(path)),
                memory=common.memory_view(mesh),
            )
            raw_variants[n] = raw
            save()
        reference = raw_variants[128]
        for n in (256, 512):
            row["comparisons"][str(n)] = {
                phase: dict(
                    output=common.compare(values["output"], raw_variants[n][phase]["output"]),
                    state={
                        key: common.compare(value, raw_variants[n][phase]["state"][key])
                        for key, value in values["state"].items()
                    },
                )
                for phase, values in reference.items()
            }
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
    if (
        os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
    ):
        raise ValueError("Tracking with program-cache checks must be enabled before Python starts")
    if args.repeats < 1:
        raise ValueError("Positive repeat count required")
    layers = [int(value) for value in args.layers.split(",")]
    if not layers or any(value not in (0, 1, 3) for value in layers):
        raise ValueError("Representative real layers are0,1,3")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = Qwen4ExpConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    policy_path = common.ROOT / "doc/datatype_sweep/selected_precision_config.json"
    policy = load_policy(policy_path)
    if cfg.max_position_embeddings != common.CONTEXT or policy["config_id"] != "combined_head4_gdn4_lofi":
        raise ValueError("Selected precision and full context are mandatory")
    # Test-process-only page mapping covering1025logical tokens. Existing helper
    # diagnostics read these same physical pages; production source is unchanged.
    common.ACTIVE_PAGES = [128 + (7 * index + 11) % 64 for index in range(33)]
    sources = [
        Path(__file__),
        Path(common.__file__),
        Path(__file__).with_name("benchmark_prefill_view_candidate.py"),
        *sorted((common.ROOT / "tt").glob("*.py")),
    ]
    report = dict(
        status="running",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={str(path): common.sha(path) for path in sources},
        policy=policy,
        policy_sha256=common.sha(policy_path),
        checkpoint=str(args.checkpoint.resolve()),
        config_sha256=common.sha(args.checkpoint / "config.json"),
        checkpoint_index_sha256=common.sha(args.checkpoint / "model.safetensors.index.json"),
        context=common.CONTEXT,
        public_pool_tokens=common.PUBLIC_TOKENS,
        public_slot=common.SLOT,
        active_pages=common.ACTIVE_PAGES,
        sentinel_pages=common.SENTINEL_PAGES,
        tracking=dict(enabled=True, skip_program_cache=False, exemptions=[]),
        timing_contract="Reduced-layer512token composite replay through four-shard blocking output readback; excludes reset/uploads; not serving/roofline evidence.",
        input_contract="Seeded synthetic residuals with real selected-policy weights; same1025tokens across all views; no full-model accuracy claim.",
        layers=[],
    )

    def save():
        common.dump(args.output / "summary.json", report)

    save()
    try:
        for layer in layers:
            run_layer(args, cfg, policy, layer, report, save)
        report["status"] = "measurements_complete_review_numerics"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
