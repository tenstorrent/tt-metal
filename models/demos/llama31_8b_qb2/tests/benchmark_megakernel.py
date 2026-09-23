# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched real-model experiment; run one hardware process at a time.

Example (from the repository, after hardware health verification)::

    python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
        --mode baseline --context 128 --output /outside/repo/baseline
    python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
        --mode mlp --context 128 --reference /outside/repo/baseline \
        --output /outside/repo/mlp

The optional --profile path is for a separate Tracy/device-profiler process.
It does not report host latency as a serving or device-latency measurement.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time

import torch


def make_prompt(tokenizer, context):
    text = "Explain how an operating system shares memory and processor time among several programs. "
    unit = tokenizer.encode(text, add_special_tokens=False)
    return ([tokenizer.bos_token_id] + unit * ((context + len(unit) - 1) // len(unit)))[:context]


def metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    if a.shape != b.shape or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError("Mismatched shapes or non-finite model outputs")
    error = a - b
    return {
        "pcc": torch.corrcoef(torch.stack((a, b)))[0, 1].item(),
        "relative_l2": (error.norm() / b.norm().clamp_min(1e-12)).item(),
        "max_abs": error.abs().max().item(),
        "exact": torch.equal(actual, expected),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "swiglu",
            "mlp",
            "mlp_reduce",
            "mlp_tail",
            "norm_mlp_tail",
            "gather_norm_mlp_tail",
            "post_attention",
            "attention_tail",
            "decoder",
            "decoder_loop",
            "decoder_loop_embedding",
            "decoder_loop_head",
            "decode_token",
        ),
        required=True,
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--hf-reference", type=Path, help="CPU reference.pt; fixes the teacher stream and adds HF accuracy evidence"
    )
    parser.add_argument(
        "--require-hf-pcc",
        type=float,
        help="Optional HF accuracy gate; otherwise report BF16-reference drift separately from matched TT checks",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--require-exact", action="store_true", help="Require exact teacher logits, touched KV and greedy output against reference")
    parser.add_argument("--projection-reader", choices=("original", "coalesced", "pipelined", "pipelined_rows"), default="original")
    parser.add_argument("--projection-lookahead", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--projection-buffers", type=int, choices=(2, 3, 4, 5), default=2)
    parser.add_argument("--coalesce-input", action="store_true")
    parser.add_argument("--scratch-init-once", choices=("off", "padding", "norm", "all"), default="off")
    parser.add_argument("--early-weight-blocks", type=int, choices=(0, 2, 3), default=0)
    parser.add_argument("--early-weight-phases", choices=("all", "qkv", "o", "gu", "down", "qkv_o_gu"), default="all")
    parser.add_argument("--share-qkv-workers", action="store_true")
    parser.add_argument("--projection-placement", choices=("row", "dram"), default="row")
    parser.add_argument("--prefetch-head-workers", action="store_true")
    parser.add_argument("--head-prefetch-targets", choices=("both", "qkv", "o"), default="both")
    parser.add_argument("--alias-projection-cbs", action="store_true")
    parser.add_argument("--prefetch-gu-blocks", type=int, choices=(0, 2, 4, 6), default=0)
    parser.add_argument("--prefetch-down-blocks", type=int, choices=(0, 2, 4, 6), default=0)
    parser.add_argument("--hoist-pack-config", action="store_true")
    parser.add_argument("--bank-vc", action="store_true")
    parser.add_argument("--wide-subblocks", action="store_true")
    parser.add_argument("--bounded-layer-barrier", action="store_true")
    parser.add_argument("--gu-workers", type=int, choices=(8, 16), default=8)
    parser.add_argument("--reuse-mlp-scratch", action="store_true")
    args = parser.parse_args()
    if args.reuse_mlp_scratch and args.mode != "mlp":
        parser.error("--reuse-mlp-scratch requires --mode mlp")
    if args.context < 1 or args.tokens < 3 or args.repeats < 1:
        parser.error("context >= 1, tokens >= 3 and repeats >= 1 are required")
    if args.profile != (os.environ.get("TT_METAL_DEVICE_PROFILER", "0") not in ("0", "")):
        parser.error("--profile must match TT_METAL_DEVICE_PROFILER; keep profiling and latency runs separate")
    if args.profile and os.environ.get("TT_METAL_WATCHER"):
        parser.error("Watcher and profiler must run separately")
    return args


def run(args):
    import ttnn
    from models.demos.llama31_8b_qb2.tt.generator import LlamaGenerator
    from models.demos.llama31_8b_qb2.tt.megakernel.decoder import enable_experimental_decode
    from models.demos.llama31_8b_qb2.tt.model import REVISION
    from models.demos.llama31_8b_qb2.tt.megakernel.tuning import ProjectionTuning
    from models.demos.llama31_8b_qb2.tt.generator_vllm import LlamaForCausalLM
    from models.demos.utils.trace_region_sizes import build_trace_device_params

    tuning = ProjectionTuning(
        reader=args.projection_reader, wide_subblocks=args.wide_subblocks,
        bounded_barrier=args.bounded_layer_barrier, buffer_count=args.projection_buffers, lookahead=args.projection_lookahead,
        hoist_pack_config=args.hoist_pack_config, bank_vc=args.bank_vc,
        prefetch_gu_blocks=args.prefetch_gu_blocks, prefetch_down_blocks=args.prefetch_down_blocks,
        alias_projection_cbs=args.alias_projection_cbs, prefetch_head_workers=args.prefetch_head_workers,
        projection_placement=args.projection_placement, coalesce_input=args.coalesce_input,
        head_prefetch_targets=args.head_prefetch_targets, share_qkv_workers=args.share_qkv_workers,
        early_weight_blocks=args.early_weight_blocks, scratch_init_once=args.scratch_init_once,
        early_weight_phases={"qkv":1, "o":2, "gu":4, "qkv_o_gu":7, "down":8, "all":15}[args.early_weight_phases],
    )
    if args.mode == "baseline" and tuning != ProjectionTuning():
        raise ValueError("Projection tuning applies only to experimental kernels")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    hf_reference = torch.load(args.hf_reference, map_location="cpu", weights_only=True) if args.hf_reference else None
    reference = None
    if args.reference:
        reference = torch.load(args.reference / "evidence.pt", map_location="cpu", weights_only=True)
        assert reference["context"] == args.context and reference["tokens"] == args.tokens
    result = {
        "mode": args.mode,
        "projection_tuning": asdict(tuning),
        "reuse_mlp_scratch": args.reuse_mlp_scratch,
        "gu_workers": args.gu_workers,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "checkpoint_revision": REVISION,
        "batch": 1,
        "context": args.context,
        "tokens": args.tokens,
        "profile_run": args.profile,
        "sampling": {"top_k": 1, "top_p": 0.0, "temperature": 1.0, "seed": 42},
        "limitations": (
            "B1; mode-dependent experimental composition. decoder_loop places all32 layers in one program; "
            "decode_token adds embedding/final gather/norm/head; sampling stays native. "
            "B1 active decode and inactive warmup; no serving qualification. See PROGRESS.md."
        ),
    }
    evidence = {"context": args.context, "tokens": args.tokens}
    ttnn.set_fabric_config(**LlamaForCausalLM.model_capabilities["fabric_config"])
    mesh = None
    generator = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=16384,
            **build_trace_device_params("llama3.1-8b-qb2-decoder"),
        )
        load_start = time.perf_counter()
        generator = LlamaGenerator(mesh, max_batch_size=1, trace_prefill=False, record_token_history=True)
        result["model_load_seconds"] = time.perf_counter() - load_start
        if args.mode != "baseline":
            enable_experimental_decode(
                generator.model,
                mode=args.mode,
                reuse_scratch=args.reuse_mlp_scratch,
                gu_workers=args.gu_workers,
                kv_cache=generator.kv_cache,
                tuning=tuning,
            )
        assert generator.model.num_layers == 32
        result["precision"] = generator.model.precision_policy
        evidence["precision"] = result["precision"]
        evidence["checkpoint_revision"] = REVISION
        evidence["sampling"] = result["sampling"]
        if reference is not None:
            for key in ("precision", "checkpoint_revision", "sampling"):
                assert evidence[key] == reference[key], f"Unmatched baseline {key}"
        prompt = make_prompt(generator.tokenizer, args.context)
        if hf_reference is not None:
            assert hf_reference["prompt"] == prompt
            assert hf_reference["checkpoint_revision"] == REVISION
            assert hf_reference["tokens"] == args.tokens and hf_reference["context"] == args.context
        if reference is not None:
            assert prompt == reference["prompt"]
        evidence["prompt"] = prompt
        settings = {**result["sampling"], "stop_on_eos": False, "host_sampling": False}

        # A full generation compiles and captures before the measurement window.
        warmup_start = time.perf_counter()
        warmup = generator.generate(prompt, args.tokens, **settings)
        ttnn.synchronize_device(mesh)
        result["warmup_capture_generation_seconds"] = time.perf_counter() - warmup_start
        result["warmup_perf"] = deepcopy(generator.last_perf)
        result["timing_denominator"] = args.tokens - 1
        evidence["warmup_tokens"] = warmup
        # Allocator metadata only, outside every measurement window. Static
        # program CBs/code are additional; this is not an execution peak.
        result["allocator_after_warmup"] = {}
        for name, kind in (("L1", ttnn.BufferType.L1), ("DRAM", ttnn.BufferType.DRAM)):
            view = ttnn.get_memory_view(mesh, kind)
            result["allocator_after_warmup"][name] = {
                key: getattr(view, key)
                for key in (
                    "num_banks",
                    "total_bytes_per_bank",
                    "total_bytes_allocated_per_bank",
                    "total_bytes_free_per_bank",
                    "largest_contiguous_bytes_free_per_bank",
                    "block_table",
                )
            }
        if args.profile:
            from tracy import signpost

            # Drain warmup/capture records before measuring. Without these
            # explicit drains, small-op records disappear from later replays
            # once the device profiler's finite DRAM buffers fill.
            ttnn.ReadDeviceProfiler(mesh)
            # Reset outside the signposts so each trace starts at the same
            # context and token. Profiling deliberately freezes this one token.
            for repeat in range(args.repeats):
                generator.refresh_decode_inputs(
                    torch.tensor([warmup[0]], dtype=torch.int32),
                    torch.tensor([args.context], dtype=torch.int32),
                )
                ttnn.synchronize_device(mesh)
                signpost(f"QB2_DECODE_BEGIN_{repeat}", f"mode={args.mode},B=1,context={args.context}")
                generator.replay_decode(sample=True)
                ttnn.synchronize_device(mesh)
                signpost(f"QB2_DECODE_END_{repeat}")
                ttnn.ReadDeviceProfiler(mesh)
                if os.environ.get("TT_METAL_PROFILER_SUM") == "1" and hasattr(generator.model, "fused_decode_loop"):
                    loop = generator.model.fused_decode_loop
                    counters = []
                    # Height-sharded storage is assembled in row-major grid
                    # order, independent of the body's role-ordered core list.
                    storage_cores = ttnn.corerange_to_cores(loop.grid, row_wise=True)
                    for device, shard in enumerate(ttnn.get_device_tensors(loop.state)):
                        tiles = ttnn.to_torch(shard).reshape(len(loop.cores), 32, 32)
                        for index, core in enumerate(storage_cores):
                            # Raw tile words480/481 map to face1,row14,col0/1.
                            cycles, count = int(tiles[index, 14, 16]), int(tiles[index, 14, 17])
                            assert cycles > 0 and count == 2 * loop.count and count <= 64
                            per_barrier = tiles[index, 16:20, :16].reshape(-1)[:count].tolist()
                            assert sum(per_barrier) == cycles and all(v > 0 for v in per_barrier)
                            counters.append(
                                {
                                    "device_shard": device,
                                    "logical_core": [core.x, core.y],
                                    "cycles": cycles,
                                    "barriers": count,
                                    "per_barrier_cycles": per_barrier,
                                }
                            )
                    result.setdefault("layer_barrier_counters", []).append({"repeat": repeat, "cores": counters})
            result["measurement"] = (
                "Read device kernel/firmware durations inside decode signposts from the profiler artifacts"
            )
            return result

        result["generation"] = []
        for repeat in range(args.repeats):
            output = generator.generate(prompt, args.tokens, **settings)
            ttnn.synchronize_device(mesh)
            assert output == warmup, "Repeated generation changed greedy output"
            assert int(generator._read_replicated(generator.positions)[0]) == args.context + args.tokens - 1
            result["generation"].append(deepcopy(generator.last_perf))
        evidence["generated_tokens"] = output
        result["generated_tokens"] = output
        result["generated_text"] = generator.tokenizer.decode(output)
        if reference is not None:
            result["greedy_token_agreement"] = (
                sum(a == b for a, b in zip(output, reference["generated_tokens"])) / args.tokens
            )

        # Teacher forcing isolates numerical drift from different token paths.
        # Readback is intentionally outside the generation performance evidence.
        forced = reference["teacher_tokens"] if reference is not None else output
        if hf_reference is not None:
            if reference is not None:
                assert forced == hf_reference["teacher_tokens"], "TT and HF references must use the same teacher stream"
            forced = hf_reference["teacher_tokens"]
        evidence["teacher_tokens"] = forced
        rows = []
        positions = []

        def next_input(step, prediction):
            rows.append(generator.read_logits().reshape(32, -1)[0].clone())
            positions.append(int(generator._read_replicated(generator.positions)[0]))
            return forced[step]

        generator.generate(prompt, args.tokens, next_input=next_input, **settings)
        logits = torch.stack(rows)
        evidence["teacher_logits"] = logits
        evidence["teacher_positions"] = positions
        result["teacher_positions"] = positions
        if reference is not None:
            result["teacher_logits"] = metrics(logits, reference["teacher_logits"])
            result["teacher_step_metrics"] = [metrics(a, b) for a, b in zip(logits, reference["teacher_logits"])]
            assert result["teacher_logits"]["pcc"] >= 0.999
            assert result["teacher_logits"]["relative_l2"] < 0.03

        if hf_reference is not None:
            result["hf_logits"] = metrics(logits, hf_reference["teacher_logits"])
            result["hf_step_metrics"] = [metrics(a, b) for a, b in zip(logits, hf_reference["teacher_logits"])]
            result["hf_teacher_top1_agreement"] = (
                (logits.argmax(-1) == hf_reference["teacher_logits"].argmax(-1)).float().mean().item()
            )
            # The original selected BFP4/BFP8 model may differ from BF16 HF
            # below 0.99 even with identical greedy tokens. Preserve that fact;
            # it is separate from the much stricter matched TT fusion check.
            result["hf_pcc_099_passed"] = result["hf_logits"]["pcc"] >= 0.99
            result["required_hf_pcc"] = args.require_hf_pcc

        # Read only pages touched by this request, but include every layer and
        # TP shard. Slice/readback programs are warmed after releasing traces.
        generator.release_traces()
        used_pages = (args.context + args.tokens - 2) // 128 + 1
        cache_evidence = []
        for pair in generator.kv_cache:
            cache_evidence.append(
                [
                    torch.stack(
                        [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor[1 : used_pages + 1])]
                    )
                    for tensor in pair
                ]
            )
        evidence["teacher_cache"] = cache_evidence
        if reference is not None:
            result["teacher_cache_metrics"] = [
                [metrics(a, b) for a, b in zip(pair, expected)]
                for pair, expected in zip(cache_evidence, reference["teacher_cache"])
            ]
            for pair in result["teacher_cache_metrics"]:
                for item in pair:
                    assert item["pcc"] >= 0.999 and item["relative_l2"] < 0.03, item
        if hf_reference is not None:
            visible = args.context + args.tokens - 1
            result["hf_cache_metrics"] = [
                [
                    metrics(a.permute(0, 2, 1, 3, 4).reshape(1, 8, -1, 128)[:, :, :visible], b)
                    for a, b in zip(pair, expected)
                ]
                for pair, expected in zip(cache_evidence, hf_reference["teacher_cache"])
            ]
        if args.require_exact:
            assert reference is not None, "--require-exact requires --reference"
            assert result["teacher_logits"]["exact"], "Teacher logits differ from reference"
            assert all(m["exact"] for pair in result["teacher_cache_metrics"] for m in pair), "Touched KV differs"
            assert output == reference["generated_tokens"], "Greedy outputs differ"
        torch.save(evidence, args.output / "evidence.pt")
        result["comparison_checks_passed"] = True
        if args.require_hf_pcc is not None:
            assert hf_reference is not None, "--require-hf-pcc needs --hf-reference"
            assert result["hf_logits"]["pcc"] >= args.require_hf_pcc
        return result
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        if "teacher_logits" in evidence:
            torch.save(evidence, args.output / "failure-evidence.pt")
        raise
    finally:
        # Preserve partial results when a numerical assertion fails.
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)
