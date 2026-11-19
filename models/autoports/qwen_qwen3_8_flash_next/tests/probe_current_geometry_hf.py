# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-owner-only reduced sampler evidence, never a cross-geometry gate waiver.

One current-production configured-B2/C128 shared-pool generator, real layers0/1/3.
Single-active controls are NOT true configured-B1 executions. All predictions
come from the common traced device sampler. Full-logit host reads are diagnostic
only and never supply the next input; exact forced histories are 42/43/44.
--self-test imports neither TTNN nor model weights. Device runs require tracking.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/full_model"
PROMPTS = ([785, 13458, 13], [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])
FAMILIES = ("solo0", "solo1", "mixed", "migrated", "survivor")
FORCED = (42, 43, 44)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_prediction(logits, token, hf_logits):
    """Score the actual device token, preserving deterministic argmax/tie distinctions."""
    logits, hf_logits = logits.float().flatten(), hf_logits.float().flatten()
    if logits.shape != hf_logits.shape or logits.numel() < 100:
        raise ValueError("Expected matching full-vocabulary rows of at least 100 logits")
    if not torch.isfinite(logits).all() or not torch.isfinite(hf_logits).all():
        raise ValueError("Nonfinite full logits cannot establish greedy correctness")
    if not 0 <= token < logits.numel():
        raise ValueError("Device prediction is outside vocabulary")
    top = hf_logits.topk(100).indices.tolist()
    return {
        "device_token": token,
        "tt_argmax": int(logits.argmax()),
        "tt_max_tie_ids": (logits == logits.max()).nonzero().flatten().tolist(),
        "device_token_is_tt_max": bool(logits[token] == logits.max()),
        "device_token_logit": float(logits[token]),
        "tt_max_logit": float(logits.max()),
        "hf_argmax": int(hf_logits.argmax()),
        "hf_max_tie_ids": (hf_logits == hf_logits.max()).nonzero().flatten().tolist(),
        "hf_top100_ids": top,
        "top1": token == int(hf_logits.argmax()),
        "top5": token in top[:5],
        "top100": token in top,
        "device_token_is_hf_max": bool(hf_logits[token] == hf_logits.max()),
        "hf_strict_rank": int((hf_logits > hf_logits[token]).sum()) + 1,
        "tt_argmax_in_hf_top5": int(logits.argmax()) in top[:5],
        "tt_argmax_in_hf_top100": int(logits.argmax()) in top,
    }


def accuracy(rows):
    if not rows:
        raise ValueError("Empty accuracy population")
    scores = {f"top{k}": sum(row[f"top{k}"] for row in rows) / len(rows) for k in (1, 5, 100)}
    return {"rows": len(rows), **scores, "bar_passed": scores["top5"] >= 0.98 and scores["top100"] == 1.0}


def exact_outputs(left, right):
    return left.keys() == right.keys() and all(
        left[r]["device_token"] == right[r]["device_token"] and torch.equal(left[r]["logits"], right[r]["logits"])
        for r in left
    )


def self_test():
    # A tied genuine greedy token is valid even when it is not torch.argmax.
    hf = -torch.arange(128, dtype=torch.float32)
    tt = hf.clone()
    tt[1] = tt[0]
    row = score_prediction(tt, 1, hf)
    assert row["device_token_is_tt_max"] and row["tt_argmax"] == 0 and not row["top1"] and row["top5"]
    hf_tied = hf.clone()
    hf_tied[1] = hf_tied[0]
    tied = score_prediction(tt, 1, hf_tied)
    assert tied["device_token_is_hf_max"] and not tied["top1"] and tied["hf_max_tie_ids"] == [0, 1]
    assert not score_prediction(tt, 4, hf)["device_token_is_tt_max"]
    wrong = tt.clone()
    wrong[127] = 10
    miss = score_prediction(wrong, 127, hf)
    assert miss["device_token_is_tt_max"] and not miss["top5"] and not miss["top100"]
    assert not accuracy([row, miss])["bar_passed"]
    assert accuracy([row])["bar_passed"]
    for token, values in ((-1, tt), (128, tt), (0, tt.clone().fill_(float("nan")))):
        try:
            score_prediction(values, token, hf)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid diagnostic accepted")
    left = {0: {"device_token": 1, "logits": tt}}
    assert exact_outputs(left, {0: {"device_token": 1, "logits": tt.clone()}})
    assert not exact_outputs(left, {0: {"device_token": 0, "logits": tt.clone()}})
    assert not exact_outputs(left, {0: {"device_token": 1, "logits": tt + 1}})
    assert "ttnn" not in sys.modules
    print("CPU self-test PASS: actual-token ranking, tie semantics, nonfinite/range/greedy negatives, repeat equality")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensors", type=Path)
    parser.add_argument("--hf-json", type=Path, default=DOC / "shared_pool_hf_control_v2.json")
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or args.tensors is None:
        parser.error("--output and --tensors are required and must be new paths")
    if args.output.exists() or args.tensors.exists() or args.output.resolve() == args.tensors.resolve():
        parser.error("Refusing to overwrite evidence")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Enable TT_METAL_TRACE_ALLOC_TRACKING=1 before process startup")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache tracking must remain enabled")
    if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        parser.error("Restore the accepted P300x2 mesh descriptor after lane-env")
    torch.set_num_threads(4)
    hf_metadata = json.loads(args.hf_json.read_text())
    hf_path = Path(hf_metadata["tensor_artifact"]["path"])
    assert digest(hf_path) == hf_metadata["tensor_artifact"]["sha256"]
    assert hf_metadata["prompts"] == list(PROMPTS) and hf_metadata["forced_tokens"] == list(FORCED)
    checkpoint = Path(args.checkpoint or hf_metadata["snapshot"]).resolve()
    assert checkpoint == Path(hf_metadata["snapshot"]).resolve(), "Checkpoint must match the saved original-HF control"
    hf = torch.load(hf_path, map_location="cpu", weights_only=False)["runs"]
    historical_path = DOC / "shared_pool_generator_v5.json"
    historical = json.loads(historical_path.read_text())
    assert historical["passed"] is False

    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving import packed, positions, trace_ids
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_generator import decode_geometry_comparison
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import SamplingParams, build_generator

    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    sources = [path for path in sorted((ROOT / "tt").rglob("*")) if path.suffix in (".py", ".cpp", ".hpp")]
    sources.append(Path(__file__))
    sources += [ROOT / "tests" / name for name in ("probe_shared_pool_generator.py", "probe_full_model_serving.py")]
    sources += [Path("models/common/sampling") / name for name in ("generator.py", "tt_sampling.py")]
    native_root = Path("ttnn/cpp/ttnn/operations/experimental/deepseek_prefill")
    for operation in ("moe_fused_swiglu", "post_combine_reduce"):
        sources += [path for path in sorted((native_root / operation).rglob("*")) if path.suffix in (".cpp", ".hpp")]
    sources.append(Path("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h"))
    source_hashes = {str(path.resolve()): digest(path) for path in sources}
    report = {
        "completed": False,
        "diagnostic_checks_passed": False,
        "acceptance_claim": False,
        "scope": "current-production real layers0/1/3; configuredB2/C128 shared pool; forced-history sampled predictions",
        "single_active_scope": "configuredB2 single-active n1; no true configuredB1 instance is executed",
        "command": sys.argv,
        "source_hashes": source_hashes,
        "prompts": PROMPTS,
        "forced_tokens": FORCED,
        "tracking": True,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_VISIBLE_DEVICES",
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
            )
        },
        "checkpoint": str(checkpoint),
        "checkpoint_revision": hf_metadata["revision"],
        "checkpoint_config_sha256": digest(checkpoint / "config.json"),
        "mesh_descriptor_sha256": digest(os.environ["TT_MESH_GRAPH_DESC_PATH"]),
        "runtime": {"torch": torch.__version__, "ttnn_module": ttnn.__file__},
        "sampling_parameters": "temperature=[0.0]*active_rows, top_k=[1]*active_rows, top_p=[1.0]*active_rows, seed=[0]*active_rows; migration resets both authoritative request rows",
        "host_logits": "diagnostic snapshots after device sampling; never used as token feedback",
        "hf_reference": {
            "json": str(args.hf_json),
            "json_sha256": digest(args.hf_json),
            "tensor": hf_metadata["tensor_artifact"],
            "backend": hf_metadata["backend"],
        },
        "historical_frozen_gate": {
            "path": str(historical_path),
            "sha256": digest(historical_path),
            "passed": False,
            "failures": historical["failures"],
            "token43": historical["solo_vs_migrated_decode"],
            "survivor44": historical["solo_vs_survivor_decode"],
            "disposition": "FAILED preserved; this probe does not waive or replace its thresholds",
        },
        "rows": [],
        "checks": [],
    }
    runs = {}
    mesh = gen = None
    started = time.monotonic()
    failure = None

    def mark(phase):
        print(json.dumps({"phase": phase, "seconds": round(time.monotonic() - started, 3)}), flush=True)

    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        assert condition, name

    try:
        mark("LOAD_REDUCED_MODEL")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        options = dict(layer_indices=(0, 1, 3), max_batch_size=2, max_seq_len=128, defer_kv_cache=True)
        options["checkpoint"] = str(checkpoint)
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator(str(ROOT), mesh, **options)
        pool = gen.allocate_kv_cache_per_layer([((16, 1, 32, 256), torch.bfloat16, 0)])
        tables = torch.tensor([[15, 14, 13, 12], [11, 10, 9, 8]], dtype=torch.int32)
        gen.warmup_model_prefill(kv_cache=pool)
        gen.warmup_model_decode(kv_cache=pool)
        report["programs_after_prepare"] = mesh.num_program_cache_entries()
        report["counters_after_prepare"] = dict(gen.counters)
        report["sampler_class"] = (
            type(gen.sampler.tt_sampling).__module__ + "." + type(gen.sampler.tt_sampling).__name__
        )
        check("common split greedy selected", not gen.sampler.tt_sampling.force_argmax_sampling)

        def params(count):
            return SamplingParams(temperature=[0.0] * count, top_k=[1] * count, top_p=[1.0] * count, seed=[0] * count)

        def pool_addresses():
            return [
                part.buffer_address()
                for pair in pool.pairs.values()
                for tensor in pair
                for part in ttnn.get_device_tensors(tensor)
            ]

        addresses = pool_addresses()

        def snapshot(family, repeat, step, requests, slots, output):
            logits = gen._read_logits(gen.sample_logits).clone()
            feedback = [ttnn.to_torch(part).flatten().clone() for part in ttnn.get_device_tensors(gen.decode_tokens)]
            result = {}
            for row, (request, slot) in enumerate(zip(requests, slots, strict=True)):
                token = int(output.flatten()[row])
                check(
                    f"{family}/{repeat}/{step}/r{request}/feedback_all_ranks",
                    all(int(x[slot]) == token for x in feedback),
                )
                logical_position = len(PROMPTS[request]) - 1 + step
                qsa_positions = positions(gen.caches[slot], logical_position + 1)
                assert gen.caches[slot].position == logical_position + 1
                reference = hf[f"{family}/repeat{repeat}"][step][request]["logits"]
                score = score_prediction(logits[slot], token, reference)
                check(f"{family}/{repeat}/{step}/r{request}/semantically_greedy", score["device_token_is_tt_max"])
                report["rows"].append(
                    {
                        "family": family,
                        "repeat": repeat,
                        "step": step,
                        "phase": "prefill" if step == 0 else f"forced{FORCED[step-1]}",
                        "request": request,
                        "physical_slot": slot,
                        "active_width": len(slots),
                        "configured_batch_size": 2,
                        "logical_logit_position": logical_position,
                        "qsa_positions": qsa_positions,
                        "hf_run": f"{family}/repeat{repeat}",
                        **score,
                    }
                )
                result[request] = {"logits": logits[slot].clone(), "device_token": token, "physical_slot": slot}
            return result

        for repeat in range(2):
            for family in FAMILIES:
                mark(f"{family}/repeat{repeat}")
                gen.reset()
                requests = (int(family[-1]),) if family.startswith("solo") else (0, 1)
                slots = tuple(range(len(requests)))
                rows = [PROMPTS[r] for r in requests]
                table = tables[list(requests)]
                before_prefill = dict(gen.counters)
                output = gen.prefill_forward(
                    packed(rows),
                    page_table=table,
                    page_tables_per_layer=[table],
                    kv_cache=pool,
                    prompt_lens=list(map(len, rows)),
                    slots=slots,
                    sampling_params=params(len(slots)),
                )
                check(
                    f"{family}/{repeat}/prefill_common_traced_sampling",
                    gen.counters["sampling_replays"] - before_prefill["sampling_replays"] == 1,
                )
                key = f"{family}/repeat{repeat}"
                runs[key] = [snapshot(family, repeat, 0, requests, slots, output)]
                for step, token in enumerate(FORCED, 1):
                    migration = family in ("migrated", "survivor") and step == 2
                    if migration:
                        requests, slots = (1, 0), (0, 1)
                    if family == "survivor" and step == 3:
                        gen.release_request(0)
                        requests, slots = (0,), (1,)
                    table = tables[list(requests)]
                    commands = (
                        dict(
                            slot_remap=(1, 0),
                            reload_sampling_params=True,
                            reset_sampling_state=True,
                            sampling_params=params(2),
                        )
                        if migration
                        else {}
                    )
                    before = dict(gen.counters)
                    output = gen.decode_forward(
                        torch.tensor([token] * len(slots)),
                        [len(PROMPTS[r]) + step - 1 for r in requests],
                        page_table=table,
                        page_tables_per_layer=[table],
                        kv_cache=pool,
                        slots=slots,
                        sample_on_device=True,
                        reload_inputs=True,
                        **commands,
                    )
                    check(
                        f"{key}/step{step}/traced_model_and_sampler",
                        all(gen.counters[name] - before[name] == 1 for name in ("model_replays", "sampling_replays")),
                    )
                    runs[key].append(snapshot(family, repeat, step, requests, slots, output))
                if repeat:
                    check(
                        f"{family}/exact_repeat_logits_and_device_tokens",
                        all(exact_outputs(a, b) for a, b in zip(runs[f"{family}/repeat0"], runs[key], strict=True)),
                    )
            if repeat == 0:
                retained_traces = trace_ids(gen)
                program_count = mesh.num_program_cache_entries()
                retained_counters = dict(gen.counters)
            else:
                check("repeat suite reuses all trace IDs", retained_traces == trace_ids(gen))
                check("repeat suite adds no compiled programs", program_count == mesh.num_program_cache_entries())
                check(
                    "repeat suite neither captures nor releases",
                    all(gen.counters[name] == retained_counters[name] for name in ("captures", "releases")),
                )
        for repeat in range(2):
            for step in range(4):
                check(
                    f"migration_vs_continuous/repeat{repeat}/step{step}",
                    exact_outputs(runs[f"mixed/repeat{repeat}"][step], runs[f"migrated/repeat{repeat}"][step]),
                )
                if step < 3:
                    check(
                        f"survivor_prefix_vs_migrated/repeat{repeat}/step{step}",
                        exact_outputs(runs[f"survivor/repeat{repeat}"][step], runs[f"migrated/repeat{repeat}"][step]),
                    )
        check("shared pool addresses unchanged", addresses == pool_addresses())
        report["current_cross_geometry_frozen_comparisons"] = {}
        for step in (1, 2):
            reference = torch.stack([runs[f"solo{r}/repeat0"][step][r]["logits"] for r in range(2)])
            actual = torch.stack([runs["migrated/repeat0"][step][r]["logits"] for r in range(2)])
            report["current_cross_geometry_frozen_comparisons"][f"forced{FORCED[step-1]}"] = decode_geometry_comparison(
                reference, actual
            )
        report["current_cross_geometry_frozen_comparisons"]["survivor44"] = decode_geometry_comparison(
            runs["solo0/repeat0"][3][0]["logits"].unsqueeze(0), runs["survivor/repeat0"][3][0]["logits"].unsqueeze(0)
        )
        # Repeats are reproducibility checks, not independent accuracy samples.
        report["hf_accuracy_by_family_phase"] = {
            f"{family}/{phase}": accuracy(
                [
                    r
                    for r in report["rows"]
                    if r["repeat"] == 0 and r["family"] == family and (r["step"] == 0) == (phase == "prefill")
                ]
            )
            for family in FAMILIES
            for phase in ("prefill", "decode")
        }
        report["hf_accuracy_bar_passed"] = all(
            row["bar_passed"] for row in report["hf_accuracy_by_family_phase"].values()
        )
        report["diagnostic_checks_passed"] = True
        report["completed"] = True
    except BaseException as error:
        failure = error
        report["error"] = traceback.format_exc()
    finally:
        report["wall_seconds"] = time.monotonic() - started
        if gen is not None:
            report["counters_before_close"] = dict(gen.counters)
            report["trace_ids_before_close"] = trace_ids(gen)
            report["programs_final"] = mesh.num_program_cache_entries()
        report["source_hashes_unchanged"] = all(digest(path) == value for path, value in source_hashes.items())
        args.tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"runs": runs, "metadata": report}, args.tensors)
        report["tensor_artifact"] = {"path": str(args.tensors), "sha256": digest(args.tensors)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        mark("EVIDENCE_SAVED")
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    if failure is not None:
        raise failure
    assert report["source_hashes_unchanged"], "Sources changed during run; provenance is ambiguous"
    assert report["hf_accuracy_bar_passed"], "Direct HF accuracy bar failed; preserve all diagnostics for investigation"
    print("Completed sampled-token/HF diagnostic; historical frozen cross-geometry FAILED remains separate", flush=True)


if __name__ == "__main__":
    main()
