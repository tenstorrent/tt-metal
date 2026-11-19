# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: real GR operands or serial serving traces, with test-local dispatch."""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_compact_gr_down import baseline, read_ranks
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving import mutable_cache, packed, trace_ids
from models.autoports.qwen_qwen3_8_flash_next.tt.compact_gr_down import compact_gr_down
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import GatheredProjection


def differences(actual, expected):
    assert len(actual) == len(expected)
    rows = []
    for rank, (left, right) in enumerate(zip(actual, expected, strict=True)):
        assert left.shape == right.shape and left.dtype == right.dtype
        integer = torch.int16 if left.dtype == torch.bfloat16 else torch.int32
        a, b = left.contiguous().view(integer), right.contiguous().view(integer)
        mask = a != b
        samples = []
        for coord in mask.nonzero()[:16].tolist():
            key = tuple(coord)
            samples.append(
                {
                    "coordinate": coord,
                    "actual_bits": int(a[key]),
                    "expected_bits": int(b[key]),
                    "actual_float": str(float(left[key])),
                    "expected_float": str(float(right[key])),
                }
            )
        rows.append(
            {
                "rank": rank,
                "changed_bits": int(mask.sum()),
                "samples": samples,
                "actual_finite": bool(torch.isfinite(left).all()),
                "expected_finite": bool(torch.isfinite(right).all()),
                "torch_equal": bool(torch.equal(left, right)),
            }
        )
    return rows


def require_exact(record):
    assert all(row["changed_bits"] == 0 for row in record), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager-operands", "serving-traces"), default="eager-operands")
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    parser.add_argument("--trace-region-size", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    assert not args.output.exists(), f"Refusing to overwrite evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__)] + sorted(
        p for p in (root / "tt").rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp")
    )
    report = {
        "passed": False,
        "mode": args.mode,
        "scope": "Reduced real layers0/1/3 diagnostic; no perf or acceptance claim",
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("TT_", "TTNN_"))},
        "calls": [],
        "comparisons": [],
        "eager_limit": "Real-operand dual reads synchronize and perturb lifetimes. The old baseline also executes untilize/retilize before the next native consumer, potentially conditioning LLK state. A pass only excludes observed operand mismatches; it does not exclude allocation/history or uninitialized compute-state dependencies. The serving-traces mode never executes the baseline compactor for a selected call.",
    }
    original = GatheredProjection.__call__
    mesh = gen = None
    enabled = False
    run_label = "setup"
    started = time.monotonic()

    def selected(instance, x):
        if not enabled or instance.valid_width != 80:
            return original(instance, x)
        value = instance.ccl.gather(x) if instance.before else x
        gathered = instance.ccl.gather(instance.projection(value))
        count = gathered.shape[-2]
        if not (tuple(gathered.shape) == (1, 1, count, 384) and 1 <= count <= 32):
            raise AssertionError(f"Unexpected diagnostic compaction shape {gathered.shape}")
        native = compact_gr_down(gathered)
        event = {
            "run": run_label,
            "logical": list(gathered.shape),
            "padded": list(gathered.padded_shape),
            "input_address": gathered.buffer_address(),
            "output_address": native.buffer_address(),
        }
        report["calls"].append(event)
        if args.mode == "eager-operands":
            reference = baseline(gathered)
            event["logical_comparison"] = differences(read_ranks(native), read_ranks(reference))
            # Padding is evidence, not an equality gate: the old slice/concat
            # may expose unspecified rows. Downstream full-model equality is strict.
            event["physical_comparison"] = differences(
                read_ranks(native, physical=True), read_ranks(reference, physical=True)
            )
            require_exact(event["logical_comparison"])
            del reference
        return native

    def compare_cache(actual, expected, result):
        assert actual.keys() == expected.keys()
        result["cache"] = {}
        for key in actual:
            result["cache"][key] = differences(actual[key], expected[key])
            require_exact(result["cache"][key])

    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the selected mesh descriptor"
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        options = {"layer_indices": (0, 1, 3), "max_batch_size": 2, "max_seq_len": 128}
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator(str(root), mesh, **options)
        GatheredProjection.__call__ = selected
        enabled = args.mode == "serving-traces"
        # Traced mode selects native before any preparation or capture. Its
        # wrapper only reads Python metadata; it never reads a device tensor.
        gen.prepare()
        report["prepared_seconds"] = time.monotonic() - started
        if args.mode == "eager-operands":
            for count in (1, 2, 3, 32):
                token_host = torch.tensor(([785, 13458, 13] * 11)[:count], dtype=torch.int32).reshape(1, count)
                word_host = torch.zeros((1, count, 16), dtype=torch.int32)
                word_host[:, :, 0], word_host[:, :, 1] = token_host, 1
                tokens, words = gen._upload(token_host), gen._upload(word_host)
                controls = None
                for use_native in (False, True):
                    enabled = use_native
                    run_label = f"q{count}:{'native' if enabled else 'baseline'}"
                    report["last_phase"] = run_label
                    print(f"GR_INTEGRATION_PHASE {run_label}", flush=True)
                    gen.reset()
                    gen._reset_cache(gen.prefill_cache)
                    output = gen.model.forward_chunk(tokens, words, cache=gen.prefill_cache)
                    logits, state = read_ranks(output), mutable_cache(gen.prefill_cache)
                    del output
                    if controls is None:
                        controls = logits, state
                    else:
                        result = {"run": run_label, "logits": differences(logits, controls[0])}
                        report["comparisons"].append(result)
                        assert all(r["actual_finite"] and r["expected_finite"] for r in result["logits"]), result
                        require_exact(result["logits"])
                        compare_cache(state, controls[1], result)
                del tokens, words
        else:
            identity = torch.arange(4, dtype=torch.int32).reshape(1, 4)
            prompts = ([785, 13458, 13], [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])

            def prefill(rows, slots):
                return gen.prefill_forward(
                    packed(rows),
                    page_table=identity.repeat(len(rows), 1),
                    kv_cache=[gen.caches[s] for s in slots],
                    prompt_lens=list(map(len, rows)),
                    slots=slots,
                )

            solo_logits, solo_states = [], []
            for index, prompt in enumerate(prompts):
                run_label = f"solo{index}"
                report["last_phase"] = run_label
                print(f"GR_INTEGRATION_PHASE {run_label}", flush=True)
                solo_logits.append(prefill([prompt], (0,))[0].clone())
                solo_states.append(mutable_cache(gen.caches[0]))
            run_label = "mixed_after_q32_capture"
            report["last_phase"] = run_label
            print(f"GR_INTEGRATION_PHASE {run_label}", flush=True)
            mixed = prefill(prompts, (0, 1))
            report["trace_ids"] = trace_ids(gen)
            for row in range(2):
                result = {"row": row, "logits": differences([mixed[row]], [solo_logits[row]])}
                report["comparisons"].append(result)
                assert all(r["actual_finite"] and r["expected_finite"] for r in result["logits"]), result
                require_exact(result["logits"])
                compare_cache(mutable_cache(gen.caches[row]), solo_states[row], result)
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        GatheredProjection.__call__ = original
        report["wall_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
