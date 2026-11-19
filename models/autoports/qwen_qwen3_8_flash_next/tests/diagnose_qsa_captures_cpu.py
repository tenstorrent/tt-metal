# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline causal checks for closed QSA projection and 105568-window captures.

Imports no TTNN and opens no device. Missing/incomplete captures or provenance
fail closed. Numeric disagreements are retained as observations, not rewritten
as acceptance. Full-run outputs do not contain projected operands or IDs; this
helper explicitly distinguishes reduced-capture observations from agreement
with the saved full-run output.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .probe_qsa import load_reference

CHUNKS = (105568, 58528, 93024, 100032, 16736, 9184, 7840)
COORDINATES = ((105584, 328), (58531, 244), (9190, 595), (7842, 514))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def cpu_rne_bf16(value):
    """Explicit IEEE FP32-to-BF16 round-to-nearest-even for finite operands."""
    assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
    bits = value.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.to(torch.int16).view(torch.bfloat16)


def tree_sum(value):
    tails = []
    while value.shape[-1] > 1:
        width = value.shape[-1]
        half = width // 2
        if width % 2:
            tails.append(value[..., -1:])
        value = value[..., :half] + value[..., half : 2 * half]
    for tail in tails:
        value = value + tail
    return value


def measure(expected, actual):
    assert expected.numel() == actual.numel()
    a, b = expected.double().flatten(), actual.double().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "pcc_fp64": None, "max_abs_error": None}
    ac, bc = a - a.mean(), b - b.mean()
    denominator = ac.norm() * bc.norm()
    pcc = float(ac @ bc / denominator) if denominator else float(torch.equal(a, b))
    return {
        "finite": True,
        "pcc_fp64": max(-1.0, min(1.0, pcc)),
        "max_abs_error": float((a - b).abs().max()),
        "exact_equal": torch.equal(expected.reshape(-1), actual.reshape(-1)),
        "bf16_equal_fraction": float((expected.bfloat16().flatten() == actual.bfloat16().flatten()).float().mean()),
    }


def source_bound_run(
    path, stage, producer, capture, output, *, capture_flag="--capture", allowed_exit=(0,), paths=None
):
    record = json.loads(path.read_text())
    assert (
        record["status"] == "finished" and record["exit_code"] in allowed_exit
    ), "Producer is incomplete or has an unexpected exit"
    log = Path(record["log"])
    if not log.is_absolute():
        log = Path(record["cwd"]) / log
    assert digest(log) == record["log_sha256"]
    argv = record["argv"]
    assert any(value.endswith("." + producer) for value in argv)
    for flag, expected in {capture_flag: capture, "--output": output, **(paths or {})}.items():
        recorded = Path(argv[argv.index(flag) + 1])
        if not recorded.is_absolute():
            recorded = Path(record["cwd"]) / recorded
        assert recorded.resolve() == expected.resolve()
    relevant = {
        name: value
        for name, value in record["source_sha256"].items()
        if name.startswith("tt/") or name == f"tests/{producer}.py"
    }
    assert relevant and f"tests/{producer}.py" in relevant
    for name, expected in relevant.items():
        assert digest(stage / name) == expected, f"Producer/runtime source changed: {name}"
    return {
        "path": str(path),
        "sha256": digest(path),
        "argv": argv,
        "source_sha256": relevant,
        "log": str(log),
        "log_sha256": digest(log),
        "exit_code": record["exit_code"],
        "current_producer_runtime_sources_exact": True,
    }


@torch.inference_mode()
def inspect_projection(model, source, capture, run_report):
    starts = capture["chunk_starts"]
    assert tuple(starts) == CHUNKS and run_report["chunk_starts"] == starts
    expected, exact, tree = [], [], []
    weight = model.indexer.index_qk_proj.weight
    assert torch.equal(capture["weight_roundtrip"], weight.T)
    assert torch.equal(capture["weight32_roundtrip"], weight.T.float())
    assert all(run_report["operand_roundtrip"]["input_equal"])
    assert len(run_report["operand_roundtrip"]["input_equal"]) == len(starts)
    for start in starts:
        hidden = source["hidden"][:, start : start + 32]
        expected.append(model.indexer.index_qk_proj(hidden))
        exact.append(F.linear(hidden.double(), weight.double()))
        tree.append(
            torch.cat(
                [
                    tree_sum(hidden.float().unsqueeze(-2) * weight[column : column + 32].float()[None, None]).squeeze(
                        -1
                    )
                    for column in range(0, 640, 32)
                ],
                -1,
            )
        )
    expected, exact, tree = (torch.cat(values, 1) for values in (expected, exact, tree))
    assert torch.equal(capture["expected"], expected), "Original32-row literal projection identity differs"
    assert torch.equal(capture["hf_fp64"], exact), "Captured FP64 control identity differs"
    raw, rounded = capture["sfpu_tree_fp32"], capture["sfpu_tree"]
    assert raw.shape == rounded.shape == expected.shape == (1, 224, 640)
    rne = cpu_rne_bf16(raw)
    result = {
        "chunks": starts,
        "literal_32row_recomputation_exact": True,
        "fp64_32row_recomputation_exact": True,
        "device_precast_vs_cpu_tree": measure(tree, raw),
        "device_rounded_vs_cpu_rne": measure(rne, rounded),
        "explicit_rne_vs_torch_cast_exact": torch.equal(rne, raw.bfloat16()),
        "device_rounded_vs_literal": measure(expected, rounded),
        "device_rounded_vs_fp64_rounded": measure(exact.bfloat16(), rounded),
        "native_vs_literal": measure(expected, capture["native"]),
        "critical_coordinates": [],
    }
    for position, column in COORDINATES:
        start = position // 32 * 32
        index = starts.index(start) * 32 + position % 32
        literal = float(expected[0, index, column])
        exact_rounded = float(exact[0, index, column].bfloat16())
        midpoint = (literal + exact_rounded) / 2 if literal != exact_rounded else None
        result["critical_coordinates"].append(
            {
                "position": position,
                "column": column,
                "original_chunk_start": start,
                "capture_row": index,
                "literal_BF16": literal,
                "FP64_dot": float(exact[0, index, column]),
                "FP64_rounded_BF16": exact_rounded,
                "CPU_tree_precast": float(tree[0, index, column]),
                "CPU_tree_rounded_BF16": float(tree[0, index, column].bfloat16()),
                "TT_tree_precast": float(raw[0, index, column]),
                "TT_tree_rounded_BF16": float(rounded[0, index, column]),
                "CPU_RNE_of_TT_precast": float(rne[0, index, column]),
                "TT_native_BF16": float(capture["native"][0, index, column]),
                "literal_FP64_adjacent_midpoint": midpoint,
            }
        )
    return result


@torch.inference_mode()
def inspect_window(model, fixture, capture, full, run_report):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import repeat_kv

    prefix, count = fixture["prefix"], fixture["rows"]
    assert (prefix, count) == (105568, 32)
    assert (run_report["prefix"], run_report["rows"], run_report["capacity"]) == (prefix, count, 262144)
    assert torch.equal(capture["query_positions"].reshape(-1), torch.arange(prefix, prefix + count).float())
    assert int(capture["state_position"].item()) == prefix
    keys = torch.cat((fixture["prefix_state"]["keys"], fixture["projections"]["key"]), 2)
    values = torch.cat((fixture["prefix_state"]["values"], fixture["projections"]["value"]), 2)
    inserted_keys = torch.cat((fixture["prefix_state"]["keys"], capture["key"]), 2)
    inserted_values = torch.cat(
        (fixture["prefix_state"]["values"], capture["value_projection"].reshape(1, count, 2, 256).transpose(1, 2)), 2
    )
    prefix_keys = capture.get("tt_compressed_prefix", fixture["prefix_state"]["compressed"].unsqueeze(1))
    if "tt_compressed_prefix" not in capture:
        assert run_report["prefix_indexer_mode"] == "exact HF setup injection"
    compressed = torch.cat((prefix_keys, capture["compressed_window"]), 2)
    result = {
        "prefix": prefix,
        "prefix_indexer_mode": run_report["prefix_indexer_mode"],
        "full_run_comparison_scope": "Reduced window uses injected HF K/V prefix; full-run operands/IDs were not captured.",
        "window_vs_saved_full": measure(full["actual"][:, prefix : prefix + count], capture["actual"]),
        "window_vs_literal_HF": measure(fixture["saved_expected"], capture["actual"]),
        "rows": [],
    }
    for row in range(count):
        position = prefix + row
        ids = capture["selected_ids"][row].long()
        valid = ids != 0xFFFFFFFF
        selected = ids[valid]
        actual_set, expected_set = set(selected.tolist()), set(fixture["selected_ids"][row].tolist())
        complete = (position + 1) // 4
        q_index = capture["index_query"][0, :, row].float()
        cpu_scores = (q_index @ compressed[0, 0, :complete].float().T).relu().sum(0) / math.sqrt(128)
        recorded_scores = capture["indexer_scores"][row, :complete]
        score_top = set(recorded_scores.topk(512).indices.tolist())
        cpu_top = set(cpu_scores.topk(512).indices.tolist())
        device_top = set(capture["indexer_topk_blocks"][row].long().tolist())
        item = {
            "position": position,
            "valid_count": len(selected),
            "required_valid_count": 2048 + (position + 1) % 4,
            "duplicates": len(selected) - len(actual_set),
            "future_or_negative_ids": selected[(selected > position) | (selected < 0)].tolist(),
            "missing_literal_token_ids": sorted(expected_set - actual_set),
            "extra_literal_token_ids": sorted(actual_set - expected_set),
            "device_topk_vs_CPU_topk_same_recorded_scores": sorted(device_top ^ score_top),
            "device_score_topk_vs_CPU_scores_same_operands": sorted(score_top ^ cpu_top),
            "device_scores_vs_CPU_same_operands": measure(cpu_scores, recorded_scores),
            "window_vs_saved_full": measure(
                full["actual"][:, position : position + 1], capture["actual"][:, row : row + 1]
            ),
            "window_vs_literal_HF": measure(
                fixture["saved_expected"][:, row : row + 1], capture["actual"][:, row : row + 1]
            ),
        }
        if item["future_or_negative_ids"] or item["duplicates"] or item["valid_count"] != item["required_valid_count"]:
            item["structural_validity_passed"] = False
            item["attention_control"] = "Not evaluated: captured selection violates causal count/range/uniqueness"
            result["rows"].append(item)
            continue
        item["structural_validity_passed"] = True
        item["gather_keys_vs_inserted"] = measure(
            inserted_keys[0, :, selected], capture["selected_keys"][row, :, valid]
        )
        item["gather_values_vs_inserted"] = measure(
            inserted_values[0, :, selected], capture["selected_values"][row, :, valid]
        )
        query = fixture["projections"]["query"][:, :, row : row + 1]
        k, v = repeat_kv(keys[:, :, selected], 12), repeat_kv(values[:, :, selected], 12)
        probabilities = torch.softmax((query @ k.transpose(-1, -2)) * (256**-0.5), -1, dtype=torch.float32).bfloat16()
        attention = probabilities @ v
        attention *= fixture["projections"]["gate"][:, :, row : row + 1].sigmoid()
        oracle = model.o_proj(attention.reshape(1, 1, 6144))
        item["HF_attention_actual_IDs_vs_literal"] = measure(fixture["saved_expected"][:, row : row + 1], oracle)
        item["HF_attention_actual_IDs_vs_window"] = measure(oracle, capture["actual"][:, row : row + 1])
        item["HF_attention_actual_IDs_vs_saved_full"] = measure(oracle, full["actual"][:, position : position + 1])
        item["HF_attention_actual_IDs_capture_recomputed_exact"] = torch.equal(
            oracle, capture["hf_attention_on_tt_selected_ids"][:, row : row + 1]
        )
        q = capture["query"][:, :, row : row + 1]
        k = repeat_kv(capture["selected_keys"][row : row + 1, :, valid], 12)
        v = repeat_kv(capture["selected_values"][row : row + 1, :, valid], 12)
        p = torch.softmax((q @ k.transpose(-1, -2)) * (256**-0.5), -1, dtype=torch.float32).bfloat16()
        same_operands = p @ v
        item["device_attention_vs_CPU_same_TT_operands"] = measure(same_operands, capture["attention_output"][row])
        item["CPU_same_operands_capture_recomputed_exact"] = torch.equal(
            same_operands, capture["cpu_attention_on_tt_operands"][:, :, row : row + 1]
        )
        result["rows"].append(item)
    result["critical_row105584"] = result["rows"][16]
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--full-device-cache", type=Path, required=True)
    parser.add_argument("--full-device-report", type=Path, required=True)
    parser.add_argument("--full-device-provenance", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    for name in ("projection", "window"):
        for kind in ("capture", "report", "provenance"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": __doc__, "accepted": False, "device_executed": False, "status": "pending"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        required = {
            name: value
            for name, value in vars(args).items()
            if isinstance(value, Path) and name not in ("output", "snapshot")
        }
        missing = [str(value) for value in required.values() if not value.is_file()]
        report["missing_inputs"] = missing
        if missing:
            raise FileNotFoundError("Required closed-capture inputs are missing: " + ", ".join(missing))
        assert "ttnn" not in sys.modules
        torch.set_num_threads(8)
        stage = Path(__file__).parents[1]
        report["input_sha256"] = {name: digest(value) for name, value in required.items()}
        report["helper_sha256"] = digest(Path(__file__))
        report["producer_runs"] = {
            name: source_bound_run(
                getattr(args, name + "_provenance"),
                stage,
                producer,
                getattr(args, name + "_capture"),
                getattr(args, name + "_report"),
                paths={"--snapshot": args.snapshot, **({"--fixture": args.fixture} if name == "window" else {})},
            )
            for name, producer in (
                ("projection", "probe_qsa_projection_precision"),
                ("window", "probe_qsa_context_window"),
            )
        }
        report["producer_runs"]["full_run_literal_failure"] = source_bound_run(
            args.full_device_provenance,
            stage,
            "probe_qsa_prefill",
            args.full_device_cache,
            args.full_device_report,
            capture_flag="--device-output-cache",
            allowed_exit=(1,),
            paths={"--snapshot": args.snapshot, "--fixture-cache": args.source_cache},
        )
        source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
        full = torch.load(args.full_device_cache, weights_only=True, mmap=True, map_location="cpu")
        fixture = torch.load(args.fixture, weights_only=True, mmap=True, map_location="cpu")
        projection = torch.load(args.projection_capture, weights_only=True, mmap=True, map_location="cpu")
        window = torch.load(args.window_capture, weights_only=True, mmap=True, map_location="cpu")
        projection_report, window_report = (
            json.loads(path.read_text()) for path in (args.projection_report, args.window_report)
        )
        full_report = json.loads(args.full_device_report.read_text())
        assert full_report["position_after_prefill"] == 262144
        assert full_report["prefill"]["finite"] and full_report["prefill"]["rows_compared"] == 262144
        assert full_report["prefill"]["passed"] is False
        assert full_report["error"] == "AssertionError: Full prefill all-row gate failed before boundary continuation"
        report["preserved_literal_full_run_gate"] = full_report["prefill"]
        _, model, _, _, provenance = load_reference(args.snapshot)
        key = source["key"]
        assert key == fixture["source_key"] == full["reference_fixture"]["source_key"]
        assert key["snapshot"] == str(args.snapshot.resolve())
        assert key["length"] == key["capacity"] == full["length"] == 262144 and key["decode_steps"] == 0
        assert key["seed"] == 1979
        assert fixture["provenance"] == projection_report["provenance"] == provenance
        for name in ("reference_sha256", "config_sha256", "transformers"):
            assert key[name] == provenance[name]
        assert torch.equal(fixture["hidden"], source["hidden"][:, 105568:105600])
        assert torch.equal(fixture["saved_expected"], source["expected"][:, 105568:105600])
        assert Path(projection_report["capture_path"]).resolve() == args.projection_capture.resolve()
        assert Path(window_report["capture_path"]).resolve() == args.window_capture.resolve()
        report["reference_provenance"] = provenance
        report["source_key"] = key
        report["projection"] = inspect_projection(model, source, projection, projection_report)
        report["window"] = inspect_window(model, fixture, window, full, window_report)
        report["status"] = "completed_capture_diagnostic"
        assert "ttnn" not in sys.modules
    except Exception as exc:
        report["status"] = "failed_closed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "critical_projection_coordinates": report["projection"]["critical_coordinates"],
                "critical_window_row": report["window"]["critical_row105584"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
