# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare bounded public prefill to the native two-real-layer wrapper.

Diagnostic only: no trace capture, no sampling, and no HF/full-model quality
claim. Run under an external bounded timeout; this script never resets devices.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import RecurrentState


def tensor_metrics(reference, actual):
    if reference.shape != actual.shape:
        return {"passed": False, "reference_shape": list(reference.shape), "actual_shape": list(actual.shape)}
    left, right = reference.reshape(-1), actual.reshape(-1)
    sums = [0.0] * 5
    max_error, finite = 0.0, True
    # Avoid materializing several full-vocabulary FP64 copies simultaneously.
    for start in range(0, left.numel(), 1_048_576):
        x, y = (value[start : start + 1_048_576].double() for value in (left, right))
        finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        terms = (x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum())
        sums = [total + float(term) for total, term in zip(sums, terms)]
        max_error = max(max_error, float((x - y).abs().max()))
    sx, sy, sxx, syy, sxy = sums
    count = left.numel()
    variance = max(0.0, sxx - sx * sx / count) * max(0.0, syy - sy * sy / count)
    equal = torch.equal(reference, actual)
    pcc = 1.0 if equal else (sxy - sx * sy / count) / variance**0.5 if variance else 0.0
    if not finite:
        pcc, max_error = 0.0, None
    return {
        "shape": list(reference.shape),
        "finite": finite,
        "equal": equal,
        "pcc": pcc,
        "max_abs_error": max_error,
        "passed": finite and pcc >= 0.995,
    }


def logit_metrics(reference, actual):
    result = tensor_metrics(reference, actual)
    if reference.shape != actual.shape:
        return result
    expected = reference.argmax(dim=-1, keepdim=True)
    candidates = actual.topk(100, dim=-1).indices
    for k in (1, 5, 100):
        result[f"top{k}_agreement"] = float((candidates[..., :k] == expected).any(dim=-1).float().mean())
    predicted = actual.argmax(dim=-1, keepdim=True)
    result["top1_agreement"] = float((predicted == expected).float().mean())
    result["first_top1_miss"] = None
    misses = (predicted != expected).squeeze(-1).nonzero()
    if misses.numel():
        result["first_top1_miss"] = misses[0].tolist()
    result["passed"] = (
        result["passed"]
        and result["top1_agreement"] >= 0.98
        and result["top5_agreement"] >= 0.98
        and result["top100_agreement"] == 1.0
    )
    return result


def read_states(generator):
    result = {}
    for index, state in zip(generator.model.layer_indices, generator.cache):
        if isinstance(state, RecurrentState):
            result[str(index)] = {
                "conv": ttnn.to_torch(state.conv, mesh_composer=ttnn.ConcatMeshToTensor(generator.mesh, dim=-1)),
                "recurrent": ttnn.to_torch(
                    state.recurrent, mesh_composer=ttnn.ConcatMeshToTensor(generator.mesh, dim=1)
                ),
            }
    return result


def addresses(generator):
    return [
        {
            name: int(getattr(state, name).buffer_address())
            for name in (("conv", "recurrent") if isinstance(state, RecurrentState) else ("key", "value"))
        }
        for state in generator.cache
    ]


def native_prefill(generator, tokens, *, start=0):
    model = generator.model
    count = tokens.shape[1]
    ids = model.io._upload(tokens.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    positions = model.io._upload(
        torch.arange(start, start + count, dtype=torch.int32)[None].repeat(tokens.shape[0], 1),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    output = model.prefill(ids, positions, cache=generator.cache, start_pos=start, return_all_logits=True)
    return ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(generator.mesh, dim=-1))


def public_prefill(generator, tokens, *, start=0, end=None):
    end = tokens.shape[1] if end is None else end
    return generator.prefill_forward(
        tokens,
        page_table=generator.page_host,
        kv_cache=generator.cache,
        prompt_lens=[end] * tokens.shape[0],
        start_pos=[start] * tokens.shape[0],
        return_all_logits=True,
    )


def compare_case(generator, tokens, *, resume_start=None):
    length = tokens.shape[1]
    label = f"length={length}" + ("" if resume_start is None else f" resume={resume_start}")
    before = addresses(generator)
    print(f"NATIVE_BEGIN {label}", flush=True)
    generator.reset()
    if resume_start is not None:
        # Match the native public BF16 state-publication boundary exactly.
        native_prefill(generator, tokens[:, :resume_start])
        reference = native_prefill(generator, tokens[:, resume_start:], start=resume_start)
    else:
        reference = native_prefill(generator, tokens)
    reference_states = read_states(generator)
    print(f"PUBLIC_BEGIN {label}", flush=True)
    generator.reset()
    if resume_start is not None:
        public_prefill(generator, tokens, end=resume_start)
        actual = public_prefill(generator, tokens, start=resume_start)
    else:
        actual = public_prefill(generator, tokens)
    actual_states = read_states(generator)
    logits = logit_metrics(reference, actual)
    states = {
        layer: {kind: tensor_metrics(value, actual_states[layer][kind]) for kind, value in values.items()}
        for layer, values in reference_states.items()
    }
    stable = before == addresses(generator)
    result = {
        "length": length,
        "resume_start": resume_start,
        "logits": logits,
        "gdn_states": states,
        "stable_cache_addresses": stable,
        "passed": logits["passed"]
        and stable
        and all(v["passed"] for values in states.values() for v in values.values()),
    }
    print(
        f"CASE_DONE {label} pass={result['passed']} pcc={logits.get('pcc')} "
        f"top1={logits.get('top1_agreement')} top5={logits.get('top5_agreement')}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=range(1, 33), default=1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print(f"MESH_OPEN batch={args.batch}", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    generator = None
    start = time.perf_counter()
    result = {
        "layers": [0, 3],
        "batch": args.batch,
        "context_allocation": 2048,
        "trace_capture": False,
        "thresholds": {"pcc": 0.995, "top1_agreement": 0.98, "top5_agreement": 0.98, "top100_agreement": 1.0},
        "cases": [],
    }
    try:
        model = Model(mesh, layer_indices=[0, 3])
        generator = QwenGenerator(model, batch_size=args.batch, max_seq_len=2048)
        # Different rows expose row/cache mixing while remaining deterministic.
        tokens = torch.arange(545, dtype=torch.int32)[None] + 100 + torch.arange(args.batch)[:, None] * 997
        for length in (33, 128, 513):
            result["cases"].append(compare_case(generator, tokens[:, :length]))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        for resume in (31, 512):
            result["cases"].append(compare_case(generator, tokens, resume_start=resume))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["passed"] = all(case["passed"] for case in result["cases"])
        result["generator_captures"] = generator.counters["captures"]
        result["helper_counters"] = {
            "single": generator.prefill_single.counters,
            "uniform": generator.prefill_uniform.counters,
        }
        if result["generator_captures"]:
            raise AssertionError("The diagnostic probe unexpectedly captured traces")
    except BaseException as error:
        result["passed"] = False
        result["exception"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)
    print(f"BOUNDED_PREFILL_PROBE pass={result['passed']} output={args.output}", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
