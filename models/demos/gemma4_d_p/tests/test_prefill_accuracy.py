# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 prefill accuracy against GPU references, outside the migration path.

The mesh, one KV slot and the traced serving runtime are built once and reused for every dataset. Each
dataset's token prefix is prefilled chunk by chunk; then every KV head is compared with the GPU capture
while the final hidden states' next-token likelihoods are scored against its final-layer stream.

A dataset is a prepared GPU trace (see ``prepare_gpu_reference.py``) whose source capture also kept the
final decoder layer's output. Select datasets with ``GEMMA4_ACCURACY_DATASETS=name,...`` (default: all).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from models.common.weight_cache import weight_cache_is_complete
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import weight_cache_identity
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_validation import (
    PREPARED_GPU_TRACE_LAYOUT,
    compare_slot_cache,
    read_cache_tensor,
)
from models.demos.gemma4_d_p.tt.runners.likelihood import (
    HIDDEN_SAMPLES,
    HiddenSampler,
    check_limits,
    compare,
    format_report,
    gpu_final_layer_stream,
)

TRACE_REGION_SIZE = 256 * 1024 * 1024
GPU_TRACES = Path("/mnt/models/huggingface/gpu_traces/gemma4_d_p")


@dataclass(frozen=True)
class Dataset:
    trace_dir: Path
    min_per_head_pcc: float
    min_overall_pcc: float
    max_overall_rrmse: float
    # Per depth bin, keyed by its first position: minimum top-1 agreement, maximum mean ΔNLL and maximum
    # mean top-20 KL against the GPU reference. Text of another kind needs its own measured limits.
    likelihood_limits: dict


DATASETS = {
    "gutenberg": Dataset(
        trace_dir=GPU_TRACES / "gutenberg-135",
        min_per_head_pcc=0.928,
        min_overall_pcc=0.978,
        max_overall_rrmse=0.208,
        # Main at chunk 8192 scores no lower than 0.899 / 0.882 / 0.890 top-1 and no higher than 0.197 / 0.300 / 0.287
        # ΔNLL and 0.128 / 0.177 / 0.178 KL across the 8K, 16K, 128K and 256K contexts.
        likelihood_limits={0: (0.89, 0.25, 0.16), 8192: (0.87, 0.33, 0.20), 65536: (0.88, 0.32, 0.20)},
    ),
}


def selected_datasets():
    names = [name for name in os.getenv("GEMMA4_ACCURACY_DATASETS", ",".join(DATASETS)).split(",") if name]
    if unknown := set(names) - set(DATASETS):
        raise ValueError(f"Unknown datasets {sorted(unknown)}; known: {sorted(DATASETS)}")
    return {name: DATASETS[name] for name in names}


def load_token_ids(adapter, dataset, context_len):
    """The dataset's token IDs, after checking it can be scored at ``context_len``; None if too short."""
    metadata = json.loads((dataset.trace_dir / "metadata.json").read_text())
    assert metadata["model_id"] == adapter.hf_model_id, f"{dataset.trace_dir} was captured from another model"
    assert metadata["layout"] == PREPARED_GPU_TRACE_LAYOUT, f"Prepare {dataset.trace_dir} with prepare_gpu_reference"
    assert metadata["n_layers"] == Gemma4ServiceConfig.NUM_LAYERS
    if len(metadata["token_ids"]) < context_len:
        return None
    for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
        assert (dataset.trace_dir / "kv_cache" / f"layer_{layer}.safetensors").is_file(), f"Missing layer {layer} KV"
    *_, stream = gpu_final_layer_stream(dataset.trace_dir)
    assert stream["row_count"] >= context_len, f"{dataset.trace_dir}: final-layer stream is shorter than its tokens"
    return metadata["token_ids"]


def run_params(adapter):
    config = Gemma4ServiceConfig
    return PrefillRunParams(
        mesh_shape=config.MESH_SHAPE,
        num_layers=config.NUM_LAYERS,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=config.MAX_SEQ_LEN,
        chunk_size=config.CHUNK_SIZE,
        num_users=1,
        capacity_factor=1,
        num_links=2,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path(config.MESH_SHAPE),
        use_trace=True,
    )


def prefill(runtime, kv_cache, token_ids, sampler):
    chunk = Gemma4ServiceConfig.CHUNK_SIZE
    for start in range(0, len(token_ids), chunk):
        runtime.prefill_chunk(
            runtime.make_chunk_input(token_ids[start : start + chunk]),
            kv_cache,
            slot_id=0,
            actual_start=start,
            actual_end=start + chunk,
        )
        sampler.add_chunk(runtime.output, start, start + chunk)


def compare_kv(kv_cache, context_len, trace_dir, output_dir):
    staging = {}

    def read_heads(layer):
        cache = kv_cache.layers[layer]
        tensors = ((0, cache.kv),) if hasattr(cache, "kv") else ((4, cache.k), (20, cache.v))
        for config_start, tensor in tensors:
            for head, actual in enumerate(read_cache_tensor(tensor, 0, context_len, staging)):
                yield config_start + head, actual

    compare_slot_cache(read_heads, 0, context_len, trace_dir, summary_dir=output_dir)
    return json.loads((output_dir / "gemma4_slot0.json").read_text())


def kv_criteria(report, dataset):
    """(name, required, achieved, passed) rows, in the form ``check_limits`` returns."""
    overall, worst = report["error_metrics"]["overall"], report["worst_head"]
    return [
        (
            "Minimum per-head PCC",
            f">= {dataset.min_per_head_pcc:.6f}",
            worst["pcc"],
            worst["pcc"] >= dataset.min_per_head_pcc,
        ),
        ("Overall PCC", f">= {dataset.min_overall_pcc:.6f}", overall["pcc"], overall["pcc"] >= dataset.min_overall_pcc),
        (
            "Overall RRMSE",
            f"< {dataset.max_overall_rrmse:.6f}",
            overall["relative_rmse"],
            overall["relative_rmse"] < dataset.max_overall_rrmse,
        ),
    ]


def print_criteria(name, criteria):
    def value(achieved):
        return "n/a" if achieved is None else f"{achieved:.6f}"

    print(f"\n{name}: accuracy criteria")
    print("┌──────────────────────┬─────────────┬──────────┬────────┐")
    print("│ Criterion            │ Required    │ Achieved │ Result │")
    print("├──────────────────────┼─────────────┼──────────┼────────┤")
    for criterion, required, achieved, passed in criteria:
        print(f"│ {criterion:<20} │ {required:<11} │ {value(achieved):>8} │ {'PASS' if passed else 'FAIL':<6} │")
    print("└──────────────────────┴─────────────┴──────────┴────────┘", flush=True)
    return [
        f"{name}: {criterion} failed: actual={value(achieved)}, required {required}"
        for criterion, required, achieved, passed in criteria
        if not passed
    ]


def check_dataset(runtime, kv_cache, name, dataset, token_ids, context_len, output_dir):
    """Prefill one dataset's first ``context_len`` tokens and return its failed criteria."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # The sampler takes the whole document: the gold token after the last prompt position lies past it.
    sampler = HiddenSampler(context_len, Gemma4ServiceConfig.CHUNK_SIZE, token_ids)
    prefill(runtime, kv_cache, token_ids[:context_len], sampler)
    sampler.save(output_dir / HIDDEN_SAMPLES)
    prefill_seconds = time.perf_counter() - started

    # Scoring only needs the saved hidden states, so it runs on the host while the KV is read back.
    with ThreadPoolExecutor(1) as scorer:
        likelihood = scorer.submit(compare, dataset.trace_dir, output_dir)
        started = time.perf_counter()
        kv = compare_kv(kv_cache, context_len, dataset.trace_dir, output_dir)
        kv_seconds = time.perf_counter() - started
        likelihood = likelihood.result()
    (output_dir / "likelihood.json").write_text(json.dumps(likelihood, indent=2) + "\n")

    overall, worst = kv["error_metrics"]["overall"], kv["worst_head"]
    print(
        f"\n{name}: {context_len} tokens; prefill {prefill_seconds:.0f} s, KV comparison {kv_seconds:.0f} s, "
        f"report in {output_dir}"
    )
    print(f"{'Layer':>7} {'PCC':>12} {'Relative RMSE':>16} {'RMSE':>12}")
    for entry in kv["error_metrics"]["layers"]:
        print(f"{entry['layer']:>7} {entry['pcc']:>12.6f} {entry['relative_rmse']:>16.6f} {entry['rmse']:>12.6f}")
    print(f"{'Overall':>7} {overall['pcc']:>12.6f} {overall['relative_rmse']:>16.6f} {overall['rmse']:>12.6f}")
    print(f"Worst head: layer={worst['layer']} head={worst['head']} type={worst['cache_type']} PCC={worst['pcc']:.6f}")
    print(f"\nNext-token likelihood vs GPU reference\n{format_report(likelihood)}", flush=True)

    return print_criteria(name, kv_criteria(kv, dataset) + check_limits(likelihood, dataset.likelihood_limits))


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric(
    [Gemma4ServiceConfig.MESH_SHAPE], device_params_extra={"trace_region_size": TRACE_REGION_SIZE}
)
@pytest.mark.parametrize("context_len", [8192, 16384, 131072, 262144], ids=["8k", "16k", "128k", "256k"])
def test_prefill_accuracy(mesh_device, context_len, tmp_path, request):
    adapter = Gemma4PrefillAdapter()
    hf_config = adapter.load_hf_config()
    cache = adapter.weight_cache_path(Gemma4ServiceConfig.MESH_SHAPE)
    identity = weight_cache_identity(
        adapter.hf_model_id,
        Gemma4ServiceConfig.NUM_LAYERS,
        Gemma4ServiceConfig.MESH_SHAPE,
        Gemma4Precision.load(adapter.hf_model_id),
    )
    assert weight_cache_is_complete(cache, **identity), f"Missing or incompatible TT weight cache: {cache}"
    # Check every dataset before the model is built, so a bad one fails in seconds, not minutes.
    datasets = {
        name: (dataset, token_ids)
        for name, dataset in selected_datasets().items()
        if (token_ids := load_token_ids(adapter, dataset, context_len)) is not None
    }
    if not datasets:
        pytest.skip(f"No selected dataset holds {context_len} tokens")
    output_root = tmp_path
    if summaries := os.getenv("PREFILL_SUMMARIES"):
        output_root = Path(summaries) / "gemma4_prefill_accuracy" / request.node.name
    print(f"Accuracy test results: {output_root}", flush=True)

    params = run_params(adapter)
    kv_cache = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_cache)
    runtime.capture_trace(kv_cache)
    failures = []
    try:
        for name, (dataset, token_ids) in datasets.items():
            failures += check_dataset(runtime, kv_cache, name, dataset, token_ids, context_len, output_root / name)
    finally:
        runtime.release_trace()
    if failures:
        pytest.fail("Accuracy checks failed:\n" + "\n".join(failures), pytrace=False)
