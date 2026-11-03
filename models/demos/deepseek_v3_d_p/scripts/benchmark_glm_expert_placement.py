# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drive the upstream GLM harness, with explicit inputs and archived raw samples."""

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["smoke", "routing", "perf", "short"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
a = p.parse_args()
out = a.output
out.mkdir(parents=True, exist_ok=False)
repo = Path(os.environ["TT_METAL_HOME"])
sys.path.insert(0, str(repo))
import pytest
from loguru import logger

mode = a.mode
layers = 10 if mode == "smoke" else 78
chunks = 2 if mode in ("smoke", "short") else 11
iters = 1 if mode == "routing" else (2 if mode == "smoke" else 10)
traced = mode != "routing"
trace = Path(os.environ["PREFILL_TRACE_DIR"])
metadata = json.loads((trace / "metadata.json").read_text())
assert len(metadata["token_ids"]) >= chunks * 5120
assert (trace / "dsa/indexer_k_layer_0").is_dir()
if mode == "routing":
    os.environ["PREFILL_DUMP_ROUTING_DIR"] = str(out / "routing")
else:
    assert not os.getenv("PREFILL_DUMP_ROUTING_DIR")
records = []
iteration_totals = []


def sink(message):
    text = message.record["message"]
    m = re.fullmatch(r"\[chunk timing\] iter=(\d+) chunk=(\d+) ([\d.]+) ms", text)
    if m:
        row = dict(iteration=int(m[1]), chunk=int(m[2]), replay_ms=float(m[3]))
        records.append(row)
        with (out / "samples.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    m = re.fullmatch(r"iter (\d+) done \((\d+) chunks\) in ([\d.]+) seconds", text)
    if m:
        iteration_totals.append(dict(iteration=int(m[1]), chunks=int(m[2]), whole_seconds=float(m[3])))


logger.add(sink, level="INFO")
state = dict(
    mode=mode,
    scope="GLM-5.3 chunked KV-producing prefill; final layer KV-only; no LM head/decode",
    layers=layers,
    chunks=chunks,
    chunk_tokens=5120,
    num_iters=iters,
    trace=traced,
    cache=str(a.cache),
    trace_directory=str(trace),
    tokens_sha256=hashlib.sha256(
        json.dumps(metadata["token_ids"][: chunks * 5120], separators=(",", ":")).encode()
    ).hexdigest(),
    samples_warmup_iterations=[0],
    status="running",
    started=time.time(),
)


def save():
    (out / "measurement.json").write_text(json.dumps(state, indent=2) + "\n")


class Plugin:
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        assert len(items) == 1, [x.nodeid for x in items]
        item = items[0]
        params = item.callspec.params
        for k, v in dict(num_layers=layers, n_chunks=chunks, num_iters=iters, preload_isl=0, use_trace=traced).items():
            assert params[k] == v, (k, params[k], v)
        module = inspect.getmodule(item.obj)
        original = module.run_chunked_transformer_updated

        def run(*args, **kwargs):
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            assert bound.arguments["variant"].name == "glm_5_3"
            bound.arguments.update(
                weight_cache_path=a.cache,
                seq_cache=module.SEQ_CACHE,
                baseline_chunk_times_s=None,
                perf_margin=None,
                check_pcc=True,
                kv_pcc_threshold=module.KV_CACHE_PCC_THRESHOLD_SCALED_FP8,
            )
            state.update(
                seq_cache=module.SEQ_CACHE,
                kv_pcc_threshold=module.KV_CACHE_PCC_THRESHOLD_SCALED_FP8,
                nodeid=item.nodeid,
            )
            save()
            # The upstream PCC reader already transfers the result after timings. Record exact valid
            # cache contents too, so the placement comparison need not rely on broad golden PCC floors.
            gather = module.gather_cache_natural

            def checked_gather(*gargs, **gkwargs):
                cache, stripes = gather(*gargs, **gkwargs)
                positions = module.blockcyclic_positions(stripes, module.CHUNK, module.SEQ_CACHE)
                rows = []
                for slot in range(cache.shape[0]):
                    natural = module.unrotate_cache_layer(cache[slot], positions, chunks * module.CHUNK).contiguous()
                    rows.append(
                        dict(
                            slot=slot,
                            shape=list(natural.shape),
                            sha256=hashlib.sha256(natural.numpy().tobytes()).hexdigest(),
                        )
                    )
                with (out / "cache-digests.jsonl").open("a") as f:
                    f.write(json.dumps(dict(shape=list(cache.shape), stripes=stripes, layers=rows)) + "\n")
                return cache, stripes

            module.gather_cache_natural = checked_gather
            try:
                return original(*bound.args, **bound.kwargs)
            finally:
                module.gather_cache_natural = gather

        module.run_chunked_transformer_updated = run

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            state["skipped"] = report.longreprtext


selector = (
    f"torus-xy-8x4 and L{layers} and preload0 and "
    + ("chunks2 and not chunks20" if mode in ("smoke", "short") else "chunks_eleven")
    + " and "
    + ("iters1" if mode == "routing" else "two_iters" if mode == "smoke" else "ten_iters")
    + " and "
    + ("traced" if traced else "notrace")
)
save()
result = pytest.main(
    [
        "-q",
        "-s",
        "--tb=short",
        "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc",
        "-k",
        selector,
    ],
    plugins=[Plugin()],
)
state.update(
    pytest_returncode=int(result), finished=time.time(), iteration_totals=iteration_totals, sample_count=len(records)
)
valid = result == 0 and not state.get("skipped") and len(records) == iters * chunks
if valid and mode == "routing":
    import torch

    files = sorted((out / "routing").glob("*.pt"))
    expected = {(layer, start) for layer in range(3, layers - 1) for start in range(0, chunks * 5120, 5120)}
    seen = set()
    capture = []
    for f in files:
        sample = torch.load(f, weights_only=True)
        key = (sample["layer_idx"], sample["actual_start"])
        assert key not in seen
        seen.add(key)
        ids = sample["expert_ids"]
        weights = sample["expert_weights"]
        assert tuple(ids.shape) == (5120, 8) and weights.shape == ids.shape
        assert ((ids >= 0) & (ids < 256)).all()
        assert torch.isfinite(weights).all()
        assert (ids.sort(1).values.diff(dim=1) > 0).all()
        capture.append(dict(file=f.name, layer=key[0], start=key[1], sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    assert seen == expected, (len(seen), len(expected))
    (out / "routing-manifest.json").write_text(json.dumps(capture, indent=2) + "\n")
    state["routing_files"] = len(files)
state["status"] = "complete" if valid else "failed"
save()
raise SystemExit(0 if valid else int(result) or 1)
