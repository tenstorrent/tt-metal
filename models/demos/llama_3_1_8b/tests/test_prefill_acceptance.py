# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline acceptance test (``method/ACCEPTANCE.md``): full-depth, full-width, real weights.

One real forward of the whole 32-layer model on the spec's 8x4 Blackhole Galaxy mesh, using the
golden trace's exact token ids, with every layer's on-device K and V compared against that trace.
Nothing here is skipped, xfailed, dimension-reduced or fed random weights, and every number in the
report is read back from the run that just happened.

Environment (all five are read, none is inferred):

| Variable | Use |
|---|---|
| ``PREFILL_SPEC`` | mesh, parallelism, dataformats and the two PCC thresholds |
| ``PREFILL_HF_MODEL`` / ``HF_MODEL`` | the real checkpoint directory |
| ``PREFILL_TRACE_DIR`` | the CPU golden trace, the SAME one in both modes |
| ``PREFILL_CHUNKED`` | ``0`` one-shot, ``1`` multi-chunk at the spec's chunk size |
| ``PREFILL_ACCEPTANCE_OUT`` | where this test writes its JSON report |

One-shot and chunked differ ONLY in ``chunk_size``: one-shot sets it to the whole padded sequence,
chunked to the spec's 5120. Everything else — weights, tokens, trace, mesh, dtypes — is identical,
which is what makes "chunked produces the same KV as one-shot" a meaningful claim rather than a
comparison of two different runs.

The assert is the spec's ``pcc_lower_bound``. Layers between that and ``pcc_target`` pass and are
reported; the reason they are there is root-caused in
``tests/torch_ref/test_golden_trace_rope_precision.py`` and recorded in the README.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference.config import LlamaConfig
from models.demos.llama_3_1_8b.tests.common import galaxy_mesh, load_spec
from models.demos.llama_3_1_8b.tt.model_config import load_state_dict, resolve_checkpoint
from models.demos.llama_3_1_8b.tt.runners.prefill_kv_validation import load_trace, per_layer_kv_pcc
from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import PrefillRuntimeConfig, TtPrefillRuntime
from models.demos.llama_3_1_8b.utils.general import weight_cache_dir

DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b, "float32": ttnn.float32}


def _trace_dir() -> Path:
    """``PREFILL_TRACE_DIR`` is set by the verifier; an inherited shell value does not override it."""
    d = os.environ.get("PREFILL_TRACE_DIR")
    assert d, "PREFILL_TRACE_DIR is not set — the acceptance run needs the resolved golden trace"
    return Path(d)


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


@galaxy_mesh()
def test_prefill_kv(mesh_device, device_params, topology_name):
    spec = load_spec()
    sp, tp = spec["parallelism"]["sp"], spec["parallelism"]["tp"]
    chunk_size = spec["shapes"]["chunk_size"]
    lower, target = float(spec["acceptance"]["pcc_lower_bound"]), float(spec["acceptance"]["pcc_target"])
    cache_dtype = DTYPES[spec["dataformats"]["kv_cache"]["default"]]
    weight_dtype = DTYPES[spec["dataformats"]["weights"]["default"]]

    assert tuple(mesh_device.shape) == (sp, tp), f"mesh is {tuple(mesh_device.shape)}, spec binds ({sp}, {tp})"

    # --- inputs ---------------------------------------------------------------------------
    checkpoint = resolve_checkpoint()
    trace_dir = _trace_dir()
    meta = load_trace(trace_dir)
    token_ids = list(meta["token_ids"])
    seq_len = len(token_ids)

    chunked = os.environ.get("PREFILL_CHUNKED", "0") == "1"
    mode = "chunked" if chunked else "one_shot"
    # The cache spans the whole prompt, rounded up to a whole number of chunks. In one-shot mode the
    # runtime's chunk IS that whole padded sequence; in chunked mode it is the spec's chunk_size, and
    # the trace must span more than one of them for the mode to mean anything.
    padded = _round_up(seq_len, chunk_size)
    runtime_chunk = padded if not chunked else chunk_size
    if chunked:
        assert seq_len > chunk_size, (
            f"the selected trace has {seq_len} tokens, which is not more than chunk_size {chunk_size}; "
            f"a single-chunk 'chunked' run does not exercise the cache-read path"
        )

    # --- dimensions, asserted against the CHECKPOINT's config, not against the spec ----------
    cfg = LlamaConfig.from_json(Path(checkpoint) / "config.json")
    assert cfg.num_hidden_layers == meta["num_layers"], "trace depth differs from the checkpoint's"
    assert cfg.num_key_value_heads == meta["num_kv_heads"] and cfg.head_dim == meta["head_dim"]

    logger.info(
        f"acceptance [{mode}, {topology_name}]: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
        f"{seq_len} tokens (padded to {padded}), chunk {runtime_chunk}, sp={sp} tp={tp}, "
        f"weights {checkpoint}, trace {trace_dir}"
    )

    state_dict = load_state_dict(checkpoint)
    runtime = TtPrefillRuntime(
        mesh_device,
        cfg,
        state_dict,
        PrefillRuntimeConfig(
            num_layers=cfg.num_hidden_layers,
            max_seq_len=padded,
            chunk_size=runtime_chunk,
            mesh_shape=tuple(mesh_device.shape),
            cache_dtype=cache_dtype,
            weight_dtype=weight_dtype,
            weight_cache_path=str(weight_cache_dir(mesh_device.shape, weight_dtype)),
            topology=ttnn.Topology.Ring if topology_name == "torus" else ttnn.Topology.Linear,
        ),
    )
    del state_dict  # the device holds the weights now; 16 GiB of host copies is not needed past here

    kv_cache = runtime.allocate_kv_cache()
    t0 = time.perf_counter()
    runtime.prefill_sequence(token_ids, kv_cache)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - t0
    logger.info(f"prefill [{mode}] {seq_len} tokens in {elapsed:.1f} s ({seq_len / elapsed:.1f} tok/s)")

    # --- the graded comparison --------------------------------------------------------------
    logger.info(f"per-layer K/V vs {trace_dir} over {seq_len} tokens:")
    rows = per_layer_kv_pcc(runtime, kv_cache, trace_dir=trace_dir, n_tokens=seq_len)

    assert len(rows) == cfg.num_hidden_layers, f"expected one row per layer, got {len(rows)}"
    assert [r["layer"] for r in rows] == list(range(cfg.num_hidden_layers)), "layer_pcc must be in order from 0"

    min_k = min(r["k"] for r in rows)
    min_v = min(r["v"] for r in rows)
    below_target = [r["layer"] for r in rows if min(r["k"], r["v"]) < target]
    logger.info(
        f"[{mode}] min K {min_k:.6f}, min V {min_v:.6f} across {len(rows)} layers "
        f"(lower bound {lower}, target {target}); {len(below_target)} layers below target"
    )

    for r in rows:
        for name in ("k", "v"):
            value = r[name]
            assert value == value, f"layer {r['layer']} {name.upper()} PCC is NaN"
            assert value >= lower, f"layer {r['layer']} {name.upper()} PCC {value:.6f} < pcc_lower_bound {lower}"

    # --- the report, written from the measured run ONLY, after the asserts -------------------
    out_path = os.environ.get("PREFILL_ACCEPTANCE_OUT")
    if out_path:
        import sys

        report = {
            "mode": mode,
            "python": sys.executable,
            "num_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "seq_len": seq_len,
            "chunk_size": chunk_size,
            "sp": sp,
            "tp": tp,
            "target_hw": spec["target_hw"],
            "layer_pcc": [{"layer": r["layer"], "k": r["k"], "v": r["v"]} for r in rows],
            # Provenance and context, beyond the required contract.
            "topology": topology_name,
            "runtime_chunk_size": runtime_chunk,
            "padded_seq_len": padded,
            "checkpoint": str(checkpoint),
            "trace_dir": str(trace_dir),
            "kv_cache_dtype": spec["dataformats"]["kv_cache"]["default"],
            "weight_dtype": spec["dataformats"]["weights"]["default"],
            "rope_freq_fp16": os.getenv("LLAMA_ROPE_FREQ_FP16", "0") in ("1", "true", "yes", "on"),
            "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(seq_len / elapsed, 1),
            "pcc_lower_bound": lower,
            "pcc_target": target,
            "min_k": min_k,
            "min_v": min_v,
            "layers_below_target": below_target,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        logger.info(f"wrote acceptance report to {out_path}")
