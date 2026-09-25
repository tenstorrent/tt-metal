# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic (not collected by the suite; run the file explicitly): per-layer KV PCC vs the golden trace
for the first ``MISTRAL_DIAG_LAYERS`` real layers, under the current ``MISTRAL_PRECISION`` knobs.
REDUCED DEPTH by design: it attributes the depth-wise KV decay to a knob, it is not the graded result.

  MISTRAL_DIAG_LAYERS=48 MISTRAL_PRECISION=sdpa_fp32_acc MISTRAL_DIAG_OUT=/tmp/d.json \\
    scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests/diag_kv_depth.py

``MISTRAL_DIAG_BF16_WEIGHTS=1`` builds attention + MLP weights in bf16 from the checkpoint (attribution
only: the spec binds bf8; bf16 weights for more than ~44 layers do not fit a chip's DRAM).
"""

import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.mistral_medium_3_5_128b.config import (
    MistralMediumConfig,
    load_prefill_spec,
    resolve_dataformats,
    ttnn_dtype,
)
from models.demos.mistral_medium_3_5_128b.tt.kv_validation import layer_kv_pcc, trace_token_ids
from models.demos.mistral_medium_3_5_128b.tt.precision import precision
from models.demos.mistral_medium_3_5_128b.tt.runtime import PrefillRuntime, PrefillRuntimeConfig
from models.demos.mistral_medium_3_5_128b.tt.weights import CacheOnlyWeights, CheckpointWeights, weight_cache_dir


@pytest.mark.timeout(7200)
def test_kv_depth_diagnostic(galaxy_mesh):
    ckpt = os.environ.get("PREFILL_HF_MODEL") or os.environ["HF_MODEL"]
    trace = os.environ["PREFILL_TRACE_DIR"]
    spec = load_prefill_spec()
    cfg = MistralMediumConfig.from_json(Path(ckpt) / "config.json")
    n_layers = int(os.environ.get("MISTRAL_DIAG_LAYERS", "48"))
    chunked = os.environ.get("PREFILL_CHUNKED", "0") == "1"
    bf16_weights = os.environ.get("MISTRAL_DIAG_BF16_WEIGHTS") == "1"
    token_ids, _ = trace_token_ids(trace)
    n = len(token_ids)
    sp = galaxy_mesh.shape[0]
    chunk = spec["shapes"]["chunk_size"] if chunked else PrefillRuntime.pad_to(n, 32 * sp)
    capacity = PrefillRuntime.pad_to(n, chunk)

    dtypes = {k: ttnn_dtype(v) for k, v in resolve_dataformats(spec).items()}
    if bf16_weights:
        for k in ("attention", "mlp_gate", "mlp_up", "mlp_down"):
            dtypes[k] = ttnn.bfloat16
        weights, cache_dir = CheckpointWeights(ckpt), None
    else:
        weights, cache_dir = CacheOnlyWeights(), str(weight_cache_dir(ckpt, tuple(galaxy_mesh.shape)))
    runtime = PrefillRuntime(
        galaxy_mesh,
        cfg,
        weights,
        PrefillRuntimeConfig(
            num_layers=n_layers,
            max_seq_len=capacity,
            chunk_size=chunk,
            dtypes=dtypes,
            cache_dtype=dtypes["kv_cache"],
            weight_cache_path=cache_dir,
        ),
    )
    kv = runtime.allocate_kv_cache()
    t0 = time.perf_counter()
    runtime.prefill(token_ids, kv)
    elapsed = time.perf_counter() - t0
    k_blk, v_blk = runtime.read_slot_kv(kv)
    rows = layer_kv_pcc(
        k_blk, v_blk, trace, n_tokens=n, sp=sp, chunk_size=chunk, max_seq_len=capacity, head_dim=cfg.head_dim
    )
    p = precision()
    tag = f"precision={p} bf16_weights={bf16_weights} chunked={chunked} layers={n_layers}"
    for r in rows:
        logger.info(f"[diag] layer {r['layer']:2d}: K={r['k']:.6f} V={r['v']:.6f}")
    logger.info(
        f"[diag] {tag}: worst K {min(r['k'] for r in rows):.6f} V {min(r['v'] for r in rows):.6f} ({elapsed:.1f}s)"
    )
    out = os.environ.get("MISTRAL_DIAG_OUT")
    if out:
        Path(out).write_text(json.dumps({"tag": tag, "layer_pcc": rows}, indent=1) + "\n")
