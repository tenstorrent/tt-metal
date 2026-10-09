# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill latency per decoder-layer kind and per module, batch 1, stage-1 precision policy.

Method (eager, no trace): build the layer once, upload the real-prompt input prefix once, run
``WARMUP`` untimed passes (lazy weight upload, program compile, program cache), then ``TIMED``
passes. Each timed pass is ``ttnn.synchronize_device`` -> ``perf_counter`` -> forward ->
``ttnn.synchronize_device`` -> ``perf_counter``, so a sample is host dispatch + device execution
of one full pass. The reported value is the median.

Targets:
- ``layer``: ``PplxDecoderLayer.forward`` (both norms, mixer, MLP, residuals, chunk loop).
- ``mixer``: gated attention (``full_attention``) or Gated DeltaNet (``linear_attention``) over the
  whole request via ``mixer_prefill`` (input already normed), including chunking and RoPE slices.
- ``mlp``: the SwiGLU MLP run on each prefill chunk exactly as the decoder does.
- ``embedding``, ``final_norm_readout``: the non-layer pieces of a full prefill.

Results are appended to ``PPLX_DECIDER_PERF_LOG`` (JSON lines) and rendered with
``python models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py``.

Device-profiler capture (one warmed layer between signposts ``PREFILL_START`` / ``PREFILL_END``)::

    python -m tracy -r -p -v -m pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py \
        -k "test_profile_layer and L0_linear and S2048"
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    bf16_round,
    build_optimizations,
    build_rotary,
    build_tt_layer,
    golden_tensor,
    model_args,
    reader,
    to_device,
)
from models.demos.pplx_decider_v1_27b.tt.optimizations import PrecisionPolicy

PERF_SEQ_LENS = [128, 1024, 2048, 4096, 8192]  # the prefill buckets
PROFILE_SEQ_LENS = [128, 2048, 8192]
WARMUP = 2
TIMED = 7
PERF_LOG = Path(
    os.environ.get("PPLX_DECIDER_PERF_LOG", "/local/ttuser/gtobar/artifacts/pplx_decider/perf/prefill_perf.jsonl")
)
LAYERS = {"L0_linear": 0, "L3_full": 3}

try:
    from tracy import signpost
except ImportError:  # profiler build without the Python helper

    def signpost(*_args, **_kwargs):
        return None


def _free(out):
    for t in out if isinstance(out, (list, tuple)) else [out]:
        if isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t)


def time_passes(device, fn, *, warmup=WARMUP, timed=TIMED) -> list[float]:
    """Seconds per pass, warm-up excluded. Host sync: synchronize_device before and after each pass."""
    for _ in range(warmup):
        _free(fn())
        ttnn.synchronize_device(device)
    samples = []
    for _ in range(timed):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        samples.append(time.perf_counter() - start)
        _free(out)
    return samples


def record(target: str, kind: str | None, layer: int | None, seq_len: int, samples: list[float]) -> dict:
    row = dict(
        target=target,
        kind=kind,
        layer=layer,
        seq_len=seq_len,
        batch=1,
        median_ms=statistics.median(samples) * 1e3,
        min_ms=min(samples) * 1e3,
        max_ms=max(samples) * 1e3,
        samples_ms=[s * 1e3 for s in samples],
        warmup=WARMUP,
        timed=len(samples),
        sync="ttnn.synchronize_device before and after each pass; eager (no trace)",
        policy=PrecisionPolicy.default().name,
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PERF_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"PERF {target:<20} {kind or '-':<17} S={seq_len:<5} median {row['median_ms']:.2f} ms")
    return row


def _layer_input(device, layer_idx: int, seq_len: int) -> ttnn.Tensor:
    return to_device(bf16_round(golden_tensor(f"L{layer_idx}_input", seq_len)), device)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("layer_name", list(LAYERS))
def test_layer_and_module_perf(device, layer_name):
    layer_idx = LAYERS[layer_name]
    layer = build_tt_layer(device, layer_idx)
    rotary = build_rotary(device) if layer.kind == "full_attention" else None
    chunk = layer.config.optimizations.prefill_chunk
    for seq_len in PERF_SEQ_LENS:
        x = _layer_input(device, layer_idx, seq_len)
        record("layer", layer.kind, layer_idx, seq_len, time_passes(device, lambda: layer(x, rotary)))

        normed = layer.input_norm(x)
        mixer_name = "attention" if layer.kind == "full_attention" else "gdn"
        record(
            mixer_name, layer.kind, layer_idx, seq_len, time_passes(device, lambda: layer.mixer_prefill(normed, rotary))
        )

        post = layer.post_norm(x)
        chunks = (
            [post] if seq_len <= chunk else [post[:, s : min(s + chunk, seq_len), :] for s in range(0, seq_len, chunk)]
        )
        record("mlp", layer.kind, layer_idx, seq_len, time_passes(device, lambda: [layer.mlp(c) for c in chunks]))
        for t in [x, normed, post, *chunks]:
            if t.is_allocated():
                ttnn.deallocate(t)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_embedding_and_head_perf(device):
    from models.demos.pplx_decider_v1_27b.tt.embedding import PplxEmbedding
    from models.demos.pplx_decider_v1_27b.tt.norm import PplxRMSNorm
    from models.demos.pplx_decider_v1_27b.tt.readout import PplxReadout
    from models.demos.pplx_decider_v1_27b.tt.weight_adapter import (
        build_embedding_weight,
        build_readout_weight,
        zero_centred_norm,
    )

    opts = build_optimizations(device)
    embedding = PplxEmbedding(build_embedding_weight(reader().embedding_weight(), opts.policy), mesh_device=device)
    norm = PplxRMSNorm(
        zero_centred_norm(reader().final_norm_weight(), opts.policy).weight, model_args().rms_norm_eps, opts.norm
    )
    readout = PplxReadout(build_readout_weight(reader().readout_weight(), opts.policy), opts.linear, mesh_device=device)
    for seq_len in PERF_SEQ_LENS:
        ids = ttnn.from_torch(
            golden_tensor("ids", seq_len).to(torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        record("embedding", None, None, seq_len, time_passes(device, lambda: embedding(ids)))
        hidden = to_device(bf16_round(golden_tensor("final_input", seq_len)), device)

        def head():
            last = hidden[:, seq_len - 1 : seq_len, :] if seq_len > 1 else hidden
            return readout(norm(last))  # norm is row-wise: last token first, as the model will do

        record("final_norm_readout", None, None, seq_len, time_passes(device, head))
        ttnn.deallocate(ids)
        ttnn.deallocate(hidden)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", PROFILE_SEQ_LENS, ids=[f"S{s}" for s in PROFILE_SEQ_LENS])
@pytest.mark.parametrize("layer_name", list(LAYERS))
def test_profile_layer(device, layer_name, seq_len):
    """One warmed layer pass between signposts, for Tracy / TT_METAL_DEVICE_PROFILER captures."""
    layer_idx = LAYERS[layer_name]
    layer = build_tt_layer(device, layer_idx)
    rotary = build_rotary(device) if layer.kind == "full_attention" else None
    x = _layer_input(device, layer_idx, seq_len)
    for _ in range(WARMUP):
        _free(layer(x, rotary))
        ttnn.synchronize_device(device)
    signpost("PREFILL_START")
    out = layer(x, rotary)
    ttnn.synchronize_device(device)
    signpost("PREFILL_END")
    _free(out)


def render(log: Path = PERF_LOG) -> str:
    latest = {}
    for line in log.read_text().splitlines():
        r = json.loads(line)
        latest[(r["target"], r["kind"], r["seq_len"])] = r
    seq_lens = sorted({k[2] for k in latest})
    out = ["| target | kind | " + " | ".join(f"S={s}" for s in seq_lens) + " |", "|---|---|" + "---:|" * len(seq_lens)]
    for target, kind in sorted({k[:2] for k in latest}, key=lambda k: (k[0], k[1] or "")):
        cells = [
            f"{latest[(target, kind, s)]['median_ms']:.2f}" if (target, kind, s) in latest else "" for s in seq_lens
        ]
        out.append(f"| {target} | {kind or '-'} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Median ms of the timed passes (warm-up excluded); batch 1; eager; synchronize_device around each pass.")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(Path(sys.argv[1]) if len(sys.argv) > 1 else PERF_LOG))
