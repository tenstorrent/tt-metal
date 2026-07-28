# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pin GEMMA4_LONG_CONTEXT_POLICY cutovers used by text_demo / text_demo_v2 / vLLM.

Expectations come from isl_sweep_logs (full_matrix + QB2 coherent bounded path):
  * P150x8 31B @ 128k unbounded+4096 → "lapped…" collapse
  * QB2 / P150x8 31B @ 128k bounded+2048 → coherent
  * P150x8 12B @ 128k unbounded+4096 → coherent
  * P150x8 31B @ 64k unbounded+4096 → coherent (keep for TTFT/perf)
"""

from __future__ import annotations

import pytest

from models.demos.gemma4.tt.generator_trace import normalize_gemma4_model_key, resolve_gemma4_demo_long_context

# (mesh, hf_model, max_seq_len) → (bounded, prefill_chunk)
_EXPECTED = [
    # ── 31B / LoudBox ─────────────────────────────────────────────────────
    ("P150x8", "google/gemma-4-31B-it", 4096, False, 4096),
    ("P150x8", "google/gemma-4-31B-it", 32768, False, 4096),
    ("P150x8", "google/gemma-4-31B-it", 65536, False, 4096),
    ("P150x8", "google/gemma-4-31B-it", 131072, True, 2048),
    ("P150x8", "google/gemma-4-31B-it", 262144, True, 2048),
    # ── 31B / QB2 ─────────────────────────────────────────────────────────
    ("P150x4", "google/gemma-4-31B-it", 32768, False, 4096),
    ("P150x4", "google/gemma-4-31B-it", 65536, True, 4096),
    ("P150x4", "google/gemma-4-31B-it", 131072, True, 2048),
    ("P300x2", "google/gemma-4-31B-it", 131072, True, 2048),
    # ── 31B / T3K (WH 12 GB/ASIC: multi-chunk 2048; bound from 32k) ───────
    ("T3K", "google/gemma-4-31B-it", 8192, False, 2048),
    ("T3K", "google/gemma-4-31B-it", 16384, False, 2048),
    ("T3K", "google/gemma-4-31B-it", 32768, True, 2048),
    ("T3K", "google/gemma-4-31B-it", 131072, True, 2048),
    # ── 12B: stay unbounded through 128k on multi-chip BH for perf ────────
    ("P150x8", "google/gemma-4-12B-it", 131072, False, 4096),
    ("P150x8", "google/gemma-4-12B-it", 262144, False, 4096),
    ("P150x4", "google/gemma-4-12B-it", 131072, False, 4096),
    ("P150x4", "google/gemma-4-12B-it", 262144, True, 4096),
    ("P150", "google/gemma-4-12B-it", 32768, False, 4096),
    ("P150", "google/gemma-4-12B-it", 65536, True, 4096),
    ("P150", "google/gemma-4-12B-it", 131072, True, 4096),
    ("P150", "google/gemma-4-12B-it", 262144, True, 4096),  # full HF ISL
    # ── 26B-A4B / LoudBox: same 128k coherency cutover as 31B ─────────────
    ("P150x8", "google/gemma-4-26B-A4B-it", 65536, False, 4096),
    ("P150x8", "google/gemma-4-26B-A4B-it", 131072, True, 2048),
    ("P150x4", "google/gemma-4-26B-A4B-it", 131072, False, 4096),
    # ── E2B/E4B: unbounded through measured 256k on P150 / LB / QB2 ───────
    ("P150", "google/gemma-4-E2B-it", 262144, False, 4096),
    ("P150", "google/gemma-4-E4B-it", 262144, False, 4096),
    ("P150x8", "google/gemma-4-E4B-it", 262144, False, 4096),
    ("P150x4", "google/gemma-4-E2B-it", 262144, False, 4096),
]


@pytest.mark.parametrize(
    "mesh,model,max_seq_len,exp_bounded,exp_chunk",
    _EXPECTED,
    ids=[f"{x[0]}-{x[1].rsplit('/', 1)[-1]}-{x[2]}" for x in _EXPECTED],
)
def test_long_context_policy_defaults(mesh, model, max_seq_len, exp_bounded, exp_chunk, monkeypatch):
    monkeypatch.delenv("GEMMA4_BOUNDED_SLIDING", raising=False)
    monkeypatch.delenv("GEMMA4_GEN_PREFILL_CHUNK", raising=False)
    monkeypatch.delenv("GEMMA4_DEMO_SINGLE_CHUNK", raising=False)
    monkeypatch.setenv("MESH_DEVICE", mesh)

    lc = resolve_gemma4_demo_long_context(max_seq_len, mesh_device=None, model_name_or_path=model)
    assert (
        lc["bounded_sliding"] is exp_bounded
    ), f"{mesh}/{model}@{max_seq_len}: bounded={lc['bounded_sliding']} want {exp_bounded}"
    assert (
        lc["prefill_chunk"] == exp_chunk
    ), f"{mesh}/{model}@{max_seq_len}: chunk={lc['prefill_chunk']} want {exp_chunk}"


def test_env_override_bounded_and_chunk(monkeypatch):
    monkeypatch.setenv("MESH_DEVICE", "P150x8")
    monkeypatch.setenv("GEMMA4_BOUNDED_SLIDING", "0")
    monkeypatch.setenv("GEMMA4_GEN_PREFILL_CHUNK", "4096")
    lc = resolve_gemma4_demo_long_context(131072, None, "google/gemma-4-31B-it")
    assert lc["bounded_sliding"] is False
    assert lc["prefill_chunk"] == 4096


def test_mesh_device_alias_p300x2(monkeypatch):
    monkeypatch.delenv("GEMMA4_BOUNDED_SLIDING", raising=False)
    monkeypatch.delenv("GEMMA4_GEN_PREFILL_CHUNK", raising=False)
    monkeypatch.setenv("MESH_DEVICE", "P300x2")
    lc = resolve_gemma4_demo_long_context(131072, None, "google/gemma-4-31B-it")
    assert lc["bounded_sliding"] is True
    assert lc["prefill_chunk"] == 2048


@pytest.mark.parametrize(
    "path,key",
    [
        ("google/gemma-4-31B-it", "31B"),
        (
            "/mnt/MLPerf/huggingface/hub/models--google--gemma-4-31B-it/snapshots/ba74f5b6c647c0911554e50278d6f6f4477f9010",
            "31B",
        ),
        (
            "/mnt/MLPerf/huggingface/hub/models--google--gemma-4-12B-it/snapshots/deadbeef",
            "12B",
        ),
    ],
)
def test_normalize_gemma4_model_key_snapshot_paths(path, key):
    assert normalize_gemma4_model_key(path) == key
