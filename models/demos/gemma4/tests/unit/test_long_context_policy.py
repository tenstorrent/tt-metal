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

from models.demos.gemma4.tt.generator_trace import (
    get_gemma4_long_context_policy,
    normalize_gemma4_model_key,
    resolve_gemma4_demo_long_context,
)

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
    # Unbounded ceiling re-measured at 32768; bounded now starts at 65536 so the
    # multi-request sliding remap stays out of the served range.
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
    # ── Wormhole, measured on a real T3K / N300 (12 GB per ASIC) ──────────
    # Bounded sliding is deliberately kept OUT of the WH serving range: its
    # sliding page-table remap is keyed on the current tensor's row index rather
    # than the request's persistent KV slot, which corrupts any multi-request
    # batch (see generator_trace 12B/T3K comment). These cases pin that
    # bounded_sliding stays False across the whole served context.
    # 12B/T3K: unbounded measured through 131072; bounded only at the full 256k.
    ("T3K", "google/gemma-4-12B-it", 8192, False, 2048),
    ("T3K", "google/gemma-4-12B-it", 16384, False, 2048),
    ("T3K", "google/gemma-4-12B-it", 131072, False, 2048),
    ("T3K", "google/gemma-4-12B-it", 262144, True, 2048),
    # 12B/N300 (24 GB, the only variant that fits one WH card): unbounded to 8k,
    # bounded above — so concurrent serving stays at/below 8192 there.
    ("N300", "google/gemma-4-12B-it", 8192, False, 2048),
    ("N300", "google/gemma-4-12B-it", 16384, True, 2048),
    ("N300", "google/gemma-4-12B-it", 131072, True, 2048),
    # 26B-A4B/T3K and 31B/T3K: unbounded measured through 32768.
    ("T3K", "google/gemma-4-26B-A4B-it", 8192, False, 2048),
    ("T3K", "google/gemma-4-26B-A4B-it", 32768, False, 2048),
    ("T3K", "google/gemma-4-26B-A4B-it", 65536, True, 2048),
    ("T3K", "google/gemma-4-31B-it", 32768, False, 2048),
    ("T3K", "google/gemma-4-31B-it", 65536, True, 2048),
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


@pytest.mark.parametrize("mesh", ["N300", "T3K", "N150", "N150x4"])
@pytest.mark.parametrize(
    "model",
    [
        "google/gemma-4-12B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
    ],
)
def test_wormhole_never_inherits_blackhole_policy(mesh, model, monkeypatch):
    """A WH board must never resolve to a Blackhole (QB2 / P150x*) entry.

    Regression for the silent cross-arch fallback: every (model, device) combo
    missing from the table used to fall back to the QB2 entry, handing a 24 GB
    N300 / 96 GB T3K Blackhole's 128 GB headroom ("unbounded KV through 128k")
    and OOMing on KV allocation. Wormhole must bound early and keep chunk 2048.

    Force the WH host view: ``_canonical_device_name`` only treats ``N150`` as a
    real WH board when the host is Wormhole (on Blackhole, ``N150`` is a
    historical P150 alias). Without this patch the N150 cases pass only on WH
    runners and falsely fail on BH.
    """
    monkeypatch.setattr(
        "models.demos.gemma4.tt.generator_trace._host_is_wormhole",
        lambda: True,
    )
    monkeypatch.delenv("GEMMA4_BOUNDED_SLIDING", raising=False)
    monkeypatch.delenv("GEMMA4_GEN_PREFILL_CHUNK", raising=False)
    monkeypatch.delenv("GEMMA4_DEMO_SINGLE_CHUNK", raising=False)
    monkeypatch.setenv("MESH_DEVICE", mesh)

    policy = get_gemma4_long_context_policy(None, model)
    assert "device_fallback" not in policy["source"], (
        f"{mesh}/{model} fell back to the Blackhole QB2 entry (source={policy['source']}); "
        "WH has 12 GB per ASIC and cannot inherit Blackhole DRAM headroom."
    )
    # Blackhole gemma4 entries all use chunk 4096; every WH entry must stay at
    # the 2048 chunk the WH boards were measured with. (unbounded_isl_max is no
    # longer a useful discriminator: WH T3K is now measured unbounded through
    # 131072 for 12B, which legitimately exceeds some Blackhole entries.)
    assert policy["prefill_chunk"] == 2048, f"{mesh}/{model}: {policy}"


def test_blackhole_qb2_fallback_still_applies(monkeypatch):
    """The QB2 fallback must stay intact for Blackhole boards (no regression)."""
    monkeypatch.setenv("MESH_DEVICE", "P150")
    policy = get_gemma4_long_context_policy(None, "google/gemma-4-31B-it")
    assert policy["source"].endswith("_device_fallback")
    assert policy["prefill_chunk"] == 4096
