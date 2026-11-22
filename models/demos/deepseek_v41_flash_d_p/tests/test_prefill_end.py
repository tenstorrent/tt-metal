# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PREFILL_END (P.49, owner 2026-11-12): prefill stops at decode's cap; every served byte below the cap is the same.

The serving runtime (tt/runners/adapter.py, built and compiled as the engine does: caches.build, the spec's served
layers, 2 slots, chunk / max seq of the golden rung) gets each request the way tt-d-gen 48b79127fb1 hands a PD_DISAGG
prefill its prompt: actual_end = prompt_compute_end = prompt_len - 1 (types.hpp:101-110), chunks [k chunk,
min((k + 1) chunk, actual_end)) (prefill_writer.cpp:125-131), the H2D payload with the PAD_ID tail (server_rules).
Decode's cap = reusable_prefix_cap(64, prompt_len) = (prompt_len - 1) // 64 x 64 (prefix_indexer.hpp:78-81); the model
computes it from actual_end as actual_end // 64 x 64 (tt/attention.snapshot_cap; DS41-219).

Every request runs twice, PREFILL_END=end into slot 0 (P.42's path: every position up to actual_end) and
PREFILL_END=cap into slot 1 (only [0, cap)), both prompts (novel: the spec's prompt; code: tt-metal C++, testing/b2b)
interleaved, so a record the cap run failed to write still holds the other prompt's bytes. Through the exported table
over UMD, as the KV Manager reads them:
  - served_bytes_differ (== 0 gates): records that differ between the two runs: every layer's 0_window_kv snapshot
    (4 records: rows [cap - 128, cap) at p % 128, zeros elsewhere) and every history config's records of [0, cap)
    (compressed KV, index keys; re-index aliases included);
  - past_cap_written: history records of [cap, actual_end) the cap run changed in slot 1 (must be 0: nothing at or
    past cap is written); end_past_written: those the end run changed in slot 0 (> 0 whenever such records exist:
    the two sides really differ);
  - acks: one per served layer per chunk, in order, naming the engine's (slot, actual_start, actual_end), also on a
    chunk with nothing below cap (computes nothing, rewrites each layer's snapshot at cap, acks).
Prompt lengths (prompt_len % 64; chunks of 5120 on galaxy_8x4): 191 (63, shorter than 128 + 64), 4097 (1, one chunk,
cap = actual_end), 4098 (2, one chunk), 5122 (2, two chunks, the last one entirely past cap), 7231 (63, two chunks),
10176 (0, two chunks). Scaled to the rung's chunk on other boxes (lengths past max seq dropped).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_prefill_end.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import b2b
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.prefill.runners.layer_completion_ring_writer import LayerCompletionSink
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)
PROMPTS = ("novel", "code")
SLOT = {"end": 0, "cap": 1}


def prompt_lens(chunk: int, max_seq: int) -> list[int]:
    """The prompt lengths above at the rung's chunk (5120 on galaxy_8x4); only those whose actual_end fits max seq."""
    lens = [191, chunk - 1023, chunk - 1022, chunk + 2, chunk + 2111, 2 * chunk - 64]
    return [n for n in lens if R.prompt_compute_end(n) <= max_seq]


def _set(monkeypatch, mode: str) -> None:
    monkeypatch.setenv(settings.env_name("PREFILL_END"), mode)
    settings.refresh()
    assert settings.get("PREFILL_END") == mode


@mesh_parametrize(timeout_s=2 * 3600)
def test_prefill_end(mesh_device, tmp_path, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    R.self_check()  # (P.49: the cap from actual_end == decode's reusable_prefix_cap(prompt_len), every length)
    layers = C.served_prefix()
    cfgs = C.served_configs(layers)
    chunk, max_seq = R.CHUNK, R.MAX_SEQ
    lens = prompt_lens(chunk, max_seq)
    tokens = {wl: torch.tensor(b2b.ids(S, wl, max(lens))).numpy() for wl in PROMPTS}
    try:
        _, runtime, kv = C.build(mesh_device, layers, 2)
    except C.NotBuilt as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    path = C.export_table(runtime, kv, mesh_device, str(tmp_path / "table.pb"), len(layers))
    reader = C.TableReader(path, C.device_map(mesh_device, str(tmp_path / "device_map.json")))
    hist = [c for c in cfgs if not c.get("snapshot")]
    snaps = [c for c in cfgs if c.get("snapshot")]

    acks = []

    class Sink(LayerCompletionSink):
        def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end):
            for layer in range(layer_start, layer_end):
                acks.append((int(layer), (int(slot_id), int(pos_start), int(pos_end))))

    runtime.set_layer_completion_sink(Sink())

    def run(wl: str, n: int, mode: str) -> list[str]:
        """One request of prompt_len n (actual_end n - 1) in mode's slot; its ack failures."""
        _set(monkeypatch, mode)
        slot, end, out = SLOT[mode], R.prompt_compute_end(n), []
        k = 0
        while k * chunk < end:
            lo, hi = k * chunk, min((k + 1) * chunk, end)
            inp = C.payload_tensor(mesh_device, R.server_payload(tokens[wl][lo : lo + chunk], lo, hi))
            n0 = len(acks)
            runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=lo, actual_end=hi, request_id=k)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(inp)
            want = [(i, (slot, lo, hi)) for i in range(len(layers))]
            if acks[n0:] != want:
                out.append(
                    f"{wl} {n} {mode} chunk [{lo}, {hi}): acks {acks[n0:][:4]}.. want one per layer {want[:2]}.."
                )
            k += 1
        return out

    def served(slot: int, cap: int) -> dict:
        got = {}
        for c in snaps:
            got.update({(c["name"],) + key: b for key, b in reader.records(c["name"], slot, c["layers"], 0, 0).items()})
        if cap > 0:
            for c in hist:
                recs = reader.records(c["name"], slot, c["layers"], 0, cap)
                got.update({(c["name"],) + key: b for key, b in recs.items()})
        return got

    def past(slot: int, cap: int, end: int) -> dict:
        got = {}
        if R.ceil_to(end, 32) > cap:
            for c in hist:
                recs = reader.records(c["name"], slot, c["layers"], cap, end)
                got.update({(c["name"],) + key: b for key, b in recs.items()})
        return got

    fails, cases, total_differ, total_past, total_recs = [], [], 0, 0, 0
    try:
        for n in lens:
            for wl in PROMPTS:
                end = R.prompt_compute_end(n)
                cap = C.snapshot_cap(end)
                assert cap == R.reusable_prefix_cap(R.KV_BLOCK, n), (n, end, cap)  # decode's cap (DS41-219)
                end_before = past(SLOT["end"], cap, end)
                fails += run(wl, n, "end")
                end_after = past(SLOT["end"], cap, end)
                ref = served(SLOT["end"], cap)
                before = past(SLOT["cap"], cap, end)
                fails += run(wl, n, "cap")
                got = served(SLOT["cap"], cap)
                after = past(SLOT["cap"], cap, end)
                differ = sorted(k for k in ref if got.get(k) != ref[k])
                written = sorted(k for k in before if after.get(k) != before[k])
                end_written = sum(end_after.get(k) != b for k, b in end_before.items())
                total_differ += len(differ)
                total_past += len(written)
                total_recs += len(ref)
                case = dict(
                    prompt=wl,
                    prompt_len=n,
                    mod64=n % 64,
                    actual_end=end,
                    cap=cap,
                    chunks=-(-end // chunk),
                    records=len(ref),
                    differ=len(differ),
                    past_cap_records=len(before),
                    past_cap_written=len(written),
                    end_past_written=end_written,
                )
                cases.append(case)
                logger.info(f"PREFILL_END {case}" + (f" first differing {differ[:6]}" if differ else ""))
                if differ:
                    fails.append(
                        f"{wl} prompt_len {n} cap {cap}: {len(differ)} of {len(ref)} records differ {differ[:6]}"
                    )
                if end_before and not end_written:  # (the two sides really differ: end computes past cap)
                    fails.append(f"{wl} prompt_len {n} cap {cap}: the end run wrote nothing in [cap, actual_end)")
                if written:
                    fails.append(f"{wl} prompt_len {n} cap {cap}: the cap run wrote past cap: {written[:6]}")
    finally:
        _set(monkeypatch, "cap")
    if reader.replica_mismatch:
        fails.append(
            f"device-group replicas differ at {len(reader.replica_mismatch)} records: {reader.replica_mismatch[:3]}"
        )
    metrics.record("served_bytes_differ", total_differ)
    metrics.record("served_records_compared", total_recs)
    metrics.record("past_cap_written", total_past)
    metrics.record("prefill_end_cases", cases)
    print(f"[prefill_end] layers {layers[0]}..{layers[-1]} ({len(layers)}), configs {[c['name'] for c in cfgs]}")
    for c in cases:
        print(f"[prefill_end] {c}")
    assert not fails, "\n".join(fails[:40])
