# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Where the time goes in chunked prefill of the MTP lanes adapter (``QWEN38_MAMBA_SPEC=1``), on device, in-process.

A ~20,000-token prompt as one call (case U, S = 0, checkpoint at the last full block) against 2048-token calls (case C,
each restoring the state saved by the call before, the last call with the same checkpoint).  Every ``prefill_forward``
call is timed with a device synchronize at both ends, and split by wrapping the real methods (the wrappers call
through): restore (``restore_prefix``), chunk prefill (``_prefill_chunked``), forced step (``_prefill``), save
(``save_prefix_state``), page table (``write_page_table``), lane admission (``admit_from_chain``).  The pass forms of
each chunk prefill (slab / 128-row / 32-row) come from the driver's own ``_cover``.  Timing only: no token gate.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_vllm_mtp_prefix_perf_device.py -s
"""

import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_mtp_prefix_device as tpl
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import needle_prompts
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as pdriver
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38MTPForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

PROMPT_LENGTH = 20000
CHUNK = 2048
tpl.REQUEST_KV_BLOCKS = 21  # 20,000 tokens need 20 blocks of 1024 (the template's helper reads this module constant)


def step(name: str) -> None:
    logger.info(f"PC4A step: {name}")


class Probe:
    """Per-call time buckets (inclusive, each bracketed by a device synchronize) and the plan of each chunk prefill."""

    def __init__(self, mesh, chain) -> None:
        self.mesh = mesh
        self.chain = chain
        self.ms = defaultdict(float)
        self.counts = defaultdict(int)
        self.plans: list[dict] = []
        self.timings: list = []

    def reset(self) -> None:
        self.ms.clear()
        self.counts.clear()
        self.plans = []
        self.timings = []

    def sync(self) -> None:
        # the chain's event sync (allowed inside the loop guard, unlike ttnn.synchronize_device)
        self.chain.event_synchronize(self.chain.record_event())

    def wrap(self, owner, name: str, bucket: str) -> None:
        real = getattr(owner, name)
        probe = self

        def wrapper(*args, **kwargs):
            probe.sync()
            t0 = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                probe.sync()
                probe.ms[bucket] += (time.perf_counter() - t0) * 1000
                probe.counts[bucket] += 1

        setattr(owner, name, wrapper)

    def wrap_plan(self) -> None:
        cls = pdriver.Qwen38ChunkPrefill
        real_cover, real_run = cls._cover, cls.run
        probe = self

        def cover(this, count, position):
            specs, used = real_cover(this, count, position)
            kinds = defaultdict(int)
            for s in specs:
                kinds[s.kind] += 1
            padded = [(s.kind, s.rows, s.real) for s in specs if s.padded]
            probe.plans.append(
                {
                    "count": count,
                    "position": position,
                    "plan": used,
                    "slab": kinds["slab"],
                    "long": kinds["long"],
                    "short": kinds["short"],
                    "padded": padded,
                    "slab_rows": this.slab_rows,
                }
            )
            return specs, used

        def run(this, *args, **kwargs):
            result = real_run(this, *args, **kwargs)
            probe.timings.append(result.timing)
            return result

        cls._cover, cls.run = cover, run


def forms_text(plan: dict) -> str:
    return f"slab {plan['slab']}x{plan['slab_rows']} / 128-row {plan['long']} / 32-row {plan['short']}"


def timed_prefill(h: tpl.Harness, probe: Probe, mesh, req, start: int, end: int, final: bool) -> dict:
    probe.reset()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    first = h.prefill([(req, start, end, final)])[0]
    ttnn.synchronize_device(mesh)
    total = (time.perf_counter() - t0) * 1000
    row = {
        "S": start,
        "E": end,
        "final": final,
        "first": first,
        "total": total,
        "restore": probe.ms["restore"],
        "chunk": probe.ms["chunk"],
        "forced": probe.ms["forced"],
        "save": probe.ms["save"],
        "page": probe.ms["page"],
        "admit": probe.ms["admit"],
        "plans": list(probe.plans),
        "timings": list(probe.timings),
        "n_save": probe.counts["save"],
        "n_forced": probe.counts["forced"],
        "n_chunk": probe.counts["chunk"],
    }
    row["other"] = total - sum(row[k] for k in ("restore", "chunk", "forced", "save", "page", "admit"))
    return row


def print_rows(title: str, rows: list[dict]) -> None:
    logger.info(f"PC4A TABLE {title}")
    logger.info(
        "PC4A  S      E      total  restore chunk   forced  save    page    admit   other   (ms)  | chunk forms (slab/128/32) align rows"
    )
    for r in rows:
        plan = r["plans"][0] if r["plans"] else None
        forms = forms_text(plan) if plan else "no chunk prefill"
        extra = ""
        if r["timings"]:
            t = r["timings"][0]
            extra = (
                f" align {t.alignment_steps}, tail {t.tail_rows}, wall {t.wall_ms:.0f} ms, handoff {t.handoff_ms:.0f} ms, "
                f"plan {t.plan}, padded {t.padded_pass}/{t.padded_real_rows}, chunk-count {len(r['plans'])}"
            )
        logger.info(
            f"PC4A {r['S']:<6} {r['E']:<6} {r['total']:<6.0f} {r['restore']:<7.0f} {r['chunk']:<7.0f} {r['forced']:<7.0f} "
            f"{r['save']:<7.0f} {r['page']:<7.0f} {r['admit']:<7.0f} {r['other']:<7.0f} | {forms};{extra}"
        )
    total = sum(r["total"] for r in rows)
    if not rows:
        return
    logger.info(
        f"PC4A {title}: total {total:.0f} ms for {PROMPT_LENGTH} tokens = {PROMPT_LENGTH / total * 1000:.0f} tok/s"
    )
    for key in ("restore", "chunk", "forced", "save", "page", "admit", "other"):
        logger.info(f"PC4A   {title} {key}: {sum(r[key] for r in rows):.0f} ms")


def run_case(h, probe, mesh, req, name: str, checkpoint_at: int, bounds: list[tuple[int, int, bool]]) -> list[dict]:
    rows = []
    try:
        for start, end, final in bounds:
            if final and not req.checkpoints:  # the checkpoint rides only the call that crosses it
                req.checkpoints[checkpoint_at] = h.blocks.take_state()
            rows.append(timed_prefill(h, probe, mesh, req, start, end, final))
            r = rows[-1]
            logger.info(
                f"PC4A {name} call S={start} E={end}: {r['total']:.0f} ms, chunk {r['chunk']:.0f}, plans {[forms_text(p) for p in r['plans']]}"
            )
    except BaseException as error:  # noqa: BLE001
        logger.error(f"PC4A {name} failed at call {len(rows)}: {type(error).__name__}: {error}")
        print_rows(name + " (partial)", rows)
        raise
    return rows


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_chunked_prefill_perf(mesh_device, monkeypatch):
    checkpoint: Path = checkpoint_directory()
    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")
    if not os.environ.get("QWEN38_VLLM_MTP_K"):
        monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "1")
    mesh_device.enable_program_cache()

    step("prompts")
    prompts, _, _ = needle_prompts(checkpoint, 1, PROMPT_LENGTH)
    prompt = prompts[0]
    length = len(prompt)
    logger.info(f"PC4A prompt length {length}")
    allocated = tpl.ALLOCATED
    step("initialize_vllm_model")
    model = Qwen38MTPForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)),
        mesh_device,
        max_batch_size=tpl.LANES,
        max_seq_len=allocated - RESIDENT_CONTEXT_HEADROOM,
    )
    assert model.lanes_mode and model.lanes == tpl.LANES and model.drafts == 3
    try:
        step("allocate_kv_cache_per_layer")
        model.kv_cache = model.allocate_kv_cache_per_layer(tpl.kv_entries())
        width = -(-(allocated - RESIDENT_CONTEXT_HEADROOM) // tpl.BLOCK)
        prefill_kwargs = dict(kv_cache=None, can_sample_on_device=False)
        decode_kwargs = dict(kv_cache=None, max_batch_size=tpl.LANES, num_blocks=width, can_sample_on_device=True)
        step("warmup_model_prefill eager")
        model.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
        step("warmup_model_decode eager")
        model.warmup_model_decode(enable_trace=False, **decode_kwargs)
        if hasattr(model, "already_warmed_up_prefill"):
            model.already_warmed_up_prefill = False
        step("warmup_model_prefill traced")
        model.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
        step("warmup_model_decode traced")
        model.warmup_model_decode(enable_trace=True, **decode_kwargs)
        step("warm-ups done")

        h = tpl.Harness(model)
        model.mtp_hybrid_single = False
        probe = Probe(mesh_device, model.driver.chain)
        probe.wrap_plan()
        probe.wrap(model.driver.session, "restore_prefix", "restore")
        probe.wrap(model.driver.session, "_prefill_chunked", "chunk")
        probe.wrap(model.driver.session, "_prefill", "forced")
        probe.wrap(model.driver.chain, "save_prefix_state", "save")
        probe.wrap(model.driver.chain, "write_page_table", "page")
        probe.wrap(model.lane_driver, "admit_from_chain", "admit")

        checkpoint_at = (length // tpl.BLOCK) * tpl.BLOCK
        if checkpoint_at == length:
            checkpoint_at -= tpl.BLOCK
        starts = list(range(0, length, CHUNK))
        c_bounds = [(s, min(s + CHUNK, length), min(s + CHUNK, length) == length) for s in starts]
        logger.info(f"PC4A checkpoint at {checkpoint_at}; C calls {[(s, e) for s, e, _ in c_bounds]}")
        results = {}
        for rep in range(2):
            for case in ("U", "C"):
                step(f"case {case} rep {rep}")
                req = h.new_req(f"{case}{rep}", 0, prompt, checkpoints=[checkpoint_at] if case == "U" else [])
                bounds = [(0, length, True)] if case == "U" else c_bounds
                try:
                    results[(case, rep)] = run_case(h, probe, mesh_device, req, f"{case}{rep}", checkpoint_at, bounds)
                finally:
                    model.release_request(req.slot)
                    h.free(req)
                print_rows(f"{case} rep {rep}", results[(case, rep)])
        u = results[("U", 1)][0]
        logger.info(
            f"PC4A first tokens: U0 {results[('U', 0)][0]['first']}, U1 {u['first']}, "
            f"C0 {results[('C', 0)][-1]['first']}, C1 {results[('C', 1)][-1]['first']}"
        )
        u_tok = u["chunk"] / length * 1000
        logger.info(
            f"PC4A reference: U chunk-prefill {u['chunk']:.0f} ms = {u_tok:.4f} ms/token = {1000 / u_tok:.0f} tok/s; "
            f"U total {u['total']:.0f} ms; C total {sum(r['total'] for r in results[('C', 1)]):.0f} ms"
        )
        for r in results[("C", 1)]:
            n = r["E"] - r["S"]
            logger.info(
                f"PC4A C call S={r['S']}: chunk {r['chunk']:.0f} ms for {n} rows = {r['chunk'] / n * 1000:.1f} us/row "
                f"(U: {u['chunk'] / length * 1e6:.1f} us/row)"
            )
    finally:
        step("release_persistent_capture")
        model.release_persistent_capture()
        step("done")
