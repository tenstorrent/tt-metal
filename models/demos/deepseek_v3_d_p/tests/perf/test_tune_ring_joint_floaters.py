# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tunes the floater schedule of a ring_joint SDPA layout (ring_joint_floater_layout.hpp's kFloaterLayoutSchedules) on
one Kimi MLA shape on the Galaxy 8x4: a coordinate search over the schedule, set per candidate through
SDPAProgramConfig.floater_schedule (ttMLA.sdpa_floater_schedule), without rebuilding. Whether a shape runs floaters
at all is the op's floater_compute_to_transfer; tune on a shape that does (5k chunks at HiFi2 and above).

Each candidate is the fused ring gather + SDPA program of one MLA forward (realtime profiler, the slowest chip, median
of 3 after a warm-up), at a short and a long cached prefix, scored as the mean of its time over the no-floater time at
each prefix. The MLA layers, caches and inputs are built once; a candidate costs a kernel compile and the runs.

Every measurement is appended to generated/floater_tune/<variant>_c<chunk>.jsonl (one JSON object per line) before
the next starts, and a rerun skips what is there, so a crash or a hung schedule costs one candidate: a candidate that
started and never finished is not tried again. Only an error from the forward marks a schedule invalid; a profiler
error stops the run. The result is printed as a kFloaterLayoutSchedules entry, or as "no floaters" when the best
schedule does not beat none by MIN_GAIN.

    scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/perf/test_tune_ring_joint_floaters.py \\
        -m floater_tune -k "k3 and c5k" -s
"""

import json
from pathlib import Path

import pytest
from loguru import logger

import models.demos.deepseek_v3_d_p.tests.test_mla as T
import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin import is_marker_explicitly_selected
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

SDPA_LABEL = "experimental/ccl/ring_attention_all_gather_async + transformer/sdpa"
RING_SIZE = 8
FRAC_ONE = 3072  # kDonationFracOne
# Cached prefixes each candidate is timed at: the 55k production context's final chunk, and a long one.
PREFIXES = (56320, 524288)
# A tuned schedule must beat no floaters by this fraction of its time, or the shape runs without floaters.
MIN_GAIN = 0.02
# A move must beat the best by this fraction of its score to be taken: the runs' noise.
NOISE = 0.003
SAMPLES = 3
LOG_DIR = Path("generated/floater_tune")

# The search's starting points: kFloaterLayoutSchedules' entries (iter0, iter1, frac0, frac1, donate_tail,
# merge0..merge4) for Kimi K3 (24 heads x 2 bands) and K2.7 (16 x 3). Keep them in step with the table.
SEEDS = {
    "kimi_k3": (1, 4, 947, 2765, 0, 4, 4, 4, 4, 4),
    "kimi_k2_7": (1, 3, 1843, 1664, 1, 2, 3, 4, 5, 5),
}

pytestmark = pytest.mark.floater_tune


@pytest.fixture(autouse=True)
def _require_marker(request):
    if not is_marker_explicitly_selected(request.config, "floater_tune"):
        pytest.skip("tuning tool: select it with -m floater_tune")


def _text(schedule):
    return "off" if schedule is None else ",".join(str(v) for v in schedule)


def _neighbours(schedule, frac_step):
    """Single-coordinate moves of a schedule that the layout accepts: each donation iteration, each fraction, the
    donated end, and all merge points together."""
    i0, i1, f0, f1, tail, *merges = schedule
    moves = []
    for d in (-1, 1):
        moves.append((i0 + d, i1, f0, f1, tail, *(max(m, i0 + d) for m in merges)))
        moves.append((i0, i1 + d, f0, f1, tail, *merges))
        moves.append((i0, i1, f0 + d * frac_step, f1, tail, *merges))
        moves.append((i0, i1, f0, f1 + d * frac_step, tail, *merges))
        moves.append((i0, i1, f0, f1, tail, *(m + d for m in merges)))
    moves.append((i0, i1, f0, f1, 1 - tail, *merges))

    def valid(s):
        a, b, fa, fb, _, *ms = s
        return (
            0 <= a < b < RING_SIZE and 0 < fa < FRAC_ONE and 0 < fb < FRAC_ONE and all(a <= m < RING_SIZE for m in ms)
        )

    return [s for s in moves if valid(s)]


def _sdpa_ns(mesh_device, forward):
    """Device time of the fused gather + SDPA program in one forward, on its slowest chip. Records of an earlier
    dispatch can arrive late; runtime ids grow, so the newest such program is this forward's."""
    outputs = []
    try:
        _, records = profile_realtime_program(mesh_device, lambda: outputs.append(forward()), collect_all=True)
    finally:
        for tt_out in outputs:
            ttnn.deallocate(tt_out)
    per_program = {}
    for record in records:
        if record["runtime_id"] and T._op_label(record["kernel_sources"]) == SDPA_LABEL:
            runtime_id = record["runtime_id"]
            per_program[runtime_id] = max(per_program.get(runtime_id, 0.0), float(record["duration_ns"]))
    assert per_program, f"no {SDPA_LABEL} program in the profile"
    return per_program[max(per_program)]


class _Log:
    """The JSON-lines record of every measurement: {"key", "schedule": text, "status": "started" | "ok" | "invalid",
    "ns": [per prefix]}."""

    def __init__(self, path, key):
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.results, started = {}, set()
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            for n, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A record cut short by a crash mid-append: the last line only.
                    assert n == len(lines) - 1, f"{self.path}:{n + 1} is not JSON"
                    continue
                if entry["key"] != key:
                    continue
                if entry["status"] == "started":
                    started.add(entry["schedule"])
                else:
                    self.results[entry["schedule"]] = entry
        # Started and never finished: it crashed or hung the device.
        for text in started - self.results.keys():
            self.results[text] = {"schedule": text, "status": "invalid", "reason": "did not finish"}
            logger.warning(f"floater tune: {text} did not finish in an earlier run; not trying it again")

    def write(self, entry):
        with self.path.open("a") as f:
            f.write(json.dumps({"key": self.key, **entry}) + "\n")
        if entry["status"] != "started":
            self.results[entry["schedule"]] = entry


def _with_forwards(request, mesh_device, topology, chunk, prefixes, body, forwards=()):
    """Builds one MLA forward per cached prefix, then runs body(forwards) with all of them alive; each is a
    (forward, mla) pair."""
    if not prefixes:
        return body(forwards)
    result = {}

    def hook(forward, mla):
        result["value"] = _with_forwards(
            request, mesh_device, topology, chunk, prefixes[1:], body, forwards + ((forward, mla),)
        )
        mla.sdpa_floater_schedule = None
        return forward()

    T._run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=topology,
        iters_isl=[chunk],
        chunk_size_global=chunk,
        prefill_len=prefixes[0],
        use_metadata_tensor=True,
        mla_kwargs=dict(expanded_k=True),
        forward_hook=hook,
    )
    return result["value"]


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k3", "kimi_k2_7"], indirect=True, ids=["k3", "k2_7"])
@pytest.mark.parametrize("chunk", [5120, 2048], ids=["c5k", "c2k"])
@pytest.mark.timeout(0)
def test_tune_ring_joint_floaters(request, mesh_device, device_params, variant, chunk, config_only):
    sp, tp = mesh_device.shape
    heads = config_only.num_attention_heads // tp
    kvpe = config_only.kv_lora_rank + config_only.qk_rope_head_dim
    qk_head_dim = config_only.qk_nope_head_dim + config_only.qk_rope_head_dim
    q_chunk, k_chunk, bands = ttMLA._EXPANDED_K_SDPA_CHUNKS[(kvpe, qk_head_dim, heads, chunk // sp)]
    key = f"{variant.name} heads={heads} bands={bands} q={q_chunk} k={k_chunk} prefixes={list(PREFIXES)}"
    log = _Log(LOG_DIR / f"{variant.name}_c{chunk}.jsonl", key)
    logger.info(f"floater tune: {key}, log {log.path}")

    def tune(forwards):
        def measure(schedule):
            text = _text(schedule)
            if text in log.results:
                entry = log.results[text]
                return entry["ns"] if entry["status"] == "ok" else None
            log.write({"schedule": text, "status": "started"})
            try:
                ns = []
                for forward, mla in forwards:
                    mla.sdpa_floater_schedule = [] if schedule is None else list(schedule)
                    try:
                        ttnn.deallocate(forward())  # compile and warm up
                    except RuntimeError as error:  # a schedule the op refuses
                        reason = str(error).splitlines()[0][:300]
                        log.write({"schedule": text, "status": "invalid", "reason": reason})
                        logger.info(f"floater tune: {text} invalid: {reason}")
                        return None
                    ns.append(sorted(_sdpa_ns(mesh_device, forward) for _ in range(SAMPLES))[SAMPLES // 2])
            finally:
                # Each candidate is its own program; drop them so the cache does not grow with the search.
                mesh_device.clear_program_cache()
            log.write({"schedule": text, "status": "ok", "ns": ns})
            logger.info(f"floater tune: {text}: " + ", ".join(f"{n / 1e6:.3f} ms" for n in ns))
            return ns

        require_realtime_profiler("floater tuning")
        off = measure(None)
        assert off is not None, "no-floater baseline failed"

        def score(ns):
            return sum(n / o for n, o in zip(ns, off)) / len(off)

        best = SEEDS[variant.name]
        best_ns = measure(best)
        best_score = score(best_ns) if best_ns else float("inf")
        for frac_step in (384, 128):
            improved = True
            while improved:
                improved = False
                for candidate in _neighbours(best, frac_step):
                    ns = measure(candidate)
                    if ns is not None and score(ns) < best_score * (1 - NOISE):
                        best, best_ns, best_score = candidate, ns, score(ns)
                        logger.info(f"floater tune: new best {_text(best)} score {best_score:.4f}")
                        improved = True
                        break
        return off, best, best_ns, best_score

    off, best, best_ns, best_score = _with_forwards(
        request, mesh_device, per_axis_topology(device_params["fabric_config"]), chunk, PREFIXES, tune
    )
    per_prefix = ", ".join(f"{p}: {o / 1e6:.3f} -> {b / 1e6:.3f} ms" for p, o, b in zip(PREFIXES, off, best_ns or off))
    if best_score > 1 - MIN_GAIN:
        logger.info(
            f"floater tune RESULT {key}: no floaters (best {_text(best)} scores {best_score:.4f}; {per_prefix})"
        )
    else:
        i0, i1, f0, f1, tail, *merges = best
        entry = (
            f"{{{{{heads}, {bands}}}, {{{{{i0}, {i1}}}, {{{f0}, {f1}}}, "
            f"{'true' if tail else 'false'}, {{{', '.join(map(str, merges))}}}}}}},"
        )
        logger.info(f"floater tune RESULT {key}: score {best_score:.4f} ({per_prefix})\n    {entry}")
