# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen's prefill rules, ported to Python for the DeepSeek-V4.1-Flash serving-contract tests (CPU only).

Every function cites the server code it copies, at tt-d-gen @ a286b622 (DGEN_SHA). The tests use these to send the
model exactly what the server sends: the chunk plan, the pad id, the SP reshuffle, the round-robin slot interleave,
the prefix-reuse start of a follow-up turn. The server's own KV checks (tools/launch_harness/tables.py,
kv_manager/tools/kv_dump_compare.py, tools/launch_harness/validation.py) are loaded from the tt-d-gen checkout at
DGEN_SHA, not copied, so a test always applies the server's code.
"""

from __future__ import annotations

import getpass
import importlib.util
import os
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- where the rules come from
DGEN_SHA = "a286b6221da5ee63fb7f7f69ed2772e51d0fd53e"  # tt-d-gen main, read 2026-10-14

# ---------------------------------------------------------------- constants (server)
PAD_ID = 0xFFFFFFFF  # engine/include/engine/runtime/types.hpp:20
TILE = 32  # engine/include/engine/runtime/types.hpp:22 TILE_ALIGNMENT (chunk bases are align_down_to_tile)
KV_BLOCK = 64  # kv_block_size in every shipped worker config (models/*/dynamo.disagg.prefill.json "kv_block_size")
RECORD_TOKENS = 32  # tools/launch_harness/tables.py:37 (every config's chunk_n_tokens must be 32)
PREFILL_GOLDEN_PCC = 0.93  # tools/launch_harness/validation.py:18-22 PCC_DEFAULTS["prefill-golden-pcc"]
LAST_BLOCK_PCC = 0.99  # validation.py:18-22 PCC_DEFAULTS["last-block-pcc"] (last block / prefix decode held)
DRAM_ALIGN = 64  # kv_manager/device_io/src/noc_address_resolver.cpp:18-22 (BH NIU: offset % 64 == 0)


# ---------------------------------------------------------------- the tests' serving geometry
def _spec():
    from models.demos.common.bringup.core.spec import Spec

    try:
        from models.demos.common.bringup.reference.golden import load_spec

        return load_spec()
    except (Exception, SystemExit):
        return Spec.load(Path(__file__).resolve().parents[3] / "bringup" / "spec.yaml")


def golden_geometry(spec=None) -> tuple[str, int, int]:
    """(rung, chunk, max_seq) the contract tests run at: the running spec's shortest ladder rung (by seq) with a golden
    of its own and full dumps (every chunk dumped, so every position of a dumped layer's window KV and block input is
    in the golden); chunk = that rung's chunk, max_seq = its seq. Never one box's literals: s4096 / 2048 / 4096 on the
    QuietBox seed, s10240 / 5120 / 10240 on galaxy_8x4. The server's rules do not depend on the size."""
    s = spec or _spec()
    own = [r for r in s.data.get("ladder", []) if r.get("golden", r["name"]) == r["name"] and r.get("full_dumps")]
    if not own:
        raise ValueError("the spec's ladder has no rung with its own full-dump golden: the contract tests need one")
    r = min(own, key=lambda x: int(x["seq"]))
    return r["name"], int(r["chunk"]), int(r["seq"])


GOLDEN_RUNG, CHUNK, MAX_SEQ = golden_geometry()
assert MAX_SEQ % CHUNK == 0 and CHUNK % TILE == 0 and MAX_SEQ % TILE == 0  # backend_runtime.cpp:124-129


def _mesh() -> tuple[int, int]:
    """spec box.mesh: SP over axis 0 (rows), TP over axis 1 (columns)."""
    rows, cols = _spec().mesh
    return int(rows), int(cols)


SP, TP = _mesh()  # 2, 2 on the QuietBox seed; 8, 4 on galaxy_8x4
W = CHUNK // SP  # tokens per SP row per chunk: 1024 on the seed (2048 / SP 2), 640 on galaxy_8x4 (5120 / SP 8)
assert CHUNK % SP == 0, "prefill_pipeline.cpp:37-39: chunk_size % sp_factor == 0"


def align_down(n: int, a: int) -> int:
    return n - n % a


def ceil_to(n: int, a: int) -> int:
    return -(-n // a) * a


# ---------------------------------------------------------------- chunk plan
def reusable_prefix_cap(kv_block_size: int, prompt_len: int) -> int:
    """engine/include/engine/control/prefix_indexer.hpp:75-78: the last block boundary strictly below prompt_len (the
    block holding the last token is always recomputed). Also the boundary decode asks for: the migrated prefix is
    [0, cap) and decode recomputes [cap, prompt_len) (validation.py:55-61 comparison_ranges)."""
    if kv_block_size == 0 or prompt_len == 0:
        return 0
    return (prompt_len - 1) // kv_block_size * kv_block_size


def follow_up_resident(prev_end: int, prompt_len: int, kv_block_size: int = KV_BLOCK) -> int:
    """Where a follow-up turn starts on the prefill node when the idle slot holding the earlier turn is remounted.

    The reader indexes every full kv block below each retired chunk's actual_end (prefill_reader.cpp:177
    advance_kv_residency); admission takes the longest indexed prefix capped by reusable_prefix_cap and remounts the
    idle holder (backend_runtime.cpp:527-557 prepare_admission: reacquire + truncate). A prefill node never copies a
    prefix into another slot (copy needs kv_owned_by_control(), DECODE only: backend_runtime.hpp:175,
    backend_runtime.cpp:616-618), so a reused prefix is always in the same slot."""
    indexed = align_down(prev_end, kv_block_size) if kv_block_size else 0
    return min(indexed, reusable_prefix_cap(kv_block_size, prompt_len))


def chunk_plan(prompt_len: int, resident: int = 0, chunk: int | None = None, max_seq: int | None = None) -> list[tuple]:
    """(actual_start, actual_end) of every chunk of one request.

    backend_runtime.cpp:1238-1247 setup_slot_for_prefill (base = align_down_to_tile(resident), n_chunks = ceil((prompt
    - base) / chunk)) and prefill_writer.cpp:51-82 (lo = base + i * chunk; a last chunk whose padded end passes max_seq
    is pulled back to max_seq - chunk, tt-d-gen #430; actual_end = min(lo + chunk, prompt_len))."""
    chunk, max_seq = chunk or CHUNK, max_seq or MAX_SEQ
    assert 0 < prompt_len <= max_seq, "admit(): prompt length exceeds max_seq_len (backend_runtime.cpp:202)"
    base = align_down(resident, TILE)
    n = max(1, -(-(prompt_len - base) // chunk))
    out = []
    for i in range(n):
        lo = base + i * chunk
        last = i + 1 >= n
        if last and lo + chunk > max_seq:
            lo = max_seq - chunk
        hi = min(lo + chunk, prompt_len)
        assert hi > lo, "prefill overcount (prefill_writer.cpp:75)"
        assert not last or hi == prompt_len, "prefill undercount (prefill_writer.cpp:79)"
        out.append((lo, hi))
    return out


@dataclass
class Push:
    slot: int
    chunk_idx: int
    start: int
    end: int
    is_last: bool
    turn: int


def interleave(turns_by_slot: dict[int, list[int]], kv_block_size: int = KV_BLOCK) -> list[Push]:
    """The order the server injects chunks for several slots, each with its turns' prompt lengths.

    Every slot's first turn is admitted at once; its next turn is admitted (to the back of the prefill queue) when the
    previous turn's last chunk is injected. PrefillWriter::step serves the queue front: a non-last chunk rotates the
    slot to the back, the last chunk pops it (prefill_writer.cpp:103-114)."""
    q = deque()
    state = {}
    for s, turns in turns_by_slot.items():
        state[s] = {"turn": 0, "chunks": chunk_plan(turns[0], 0), "i": 0}
        q.append(s)
    out = []
    while q:
        s = q[0]
        st = state[s]
        lo, hi = st["chunks"][st["i"]]
        last = st["i"] + 1 >= len(st["chunks"])
        out.append(Push(s, st["i"], lo, hi, last, st["turn"]))
        st["i"] += 1
        if not last:
            q.rotate(-1)
            continue
        q.popleft()
        st["turn"] += 1
        turns = turns_by_slot[s]
        if st["turn"] < len(turns):
            resident = follow_up_resident(hi, turns[st["turn"]], kv_block_size)
            st["chunks"], st["i"] = chunk_plan(turns[st["turn"]], resident), 0
            q.append(s)
    return out


def follow_up(prev_end: int, prompt_len: int, kv_block_size: int = KV_BLOCK) -> tuple[int, list[tuple]]:
    """(resident, chunk plan) of a follow-up turn on the slot whose last turn ended at prev_end."""
    res = follow_up_resident(prev_end, prompt_len, kv_block_size)
    return res, chunk_plan(prompt_len, res)


# ---------------------------------------------------------------- the chunk payload
def ring_sdpa_reshuffle(tokens, c_start: int, intra: int, n_c: int, w: int) -> np.ndarray:
    """Literal port of engine/include/engine/runtime/ring_sdpa_reshuffle.hpp:31-68 (output device-major)."""
    inp = np.asarray(tokens).reshape(-1)
    vol = n_c * w
    assert inp.size == vol, (inp.size, vol)
    out = np.empty(vol, dtype=inp.dtype)
    out[c_start * w : c_start * w + (w - intra)] = inp[: w - intra]
    out[c_start * w + (w - intra) : c_start * w + w] = inp[vol - intra :]
    for k in range(1, n_c):
        col = (c_start + k) % n_c
        s = (w - intra) + (k - 1) * w
        out[col * w : col * w + w] = inp[s : s + w]
    return out


def device_of(g: int, w: int = W, sp: int = SP) -> int:
    """SP row that receives absolute position g (ring_sdpa_reshuffle.hpp:9-11): (g / W) % N_C."""
    return (g // w) % sp


def placement(start: int, chunk: int = CHUNK, sp: int = SP) -> list[list[int]]:
    """Absolute positions each SP row receives, in its local order, for a chunk whose first token is at `start` (the
    reshuffle with kv_offset = actual_start, prefill_pipeline.cpp:146-153)."""
    w = chunk // sp
    pos = ring_sdpa_reshuffle(np.arange(start, start + chunk, dtype=np.int64), (start // w) % sp, start % w, sp, w)
    return [pos[r * w : (r + 1) * w].tolist() for r in range(sp)]


def server_payload(chunk_tokens, start: int, end: int, chunk: int = CHUNK, sp: int = SP) -> np.ndarray:
    """What the H2D stream delivers for one chunk: the prompt slice padded with PAD_ID to `chunk`
    (prefill_writer.cpp:93-97), reshuffled with kv_offset = actual_start when sp > 1 (prefill_pipeline.cpp:146-153),
    as the [sp, 1, chunk / sp] uint32 global tensor (runner_utils.make_h2d_spec, sharded over mesh axis 0)."""
    w = chunk // sp
    t = np.asarray(chunk_tokens, dtype=np.int64).reshape(-1)[: end - start].astype(np.uint32)
    buf = np.full(chunk, PAD_ID, dtype=np.uint32)
    buf[: t.size] = t
    if sp > 1:
        buf = ring_sdpa_reshuffle(buf, (start // w) % sp, start % w, sp, w)
    return buf.reshape(sp, 1, w)


def header(slot: int, start: int, end: int) -> bytes:
    """The 12-byte chunk header {slot_id, actual_start, actual_end} (pipeline_types.hpp:42-53)."""
    import struct

    return struct.pack("<3I", slot, start, end)


# ---------------------------------------------------------------- the tests' prompts
# Chunk plans for the tests, from CHUNK / MAX_SEQ (H = CHUNK / 2; chunk_plan / interleave are the server's rules;
# self_check() asserts the shape each test relies on). The lengths keep the seed's offsets, so the QuietBox seed (CHUNK
# 2048, MAX_SEQ 4096, W 1024) gets exactly its numbers in brackets; galaxy_8x4 (5120 / 10240, W 640) the second ones.
#   cold_mid_end   cold CHUNK + 969 [3017; 6089]: (0, CHUNK), (CHUNK, end): a second chunk ending mid-record and
#                  inside a ratio-2 group
#   follow_block   turn H - 24 [1000; 2536] then CHUNK + 952 [3000; 6072] over a resident prefix H - 64 [960; 2496]
#                  (kv blocks of 64): one chunk from a start off the SP period, so every SP row's slab of the chunk wraps
#                  (prefill_pipeline.cpp:146-153)
#   follow_tile    kv_block_size 32 (legal: only a multiple of the tile is required): turn H + 306 [1330; 2866] then
#                  CHUNK + 953 [3001; 6073] over H + 288 [1312; 2848]: a start that is a multiple of 32 but not of 64
#   pulled_back    turn H - 24 then MAX_SEQ - 96 [4000; 10144] over H - 64: (H - 64, H - 64 + CHUNK), then the last
#                  chunk pulled back to (MAX_SEQ - CHUNK, MAX_SEQ - 96) over rows the chunk before it wrote
#                  (prefill_writer.cpp:63-67)
H = CHUNK // 2
assert H % KV_BLOCK == 0 and H >= 1024, f"the scenario offsets need CHUNK / 2 a multiple of 64 and >= 1024 ({H})"
SCENARIOS = {
    "cold_mid_end": {"turns": [CHUNK + 969], "kv_block": KV_BLOCK},
    "follow_block": {"turns": [H - 24, CHUNK + 952], "kv_block": KV_BLOCK},
    "follow_tile": {"turns": [H + 306, CHUNK + 953], "kv_block": 32},
    "pulled_back": {"turns": [H - 24, MAX_SEQ - 96], "kv_block": KV_BLOCK},
}
# The runner test's requests. Producer fallback: slot 0 H - 24 then MAX_SEQ - 96 (follow-up from H - 64, pulled back),
# slot 1 H + 476 [1500; 3036] then CHUNK + 1452 [3500; 6572] (follow-up from H + 448 [1472; 3008]), interleaved
# round-robin. Engine: see test_runner_contract.ENGINE_CASE.
RUNNER_TURNS = {0: [H - 24, MAX_SEQ - 96], 1: [H + 476, CHUNK + 1452]}
# The adapter test's turns: slot 0 one chunk then a follow-up from its reused prefix; slot 1 cold over two chunks.
ACK_TURNS = {0: [H - 24, CHUNK + 952], 1: [CHUNK + 952]}


def scenario_plan(name: str) -> list[tuple[int, list[tuple]]]:
    """[(resident, chunk plan)] per turn of SCENARIOS[name] on one slot."""
    sc = SCENARIOS[name]
    out, prev = [], None
    for n in sc["turns"]:
        res = 0 if prev is None else follow_up_resident(prev, n, sc["kv_block"])
        plan = chunk_plan(n, res)
        out.append((res, plan))
        prev = plan[-1][1]
    return out


def self_check() -> None:
    """The ports against the server's own test rule (engine/tests/test_ring_sdpa_reshuffle.cpp expected_by_ownership)
    and the scenario shapes the tests rely on."""
    for sp_, w_ in ((1, 4), (2, 4), (4, 8), (SP, W)):
        for start in (0, 1, 3, 7, w_, w_ + 5, 2 * w_ * sp_ + 3, 960, 1312, 2048):
            nat = np.arange(sp_ * w_, dtype=np.uint32)
            want = np.empty_like(nat)
            fill = [0] * sp_
            for i in range(sp_ * w_):
                d = ((start + i) // w_) % sp_
                want[d * w_ + fill[d]] = nat[i]
                fill[d] += 1
            got = ring_sdpa_reshuffle(nat, (start // w_) % sp_, start % w_, sp_, w_)
            assert np.array_equal(got, want), f"ring_sdpa_reshuffle port wrong at sp={sp_} w={w_} start={start}"
    big = {"chunk": 8192, "max_seq": 1048576}
    assert chunk_plan(1048576, 2944, **big)[-1] == (1048576 - 8192, 1048576)
    assert chunk_plan(3000, **big) == [(0, 3000)] and follow_up_resident(3000, 9000) == 2944
    assert scenario_plan("cold_mid_end") == [(0, [(0, CHUNK), (CHUNK, CHUNK + 969)])]
    assert scenario_plan("follow_block")[1] == (H - 64, [(H - 64, CHUNK + 952)])
    assert (H - 64) % W, "follow_block: the start must be off the SP period"
    assert scenario_plan("follow_tile")[1] == (H + 288, [(H + 288, CHUNK + 953)]) and (H + 288) % 64 == 32
    res, plan = scenario_plan("pulled_back")[1]
    assert res == H - 64 and plan == [(H - 64, H - 64 + CHUNK), (MAX_SEQ - CHUNK, MAX_SEQ - 96)]
    assert plan[-1][0] == MAX_SEQ - CHUNK < plan[-2][1]
    for turns in (RUNNER_TURNS, ACK_TURNS):
        assert max(n for t in turns.values() for n in t) <= MAX_SEQ
    assert [(p.slot, p.start, p.end) for p in interleave(RUNNER_TURNS)] == [
        (0, 0, H - 24),
        (1, 0, H + 476),
        (0, H - 64, H - 64 + CHUNK),
        (1, H + 448, CHUNK + 1452),
        (0, MAX_SEQ - CHUNK, MAX_SEQ - 96),
    ]
    assert [(p.slot, p.start, p.end) for p in interleave(ACK_TURNS)] == [
        (0, 0, H - 24),
        (1, 0, CHUNK),
        (0, H - 64, CHUNK + 952),
        (1, CHUNK, CHUNK + 952),
    ]


# ---------------------------------------------------------------- the server's own KV checks (loaded, not copied)
def server_repo() -> Path:
    r = os.environ.get("BRINGUP_SERVER_REPO") or _spec().get("serving.server_repo")
    return Path(os.path.expandvars(r) if r else f"/localdev/{getpass.getuser()}/tt-d-gen")


_HARNESS_FILES = (
    "tools/launch_harness/tables.py",
    "tools/launch_harness/modes.py",
    "tools/launch_harness/validation.py",
    "tools/launch_harness/kv_chunk_address_table_pb2.py",
    "kv_manager/tools/kv_dump_compare.py",
)
_LOADED = {}


def _server_source(path: str) -> str:
    """The file at DGEN_SHA (what the contract was written against), else the working tree."""
    repo = server_repo()
    if not repo.exists():
        raise FileNotFoundError(f"tt-d-gen checkout not found at {repo} (set BRINGUP_SERVER_REPO)")
    r = subprocess.run(["git", "-C", str(repo), "show", f"{DGEN_SHA}:{path}"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout
    p = repo / path
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(f"{path} is neither at {DGEN_SHA} nor in {repo}")


def harness():
    """(tables, kv_dump_compare, validation) from tt-d-gen: the launch harness's table rules, the KV dump comparer
    and the harness's PCC limits / comparison ranges."""
    if "mods" in _LOADED:
        return _LOADED["mods"]
    root = Path(tempfile.mkdtemp(prefix="ds41_dgen_harness_"))
    pkg = root / "launch_harness"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for f in _HARNESS_FILES:
        (pkg / Path(f).name).write_text(_server_source(f))
    sys.path.insert(0, str(root))
    tables = importlib.import_module("launch_harness.tables")
    validation = importlib.import_module("launch_harness.validation")
    spec = importlib.util.spec_from_file_location("ds41_kv_dump_compare", pkg / "kv_dump_compare.py")
    kvc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kvc)
    assert validation.PCC_DEFAULTS["prefill-golden-pcc"] == PREFILL_GOLDEN_PCC, validation.PCC_DEFAULTS
    assert validation.PCC_DEFAULTS["last-block-pcc"] == LAST_BLOCK_PCC, validation.PCC_DEFAULTS
    _LOADED["mods"] = (tables, kvc, validation)
    return _LOADED["mods"]


def record_name(slot: int, layer: int, pos: int) -> str:
    """kv_dram_poke --dump file name (kv_dump_compare.py:7, FNAME)."""
    return f"s{slot}_l{layer}_p{pos}.bin"


STATE_OF_KIND = {"window": "window_kv", "compress": "compress_kv", "index": "index_k"}  # cache config kind -> state


def state_threshold():
    """KV vs golden limit per (config, layer): the stricter of the server's prefill-golden PCC (its own rule) and
    calib x the reference's drift for that state and layer on the GOLDEN_RUNG rung (F128, testing/e2e_limits.py)."""
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.common.bringup.testing import e2e_limits as E

    drift = E.load(Golden.for_rung(_spec(), GOLDEN_RUNG).dir, GOLDEN_RUNG)

    def limit(cfg: dict, layer: int) -> float:
        return max(PREFILL_GOLDEN_PCC, E.pcc_limit(drift, f"state_{STATE_OF_KIND[cfg['kind']]}_L{layer:02d}"))

    return limit
