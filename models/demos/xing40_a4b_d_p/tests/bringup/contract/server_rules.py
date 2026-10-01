# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen's prefill rules, ported to Python for the Xing serving-contract tests (CPU only, no device).

Every function cites the server code it copies (tt-d-gen @ 93e77b80, launch harness PR #1088 @ 7d4bee4e). The tests
use these to send the model exactly what the server sends: the chunk plan, the pad id, the SP reshuffle, the
round-robin slot interleave, the prefix-reuse start of a follow-up turn. The server's own KV checks
(`tools/launch_harness/tables.py`, `kv_manager/tools/kv_dump_compare.py`) are loaded from the tt-d-gen checkout, not
copied, so a test always applies the server's current code.
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

# ---------------------------------------------------------------- constants (server)
PAD_ID = 0xFFFFFFFF  # engine/include/engine/runtime/types.hpp:20
TILE = 32  # engine/include/engine/runtime/types.hpp:22 TILE_ALIGNMENT
KV_BLOCK = 64  # kv_block_size in every shipped prefill config (models/*/dynamo.disagg.prefill.json)
RECORD_TOKENS = 32  # tools/launch_harness/tables.py:23 (chunk_n_tokens must be 32)
MLA_WIDTH = 576  # tables.py:129
NOPE_SPLIT = 512  # kv_dump_compare.py:41 (nope = dims 0:512, pe = 512:576)
PREFILL_GOLDEN_PCC = 0.93  # launch_harness validation.py PCC_DEFAULTS["prefill-golden-pcc"]
LAST_BLOCK_PCC = 0.99  # launch_harness README "KV validation": last KV block PCC floor

# ---------------------------------------------------------------- this model's serving geometry (spec.yaml)
CHUNK = 5120  # spec target.chunk
MAX_SEQ = 56320  # spec target.seq (11 chunks)
SP, TP = 4, 2  # spec box.mesh [4, 2]: SP over axis 0, TP over axis 1
W = CHUNK // SP  # 1280 tokens per SP row per chunk
NUM_LAYERS = 40  # spec num_layers (MTP layer 40 never served)

# Where the server's code lives (the serving-contract agent read it there).
DGEN_MAIN_SHA = "93e77b802999fdb4303e61195373fdfd76f48b3c"
HARNESS_REF = "7d4bee4ecdcebb2d8f90b8519e471319400781cb"  # tt-d-gen PR #1088 (bzhang/launch-harness-tools)


def align_down(n: int, a: int) -> int:
    return n - n % a


def ceil_to(n: int, a: int) -> int:
    return -(-n // a) * a


# ---------------------------------------------------------------- chunk plan
def reusable_prefix_cap(kv_block_size: int, prompt_len: int) -> int:
    """engine/include/engine/control/prefix_indexer.hpp:75-78."""
    if kv_block_size == 0 or prompt_len == 0:
        return 0
    return (prompt_len - 1) // kv_block_size * kv_block_size


def follow_up_resident(prev_end: int, prompt_len: int, kv_block_size: int = KV_BLOCK) -> int:
    """Where a follow-up turn on the slot that holds the earlier turn starts on the prefill node.

    The reader indexes every full kv block below each retired chunk's actual_end (prefill_reader.cpp:121
    advance_kv_residency); admission matches the longest indexed prefix capped by reusable_prefix_cap and remounts the
    idle holder slot (backend_runtime.cpp:461-560). A follow-up turn's prompt extends the earlier one, so the match is
    every indexed block."""
    indexed = align_down(prev_end, kv_block_size) if kv_block_size else 0
    matched = min(indexed, reusable_prefix_cap(kv_block_size, prompt_len))
    return matched  # 0 = cold admit


def chunk_plan(prompt_len: int, resident: int = 0, chunk: int = CHUNK, max_seq: int = MAX_SEQ) -> list[tuple]:
    """The (actual_start, actual_end) of every chunk of one request.

    backend_runtime.cpp:841-850 setup_slot_for_prefill (base = align_down_to_tile(resident), n_chunks) and
    prefill_writer.cpp:51-82 (lo = base + i * chunk; the last chunk pulled back to max_seq - chunk when its padded end
    passes max_seq, workaround tt-d-gen #430; actual_end = min(lo + chunk, prompt_len))."""
    assert 0 < prompt_len <= max_seq, "admit(): prompt length exceeds max_seq_len (backend_runtime.cpp:168)"
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
    """The order the server injects chunks for several slots, each with a list of prompt lengths (its turns).

    The first turn of every slot is admitted at once; a slot's next turn is admitted (at the back of the prefill queue)
    when its previous turn's last chunk is injected. PrefillWriter::step serves the queue front: a non-last chunk
    rotates the slot to the back, the last chunk pops it (prefill_writer.cpp:103-114)."""
    q = deque()
    state = {}
    for s, turns in turns_by_slot.items():
        state[s] = {"turn": 0, "chunks": chunk_plan(turns[0], 0), "i": 0, "prev_end": 0}
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
        st["prev_end"] = hi
        st["turn"] += 1
        turns = turns_by_slot[s]
        if st["turn"] < len(turns):
            resident = follow_up_resident(hi, turns[st["turn"]], kv_block_size)
            st["chunks"], st["i"] = chunk_plan(turns[st["turn"]], resident), 0
            q.append(s)
    return out


# ---------------------------------------------------------------- the chunk payload
def ring_sdpa_reshuffle(tokens, c_start: int, intra: int, n_c: int, w: int) -> np.ndarray:
    """Literal port of engine/include/engine/runtime/ring_sdpa_reshuffle.hpp:31-68 (output device-major)."""
    inp = np.asarray(tokens, dtype=np.uint32).reshape(-1)
    vol = n_c * w
    assert inp.size == vol, (inp.size, vol)
    out = np.empty(vol, dtype=np.uint32)
    out[c_start * w : c_start * w + (w - intra)] = inp[: w - intra]
    out[c_start * w + (w - intra) : c_start * w + w] = inp[vol - intra :]
    for k in range(1, n_c):
        col = (c_start + k) % n_c
        s = (w - intra) + (k - 1) * w
        out[col * w : col * w + w] = inp[s : s + w]
    return out


def device_of(g: int, w: int = W, sp: int = SP) -> int:
    """SP row that receives absolute position g (ring_sdpa_reshuffle.hpp:10-11)."""
    return (g // w) % sp


def placement(start: int, chunk: int = CHUNK, sp: int = SP) -> list[list[int]]:
    """Absolute positions held by each SP row, in local order, for a chunk at `start` (the reshuffle's output)."""
    w = chunk // sp
    pos = ring_sdpa_reshuffle(np.arange(start, start + chunk, dtype=np.int64), (start // w) % sp, start % w, sp, w)
    return [pos[r * w : (r + 1) * w].astype(np.int64).tolist() for r in range(sp)]


def server_payload(chunk_tokens, start: int, end: int, chunk: int = CHUNK, sp: int = SP) -> np.ndarray:
    """What the H2D stream delivers for one chunk: the prompt slice padded with PAD_ID to `chunk`
    (prefill_writer.cpp:93-97), reshuffled with kv_offset = actual_start (prefill_pipeline.cpp:146-152), as the
    [sp, 1, chunk / sp] uint32 global tensor (runner_utils.make_global_spec)."""
    w = chunk // sp
    t = np.asarray(chunk_tokens, dtype=np.int64).reshape(-1)[: end - start].astype(np.uint32)
    buf = np.full(chunk, PAD_ID, dtype=np.uint32)
    buf[: t.size] = t
    if sp > 1:
        buf = ring_sdpa_reshuffle(buf, (start // w) % sp, start % w, sp, w)
    return buf.reshape(sp, 1, w)


def self_check() -> None:
    """The port against the server's own test rule (engine/tests/test_ring_sdpa_reshuffle.cpp expected_by_ownership)."""
    for sp_, w_ in ((1, 4), (2, 4), (4, 8), (SP, W)):
        for start in (0, 1, 3, 7, w_, w_ + 5, 2 * w_ * sp_ + 3, 2944, 12288, 51200):
            nat = np.arange(sp_ * w_, dtype=np.uint32)
            want = np.empty_like(nat)
            fill = [0] * sp_
            for i in range(sp_ * w_):
                d = ((start + i) // w_) % sp_
                want[d * w_ + fill[d]] = nat[i]
                fill[d] += 1
            got = ring_sdpa_reshuffle(nat, (start // w_) % sp_, start % w_, sp_, w_)
            assert np.array_equal(got, want), f"ring_sdpa_reshuffle port wrong at sp={sp_} w={w_} start={start}"
    assert chunk_plan(56000, 2944)[-1] == (51200, 56000) and len(chunk_plan(56000, 2944)) == 11
    assert chunk_plan(52000)[-1] == (51200, 52000) and chunk_plan(3000) == [(0, 3000)]
    assert follow_up_resident(3000, 56000) == 2944 and follow_up_resident(12345, 20000) == 12288


# ---------------------------------------------------------------- the server's own KV checks (loaded, not copied)
def server_repo() -> Path:
    r = os.environ.get("BRINGUP_SERVER_REPO")
    if not r:
        try:
            from models.demos.common.bringup.reference.golden import load_spec

            r = load_spec().get("serving.server_repo")
        except Exception:
            r = None
    return Path(r or f"/localdev/{getpass.getuser()}/tt-d-gen")


_HARNESS_FILES = (
    "tools/launch_harness/tables.py",
    "tools/launch_harness/modes.py",
    "tools/launch_harness/validation.py",
    "tools/launch_harness/kv_chunk_address_table_pb2.py",
    "kv_manager/tools/kv_dump_compare.py",
)
_LOADED = {}


def _server_source(path: str) -> str:
    """The file at the pinned harness ref (what the contract was written against), else the working tree."""
    repo = server_repo()
    if not repo.exists():
        raise FileNotFoundError(f"tt-d-gen checkout not found at {repo} (set BRINGUP_SERVER_REPO)")
    r = subprocess.run(["git", "-C", str(repo), "show", f"{HARNESS_REF}:{path}"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout
    p = repo / path
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(
        f"{path} is neither at {HARNESS_REF} nor in {repo}; fetch tt-d-gen PR #1088 "
        f"(git -C {repo} fetch origin bzhang/launch-harness-tools)"
    )


def harness():
    """(tables, kv_dump_compare) modules from tt-d-gen: the launch harness's table rules and the KV dump comparer."""
    if "mods" in _LOADED:
        return _LOADED["mods"]
    root = Path(tempfile.mkdtemp(prefix="xing_dgen_harness_"))
    pkg = root / "launch_harness"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for f in _HARNESS_FILES:
        (pkg / Path(f).name).write_text(_server_source(f))
    sys.path.insert(0, str(root))
    tables = importlib.import_module("launch_harness.tables")
    spec = importlib.util.spec_from_file_location("xing_kv_dump_compare", pkg / "kv_dump_compare.py")
    kvc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kvc)
    _LOADED["mods"] = (tables, kvc)
    return tables, kvc


def record_name(slot: int, layer: int, pos: int) -> str:
    """kv_dram_poke --dump file name (kv_dump_compare.py:7, FNAME)."""
    return f"s{slot}_l{layer}_p{pos}.bin"


# ---------------------------------------------------------------- golden
def golden():
    """The bring-up's s56320 golden: tokens [56320] and per-layer kv_latent [56320, 576] (spec state.tensors)."""
    from models.demos.common.bringup.reference.golden import Golden, load_spec

    return Golden.for_rung(load_spec(), "s56320")


def golden_kv(g, layer: int):
    return g.state(layer)["kv_latent"].float()


def kv_pcc_failures(kvc, got: np.ndarray, want: np.ndarray, layer: int, threshold: float, what: str) -> list[str]:
    """kv_dump_compare.tensor_pcc (nope / pe channels scored separately) -> failure strings."""
    out = []
    for f in kvc.tensor_pcc(np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32), layer, threshold):
        print(f"  {what} layer {layer} {f['channel']}: pcc {f['pcc']} (>= {threshold})")
        if not f["passed"]:
            out.append(f"{what}: layer {layer} {f['channel']} pcc {f['pcc']} < {threshold}")
    return out


def state_threshold() -> float:
    """The KV-vs-golden floor: the stricter of the server's prefill-golden PCC and the spec's state threshold."""
    from models.demos.common.bringup.reference.golden import load_spec
    from models.demos.common.bringup.testing.harness import threshold

    return max(PREFILL_GOLDEN_PCC, threshold(load_spec(), "state"))
