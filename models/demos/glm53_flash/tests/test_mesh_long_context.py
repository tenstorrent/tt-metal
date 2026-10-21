# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context end-to-end run on the mesh: one user, a needle-in-a-haystack prompt of ``GLM53_LONG_N`` tokens
(chat template), chunked prefill (``GLM53_LONG_CHUNK`` tokens per chunk, every chunk but the last a multiple of
32 * rows), then ``GLM53_LONG_GEN`` greedy decode steps through the captured decode trace. Reports the prefill time
per chunk and in total (TTFT), the decode step time as tok/s per user, the generated text and whether the needle was
retrieved (``GLM53_LONG_ASSERT=1`` makes a missing needle a failure).

    GLM53_MESH_SHAPE=8,4 GLM53_LONG_N=16384 pytest models/demos/glm53_flash/tests/test_mesh_long_context.py -s
    GLM53_MESH_SHAPE=8,4 GLM53_LONG_N=1048576 GLM53_PARTITION_USERS=0 GLM53_LONG_B=1 pytest ... -s   # the full context, one slot per chip
    GLM53_MESH_SHAPE=8,4 GLM53_LONG_N=1048576 GLM53_PREFILL_PARTITION=heads2users GLM53_LONG_USERS=all pytest ... -s
        # 4 users per row at the full context: one slot per chip, every user prefilled (sparse_1m.md section 7)
Knobs shared with the e2e test / demo: GLM53_MODEL_LAYERS (default 0-44), GLM53_MODEL_EXPERT_DTYPE /
GLM53_MODEL_EXPERT_DOWN_DTYPE (bfp8), GLM53_MESH_CACHE (tensorbin weight cache), GLM53_MODEL_TRACE_MB.
The DSA cache is user-partitioned over the columns (GLM53_PARTITION_USERS=1 is set here unless overridden) so a 1 M
user costs 1/4 of its latent + index per column chip. ``GLM53_LONG_USERS`` (default 1): how many user slots per row
are prefilled, one after the other, with the same prompt (``all`` = every slot: one per column with the partitioned
cache); the decode then runs every slot on its own greedy chain and the needle / first token are checked per
(row, user). The JSON keeps user 0's chain in ``generated`` / ``text`` (what ``reference/compare_to_tt.py`` grades)
and every chain under ``users``. The allocator's DRAM view is logged after the build and after the prefills.
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
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import log_dram
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

os.environ.setdefault("GLM53_PARTITION_USERS", "1")

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
N = int(os.environ.get("GLM53_LONG_N", "16384"))  # prompt tokens (approximate: the filler is scaled to it)
CHUNK = int(
    os.environ.get("GLM53_LONG_CHUNK", "16384")
)  # 32 K chunks exceed the KDA grouped-scan owner cores on 8 rows
GEN = int(os.environ.get("GLM53_LONG_GEN", "32"))
DEPTH = float(os.environ.get("GLM53_LONG_NEEDLE_DEPTH", "0.5"))  # needle position as a fraction of the filler
B = int(os.environ.get("GLM53_LONG_B", "4"))  # user slots per row (one per column with the partitioned cache)
_users_env = os.environ.get("GLM53_LONG_USERS", "1")
N_USERS = B if _users_env == "all" else min(B, int(_users_env))  # slots per row prefilled (sequentially)
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
NEEDLE = "The secret passphrase for the vault is 'amber-falcon-7391'."
QUESTION = "What is the secret passphrase for the vault? Answer with the passphrase only."
NEEDLE_KEY = "amber-falcon-7391"

_TOPICS = [
    "the harbour",
    "a mountain pass",
    "the old library",
    "a night market",
    "the river delta",
    "an orchard",
    "the observatory",
    "a tram depot",
    "the glass factory",
    "a lighthouse",
    "the salt flats",
    "a bakery",
]
_VERBS = ["described", "measured", "repainted", "catalogued", "photographed", "mapped", "restored", "visited"]


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-44")
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def filler_sentences(n: int, seed: int = 0) -> str:
    """``n`` varied, non-repeating filler sentences (counters keep every sentence distinct)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        t = _TOPICS[int(torch.randint(len(_TOPICS), (1,), generator=g))]
        v = _VERBS[int(torch.randint(len(_VERBS), (1,), generator=g))]
        k = int(torch.randint(10, 999, (1,), generator=g))
        out.append(f"On day {i + 1} the survey team {v} {t} and logged {k} entries before noon.")
    return " ".join(out)


def build_needle_ids(tok, n_tokens: int, depth: float) -> tuple[list[int], str]:
    """Chat-templated prompt of about ``n_tokens`` tokens: filler with the needle at ``depth``, then the question."""

    # the generation prompt ends with ``<|assistant|><think>`` (this template has no no-think switch); closing the
    # think block right away makes the greedy continuation the answer itself, so a short generation can contain the
    # needle (the HF reference builds its prompt through this same function)
    close_think = tok.encode("</think>", add_special_tokens=False)

    def ids_of(text: str) -> list[int]:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):  # transformers >= 5 returns a BatchEncoding
            ids = ids["input_ids"]
        return [int(t) for t in ids] + close_think

    def prompt(n_sent: int) -> str:
        a = filler_sentences(max(1, int(n_sent * depth)), seed=1)
        b = filler_sentences(max(1, n_sent - int(n_sent * depth)), seed=2)
        return f"{a} {NEEDLE} {b}\n\n{QUESTION}"

    n_sent = max(2, n_tokens // 20)
    for _ in range(4):  # converge the filler length to the requested token count
        text = prompt(n_sent)
        ids = ids_of(text)
        if abs(len(ids) - n_tokens) <= max(64, n_tokens // 200) or n_sent <= 2:
            return ids, text
        n_sent = max(2, int(n_sent * n_tokens / len(ids)))
    return ids, text


def eos_ids(snapshot: Path, tok) -> set[int]:
    ids = set()
    gc = snapshot / "generation_config.json"
    if gc.exists():
        v = json.loads(gc.read_text()).get("eos_token_id", [])
        ids.update(v if isinstance(v, list) else [v])
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1024 * 1024)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mesh_long_context(mesh_device):
    from transformers import AutoTokenizer

    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    stop = eos_ids(snapshot, tok)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    layers = _layers()
    rows, cols = MESH
    ids, text = build_needle_ids(tok, N, DEPTH)
    S = len(ids)
    max_seq_len = S + GEN + 256
    logger.info(f"long context: {S} prompt tokens (requested {N}), chunk {CHUNK}, gen {GEN}, max_seq_len {max_seq_len}")
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=max_seq_len,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    logger.info(f"model built in {time.perf_counter() - t0:.0f} s ({len(layers)} layers, {U} slots)")
    dsa = [d for d in model.layers if d.cache is not None]
    if dsa:
        c = dsa[0].cache
        logger.info(
            f"DSA cache per chip: partition {c.partition} prefill_mode {c.prefill_mode} slots {c.slots} "
            f"(users_per_chip {c.users_per_chip}, scratch {c.scratch}, dummy blocks {c.dummy_blocks}), "
            f"{c.blocks_per_seq} blocks x 32 x {c.latent_dim} per slot; pools per user {c.index.pools_per_user}"
        )
    log_dram(mesh_device, "after build")
    # decode trace first (compile + capture at position 0 on dummy ids; the prefill below overwrites the state)
    trace_id, (_, _, ids_tr) = model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
    model.reset_kda_states()
    # chunked prefill of users 0 .. N_USERS-1, one after the other (the SP prefill writes slot ``user`` of every
    # row: the rows hold the same prompts; with the heads2users partitioned cache column u of a row holds user u)
    chunk_log: list[tuple[int, int, int, float]] = []
    user_in_flight = [0]

    def on_chunk(s0: int, e: int, sec: float):
        chunk_log.append((user_in_flight[0], s0, e, sec))
        logger.info(
            f"user {user_in_flight[0]} prefill chunk [{s0}, {e}) {e - s0} tokens: {sec:.2f} s ({(e - s0) / sec:.0f} tok/s)"
        )

    firsts, ttfts = [], []
    for u in range(N_USERS):
        user_in_flight[0] = u
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        _, logits_last, owner_row = model.prefill_chunked(
            torch.tensor(ids, dtype=torch.long), user=u, chunk=CHUNK, on_chunk=on_chunk
        )
        firsts.append(model.first_token(logits_last, owner_row))
        ttfts.append(time.perf_counter() - t0)
        logger.info(
            f"user {u}: TTFT {ttfts[-1]:.1f} s for {S} tokens ({S / ttfts[-1]:.0f} tok/s prefill); first token "
            f"{firsts[-1]} {tok.decode([firsts[-1]])!r}"
        )
    log_dram(mesh_device, "after prefill")
    first, ttft_s = firsts[0], ttfts[0]
    # greedy decode: every slot runs its own chain (slot u of every row carries prompt u for u < N_USERS; the other
    # slots are ignored); a chain stops at EOS, the step stops when every prefilled chain has
    live = [(r, u) for r in range(rows) for u in range(N_USERS)]
    gens = {(r, u): [firsts[u]] for r, u in live}
    done = {k: False for k in live}
    positions = [S] * U
    step_s = []
    for _ in range(GEN - 1):
        feed = torch.tensor([gens[(r, u)][-1] if (r, u) in gens else firsts[0] for r in range(rows) for u in range(B)])
        model.load_ids(feed)
        model.set_positions(positions)
        t0 = time.perf_counter()
        model.replay(trace_id)
        nxt = model.ids_to_host(ids_tr)
        step_s.append(time.perf_counter() - t0)
        for r, u in live:
            if done[(r, u)]:
                continue
            tid = int(nxt[r * B + u])
            gens[(r, u)].append(tid)
            done[(r, u)] = tid in stop
        positions = [p + 1 for p in positions]
        if all(done.values()):
            break
    texts = {k: tok.decode([t for t in g if t not in stop], skip_special_tokens=True) for k, g in gens.items()}
    founds = {k: NEEDLE_KEY in t for k, t in texts.items()}
    gen, out_text, found = gens[(0, 0)], texts[(0, 0)], founds[(0, 0)]
    steady = step_s[2:] or step_s
    step_ms = 1000 * sum(steady) / len(steady) if steady else float("nan")
    for r, u in live:
        logger.info(
            f"=== OUTPUT row {r} user {u} ({len(gens[(r, u)])} tokens, first {gens[(r, u)][0]}): {texts[(r, u)]!r}\n"
            f"needle {'FOUND' if founds[(r, u)] else 'MISSING'} ({NEEDLE_KEY})"
        )
    chains_equal = all(gens[(r, u)] == gens[(0, u)] for r, u in live)
    users_equal = all(gens[(0, u)] == gens[(0, 0)] for u in range(N_USERS))
    logger.info(
        f"PERF-LONG mesh {MESH} layers {len(layers)} ISL {S} chunk {CHUNK} users/row {N_USERS}: prefill "
        f"{ttft_s:.1f} s ({S / ttft_s:.0f} tok/s) [per user {[f'{t:.1f}' for t in ttfts]}], decode step {step_ms:.1f} ms = "
        f"{1000 / step_ms:.1f} tok/s per user at position {S} ({U} slots: {U * 1000 / step_ms:.1f} tok/s aggregate); "
        f"needle {'found' if found else 'missing'}; needles found {sum(founds.values())}/{len(founds)}; "
        f"rows agree {chains_equal}; users agree {users_equal}"
    )
    out_dir = Path("generated/glm53_flash_long_context")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"mesh{rows}x{cols}_N{S}.json").write_text(
        json.dumps(
            {
                "mesh": MESH,
                "layers": len(layers),
                "isl": S,
                "chunk": CHUNK,
                "users_per_row": N_USERS,
                "partition": os.environ.get("GLM53_PARTITION_USERS"),
                "prefill_partition": os.environ.get("GLM53_PREFILL_PARTITION", "scratch"),
                "ttft_s": ttft_s,
                "ttft_s_per_user": ttfts,
                "first_tokens": firsts,
                "chunks": chunk_log,
                "decode_step_ms": step_ms,
                "generated": gen,
                "text": out_text,
                "needle_found": found,
                "users": {
                    f"r{r}u{u}": {"generated": gens[(r, u)], "text": texts[(r, u)], "needle_found": founds[(r, u)]}
                    for r, u in live
                },
            },
            indent=1,
        )
    )
    if os.environ.get("GLM53_LONG_ASSERT", "0") == "1":
        assert all(founds.values()), texts
