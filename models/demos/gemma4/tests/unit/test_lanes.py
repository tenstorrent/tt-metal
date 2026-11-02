# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lane-sharded e2e: one user per lane with its own KV, prefilled per user and decoded as one lane-major
global batch; each lane must answer its own question, so cross-lane KV or logit mixing fails."""

import os
import random
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import (
    lanes_fabric_config,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.lane_prefill import check_lane_decode_frame
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import TestFactory, lanes_fabric_config, parametrize_mesh_with_fabric, skip_if_not_moe
from .test_layer import (
    _create_gemma4_model_args,
    _create_hf_reference_layer,
    _create_hf_text_config,
    _hf_state_to_tt_state,
    _reduced_experts,
)

USERS = [
    ("What is the capital of France? Answer in one word.", "paris"),
    ("What is 2+2? Answer with just the number.", "4"),
    ("What is the chemical symbol for gold? Answer with just the symbol.", "au"),
    ("What is the opposite of hot? Answer in one word.", "cold"),
]
GEN_TOKENS = 24
SLOTS_PER_LANE = 32


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_e2e(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    # Per-column pool: block 0 is the lane scratch, then 32 slots per lane.
    blocks_per_user = 4  # 256 tokens: short chat prompts + generation
    pool_blocks = 1 + SLOTS_PER_LANE * blocks_per_user
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=pool_blocks)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE,
        max_seq_len=4096,
        paged_attention_config=paged_cfg,
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    lanes = mesh_cfg.lanes
    assert lanes == 4
    B_g = lanes * SLOTS_PER_LANE

    # Encode with the chat template; raw completions on instruct models look degenerate by construction.
    prompts_tok = []
    for q, _ in USERS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and isinstance(ids[0], str):  # some wrappers return tokens, not ids
            ids = tokenizer.convert_tokens_to_ids(ids)
        prompts_tok.append(torch.tensor(ids, dtype=torch.long))

    # One user per lane, slot 0: global row = lane * 32.
    rows = [lane * SLOTS_PER_LANE for lane in range(lanes)]
    page_rows = {}
    for lane, row in enumerate(rows):
        ids = torch.arange(1, 1 + blocks_per_user, dtype=torch.int32)  # slot 0 of this lane's pool
        page_rows[row] = ids

    # ── Prefill each user into its lane slot (B=1 batched path per call) ──
    positions = {}
    cur_tok = {}
    for i, row in enumerate(rows):
        t = prompts_tok[i]
        plen = int(t.shape[-1])
        padded = 1 << max(int(plen - 1).bit_length(), 7)
        tok1 = torch.zeros(1, padded, dtype=torch.long)
        tok1[0, :plen] = t
        out = generator.prefill_forward_text(
            tok1,
            page_table=page_rows[row].unsqueeze(0),
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([plen]),
            empty_slots=[row],  # global slot = decode row (block convention: lane = slot // 32)
            enable_trace=False,
            sampling_params=None,
            warmup_prefill=False,
        )
        lg = out[0] if isinstance(out, (list, tuple)) else out
        positions[row] = plen
        cur_tok[row] = int(torch.argmax(lg.reshape(-1)).item())

    # ── Lane-major decode loop (host greedy) ──────────────────────────────
    # The prefill logits already produced each user's first token; a model that answers in one token
    # (26B-A4B: "Paris" then <end_of_turn>) only shows it here.
    outputs = {row: [cur_tok[row]] for row in rows}
    for _step in range(GEN_TOKENS):
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, blocks_per_user, dtype=torch.int32)
        for row in rows:
            tokens_g[row] = cur_tok[row]
            pos_g[row] = positions[row]
            pt_g[row] = page_rows[row]
        logits = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=False,
            read_from_device=True,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        lg = logits[0] if isinstance(logits, (list, tuple)) else logits
        lg = lg.reshape(B_g, -1)
        for row in rows:
            nxt = int(torch.argmax(lg[row]).item())
            outputs[row].append(nxt)
            cur_tok[row] = nxt
            positions[row] += 1

    # ── Oracle: each lane answered ITS OWN question ───────────────────────
    for i, row in enumerate(rows):
        text = tokenizer.decode(outputs[row], skip_special_tokens=True)
        expect = USERS[i][1]
        print(f"[lane {i}] {USERS[i][0]!r} -> {text!r}")
        assert expect in text.lower(), f"lane {i} answer missing {expect!r}: {text!r}"


# ===== merged from tests/unit/test_lanes_capacity.py: Lane capacity: fill all 128 slots (32 per lane x 4 lanes) by lane-parallel prefill and decode the full
QA = [
    ("What is the capital of France? Answer in one word.", "paris"),
    ("What is 2+2? Answer with just the number.", "4"),
    ("What is the chemical symbol for gold? Answer with just the symbol.", "au"),
    ("What is the opposite of hot? Answer in one word.", "cold"),
]


GEN_TOKENS_lanes_capacity = 24


SLOTS_PER_LANE_lanes_capacity = 32


ENABLE_TRACE = os.environ.get("G4_CAP_TRACE", "0") == "1"


DEVICE_SAMPLE = os.environ.get("G4_CAP_DEVSAMPLE", "0") == "1"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_capacity(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"

    from models.common.sampling.sampling_params import SamplingParams
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    blocks_per_user = 4  # 256 tokens: prompt <=128 + generation
    pool_blocks = 1 + SLOTS_PER_LANE_lanes_capacity * blocks_per_user
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=pool_blocks)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE_lanes_capacity,
        max_seq_len=4096,
        paged_attention_config=paged_cfg,
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    lanes = mesh_cfg.lanes
    B_g = lanes * SLOTS_PER_LANE_lanes_capacity  # 128

    toks = []
    for q, _ in QA:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and isinstance(ids[0], str):
            ids = tokenizer.convert_tokens_to_ids(ids)
        toks.append(torch.tensor(ids, dtype=torch.long))
    plens_qa = [int(t.shape[-1]) for t in toks]
    padded = 1 << max(int(max(plens_qa) - 1).bit_length(), 7)

    # Block convention: global slot u is the decode row, lane = u // 32, per-lane slot = u % 32.
    # User u asks question u % 4, so each lane's slots cycle all four answers.
    def lane_of(u):
        return u // SLOTS_PER_LANE_lanes_capacity

    def slot_of(u):
        return u % SLOTS_PER_LANE_lanes_capacity

    def row_of(u):
        return u

    def blocks_of(u):
        s = slot_of(u)
        return torch.arange(1 + s * blocks_per_user, 1 + (s + 1) * blocks_per_user, dtype=torch.int32)

    # ── Prefill all 128 users: 32 lane-parallel rounds of 4 ───────────────
    n_users = B_g
    positions, cur_tok = {}, {}
    t0 = time.perf_counter()
    total_prefill_tokens = 0
    for r in range(SLOTS_PER_LANE_lanes_capacity):
        group = [lane * SLOTS_PER_LANE_lanes_capacity + r for lane in range(lanes)]  # one user per lane
        tokens_l = torch.zeros(lanes, padded, dtype=torch.long)
        tables_l = torch.zeros(lanes, 1, blocks_per_user, dtype=torch.int32)
        plens_l = []
        for u in group:
            q = u % len(QA)
            tokens_l[lane_of(u), : plens_qa[q]] = toks[q]
            tables_l[lane_of(u), 0] = blocks_of(u)
            plens_l.append(plens_qa[q])
            total_prefill_tokens += plens_qa[q]
        logits = generator.prefill_forward_lanes(tokens_l, tables_l, tt_kv_cache, plens_l)
        for u in group:
            positions[u] = plens_l[lane_of(u)]
            cur_tok[u] = int(torch.argmax(logits[lane_of(u)]).item())
    prefill_s = time.perf_counter() - t0
    print(
        f"[capacity] prefill {n_users} users in {prefill_s:.2f}s "
        f"({n_users / lanes:.0f} lane-parallel rounds, {total_prefill_tokens / prefill_s:.0f} tok/s)"
    )

    # ── Decode the full 128-row global batch ───────────────────────────────
    outputs = {u: [] for u in range(n_users)}
    step_times = []
    for step in range(GEN_TOKENS_lanes_capacity):
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, blocks_per_user, dtype=torch.int32)
        for u in range(n_users):
            r = row_of(u)
            tokens_g[r] = cur_tok[u]
            pos_g[r] = positions[u]
            pt_g[r] = blocks_of(u)
        sp = SamplingParams(temperature=0.0, top_k=1, top_p=1.0) if DEVICE_SAMPLE else None
        t0 = time.perf_counter()
        ret = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=ENABLE_TRACE,
            read_from_device=True,
            sampling_params=sp,
        )
        out0 = ret[0] if isinstance(ret, (list, tuple)) else ret
        if DEVICE_SAMPLE:
            next_tok = out0.reshape(-1)[:B_g].to(torch.long)
        else:
            lg = out0.reshape(B_g, -1)
        step_times.append(time.perf_counter() - t0)
        for u in range(n_users):
            if DEVICE_SAMPLE:
                nxt = int(next_tok[row_of(u)].item())
            else:
                nxt = int(torch.argmax(lg[row_of(u)]).item())
            outputs[u].append(nxt)
            cur_tok[u] = nxt
            positions[u] += 1

    steady = sorted(step_times[3:])[len(step_times[3:]) // 2]
    print(
        f"[capacity] decode step median {steady * 1e3:.1f}ms -> "
        f"{B_g / steady:.0f} tok/s aggregate at {B_g} users (host-loop"
        f"{', traced' if ENABLE_TRACE else ', eager'}{', device-sampled' if DEVICE_SAMPLE else ''})"
    )

    # ── Oracle: all 128 rows answer their own question ─────────────────────
    bad = []
    for u in range(n_users):
        text = tokenizer.decode(outputs[u], skip_special_tokens=True).lower()
        expect = QA[u % len(QA)][1]
        if expect not in text:
            bad.append((u, lane_of(u), slot_of(u), expect, text[:60]))
    for u in (0, 1, 2, 3, 124, 125, 126, 127):
        text = tokenizer.decode(outputs[u], skip_special_tokens=True)
        print(f"[user {u:3d} lane {lane_of(u)} slot {slot_of(u):2d}] -> {text!r}")
    assert not bad, f"{len(bad)}/{n_users} users missing their answer; first: {bad[:4]}"
    print(f"[capacity] oracle: {n_users}/{n_users} users answered their own question")


# ===== merged from tests/unit/test_lane_prefill_parallel.py: Lane-parallel prefill (galaxy lanes): four users in one forward must match their serial single-user
USERS_lane_prefill_parallel = [
    "What is the capital of France? Answer in one word.",
    "What is 2+2? Answer with just the number.",
    "What is the chemical symbol for gold? Answer with just the symbol.",
    "What is the opposite of hot? Answer in one word.",
]


SLOTS_PER_LANE_lane_prefill_parallel = 32


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lane_prefill_parallel(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    # CP prefill splits ONE user's sequence over the lane axis; lane-parallel
    # prefill gives each lane its own user. Mutually exclusive per call.
    os.environ["GEMMA4_CP_PREFILL"] = "0"

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    blocks_per_user = 33  # 2112 tokens: covers the timing ISL below
    pool_blocks = 1 + SLOTS_PER_LANE_lane_prefill_parallel * blocks_per_user
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=pool_blocks)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE_lane_prefill_parallel,
        max_seq_len=4096,
        paged_attention_config=paged_cfg,
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    lanes = mesh_cfg.lanes
    assert lanes == len(USERS_lane_prefill_parallel)
    vocab = generator.model[0].vocab_size

    prompts_tok = []
    for q in USERS_lane_prefill_parallel:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and isinstance(ids[0], str):
            ids = tokenizer.convert_tokens_to_ids(ids)
        prompts_tok.append(torch.tensor(ids, dtype=torch.long))

    block_ids = torch.arange(1, 1 + blocks_per_user, dtype=torch.int32)  # slot 0 of each lane's pool

    def serial_prefill(toks_list, plens, padded):
        """4x the proven single-user path (lane = slot % lanes)."""
        outs = []
        for i, t in enumerate(toks_list):
            tok1 = torch.zeros(1, padded, dtype=torch.long)
            tok1[0, : plens[i]] = t[: plens[i]]
            out = generator.prefill_forward_text(
                tok1,
                page_table=block_ids.unsqueeze(0),
                kv_cache=tt_kv_cache,
                prompt_lens=torch.tensor([plens[i]]),
                empty_slots=[i * SLOTS_PER_LANE_lane_prefill_parallel],  # lane i's slot 0 (block convention)
                enable_trace=False,
                sampling_params=None,
                warmup_prefill=False,
            )
            lg = out[0] if isinstance(out, (list, tuple)) else out
            outs.append(lg.reshape(-1)[:vocab].float())
        return outs

    def lane_prefill(toks_list, plens, padded):
        tokens4 = torch.zeros(lanes, padded, dtype=torch.long)
        for i, t in enumerate(toks_list):
            tokens4[i, : plens[i]] = t[: plens[i]]
        tables = block_ids.reshape(1, 1, -1).repeat(lanes, 1, 1).clone()
        return generator.prefill_forward_lanes(tokens4, tables, tt_kv_cache, plens)

    # ── Correctness on the real chat prompts ──────────────────────────────
    plens = [int(t.shape[-1]) for t in prompts_tok]
    assert len({(p - 1) // 32 for p in plens}) == 1, "test prompts must share a last tile"
    padded = 1 << max(int(max(plens) - 1).bit_length(), 7)

    serial_logits = serial_prefill(prompts_tok, plens, padded)
    lane_logits = lane_prefill(prompts_tok, plens, padded)

    for i in range(lanes):
        s_arg = int(torch.argmax(serial_logits[i]).item())
        l_arg = int(torch.argmax(lane_logits[i]).item())
        pcc = _pcc(serial_logits[i], lane_logits[i])
        print(f"[lane {i}] serial argmax={s_arg} lane argmax={l_arg} pcc={pcc:.5f}")
        assert pcc > 0.99, f"lane {i} logits diverge from serial prefill: pcc={pcc}"
        assert s_arg == l_arg, f"lane {i} next-token mismatch: serial {s_arg} vs lane {l_arg}"

    # ── k = 2 users per lane as one batched stream per column ─────────────
    # Round r shares one prompt length across lanes: truncate the prompts to the shortest and rotate them
    # so lane i prefills prompts i and (i + 1) % lanes at its slots 0 and 1. Serial references on the
    # same truncated prompts.
    L = min(plens)
    trunc = [t[:L] for t in prompts_tok]
    padded2 = 1 << max(int(L - 1).bit_length(), 7)
    assert 2 * padded2 <= int(generator.model_args[0].max_prefill_chunk_size), "k=2 stream must fit the chunk"
    serial2 = serial_prefill(trunc, [L] * lanes, padded2)
    tokens_k = torch.zeros(lanes, 2, padded2, dtype=torch.long)
    tables_k = torch.zeros(lanes, 2, blocks_per_user, dtype=torch.int32)
    for i in range(lanes):
        for r in range(2):
            tokens_k[i, r, :L] = trunc[(i + r) % lanes]
            tables_k[i, r] = block_ids + r * blocks_per_user  # slot r of lane i's pool
    lane_k = generator.prefill_forward_lanes(
        tokens_k,
        tables_k,
        tt_kv_cache,
        [L] * (lanes * 2),
        slot_ids=[i * SLOTS_PER_LANE_lane_prefill_parallel + r for i in range(lanes) for r in range(2)],
    )
    assert tuple(lane_k.shape) == (lanes * 2, vocab)
    for i in range(lanes):
        for r in range(2):
            ref = serial2[(i + r) % lanes]
            got = lane_k[i * 2 + r]
            s_arg, l_arg = int(torch.argmax(ref).item()), int(torch.argmax(got).item())
            pcc = _pcc(ref, got)
            print(f"[lane {i} user {r}] serial argmax={s_arg} batched argmax={l_arg} pcc={pcc:.5f}")
            assert pcc > 0.99, f"lane {i} user {r}: batched lane prefill diverges from serial: pcc={pcc}"
            assert s_arg == l_arg, f"lane {i} user {r}: next-token mismatch: serial {s_arg} vs batched {l_arg}"


# ===== merged from tests/unit/test_lane_prefill_trace.py: The traced lane prefill on the lane engine: the 1x128 lane frame captured at boot and replayed for the 32
SLOTS_PER_LANE_lane_prefill_trace = 32


BLOCKS_PER_USER = 4  # 256 tokens at block 64: the 128-token frame plus headroom


BUCKETS = [(1, 128), (8, 128), (32, 128)]  # rounds per column x padded length; 32x128 = the 4096-token stream


COUNTRIES = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
    ("Egypt", "Cairo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Greece", "Athens"),
    ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Hungary", "Budapest"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Belgium", "Brussels"), ("Russia", "Moscow"), ("China", "Beijing"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Canada", "Ottawa"), ("Cuba", "Havana"),
    ("Chile", "Santiago"), ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Nigeria", "Abuja"), ("Morocco", "Rabat"),
]  # fmt: skip


def _pcc_lane_prefill_trace(a, b):
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


MESH_PARAMS = dict(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={
        "fabric_config": lanes_fabric_config(),
        "l1_small_size": 65536,
        "trace_region_size": 1_200_000_000,  # the Galaxy entry's region: the 32x128 frame is ~0.5 GB
    },
)


def _lane_engine(mesh_device, monkeypatch, *, max_seq_len, pool_blocks, slots=SLOTS_PER_LANE_lane_prefill_trace):
    """The lane engine on this mesh (skips off the (2,4)/(8,4) shapes and without the lanes fabric): the 26B model
    with ``slots`` slots per lane, a ``max_seq_len`` context and a per-column paged pool of ``pool_blocks`` blocks
    of 64 tokens (block 0 is the lane scratch)."""
    skip_if_config_only_checkpoint()
    shape = tuple(int(x) for x in mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    if shape not in ((2, 4), (8, 4)):
        pytest.skip(f"the lane engine's shapes are (2,4) and (8,4); got {shape}")
    try:
        fabric = str(ttnn.get_fabric_config())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"fabric config unavailable: {exc}")
    if "FABRIC_1D" not in fabric:
        pytest.skip(f"lanes fabric (FABRIC_1D / FABRIC_1D_RING) not active: {fabric}")
    for k, v in {"GEMMA4_GALAXY_FRACTURE": "1", "GEMMA4_GALAXY_LANES": "1", "GEMMA4_CP_PREFILL": "0"}.items():
        monkeypatch.setenv(k, v)
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it")
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=int(pool_blocks))
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=int(slots),
        max_seq_len=int(max_seq_len),
        paged_attention_config=paged_cfg,
    )
    model = generator.model[0]
    mesh_cfg = model.mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    block_size = generator._effective_paged_block_size(
        tt_kv_cache[0] if isinstance(tt_kv_cache[0], list) else tt_kv_cache
    )
    return SimpleNamespace(
        generator=generator,
        kv=tt_kv_cache,
        tokenizer=tokenizer,
        model=model,
        lanes=int(mesh_cfg.lanes),
        vocab=model.vocab_size,
        block_size=block_size,
    )


def _lane_setup(mesh_device, monkeypatch, frame_len):
    """The lane engine (``_lane_engine``) with slots for ``SLOTS_PER_LANE_lane_prefill_trace`` users per lane, and
    the 32 coherence questions padded to one length that fits the ``frame_len``-token frame."""
    e = _lane_engine(
        mesh_device,
        monkeypatch,
        max_seq_len=4096,
        pool_blocks=1 + SLOTS_PER_LANE_lane_prefill_trace * BLOCKS_PER_USER,
    )
    generator, tt_kv_cache, tokenizer, model = e.generator, e.kv, e.tokenizer, e.model
    lanes, vocab, block_size = e.lanes, e.vocab, e.block_size

    # the 32 coherence questions, every prompt padded to one token length after the user-turn header
    def ids_for(country):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"What is the capital of {country}? Answer in one word."}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.convert_tokens_to_ids(t) if isinstance(t, str) else t for t in ids]

    nl = tokenizer.encode("\n", add_special_tokens=False)[-1]
    prompts = [ids_for(c) for c, _ in COUNTRIES]
    L = max(len(p) for p in prompts)
    assert L <= frame_len, f"the questions must fit the {frame_len}-token frame, longest is {L}"
    padded = []
    for p in prompts:
        cut = p.index(nl) + 1 if nl in p else 1
        padded.append(p[:cut] + [nl] * (L - len(p)) + p[cut:])

    return SimpleNamespace(
        generator=generator,
        kv=tt_kv_cache,
        tokenizer=tokenizer,
        model=model,
        lanes=lanes,
        vocab=vocab,
        block_size=block_size,
        padded=padded,
        L=L,
    )


@parametrize_mesh_with_fabric(**MESH_PARAMS)
@pytest.mark.parametrize("bucket", BUCKETS, ids=[f"b{k}x{n}" for k, n in BUCKETS])
def test_lane_prefill_trace_matches_eager(mesh_device, reset_seeds, monkeypatch, bucket):
    s = _lane_setup(mesh_device, monkeypatch, bucket[1])
    generator, tt_kv_cache, tokenizer, model = s.generator, s.kv, s.tokenizer, s.model
    lanes, vocab, block_size, padded, L = s.lanes, s.vocab, s.block_size, s.padded, s.L
    k, S = bucket
    try:
        assert generator._g4_capture_lane_prefill_trace(bucket, tt_kv_cache), "capture failed"
    except Exception as exc:  # noqa: BLE001
        if k > 1 and "trace" in str(exc).lower():
            pytest.skip(f"the {k}x{S} frame does not fit this trace region: {exc}")
        raise
    entry = generator._g4_lane_trace_registry()[bucket]
    logger.info(f"LANE_TRACE captured {bucket} widths={entry['widths'][:3]}... needed_blocks={entry['needed_blocks']}")

    # Every call fills k rounds x lanes rows with the questions in row-major order (cycling past 32 for the 32-round
    # frame): row (lane, r) holds question (call * k + r) * lanes + lane, at slot r of its lane.
    rows_per_call = k * lanes
    n_rows = max(len(COUNTRIES), rows_per_call)
    n_calls = (n_rows + rows_per_call - 1) // rows_per_call

    def frame_for(call):
        tokens = torch.zeros(lanes, k, S, dtype=torch.long)
        tables = torch.zeros(lanes, k, BLOCKS_PER_USER, dtype=torch.int32)
        qs = {}
        for r in range(k):
            for lane in range(lanes):
                q = ((call * k + r) * lanes + lane) % len(COUNTRIES)
                qs[(lane, r)] = q
                tokens[lane, r, :L] = torch.tensor(padded[q], dtype=torch.long)
                tables[lane, r] = torch.arange(
                    1 + r * BLOCKS_PER_USER, 1 + (r + 1) * BLOCKS_PER_USER, dtype=torch.int32
                )
        slots = [ln * SLOTS_PER_LANE_lane_prefill_trace + r for ln in range(lanes) for r in range(k)]
        per_layer = generator._g4_lane_frame_page_tables(model, tables, slots, block_size)
        return tokens, tables, per_layer, slots, qs

    # The replay projects with the gathered lm_head (one tile, one readback), the eager walk with the per-round
    # form, so the parity also checks the gather's row arithmetic; the decoded token must start the capital.
    mismatches, pccs, checked, hits = [], [], 0, 0
    for call in range(n_calls):
        tokens, tables, per_layer, slots, qs = frame_for(call)
        plens = [L] * (lanes * k)
        monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "1")
        traced = generator._g4_lane_prefill_trace_replay(bucket, tokens, tables, per_layer, tt_kv_cache, plens)
        assert traced is not None, f"call {call}: the replay fell back to eager"
        monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "0")
        eager = generator.prefill_forward_lanes(
            tokens if k > 1 else tokens[:, 0], tables, tt_kv_cache, plens, slot_ids=slots
        )
        for (lane, r), q in qs.items():
            row = lane * k + r
            t_arg, e_arg = int(torch.argmax(traced[row]).item()), int(torch.argmax(eager[row]).item())
            pcc = _pcc_lane_prefill_trace(traced[row][:vocab], eager[row][:vocab])
            pccs.append(pcc)
            checked += 1
            word = tokenizer.decode([t_arg]).strip()
            hits += int(bool(word) and COUNTRIES[q][1].lower().startswith(word.lower()))
            logger.info(
                f"[call {call} lane {lane} round {r}] {COUNTRIES[q][0]}: traced {t_arg} ({word!r}) eager {e_arg} pcc={pcc:.5f}"
            )
            if t_arg != e_arg:
                mismatches.append((call, lane, r, COUNTRIES[q][0], t_arg, e_arg))
    logger.info(
        f"LANE_TRACE parity {k}x{S}: {checked - len(mismatches)}/{checked} token-exact, min pcc {min(pccs):.5f} "
        f"over {n_calls} call(s)"
    )
    logger.info(f"LANE_TRACE capitals {k}x{S}: {hits}/{checked} first tokens start the capital")
    assert not mismatches, f"traced vs eager first token differs: {mismatches}"
    assert min(pccs) > 0.99
    assert hits >= 0.9 * checked, f"only {hits}/{checked} first tokens start the capital (window 11: 32/32)"


def _lane_trace_programs(generator, model, entry, k, kv):
    """Device programs of the frame's trace body (one eager pass of ``_g4_lane_trace_forward`` under the graph
    capture): the launch-bound floor of a replay is this count times the per-program launch cost."""
    model_kv = kv[0] if isinstance(kv, (list, tuple)) else kv  # the model's per-layer list, as the capture unwraps it
    # the body runs under the capture's mode (the forward returns post-norm hidden states, lm_head deferred to the
    # body's own gather); outside it the forward projects every row itself and the gather would see [k * S, vocab]
    model._prefill_trace_mode = True
    try:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = generator._g4_lane_trace_forward(model, entry["inputs"], k, model_kv)
        ttnn.synchronize_device(model.mesh_device)
        captured = ttnn.graph.end_graph_capture()
    finally:
        model._prefill_trace_mode = False
    out.deallocate(True)
    names = [
        str(n["params"].get("name", "?")).replace("DeviceOperation", "")
        for n in captured
        if n["node_type"] == "function_start" and str(n["params"].get("name", "")).endswith("DeviceOperation")
    ]
    hist = {}
    for n in names:
        hist[n] = hist.get(n, 0) + 1
    ccl = sum(
        v for n, v in hist.items() if any(x in n for x in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))
    )
    top = ", ".join(f"{n}={v}" for n, v in sorted(hist.items(), key=lambda kv: -kv[1]))
    return len(names), ccl, top


@parametrize_mesh_with_fabric(**MESH_PARAMS)
def test_lane_prefill_trace_call_walls(mesh_device, reset_seeds, monkeypatch):
    """The lane frames' per-call walls by host phase (install / stage / replay / readback; the replay synchronized
    under GEMMA4_LANE_TRACE_SYNC_PROFILE=1 so it is the device wall) over the served call shapes: the c1 frame
    (one prompt lane of four), full and padded 8x128 / 32x128 calls, every shape twice so the first call shows a
    compile and the second the steady state (window 12: the 8x128 calls of 4 / 5 / 7 real rounds cost 860-1125 ms
    against 295 for 8, the eager gathered lm_head compiling per real-round count); plus the trace body's program
    count and, per shape, the real cells' first tokens against the eager walk (the gather index for padded calls).
    """
    from models.demos.gemma4.tt.lane_prefill import lane_trace_readback_rows

    monkeypatch.setenv("GEMMA4_LANE_TRACE_SYNC_PROFILE", "1")
    monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "0")  # the eager walk's per-round form
    s = _lane_setup(mesh_device, monkeypatch, 128)
    generator, tt_kv_cache, tokenizer, model = s.generator, s.kv, s.tokenizer, s.model
    lanes, block_size, padded, L = s.lanes, s.block_size, s.padded, s.L
    captured = []
    for bucket in BUCKETS:
        try:
            assert generator._g4_capture_lane_prefill_trace(bucket, tt_kv_cache), "capture failed"
            captured.append(bucket)
        except Exception as exc:  # noqa: BLE001
            if bucket[0] > 1 and "trace" in str(exc).lower():
                logger.info(
                    f"LANE_TRACE walls: the {bucket[0]}x{bucket[1]} frame does not fit this trace region: {exc}"
                )
                continue
            raise
    reg = generator._g4_lane_trace_registry()
    in_trace = all(reg[b].get("logits_in_trace") for b in captured)
    logger.info(f"LANE_TRACE walls: frames {captured} lm_head_in_trace={in_trace} mesh={tuple(mesh_device.shape)}")
    for bucket in captured:
        n_prog, n_ccl, top = _lane_trace_programs(generator, model, reg[bucket], bucket[0], tt_kv_cache)
        logger.info(f"LANE_TRACE programs {bucket[0]}x{bucket[1]}: total={n_prog} ccl={n_ccl} | {top}")

    def frame(k, S, real_rounds, real_lanes, call=0):
        tokens = torch.zeros(lanes, k, S, dtype=torch.long)
        tables = torch.zeros(lanes, k, BLOCKS_PER_USER, dtype=torch.int32)
        qs = {}
        for r in range(k):
            for lane in range(lanes):
                tables[lane, r] = torch.arange(
                    1 + r * BLOCKS_PER_USER, 1 + (r + 1) * BLOCKS_PER_USER, dtype=torch.int32
                )
                if r in real_rounds and lane in real_lanes:
                    q = ((call * k + r) * lanes + lane) % len(COUNTRIES)
                    qs[(lane, r)] = q
                    tokens[lane, r, :L] = torch.tensor(padded[q], dtype=torch.long)
        slots = [ln * SLOTS_PER_LANE_lane_prefill_trace + r for ln in range(lanes) for r in range(k)]
        per_layer = generator._g4_lane_frame_page_tables(model, tables, slots, block_size)
        return tokens, tables, per_layer, slots, qs

    walls = []

    def run(bucket, real_rounds, real_lanes, *, check):
        k, S = bucket
        tokens, tables, per_layer, slots, qs = frame(k, S, real_rounds, real_lanes)
        plens = [L] * (lanes * k)
        model._g4_pf_ms = {}
        kw = {"rounds": list(real_rounds)} if len(real_rounds) < k else {}
        if len(real_lanes) < lanes:
            kw["lanes"] = list(real_lanes)
        t0 = time.perf_counter()
        traced = generator._g4_lane_prefill_trace_replay(bucket, tokens, tables, per_layer, tt_kv_cache, plens, **kw)
        total = (time.perf_counter() - t0) * 1e3
        assert traced is not None, f"{k}x{S}: the replay fell back to eager"
        ph = dict(model._g4_pf_ms)
        n = len(real_rounds)
        line = (
            f"LANE_TRACE walls {k}x{S} n={n} lanes={len(real_lanes)} rows_read={lane_trace_readback_rows(n)}: "
            f"total={total:.1f} install={ph.get('install', 0):.1f}x{int(round(ph.get('install_n', 0)))} "
            f"stage={ph.get('stage', 0):.1f} replay={ph.get('replay', 0):.1f} gather={ph.get('gather', 0):.1f} "
            f"readback={ph.get('readback', 0):.1f} ms"
        )
        logger.info(line)
        walls.append((k, S, n, len(real_lanes), total, ph))
        if check:
            eager = generator.prefill_forward_lanes(
                tokens if k > 1 else tokens[:, 0], tables, tt_kv_cache, plens, slot_ids=slots
            )
            bad, hits = [], 0
            for (lane, r), q in qs.items():
                row = lane * k + r
                t_arg, e_arg = int(torch.argmax(traced[row]).item()), int(torch.argmax(eager[row]).item())
                word = tokenizer.decode([t_arg]).strip()
                hits += int(bool(word) and COUNTRIES[q][1].lower().startswith(word.lower()))
                if t_arg != e_arg:
                    bad.append((lane, r, COUNTRIES[q][0], t_arg, e_arg))
            logger.info(
                f"LANE_TRACE walls parity {k}x{S} n={n} lanes={len(real_lanes)}: {len(qs) - len(bad)}/{len(qs)} token-exact, capitals {hits}/{len(qs)}"
            )
            assert not bad, f"{k}x{S} n={n}: traced vs eager first token differs: {bad}"
            assert hits >= 0.9 * len(qs), f"{k}x{S} n={n}: only {hits}/{len(qs)} first tokens start the capital"

    all_lanes = list(range(lanes))
    if (1, 128) in captured:
        for i in range(3):
            run((1, 128), [0], [0], check=i == 0)  # the served c1 frame: one prompt lane
        for i in range(2):
            run((1, 128), [0], all_lanes, check=i == 0)
    if (8, 128) in captured:
        for i in range(3):
            run((8, 128), list(range(8)), all_lanes, check=i == 0)
        for n in (4, 5, 7):
            for i in range(2):
                run((8, 128), list(range(n)), all_lanes, check=i == 0)
    if (32, 128) in captured:
        for i in range(2):
            run((32, 128), list(range(32)), all_lanes, check=i == 0)
        for n in (29, 25):
            for i in range(2):
                run((32, 128), list(range(n)), all_lanes, check=i == 0 and n == 29)
    # the steady state must not depend on the real-round count: the second call of every padded shape within
    # 1.5x of the second full call of its frame (window 12: 3-4x)
    by = {}
    for k, S, n, nl, total, ph in walls:
        by.setdefault((k, nl), []).append((n, total))
    for (k, nl), rows in by.items():
        seconds = {}
        for n, total in rows:
            seconds.setdefault(n, []).append(total)
        full = seconds.get(k)
        if not full or len(full) < 2:
            continue
        ref = min(full[1:])
        for n, ts in seconds.items():
            if n != k and len(ts) >= 2:
                assert (
                    min(ts[1:]) <= 1.5 * ref + 20
                ), f"{k}x128 n={n}: steady call {min(ts[1:]):.0f} ms vs full {ref:.0f}"


@parametrize_mesh_with_fabric(**MESH_PARAMS)
def test_lane_prefill_trace_single_prompt_after_a_wave(mesh_device, reset_seeds, monkeypatch):
    """The served sequence window 9 died in (2026-10-25 03:26, the MoE combine of the lane trace on half the rows):
    a wave of full-frame replays (the c128 bench's gathered steps), the decode frame's own per-layer tables written
    in between (another key: the realloc the boot log reports), then one short prompt alone in one lane (B=1 <
    lanes: the frame with three scratch lanes, the prompt at a non-zero slot). Traced vs eager, token-exact. Then
    the guard: a per-layer install wider than the capture grows the key's buffers, and the next replay refuses and
    drops the frame instead of reading the freed buffer."""
    from models.demos.gemma4.tt.lane_prefill import fit_table_width

    s = _lane_setup(mesh_device, monkeypatch, 128)
    generator, model, lanes, L = s.generator, s.model, s.lanes, s.L
    bucket = (1, 128)
    assert generator._g4_capture_lane_prefill_trace(bucket, s.kv), "capture failed"
    entry = generator._g4_lane_trace_registry()[bucket]
    assert entry["tables"] is not None, "the capture records the key's persistent buffers"
    width = max(w for w in entry["widths"] if w)
    logger.info(f"LANE_TRACE captured {bucket} widths={entry['widths'][:3]}... rows={entry['rows']} width={width}")

    def frame(rows):
        """A 1-round frame: ``rows`` maps lane -> (question, slot); other lanes are scratch (null block 0)."""
        tokens = torch.zeros(lanes, 1, 128, dtype=torch.long)
        tables = torch.zeros(lanes, 1, BLOCKS_PER_USER, dtype=torch.int32)
        slots = [ln * SLOTS_PER_LANE_lane_prefill_trace for ln in range(lanes)]
        for ln, (q, slot) in rows.items():
            tokens[ln, 0, :L] = torch.tensor(s.padded[q], dtype=torch.long)
            tables[ln, 0] = torch.arange(1, 1 + BLOCKS_PER_USER, dtype=torch.int32)
            slots[ln] = slot
        per_layer = generator._g4_lane_frame_page_tables(model, tables, slots, s.block_size)
        return tokens, tables, per_layer, slots

    def check(label, rows, traced, eager):
        bad, pccs, hits = [], [], 0
        for ln, (q, _slot) in rows.items():
            t_arg, e_arg = int(torch.argmax(traced[ln]).item()), int(torch.argmax(eager[ln]).item())
            pccs.append(_pcc_lane_prefill_trace(traced[ln][: s.vocab], eager[ln][: s.vocab]))
            word = s.tokenizer.decode([t_arg]).strip()
            hits += int(bool(word) and COUNTRIES[q][1].lower().startswith(word.lower()))
            if t_arg != e_arg:
                bad.append((ln, COUNTRIES[q][0], t_arg, e_arg))
        logger.info(
            f"LANE_TRACE {label} parity: {len(rows) - len(bad)}/{len(rows)} token-exact, pcc {min(pccs):.5f}, "
            f"capitals {hits}/{len(rows)}"
        )
        assert not bad, f"{label}: traced vs eager first token differs: {bad}"
        assert min(pccs) > 0.99
        assert hits == len(rows), f"{label}: {hits}/{len(rows)} first tokens start the capital"

    def replay(tokens, tables, per_layer, plens, **kw):
        monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "1")  # the gathered lm_head on the traced side
        return generator._g4_lane_prefill_trace_replay(bucket, tokens, tables, per_layer, s.kv, plens, **kw)

    def eager_walk(tokens, tables, plens, slots):
        monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "0")  # the per-round form on the eager side
        return generator.prefill_forward_lanes(tokens[:, 0], tables, s.kv, plens, slot_ids=slots)

    # the wave: every lane with a prompt, the 32 questions over eight replays (the c128 bench's steps)
    wave_bad = 0
    for call in range(len(COUNTRIES) // lanes):
        rows = {ln: (call * lanes + ln, ln * SLOTS_PER_LANE_lane_prefill_trace) for ln in range(lanes)}
        tokens, tables, per_layer, slots = frame(rows)
        traced = replay(tokens, tables, per_layer, [L] * lanes)
        assert traced is not None, f"wave call {call}: the replay fell back to eager"
        eager = eager_walk(tokens, tables, [L] * lanes, slots)
        for ln, (q, _slot) in rows.items():
            wave_bad += int(torch.argmax(traced[ln]).item()) != int(torch.argmax(eager[ln]).item())
    logger.info(f"LANE_TRACE wave parity: {len(COUNTRIES) - wave_bad}/{len(COUNTRIES)} token-exact")
    assert wave_bad == 0

    # decode wrote the lanes' decode frame (its own key: the per-layer realloc the boot log reports)
    frame_tables = generator._g4_lane_frame_page_tables(
        model,
        torch.zeros(lanes, 2, BLOCKS_PER_USER, dtype=torch.int32),
        [ln * SLOTS_PER_LANE_lane_prefill_trace + r for ln in range(lanes) for r in range(2)],
        s.block_size,
    )
    generator._install_per_layer_page_tables(model, frame_tables, writer="decode")
    model._g4_persistent_alloc_event = None
    if hasattr(model, "_active_page_tables_per_layer"):
        del model._active_page_tables_per_layer

    # one short prompt alone: lane 2 at slot 5, scratch rows elsewhere
    rows = {2: (0, 2 * SLOTS_PER_LANE_lane_prefill_trace + 5)}
    tokens, tables, per_layer, slots = frame(rows)
    traced = replay(tokens, tables, per_layer, [L] * lanes, rounds=[0])
    assert traced is not None, "the single prompt's replay fell back to eager"
    eager = eager_walk(tokens, tables, [L] * lanes, slots)
    check("single-prompt", rows, traced, eager)

    # the guard: a wider install (an eager call at an uncaptured length) grows the key's buffers
    wide = [fit_table_width(t, width + 8) if t is not None else t for t in per_layer]
    model._active_page_tables_per_layer = wide
    generator._install_per_layer_page_tables(model, wide, writer="prefill")
    grown = model._g4_persistent_alloc_event
    model._g4_persistent_alloc_event = None
    del model._active_page_tables_per_layer
    refused = replay(tokens, tables, per_layer, [L] * lanes)
    logger.info(
        f"LANE_TRACE guard: grow event={grown!r} refused={refused is None} dropped={bucket not in generator._g4_lane_trace_registry()}"
    )
    assert refused is None and bucket not in generator._g4_lane_trace_registry(), "a grown key must drop the frame"
    eager2 = eager_walk(tokens, tables, [L] * lanes, slots)
    assert int(torch.argmax(eager2[2]).item()) == int(
        torch.argmax(eager[2]).item()
    ), "eager still answers after the drop"


# ===== merged from tests/unit/test_layer_lanes_traced.py: One decoder layer of the lane engine, traced: the smallest full-mesh unit for a decode-step knob.
ROWS_PER_LANE = 32


CACHE_LEN = 96


BLOCK_SIZE = 64


BLOCKS_PER_USER_layer_lanes_traced = (
    CACHE_LEN + 64 + BLOCK_SIZE - 1
) // BLOCK_SIZE  # the decode position plus headroom


REPLAYS = int(os.environ.get("LANE_LAYER_REPLAYS", "20"))


KNOBS = (
    "GEMMA4_CCL_NUM_LINKS",
    "GEMMA4_CCL_ALLREDUCE_AG",
    "GEMMA4_CCL_CHUNKS_PER_SYNC",
    "GEMMA4_CCL_NUM_WORKERS",
    "GEMMA4_CCL_NUM_BUFFERS",
    "GEMMA4_MOE_COMPUTE_CUSTOM_OP",
    "GEMMA4_NORM_FUSED",
    "GEMMA4_LAYER_NORM_FUSED",
    "GEMMA4_QKV_NORM_ROPE_FUSED",
)


def _engine_mesh_config(mesh_device):
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    cfg = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    cfg.lane_sharded = True
    return cfg


@skip_if_not_moe
@pytest.mark.skipif(not is_blackhole(), reason="the decode MoE kernel is Blackhole-only")
@parametrize_mesh_with_fabric(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={
        "fabric_config": lanes_fabric_config(),
        "l1_small_size": 65536,
        "trace_region_size": 64_000_000,
    },
)
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding_layer0"])
@pytest.mark.parametrize("rope", ["cache", "presliced"], ids=["rope_cache", "rope_presliced"])
def test_layer_lanes_traced_step(mesh_device, layer_idx, rope, reset_seeds, monkeypatch):
    # CI runs the whole dir on 4-chip, 1-chip and Wormhole boxes: skip, never error, off the lane engine's shapes
    shape = tuple(int(x) for x in mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    if shape not in ((2, 4), (8, 4)):
        pytest.skip(f"the lane engine's shapes are (2,4) and (8,4); got {shape}")
    try:
        fabric = str(ttnn.get_fabric_config())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"fabric config unavailable: {exc}")
    if "FABRIC_1D" not in fabric:
        pytest.skip(f"lanes fabric (FABRIC_1D / FABRIC_1D_RING) not active: {fabric}")
    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "1"))
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    if tuple(mesh_device.shape) == (2, 4):
        monkeypatch.setenv("GEMMA4_MOE_DECODE_FRAME", os.environ.get("GEMMA4_MOE_DECODE_FRAME", "128"))
    mesh_config = _engine_mesh_config(mesh_device)
    lanes = mesh_config.lanes
    users = lanes * ROWS_PER_LANE

    hf_text_config = _create_hf_text_config(num_experts=_reduced_experts(mesh_device), top_k=2)
    # One layer: the decode MoE module is shared by every layer of a model and built once the last one registers
    # (Gemma4MoEDecode, num_layers from the config's num_hidden_layers); a 30-layer config with one built layer never
    # builds it (window 5 arms workers4 / buffers4 / ag_sum_links1: "decode MoE module not built"). The 26B has no
    # KV-shared layers (num_kv_shared_layers 0), so a one-layer config keeps layer 0 a plain sliding layer.
    assert layer_idx == 0 and hf_text_config.layer_types[0] == "sliding_attention"
    assert int(getattr(hf_text_config, "num_kv_shared_layers", 0) or 0) == 0, "a one-layer config would share KV"
    hf_text_config.num_hidden_layers = 1
    hf_text_config.layer_types = list(hf_text_config.layer_types[:1])
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)
    tt_state = _hf_state_to_tt_state(hf_layer.state_dict(), layer_idx)
    model_args = _create_gemma4_model_args(hf_text_config)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)
    ccl_manager = CCLManager(mesh_device)  # the engine's default links
    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=CACHE_LEN + 64,
        max_local_batch_size=ROWS_PER_LANE,
    )
    # The engine's paged cache: per column block 0 is the lane scratch, then one block run per row
    # (test_decode_roofline_lanes.py's pool); the attention's heads come from mesh_config (effective block 64).
    paged_cfg = PagedAttentionConfig(
        block_size=BLOCK_SIZE, max_num_blocks=1 + ROWS_PER_LANE * BLOCKS_PER_USER_layer_lanes_traced
    )
    kv_cache = init_kv_cache(
        mesh_device,
        attn_cfg,
        max_batch_size=ROWS_PER_LANE,
        max_seq_len=CACHE_LEN + 64,
        paged_attention_config=paged_cfg,
        cache_dtype=ttnn.bfloat16,
        mesh_config=mesh_config,
    )
    tt_layer.self_attn.kv_cache = kv_cache
    moe_decode = getattr(getattr(tt_layer, "moe", None), "experts", None)
    moe_decode = getattr(moe_decode, "moe_decode", None)
    if moe_decode is not None:
        assert moe_decode.num_layers == 1 and moe_decode._reg["module"] is not None, "decode MoE module not built"

    # The decode frame exactly as prepare_decode_inputs_host hands it to the layer (lane-major batch of
    # lanes x 32 rows, each tensor lane-sharded over the columns): hidden rows on dim 2, the RoPE positions
    # [1, batch] uint32 on dim 1, the cache positions [batch] int32 on dim 0 (the paged cache update and the
    # decode SDPA read their batch from dim 0: a [1, rows] tensor here is ONE update index for 32 rows, window 5),
    # the page table [batch, blocks] int32 on dim 0. Checked on the host before anything reaches the device.
    x_torch = torch.randn(1, 1, users, model_args.hidden_size, dtype=torch.bfloat16)
    rope_pos = torch.full((1, users), CACHE_LEN, dtype=torch.int64)
    cache_pos = torch.full((users,), CACHE_LEN, dtype=torch.int32)
    page_table = torch.zeros(users, BLOCKS_PER_USER_layer_lanes_traced, dtype=torch.int32)
    for row in range(users):
        slot = row % ROWS_PER_LANE
        page_table[row] = torch.arange(
            1 + slot * BLOCKS_PER_USER_layer_lanes_traced,
            1 + (slot + 1) * BLOCKS_PER_USER_layer_lanes_traced,
            dtype=torch.int32,
        )
    columns = check_lane_decode_frame(
        lanes,
        ROWS_PER_LANE,
        hidden=tuple(x_torch.shape),
        rope_positions=tuple(rope_pos.shape),
        cache_positions=tuple(cache_pos.shape),
        page_table=tuple(page_table.shape),
    )
    logger.info(f"LANE_LAYER frame per column: {columns}")
    lane_map = lambda d: mesh_config.lane_shard_mapper(mesh_device, d)  # noqa: E731
    # The layer CONSUMES its hidden-state input (the fused norm chain frees the residual, as the engine's fresh
    # per-step embedding allows), so the persistent input is cloned per step inside the step itself -- a device
    # copy the trace records, as the engine's trace embeds its token buffer inside the trace. Window 5b: the
    # second step's first norm found the first step's freed input (x.device() None).
    x_in = ttnn.from_torch(
        x_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=lane_map(2)
    )
    rope_pos_tt = ttnn.from_torch(
        rope_pos, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=lane_map(1)
    )
    cache_pos_tt = ttnn.from_torch(
        cache_pos, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=lane_map(0)
    )
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=lane_map(0)
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, CACHE_LEN + 64, layer_idx)
    # ``rope_presliced``: the served decode loop's form (model.py: the per-layer-type cos / sin gathered once per step
    # by ``ttnn.embedding`` over the positions, handed down with ``rope_presliced=True``): the form the fused QKV
    # norm + RoPE program (GEMMA4_QKV_NORM_ROPE_FUSED) engages on; the full-cache form keeps the attention's own
    # per-user gather and never fuses.
    presliced = rope == "presliced"
    cos_2d, sin_2d = (
        TestFactory.create_tt_rope_cache_2d(mesh_device, hf_text_config, CACHE_LEN + 64, layer_idx)
        if presliced
        else (None, None)
    )

    for name, t in (
        ("hidden", x_in),
        ("rope_positions", rope_pos_tt),
        ("cache_positions", cache_pos_tt),
        ("page_table", page_table_tt),
    ):
        assert t.device() is not None and t.is_allocated(), f"{name}: not a live device tensor"

    def step():
        x_step = ttnn.clone(x_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # the layer frees this copy, not x_in
        rope_mats, freed = (cos_tt, sin_tt), ()
        if presliced:
            cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(rope_pos_tt, cos_2d, layout=ttnn.TILE_LAYOUT))
            sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(rope_pos_tt, sin_2d, layout=ttnn.TILE_LAYOUT))
            rope_mats, freed = (cos_pos, sin_pos), (cos_pos, sin_pos)
        out = tt_layer(
            x_step,
            rope_mats=rope_mats,
            position_idx=rope_pos_tt,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            is_decode=True,
            token_index=None,  # per-user RoPE from the positions (the engine passes None under its 2-D rope caches)
            position_idx_cache=cache_pos_tt,
            rope_presliced=presliced,
        )
        for t in freed:
            t.deallocate(True)
        return out

    # warm (compiles every program), then one graph-captured eager step: the program count and the collectives of
    # this arm (the inert-knob rule: an arm that leaves the collectives unchanged says so), and the eager reference
    out = step()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    for name, t in (
        ("hidden", x_in),
        ("rope_positions", rope_pos_tt),
        ("cache_positions", cache_pos_tt),
        ("page_table", page_table_tt),
    ):
        assert (
            t.is_allocated()
        ), f"{name}: the step consumed a persistent input (the engine binds these as trace inputs)"
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = step()
    ttnn.synchronize_device(mesh_device)
    captured = ttnn.graph.end_graph_capture()
    eager = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)
    dev_ops = [
        n["params"].get("name", "?")
        for n in captured
        if n["node_type"] == "function_start" and str(n["params"].get("name", "")).endswith("DeviceOperation")
    ]
    ccl_ops = [n for n in dev_ops if any(k in n for k in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))]
    ccl_counts = {}
    for n in ccl_ops:
        ccl_counts[n] = ccl_counts.get(n, 0) + 1
    knobs = " ".join(f"{k}={os.environ.get(k, '-')}" for k in KNOBS)
    ccl_line = " ".join(f"{k.replace('DeviceOperation', '')}={v}" for k, v in sorted(ccl_counts.items()))
    logger.info(f"LANE_LAYER programs/step={len(dev_ops)} ccl_ops={len(ccl_ops)} {ccl_line} | rope={rope} {knobs}")
    hist = {}
    for n in dev_ops:
        hist[n] = hist.get(n, 0) + 1
    logger.info(
        "LANE_LAYER ops: "
        + ", ".join(f"{n.replace('DeviceOperation', '')}={v}" for n, v in sorted(hist.items(), key=lambda kv: -kv[1]))
    )
    logger.info("LANE_LAYER seq: " + " ".join(n.replace("DeviceOperation", "") for n in dev_ops))

    # GEMMA4_LANE_LAYER_REF_FLAGS="NAME=VALUE ..." (an arm's job): one more eager step under those flag overrides
    # (the composite forms of the arm's fusions: decode_flag reads a set env at call time) as the numerics reference
    # of the arm's eager step, so a fusion is gated on the lane frame before any served use
    ref_flags = os.environ.get("GEMMA4_LANE_LAYER_REF_FLAGS", "").split()
    if ref_flags:
        arm_values = {kv_pair.partition("=")[0]: os.environ.get(kv_pair.partition("=")[0]) for kv_pair in ref_flags}
        for kv_pair in ref_flags:
            name, _, value = kv_pair.partition("=")
            monkeypatch.setenv(name, value)
        ref_out = step()
        ttnn.synchronize_device(mesh_device)
        ref = ttnn.to_torch(ttnn.get_device_tensors(ref_out)[0]).float()
        ttnn.deallocate(ref_out)
        for name, value in arm_values.items():  # the arm's own values back before the traced step
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        ref_pcc = torch.corrcoef(torch.stack([eager.flatten(), ref.flatten()]))[0, 1].item()
        ref_max = float((eager - ref).abs().max())
        logger.info(f"LANE_LAYER fused_vs_reference pcc={ref_pcc:.6f} max_abs={ref_max:.4f} ref_flags={ref_flags}")
        assert ref_pcc > 0.999, f"the arm's eager step diverges from its reference flags {ref_flags}: pcc {ref_pcc}"

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tr = step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)  # one untimed replay
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) / REPLAYS * 1e3
    traced = ttnn.to_torch(ttnn.get_device_tensors(out_tr)[0]).float()
    logger.info(
        f"LANE_LAYER layer={layer_idx} mesh={mesh_device.shape[0]}x{mesh_device.shape[1]} users={users}: "
        f"{ms:.3f} ms/replay over {REPLAYS} replays (x30 layers = {30 * ms:.1f} ms) | rope={rope} {knobs}"
    )
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.isfinite(traced).all(), "traced layer output is not finite"
    pcc = torch.corrcoef(torch.stack([eager.flatten(), traced.flatten()]))[0, 1].item()
    logger.info(f"LANE_LAYER parity traced vs eager pcc={pcc:.6f}")
    assert pcc > 0.999, f"traced step diverges from the eager one: pcc {pcc}"


# ===== the serial prefill step of the lane engine: several users on DIFFERENT lanes in one prefill_forward_text call
# (the vLLM chunked continuations and the cached-prefix prefills of a concurrent burst: start_pos > 0, which the lane
# router declines), each user's first-token logits against the same user prefilled alone. 2026-10-25, the 26B Galaxy
# multidoc runs (4 distinct ~28K documents, 20 cached-prefix followers): the readback after the loop took every user's
# logits tile from the LAST user's lane (a non-owner lane's corrupted copy of its logits), a wrong first token on 2-6 of
# 20 concurrent requests and never on serial or staggered ones. Case 1: cold multi-chunk prompts of distinct lengths.
# Case 2: cached-prefix restarts at distinct SDPA-aligned offsets with ~130 new tokens each.
_VAULT_CODES = ["CRIMSON-FALCON-7", "AMBER-LANTERN-3", "VIOLET-HARBOR-9", "COBALT-MEADOW-5"]
_COLD_LENGTHS = [4300, 6250, 9200, 5100]  # distinct multi-chunk prompts at chunk 4096: 2, 2, 3 and 2 chunks
_RESTART_OFFSETS = [4096, 2048, 6144, 3072]  # SDPA-aligned cached prefixes
_RESTART_NEW = 130  # new tokens after the cached prefix (the served followers: ~130-210)
_SERIAL_MAX_SEQ = 16384
_PROSE = (
    ("the night-shift supervisor", "a forklift operator", "the regional auditor", "our logistics partner",
     "the dock foreman", "a visiting inspector", "the maintenance crew", "the procurement lead",
     "a seasonal contractor", "the safety officer"),
    ("reported", "questioned", "approved", "delayed", "re-checked", "signed off on", "postponed", "measured",
     "replaced", "catalogued"),
    ("the pallet racks in aisle {n}", "the cold-storage thermostat", "the {n} damaged crates",
     "the conveyor belt's drive motor", "the overtime sheet for week {n}", "the forklift battery chargers",
     "the loading-bay door seals", "{n} mislabelled shipments", "the sprinkler inspection tags",
     "the mezzanine load limit"),
    ("before the audit closes", "because the manifest did not match", "while the second shift was short-staffed",
     "after a customer complaint", "to keep the insurance certificate valid", "since the last storm damaged the roof",
     "as the quarterly review required", "once the spare parts arrived", "despite the budget freeze",
     "ahead of the holiday peak"),
)  # fmt: skip

SERIAL_MESH_PARAMS = dict(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={"fabric_config": lanes_fabric_config(), "l1_small_size": 65536},
)


def _needle_prompt(tokenizer, user, length):
    """A chat-template prompt of exactly ``length`` tokens for ``user``: its vault code in the first sentence (the
    first chunk / the cached prefix), seeded prose filler, the question last. Returns ``(ids, code)``."""

    def enc(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.convert_tokens_to_ids(t) if isinstance(t, str) else int(t) for t in ids]

    code = _VAULT_CODES[user % len(_VAULT_CODES)]
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "@@"}], add_generation_prompt=True, tokenize=False
    )
    head_text, tail_text = template.split("@@")
    head, tail = enc(head_text), enc(tail_text)
    needle = enc(f"Vault {user} memo. The access code for vault {user} is {code}. ")
    question = enc(f"\n\nWhat is the access code for vault {user}? Answer with the code only.")
    rng = random.Random(100 + user)
    subjects, verbs, objects, tails = _PROSE
    sentences, chars = [], 0
    while chars < 6 * length:
        s = (
            f"On day {len(sentences) + 1} {rng.choice(subjects)} {rng.choice(verbs)} "
            f"{rng.choice(objects).format(n=rng.randint(2, 97))} {rng.choice(tails)}. "
        )
        sentences.append(s)
        chars += len(s)
    filler = enc("".join(sentences))
    n = length - len(head) - len(needle) - len(question) - len(tail)
    assert 0 < n <= len(filler), f"filler too short for {length} tokens: {len(filler)} < {n}"
    return head + needle + filler[:n] + question + tail, code


def _serial_prefill_logits(s, prompts, lengths, users, start_pos=None, sampling_params=None):
    """One ``prefill_forward_text`` call for ``users`` (user i at lane i's slot 0, blocks 1.. of its column's pool):
    the first-token logits per user, ``[len(users), vocab]``, or with ``sampling_params`` the on-device sampled first
    tokens ``[len(users)]``. ``lengths[i]`` tokens of ``prompts[i]`` are the prompt; ``start_pos[i]`` of them are the
    cached prefix (prefilled earlier)."""
    B, S = len(users), max(int(lengths[i]) for i in users)
    blocks = -(-_SERIAL_MAX_SEQ // int(s.block_size))
    tokens = torch.zeros(B, S, dtype=torch.long)
    for b, i in enumerate(users):
        tokens[b, : int(lengths[i])] = torch.tensor(prompts[i][: int(lengths[i])], dtype=torch.long)
    table = torch.arange(1, 1 + blocks, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    extra = {} if start_pos is None else {"start_pos": [int(start_pos[i]) for i in users]}
    out = s.generator.prefill_forward_text(
        tokens,
        page_table=table,
        kv_cache=s.kv,
        prompt_lens=torch.tensor([int(lengths[i]) for i in users]),
        empty_slots=[i * SLOTS_PER_LANE_lane_prefill_trace for i in users],  # lane i's slot 0 (lane = slot // 32)
        enable_trace=False,
        sampling_params=sampling_params,
        warmup_prefill=False,
        **extra,
    )
    if sampling_params is not None:
        assert isinstance(out, (list, tuple)), "on-device sampling did not run: the call returned logits"
        return out[0].reshape(B).to(torch.long)
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out.reshape(B, -1)[:, : s.vocab].float()


def _check_serial_step_parity(s, label, prompts, codes, lengths, start_pos=None):
    """Every user prefilled alone (the proven one-user call), then all users in ONE call; per user the first
    token must match (and the logits agree), and the alone references must answer their codes."""
    users = list(range(s.lanes))
    alone = [_serial_prefill_logits(s, prompts, lengths, [i], start_pos)[0] for i in users]
    together = _serial_prefill_logits(s, prompts, lengths, users, start_pos)
    bad, pccs, hits_alone, hits_together = [], [], 0, 0
    for i in users:
        a_arg, t_arg = int(torch.argmax(alone[i]).item()), int(torch.argmax(together[i]).item())
        pcc = _pcc(alone[i], together[i])
        pccs.append(pcc)
        a_word, t_word = s.tokenizer.decode([a_arg]).strip(), s.tokenizer.decode([t_arg]).strip()
        hits_alone += int(bool(a_word) and codes[i].lower().startswith(a_word.lower()))
        hits_together += int(bool(t_word) and codes[i].lower().startswith(t_word.lower()))
        logger.info(
            f"[lane {i}] {label} len={lengths[i]} cached={0 if start_pos is None else start_pos[i]}: "
            f"alone {a_arg} ({a_word!r}) together {t_arg} ({t_word!r}) pcc={pcc:.5f} code={codes[i]}"
        )
        if a_arg != t_arg:
            bad.append((i, a_arg, a_word, t_arg, t_word))
    logger.info(
        f"LANE_SERIAL {label} parity: {len(users) - len(bad)}/{len(users)} token-exact, min pcc {min(pccs):.5f}, "
        f"code hits alone {hits_alone}/{len(users)} together {hits_together}/{len(users)}"
    )
    assert not bad, f"{label}: first token differs between the one-user call and the {len(users)}-user step: {bad}"
    assert min(pccs) > 0.99, f"{label}: logits diverge between the one-user call and the step: min pcc {min(pccs)}"
    assert hits_alone == len(users), f"{label}: the one-user call answered {hits_alone}/{len(users)} codes"


@parametrize_mesh_with_fabric(**SERIAL_MESH_PARAMS)
def test_lane_serial_prefill_step_distinct_prompts(mesh_device, reset_seeds, monkeypatch):
    """One user per lane, distinct multi-chunk prompts of distinct lengths (cold), prefilled in one call."""
    s = _lane_engine(
        mesh_device,
        monkeypatch,
        max_seq_len=_SERIAL_MAX_SEQ,
        pool_blocks=1 + 2 * (_SERIAL_MAX_SEQ // 64),
    )
    lengths = _COLD_LENGTHS[: s.lanes]
    assert len(lengths) == s.lanes, f"{s.lanes} lanes need {s.lanes} distinct prompt lengths"
    chunk = int(s.generator.model_args[0].max_prefill_chunk_size)
    assert all(n > chunk for n in lengths), f"the prompts must be multi-chunk at chunk {chunk}: {lengths}"
    built = [_needle_prompt(s.tokenizer, i, n) for i, n in enumerate(lengths)]
    prompts, codes = [b[0] for b in built], [b[1] for b in built]
    _check_serial_step_parity(s, "cold", prompts, codes, lengths)


@parametrize_mesh_with_fabric(**SERIAL_MESH_PARAMS)
def test_lane_serial_prefill_step_cached_prefix(mesh_device, reset_seeds, monkeypatch):
    """One user per lane, each restarting after a cached prefix of a different SDPA-aligned length with ~130 new
    tokens (the served followers), the restarts prefilled in one call."""
    s = _lane_engine(
        mesh_device,
        monkeypatch,
        max_seq_len=_SERIAL_MAX_SEQ,
        pool_blocks=1 + 2 * (_SERIAL_MAX_SEQ // 64),
    )
    offsets = _RESTART_OFFSETS[: s.lanes]
    assert len(offsets) == s.lanes, f"{s.lanes} lanes need {s.lanes} distinct restart offsets"
    lengths = [p + _RESTART_NEW for p in offsets]
    built = [_needle_prompt(s.tokenizer, i, n) for i, n in enumerate(lengths)]
    prompts, codes = [b[0] for b in built], [b[1] for b in built]
    # the cached prefixes: each user's first ``offset`` tokens, prefilled alone (positions below the offset)
    for i in range(s.lanes):
        _serial_prefill_logits(s, prompts, offsets, [i])
    _check_serial_step_parity(s, "restart", prompts, codes, lengths, start_pos=offsets)


_SAMPLING_LENGTHS = [1500, 2500, 3500, 4000]  # single-chunk prompts of distinct lengths


@parametrize_mesh_with_fabric(**SERIAL_MESH_PARAMS)
def test_lane_serial_prefill_step_device_sampling(mesh_device, reset_seeds, monkeypatch):
    """One user per lane in one call with on-device (greedy) sampling: each user's sampled first token must be the
    argmax of its alone logits (the fixed logits readback is the reference: the sampled tokens of an alone call off
    lane 0 were wrong the same way, the shared loop reads the tokens from the first shard) and start its code. The
    sampler needs a per-device vocab within 64K (the Galaxy's TP8 lanes: 32K); a TP2 lane (the (2,4) block: 128K)
    builds none, so the case skips there."""
    from models.common.sampling import SamplingParams

    s = _lane_engine(
        mesh_device,
        monkeypatch,
        max_seq_len=_SERIAL_MAX_SEQ,
        pool_blocks=1 + 2 * (_SERIAL_MAX_SEQ // 64),
    )
    if getattr(s.model, "sampling", None) is None:
        pytest.skip(f"no on-device sampler on this mesh (per-device vocab over 64K at tp={s.model.mesh_config.tp})")
    lengths = _SAMPLING_LENGTHS[: s.lanes]
    assert len(lengths) == s.lanes, f"{s.lanes} lanes need {s.lanes} distinct prompt lengths"
    built = [_needle_prompt(s.tokenizer, i, n) for i, n in enumerate(lengths)]
    prompts, codes = [b[0] for b in built], [b[1] for b in built]
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    users = list(range(s.lanes))
    ref = [int(torch.argmax(_serial_prefill_logits(s, prompts, lengths, [i])[0]).item()) for i in users]
    alone = [int(_serial_prefill_logits(s, prompts, lengths, [i], sampling_params=greedy)[0].item()) for i in users]
    together = _serial_prefill_logits(s, prompts, lengths, users, sampling_params=greedy).tolist()
    bad, hits = [], 0
    for i in users:
        word = s.tokenizer.decode([together[i]]).strip()
        hits += int(bool(word) and codes[i].lower().startswith(word.lower()))
        logger.info(
            f"[lane {i}] sampled len={lengths[i]}: alone argmax {ref[i]} alone sampled {alone[i]} "
            f"together sampled {together[i]} ({word!r}) code={codes[i]}"
        )
        if together[i] != ref[i] or alone[i] != ref[i]:
            bad.append((i, ref[i], alone[i], together[i], word))
    logger.info(
        f"LANE_SERIAL sampled parity: {len(users) - len(bad)}/{len(users)} token-exact vs the alone logits, "
        f"code hits {hits}/{len(users)}"
    )
    assert not bad, f"sampled first tokens differ from the alone logits' argmax (lane, ref, alone, step, word): {bad}"
    assert hits == len(users), f"the step's sampled tokens start {hits}/{len(users)} codes"
