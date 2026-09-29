# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lane capacity reference (galaxy slice 4b): 128 users on one weight copy.

Fills every slot (32 per lane x 4 lanes) via lane-parallel prefill, then
decodes the full 128-row global batch. Every user asks one of four known
questions; every row's greedy continuation must contain ITS answer, so any
page-table collision or cross-lane mixing fails loudly. Reports aggregate
prefill throughput and decode steps/s at capacity.

    GEMMA4_GALAXY_FRACTURE=1 GEMMA4_GALAXY_LANES=1 HF_MODEL=google/gemma-4-31B-it \
    pytest models/demos/gemma4/tests/unit/test_lanes_capacity.py -k 8x4 -s
"""

import os
import time

import torch

from ...tests.test_factory import parametrize_mesh_with_fabric

QA = [
    ("What is the capital of France? Answer in one word.", "paris"),
    ("What is 2+2? Answer with just the number.", "4"),
    ("What is the chemical symbol for gold? Answer with just the symbol.", "au"),
    ("What is the opposite of hot? Answer in one word.", "cold"),
]
GEN_TOKENS = 24
SLOTS_PER_LANE = 32
ENABLE_TRACE = os.environ.get("G4_CAP_TRACE", "0") == "1"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_capacity(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    blocks_per_user = 4  # 256 tokens: prompt <=128 + generation
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
    B_g = lanes * SLOTS_PER_LANE  # 128

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

    # User u: lane u % lanes, per-lane slot u // lanes (modulo convention),
    # question u % 4 (so each lane cycles all four answers down its slots).
    def lane_of(u):
        return u % lanes

    def slot_of(u):
        return u // lanes

    def row_of(u):  # decode global row (lane-major)
        return lane_of(u) * SLOTS_PER_LANE + slot_of(u)

    def blocks_of(u):
        s = slot_of(u)
        return torch.arange(1 + s * blocks_per_user, 1 + (s + 1) * blocks_per_user, dtype=torch.int32)

    # ── Prefill all 128 users: 32 lane-parallel rounds of 4 ───────────────
    n_users = B_g
    positions, cur_tok = {}, {}
    t0 = time.perf_counter()
    total_prefill_tokens = 0
    for base in range(0, n_users, lanes):
        group = list(range(base, base + lanes))  # lane_of(u) == u - base
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
    for step in range(GEN_TOKENS):
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, blocks_per_user, dtype=torch.int32)
        for u in range(n_users):
            r = row_of(u)
            tokens_g[r] = cur_tok[u]
            pos_g[r] = positions[u]
            pt_g[r] = blocks_of(u)
        t0 = time.perf_counter()
        logits = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=ENABLE_TRACE,
            read_from_device=True,
            sampling_params=None,
        )
        lg = logits[0] if isinstance(logits, (list, tuple)) else logits
        lg = lg.reshape(B_g, -1)
        step_times.append(time.perf_counter() - t0)
        for u in range(n_users):
            nxt = int(torch.argmax(lg[row_of(u)]).item())
            outputs[u].append(nxt)
            cur_tok[u] = nxt
            positions[u] += 1

    steady = sorted(step_times[3:])[len(step_times[3:]) // 2]
    print(
        f"[capacity] decode step median {steady * 1e3:.1f}ms -> "
        f"{B_g / steady:.0f} tok/s aggregate at {B_g} users (host-loop"
        f"{', traced' if ENABLE_TRACE else ', eager'})"
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
