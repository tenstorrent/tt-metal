# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lane capacity: fill all 128 slots (32 per lane x 4 lanes) by lane-parallel prefill and decode the full
global batch; every row must answer its own question, so page-table collisions or cross-lane mixing fail."""

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

    # Block convention: global slot u is the decode row, lane = u // 32, per-lane slot = u % 32.
    # User u asks question u % 4, so each lane's slots cycle all four answers.
    def lane_of(u):
        return u // SLOTS_PER_LANE

    def slot_of(u):
        return u % SLOTS_PER_LANE

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
    for r in range(SLOTS_PER_LANE):
        group = [lane * SLOTS_PER_LANE + r for lane in range(lanes)]  # one user per lane
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
