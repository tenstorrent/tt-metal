# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lane-sharded one-instance e2e (galaxy slice 3b): 4 users, one per lane.

Each lane's column serves a DIFFERENT user with its own KV: per-user prefill
lands on the owner column (scratch block 0 elsewhere), decode runs one
lane-major global batch (lanes x 32 with pad rows), and the host reassembles
per-lane logits. Greedy continuations must answer each user's own question —
any cross-lane KV or logit mixing breaks the oracle immediately.

    GEMMA4_GALAXY_FRACTURE=1 GEMMA4_GALAXY_LANES=1 HF_MODEL=google/gemma-4-31B-it \
    pytest models/demos/gemma4/tests/unit/test_lanes_e2e.py -k 8x4 -s
"""

import os

import torch


from ...tests.test_factory import parametrize_mesh_with_fabric

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
    os.environ.setdefault("GEMMA4_CP_PREFILL", "1")

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.model_config import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    # Per-column pool: block 0 is the lane scratch, then 32 slots per lane.
    blocks_per_user = 4  # 256 tokens: short chat prompts + generation
    pool_blocks = 1 + SLOTS_PER_LANE * blocks_per_user
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=pool_blocks)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        model_path,
        mesh_device=mesh_device,
        max_batch_size=SLOTS_PER_LANE,
        max_seq_len=4096,
        paged_attention_config=paged_cfg,
        instruct=True,
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    lanes = mesh_cfg.lanes
    assert lanes == 4
    B_g = lanes * SLOTS_PER_LANE

    # Encode with the chat template (raw completions on instruct models look
    # degenerate by construction — see gemma4-probe-coherence-correctly).
    prompts_tok = []
    for q, _ in USERS:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
        prompts_tok.append(torch.tensor(ids, dtype=torch.long))

    # One user per lane, slot 0: global row = lane * 32.
    rows = [lane * SLOTS_PER_LANE for lane in range(lanes)]
    page_rows = {}
    for lane, row in enumerate(rows):
        ids = torch.arange(1, 1 + blocks_per_user, dtype=torch.int32)  # slot 0 of this lane's pool
        page_rows[row] = ids

    # ── Per-user prefill on the owner lane ────────────────────────────────
    positions = {}
    for (lane, row), toks in zip(enumerate(rows), prompts_tok):
        plen = toks.shape[-1]
        padded = 1 << max(int(plen - 1).bit_length(), 7)
        toks_padded = torch.nn.functional.pad(toks, (0, padded - plen), value=0).unsqueeze(0)
        logits = generator.prefill_forward_single_user_text(
            toks_padded,
            page_table=page_rows[row].unsqueeze(0),
            user_id=row,
            last_token_idx=plen - 1,
            kv_cache=tt_kv_cache,
        )
        positions[row] = plen

    # Prefill logits handling differs per path; take the safe route and start
    # decode from the last prompt token's argmax computed in the first decode
    # step instead: seed decode with the final prompt token.
    seeds = {row: int(prompts_tok[i][-1]) for i, row in enumerate(rows)}

    # ── Lane-major decode loop (host greedy) ──────────────────────────────
    outputs = {row: [] for row in rows}
    cur_tok = dict(seeds)
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
