# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lane-sharded e2e: one user per lane with its own KV, prefilled per user and decoded as one lane-major
global batch; each lane must answer its own question, so cross-lane KV or logit mixing fails."""

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
    outputs = {row: [] for row in rows}
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
