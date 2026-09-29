# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lane-parallel prefill (galaxy slice 4): 4 users prefill in ONE forward.

Correctness: each lane's last-token logits from the single lane-parallel
forward must match that user's serial single-user prefill (argmax equal,
PCC high). Timing: 4 serial prefills vs one lane-parallel forward at the
largest single-chunk ISL, reported (not asserted).

    GEMMA4_GALAXY_FRACTURE=1 GEMMA4_GALAXY_LANES=1 HF_MODEL=google/gemma-4-31B-it \
    pytest models/demos/gemma4/tests/unit/test_lane_prefill_parallel.py -k 8x4 -s
"""

import os
import time

import torch

from ...tests.test_factory import parametrize_mesh_with_fabric

USERS = [
    "What is the capital of France? Answer in one word.",
    "What is 2+2? Answer with just the number.",
    "What is the chemical symbol for gold? Answer with just the symbol.",
    "What is the opposite of hot? Answer in one word.",
]
SLOTS_PER_LANE = 32


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
    assert lanes == len(USERS)
    vocab = generator.model[0].vocab_size

    prompts_tok = []
    for q in USERS:
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
                empty_slots=[i * SLOTS_PER_LANE],  # lane i's slot 0 (block convention)
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

    # ── Phase A: correctness on the real chat prompts ─────────────────────
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

    # ── Phase B: timing at the largest single-chunk ISL ───────────────────
    max_chunk = int(generator.model_args[0].max_prefill_chunk_size)
    s_time = min(2048, max_chunk)
    base = prompts_tok[0]
    reps = (s_time + base.shape[-1] - 1) // base.shape[-1]
    long_tok = base.repeat(reps)[:s_time]
    toks_list = [long_tok.clone() for _ in range(lanes)]
    plens_t = [s_time] * lanes

    serial_prefill(toks_list, plens_t, s_time)  # warm/compile
    lane_prefill(toks_list, plens_t, s_time)

    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        serial_prefill(toks_list, plens_t, s_time)
        t.append(time.perf_counter() - t0)
    serial_ms = sorted(t)[1] * 1e3

    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        lane_prefill(toks_list, plens_t, s_time)
        t.append(time.perf_counter() - t0)
    lanes_ms = sorted(t)[1] * 1e3

    print(
        f"[lane-prefill] isl={s_time} serial4={serial_ms:.1f}ms lanes={lanes_ms:.1f}ms "
        f"speedup={serial_ms / lanes_ms:.2f}x (4 users)"
    )
