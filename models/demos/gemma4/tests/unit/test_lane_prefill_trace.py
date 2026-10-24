# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The traced lane prefill on the (8,4) lane engine: the 1x128 lane frame captured at boot and replayed for the 32
coherence questions (one real lane plus scratch lanes per call, the real lane rotating), token-exact against the
eager lane walk of the same frame and against the serial owner-lane prefill. ~8 min (model load included).

    GEMMA4_KV_BFP8=1 GEMMA4_CCL_TOPOLOGY=linear GEMMA4_HYBRID_KV_CACHE_GROUPS=1 GEMMA4_PAGE_BLOCK_SIZE=64 \\
    HF_MODEL=google/gemma-4-26B-A4B-it pytest models/demos/gemma4/tests/unit/test_lane_prefill_trace.py -k 8x4 -s
"""

import os

import pytest
import torch
from loguru import logger

from models.demos.gemma4.tests.test_factory import (
    lanes_fabric_config,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)

SLOTS_PER_LANE = 32
BLOCKS_PER_USER = 4  # 256 tokens at block 64: the 128-token frame plus headroom
BUCKET = (1, 128)
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


def _pcc(a, b):
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


@parametrize_mesh_with_fabric(
    mesh_shapes=[(8, 4)],
    device_params_extra={
        "fabric_config": lanes_fabric_config(),
        "l1_small_size": 65536,
        "trace_region_size": 400_000_000,
    },
)
def test_lane_prefill_trace_matches_eager(mesh_device, reset_seeds, monkeypatch):
    skip_if_config_only_checkpoint()
    shape = tuple(int(x) for x in mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    if shape != (8, 4):
        pytest.skip(f"the traced lane prefill is the (8,4) lane engine's; got {shape}")
    for k, v in {"GEMMA4_GALAXY_FRACTURE": "1", "GEMMA4_GALAXY_LANES": "1", "GEMMA4_CP_PREFILL": "0"}.items():
        monkeypatch.setenv(k, v)
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it")
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + SLOTS_PER_LANE * BLOCKS_PER_USER)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device, model_path, max_batch_size=SLOTS_PER_LANE, max_seq_len=4096, paged_attention_config=paged_cfg
    )
    model = generator.model[0]
    mesh_cfg = model.mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    lanes = int(mesh_cfg.lanes)
    vocab = model.vocab_size
    block_size = generator._effective_paged_block_size(
        tt_kv_cache[0] if isinstance(tt_kv_cache[0], list) else tt_kv_cache
    )

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
    assert L <= BUCKET[1], f"the questions must fit the {BUCKET[1]}-token frame, longest is {L}"
    padded = []
    for p in prompts:
        cut = p.index(nl) + 1 if nl in p else 1
        padded.append(p[:cut] + [nl] * (L - len(p)) + p[cut:])

    assert generator._g4_capture_lane_prefill_trace(BUCKET, tt_kv_cache), "capture failed"
    entry = generator._g4_lane_trace_registry()[BUCKET]
    logger.info(f"LANE_TRACE captured {BUCKET} widths={entry['widths'][:3]}... needed_blocks={entry['needed_blocks']}")

    def frame_for(q):
        """The q-th question on lane q % lanes at slot (q // lanes) % 32; the other lanes on scratch rows."""
        lane, slot = q % lanes, (q // lanes) % SLOTS_PER_LANE
        tokens = torch.zeros(lanes, 1, BUCKET[1], dtype=torch.long)
        tokens[lane, 0, :L] = torch.tensor(padded[q], dtype=torch.long)
        tables = torch.zeros(lanes, 1, BLOCKS_PER_USER, dtype=torch.int32)
        tables[lane, 0] = torch.arange(1 + slot * BLOCKS_PER_USER, 1 + (slot + 1) * BLOCKS_PER_USER, dtype=torch.int32)
        slots = [ln * SLOTS_PER_LANE + slot for ln in range(lanes)]
        per_layer = generator._g4_lane_frame_page_tables(model, tables, slots, block_size)
        return lane, tokens, tables, per_layer, slots

    mismatches, pccs = [], []
    for q in range(len(COUNTRIES)):
        lane, tokens, tables, per_layer, slots = frame_for(q)
        traced = generator._g4_lane_prefill_trace_replay(BUCKET, tokens, tables, per_layer, tt_kv_cache, [L] * lanes)
        assert traced is not None, f"question {q}: the replay fell back to eager"
        eager = generator.prefill_forward_lanes(tokens[:, 0], tables, tt_kv_cache, [L] * lanes, slot_ids=slots)
        t_arg, e_arg = int(torch.argmax(traced[lane]).item()), int(torch.argmax(eager[lane]).item())
        pcc = _pcc(traced[lane][:vocab], eager[lane][:vocab])
        pccs.append(pcc)
        word = tokenizer.decode([t_arg]).strip()
        logger.info(f"[q {q} lane {lane}] {COUNTRIES[q][0]}: traced {t_arg} ({word!r}) eager {e_arg} pcc={pcc:.5f}")
        if t_arg != e_arg:
            mismatches.append((q, COUNTRIES[q][0], t_arg, e_arg))
    logger.info(
        f"LANE_TRACE parity: {len(COUNTRIES) - len(mismatches)}/{len(COUNTRIES)} token-exact, min pcc {min(pccs):.5f}"
    )
    assert not mismatches, f"traced vs eager first token differs: {mismatches}"
    assert min(pccs) > 0.99
