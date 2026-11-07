# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A prefix hit on KV blocks its own prefill step computes, through the vLLM adapter on one TP=8 row (full 48 layers).

vLLM caches a block when it schedules the block's computation, so a request admitted in the step whose chunk computes a
shared prefix hits blocks no pass has written yet (Video-MME: three questions on one video arrive together; the
first one's last video chunk and the others' question tails run in one step). prefill_forward runs every row as a pass
of its own; it must run such a row after the row that writes its blocks (``_prefill_readers``).

Request A (slot 0) prefills a prompt in two steps: [0, 1024), then [1024, N). In A's second step request B (slot 1,
row 0 of the step, so first in row order) is admitted with a prefix hit at HIT: its tables share A's blocks below HIT,
and A writes [1024, HIT) of them in that very step. After A is done, request C (slot 3, the same parity as B) gets
the same hit on A's blocks in a step of its own. B must compute exactly what C computes (last-token logits bit for
bit). With the ordering turned off (``_prefill_readers`` returning nothing) the same setup on fresh slots (A' 4, B' 5)
must differ: B' attends A''s blocks before A' wrote them, so the test sees the bug it guards against.

    TT_VISIBLE_DEVICES=<the row's 8 chips> pytest models/experimental/mimo_v2_6_flash/tests/test_prefill_shared_blocks_tp8.py
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, model_path
from models.experimental.mimo_v2_6_flash.tests.test_spec_decode_tp8_dp4 import kv_cache_and_tables, vllm_kv_layout
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import BLOCK_SIZE, MiMoV2ForCausalLM

PROMPT_TOKENS = 1400
FIRST_CHUNK = 1024
HIT = 1280  # a multiple of 128 (vLLM's scheduler block with the pair split) inside A's second chunk
SLOTS = 8
MAX_SEQ_LEN = 2048


def prompt_ids(path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "reference", "model.py")) as f:
        text = "Explain what the following Python module does.\n\n" + f.read()
    ids = torch.tensor(AutoTokenizer.from_pretrained(path)(text)["input_ids"], dtype=torch.int32)
    assert ids.numel() >= PROMPT_TOKENS, ids.numel()
    return ids[:PROMPT_TOKENS]


def share_prefix(group_tables, block_sizes, src, dst, positions):
    """Point slot ``dst``'s rows at slot ``src``'s blocks for positions [0, positions) in every KV cache group."""
    for g, table in enumerate(group_tables):
        n = positions // block_sizes[g]
        table[dst, :n] = table[src, :n]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_prefill_shared_blocks_tp8(mesh_device, monkeypatch):
    path = model_path()
    cfg = MiMoConfig.from_model_path(path)
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "")
    groups, buffers = vllm_kv_layout(cfg)
    monkeypatch.setattr(MiMoV2ForCausalLM, "_layer_groups", groups)
    t0 = time.time()
    hf = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers, _name_or_path=path)
    model = MiMoV2ForCausalLM.initialize_vllm_model(hf, mesh_device, SLOTS, MAX_SEQ_LEN, tt_data_parallel=1)
    kv, page_table, layer_tables = kv_cache_and_tables(model, cfg, groups, buffers, SLOTS, MAX_SEQ_LEN // BLOCK_SIZE)
    logger.info(f"model and KV cache ready in {time.time() - t0:.0f}s")
    num_groups = max(groups) + 1
    group_tables = [layer_tables[groups.index(g)] for g in range(num_groups)]
    block_sizes = [model.group_block_sizes[g] for g in range(num_groups)]
    ids = prompt_ids(path)
    n = ids.numel()

    def step(rows):
        """One prefill_forward: rows [(slot, start, end)] -> each row's last-token logits [vocab]."""
        slots = [s for s, _, _ in rows]
        out = model.prefill_forward(
            ids[None].expand(len(rows), -1).contiguous(),
            torch.tensor([a for _, a, _ in rows], dtype=torch.int32),
            torch.tensor([b for _, _, b in rows]),
            page_table[slots],
            kv,
            empty_slots=slots,
            page_tables_per_layer=[t[slots] for t in layer_tables],
        )
        return [out[k, -1] for k in range(len(rows))]

    def shared_step_pair(writer, reader):
        """The writer's first chunk alone, then its second chunk with the reader (row 0) hitting its blocks."""
        share_prefix(group_tables, block_sizes, writer, reader, HIT)
        step([(writer, 0, FIRST_CHUNK)])
        reader_logits, writer_logits = step([(reader, HIT, n), (writer, FIRST_CHUNK, n)])
        return reader_logits, writer_logits

    # The fixed order: B runs after A in A's second step.
    b, a = shared_step_pair(writer=0, reader=1)
    share_prefix(group_tables, block_sizes, 0, 3, HIT)
    (c,) = step([(3, HIT, n)])
    diff = float((b - c).abs().max())
    logger.info(
        f"B (hit at {HIT} on blocks of its own step) vs C (same hit, later step): max |diff| {diff}; top-1 A "
        f"{int(a.argmax())}, B {int(b.argmax())}, C {int(c.argmax())}"
    )
    assert torch.equal(b, c), f"B's logits differ from C's (max |diff| {diff}): it read A's blocks before A wrote them"
    assert int(b.argmax()) == int(a.argmax())

    # The old order (rows as they come): B' reads A''s blocks before A' writes them.
    monkeypatch.setattr(MiMoV2ForCausalLM, "_prefill_readers", lambda self, tables, placed: {})
    b_old, _ = shared_step_pair(writer=4, reader=5)
    old_diff = float((b_old - c).abs().max())
    logger.info(f"without the ordering: B' vs C max |diff| {old_diff}, top-1 B' {int(b_old.argmax())}")
    assert not torch.equal(b_old, c), "without the ordering the reader should have attended unwritten blocks"
