# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MTP speculative decoding through the vLLM adapter (generator_vllm.py) on TP=8 x DP=4, without vLLM.

Lane 0 decodes a prompt greedily one token per step; lane 1 serves the same prompt the way the plugin drives a
speculating request: a plain step, then propose (MTP drafts) / verify (the request in every slot of its replica) /
greedy acceptance. The speculative tokens must equal the plain greedy tokens. Logs the mean accepted length and the
time per token of both loops. Needs vllm-tt-plugin (its spec_decode value types) and the MiMo checkpoint.
"""

import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS, accept_greedy
from models.experimental.mimo_v2_6_flash.tests.common import DP, GALAXY_DEVICE_PARAMS, model_path
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import BLOCK_SIZE, MiMoV2ForCausalLM

pytest.importorskip("vllm_tt_plugin")

SPEC_DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": 1_200_000_000}
PER_LANE = 4
MAX_SEQ_LEN = 1024
STEPS = 96
PROMPT = (
    "Write a Python function that returns the n-th Fibonacci number using memoization, then explain how the "
    "memoization changes its time complexity."
)


@pytest.mark.parametrize("device_params", [SPEC_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_decode_tp8_dp4(mesh_device, monkeypatch):
    path = model_path()
    cfg = MiMoConfig.from_model_path(path)
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    # One KV cache group (no vLLM to group the specs): every layer reads one page table, a buffer per layer.
    monkeypatch.setattr(MiMoV2ForCausalLM, "_layer_groups", [0] * cfg.num_hidden_layers)
    monkeypatch.setattr(MiMoV2ForCausalLM, "prefill_token_budget", 2048)

    rows = DP * PER_LANE
    width = MAX_SEQ_LEN // BLOCK_SIZE
    num_blocks = 1 + rows * width
    hf = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers, _name_or_path=path)
    model = MiMoV2ForCausalLM.initialize_vllm_model(hf, mesh_device, rows, MAX_SEQ_LEN, tt_data_parallel=DP)
    specs = []
    for i in range(cfg.num_hidden_layers):
        dims = cfg.attention_dims(i)
        specs.append(((num_blocks, 1, BLOCK_SIZE, dims.head_dim), torch.bfloat16, i))
    kv = model.allocate_kv_cache_per_layer(specs)
    model.warmup_model_prefill(kv)
    model.warmup_model_decode(kv, enable_trace=True)
    page_table = 1 + torch.arange(rows * width, dtype=torch.int32).reshape(rows, width)

    prompt = AutoTokenizer.from_pretrained(path)(PROMPT, return_tensors="pt")["input_ids"][0].to(torch.int32)
    plain_row, spec_row = 0, PER_LANE  # lane 0 slot 0, lane 1 slot 0

    def prefill(row):
        logits = model.prefill_forward(
            prompt[None],
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([prompt.numel()]),
            page_table,
            kv,
            empty_slots=[row],
        )
        return int(logits[0, -1].argmax())

    def decode(row, token, pos):
        tokens = torch.zeros(rows, 1, dtype=torch.int32)
        positions = torch.full((rows,), -1, dtype=torch.int32)
        tokens[row, 0], positions[row] = token, pos
        logits = model.decode_forward(tokens, positions, page_table, kv, enable_trace=True)
        return int(logits[row, 0].argmax())

    # Plain greedy on lane 0.
    plain = prompt.tolist() + [prefill(plain_row)]
    t0 = time.perf_counter()
    while len(plain) < prompt.numel() + STEPS:
        plain.append(decode(plain_row, plain[-1], len(plain) - 1))
    plain_s = time.perf_counter() - t0

    # Speculative greedy on lane 1: a plain step, then propose / verify / accept.
    seq = prompt.tolist() + [prefill(spec_row)]
    t0 = time.perf_counter()
    seq.append(decode(spec_row, seq[-1], len(seq) - 1))
    committed, accepted, counts = [seq[-1]], 1, []

    def propose():
        tok = torch.zeros(rows, 1 + MAX_DRAFTS, dtype=torch.int32)
        pos = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
        tok[spec_row, : len(committed)] = torch.tensor(committed, dtype=torch.int32)
        pos[spec_row, : len(committed)] = torch.arange(len(seq) - len(committed), len(seq), dtype=torch.int32)
        out = model.propose_draft_tokens(MAX_DRAFTS, tok, pos, torch.ones(rows, dtype=torch.int32) * accepted)
        assert int(out.num_valid[spec_row]) == MAX_DRAFTS and int(out.num_valid.sum()) == MAX_DRAFTS
        return out.draft_token_ids[spec_row].tolist()

    while len(seq) < prompt.numel() + STEPS:
        drafts = propose()
        p = len(seq) - 1
        tokens = torch.zeros(rows, 1 + MAX_DRAFTS, dtype=torch.int32)
        positions = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
        tokens[spec_row] = torch.tensor([seq[-1]] + drafts, dtype=torch.int32)
        positions[spec_row] = torch.arange(p, p + 1 + MAX_DRAFTS, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        valid[spec_row] = MAX_DRAFTS
        out = model.decode_forward(
            tokens,
            positions,
            page_table,
            kv,
            spec_mode="argmax_ids",
            num_valid_drafts=valid,
            accepted_counts=torch.ones(rows, dtype=torch.int32),
        )
        committed = accept_greedy(out.argmax_ids[spec_row].tolist(), drafts, MAX_DRAFTS)
        accepted = len(committed)
        counts.append(accepted)
        seq += committed
    spec_s = time.perf_counter() - t0

    n = prompt.numel() + STEPS
    logger.info(
        f"MTP spec decode: mean accepted {sum(counts) / len(counts):.2f} over {len(counts)} verifies "
        f"(histogram {[counts.count(m) for m in range(1, MAX_DRAFTS + 2)]}); "
        f"{1e3 * plain_s / STEPS:.1f} ms/token plain, {1e3 * spec_s / (len(seq) - prompt.numel() - 1):.1f} ms/token "
        "speculative (host-driven, sync reads)"
    )
    first_diff = next((i for i in range(prompt.numel(), n) if seq[i] != plain[i]), None)
    assert first_diff is None, f"speculative tokens diverge from greedy at position {first_diff}"
