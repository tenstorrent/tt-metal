# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 2 (a): the CPU oracle at R = 5 rows equals its per-row results, stage by stage.

No device, synthetic weights at the model's widths.  GR read/write (full and TP4 forms), final mixer (full
and TP4), MTP input mixer, PLE (post-lookup equations with the real n-gram hash and a fake table), LM head
(sharded logits and the owner argmax), and MoE (router/top-k/shared/expert combine, full and EP4 forms).
Row independence is the property the (k+1)-row verify body relies on; each stage is compared bitwise under
``row_serial_torch`` (torch's own bf16 GEMM/reduction blocking differs between M = 1 and M = 5 by up to
1 bf16 ULP, which is a torch artifact, not a coupling between rows).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.reference import build_ngram_hash_spec, mtp_input_fusion, ngram_token_ids
from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import row_serial_torch
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixer, Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.tt.moe import (
    Qwen38ExpertWeights,
    Qwen38MoE,
    Qwen38SharedExpertShard,
    weighted_routed_reduce,
)
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38PLE, Qwen38PLEState, Qwen38PLEWeights

ROWS = 5
HIDDEN = 2560
BRANCHES = 4
RANK = 320
EPS = 1e-6
TP = 4
VOCAB_SHARD = 2048  # synthetic LM head: 4 x 2048 rows
EOS = 248044


def _ranges(total: int) -> tuple[tuple[int, int], ...]:
    width = total // TP
    return tuple((d * width, (d + 1) * width) for d in range(TP))


def _placement(hidden: int, experts: int = 64) -> SimpleNamespace:
    config = SimpleNamespace(
        hidden_size=hidden,
        residual_branches=BRANCHES,
        residual_rank=RANK,
        residual_width=BRANCHES * hidden,
        rms_norm_eps=EPS,
        num_experts=experts,
        top_k=10,
        norm_topk_prob=True,
    )
    return SimpleNamespace(config=config, hidden_ranges=_ranges(hidden), expert_ranges=_ranges(experts))


def _bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _assert_rows_equal(batched: torch.Tensor, per_row: list[torch.Tensor], label: str) -> None:
    stacked = torch.cat(per_row, dim=1)
    assert stacked.shape == batched.shape, (label, stacked.shape, batched.shape)
    assert stacked.dtype == batched.dtype, (label, stacked.dtype, batched.dtype)
    if not torch.equal(stacked, batched):
        gap = (stacked.float() - batched.float()).abs()
        raise AssertionError(f"{label}: rows differ, max abs {gap.max().item():.3e} at {gap.argmax().item()}")


@pytest.fixture(scope="module")
def residual() -> torch.Tensor:
    torch.manual_seed(1)
    return _bf16(1, ROWS, BRANCHES * HIDDEN)


@pytest.fixture(autouse=True)
def _row_serial():
    with row_serial_torch():
        yield


def test_gr_read_and_write_are_row_independent(residual) -> None:
    torch.manual_seed(2)
    weights = Qwen38GatedResidualWeights(
        placement=_placement(HIDDEN),
        layer_index=0,
        block="attn",
        norm=_bf16(BRANCHES * HIDDEN, scale=0.1),
        down=_bf16(RANK, BRANCHES * HIDDEN, scale=BRANCHES * HIDDEN**-0.5),
        up=_bf16(BRANCHES * HIDDEN, RANK, scale=RANK**-0.5),
        inject=_bf16(BRANCHES, BRANCHES * HIDDEN, scale=(BRANCHES * HIDDEN) ** -0.5),
    )
    gr = Qwen38GatedResidual(weights)
    block_output = _bf16(1, ROWS, HIDDEN)

    block_input, state = gr.read(residual)
    written = gr.write(block_output, state)
    rows = [gr.read(residual[:, r : r + 1]) for r in range(ROWS)]
    _assert_rows_equal(block_input, [row[0] for row in rows], "GR read block input")
    _assert_rows_equal(state.injection, [row[1].injection for row in rows], "GR read injection")
    _assert_rows_equal(written, [gr.write(block_output[:, r : r + 1], rows[r][1]) for r in range(ROWS)], "GR write")

    shards = gr.shard_residual(residual)
    block_shards, tp_state = gr.read_tp4(shards)
    output_shards = gr.shard_hidden(block_output)
    written_shards = gr.write_tp4(output_shards, tp_state)
    for d in range(TP):
        per_row = []
        per_row_written = []
        for r in range(ROWS):
            row_shards = tuple(shard[:, r : r + 1] for shard in shards)
            row_blocks, row_state = gr.read_tp4(row_shards)
            per_row.append(row_blocks[d])
            per_row_written.append(gr.write_tp4(tuple(s[:, r : r + 1] for s in output_shards), row_state)[d])
        _assert_rows_equal(block_shards[d], per_row, f"GR TP4 read shard {d}")
        _assert_rows_equal(written_shards[d], per_row_written, f"GR TP4 write shard {d}")
    _assert_rows_equal(
        tp_state.injection,
        [gr.read_tp4(tuple(s[:, r : r + 1] for s in shards))[1].injection for r in range(ROWS)],
        "GR TP4 injection",
    )
    assert torch.equal(gr.combine_residual_shards(written_shards), written)


def test_final_mixer_is_row_independent(residual) -> None:
    torch.manual_seed(3)
    weights = Qwen38FinalMixerWeights(
        placement=_placement(HIDDEN),
        norm=_bf16(BRANCHES * HIDDEN, scale=0.1),
        down=_bf16(RANK, BRANCHES * HIDDEN, scale=(BRANCHES * HIDDEN) ** -0.5),
        up=_bf16(BRANCHES * HIDDEN, RANK, scale=RANK**-0.5),
    )
    mixer = Qwen38FinalMixer(weights)
    _assert_rows_equal(mixer(residual), [mixer(residual[:, r : r + 1]) for r in range(ROWS)], "final mixer")
    shards = mixer.shard_residual(residual)
    outputs = mixer.forward_tp4(shards)
    for d in range(TP):
        per_row = [mixer.forward_tp4(tuple(s[:, r : r + 1] for s in shards))[d] for r in range(ROWS)]
        _assert_rows_equal(outputs[d], per_row, f"final mixer TP4 shard {d}")


def test_mtp_input_mixer_is_row_independent(residual) -> None:
    torch.manual_seed(4)
    embedding = _bf16(1, ROWS, HIDDEN)
    weights = dict(
        embedding_norm_weight=_bf16(HIDDEN, scale=0.1),
        hidden_norm_weight=_bf16(BRANCHES * HIDDEN, scale=0.1),
        fc_embedding_weight=_bf16(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        fc_hidden_weight=_bf16(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        hc_count=BRANCHES,
        hidden_size=HIDDEN,
        eps=EPS,
    )
    mixed = mtp_input_fusion(embedding, residual, **weights)
    per_row = [mtp_input_fusion(embedding[:, r : r + 1], residual[:, r : r + 1], **weights) for r in range(ROWS)]
    _assert_rows_equal(mixed, per_row, "MTP input mixer")


class _FakeHostPLEEmbedding:
    """The real n-gram hash and context threading over a small fake table (the checkpoint table is 360 GB away)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(ngram_size=3, eos_token_id=EOS, ple_checkpoint_layer=1)
        self.spec = build_ngram_hash_spec(
            unigram_vocab_size=248320,
            ngram_size=3,
            heads_per_ngram=8,
            ngram_vocab_size_base=20_000_000,
            ple_layer_index=0,
            seed=1234,
            divisible_by=128,
        )
        self.embedding_head_dim = HIDDEN // 16
        self.table = _bf16(4096, self.embedding_head_dim)

    def lookup(self, input_ids: torch.Tensor, previous_context: torch.Tensor | None):
        ids, next_context = ngram_token_ids(input_ids, previous_context, EOS, self.spec)
        rows = self.table[ids.reshape(-1) % self.table.shape[0]]
        return rows.reshape(*ids.shape, self.embedding_head_dim).flatten(-2), next_context


def test_ple_rows_equal_sequential_steps_with_threaded_state(residual) -> None:
    torch.manual_seed(5)
    weights = Qwen38PLEWeights(
        layer_idx=1,
        hidden_size=HIDDEN,
        residual_branches=BRANCHES,
        embedding_width=HIDDEN,
        conv_kernel=4,
        conv_dilation=3,
        rms_norm_eps=EPS,
        key=_bf16(BRANCHES * HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        value=_bf16(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        norm_key=_bf16(BRANCHES * HIDDEN, scale=0.1),
        norm_query=_bf16(BRANCHES * HIDDEN, scale=0.1),
        norm_conv=_bf16(BRANCHES * HIDDEN, scale=0.1),
        conv=_bf16(BRANCHES * HIDDEN, 1, 4, scale=0.5),
    )
    ple = Qwen38PLE(_FakeHostPLEEmbedding(), weights)
    input_ids = torch.tensor([[17, 15, 16, 95859, 20]], dtype=torch.long)
    state = Qwen38PLEState(
        token_context=torch.tensor([[EOS, 12]], dtype=torch.long),
        conv=_bf16(1, BRANCHES * HIDDEN, 9),
    )

    output, next_state = ple.forward(residual, input_ids, state)
    per_row = []
    threaded = state
    for r in range(ROWS):
        row_output, threaded = ple.forward(residual[:, r : r + 1], input_ids[:, r : r + 1], threaded)
        per_row.append(row_output)
    _assert_rows_equal(output, per_row, "PLE output")
    assert torch.equal(next_state.token_context, threaded.token_context)
    assert torch.equal(next_state.conv, threaded.conv)
    # The conv state after R rows is the last 9 rows of [history | normalized new rows], as the design's window select.
    assert next_state.conv.shape == (1, BRANCHES * HIDDEN, 9)


def _sharded_greedy(hidden: torch.Tensor, head_shards: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """LM head per vocabulary shard, local argmax, owner = first shard holding the maximum (resolve_greedy)."""

    logits = torch.cat([F.linear(hidden, shard) for shard in head_shards], dim=-1)
    values, indices = zip(*(F.linear(hidden, shard).max(dim=-1) for shard in head_shards))
    values = torch.stack(values, dim=-1).float()
    indices = torch.stack(indices, dim=-1)
    owner = values.argmax(dim=-1, keepdim=True)
    starts = torch.arange(TP) * VOCAB_SHARD
    return logits, (indices.gather(-1, owner) + starts[owner]).squeeze(-1)


def test_lm_head_logits_and_owner_argmax_are_row_independent() -> None:
    torch.manual_seed(6)
    hidden = _bf16(1, ROWS, HIDDEN)
    head_shards = [_bf16(VOCAB_SHARD, HIDDEN, scale=HIDDEN**-0.5) for _ in range(TP)]
    # Row 3: copy the winning head row into the next shard, so two owners tie at the maximum.
    logits, _ = _sharded_greedy(hidden, head_shards)
    winner = int(logits[0, 3].argmax())
    head_shards[(winner // VOCAB_SHARD + 1) % TP][winner % VOCAB_SHARD] = head_shards[winner // VOCAB_SHARD][
        winner % VOCAB_SHARD
    ]
    logits, tokens = _sharded_greedy(hidden, head_shards)
    per_row = [_sharded_greedy(hidden[:, r : r + 1], head_shards) for r in range(ROWS)]
    _assert_rows_equal(logits, [row[0] for row in per_row], "LM head logits")
    assert tokens.shape == (1, ROWS)
    assert tokens[0].tolist() == [int(row[1]) for row in per_row]
    # The tie on row 3 resolves to the lowest owner in both forms.
    row3 = logits[0, 3]
    tied = torch.nonzero(row3 == row3.max()).reshape(-1)
    assert tied.numel() >= 2 and int(tokens[0, 3]) == int(tied.min())


class _FakeMoEWeights:
    """Router, shared expert and per-expert weights at a small width; the oracle math is width-generic."""

    def __init__(self, hidden: int, intermediate: int, experts: int) -> None:
        self.placement = _placement(hidden, experts)
        self.config = self.placement.config
        self.expert_ranges = self.placement.expert_ranges
        self.router_weight = _bf16(experts, hidden, scale=hidden**-0.5)
        self.shared_gate_proj = _bf16(intermediate, hidden, scale=hidden**-0.5)
        self.shared_up_proj = _bf16(intermediate, hidden, scale=hidden**-0.5)
        self.shared_down_proj = _bf16(hidden, intermediate, scale=intermediate**-0.5)
        self.shared_scalar_gate = _bf16(1, hidden, scale=hidden**-0.5)
        self.experts = [
            Qwen38ExpertWeights(
                _bf16(2 * intermediate, hidden, scale=hidden**-0.5),
                _bf16(hidden, intermediate, scale=intermediate**-0.5),
                intermediate,
            )
            for _ in range(experts)
        ]

    def owner(self, expert_index: int) -> int:
        return next(d for d, (start, end) in enumerate(self.expert_ranges) if start <= expert_index < end)

    def expert(self, expert_index: int) -> Qwen38ExpertWeights:
        return self.experts[expert_index]

    def shared_shard(self, device_index: int) -> Qwen38SharedExpertShard:
        width = self.shared_gate_proj.shape[0] // TP
        start, end = device_index * width, (device_index + 1) * width
        hidden_start, hidden_end = self.placement.hidden_ranges[device_index]
        return Qwen38SharedExpertShard(
            gate_proj=self.shared_gate_proj[start:end],
            up_proj=self.shared_up_proj[start:end],
            down_proj=self.shared_down_proj[:, start:end],
            scalar_gate=self.shared_scalar_gate[:, hidden_start:hidden_end],
        )


def test_moe_router_topk_shared_and_expert_combine_are_row_independent() -> None:
    torch.manual_seed(8)
    moe = Qwen38MoE(_FakeMoEWeights(hidden=512, intermediate=128, experts=64))
    hidden = _bf16(1, ROWS, 512)

    output, routing = moe(hidden)
    per_row = [moe(hidden[:, r : r + 1]) for r in range(ROWS)]
    _assert_rows_equal(output, [row[0] for row in per_row], "MoE output")
    assert torch.equal(routing.indices, torch.cat([row[1].indices for row in per_row], dim=0))
    assert torch.equal(routing.scores, torch.cat([row[1].scores for row in per_row], dim=0))
    assert torch.equal(routing.logits, torch.cat([row[1].logits for row in per_row], dim=0))
    # The five rows touch more distinct experts than any single row: the R1 cost driver, not a numerics change.
    assert len(set(routing.indices.reshape(-1).tolist())) > 10

    ep4 = moe.expert_parallel_forward(hidden)
    per_row_ep4 = [moe.expert_parallel_forward(hidden[:, r : r + 1]) for r in range(ROWS)]
    for d in range(TP):
        _assert_rows_equal(ep4.hidden_shards[d], [row.hidden_shards[d] for row in per_row_ep4], f"MoE EP4 shard {d}")
        # Local selection over five rows is the union of the per-row selections, in row order.
        assert ep4.local_selected_experts[d] == tuple(
            expert for row in per_row_ep4 for expert in row.local_selected_experts[d]
        )
    # The score-weighted combine (the device fast-reduce input form [TOP_K, R, H]) is per row.
    expert_outputs = _bf16(ROWS, 10, 512)
    combined = weighted_routed_reduce(expert_outputs, routing.scores)
    _assert_rows_equal(
        combined.unsqueeze(0),
        [
            weighted_routed_reduce(expert_outputs[r : r + 1], routing.scores[r : r + 1]).unsqueeze(0)
            for r in range(ROWS)
        ],
        "weighted routed reduce",
    )
