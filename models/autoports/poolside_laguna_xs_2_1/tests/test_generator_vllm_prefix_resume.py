# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract tests for resumed/chunked vLLM prefill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.autoports.poolside_laguna_xs_2_1.tt import multichip_decoder as multichip_decoder_module
from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM, _prefill_rope_capacity
from models.autoports.poolside_laguna_xs_2_1.tt.model import LagunaModel
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder


class _RecordingGenerator:
    def __init__(self):
        self.uploads = []

    def _tokens_to_device(self, tokens):
        copied = tokens.clone()
        self.uploads.append(copied)
        return copied


def test_prefix_cache_capabilities_are_fail_closed_by_default():
    assert LagunaForCausalLM._PREFIX_CACHE_ENABLED is False
    assert LagunaForCausalLM.model_capabilities["supports_prefix_caching"] is False
    assert LagunaForCausalLM.model_capabilities["supports_prefix_caching_with_sliding_window"] is False


class _RecordingModel:
    def __init__(self):
        self.prefills = []

    def embed_prefill(self, tokens):
        return tokens

    def prefill_layers(
        self,
        hidden,
        kv_cache,
        page_table,
        *,
        fill_page_table,
        fill_page_table_base_pos,
        user_id,
        start_pos,
        runtime_offsets,
    ):
        self.prefills.append(
            {
                "hidden": hidden.clone(),
                "kv_cache": kv_cache,
                "page_table": page_table,
                "fill_page_table": fill_page_table,
                "fill_page_table_base_pos": fill_page_table_base_pos,
                "user_id": user_id,
                "start_pos": start_pos,
                "runtime_offsets": runtime_offsets,
            }
        )
        return hidden

    @staticmethod
    def logits_to_host(shards):
        return shards


def _bridge(*, max_model_len=32, spec_mode=""):
    """Build only the adapter surface exercised by prefill_forward; no TT device is opened."""
    bridge = object.__new__(LagunaForCausalLM)
    bridge.gen = _RecordingGenerator()
    bridge.model = _RecordingModel()
    bridge.D = 2
    bridge.max_model_len = max_model_len
    bridge.vocab = 3
    bridge._spec_mode = spec_mode
    bridge._spec_prefill_seq = []
    bridge._spec_next_pos = None
    bridge._test_pt_calls = []
    bridge._test_fill_pt_calls = []
    bridge._test_bucket_args = []
    bridge._test_last_rows = []

    def prefill_pt(page_table):
        bridge._test_pt_calls.append(page_table)
        return page_table

    def prefill_fill_pt(page_table):
        bridge._test_fill_pt_calls.append(page_table)
        return page_table

    def bucket_len(chunk_len):
        bridge._test_bucket_args.append(chunk_len)
        return 4 if chunk_len <= 4 else 8

    def last_token_shards(hidden, chunk_len, bucket_len):
        bridge._test_last_rows.append((chunk_len, bucket_len))
        return torch.zeros((1, bridge.vocab), dtype=torch.float32)

    bridge._prefill_pt = prefill_pt
    bridge._prefill_fill_pt = prefill_fill_pt
    bridge._bucket_len = bucket_len
    bridge._prefill_page_table_width = lambda _block_size: 10
    bridge._runtime_offsets_for_prefill = lambda L, start, bs: ("runtime", L, start, bs)
    bridge._last_token_shards = last_token_shards
    return bridge


def _prefix_bucket_bridge():
    bridge = object.__new__(LagunaForCausalLM)
    bridge.D = 2
    bridge._PREFIX_CACHE_ENABLED = True
    bridge.model = SimpleNamespace(layers=[SimpleNamespace(_prefill_pipe_chunk=8192)])
    buckets = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    bridge._prefill_bucket_lens = lambda: buckets
    bridge._bucket_len = lambda length: next(bucket for bucket in buckets if int(length) <= bucket)
    return bridge


@pytest.mark.parametrize(
    ("label", "start", "real_end", "expected_bucket"),
    [
        ("short_after_first_chunk", 8192, 8257, 8192),
        ("partial_32k", 24576, 25000, 8192),
        ("full_32k_tail", 24576, 32768, 8192),
        ("partial_65k", 57344, 57409, 8192),
        ("full_65k_tail", 57344, 65536, 8192),
        ("multi_chunk_suffix", 16384, 25384, 16384),
    ],
)
def test_prefix_resume_uses_aligned_scheduler_start_and_canonical_minimum_bucket(
    label, start, real_end, expected_bucket
):
    bridge = _prefix_bucket_bridge()

    assert bridge._prefill_bucket_for_range(real_end - start, start, 64) == expected_bucket, label


@pytest.mark.parametrize("start", [64, 2048, 32704, 65472])
def test_prefix_resume_rejects_noncanonical_scheduler_start(start):
    bridge = _prefix_bucket_bridge()

    with pytest.raises(ValueError, match="not aligned to canonical outer-chunk quantum 8192"):
        bridge._prefill_bucket_for_range(65, start, 64)


def test_prefix_bucket_rule_preserves_cold_cache_off_and_non_d2_paths():
    bridge = _prefix_bucket_bridge()
    assert bridge._prefill_bucket_for_range(65, 0, 64) == 128

    bridge._PREFIX_CACHE_ENABLED = False
    assert bridge._prefill_bucket_for_range(65, 2048, 64) == 128

    bridge._PREFIX_CACHE_ENABLED = True
    bridge.D = 1
    assert bridge._prefill_bucket_for_range(65, 2048, 64) == 128


def test_prefix_resume_rejects_model_admission_quantum_drift():
    bridge = _prefix_bucket_bridge()
    bridge.model.layers[0]._prefill_pipe_chunk = 4096

    with pytest.raises(RuntimeError, match="requires prefill outer-chunk quantum 8192.*configured for 4096"):
        bridge._prefill_bucket_for_range(65, 8192, 64)


def test_prefix_resume_rejects_partial_outer_chunk_bucket():
    bridge = _prefix_bucket_bridge()
    bridge._prefill_bucket_lens = lambda: [8192, 12_000]
    bridge._bucket_len = lambda _length: 12_000

    with pytest.raises(ValueError, match="bucket 12000 is not a whole multiple.*8192"):
        bridge._prefill_bucket_for_range(9000, 8192, 64)


def test_uncanonical_prefix_hit_fails_before_page_table_or_device_work():
    bridge = _bridge(max_model_len=8192)
    bridge._PREFIX_CACHE_ENABLED = True
    bridge.model.layers = [SimpleNamespace(_prefill_pipe_chunk=8192)]
    bridge._prefill_bucket_lens = lambda: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    with pytest.raises(ValueError, match="start_pos 2048 is not aligned.*8192"):
        bridge.prefill_forward(
            torch.arange(2113, dtype=torch.int64).reshape(1, -1),
            page_table=torch.arange(34, dtype=torch.int32).reshape(1, -1),
            kv_cache=[{"block_size": 64, "scratch_block_idx": 99}],
            prompt_lens=[2113],
            start_pos=[2048],
        )

    assert bridge._test_pt_calls == []
    assert bridge._test_fill_pt_calls == []
    assert bridge.gen.uploads == []
    assert bridge.model.prefills == []


def test_prefix_resume_never_reads_or_writes_before_scheduler_start(capsys):
    bridge = _bridge(max_model_len=16384)
    bridge._PREFIX_CACHE_ENABLED = True
    bridge.model.layers = [SimpleNamespace(_prefill_pipe_chunk=8192)]
    bridge._prefill_bucket_lens = lambda: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    bridge._prefill_page_table_width = lambda _block_size: 256
    start, end = 8192, 8257
    tokens = torch.arange(end, dtype=torch.int64).reshape(1, -1)
    page_table = torch.arange(130, dtype=torch.int32).reshape(1, -1)
    kv_cache = [{"block_size": 64, "scratch_block_idx": 999}]

    bridge.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[end],
        start_pos=[start],
    )

    uploaded = bridge.gen.uploads[0]
    call = bridge.model.prefills[0]
    assert uploaded.shape == (8192,)
    assert torch.equal(uploaded[: end - start], tokens[0, start:end])
    assert torch.count_nonzero(uploaded[end - start :]) == 0
    assert call["start_pos"] == start
    assert call["fill_page_table_base_pos"] == start
    assert call["runtime_offsets"] == ("runtime", 8192, start, 64)
    assert call["fill_page_table"][0, :2].tolist() == [128, 129]
    assert torch.all(call["fill_page_table"][0, 2:] == -1)
    assert bridge._test_last_rows == [(end - start, 8192)]
    log = capsys.readouterr().out
    assert (
        "scheduled_start=8192 effective_start=8192 real_end=8257 "
        "compute_bucket=8192 canonical_chunk=8192"
    ) in log


def test_resumed_prefill_slices_absolute_range_but_retains_absolute_start():
    bridge = _bridge()
    tokens = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ]
    )
    page_table = torch.arange(20, dtype=torch.int32).reshape(2, 10)
    kv_cache = [{"block_size": 1, "scratch_block_idx": 99}]

    logits = bridge.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[5, 7],  # absolute exclusive ends
        start_pos=[2, 5],  # absolute cached-prefix lengths
    )

    assert bridge._test_bucket_args == [3, 2]
    assert bridge.gen.uploads[0].tolist() == [12, 13, 14, 0]
    assert bridge.gen.uploads[1].tolist() == [25, 26, 0, 0]
    assert [call["start_pos"] for call in bridge.model.prefills] == [2, 5]
    assert [call["user_id"] for call in bridge.model.prefills] == [0, 1]
    guarded_page_table = bridge._test_pt_calls[0]
    fill_page_table = bridge._test_fill_pt_calls[0]
    assert guarded_page_table[0].tolist() == [0, 1, 2, 3, 4, 99, 99, 99, 99, 99]
    assert guarded_page_table[1].tolist() == [10, 11, 12, 13, 14, 15, 16, 99, 99, 99]
    assert fill_page_table[0].tolist() == [2, 3, 4, -1, -1, -1, -1, -1, -1, -1]
    assert fill_page_table[1].tolist() == [15, 16, -1, -1, -1, -1, -1, -1, -1, -1]
    assert all(torch.equal(call["page_table"], guarded_page_table) for call in bridge.model.prefills)
    assert all(torch.equal(call["fill_page_table"], fill_page_table) for call in bridge.model.prefills)
    assert all(call["kv_cache"] is kv_cache for call in bridge.model.prefills)
    assert [call["fill_page_table_base_pos"] for call in bridge.model.prefills] == [2, 5]
    assert [call["runtime_offsets"] for call in bridge.model.prefills] == [
        ("runtime", 4, 2, 1),
        ("runtime", 4, 5, 1),
    ]
    assert bridge._test_last_rows == [(3, 4), (2, 4)]
    assert logits.shape == (2, 1, bridge.vocab)


def test_d1_cold_prefill_keeps_legacy_runtime_path():
    bridge = _bridge(max_model_len=16)
    bridge.D = 1
    bridge._runtime_offsets_for_prefill = lambda L, start, bs: LagunaForCausalLM._runtime_offsets_for_prefill(
        bridge, L, start, bs
    )

    bridge.prefill_forward(
        torch.arange(8).reshape(1, -1),
        page_table=torch.arange(10, dtype=torch.int32).reshape(1, -1),
        kv_cache=[{"block_size": 1, "scratch_block_idx": 99}],
        prompt_lens=[8],
        start_pos=[0],
    )

    assert bridge.model.prefills[0]["runtime_offsets"] is None
    assert bridge.model.prefills[0]["fill_page_table_base_pos"] == 0


def test_resumed_prefill_stashes_only_the_scheduled_suffix_for_spec_decode():
    bridge = _bridge(spec_mode="1")
    bridge._spec_prefill_seq = [10, 11, 12, 13]

    bridge.prefill_forward(
        torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
        page_table=torch.arange(8, dtype=torch.int32).reshape(1, 8),
        kv_cache=[{"block_size": 1, "scratch_block_idx": 99}],
        prompt_lens=[6],
        start_pos=[4],
    )

    assert bridge._spec_prefill_seq == [10, 11, 12, 13, 14, 15]


def test_cold_bucket_padding_uses_scratch_for_attention_and_skip_for_fill():
    original = torch.tensor([[41, 7, 23, 0, 0]], dtype=torch.int32)

    attention, fill = LagunaForCausalLM._protect_prefill_padding_blocks(
        original,
        ranges=[(0, 129, 129)],
        bucket_lens=[256],
        block_size=64,
        scratch_block_idx=101,
    )

    # Real positions 0..128 own logical blocks 0..2. Bucket padding reaches block 3,
    # whose plugin-provided zero would otherwise overwrite real physical block 0.
    assert attention.tolist() == [[41, 7, 23, 101, 101]]
    assert fill.tolist() == [[41, 7, 23, -1, -1]]
    assert original.tolist() == [[41, 7, 23, 0, 0]]  # scheduler/decode table stays untouched


def test_resumed_bucket_padding_splits_attention_and_fill_after_absolute_real_end():
    original = torch.tensor([[31, 32, 33, 34, 35, 0, 0]], dtype=torch.int32)

    attention, fill = LagunaForCausalLM._protect_prefill_padding_blocks(
        original,
        ranges=[(128, 257, 129)],
        bucket_lens=[256],
        block_size=64,
        scratch_block_idx=101,
    )

    # The resumed write starts at logical block 2. The real suffix owns blocks 2..4;
    # the fourth bucket block (logical 5) is adapter padding and must not alias block 0.
    assert attention.tolist() == [[31, 32, 33, 34, 35, 101, 101]]
    assert fill.tolist() == [[33, 34, 35, -1, -1, -1, -1]]
    assert original.tolist() == [[31, 32, 33, 34, 35, 0, 0]]


def test_padding_guard_rejects_unaligned_resume_and_insufficient_table_width():
    with pytest.raises(ValueError, match="not aligned"):
        LagunaForCausalLM._protect_prefill_padding_blocks(
            torch.zeros((1, 8), dtype=torch.int32),
            ranges=[(1, 130, 129)],
            bucket_lens=[256],
            block_size=64,
            scratch_block_idx=101,
        )

    with pytest.raises(ValueError, match="beyond page-table width"):
        LagunaForCausalLM._protect_prefill_padding_blocks(
            torch.zeros((1, 3), dtype=torch.int32),
            ranges=[(0, 129, 129)],
            bucket_lens=[256],
            block_size=64,
            scratch_block_idx=101,
        )


def test_near_cap_suffix_extends_prefill_table_and_rope_horizon():
    max_model_len = 131072
    start = 98240
    end = max_model_len
    chunk_len = end - start
    bucket_len = 65536
    scheduler_width = max_model_len // 64
    fixed_prefill_width = (max_model_len + max_model_len) // 64
    original = torch.arange(scheduler_width, dtype=torch.int32).reshape(1, -1)

    attention, fill = LagunaForCausalLM._protect_prefill_padding_blocks(
        original,
        ranges=[(start, end, chunk_len)],
        bucket_lens=[bucket_len],
        block_size=64,
        scratch_block_idx=scheduler_width,
        target_width=fixed_prefill_width,
    )

    padded_block_end = (start + bucket_len) // 64
    assert attention.shape == fill.shape == (1, 4096)
    assert attention[0, scheduler_width - 1].item() == scheduler_width - 1
    suffix_blocks = (end - start) // 64
    assert fill[0, :suffix_blocks].tolist() == list(range(start // 64, scheduler_width))
    assert torch.all(fill[0, suffix_blocks:] == -1)
    assert torch.all(attention[0, scheduler_width:padded_block_end] == scheduler_width)
    assert torch.all(fill[0, scheduler_width:padded_block_end] == -1)
    assert _prefill_rope_capacity(65536) == 131072
    assert _prefill_rope_capacity(max_model_len) == 262144


def test_verify_prefill_uses_paired_attention_and_fill_guards():
    bridge = _bridge(max_model_len=512)
    bridge._bucket_len = lambda _chunk_len: 256
    bridge._prefill_page_table_width = lambda _block_size: 16
    bridge._prefill_state = lambda: object()
    selected_rows = []

    def row_logits(_hidden, row, bucket_len, _state):
        selected_rows.append((row, bucket_len))
        return torch.zeros(bridge.vocab)

    bridge._row_logits = row_logits
    page_table = torch.tensor([[4, 8, 12, 16, 20, 0, 0, 0]], dtype=torch.int32)
    kv_cache = [{"block_size": 64, "scratch_block_idx": 99}]

    logits = bridge.verify_forward(
        torch.arange(129).reshape(1, -1),
        start_pos=128,
        page_table=page_table,
        kv_cache=kv_cache,
        logit_rows=[127, 128],
    )

    guarded = bridge._test_pt_calls[0]
    fill = bridge._test_fill_pt_calls[0]
    assert guarded.shape == (1, 16)
    assert fill.shape == guarded.shape
    assert guarded[0, :5].tolist() == [4, 8, 12, 16, 20]
    assert torch.all(guarded[0, 5:] == 99)
    assert fill[0, :3].tolist() == [12, 16, 20]
    assert torch.all(fill[0, 3:] == -1)
    assert bridge.model.prefills[0]["start_pos"] == 128
    assert bridge.model.prefills[0]["fill_page_table_base_pos"] == 128
    assert torch.equal(bridge.model.prefills[0]["page_table"], guarded)
    assert torch.equal(bridge.model.prefills[0]["fill_page_table"], fill)
    assert selected_rows == [(127, 256), (128, 256)]
    assert logits.shape == (2, bridge.vocab)


def test_hybrid_list_path_builds_distinct_attention_and_fill_tables_without_mutating_inputs():
    bridge = _bridge(max_model_len=16)
    bridge._group_kinds = lambda: ["full", "sliding"]
    bridge._prefill_pt_grouped = lambda tables: tables
    bridge._prefill_fill_pt_grouped = lambda tables: tables
    original = [
        torch.tensor([[2, 3, 4, 0, 0, 0]], dtype=torch.int32),
        torch.tensor([[12, 13, 14, 0, 0, 0]], dtype=torch.int32),
    ]
    snapshots = [table.clone() for table in original]
    kv_cache = [
        {"block_size": 1, "scratch_block_idx": 90},
        {"block_size": 1, "scratch_block_idx": 91},
    ]

    attention, fill = bridge._prepare_prefill_page_tables(
        None,
        original,
        kv_cache,
        ranges=[(0, 3, 3)],
        bucket_lens=[4],
        operation="test prefill",
    )

    assert [table[0].tolist() for table in attention] == [
        [2, 3, 4, 90, 90, 90, 90, 90, 90, 90],
        [12, 13, 14, 91, 91, 91, 91, 91, 91, 91],
    ]
    assert [table[0].tolist() for table in fill] == [
        [2, 3, 4, -1, -1, -1, -1, -1, -1, -1],
        [12, 13, 14, -1, -1, -1, -1, -1, -1, -1],
    ]
    assert all(torch.equal(table, snapshot) for table, snapshot in zip(original, snapshots))


def test_laguna_model_propagates_per_layer_fill_tables_independently(monkeypatch):
    calls = []

    class _Layer:
        def __init__(self, index):
            self.index = index
            self.cfg = type("Cfg", (), {"attention_type": "full_attention"})()

        def prefill_forward(
            self,
            hidden,
            kv_cache,
            page_table,
            *,
            fill_page_table,
            fill_page_table_base_pos,
            user_id,
            start_pos,
            rope_mats,
            runtime_offsets,
        ):
            calls.append(
                (
                    self.index,
                    kv_cache,
                    page_table,
                    fill_page_table,
                    fill_page_table_base_pos,
                    user_id,
                    start_pos,
                    rope_mats,
                    runtime_offsets,
                )
            )
            return hidden

    model = object.__new__(LagunaModel)
    model.layers = [_Layer(0), _Layer(1)]
    monkeypatch.setenv("TT_LAGUNA_NO_ROPE_HOIST", "1")
    hidden = torch.zeros((1, 4, 1))
    attention = [object(), object()]
    fill = [object(), object()]
    kv_cache = [object(), object()]

    out = model.prefill_layers(
        hidden,
        kv_cache,
        attention,
        fill_page_table=fill,
        fill_page_table_base_pos=64,
        user_id=3,
        start_pos=64,
    )

    assert out is hidden
    assert calls == [
        (0, kv_cache[0], attention[0], fill[0], 64, 3, 64, None, None),
        (1, kv_cache[1], attention[1], fill[1], 64, 3, 64, None, None),
    ]


def test_multichip_decoder_reuses_cold_fill_table_but_keeps_attention_table(monkeypatch):
    dec = object.__new__(MultichipDecoder)
    dec.PIPE_CHUNK = 256
    dec.cfg = type("Cfg", (), {"num_heads": 1, "head_dim": 1, "hidden": 1})()
    dec.w = {"input_ln": object(), "wo": object(), "post_ln": object()}
    dec._ck_o = object()
    dec._rms = lambda x, _weight: x
    q = k = v = torch.zeros((1, 1, 128, 1))
    dec._qkv_roped = lambda *_args, **_kwargs: (q, k, v)
    dec._cast_fill = lambda tensor, _dtype: tensor
    attention_calls = []
    dec._prefill_attention = lambda _q, _k, _v, _cache, pt, *_args, **_kwargs: (
        attention_calls.append(pt) or torch.zeros((1, 1, 128, 1))
    )
    dec._gate = lambda attention, _hidden: attention
    dec._reduce = lambda tensor: tensor
    dec._mlp = lambda _hidden, seq, sharded: torch.zeros((1, 1, seq, 1))

    fill_calls = []

    def fake_slice(tensor, starts, ends):
        slices = tuple(slice(start, end) for start, end in zip(starts, ends))
        return tensor[slices]

    monkeypatch.setattr(multichip_decoder_module.ttnn, "slice", fake_slice)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "linear", lambda tensor, *_args, **_kwargs: tensor)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "add", torch.add)
    monkeypatch.setattr(
        multichip_decoder_module.ttnn.experimental,
        "paged_fill_cache",
        lambda _cache, _value, pt, **_kwargs: fill_calls.append(pt.clone()),
    )
    monkeypatch.setattr(
        multichip_decoder_module.ttnn.experimental,
        "nlp_concat_heads",
        lambda tensor, **_kwargs: tensor,
    )
    attention = torch.tensor([[10, 99, 99, 99]], dtype=torch.int32)
    fill = torch.tensor([[10, -1, -1, -1]], dtype=torch.int32)
    cache = {"k": object(), "v": object(), "dtype": torch.float32, "block_size": 64}

    MultichipDecoder.prefill_forward(
        dec,
        torch.zeros((1, 128, 1)),
        cache,
        attention,
        fill_page_table=fill,
        user_id=0,
        start_pos=0,
    )

    assert len(fill_calls) == 2
    assert all(table.tolist() == [[10, -1, -1, -1]] for table in fill_calls)
    assert len(attention_calls) == 1 and attention_calls[0] is attention


def test_multichip_decoder_threads_fill_table_into_pipelined_path():
    dec = object.__new__(MultichipDecoder)
    dec.PIPE_CHUNK = 64
    calls = []
    dec._prefill_pipelined = lambda *args, **kwargs: calls.append((args, kwargs)) or "done"
    attention = object()
    fill = object()
    cache = object()
    hidden = torch.zeros((1, 128, 1))

    result = MultichipDecoder.prefill_forward(
        dec,
        hidden,
        cache,
        attention,
        fill_page_table=fill,
        user_id=2,
        start_pos=64,
    )

    assert result == "done"
    assert calls == [
        (
            (hidden, cache, attention, fill, 2, 64),
            {
                "fill_page_table_base_pos": 0,
                "rope_mats": None,
                "runtime_offsets": None,
            },
        )
    ]


def test_terminal_selector_uses_relative_row_and_persistent_matmul_output(monkeypatch):
    bridge = object.__new__(LagunaForCausalLM)
    bridge.hidden = 4
    bridge.vocab = 3
    bridge.mesh_device = object()
    hidden = object()
    hidden_4d = object()
    selector = object()
    persistent_output = object()
    host_selector = object()
    state = {"sel": {8: selector}, "last_h": persistent_output}
    bridge._pf = state
    bridge._prefill_state = lambda: state

    lm_head_inputs = []

    class _Model:
        @staticmethod
        def lm_head_shards_decode(value):
            lm_head_inputs.append(value)
            return object()

        @staticmethod
        def logits_to_host(_value):
            return torch.arange(bridge.vocab, dtype=torch.float32).reshape(1, -1)

    bridge.model = _Model()
    module = __import__(LagunaForCausalLM.__module__, fromlist=["generator_vllm"])
    uploads = []
    matmuls = []

    def fake_from_torch(tensor, **kwargs):
        assert "device" not in kwargs  # host TT tensor: no device allocation under a resident trace
        uploads.append((tensor.clone(), kwargs))
        return host_selector

    def fake_copy(source, destination):
        uploads.append((source, destination))

    def fake_reshape(value, shape):
        assert value is hidden
        assert shape == (1, 1, 8, bridge.hidden)
        return hidden_4d

    def fake_matmul(lhs, rhs, **kwargs):
        matmuls.append((lhs, rhs, kwargs))
        return object()  # callers must use the persistent output, not this return handle

    monkeypatch.setattr(module, "_replicate", lambda mesh: ("replicate", mesh))
    monkeypatch.setattr(module.ttnn, "from_torch", fake_from_torch)
    monkeypatch.setattr(module.ttnn, "copy_host_to_device_tensor", fake_copy)
    monkeypatch.setattr(module.ttnn, "reshape", fake_reshape)
    monkeypatch.setattr(module.ttnn, "matmul", fake_matmul)
    monkeypatch.setenv("TT_LAGUNA_SELECTOR", "0")  # retired debug knob must not disable the safe path

    bridge._last_token_shards(hidden, plen=3, L=8)
    logits = bridge._row_logits(hidden, row=5, L=8, st=state)

    # ``plen`` is the relative resumed-chunk length, so its terminal row is plen - 1, not an
    # absolute prompt position. The verify path selects its explicitly requested relative row.
    assert uploads[0][0].reshape(-1).nonzero().reshape(-1).tolist() == [2]
    assert uploads[2][0].reshape(-1).nonzero().reshape(-1).tolist() == [5]
    assert uploads[1] == (host_selector, selector)
    assert uploads[3] == (host_selector, selector)
    assert all(call[:2] == (selector, hidden_4d) for call in matmuls)
    assert all(call[2]["optional_output_tensor"] is persistent_output for call in matmuls)
    assert lm_head_inputs == [persistent_output, persistent_output]
    assert logits.tolist() == [0.0, 1.0, 2.0]


def test_uniform_allocator_reserves_private_scratch_without_growing_logical_pool(monkeypatch):
    bridge = object.__new__(LagunaForCausalLM)
    bridge._kv_dtype = object()
    bridge.mesh_device = object()
    bridge._decode = {"stale": object()}
    bridge._verify_dec = {"stale": object()}
    bridge._report_dram = lambda *_args, **_kwargs: None
    allocations = []

    def fake_from_torch(tensor, **_kwargs):
        allocations.append(tensor.clone())
        return tensor

    module = __import__(LagunaForCausalLM.__module__, fromlist=["generator_vllm"])
    monkeypatch.setattr(module.ttnn, "from_torch", fake_from_torch)
    monkeypatch.setattr(module, "_replicate", lambda _mesh: None)

    cache = bridge.allocate_kv_cache_per_layer([((100, 2, 64, 128), object(), 0)])

    assert [tuple(t.shape) for t in allocations] == [(101, 2, 64, 128), (101, 2, 64, 128)]
    assert cache[0]["blocks_per_user"] == 100
    assert cache[0]["scratch_block_idx"] == 100
    assert tuple(cache[0]["k"].shape) == (101, 2, 64, 128)
    assert bridge._decode == {}
    assert bridge._verify_dec == {}


def test_prefill_bucket_ladder_warms_once_across_plugin_two_phase_calls():
    bridge = object.__new__(LagunaForCausalLM)
    bridge.max_model_len = 128
    bridge.max_batch_size = 2
    bridge.D = 1  # legacy topology: no D2 runtime-offset or resumed-program warm passes
    bridge._max_blocks = None
    bridge.already_warmed_up_prefill = False
    bridge._prefill_programs_warmed = False
    bridge._in_prefill_warmup = False
    state_calls = []
    prefill_calls = []
    pt_shapes = []
    fill_pt_shapes = []
    dram_reports = []
    bridge._prefill_state = lambda *_args: state_calls.append(True)
    bridge._prefill_bucket_lens = lambda: [32, 64, 128]
    bridge.prefill_forward = lambda *args, **kwargs: prefill_calls.append((args, kwargs))
    bridge._prefill_pt = lambda pt: pt_shapes.append(tuple(pt.shape))
    bridge._prefill_fill_pt = lambda pt: fill_pt_shapes.append(tuple(pt.shape))
    bridge._report_dram = lambda stage: dram_reports.append(stage)
    kv_cache = [
        {
            "block_size": 64,
            "blocks_per_user": 2,
            "scratch_block_idx": 2,
        }
    ]

    bridge.warmup_model_prefill(kv_cache=kv_cache, enable_trace=False)
    # This is exactly what TTModelRunner does between phases.
    bridge.already_warmed_up_prefill = False
    bridge.warmup_model_prefill(kv_cache=kv_cache, enable_trace=True)

    assert bridge._max_blocks == 2  # derived before phase-1 prefill, not learned later from decode
    assert len(prefill_calls) == 3  # one 32/64/128 ladder, not two
    assert state_calls == [True]
    assert pt_shapes == [(1, 8), (2, 8)]
    assert fill_pt_shapes == [(1, 8), (2, 8)]
    assert dram_reports == ["prefill_warmup"]
    assert bridge.already_warmed_up_prefill is True


def test_cache_off_d2_warmup_preserves_resumed_single_shot_shapes():
    bridge = object.__new__(LagunaForCausalLM)
    bridge.max_model_len = 128
    bridge.max_batch_size = 1
    bridge.D = 2
    bridge._PREFIX_CACHE_ENABLED = False
    bridge.model = type("Model", (), {"layers": [type("Layer", (), {"PIPE_CHUNK": 64})()]})()
    bridge._max_blocks = None
    bridge.already_warmed_up_prefill = False
    bridge._prefill_programs_warmed = False
    bridge._in_prefill_warmup = False
    calls = []
    bridge._prefill_state = lambda *_args: None
    bridge._prefill_bucket_lens = lambda: [32, 64, 128]
    bridge.prefill_forward = lambda *args, **kwargs: calls.append(kwargs)
    bridge._prefill_pt = lambda pt: pt
    bridge._prefill_fill_pt = lambda pt: pt
    bridge._report_dram = lambda _stage: None
    kv_cache = [{"block_size": 64, "blocks_per_user": 4, "scratch_block_idx": 4}]

    bridge.warmup_model_prefill(kv_cache=kv_cache, enable_trace=False)

    assert [call["start_pos"] for call in calls] == [[0], [0], [0], [64], [64]]
    assert [call["prompt_lens"] for call in calls] == [[32], [64], [128], [96], [128]]


@pytest.mark.parametrize("prefix_enabled", [False, True])
def test_production_d2_warmup_has_exact_cold_ladder_and_only_cache_off_resumed_shapes(prefix_enabled):
    buckets = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    bridge = object.__new__(LagunaForCausalLM)
    bridge.max_model_len = 131072
    bridge.max_batch_size = 1
    bridge.D = 2
    bridge._PREFIX_CACHE_ENABLED = prefix_enabled
    bridge.model = type(
        "Model",
        (),
        {"layers": [type("Layer", (), {"PIPE_CHUNK": 2048, "_prefill_pipe_chunk": 8192})()]},
    )()
    bridge._max_blocks = None
    bridge.already_warmed_up_prefill = False
    bridge._prefill_programs_warmed = False
    bridge._in_prefill_warmup = False
    calls = []
    bridge._prefill_state = lambda *_args: None
    bridge._prefill_bucket_lens = lambda: buckets
    bridge.prefill_forward = lambda *args, **kwargs: calls.append(kwargs)
    bridge._prefill_pt = lambda pt: pt
    bridge._prefill_fill_pt = lambda pt: pt
    bridge._report_dram = lambda _stage: None
    kv_cache = [{"block_size": 64, "blocks_per_user": 2048, "scratch_block_idx": 2048}]

    bridge.warmup_model_prefill(kv_cache=kv_cache, enable_trace=False)

    cold = [call["prompt_lens"][0] for call in calls if call["start_pos"] == [0]]
    resumed = [call["prompt_lens"][0] - call["start_pos"][0] for call in calls if call["start_pos"] == [64]]
    assert cold == buckets
    assert resumed == ([] if prefix_enabled else [32, 64, 128, 256, 512, 1024, 2048])

@pytest.mark.parametrize(
    ("tokens", "prompt_lens", "start_pos", "max_model_len", "match"),
    [
        (torch.zeros((1, 1, 4)), [4], [0], 8, "rank 1 or 2"),
        (torch.zeros((0, 4)), None, None, 8, "at least one request"),
        (torch.zeros((1, 4)), [4, 4], [0], 8, "prompt_lens has 2 entries"),
        (torch.zeros((1, 4)), [4], [0, 0], 8, "start_pos has 2 entries"),
        (torch.zeros((1, 4)), [4], [-1], 8, "negative start_pos"),
        (torch.zeros((1, 4)), [2], [2], 8, "empty or reversed range"),
        (torch.zeros((1, 4)), [2], [3], 8, "empty or reversed range"),
        (torch.zeros((1, 4)), [5], [0], 8, "beyond supplied token width"),
        (torch.zeros((1, 4)), [4], [0], 3, "beyond max_model_len"),
    ],
)
def test_prefill_rejects_invalid_absolute_ranges_before_device_work(
    tokens, prompt_lens, start_pos, max_model_len, match
):
    bridge = _bridge(max_model_len=max_model_len)

    with pytest.raises(ValueError, match=match):
        bridge.prefill_forward(tokens, prompt_lens=prompt_lens, start_pos=start_pos)

    assert bridge._test_pt_calls == []
    assert bridge._test_fill_pt_calls == []
    assert bridge.gen.uploads == []
    assert bridge.model.prefills == []
