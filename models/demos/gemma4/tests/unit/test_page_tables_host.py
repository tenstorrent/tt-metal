# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: when a prefill's per-layer page-table install forces the H2D (``Gemma4Generator._install_per_layer_page_tables``):
always after a decode wrote the row count's device buffers (a decode trace input rewrites them) and on the first
write, never when the last writer was prefill (the host-unchanged shortcut then decides). No device, no conftest."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import models.demos.gemma4.tt.ccl as ccl_module
import models.demos.gemma4.tt.model as model_module
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tests.hf_reference_cache import hf_reference_cache_dir, hf_reference_key, hf_reference_logits
from models.demos.gemma4.tt.async_decode import merge_async_ahead_decode_tokens
from models.demos.gemma4.tt.attention import operations as ops
from models.demos.gemma4.tt.capacity import (
    BFP8_TILE_BYTES,
    core_split,
    global_layer_plan,
    kv_bytes_per_core_per_layer,
    max_cores_for_queue_depth,
    reads_in_flight,
)
from models.demos.gemma4.tt.dram_sharded import prefill_row_chunks
from models.demos.gemma4.tt.generator import (
    _BLACKHOLE_MAX_BATCHED_PREFILL_USERS,
    _DEFAULT_MAX_BATCHED_PREFILL_USERS,
    SDPA_CHUNK_ALIGN,
    ChunkedPrefillPageTableGuardMixin,
    Gemma4Generator,
    align_num_cached_tokens_to_sdpa,
    layer_local_kv_heads,
    mask_page_table_columns_past_allocation,
    max_batched_prefill_users,
    paged_block_size_for_view,
    resolve_batched_prefill_chunk_users,
)
from models.demos.gemma4.tt.model import Gemma4Model, _decode_logits_rows
from models.tt_transformers.tt.common import get_padded_prefill_len, num_blocks_in_seq
from models.tt_transformers.tt.generator import Generator

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:  # the CI unit runners have no vLLM: those sections skip
    _HAS_VLLM = False


def _model():
    calls = []

    def update(tables, force=False):
        calls.append(force)

    return SimpleNamespace(update_persistent_per_layer_page_tables=update, _calls=calls)


def test_prefill_forces_after_decode_and_first_write_only():
    m = _model()
    tables = [torch.zeros(1, 8, dtype=torch.int32)] * 3
    Gemma4Generator._install_per_layer_page_tables(m, tables, writer="prefill")  # first write: forced
    Gemma4Generator._install_per_layer_page_tables(
        m, tables, writer="prefill"
    )  # chunk 2 of the same prompt: not forced
    Gemma4Generator._install_per_layer_page_tables(m, tables, writer="decode")  # decode never forces
    Gemma4Generator._install_per_layer_page_tables(m, tables, writer="prefill")  # after decode: forced again
    assert m._calls == [True, False, False, True], m._calls
    assert m._g4_page_table_writer_by_rows == {1: "prefill"}


def test_row_counts_are_tracked_independently():
    m = _model()
    one = [torch.zeros(1, 8, dtype=torch.int32)]
    many = [torch.zeros(32, 8, dtype=torch.int32)]
    Gemma4Generator._install_per_layer_page_tables(m, one, writer="prefill")
    Gemma4Generator._install_per_layer_page_tables(m, many, writer="decode")
    Gemma4Generator._install_per_layer_page_tables(
        m, one, writer="prefill"
    )  # the 1-row buffers were last written by prefill
    assert m._calls == [True, False, False], m._calls
    assert m._g4_page_table_writer_by_rows == {1: "prefill", 32: "decode"}


def test_models_without_persistent_tables_are_left_alone():
    m = SimpleNamespace()
    Gemma4Generator._install_per_layer_page_tables(m, [torch.zeros(1, 8, dtype=torch.int32)], writer="prefill")
    assert not hasattr(m, "_g4_page_table_writer_by_rows")


# --- the host side of one install: pad / compare / snapshot once per distinct table (layers of a kv-cache group
# share the host tensor), one H2D per device buffer; a one-row readback and the batch pad keep tables shared ---


def _stub_model(n_layers, rows, width):
    from models.demos.gemma4.tt.model import Gemma4Model

    pads = []

    def pad(pt_host, target_b, target_w):
        pads.append(id(pt_host))
        return Gemma4Model._pad_page_table_host_to_shape(pt_host, target_b, target_w)

    persistent = [SimpleNamespace(shape=(rows, width), idx=i) for i in range(n_layers)]
    stub = SimpleNamespace(
        _host_page_tables_batch=Gemma4Model._host_page_tables_batch,
        _page_tables_to_ttnn=lambda tables: persistent,
        _pad_page_table_host_to_shape=pad,
        _replicate_to_mesh_mapper=lambda: None,
        mesh_config=None,
        _pads=pads,
    )
    return stub


def test_install_pads_and_compares_each_distinct_table_once(monkeypatch):
    import ttnn
    from models.demos.gemma4.tt.model import Gemma4Model

    n_layers, rows, width = 30, 32, 64
    stub = _stub_model(n_layers, rows, width)
    copies, from_torch_calls, equal_calls = [], [], []
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: copies.append(dev.idx))
    real_from_torch = ttnn.from_torch
    monkeypatch.setattr(ttnn, "from_torch", lambda *a, **k: (from_torch_calls.append(1), real_from_torch(*a, **k))[1])
    real_equal = torch.equal
    monkeypatch.setattr(torch, "equal", lambda a, b: (equal_calls.append(1), real_equal(a, b))[1])
    full = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    sliding = torch.arange(11, 19, dtype=torch.int32).reshape(1, 8)
    tables = [full if i % 2 == 0 else sliding for i in range(n_layers)]  # two groups, one host tensor each
    Gemma4Model.update_persistent_per_layer_page_tables(stub, tables, force=True)
    assert len(stub._pads) == 2, len(stub._pads)  # not once per layer
    assert len(from_torch_calls) == 2
    assert len(copies) == n_layers and sorted(copies) == list(range(n_layers))  # still one H2D per device buffer
    # the same tables again without force: compared once per distinct table, nothing uploaded
    stub._pads.clear()
    copies.clear()
    equal_calls.clear()
    Gemma4Model.update_persistent_per_layer_page_tables(stub, tables, force=False)
    assert len(stub._pads) == 2 and len(equal_calls) == 2 and copies == [], (len(stub._pads), len(equal_calls), copies)
    # a changed sliding table re-uploads exactly the sliding layers
    tables2 = [full if i % 2 == 0 else sliding + 1 for i in range(n_layers)]
    copies.clear()
    Gemma4Model.update_persistent_per_layer_page_tables(stub, tables2, force=False)
    assert sorted(copies) == [i for i in range(n_layers) if i % 2 == 1], copies


def test_batch_pad_keeps_group_tables_shared():
    pytest.importorskip("vllm")  # generator_vllm imports vLLM; the unit jobs run without it
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    stub = SimpleNamespace(model_args=[SimpleNamespace(max_batch_size=32)])
    full = torch.ones(1, 8, dtype=torch.int32)
    sliding = torch.full((1, 8), 2, dtype=torch.int32)
    out = Gemma4ForCausalLM._pad_page_tables_batch_to_max(stub, [full, sliding, full, sliding, None])
    assert out[0] is out[2] and out[1] is out[3] and out[4] is None
    assert tuple(out[0].shape) == (32, 8) and int(out[0][0, 0]) == 1 and int(out[0][1, 0]) == 0
    assert int(out[1][0, 0]) == 2


def test_sequential_row_activation_slices_each_group_table_once():
    from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin as Mixin

    installs = []
    m = SimpleNamespace(
        update_persistent_per_layer_page_tables=lambda tables, force=False: installs.append(tables),
        _active_page_tables_per_layer=None,
    )
    full = torch.arange(64, dtype=torch.int32).reshape(4, 16)
    sliding = torch.arange(100, 164, dtype=torch.int32).reshape(4, 16)
    m._active_page_tables_per_layer = [full, sliding, full, sliding]
    me = SimpleNamespace(
        model=[m],
        _page_table_row_matches=Mixin._page_table_row_matches,
        _install_per_layer_page_tables=Mixin._install_per_layer_page_tables,
    )
    Mixin._activate_sequential_per_layer_row(me, full[0:1])  # the row cursor starts at row 0
    Mixin._activate_sequential_per_layer_row(me, full[1:2])  # the next user: row 1
    sliced = m._active_page_tables_per_layer
    assert sliced[0] is sliced[2] and sliced[1] is sliced[3]
    assert torch.equal(sliced[0], full[1:2]) and torch.equal(sliced[1], sliding[1:2])
    assert len(installs) == 2 and installs[-1] is sliced
    assert m._sequential_row_cursor == 2


def test_process_output_prefill_takes_a_one_row_host_tensor():
    import ttnn
    from models.demos.gemma4.tt.model import Gemma4Model

    vocab = 48
    row = torch.arange(64, dtype=torch.bfloat16).reshape(1, 1, 1, 64)
    host = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    stub = SimpleNamespace(vocab_size=vocab, _lane_sharded=False, mesh_config=None, _g4_pf_add=lambda k, s: None)
    out = Gemma4Model.process_output_prefill(stub, host, last_token_idx=17)  # the row index is already applied
    assert tuple(out.shape) == (1, 1, vocab)
    assert torch.equal(out.reshape(-1).float(), torch.arange(vocab, dtype=torch.float32))


def test_prefill_phase_line_sums_the_models():
    pytest.importorskip("vllm")
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    stub = SimpleNamespace(
        model=[
            SimpleNamespace(
                _g4_pf_ms={"install": 1.5, "install_n": 2.0, "trace": 50.0, "lm_head": 0.5, "readback": 0.25}
            ),
            SimpleNamespace(_g4_pf_ms={"install": 0.5, "install_n": 1.0}),
        ]
    )
    line = Gemma4ForCausalLM._g4_prefill_phases(stub, 10.0, 10.012, 0.75)
    assert line == "prelude=12.0 install=2.0x3 trace=50.0 lm_head=0.5 readback=0.2 clear=0.8 ms", line


def test_prefill_row_record_is_off_for_the_batched_form():
    from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin as Mixin

    assert Mixin._g4_prefill_row_record(127, 0) == (127, 31)
    assert Mixin._g4_prefill_row_record(200, 64) == (200, 8)  # a cached continuation: the row is relative
    assert Mixin._g4_prefill_row_record(torch.tensor(33), None) == (33, 1)
    assert Mixin._g4_prefill_row_record([127, 255], 0) is None  # batched warmup / batched prefill: per-slot list
    assert Mixin._g4_prefill_row_record((127,), 0) is None
    assert Mixin._g4_prefill_row_record(None, 0) is None
    assert Mixin._g4_prefill_row_record("x", 0) is None


# ===== merged from tests/unit/test_persistent_page_table_sharing.py: Layers handed the same host page table share one persistent device buffer,
class _Dev:
    _n = 0

    def __init__(self, shape):
        self.shape = shape
        _Dev._n += 1
        self.uid = _Dev._n


def _model_persistent_page_table_sharing(monkeypatch, copies):
    monkeypatch.setenv("GEMMA4_SHARE_PAGE_TABLES", "1")  # sharing is opt-in
    m = Gemma4Model.__new__(Gemma4Model)
    m.mesh_config = None
    monkeypatch.setattr(m, "_page_table_torch_to_ttnn", lambda pt, layer_idx=None: _Dev(list(pt.shape)), raising=False)
    monkeypatch.setattr(m, "_replicate_to_mesh_mapper", lambda: None, raising=False)
    monkeypatch.setattr(model_module.ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(model_module.ttnn, "copy_host_to_device_tensor", lambda host, dev: copies.append(dev.uid))
    return m


def _tables(seed):
    sliding = torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8) + seed
    full = torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8) + 100 + seed
    return [sliding, sliding, sliding, full, sliding, full]


def test_layers_with_one_host_table_share_one_device_buffer(monkeypatch):
    copies = []
    m = _model_persistent_page_table_sharing(monkeypatch, copies)
    persistent = m._page_tables_to_ttnn(_tables(0))
    assert len(persistent) == 6
    assert persistent[0] is persistent[1] is persistent[2] is persistent[4]
    assert persistent[3] is persistent[5] and persistent[3] is not persistent[0]


def test_update_writes_each_shared_buffer_once_and_skips_unchanged_tables(monkeypatch):
    copies = []
    m = _model_persistent_page_table_sharing(monkeypatch, copies)
    m.update_persistent_per_layer_page_tables(_tables(0))
    assert sorted(copies) == sorted({d.uid for d in m._persistent_per_layer_page_tables}) and len(copies) == 2
    copies.clear()
    m.update_persistent_per_layer_page_tables(_tables(0))
    assert copies == []
    m.update_persistent_per_layer_page_tables(_tables(7))
    assert len(copies) == 2


def test_layers_that_stop_sharing_content_get_their_own_buffers(monkeypatch):
    copies = []
    m = _model_persistent_page_table_sharing(monkeypatch, copies)
    m.update_persistent_per_layer_page_tables(_tables(0))
    tables = _tables(3)
    tables[1] = tables[1].clone() + 1000  # layer 1 diverges from the sliding group
    copies.clear()
    m.update_persistent_per_layer_page_tables(tables)
    persistent = m._persistent_per_layer_page_tables
    assert persistent[0] is not persistent[1]
    assert m._invalidate_decode_traces_after_page_table_realloc is True
    # Rebuilt unshared: every layer owns a buffer and each was written (the one
    # copy made before the divergence was noticed lands on a replaced buffer).
    assert len({id(d) for d in persistent}) == 6
    assert len(copies) >= 6 and set(d.uid for d in persistent) <= set(copies)


def test_lane_parallel_prefill_table_shards_to_one_row_per_column():
    # A lanes-row table is the lane-parallel prefill layout: one row per column, like a
    # single-row table; it must not be sized as the lane_slots-row frame (that grew the
    # persistent buffers and dropped the decode traces after every lane-router prefill).
    rows = model_module.lane_page_table_device_rows
    assert rows(1, 4, 32) == 1
    assert rows(4, 4, 32) == 1
    assert rows(8, 4, 32) == 32
    assert rows(128, 4, 32) == 32
    assert rows(4, 4, 1) == 1  # single-slot lanes: the frame is lanes rows


def test_lane_parallel_prefill_table_does_not_regrow_its_buffer(monkeypatch):
    from types import SimpleNamespace

    copies = []
    m = _model_persistent_page_table_sharing(monkeypatch, copies)
    m.mesh_config = SimpleNamespace(lane_sharded=True, lanes=4, lane_shard_mapper=lambda dev, dim: None)
    m.lane_slots = 32
    m.mesh_device = None
    m._invalidate_decode_traces_after_page_table_realloc = False
    lane_parallel = [torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8)] * 2  # one row per lane
    first = m._page_tables_to_ttnn(lane_parallel)
    again = m._page_tables_to_ttnn([t + 1 for t in lane_parallel])
    assert again[0] is first[0]  # same batch key, same device rows: no reallocation
    assert m._invalidate_decode_traces_after_page_table_realloc is False
    frame = [torch.zeros((128, 8), dtype=torch.int32)] * 2  # the decode frame keys its own buffers ...
    m._page_tables_to_ttnn(frame)
    # ... but a batch key first seen after the process's first key allocates where captured replays
    # scribble, so the next decode re-captures (the first key, the warmup's own, needs nothing).
    assert m._invalidate_decode_traces_after_page_table_realloc is True
    assert m._g4_page_table_realloc_reason.startswith("new batch key")


# ===== merged from tests/unit/test_decode_page_table_refresh.py: Decode must re-upload a per-layer page-table bucket after prefill wrote the same persistent buffers,
class _Model:
    def __init__(self, tables):
        self._active_page_tables_per_layer = tables
        self.staged = []

    def update_persistent_per_layer_page_tables(self, tables, force=False):
        self.staged.append([t.clone() if isinstance(t, torch.Tensor) else t for t in tables])


class _Generator(ChunkedPrefillPageTableGuardMixin):
    def __init__(self, model):
        self.model = [model]


def _tables_decode_page_table_refresh(rows):
    sliding = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4) + 1
    return [sliding, sliding + 100, sliding]


def test_prefill_write_forces_decode_reupload_of_the_same_bucket():
    model = _Model(_tables_decode_page_table_refresh(32))
    gen = _Generator(model)
    decode_rows = _tables_decode_page_table_refresh(1)
    gen._install_per_layer_page_tables(model, decode_rows, writer="decode")
    assert not gen._page_tables_written_by_prefill([model], [decode_rows])
    # A sequential prefill user slices the batch to one row: same bucket key.
    gen._activate_sequential_per_layer_row(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    assert gen._page_tables_written_by_prefill([model], [decode_rows])
    gen._install_per_layer_page_tables(model, decode_rows, writer="decode")
    assert not gen._page_tables_written_by_prefill([model], [decode_rows])


def test_other_buckets_are_not_marked():
    model = _Model(_tables_decode_page_table_refresh(32))
    gen = _Generator(model)
    gen._install_per_layer_page_tables(model, _tables_decode_page_table_refresh(4), writer="decode")
    gen._activate_sequential_per_layer_row(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    # 1-row buffers were written by prefill; the 4-row bucket was not.
    assert gen._page_tables_written_by_prefill([model], [_tables_decode_page_table_refresh(1)])
    assert not gen._page_tables_written_by_prefill([model], [_tables_decode_page_table_refresh(4)])


def test_clearing_the_sequential_batch_marks_the_full_batch_rows():
    model = _Model(_tables_decode_page_table_refresh(32))
    gen = _Generator(model)
    gen._install_per_layer_page_tables(model, _tables_decode_page_table_refresh(32), writer="decode")
    gen._activate_sequential_per_layer_row(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    gen._clear_sequential_batch_page_tables()
    assert gen._page_tables_written_by_prefill([model], [_tables_decode_page_table_refresh(32)])
    assert len(model.staged) == 3


def test_rows_key_matches_the_model_rule():
    rows = ChunkedPrefillPageTableGuardMixin._page_tables_rows
    assert rows([None, torch.zeros(5, 3, dtype=torch.int32)]) == 5
    assert rows([torch.zeros(7, dtype=torch.int32)]) == 1
    assert rows([None, "device-tensor-stand-in"]) is None
    assert rows(None) is None


# ===== merged from tests/unit/test_decode_page_table_rows.py: Decode per-layer page tables are sliced to the step's bucket rows: the plugin pads them to
try:
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


if _HAS_VLLM:
    _slice = Gemma4ForCausalLM._slice_page_tables_rows


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_slice_keeps_bucket_rows_and_shared_identity():
    sliding = torch.arange(32 * 8, dtype=torch.int32).reshape(32, 8)
    full = sliding + 1000
    tables = [sliding, sliding, full, sliding]
    out = _slice(tables, 4)
    assert all(tuple(t.shape) == (4, 8) for t in out)
    assert torch.equal(out[0], sliding[:4]) and torch.equal(out[2], full[:4])
    assert out[0] is out[1] is out[3] and out[2] is not out[0]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_slice_is_noop_when_tables_fit_or_rows_unknown():
    t = torch.zeros(4, 8, dtype=torch.int32)
    assert _slice([t], 4)[0] is t
    assert _slice([t], 32)[0] is t
    assert _slice([t], None)[0] is t
    assert _slice(None, 4) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_slice_passes_non_tensor_entries_through():
    t = torch.zeros(32, 8, dtype=torch.int32)
    out = _slice([None, t, "device-tensor-stand-in"], 2)
    assert out[0] is None and out[2] == "device-tensor-stand-in" and tuple(out[1].shape) == (2, 8)


# ===== merged from tests/unit/test_prefill_page_table_mask.py: Prefill zeroes page-table columns past each request's allocation: vLLM leaves stale block ids there,
def _table(rows, cols, base=1):
    return (torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols) + base).clone()


def test_columns_past_allocation_are_zeroed_per_row():
    pt = _table(2, 20)
    orig = pt.clone()
    (out,), masked = mask_page_table_columns_past_allocation([pt], prompt_lens=[100, 1000], block_sizes=[64])
    # 100 tokens -> 2 blocks; 1000 tokens -> 16 blocks.
    assert torch.equal(out[0, :2], orig[0, :2]) and int(out[0, 2:].abs().sum()) == 0
    assert torch.equal(out[1, :16], orig[1, :16]) and int(out[1, 16:].abs().sum()) == 0
    assert masked == 18 + 4
    # The plugin's tensor is left untouched.
    assert torch.equal(pt, orig)


def test_block_size_is_per_layer():
    pt_a = _table(1, 12)
    pt_b = _table(1, 12, base=100)
    (a, b), _ = mask_page_table_columns_past_allocation([pt_a, pt_b], prompt_lens=[256], block_sizes=[64, 128])
    assert int((a[0] != 0).sum()) == 4  # 256 / 64
    assert int((b[0] != 0).sum()) == 2  # 256 / 128


def test_shared_table_object_stays_shared_and_one_dim_rows_work():
    shared = _table(1, 8)[0]  # 1-D row, as the kv-share alias passes it
    (a, b), masked = mask_page_table_columns_past_allocation([shared, shared], prompt_lens=[65], block_sizes=[64, 64])
    assert a is b
    assert a.dim() == 1 and int((a != 0).sum()) == 2
    assert masked == 6  # counted once for the shared object


def test_rows_without_a_prompt_length_and_clean_tables_pass_through():
    pt = _table(3, 6)
    pt[0, 2:] = 0  # already clean
    (out,), masked = mask_page_table_columns_past_allocation([pt], prompt_lens=[100, 100], block_sizes=[64])
    assert torch.equal(out[2], pt[2])  # no prompt length -> untouched
    assert int(out[1, 2:].abs().sum()) == 0 and masked == 4
    # Nothing to mask -> the very same object comes back.
    clean = pt.clone()
    clean[:, 2:] = 0
    (same,), masked = mask_page_table_columns_past_allocation([clean], prompt_lens=[100, 100, 100], block_sizes=[64])
    assert same is clean and masked == 0


def test_non_tensor_entries_and_missing_inputs_are_left_alone():
    pt = _table(1, 4)
    out, masked = mask_page_table_columns_past_allocation(
        [None, "device-tensor", pt], prompt_lens=[1], block_sizes=[64, 64, 64]
    )
    assert out[0] is None and out[1] == "device-tensor" and int((out[2] != 0).sum()) == 1 and masked == 3
    assert mask_page_table_columns_past_allocation([pt], prompt_lens=None, block_sizes=[64]) == ([pt], 0)
    assert mask_page_table_columns_past_allocation(None, prompt_lens=[1], block_sizes=[64]) == (None, 0)


def test_block_sizes_resolve_from_the_plugins_nested_kv_cache_list():
    """The plugin passes ``kv_caches`` as [submesh][layer][k, v]; both nestings must resolve block sizes."""
    pytest.importorskip("vllm")
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    class _Cache:
        shape = [100, 2, 64, 256]

    cfg = SimpleNamespace(head_dim=256, num_key_value_heads=2)  # two heads per chip, as the cache holds
    layers = [SimpleNamespace(self_attn=SimpleNamespace(config=cfg, mesh_config=None, weights=None)) for _ in range(3)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    per_layer = [[_Cache(), _Cache()] for _ in layers]
    assert Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, per_layer) == [64, 64, 64]
    assert Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, [per_layer]) == [64, 64, 64]


_LAYERS = ((8, 256), (2, 512))


def _lane_views(mesh_shape, block_size, cache_tp):
    """Per layer (cache shape, head_dim, attention kv heads per chip) when the cache is allocated at ``cache_tp``."""
    attn_tp = mesh_shape[0]
    out = []
    for kv, hd in _LAYERS:
        attn = SimpleNamespace(
            mesh_config=SimpleNamespace(tp=attn_tp), weights=SimpleNamespace(kv_replicated=kv < attn_tp)
        )
        cfg = SimpleNamespace(head_dim=hd, num_key_value_heads=kv)
        cache_heads = 1 if kv < cache_tp else kv // cache_tp
        out.append(([256, cache_heads, block_size, hd], hd, layer_local_kv_heads(attn, cfg)))
    return out


@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("mesh_shape", [(2, 4), (8, 4)], ids=["2x4", "8x4"])
def test_lane_engine_cache_from_its_tp_keeps_declared_block_size(block_size, mesh_shape):
    """Caches allocated with the attention's own TP (init_kv_cache from mesh_config.tp) are addressed at the
    declared block size on every layer, so a 128-token prompt needs exactly 128 / block_size columns."""
    sizes = [
        paged_block_size_for_view(shape, hd, kv_local)
        for shape, hd, kv_local in _lane_views(mesh_shape, block_size, mesh_shape[0])
    ]
    assert sizes == [block_size, block_size]
    cols = num_blocks_in_seq(128, min(sizes))
    assert cols == 128 // block_size
    pt = _table(1, 8)
    (out,), _ = mask_page_table_columns_past_allocation([pt], prompt_lens=[128], block_sizes=[min(sizes)])
    assert int((out[0] != 0).sum()) == cols


@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("mesh_shape, sliding_factor", [((2, 4), 0.5), ((8, 4), 2.0)], ids=["2x4_half", "8x4_double"])
def test_lane_engine_cache_from_the_mesh_columns_shifts_the_effective_block_size(
    block_size, mesh_shape, sliding_factor
):
    """The former allocation (heads from the 4 mesh columns) holds 2 sliding heads per chip on both shapes while the
    attention writes 4 on the (2,4) and 1 on the (8,4): the sliding layers' effective block size is half the declared
    one on the block (a 128-token prompt overran its 2 columns) and double it on the Galaxy (benign: wider is fine).
    The page-table math must follow the effective size on every layer."""
    views = _lane_views(mesh_shape, block_size, cache_tp=mesh_shape[1])
    sliding, full = (paged_block_size_for_view(shape, hd, kv_local) for shape, hd, kv_local in views)
    assert sliding == int(block_size * sliding_factor)
    assert full == block_size  # 2 full heads over 4 columns and over the rows both give one head per chip
    eff = min(sliding, full)
    cols = num_blocks_in_seq(128, eff)
    assert cols == -(-128 // eff)
    if sliding_factor < 1:
        assert cols == 2 * (128 // block_size), "the block needs twice the columns the declared size suggests"
    pt = _table(1, 16)
    (out,), _ = mask_page_table_columns_past_allocation([pt], prompt_lens=[128], block_sizes=[eff])
    assert int((out[0] != 0).sum()) == cols


def test_paged_block_size_for_view_rejects_an_unaddressable_cache(expect_error):
    with expect_error(ValueError, "not addressable"):
        paged_block_size_for_view([4, 1, 64, 256], head_dim=512, num_local_kv_heads=3)


# ===== merged from tests/unit/test_sequential_row_activation.py: Sequential prefill picks each request's per-layer page-table row by batch position, not by matching
class _Model_sequential_row_activation:
    def __init__(self, tables):
        self._active_page_tables_per_layer = tables
        self.staged = []

    def update_persistent_per_layer_page_tables(self, tables, force=False):
        self.staged.append([t.clone() for t in tables])


class _Generator_sequential_row_activation(ChunkedPrefillPageTableGuardMixin):
    def __init__(self, model):
        self.model = [model]


def _tables_sequential_row_activation():
    sliding = torch.tensor([[7, 8, 0, 0], [7, 8, 0, 0]], dtype=torch.int32)
    full = torch.tensor([[101, 102, 0, 0], [201, 202, 0, 0]], dtype=torch.int32)
    return [sliding, full, sliding]


def _slice_sequential_row_activation():
    return torch.tensor([[7, 8, 0, 0]], dtype=torch.int32)


def test_second_request_gets_its_own_full_attention_row():
    model = _Model_sequential_row_activation(_tables_sequential_row_activation())
    gen = _Generator_sequential_row_activation(model)
    gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())
    assert model._active_page_tables_per_layer[1].tolist() == [[101, 102, 0, 0]]
    gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())
    assert model._active_page_tables_per_layer[1].tolist() == [[201, 202, 0, 0]]
    assert [s[1].tolist() for s in model.staged] == [[[101, 102, 0, 0]], [[201, 202, 0, 0]]]
    gen._clear_sequential_batch_page_tables()
    assert int(model._active_page_tables_per_layer[1].shape[0]) == 2
    assert model._sequential_row_cursor == 0


def test_nested_activation_within_one_request_is_a_noop():
    model = _Model_sequential_row_activation(_tables_sequential_row_activation())
    gen = _Generator_sequential_row_activation(model)
    gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())
    model._sequential_row_active = True  # the traced entry holds this across its capture/replay
    gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())
    assert model._active_page_tables_per_layer[1].tolist() == [[101, 102, 0, 0]]
    assert model._sequential_row_cursor == 1


def test_more_requests_than_rows_raises(expect_error):
    model = _Model_sequential_row_activation(_tables_sequential_row_activation())
    gen = _Generator_sequential_row_activation(model)
    gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())
    gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())
    with expect_error(ValueError, "row 2"):
        gen._activate_sequential_per_layer_row(_slice_sequential_row_activation())


def test_bind_and_release_walk_every_layer():
    from types import SimpleNamespace

    class _Attn:
        def __init__(self):
            self.config = SimpleNamespace()
            self.released = []

        def _release_sliding_prefill_tail(self, req_key=None, clear_persistent=False):
            self.released.append(req_key)

    model = _Model_sequential_row_activation(_tables_sequential_row_activation())
    model.layers = [SimpleNamespace(self_attn=_Attn()) for _ in range(3)]
    gen = _Generator_sequential_row_activation(model)
    gen._bind_sliding_tail_key(77)
    assert [layer.self_attn.config._g4_active_req_key for layer in model.layers] == [77, 77, 77]
    gen._release_all_sliding_prefill_tails(req_key=77)
    assert [layer.self_attn.released for layer in model.layers] == [[77], [77], [77]]


def test_legacy_slice_must_match_the_positional_row(expect_error):
    model = _Model_sequential_row_activation(_tables_sequential_row_activation())
    gen = _Generator_sequential_row_activation(model)
    with expect_error(ValueError, "does not match"):
        gen._activate_sequential_per_layer_row(torch.tensor([[9, 9, 0, 0]], dtype=torch.int32))


# ===== merged from tests/unit/test_prefill_over_user_cap.py: When batch exceeds the batched-prefill user cap, metal must not remap slots.
def plan_metal_prefill_strategy(
    batch_size: int,
    prefill_seq_len: int = 128,
    actual_lens=None,
) -> str:
    """Mirror Gemma4Generator.prefill_forward_text batching decision (host-only).

    Hetero actual lengths in one pad bucket are still batched (KV fill capped
    per-slot via ``valid_seq_lens``). B>user_cap uses remapped microbatch chunks.
    """
    del actual_lens  # no longer gates batching once pad-fill is capped per slot
    if batch_size <= 1:
        return "single"
    padded_batch = batch_size
    for b in (1, 2, 4, 8, 16, 32):
        if b >= batch_size:
            padded_batch = b
            break
    max_users = resolve_batched_prefill_chunk_users(padded_batch, prefill_seq_len)
    if batch_size > max_users:
        return "microbatch_remapped_slots"
    return "true_batched"


def test_user_cap_default_is_arch_gated():
    """Blackhole raises the cap (the B>=8 all_gather wedge no longer reproduces
    there and the cap costs ~4.6x TTFT at batch-32); Wormhole keeps 4."""
    assert max_batched_prefill_users() in (
        _DEFAULT_MAX_BATCHED_PREFILL_USERS,
        _BLACKHOLE_MAX_BATCHED_PREFILL_USERS,
    )


def test_batch_gt_user_cap_uses_microbatch(monkeypatch):
    monkeypatch.setenv("GEMMA4_MAX_BATCHED_PREFILL_USERS", "4")
    assert plan_metal_prefill_strategy(8) == "microbatch_remapped_slots"
    assert plan_metal_prefill_strategy(32) == "microbatch_remapped_slots"


def test_hetero_actual_lens_same_pad_bucket_still_batches(monkeypatch):
    monkeypatch.setenv("GEMMA4_MAX_BATCHED_PREFILL_USERS", "4")
    # Pad-fill fix: hetero actual + same pad bucket stays on batched/microbatch path.
    lens = [87, 82, 80, 66, 100, 91, 83, 78]
    assert plan_metal_prefill_strategy(8, prefill_seq_len=128, actual_lens=lens) == "microbatch_remapped_slots"
    assert plan_metal_prefill_strategy(4, prefill_seq_len=128, actual_lens=lens[:4]) == "true_batched"


def test_batch_le_user_cap_uses_true_batched():
    assert plan_metal_prefill_strategy(1) == "single"
    assert plan_metal_prefill_strategy(2) == "true_batched"
    assert plan_metal_prefill_strategy(4) == "true_batched"


# ===== merged from tests/unit/test_batched_prefill_actual_lens.py: Batched prefill gates on identical *padded* buckets (hetero actual OK).
def can_batch_prefill(prompt_lens, num_cached=None, page_table_ok=True, disable=False):
    """Mirror Gemma4 can_batch_prefill gating after valid_seq_lens pad-fill fix."""
    batch_size = len(prompt_lens)
    num_cached = num_cached or [0] * batch_size
    prefill_seq_lens = [get_padded_prefill_len(int(seq_len) - int(n)) for seq_len, n in zip(prompt_lens, num_cached)]
    return (
        page_table_ok
        and batch_size > 1
        and len(set(prefill_seq_lens)) == 1
        and not disable
        and all(n == 0 for n in num_cached)
    )


def test_same_padded_bucket_different_actual_lens_allows_batch():
    # GPQA-scale prompts all pad to 1024 but differ in real length — batch OK
    # once per-slot valid_seq_lens caps KV fill (attention/prefill.py).
    lens = [154, 159, 184, 165, 380, 231, 191, 478]
    assert len({get_padded_prefill_len(n) for n in lens}) == 1
    assert can_batch_prefill(lens) is True


def test_identical_actual_lens_allows_batch():
    lens = [200, 200, 200, 200]
    assert can_batch_prefill(lens) is True


def test_single_user_never_batches():
    assert can_batch_prefill([200]) is False


def test_different_padded_buckets_disables_batch():
    # 200 → 1024 bucket vs 2000 → 2048 bucket
    assert get_padded_prefill_len(200) != get_padded_prefill_len(2000)
    assert can_batch_prefill([200, 2000]) is False


# ===== merged from tests/unit/test_batched_prefill_valid_seq_lens.py: Per-slot valid_seq_lens for batched prefill KV fill (host-only).
TILE_HEIGHT = 32


def fill_len_for_slot(valid_seq_len, slot_idx, seq_len_per_user, page_len):
    """Mirror attention/prefill.py batched fill length selection."""
    fill_len = seq_len_per_user
    if isinstance(valid_seq_len, (list, tuple)):
        if 0 <= int(slot_idx) < len(valid_seq_len) and valid_seq_len[int(slot_idx)] is not None:
            fill_len = min(fill_len, max(0, int(valid_seq_len[int(slot_idx)])))
    elif valid_seq_len is not None:
        fill_len = min(fill_len, max(0, int(valid_seq_len)))
    fill_len = min(fill_len, page_len)
    tile_end = ((fill_len + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = min(tile_end, seq_len_per_user, page_len) if fill_len > 0 else 0
    return fill_len, tile_end


def test_hetero_actual_lens_tile_ceil_per_slot():
    # batch-32 demo: pad bucket 128, actual lens differ
    last_token_idx = [86, 81, 65, 99]  # actual-1
    valid = [i + 1 for i in last_token_idx]
    page_len = 1024
    seq = 128
    for slot, actual in enumerate(valid):
        fill, tile = fill_len_for_slot(valid, slot, seq, page_len)
        assert fill == actual
        assert tile == ((actual + 31) // 32) * 32
        assert tile <= seq


def test_missing_valid_seq_len_fills_full_pad_bucket():
    fill, tile = fill_len_for_slot(None, 0, 128, 1024)
    assert fill == 128
    assert tile == 128


def test_plan_allows_hetero_actual_with_same_pad():
    pass  # plan_metal_prefill_strategy is defined in this module now

    # After pad-fill fix, hetero actual + same pad should true-batch / microbatch,
    # not force sequential-only. B=4 → true_batched; B=32 → sequential_global
    # was the *old* plan — update expectation to microbatch/chunk path.
    assert plan_metal_prefill_strategy(4, 128, actual_lens=[80, 90, 70, 100]) == "true_batched"


# ===== merged from tests/unit/test_prefill_row_chunks_host.py: Host-only check of the prefill matmul row chunking (tt/dram_sharded.py prefill_row_chunks): full slabs first, then
def test_full_slabs_then_small_remainders():
    assert prefill_row_chunks(4096, slab=2560, remainder=1024) == [(0, 2560), (2560, 3584), (3584, 4096)]
    assert prefill_row_chunks(8192, slab=2560, remainder=1024) == [(0, 2560), (2560, 5120), (5120, 7680), (7680, 8192)]
    assert prefill_row_chunks(3072, slab=2560, remainder=1024) == [(0, 2560), (2560, 3072)]
    assert prefill_row_chunks(2561, slab=2560, remainder=1024) == [(0, 2560), (2560, 2561)]


def test_chunks_cover_m_exactly_and_never_exceed_the_bounds():
    for m in range(32, 40000, 992):
        chunks = prefill_row_chunks(m, slab=2560, remainder=1024)
        assert chunks[0][0] == 0 and chunks[-1][1] == m
        assert all(a < b for a, b in chunks)
        assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
        assert all(b - a == 2560 or b - a <= 1024 for a, b in chunks)
        assert all(b - a == 2560 for a, b in chunks[: m // 2560])


def test_defaults_are_the_blackhole_slab_and_the_pre_slab_chunk():
    from models.demos.gemma4.tt import dram_sharded as ds

    assert ds._PREFILL_REMAINDER_CHUNK <= 1024 and ds._PREFILL_REMAINDER_CHUNK <= ds._PREFILL_CUTOFF
    assert prefill_row_chunks(4096)[0] == (0, ds._PREFILL_M_CHUNK) if ds._PREFILL_M_CHUNK < 4096 else True


def test_slab_rows_follow_the_l1_headroom():
    import ttnn
    from models.demos.gemma4.tt.dram_sharded import prefill_cb_bytes, prefill_slab_rows

    # the two CI numbers the footprint model is calibrated on (Blackhole, 2D matmul, k-block 4)
    assert (
        prefill_cb_bytes(8, 48, 4, 2048) == 1815936
    )  # 31B gate_up with bf16 weights: "beyond max L1" in the unit test
    assert prefill_cb_bytes(8, 40, 4, 2048) == 1553792  # 12B gate_up bf16: the CB region end that clashed at 1539200
    bh = dict(grid=(8, 10), l1_bytes=1572864, headroom=128 * 1024, max_cols=8)
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat8_b, **bh) == 2240  # the 31B model's MLP weights
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat16, **bh) == 1280  # the same weight in the unit test
    assert prefill_slab_rows(3840, 2 * 15360 // 4, ttnn.bfloat16, **bh) == 1920  # the 12B's MLP weights
    assert prefill_slab_rows(3840, (4096 + 4096) // 4, ttnn.bfloat16, **bh) == 2560  # a narrow weight: the full slab
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat16, **dict(bh, headroom=0)) == 2560  # 0 = no cap
    h64 = dict(bh, headroom=64 * 1024)  # the default: the 31B keeps its slab, the 12B's 2K prompt stays one matmul
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat8_b, **h64) == 2560
    assert prefill_slab_rows(3840, 2 * 15360 // 4, ttnn.bfloat16, **h64) == 2240
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat16, **h64) == 1600  # the unit test's bf16 31B weight
    from models.demos.gemma4.tt import dram_sharded as ds

    assert ds._PREFILL_L1_HEADROOM == 64 * 1024 or "GEMMA4_PREFILL_L1_HEADROOM_KB" in __import__("os").environ
    # every slab's CBs stay under the budget and the next size up would not
    for k, n, dt in ((5376, 10752, ttnn.bfloat8_b), (5376, 10752, ttnn.bfloat16), (3840, 7680, ttnn.bfloat16)):
        rows = prefill_slab_rows(k, n, dt, **bh)
        assert rows % 320 == 0 and 320 <= rows <= 2560


# ===== merged from tests/unit/test_prefill_sdpa_config_host.py: Host-only checks of the prefill SDPA program config (tt/attention/operations.py): every grid derives from the
KNOBS = (
    "GEMMA4_PREFILL_SDPA_WIDE",
    "GEMMA4_PREFILL_SDPA_QCHUNK",
    "GEMMA4_PREFILL_SDPA_KCHUNK",
    "GEMMA4_PREFILL_SDPA_GRID",
    "GEMMA4_PREFILL_SDPA_EXP_APPROX",
    "GEMMA4_PREFILL_SDPA_PROGCFG",
    "GEMMA4_TEST_GRID",
)


GRIDS = ((13, 10), (12, 10), (11, 10), (10, 10), (8, 8), (7, 8))


def _grid(pc):
    return (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)


@pytest.fixture(params=[True, False], ids=["blackhole", "wormhole"])
def _sdpa_arch(request, monkeypatch):
    """The arch the prefill SDPA config reads, stubbed on both the op module and this test module (no metal context)."""
    monkeypatch.setattr(ops, "is_blackhole", lambda: request.param)
    monkeypatch.setattr(sys.modules[__name__], "is_blackhole", lambda: request.param)
    return request.param


@pytest.fixture()
def _clean(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("_sdpa_arch")
@pytest.mark.usefixtures("_clean")
@pytest.mark.parametrize("device_grid", GRIDS, ids=[f"{x}x{y}" for x, y in GRIDS])
@pytest.mark.parametrize("heads", (4, 8, 16))
@pytest.mark.parametrize("head_dim", (256, 512))
def test_every_config_fits_the_device_grid(device_grid, heads, head_dim):
    monkeypatch_env = {}
    pc = ops.prefill_sdpa_program_config(head_dim, 2048, sliding_window=None, num_heads=heads, device_grid=device_grid)
    if pc is None:  # narrow chips on Blackhole take the op defaults
        assert is_blackhole() and heads < ops.PREFILL_SDPA_WIDE_MIN_HEADS and head_dim <= 256
        return
    gx, gy = _grid(pc)
    assert gx * gy <= device_grid[0] * device_grid[1] and gx <= device_grid[0] and gy <= device_grid[1]
    if head_dim >= 512:
        assert (gx, gy) == (min(8, device_grid[0]), min(4, device_grid[1])) and pc.q_chunk_size == 128
    elif heads >= ops.PREFILL_SDPA_WIDE_MIN_HEADS and is_blackhole() and device_grid[0] * device_grid[1] > 64:
        assert (gx, gy) == (min(12, device_grid[0]), min(10, device_grid[1])) and pc.q_chunk_size == 128  # wide
    else:
        assert (gx, gy) == (min(8, device_grid[0]), min(8, device_grid[1])) and pc.q_chunk_size == 256
    assert pc.k_chunk_size == 128 and pc.exp_approx_mode is False
    assert not monkeypatch_env


@pytest.mark.usefixtures("_sdpa_arch")
@pytest.mark.usefixtures("_clean")
def test_unknown_device_grid_is_never_wide():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8)
    assert _grid(pc) == (8, 8) and pc.q_chunk_size == 256
    pc = ops.prefill_sdpa_program_config(512, 2048, sliding_window=None, num_heads=8)
    assert _grid(pc) == (8, 4) and pc.q_chunk_size == 128


@pytest.mark.usefixtures("_sdpa_arch")
@pytest.mark.usefixtures("_clean")
def test_wide_on_the_galaxy_grid_and_the_12b_defaults(monkeypatch):
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(13, 10))
    if is_blackhole():
        assert _grid(pc) == (12, 10) and pc.q_chunk_size == 128
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=4, device_grid=(13, 10))
    if is_blackhole():
        assert pc is None
        monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "1")  # opt back in to the fixed config
        pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=4, device_grid=(13, 10))
    assert _grid(pc) == (8, 8) and (pc.q_chunk_size, pc.k_chunk_size) == (256, 128)
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "0")  # the op defaults everywhere
    assert ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(13, 10)) is None


@pytest.mark.usefixtures("_sdpa_arch")
@pytest.mark.usefixtures("_clean")
def test_window_cap_and_short_sequences():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=128, num_heads=8, device_grid=(13, 10))
    assert pc.k_chunk_size == 64
    pc = ops.prefill_sdpa_program_config(256, 64, sliding_window=None, num_heads=8, device_grid=(13, 10))
    assert pc.q_chunk_size == 64 and pc.k_chunk_size == 64


@pytest.mark.usefixtures("_sdpa_arch")
@pytest.mark.usefixtures("_clean")
def test_env_overrides_win_but_stay_on_the_device(monkeypatch):
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_WIDE", "0")
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(13, 10))
    assert _grid(pc) == (8, 8) and pc.q_chunk_size == 256
    monkeypatch.delenv("GEMMA4_PREFILL_SDPA_WIDE")
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_QCHUNK", "64")
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_GRID", "12x10")
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(10, 10))
    assert pc.q_chunk_size == 64 and _grid(pc) == (10, 10)  # the env grid is clamped to the device


# ===== merged from tests/unit/test_sdpa_chunk_align.py
def test_align_num_cached_tokens_to_sdpa_rounds_down():
    assert SDPA_CHUNK_ALIGN == 128
    # Nightly 31B QB2 crash: vLLM continuation at start_pos=48.
    assert align_num_cached_tokens_to_sdpa([48]) == [0]
    assert align_num_cached_tokens_to_sdpa([0, 128, 129, 256, 47]) == [0, 128, 128, 256, 0]


# ===== merged from tests/unit/test_sdpa_decode_split_host.py: The paged SDPA decode op's core split and per-core KV stream on the QB2 column (``tt/sdpa_decode_split.py``,
BH_GRID = 13 * 10  # Blackhole worker grid the batched decode SDPA runs on


B1_GRID = 8 * 4  # decode.py: single-user global layers (head_dim 512) use an (8, 4) grid


K256 = 256000


def test_the_factory_split_matches_the_program_factory_formula():
    # max_num_cores_for_compute = max_cores_per_head_batch * B * num_kv_heads, capped by the grid, split per batch and head
    assert core_split(grid_cores=B1_GRID, batch=1, local_kv_heads=1, max_cores_per_head_batch=16) == (16, 16)
    assert core_split(grid_cores=BH_GRID, batch=8, local_kv_heads=1, max_cores_per_head_batch=16) == (16, 128)
    assert core_split(grid_cores=BH_GRID, batch=32, local_kv_heads=1, max_cores_per_head_batch=16) == (4, 128)
    assert core_split(grid_cores=BH_GRID, batch=32, local_kv_heads=2, max_cores_per_head_batch=16) == (2, 128)
    # more heads than cores per batch: heads fold onto one core
    assert core_split(grid_cores=BH_GRID, batch=32, local_kv_heads=4, max_cores_per_head_batch=16) == (1, 128)


def test_the_readers_queue_depth_collapses_as_the_grid_fills():
    # reader_decode_all: barrier every ((512 / num_active_cores) * 1152) / q_tile_bytes reads
    assert reads_in_flight(16) == 18
    assert reads_in_flight(32) == 9
    assert reads_in_flight(64) == 4
    assert reads_in_flight(128) == 2
    assert reads_in_flight(130) == 1


def test_a_lone_256k_user_streams_its_global_layers_on_16_cores():
    p = global_layer_plan(
        grid_cores=B1_GRID, batch=1, local_kv_heads=1, max_cores_per_head_batch=16, context=K256, head_dim=512
    )
    assert p["cores_per_head"] == 16 and p["active_cores"] == 16 and p["reads_in_flight"] == 18
    # 8000 tile rows x 16 tile columns x (K, V) of bfloat8_b tiles over 16 cores = 17.4 MB and 16,000 one-tile reads per core
    assert p["bytes_per_core"] == pytest.approx(8000 * 16 * 2 * BFP8_TILE_BYTES / 16)
    assert p["reads_per_core"] == 16000
    assert p["layer_bytes"] == pytest.approx(0.2785e9, rel=1e-3)


def test_the_26b_floor_fills_the_grid_with_two_reads_in_flight_per_core():
    # 8 users x 256K (devq 0560): every user's 16-core group is live, 128 of 130 cores, and the read queue is 2 deep
    p = global_layer_plan(
        grid_cores=BH_GRID, batch=8, local_kv_heads=1, max_cores_per_head_batch=16, context=K256, head_dim=512
    )
    assert (p["cores_per_head"], p["active_cores"], p["reads_in_flight"]) == (16, 128, 2)
    assert p["layer_bytes"] == pytest.approx(8 * 0.2785e9, rel=1e-3)
    # 32 users x 64K move the same bytes with 4 cores per user
    q = global_layer_plan(
        grid_cores=BH_GRID, batch=32, local_kv_heads=1, max_cores_per_head_batch=16, context=65536, head_dim=512
    )
    assert (q["cores_per_head"], q["active_cores"], q["reads_in_flight"]) == (4, 128, 2)
    assert q["layer_bytes"] == pytest.approx(p["layer_bytes"], rel=0.03)


def test_the_knob_that_keeps_a_read_queue_at_the_floor():
    # GEMMA4_SDPA_MAX_CORES_PER_HEAD: the largest cap (power of two <= 64) that keeps >= N tile reads queued per core
    assert max_cores_for_queue_depth(grid_cores=BH_GRID, batch=1, local_kv_heads=1, min_reads_in_flight=8) == 32
    assert max_cores_for_queue_depth(grid_cores=BH_GRID, batch=1, local_kv_heads=1, min_reads_in_flight=4) == 64
    assert max_cores_for_queue_depth(grid_cores=BH_GRID, batch=8, local_kv_heads=1, min_reads_in_flight=8) == 4
    assert max_cores_for_queue_depth(grid_cores=BH_GRID, batch=8, local_kv_heads=1, min_reads_in_flight=4) == 8
    # halving the cores per user at the floor doubles the per-core stream: the knob trades parallel issuers for queue depth,
    # which is why the lever that does not trade is the reader's granularity (one NoC read per 1 KB tile today)
    assert kv_bytes_per_core_per_layer(context=K256, head_dim=512, cores_per_head=8) == pytest.approx(
        2 * kv_bytes_per_core_per_layer(context=K256, head_dim=512, cores_per_head=16)
    )


# ===== merged from tests/unit/test_decode_inactive_positions.py: Host-side decode position pad/sentinel handling (no device).
def prepare_rope_and_cache_positions(pos_flat: torch.Tensor, pad_to: int = 32):
    """Mirror Gemma4 prepare_decode_inputs_host position logic."""
    batch = pos_flat.numel()
    pos_i64 = pos_flat.to(torch.int64).clone()
    pos_rope = pos_i64.clone()
    pos_rope[pos_rope < 0] = 0
    pos_rope = pos_rope.reshape(1, batch)
    pos_padded = F.pad(pos_rope, (0, pad_to - batch), "constant", 0) if batch < pad_to else pos_rope
    pos_cache = pos_i64.to(torch.int32)
    return pos_padded, pos_cache


def test_inactive_rows_keep_minus_one_for_cache_but_zero_for_rope():
    # 8 active users + 24 vLLM pad rows at -1 (nearest-bucket pad to 32)
    pos = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80] + [-1] * 24, dtype=torch.int32)
    rope, cache = prepare_rope_and_cache_positions(pos, pad_to=32)
    assert rope.shape == (1, 32)
    assert cache.shape == (32,)
    # RoPE must never see -1 (would become UINT32_MAX and OOB the cos/sin table)
    assert int(rope.min()) >= 0
    assert torch.equal(rope[0, :8], torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64))
    assert torch.equal(rope[0, 8:], torch.zeros(24, dtype=torch.int64))
    # Cache/SDPA keep -1 skip sentinel on inactive rows
    assert torch.equal(cache[:8], torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32))
    assert torch.equal(cache[8:], torch.full((24,), -1, dtype=torch.int32))


def test_batch_lt_32_pads_rope_with_zeros():
    pos = torch.tensor([3, 4, -1, -1], dtype=torch.int32)
    rope, cache = prepare_rope_and_cache_positions(pos, pad_to=32)
    assert rope.shape == (1, 32)
    assert torch.equal(rope[0, :4], torch.tensor([3, 4, 0, 0], dtype=torch.int64))
    assert torch.equal(rope[0, 4:], torch.zeros(28, dtype=torch.int64))
    assert torch.equal(cache, torch.tensor([3, 4, -1, -1], dtype=torch.int32))


# ===== merged from tests/unit/test_decode_logits_rows.py: Host-sampled decode logits keep their vocab axis when the decode bucket is
def test_narrow_bucket_is_padded_not_folded():
    vocab = 1024
    logits = torch.arange(4 * vocab, dtype=torch.float32).reshape(1, 1, 4, vocab)
    out = _decode_logits_rows(logits, B=32, S=1)
    assert tuple(out.shape) == (32, 1, vocab)
    assert torch.equal(out[:4, 0, :], logits[0, 0])
    assert out[4:].abs().sum() == 0
    assert int(out[0, 0].argmax()) == vocab - 1  # the full row, not its first 1/8th


def test_full_batch_passes_through():
    logits = torch.randn(1, 1, 32, 64)
    out = _decode_logits_rows(logits, B=32, S=1)
    assert torch.equal(out, logits.reshape(32, 1, 64))


# ===== merged from tests/unit/test_prefill_sampling_layout.py: Host-only regression for the prefill lm_head / device-sampler boundary.
@pytest.fixture
def prefill_projection(monkeypatch):
    shard, gathered = object(), object()
    hidden = SimpleNamespace(shape=(1, 1, 32, 8), deallocate=lambda force: None)
    model = object.__new__(model_module.Gemma4Model)
    model.mesh_device = object()
    model.mesh_config = SimpleNamespace(tp=2)
    model.ccl_manager = object()
    model.hidden_size = 8
    model.lm_head_weight = SimpleNamespace(shape=(1, 1, 8, 16))
    model.final_logit_softcapping = None
    model.sampling = object()
    model._supports_on_device_sampling = True
    generator = object.__new__(Gemma4Generator)
    generator.model = [model]
    generator.data_parallel = 1
    monkeypatch.setattr(model_module, "_get_lm_head_program_config", lambda *a, **kw: None)
    monkeypatch.setattr(model_module.ttnn, "linear", lambda *a, **kw: shard)
    monkeypatch.setattr(ccl_module, "ccl_allgather", lambda *a, **kw: gathered)
    return generator, model, hidden, shard, gathered


def test_single_user_prefill_preserves_sampler_shards_and_host_vocabulary(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection

    def forward(self, *args, **kwargs):
        # Both eager prefill and traced prefill's deferred lm_head call this
        # without an explicit keep_sharded_for_sampling argument.
        return model._apply_lm_head(hidden, is_decode=False)

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    assert generator._prefill_forward_text_impl(sampling_params=object()) is shard
    assert generator._prefill_forward_text_impl(sampling_params=None) is gathered
    assert model._prefill_keep_logits_sharded is False
    model._supports_on_device_sampling = False
    assert generator._prefill_forward_text_impl(sampling_params=object()) is gathered


def test_nested_host_warmup_restores_sampling_layout(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection

    def forward(self, *args, sampling_params=None, **kwargs):
        if sampling_params is not None:
            assert self._prefill_forward_text_impl(sampling_params=None) is gathered
        return model._apply_lm_head(hidden, is_decode=False)

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    assert generator._prefill_forward_text_impl(sampling_params=object()) is shard
    assert model._prefill_keep_logits_sharded is False


def test_deferred_prefill_projection_keeps_sampling_layout_until_consumed(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection
    model._tail_slice_start, model._tail_slice_end = object(), object()
    model._replicate_to_mesh_mapper = lambda: None
    model._flush_deferred_bounded_fills_if_needed = lambda: None
    monkeypatch.setattr(model_module.ttnn, "from_torch", lambda tensor, **kw: tensor)
    monkeypatch.setattr(model_module.ttnn, "copy_host_to_device_tensor", lambda *a: None)
    monkeypatch.setattr(model_module.ttnn, "slice", lambda **kw: hidden)

    def forward(self, *args, **kwargs):
        # The shared prefill coordinator consumes the trace's hidden states
        # before it returns. Exercise Gemma's real deferred consumer + lm_head.
        return model.process_logits_after_prefill_trace(hidden, last_token_idx=17)

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    assert generator._prefill_forward_text_impl(sampling_params=object()) is shard
    assert model._prefill_keep_logits_sharded is False
    assert generator._prefill_forward_text_impl(sampling_params=None) is gathered


def test_failed_sampling_prefill_restores_host_layout(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection

    def forward(self, *args, **kwargs):
        assert model._apply_lm_head(hidden, is_decode=False) is shard
        raise RuntimeError("prefill failed")

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    with pytest.raises(RuntimeError, match="prefill failed"):  # allow-pytest.raises: host-only layout regression
        generator._prefill_forward_text_impl(sampling_params=object())
    assert model._apply_lm_head(hidden, is_decode=False) is gathered


# ===== merged from tests/unit/test_async_ahead_decode_tokens.py: Host-only tests for Gemma4 async-ahead decode token keep / slot_remap safety.
def test_matching_buffers_merge_device_token_when_pos_ahead():
    host_toks = torch.tensor([10, 20, 30], dtype=torch.int32)
    host_pos = torch.tensor([5, 6, 7], dtype=torch.int64)
    # Device is one step ahead on slots 0 and 2.
    dev_toks = torch.tensor([11, 20, 31], dtype=torch.int32)
    dev_pos = torch.tensor([6, 6, 8], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [11, 20, 31]
    assert pos.tolist() == [6, 6, 8]


def test_wider_device_feedback_merges_prefix():
    """Gemma4 pads feedback to width 32; host nearest-bucket B may be smaller.

    Slot 0 is async-ahead (pos host+1) → take device token. Remaining padded
    device rows are ignored via ``[:host_b]``.
    """
    host_toks = torch.tensor([99], dtype=torch.int32)
    host_pos = torch.tensor([3], dtype=torch.int64)
    dev_toks = torch.tensor([100, 2, 3, 4], dtype=torch.int32)
    dev_pos = torch.tensor([4, 2, 3, 4], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [100]
    assert pos.tolist() == [4]


def test_wider_device_unrelated_pos_keeps_host():
    """After batch recomposition, slot0 device pos may not match host → host wins."""
    host_toks = torch.tensor([99], dtype=torch.int32)
    host_pos = torch.tensor([3], dtype=torch.int64)
    dev_toks = torch.tensor([100, 2, 3, 4], dtype=torch.int32)
    # Device slot0 at unrelated position — must not clobber host.
    dev_pos = torch.tensor([10, 2, 3, 4], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [99]
    assert pos.tolist() == [3]


def test_sharded_narrow_device_buffer_falls_back():
    """Shard-0 exposes B/num_shards entries; must not index into host_b."""
    host_toks = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    host_pos = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    dev_toks = torch.tensor([10], dtype=torch.int32)  # one shard only
    dev_pos = torch.tensor([1], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "host_fallback"
    assert toks.tolist() == [10, 20, 30, 40]


def test_oob_slot_remap_falls_back_no_indexerror():
    """Remap indices from a larger previous batch must not gather."""
    host_toks = torch.tensor([10, 20], dtype=torch.int32)
    host_pos = torch.tensor([5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([11, 21], dtype=torch.int32)
    dev_pos = torch.tensor([6, 7], dtype=torch.int64)
    # Index 3 is OOB for host_b=2 (the pre-fix IndexError path).
    remap = torch.tensor([0, 3], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=remap)
    assert src == "host_fallback"
    assert toks.tolist() == [10, 20]


def test_valid_slot_remap_permutes_device_buffers():
    # Host positions already match the post-condense layout; remapped device
    # buffers are one step ahead so use_dev stays True for every slot.
    host_toks = torch.tensor([10, 20, 30], dtype=torch.int32)
    host_pos = torch.tensor([7, 5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([100, 200, 300], dtype=torch.int32)
    dev_pos = torch.tensor([6, 7, 8], dtype=torch.int64)
    remap = torch.tensor([2, 0, 1], dtype=torch.int64)  # -> toks [300,100,200]

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=remap)
    assert src == "merged"
    assert toks.tolist() == [300, 100, 200]
    assert pos.tolist() == [8, 6, 7]


def test_prefilled_slots_take_host_tokens():
    host_toks = torch.tensor([10, 20], dtype=torch.int32)
    host_pos = torch.tensor([5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([11, 21], dtype=torch.int32)
    dev_pos = torch.tensor([6, 7], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(
        host_toks,
        host_pos,
        dev_toks,
        dev_pos,
        prefilled_local={0},
    )
    assert src == "merged"
    assert toks.tolist() == [10, 21]
    assert pos.tolist() == [5, 7]


def test_mismatched_tok_pos_lengths_fall_back():
    """pos narrower than host_b → fallback (cannot form a full batch)."""
    host_toks = torch.tensor([10, 20], dtype=torch.int32)
    host_pos = torch.tensor([5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([11, 21], dtype=torch.int32)
    dev_pos = torch.tensor([6], dtype=torch.int64)

    toks, _, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "host_fallback"
    assert toks.tolist() == [10, 20]


def test_tok_pos_length_mismatch_still_merges_when_both_wide_enough():
    """Sampling writeback may shrink the token buffer's logical shape to 1 while
    RoPE pos stays pad-32. Both are still wide enough for host_b=1 → merge.
    Previous equality check forced host_fallback and async token doubling.
    """
    host_toks = torch.tensor([99], dtype=torch.int32)
    host_pos = torch.tensor([3], dtype=torch.int64)
    # Token buffer reports width 1 (post-sampling logical shape); pos is pad-32.
    dev_toks = torch.tensor([100], dtype=torch.int32)
    dev_pos = torch.tensor([4, 0, 0, 0], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [100]
    assert pos.tolist() == [4]


def test_slot_remap_beyond_host_batch_indexes_device_rows():
    """A slot id >= host_b is a valid *device row*, not out of bounds.

    Device feedback is padded to width 32 while the host batch can be 1, and a
    request may occupy any slot. The bound was previously ``host_b``, so a
    request living in slot 1 with host_b=1 was rejected as OOB and silently fell
    back to host tokens. The bound is the device buffer width.
    """
    host_toks = torch.tensor([111], dtype=torch.int32)
    host_pos = torch.tensor([100], dtype=torch.int64)
    dev_toks = torch.zeros(32, dtype=torch.int32)
    dev_pos = torch.zeros(32, dtype=torch.int64)
    # slot 0 is stale; slot 1 is this request's real device row.
    dev_toks[0], dev_pos[0] = 999, 7
    dev_toks[1], dev_pos[1] = 222, 100

    merged, merged_pos, src = merge_async_ahead_decode_tokens(
        host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=torch.tensor([1])
    )
    assert src == "merged"
    # Must read slot 1, never slot 0's stale 999.
    assert int(merged[0]) == 222
    assert int(merged_pos[0]) == 100


def test_slot_remap_beyond_device_width_still_falls_back():
    """Past the device buffer width is still OOB and must fall back."""
    host_toks = torch.tensor([111], dtype=torch.int32)
    host_pos = torch.tensor([100], dtype=torch.int64)
    dev_toks = torch.zeros(32, dtype=torch.int32)
    dev_pos = torch.zeros(32, dtype=torch.int64)
    merged, merged_pos, src = merge_async_ahead_decode_tokens(
        host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=torch.tensor([32])
    )
    assert src == "host_fallback"
    assert int(merged[0]) == 111


def test_inactive_host_rows_are_not_reactivated_by_a_device_position_of_zero():
    """A row the host disabled sits at -1. Device position 0 is "one step
    ahead" of it arithmetically, so the row came back as live with the
    device's stale token."""
    host_toks = torch.tensor([10, 0, 0], dtype=torch.int32)
    host_pos = torch.tensor([5, -1, -1], dtype=torch.int64)
    dev_toks = torch.tensor([11, 77, 78], dtype=torch.int32)
    dev_pos = torch.tensor([6, 0, -1], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [11, 0, 0]
    assert pos.tolist() == [6, -1, -1]


def test_live_rows_keep_device_feedback_next_to_inactive_rows():
    host_toks = torch.tensor([10, 20, 0], dtype=torch.int32)
    host_pos = torch.tensor([5, 9, -1], dtype=torch.int64)
    dev_toks = torch.tensor([11, 20, 78], dtype=torch.int32)
    dev_pos = torch.tensor([6, 9, 0], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [11, 20, 0]
    assert pos.tolist() == [6, 9, -1]


# ===== merged from tests/unit/test_hf_reference_cache_host.py: Host tests for the HF reference logits cache behind test_full_model (no device, no HF model).
@pytest.fixture
def ids():
    return torch.tensor([[2, 651, 6037, 576, 7001, 603] + [0] * 26], dtype=torch.int64)


def test_cache_dir_precedence(monkeypatch):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", "/x/explicit")
    monkeypatch.setenv("TT_CACHE_PATH", "/x/tt")
    assert hf_reference_cache_dir() == "/x/explicit"
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", "")
    assert hf_reference_cache_dir() == os.path.join("/x/tt", "hf_reference")
    monkeypatch.delenv("TT_CACHE_PATH")
    assert hf_reference_cache_dir() is None


def test_key_is_a_function_of_model_prompt_and_dtype(ids):
    k = hf_reference_key("google/gemma-4-31B-it", ids)
    assert k.startswith("google--gemma-4-31B-it__bfloat16__32__") and k.endswith(".pt")
    assert k == hf_reference_key("google/gemma-4-31B-it/", ids.clone())
    other = ids.clone()
    other[0, 3] = 999
    assert k != hf_reference_key("google/gemma-4-31B-it", other)
    assert k != hf_reference_key("google/gemma-4-12B-it", ids)
    assert k != hf_reference_key("google/gemma-4-31B-it", ids, dtype="float32")


def test_miss_computes_and_hit_loads(tmp_path, monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", str(tmp_path / "ref"))
    calls = []

    def compute():
        calls.append(1)
        torch.manual_seed(len(calls))
        return torch.randn(1, 32, 1000)

    a = hf_reference_logits("google/gemma-4-31B-it", ids, compute)
    b = hf_reference_logits("google/gemma-4-31B-it", ids, compute)
    assert len(calls) == 1 and torch.equal(a, b)
    files = os.listdir(tmp_path / "ref")
    assert files == [hf_reference_key("google/gemma-4-31B-it", ids)]  # no temp file left behind
    other = ids.clone()
    other[0, 3] = 999
    hf_reference_logits("google/gemma-4-31B-it", other, compute)
    assert len(calls) == 2 and len(os.listdir(tmp_path / "ref")) == 2


def test_prompt_mismatch_in_a_file_recomputes(tmp_path, monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", str(tmp_path))
    path = tmp_path / hf_reference_key("m", ids)
    torch.save({"input_ids": ids.clone().fill_(7), "logits": torch.zeros(1, 32, 4), "model": "m"}, path)
    out = hf_reference_logits("m", ids, lambda: torch.ones(1, 32, 4))
    assert torch.equal(out, torch.ones(1, 32, 4))
    assert torch.equal(torch.load(path)["input_ids"], ids)  # rewritten for the real prompt


def test_no_cache_dir_computes_every_time(monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", "")
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    calls = []
    for _ in range(2):
        hf_reference_logits("m", ids, lambda: calls.append(1) or torch.zeros(1))
    assert len(calls) == 2


def test_unwritable_cache_dir_keeps_the_computed_reference(tmp_path, monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", str(tmp_path / "file_not_dir" / "x"))
    (tmp_path / "file_not_dir").write_text("not a directory")
    out = hf_reference_logits("m", ids, lambda: torch.full((1,), 3.0))
    assert float(out) == 3.0
