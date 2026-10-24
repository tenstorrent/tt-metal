# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: when a prefill's per-layer page-table install forces the H2D (``Gemma4Generator._install_per_layer_page_tables``):
always after a decode wrote the row count's device buffers (a decode trace input rewrites them) and on the first
write, never when the last writer was prefill (the host-unchanged shortcut then decides). No device, no conftest."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.generator import Gemma4Generator


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
