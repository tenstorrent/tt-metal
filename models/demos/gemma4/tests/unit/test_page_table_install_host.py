# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: when a prefill's per-layer page-table install forces the H2D (``Gemma4Generator._install_per_layer_page_tables``):
always after a decode wrote the row count's device buffers (a decode trace input rewrites them) and on the first
write, never when the last writer was prefill (the host-unchanged shortcut then decides). No device, no conftest."""

from types import SimpleNamespace

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
