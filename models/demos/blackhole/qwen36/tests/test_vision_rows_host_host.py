# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the r13 vision-rows hotfix: the MTP serving path keeps vision rows on the HOST (a device tensor made
after trace capture was overwritten by trace replays between prefill segments -> garbage answers for an image that is
not in the first prefill segment, d30u20 11-12)."""
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import qwen36_vllm
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

H = 4


def test_set_vision_merge_places_host_rows_without_a_device_read(monkeypatch):
    staged = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: t.clone())
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda src, dst: staged.append((dst, src)))
    monkeypatch.setattr(ttnn, "to_torch", lambda *a, **k: pytest.fail("host rows must not be read back from device"))
    m = types.SimpleNamespace(
        _vis_buf=types.SimpleNamespace(shape=(1, 8, H)),
        _vis_mask_buf="MASK",
        num_devices=1,
        args=types.SimpleNamespace(dim=H),
    )
    m._vision_placeholder_token_id = lambda: 7
    m.vision_rows_host = lambda vt: Qwen36Model.vision_rows_host(m, vt)
    rows = torch.arange(5, dtype=torch.bfloat16).reshape(5, 1).repeat(1, H)  # row r = value r
    Qwen36Model._set_vision_merge(m, torch.tensor([[1, 1, 7, 7, 1, 1, 1, 1]]), rows, vis_row_offset=3)
    vis = [s for d, s in staged if d is m._vis_buf][0]
    mask = [s for d, s in staged if d == "MASK"][0]
    assert vis[0, 2:4, 0].tolist() == [3.0, 4.0] and mask[0, :, 0].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert Qwen36Model.vision_rows_host(m, None) is None


def test_mtp_prefill_reads_tower_rows_to_host_before_the_engine(monkeypatch):
    g = Qwen36MTPContractForCausalLM.__new__(Qwen36MTPContractForCausalLM)
    calls = []

    class _Eng:
        def prefill(self, slot, ids, row, *a, **kw):
            calls.append(kw)
            return torch.zeros(16)

    model = types.SimpleNamespace()
    model.get_image_features = lambda pix, grid: "DEVICE_ROWS"
    model.vision_rows_host = lambda vt: calls.append(("to_host", vt)) or torch.ones(4, H)
    g.model = [model]
    g._ensure_ready = lambda kv_cache=None: _Eng()
    g._table = lambda pt: pt.to(torch.int32)
    monkeypatch.setattr(qwen36_vllm, "_spec_prefix_cache_enabled", lambda: True)
    g.prefill_forward(
        torch.arange(1, 13, dtype=torch.int32).reshape(1, 12),
        torch.ones(1, 4, dtype=torch.int32),
        None,
        torch.tensor([10]),
        empty_slots=[0],
        pixel_values=[torch.zeros(16, 8)],
        image_grid_thw=[torch.tensor([[1, 4, 4]])],
        start_pos=torch.tensor([0]),
    )
    assert calls[0] == ("to_host", "DEVICE_ROWS")
    assert isinstance(calls[1]["vision_tokens"], torch.Tensor)
