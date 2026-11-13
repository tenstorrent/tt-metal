# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the exact drafter and verify-tail changes of Qwen36MTPContractForCausalLM.

* The fused per-shard argmax (tt/spec_argmax_fused.py, kernels/argmax_shard_local.cpp and argmax_shard_pick.cpp), its
  ranking in torch: per shard the order value of the row max and its lowest column, then the device with the strictly
  greatest order value scanning upward. On bf16 and fp32 rows with every tie case (inside a shard, across shards, all
  equal, +0 / -0 maxima, -inf, 1-ulp pairs, quantized rows) it returns torch.argmax over the full row, the lowest id
  among the maxima, which is also the tie rule of ttnn.argmax that the replaced tails use.
* Where the ids are taken: draft_argmax takes the fused path for a shard it supports and _argmax_sharded otherwise;
  the verify tail runs the fused argmax on the vocab shard before the all-gather.
* KV-only maintenance: a need_logits=False drafter forward under TP stops the MTP layer after its paged KV write
  (the attention gets kv_only, the layer returns before the residual and MLP), and the engine's callers free a None
  hidden.
* The one-row embedding hands the drafter's fc the same [1,1,B,dim/tp] row block.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP, draft_argmax
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, ROWS

VOCAB = 248320


def _order(x):
    """The stage-1 kernel's order value of a float tensor in its own bit width: |v| for v >= +0, -|v| for v <= -0, as
    int64 (bf16 bits are 16 wide, fp32 32 wide)."""
    if x.dtype == torch.bfloat16:
        bits = x.view(torch.int16).to(torch.int64) & 0xFFFF
        sign, mag = bits >> 15, bits & 0x7FFF
    else:
        bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sign, mag = bits >> 31, bits & 0x7FFFFFFF
    return torch.where(sign == 1, -mag, mag)


def fused_argmax_reference(x, num_devices):
    """The fused per-shard argmax of rows ``x`` [R, V] in torch: per device shard the max order value and its lowest
    column (a strict '>' over increasing columns), then the device with the strictly greatest order value, scanning
    devices upward."""
    rows, vocab = x.shape
    shard = vocab // num_devices
    o = _order(x).reshape(rows, num_devices, shard)
    best = o.max(dim=-1).values  # [R, D]
    col = (o == best.unsqueeze(-1)).to(torch.int64).argmax(dim=-1)  # the first column holding the max
    dev = (best == best.max(dim=-1, keepdim=True).values).to(torch.int64).argmax(dim=-1)  # the first such device
    return dev * shard + col.gather(1, dev.unsqueeze(1)).squeeze(1)


def tie_rows(dtype, num_devices=8, rows=12, seed=0):
    """[rows, VOCAB] logits in ``dtype``: row r is case r % 12 (ties across and inside shards, all equal, negative,
    +0 / -0 maxima, -inf, the last id, a shard boundary, a 1-ulp pair, quantized ties, shifted, random). The cases of
    the device test of the fused argmax (atupe's)."""
    shard = VOCAB // num_devices
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, VOCAB, generator=g, dtype=torch.float32)
    for r in range(rows):
        row, case = x[r], r % 12
        if case == 0:
            row[100] = row[2 * shard + 500] = 20.0
        elif case == 1:
            row[(num_devices - 1) * shard + 1000] = row[(num_devices - 1) * shard + 30000] = 21.0
        elif case == 2:
            row.fill_(1.0)
        elif case == 3:
            row.copy_(-row.abs() - 3.0)
            row[shard + 17] = row[5 * shard + 3] = -0.5
        elif case == 4:
            row.copy_(-row.abs() - 1.0)
            row[shard + 40] = -0.0
            row[shard + 41] = 0.0
            row[4 * shard + 9] = 0.0
        elif case == 5:
            row[VOCAB - 1] = 22.0
            row[7] = row[shard + 9] = float("-inf")
        elif case == 6:
            row[0] = row[3 * shard] = 19.0
        elif case == 7:
            row[6 * shard - 1] = row[6 * shard] = 18.0
        elif case == 8:
            hi = torch.tensor(30.0, dtype=dtype)
            lo = (
                (hi.view(torch.int16) - 1).view(torch.bfloat16)
                if dtype == torch.bfloat16
                else torch.nextafter(hi, torch.tensor(0.0))
            )
            row[shard + 7] = float(lo)
            row[3 * shard + 9] = float(hi)
        elif case == 9:
            row.copy_(torch.clamp(torch.round(row * 2) / 2, max=2.5))
        elif case == 10:
            row.add_(-50.0)
            row[4 * shard + 31] = -40.0
    return x.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16-verify", "fp32-drafter"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fused_argmax_ranking_is_torch_argmax_with_every_tie(dtype, seed):
    x = tie_rows(dtype, seed=seed)
    want = torch.argmax(x.float(), dim=-1)
    got = fused_argmax_reference(x, 8)
    assert torch.equal(got, want), f"{got.tolist()} != {want.tolist()}"
    # the row cases resolve as their comments say: the lowest id among the maxima
    shard = VOCAB // 8
    assert got[0] == 100 and got[1] == 7 * shard + 1000 and got[2] == 0 and got[3] == shard + 17
    assert got[4] == shard + 40 and got[5] == VOCAB - 1 and got[6] == 0 and got[7] == 6 * shard - 1
    assert got[8] == 3 * shard + 9


def test_order_values_rank_like_the_floats():
    v = torch.tensor([float("-inf"), -3.0, -1e-30, -0.0, 0.0, 1e-30, 2.0, float("inf")])
    for dtype in (torch.bfloat16, torch.float32):
        o = _order(v.to(dtype))
        assert o.tolist() == sorted(o.tolist()) and o[3] == o[4] == 0, f"{dtype}: {o.tolist()}"


class _Fused:
    def __init__(self, supported):
        self.supported, self.calls = supported, []

    def supports(self, logits):
        return None if self.supported else "not supported"

    def __call__(self, logits, mode):
        self.calls.append(mode)
        return "fused ids"


@pytest.mark.parametrize("supported", [True, False])
def test_draft_argmax_takes_the_fused_path_for_a_supported_shard(supported):
    fused = _Fused(supported)
    mtp = SimpleNamespace(_sharded_argmax=True, _fused_argmax=fused, _argmax_sharded=lambda logits: "sharded ids")
    assert draft_argmax(mtp, "shard") == ("fused ids" if supported else "sharded ids")
    assert fused.calls == ([ROWS] if supported else [])


def test_the_verify_takes_its_ids_from_the_shard_before_the_gather(monkeypatch):
    import ttnn

    log = []
    model = Qwen36Model.__new__(Qwen36Model)
    model._decode_ar = None  # the fractured residual
    model._lmhead_vocab_sharded = True
    model.lm_head_weight = "w"
    model.norm = lambda rows, mode: "normed"
    model.spec_feed_rows = lambda rows: "feed"
    model._gather_vocab = lambda shard: log.append(("gather", shard)) or "logits"
    model._lm_head = lambda normed: log.append(("lm_head", normed)) or "logits"
    monkeypatch.setattr(ttnn, "clone", lambda x, memory_config=None: "rows")
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "linear", lambda x, w: log.append(("linear", x, w)) or "shard")
    ids = lambda shard: log.append(("ids", shard)) or "ids"
    assert model._spec_tail_ids("x", ids) == ("logits", "feed", "ids")
    assert log == [("linear", "normed", "w"), ("ids", "shard"), ("gather", "shard")]
    log.clear()
    assert model._spec_tail("x") == ("logits", "feed")
    assert log == [("lm_head", "normed")]
    fused = _Fused(True)
    model._vfy_argmax_fused = fused
    assert model._vfy_shard_ids("shard") == "fused ids" and fused.calls == [LANES]
    model._vfy_argmax_fused = _Fused(False)
    assert model._vfy_shard_ids("shard") is None


def test_the_verify_on_the_replicated_residual_takes_its_ids_from_the_shard(monkeypatch):
    """On the replicated decode residual (decode_allreduce.py: a full-width x), the verify tail takes the final norm
    locally, runs the per-shard argmax on this device's vocab shard before the gather, and hands the drafter this
    device's span of the norm; without shard_ids it takes the plain LM head."""
    import ttnn

    log = []
    model = Qwen36Model.__new__(Qwen36Model)
    model._decode_ar = object()
    model.args = SimpleNamespace(dim=64)
    model._lmhead_vocab_sharded = True
    model.lm_head_weight = "w"
    model._final_norm_decode = lambda x: log.append(("final norm", x.name)) or "normed"
    model._gather_vocab = lambda shard: log.append(("gather", shard)) or "logits"
    model._lm_head = lambda normed: log.append(("lm_head", normed)) or "logits"
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "linear", lambda x, w: log.append(("linear", x, w)) or "shard")
    monkeypatch.setattr(ttnn, "mesh_partition", lambda t, **kw: log.append(("partition", t, kw["dim"])) or "feed")
    x = SimpleNamespace(shape=(1, 1, 32, 64), name="x")
    ids = lambda shard: log.append(("ids", shard)) or "ids"
    assert model._spec_tail_ids(x, ids) == ("logits", "feed", "ids")
    assert log == [
        ("final norm", "x"),
        ("linear", "normed", "w"),
        ("ids", "shard"),
        ("gather", "shard"),
        ("partition", "normed", 3),
    ]
    log.clear()
    assert model._spec_tail(x) == ("logits", "feed")
    assert log == [("final norm", "x"), ("lm_head", "normed"), ("partition", "normed", 3)]


def _layer(full_attention=True, num_devices=8):
    layer = MagicMock()
    layer.is_full_attention, layer.num_devices = full_attention, num_devices
    layer._decode_ar = None  # the fractured residual
    layer.args.get_norm_config.return_value = {}
    layer.attention_norm.return_value = "attn_input"
    layer.attention.forward_decode.return_value = None
    return layer


def test_a_kv_only_layer_stops_after_the_attentions_kv_write(monkeypatch, expect_error):
    import ttnn

    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    layer = _layer()
    out = Qwen36DecoderLayer.forward(layer, "x", mode="decode", page_table="pt", alias_kv_write=True, kv_only=True)
    assert out is None
    assert layer.attention.forward_decode.call_args.kwargs["kv_only"] is True
    layer.ffn_norm.assert_not_called()
    layer.feed_forward.forward.assert_not_called()
    for bad in (_layer(full_attention=False), _layer(num_devices=1)):
        with expect_error(ValueError, "kv_only needs a TP full-attention decode"):
            Qwen36DecoderLayer.forward(bad, "x", mode="decode", kv_only=True)
    with expect_error(ValueError, "kv_only needs a TP full-attention decode"):
        Qwen36DecoderLayer.forward(_layer(), "x", mode="prefill", kv_only=True)


@pytest.mark.parametrize("need_logits, num_devices, page_table", [(False, 8, "pt"), (False, 1, "pt"), (False, 8, None)])
def test_drafter_kv_maintenance_is_kv_only_under_tp_with_a_page_table(
    monkeypatch, need_logits, num_devices, page_table
):
    import ttnn

    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    mtp = Qwen36MTP.__new__(Qwen36MTP)
    mtp.num_devices = num_devices
    mtp._embed_rows = lambda token_ids: "emb"
    mtp._fuse = lambda emb, hidden, mode: "fused"
    mtp.decoder = MagicMock()
    mtp.decoder.forward.return_value = None if (num_devices > 1 and page_table) else "block out"
    out = Qwen36MTP.forward_decode(mtp, "h", "tok", "pos", "cos", "sin", page_table, need_logits=need_logits)
    kv_only = num_devices > 1 and page_table is not None
    assert mtp.decoder.forward.call_args.kwargs["kv_only"] is kv_only
    assert out == ((None, None) if kv_only else (None, "block out"))


def test_the_engine_frees_a_kv_only_hidden(monkeypatch):
    import ttnn

    freed = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t: freed.append(t))
    spec_engine.free_tensors("a", None, "b")
    assert freed == ["a", "b"]


def test_the_one_row_embedding_is_one_row_block(monkeypatch):
    """The drafter's fc reads [1,1,B,dim/tp]: _embed_rows views the row-major lookup as that block, pads it to whole
    tiles in row major and tilizes once."""
    import ttnn

    log = []
    mtp = Qwen36MTP.__new__(Qwen36MTP)
    mtp.embd = SimpleNamespace(weights="table")
    monkeypatch.setattr(
        ttnn,
        "embedding",
        lambda ids, w, layout, memory_config: log.append(("lookup", w, layout)) or SimpleNamespace(shape=(8, 1, 640)),
    )
    monkeypatch.setattr(
        ttnn, "reshape", lambda t, *shape: log.append(("reshape", shape)) or SimpleNamespace(shape=shape)
    )
    monkeypatch.setattr(ttnn, "pad", lambda t, padding, value: log.append(("pad", padding, value)) or "full")
    monkeypatch.setattr(ttnn, "to_layout", lambda t, layout, memory_config: log.append(("tilize", t)) or "tiled")
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "TILE_SIZE", 32)
    out = Qwen36MTP._embed_rows(mtp, SimpleNamespace(shape=(8, 1)))
    assert log[0] == ("lookup", "table", ttnn.ROW_MAJOR_LAYOUT)
    assert log[1] == ("reshape", ((1, 1, 8, 640),))
    assert log[2] == ("pad", [(0, 0), (0, 0), (0, 24), (0, 0)], 0.0)
    assert log[3] == ("tilize", "full")
    assert log[4][0] == "reshape" and out.shape == log[4][1]
