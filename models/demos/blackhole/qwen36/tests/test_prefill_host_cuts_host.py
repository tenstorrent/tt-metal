# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the prefill host cuts of Qwen36MTPContractForCausalLM.

* The drafter's prompt KV warm-up (Qwen36MTP.forward_prefill, run per prompt chunk by MTPBlockEngine.prefill_hook)
  stops after the drafter layer's paged K and V fill: TPAttention.forward_prefill_paged(kv_only=True) computes and
  writes K and V by the operations of the full forward and runs nothing after the fill.
* The prompt ids become Python ints by one tolist() (spec_engine.prompt_ints), not a 0-d tensor per token.
* GdnBlockState.zero_blocks replays one captured trace per group of blocks instead of dispatching 2 fills per layer
  slot.

The ttnn calls are replaced by functions that build symbolic tensors, so the K and V a fill writes can be compared as
expressions of the head-split outputs.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp


class _Stop(Exception):
    """Raised where the full attention forward first uses Q after the K and V fill."""

    def __init__(self):
        super().__init__("the full forward reached Q after the K and V fill")


def _leaves(x):
    if isinstance(x, tuple):
        for y in x[1:]:
            yield from _leaves(y)
    elif isinstance(x, str):
        yield x


@pytest.fixture
def ops(monkeypatch):
    """Record the attention prefill's paged fills and frees over symbolic tensors."""
    rec = SimpleNamespace(fills=[], freed=[], calls=[])

    def typecast(t, dtype=None, **kw):
        if "q" in set(_leaves(t)):
            raise _Stop
        return ("bf8", t)

    def call(name):
        def f(*a, **kw):
            rec.calls.append(name)
            return (name, *[x for x in a if isinstance(x, (str, tuple))])

        return f

    monkeypatch.setattr(ttnn, "rms_norm", lambda t, **kw: ("rms_norm", t))
    monkeypatch.setattr(ttnn, "multiply", lambda a, b, **kw: ("mul", a, b))
    monkeypatch.setattr(ttnn, "typecast", typecast)
    monkeypatch.setattr(ttnn, "slice", call("slice"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: rec.freed.append(t))
    monkeypatch.setattr(
        ttnn.experimental,
        "paged_fill_cache",
        lambda cache, src, table, batch_idx=0: rec.fills.append((cache.name, src, table)),
        raising=False,
    )
    monkeypatch.setattr(attn_tp, "apply_partial_rope_prefill", lambda t, cos, sin, n, rd: ("rope", t, cos, sin))
    monkeypatch.setattr(tpc, "prefill_act_mem", lambda S, cfg: "act memory")
    return rec


def _attention():
    att = object.__new__(attn_tp.TPAttention)
    att.tw = {"q_norm": "q_norm", "k_norm": "k_norm"}
    att.NH, att.NKV, att.HD, att.rope_dim = 3, 1, 256, 64
    att.use_paged = True
    att.paged_k = SimpleNamespace(name="K cache", shape=(64, 1, 1024, 256))
    att.paged_v = SimpleNamespace(name="V cache", shape=(64, 1, 1024, 256))
    att._sdpa_bf8 = True
    att.args = SimpleNamespace()
    att._qkv = lambda x: ("qg", "kp", "vp")
    att._make_heads = lambda qg, kp, vp, S: ("q", "gate", "k", "v")
    return att


def test_the_prompt_kv_warm_up_writes_the_full_forwards_k_and_v_and_stops(ops, expect_error):
    """kv_only computes K (norm, RoPE, bfloat8_b cast) and V (cast) by the full forward's operations from the same
    head-split outputs, writes them through the same page table, frees Q and the gate unused and returns None: no Q
    norm or RoPE, no SDPA, gate or out-projection. The drafter's K and V are unchanged."""
    att = _attention()
    x = SimpleNamespace(shape=(1, 1, 2048, 640))
    chunk_table = SimpleNamespace(shape=(1, 2))  # two 1024-token blocks hold the 2048 rows
    kw = dict(chunk_page_table=chunk_table, chunk_start_idx=0, chunk_start_idx_tensor="csi")
    with expect_error(_Stop, "reached Q after the K and V fill"):
        att.forward_prefill_paged(x, "cos", "sin", "page table", **kw)
    full = list(ops.fills)
    ops.fills.clear()
    ops.freed.clear()
    assert att.forward_prefill_paged(x, "cos", "sin", "page table", kv_only=True, **kw) is None
    assert (
        ops.fills
        == full
        == [
            ("K cache", ("bf8", ("rope", ("mul", ("rms_norm", "k"), "k_norm"), "cos", "sin")), chunk_table),
            ("V cache", ("bf8", "v"), chunk_table),
        ]
    )
    assert "q" in ops.freed and "gate" in ops.freed
    assert not any("q" in set(_leaves(t)) for t in ops.freed if isinstance(t, tuple)), "Q was never normed or rotated"


def test_kv_only_refuses_a_sequence_parallel_split(ops, expect_error):
    with expect_error(ValueError, "no sequence-parallel split"):
        _attention().forward_prefill_paged(
            SimpleNamespace(shape=(1, 1, 2048, 640)), "cos", "sin", "page table", sp=object(), kv_only=True
        )


def test_the_drafter_prompt_forward_is_kv_only_when_paged_at_tp(monkeypatch):
    """Qwen36MTP.forward_prefill asks its layer for the K and V fill only under TP with a page table (the prompt KV
    warm-up, whose callers free the output unread); a forward without a page table runs the whole layer."""
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

    monkeypatch.setattr(ttnn, "reshape", lambda t, shape, **kw: t)
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, cfg: t)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    seen = []

    def forward(fused, **kw):
        seen.append(kw["kv_only"])
        return None if kw["kv_only"] else "block output"

    mtp = object.__new__(Qwen36MTP)
    mtp.embd = lambda t: SimpleNamespace(shape=(1, 8, 640))
    mtp._fuse = lambda a, b, mode: "fused"
    mtp.decoder = SimpleNamespace(forward=forward)
    tokens = SimpleNamespace(shape=(1, 8))
    mtp.num_devices = 8
    assert mtp.forward_prefill("hidden", tokens, "cos", "sin", "page table") is None
    assert mtp.forward_prefill("hidden", tokens, "cos", "sin", None) == "block output"
    mtp.num_devices = 1
    assert mtp.forward_prefill("hidden", tokens, "cos", "sin", "page table") == "block output"
    assert seen == [True, False, False]


def test_a_kv_only_prefill_layer_stops_after_the_attention(monkeypatch, expect_error):
    """Qwen36DecoderLayer.forward(kv_only=True) in a paged TP prefill hands kv_only to the attention and returns None
    without the residual add, the ffn norm or the MLP; without a page table it refuses."""
    from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer

    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(tpc, "prefill_act_mem", lambda S, cfg: "act memory")
    seen = {}

    def forward_prefill_paged(x, cos, sin, page_table, **kw):
        seen.update(kw)
        return None

    def unused(*a, **kw):
        raise AssertionError("ran past the attention")

    layer = object.__new__(Qwen36DecoderLayer)
    layer.is_full_attention, layer.num_devices, layer._decode_ar = True, 8, None
    layer.args = SimpleNamespace(get_norm_config=lambda name, mode: {}, dim=5120)
    layer.attention_norm = lambda x, mode=None, norm_config=None: "normed"
    layer.attention = SimpleNamespace(forward_prefill_paged=forward_prefill_paged)
    layer.ffn_norm = layer.feed_forward = SimpleNamespace(forward=unused, __call__=unused)
    x = SimpleNamespace(shape=(1, 1, 2048, 640))
    assert layer.forward(x, "cos", "sin", mode="prefill", page_table="page table", kv_only=True) is None
    assert seen["kv_only"] is True
    with expect_error(ValueError, "kv_only needs a TP full-attention decode or paged prefill"):
        layer.forward(x, "cos", "sin", mode="prefill", kv_only=True)


# ------------------------------------------------------------------------------------------------ prompt ids


class _NoIter(torch.Tensor):
    """A token tensor that refuses per-token iteration."""

    def __iter__(self):
        raise AssertionError("the prompt was read one 0-d tensor per token")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint8])
def test_prompt_ints_are_the_per_token_ints_by_one_tolist(dtype):
    from models.demos.blackhole.qwen36.tt.spec_engine import prompt_ints

    tokens = torch.randint(0, 200, (1, 300)).to(dtype)
    want = [int(t) for t in tokens.reshape(-1)]
    got = prompt_ints(tokens.as_subclass(_NoIter))
    assert got == want and all(type(t) is int for t in got)
    assert prompt_ints([5, 6, 7]) == [5, 6, 7]


def test_prompt_ints_of_floating_or_bool_tokens_keep_int_per_token():
    from models.demos.blackhole.qwen36.tt.spec_engine import prompt_ints

    assert prompt_ints(torch.tensor([1.9, -2.5, 3.0])) == [1, -2, 3]
    got = prompt_ints(torch.tensor([True, False]))
    assert got == [1, 0] and all(type(t) is int for t in got)


def test_the_prefill_hook_reads_the_row_tokens_without_iterating_them(monkeypatch):
    """MTPBlockEngine.prefill_hook hands the drafter's chunk warm-up the row's ids as ints, read by one tolist()."""
    from models.demos.blackhole.qwen36.tt.spec_engine import MTPBlockEngine

    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    warmed = []
    eng = SimpleNamespace(
        model=SimpleNamespace(_fit_spec_page_table=lambda pt, nb: pt, spec_feed_rows=lambda hidden: "feed", rope=None),
        nb=4,
        _check_scratch=lambda pt: None,
        _warm_mtp_chunk=lambda feed, cs, n, prompt, pt: warmed.append(prompt),
    )
    tokens = torch.arange(100, 400, dtype=torch.int64)
    on_chunk = MTPBlockEngine.prefill_hook(eng, 0, tokens.as_subclass(_NoIter), "page table", False)
    on_chunk("hidden", 0, 300)
    assert warmed == [list(range(100, 400))]


# ------------------------------------------------------------------------------------------------ block zeroing


@pytest.fixture
def zeroing(monkeypatch):
    """A GdnBlockState over 4 layer slots that zeroes blocks 2 at a time, and a record of its device work."""
    dev = SimpleNamespace(fills=[], copies=[], traces={}, capturing=None, replays=[], released=[], syncs=0)

    def fill(cache, src, table, batch_idx=0):
        (dev.traces[dev.capturing] if dev.capturing is not None else dev.fills).append((cache, src, table))

    def begin(mesh, cq_id=0):
        dev.capturing = len(dev.traces) + 1
        dev.traces[dev.capturing] = []
        return dev.capturing

    def end(mesh, tid, cq_id=0):
        dev.capturing = None

    def execute(mesh, tid, cq_id=0, blocking=True):
        assert not blocking
        dev.replays.append((tid, list(dev.copies[-1])))

    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: SimpleNamespace(host=t.clone()))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, d: dev.copies.append(host.host.tolist()[0]))
    monkeypatch.setattr(ttnn.experimental, "paged_fill_cache", fill, raising=False)
    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", end)
    monkeypatch.setattr(ttnn, "execute_trace", execute)
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: dev.released.append(tid))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: setattr(dev, "syncs", dev.syncs + 1))
    st = object.__new__(gbs.GdnBlockState)
    st.mesh = None
    st.kv_caches = [(f"k{j}", f"v{j}") for j in range(4)]
    st._zero, st._zero_table, st._zero_refs, st._zero_trace = "zero", SimpleNamespace(shape=[1, 2]), [], None
    st._warmed = False
    st.layers = {}
    st._write_traces, st._warmed_variants = {}, ()
    return st, dev


def test_zeroing_replays_one_capture_per_group_of_blocks(zeroing):
    """After warmup, record_traces captures the 2 fills per layer slot once; zero_blocks then stages each group's
    table and replays the capture, without an eager fill or a wait. Before the capture it fills eagerly, as before."""
    st, dev = zeroing
    st.zero_blocks([5, 6, 7])
    assert len(dev.fills) == 2 * 2 * len(st.kv_caches) and dev.replays == []
    st.record_traces()
    assert dev.traces == {}, "no capture before warmup compiled the fills"
    st._warmed = True
    st.record_traces()
    st.record_traces()
    assert list(dev.traces) == [1]
    assert dev.traces[1] == [(c, "zero", st._zero_table) for kv in st.kv_caches for c in kv]
    dev.fills.clear()
    dev.copies.clear()
    st.zero_blocks([5, 6, 7])
    assert dev.fills == [] and dev.syncs == 0
    assert dev.replays == [(1, [5, 6]), (1, [7, 0])]
    assert [h.host.tolist() for h in st._zero_refs] == [[[5, 6]], [[7, 0]]]
    st.release()
    assert dev.released == [1] and st._zero_trace is None
    st.zero_blocks([9])
    assert len(dev.fills) == 2 * len(st.kv_caches)
