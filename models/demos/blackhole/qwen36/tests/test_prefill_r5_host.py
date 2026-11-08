# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests (no device; ttnn importable) of the three r5 prefill flags:

* QWEN36_PREFILL_AGMM_TUNED (tp_common.agmm_tuned_config): the per-shape, per-row-count AGMM configs, K_block unchanged;
* QWEN36_PREFILL_RS_WORKERS (tp_common.prefill_rs_workers / prefill_rs_kwargs) and how tt_all_reduce passes it on;
* QWEN36_PREFILL_MTP_LEAN: the exact-multiple fast-prefill segment runs on_chunk once per chunk and, for a non-final
  vLLM chunk, no last-row LM head (MTPSpecEngine._fast_segment with the fakes of test_spec_fastprefill_host.py).

  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest -q \
      models/demos/blackhole/qwen36/tests/test_prefill_r5_host.py
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import _engine, _fake_model, _row, fake_ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tc
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine
from models.tt_transformers.tt import ccl

__all__ = ["fake_ttnn"]  # fixture re-export


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for name in ("_PREFILL_AGMM_TUNED", "_PREFILL_RS_WORKERS", "_PREFILL_RS_LINKS", "_PREFILL_MTP_LEAN"):
        monkeypatch.setattr(tc, name, None)
    for env in ("QWEN36_PREFILL_AGMM_TUNED", "QWEN36_PREFILL_RS_WORKERS", "QWEN36_PREFILL_RS_LINKS", "QWEN36_PREFILL_MTP_LEAN"):
        monkeypatch.delenv(env, raising=False)
    yield


# --------------------------------------------------------------------------- #
# QWEN36_PREFILL_AGMM_TUNED
# --------------------------------------------------------------------------- #
class _Mesh:
    def __init__(self, x=12, y=10):
        self.grid = types.SimpleNamespace(x=x, y=y)

    def compute_with_storage_grid_size(self):
        return self.grid


def _w(k, n):
    return types.SimpleNamespace(shape=(1, 1, k, n))


# (swiglu, K, N/tp as the model's weight has it, K/tp) of the four prefill AGMM call sites on TP8
SHAPES = {
    "mlp_gateup": (True, 5120, 4352, 640),
    "gdn_qkvzab": (False, 5120, 2060, 640),
    "attn_qkv": (False, 5120, 2048, 640),
    "gdn_out": (False, 6144, 640, 768),
}
# expected (grid_x, grid_y, M_block, K_block, N_block, transpose, workers) per (shape, rows); absent = today's config
EXPECTED = {
    ("mlp_gateup", 128): (12, 9, 8, 4, 16, False, 5),
    ("mlp_gateup", 512): (12, 9, 8, 4, 16, False, 5),
    ("mlp_gateup", 1024): (12, 9, 8, 4, 16, False, 5),
    ("mlp_gateup", 2048): (11, 9, 8, 4, 16, True, 6),
    ("mlp_gateup", 8192): (12, 9, 8, 4, 16, True, 6),
    ("gdn_qkvzab", 128): (12, 9, 4, 4, 8, False, 5),
    ("gdn_qkvzab", 8192): (12, 9, 8, 4, 8, True, 6),
    ("attn_qkv", 128): (12, 9, 4, 4, 8, False, 5),
    ("attn_qkv", 8192): (12, 9, 8, 4, 16, True, 6),
    ("gdn_out", 1024): (12, 9, 4, 8, 8, True, 6),
    ("gdn_out", 2048): (11, 9, 4, 8, 8, True, 6),
    ("gdn_out", 8192): (11, 9, 4, 8, 8, True, 6),
}
ROWS = (32, 128, 512, 1024, 2048, 4096, 8192)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("rows", ROWS)
def test_agmm_tuned_off_is_none(shape, rows):
    swiglu, k, n, kl = SHAPES[shape]
    assert tc.agmm_tuned_config(rows, _w(k, n), kl, swiglu, _Mesh()) is None


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("rows", ROWS)
def test_agmm_tuned_table(monkeypatch, shape, rows):
    monkeypatch.setenv("QWEN36_PREFILL_AGMM_TUNED", "1")
    swiglu, k, n, kl = SHAPES[shape]
    r = tc.agmm_tuned_config(rows, _w(k, n), kl, swiglu, _Mesh())
    exp = EXPECTED.get((shape, rows))
    if exp is None:
        assert r is None
        return
    cfg, links, workers, transpose = r
    gx, gy, mb, kb, nb, tr, wk = exp
    assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (gx, gy)
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (mb, kb, nb)
    assert (cfg.subblock_h, cfg.subblock_w) == (1, 4)
    # the K blocking (hence each output tile's accumulation order) is today's
    assert cfg.K_block_size == tc.agmm_k_block_size(kl)
    assert (links, workers, transpose) == (2, wk, tr)
    # the op's mux grouping check: the in0 sender axis forms exactly num_links groups of `workers`
    assert -(-(gx if transpose else gy) // workers) == links


def test_agmm_tuned_needs_a_wide_enough_device(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_AGMM_TUNED", "1")
    swiglu, k, n, kl = SHAPES["mlp_gateup"]
    assert tc.agmm_tuned_config(8192, _w(k, n), kl, swiglu, _Mesh(x=11)) is None  # harvested P150: 11 wide
    assert tc.agmm_tuned_config(2048, _w(k, n), kl, swiglu, _Mesh(x=11)) is not None  # the 11-wide entry fits
    swiglu, k, n, kl = SHAPES["gdn_out"]
    assert tc.agmm_tuned_config(8192, _w(k, n), kl, swiglu, _Mesh(x=11)) is not None  # the 11-wide entry fits
    assert tc.agmm_tuned_config(8192, _w(k, n), kl, swiglu, _Mesh(x=12, y=9)) is None  # no free mux row


def test_agmm_tuned_read_once(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_AGMM_TUNED", "1")
    assert tc.prefill_agmm_tuned_enabled() is True
    monkeypatch.setenv("QWEN36_PREFILL_AGMM_TUNED", "0")
    assert tc.prefill_agmm_tuned_enabled() is True


class _AgTensor:
    def __init__(self, s, k):
        self.shape = (1, 1, s, k)


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("swiglu", [False, True])
def test_agmm_call_kwargs(monkeypatch, flag, swiglu):
    """all_gather_matmul_prefill / all_gather_swiglu_prefill pass the tuned config, links, workers and transpose
    through; flag off passes today's (8, 9) grid, 4 workers, forced transpose."""
    monkeypatch.setenv("QWEN36_PREFILL_AGMM_TUNED", flag)
    seen = {}

    def fake_agmm(**kw):
        seen.update(kw)
        return ["OUT"]

    monkeypatch.setattr(ttnn.experimental, "all_gather_minimal_matmul_async", fake_agmm, raising=False)
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: t)
    tt_ccl = types.SimpleNamespace(mesh_device=_Mesh(), get_and_cycle_ag_semaphore_handles=lambda axis=None: "sem")
    if swiglu:
        out = tc.all_gather_swiglu_prefill(_AgTensor(128, 640), _w(5120, 4352), tt_ccl, "cfg", ttnn.Topology.Ring)
    else:
        out = tc.all_gather_matmul_prefill(_AgTensor(128, 640), _w(5120, 2048), tt_ccl, "cfg", ttnn.Topology.Ring)
    assert out == "OUT"
    g = seen["config"].compute_with_storage_grid_size
    if flag == "1":
        assert (g.x, g.y) == (12, 9) and seen["num_workers_per_link"] == 5 and seen["force_transpose"] is False
    else:
        assert (g.x, g.y) == (8, 9) and seen["num_workers_per_link"] == 4 and seen["force_transpose"] is True
    assert seen["num_links"] == 2


# --------------------------------------------------------------------------- #
# QWEN36_PREFILL_RS_WORKERS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", [None, "", "0", "-3", " "])
def test_rs_workers_default_is_today(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv("QWEN36_PREFILL_RS_WORKERS", raw)
    assert tc.prefill_rs_workers() is None
    assert tc.prefill_rs_workers(8192) is None
    assert tc.prefill_rs_kwargs(8192) == {}


def test_rs_workers_prefill_rows_only(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_RS_WORKERS", "8")
    assert tc.prefill_rs_workers(8192) == 8 and tc.prefill_rs_workers(33) == 8
    assert tc.prefill_rs_kwargs(2048) == {"num_workers_per_link": 8}
    # decode / verify widths keep tt_all_reduce's default
    assert tc.prefill_rs_workers(32) is None and tc.prefill_rs_kwargs(32) == {} and tc.prefill_rs_kwargs(1) == {}


def test_rs_workers_read_once_and_bad_value(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_RS_WORKERS", "8")
    assert tc.prefill_rs_workers(64) == 8
    monkeypatch.setenv("QWEN36_PREFILL_RS_WORKERS", "2")
    assert tc.prefill_rs_workers(64) == 8
    monkeypatch.setattr(tc, "_PREFILL_RS_WORKERS", None)
    monkeypatch.setenv("QWEN36_PREFILL_RS_WORKERS", "eight")
    with pytest.raises(ValueError):
        tc.prefill_rs_workers(64)


class _RsTensor:
    shape = (1, 1, 2048, 5120)

    def is_sharded(self):
        return False

    def deallocate(self, *_):
        pass


class _FakeCCL:
    def get_num_links(self, cluster_axis=None):
        return 1

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        return None

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return None


@pytest.mark.parametrize("workers", [None, "8"])
def test_tt_all_reduce_gets_the_workers(monkeypatch, workers):
    if workers is not None:
        monkeypatch.setenv("QWEN36_PREFILL_RS_WORKERS", workers)
    monkeypatch.setenv("QWEN36_PREFILL_RS_LINKS", "2")
    seen = {}
    monkeypatch.setattr(
        ttnn.experimental, "reduce_scatter_minimal_async", lambda t, **kw: seen.update(kw) or "R", raising=False
    )
    out = ccl.tt_all_reduce(
        _RsTensor(),
        types.SimpleNamespace(shape=(1, 8)),
        _FakeCCL(),
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=tc.prefill_rs_links(2048),
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **tc.prefill_rs_kwargs(2048),
    )
    assert out == "R" and seen["num_links"] == 2
    assert seen["num_workers_per_link"] == (2 if workers is None else 8)
    assert seen["chunks_per_sync"] == 10 and seen["num_buffers_per_channel"] == 2  # unchanged


# --------------------------------------------------------------------------- #
# QWEN36_PREFILL_MTP_LEAN: one on_chunk per chunk on an exact-multiple segment
# --------------------------------------------------------------------------- #
def test_mtp_lean_flag_default_off_read_once(monkeypatch):
    assert tc.prefill_mtp_lean_enabled() is False
    monkeypatch.setattr(tc, "_PREFILL_MTP_LEAN", None)
    monkeypatch.setenv("QWEN36_PREFILL_MTP_LEAN", "1")
    assert tc.prefill_mtp_lean_enabled() is True
    monkeypatch.setenv("QWEN36_PREFILL_MTP_LEAN", "0")
    assert tc.prefill_mtp_lean_enabled() is True


def _segment(calls, lean, skip_logits, num_full=2, chunk=2048):
    m = _fake_model(calls, chunk=chunk)

    def _chunks(prompt, pt_rec, b, nf, ch, tail, on_chunk=None, run_tail=True, start=0, **k):
        calls.append(("chunks", nf, tail, start))
        for c in range(nf):  # the real replay loop runs on_chunk after every chunk replay
            on_chunk("CHUNK_OUT", start + c * ch, ch)
        m._last_chunk = ("CHUNK_OUT", ch)

    m._prefill_traced_chunked_tp = _chunks
    e = _engine(2, model=m, _skip_logits_read=skip_logits)
    seen = []
    Tp = num_full * chunk
    prompt = torch.arange(1, Tp + 1, dtype=torch.int32).reshape(1, Tp)
    out = MTPSpecEngine._fast_segment(
        e, 1, prompt[0].tolist(), prompt, _row(), 0, Tp, True, False, lambda h, cs, n: seen.append((h, cs, n)), [False], True
    )
    return out, seen


@pytest.mark.parametrize("skip_logits", [False, True])
def test_exact_multiple_segment_today_warms_last_chunk_twice(monkeypatch, fake_ttnn, skip_logits):
    monkeypatch.setattr(tc, "_PREFILL_MTP_LEAN", False)
    calls = []
    out, seen = _segment(calls, lean=False, skip_logits=skip_logits)
    assert seen == [("CHUNK_OUT", 0, 2048), ("CHUNK_OUT", 2048, 2048), ("CHUNK_OUT", 2048, 2048)]
    assert [c for c in calls if c[0] == "last_logits"] == [("last_logits", "CHUNK_OUT", 2048, 2048)]


@pytest.mark.parametrize("skip_logits", [False, True])
def test_exact_multiple_segment_lean_warms_each_chunk_once(monkeypatch, fake_ttnn, skip_logits):
    monkeypatch.setattr(tc, "_PREFILL_MTP_LEAN", True)
    calls = []
    out, seen = _segment(calls, lean=True, skip_logits=skip_logits)
    assert seen == [("CHUNK_OUT", 0, 2048), ("CHUNK_OUT", 2048, 2048)]
    ll = [c for c in calls if c[0] == "last_logits"]
    if skip_logits:  # non-final vLLM chunk: no LM head, nothing read
        assert ll == [] and isinstance(out, torch.Tensor) and out.numel() == 0 and fake_ttnn["to_torch"] == []
    else:  # the final chunk still reads the same logits row
        assert ll == [("last_logits", "CHUNK_OUT", 2048, 2048)] and len(fake_ttnn["to_torch"]) == 1
