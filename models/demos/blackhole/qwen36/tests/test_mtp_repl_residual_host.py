# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of QWEN36_MTP_REPL_RESIDUAL (K5: the MTP drafter's decode forwards on the replicated RES40 residual;
tp_common block comment "QWEN36_MTP_REPL_RESIDUAL").

  * flag parsing (QWEN36_MTP_REPL_RESIDUAL / _ENORM / _FC_CORES / _FC_IBW) and the row-parallel fc program config;
  * the math the device path relies on: the device-major fc input order makes the per-device K-slice partials sum to
    the column-parallel fc exactly; the per-token stats table (hi + lo bf16) gives rms_norm_post_all_gather the full-dim
    mean of squares;
  * flag off: forward_decode issues today's op sequence (the LM-head branch moved to _draft_logits verbatim) and the
    drafter body keeps its anchor copy;
  * flag on: _init_repl_residual's weights / routing, _forward_decode_repl's op sequence for fractured / replicated
    hidden, table / gather embedding norm, need_logits True / False, kv_only, and the drafter body's single feed gather.

Needs ttnn importable (no device: every ttnn op used here is monkeypatched).

  pytest -q models/demos/blackhole/qwen36/tests/test_mtp_repl_residual_host.py
"""
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import mtp as mtp_mod
from models.demos.blackhole.qwen36.tt import spec_engine as se
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

_FLAGS = ("QWEN36_MTP_REPL_RESIDUAL", "QWEN36_MTP_REPL_ENORM", "QWEN36_MTP_REPL_FC_CORES", "QWEN36_MTP_REPL_FC_IBW")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in _FLAGS:
        monkeypatch.delenv(f, raising=False)
    yield


# ------------------------------------------------------------------------------------------------------------------ #
# flag parsing + program config
# ------------------------------------------------------------------------------------------------------------------ #
def test_flags_default_off(monkeypatch):
    assert tpc.mtp_repl_residual_enabled() is False
    assert tpc.mtp_repl_enorm_mode() == "table"
    monkeypatch.setenv("QWEN36_MTP_REPL_RESIDUAL", "1")
    assert tpc.mtp_repl_residual_enabled() is True
    monkeypatch.setenv("QWEN36_MTP_REPL_RESIDUAL", "0")
    assert tpc.mtp_repl_residual_enabled() is False


@pytest.mark.parametrize("raw,want", [("table", "table"), ("gather", "gather"), (" Gather ", "gather"), ("", "table")])
def test_enorm_mode(monkeypatch, raw, want):
    monkeypatch.setenv("QWEN36_MTP_REPL_ENORM", raw)
    assert tpc.mtp_repl_enorm_mode() == want


@pytest.mark.parametrize("raw", ["stats", "1", "allgather"])
def test_enorm_mode_bad(monkeypatch, raw):
    monkeypatch.setenv("QWEN36_MTP_REPL_ENORM", raw)
    with pytest.raises(ValueError):
        tpc.mtp_repl_enorm_mode()


def test_fc_progcfg_defaults_and_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(tpc, "create_matmul_1d_decode_progcfg", lambda *a, **k: seen.append((a, k)) or "PC")
    assert tpc.mtp_repl_fc_progcfg(1280, 5120, 11) == "PC"
    assert seen[-1] == ((32, 1280, 5120), {"num_cores": 84, "grid_w": 11, "max_in0_block_w": 40})
    monkeypatch.setenv("QWEN36_MTP_REPL_FC_CORES", "40")
    monkeypatch.setenv("QWEN36_MTP_REPL_FC_IBW", "20")
    tpc.mtp_repl_fc_progcfg(1280, 5120, 11)
    assert seen[-1] == ((32, 1280, 5120), {"num_cores": 40, "grid_w": 11, "max_in0_block_w": 20})
    monkeypatch.setenv("QWEN36_MTP_REPL_FC_IBW", "0")
    with pytest.raises(ValueError):
        tpc.mtp_repl_fc_progcfg(1280, 5120, 11)


def test_fc_progcfg_real_config_shape():
    """Default on the BH grid (11 wide): 11x8 grid, 2 output tile columns per core (80 active), the whole K (40 tiles)
    in one in0 block."""
    pc = tpc.mtp_repl_fc_progcfg(1280, 5120, 11)
    assert pc.in0_block_w == 40 and pc.per_core_M == 1 and pc.per_core_N == 2
    grid = pc.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (11, 8)
    assert pc.mcast_in0


# ------------------------------------------------------------------------------------------------------------------ #
# the math: row-parallel fc input order, per-token stats table
# ------------------------------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("dim,tp", [(64, 8), (5120, 8), (96, 4), (8, 2)])
def test_fc_rp_order_is_a_device_major_permutation(dim, tp):
    perm = tpc.mtp_fc_rp_input_order(dim, tp)
    assert perm.shape == (2 * dim,)
    assert torch.equal(torch.sort(perm).values, torch.arange(2 * dim))
    s = dim // tp
    for d in range(tp):
        blk = perm[d * 2 * s : (d + 1) * 2 * s]
        assert torch.equal(blk[:s], torch.arange(d * s, (d + 1) * s))  # embedding span d
        assert torch.equal(blk[s:], torch.arange(dim + d * s, dim + (d + 1) * s))  # hidden span d


@pytest.mark.parametrize("dim,tp,rows", [(64, 8, 8), (96, 4, 32), (40, 2, 3)])
def test_fc_rp_partials_sum_to_the_column_parallel_fc(dim, tp, rows):
    """The device math: device d multiplies its [e span d | h span d] by its K-row block of fc_w[:, perm].T; the
    all-reduce sums the partials. Exact in fp64 (only the summation order differs)."""
    g = torch.Generator().manual_seed(0)
    fc_w = torch.randn(dim, 2 * dim, generator=g, dtype=torch.float64)
    e = torch.randn(rows, dim, generator=g, dtype=torch.float64)
    h = torch.randn(rows, dim, generator=g, dtype=torch.float64)
    full = torch.cat([e, h], -1) @ fc_w.T  # today's math (any parallel layout)
    perm = tpc.mtp_fc_rp_input_order(dim, tp)
    w_rp = fc_w[:, perm].T.contiguous()  # what _init_repl_residual's preprocess hands ShardTensorToMesh(dim=0)
    s = dim // tp
    blocks = torch.chunk(w_rp, tp, dim=0)  # ShardTensorToMesh(dim=0): device d's row block
    total = torch.zeros_like(full)
    for d in range(tp):
        x_d = torch.cat([e[:, d * s : (d + 1) * s], h[:, d * s : (d + 1) * s]], -1)  # concat([en_d, hn_d])
        total += x_d @ blocks[d]
    assert torch.allclose(total, full, rtol=1e-12, atol=1e-10)


def test_enorm_stats_table_hi_lo_and_chunking():
    g = torch.Generator().manual_seed(1)
    vocab, dim = 1000, 512  # vocab not a multiple of the chunk: the last chunk is short
    emb = (torch.randn(vocab, dim, generator=g) * torch.rand(vocab, 1, generator=g) * 0.05).to(torch.bfloat16)
    emb[7] = 0  # an all-zero row (unused token) stays finite
    tbl = tpc.mtp_enorm_stats_table(emb, chunk=300)
    assert tbl.shape == (vocab, tpc.MTP_ENORM_STATS_W) and tbl.dtype == torch.bfloat16
    assert torch.count_nonzero(tbl[:, 2:]) == 0
    want = emb.double().square().mean(-1)  # the full-dim mean of squares
    got = tbl[:, 0].double() + tbl[:, 1].double()
    rel = ((got - want).abs() / want.clamp_min(1e-30))[want > 0]
    assert float(rel.max()) < 2.0**-15  # one bf16 alone: up to 2^-9
    assert float(got[7]) == 0.0
    # hi alone is the bf16 rounding of the value; lo carries the remainder
    assert torch.equal(tbl[:, 0], want.to(torch.bfloat16))


def test_enorm_stats_table_matches_the_device_dtype():
    """The table is built from the bf16 values the device stores, whatever dtype the checkpoint tensor has."""
    g = torch.Generator().manual_seed(2)
    emb32 = torch.randn(64, 128, generator=g) * 0.03
    assert torch.equal(tpc.mtp_enorm_stats_table(emb32), tpc.mtp_enorm_stats_table(emb32.to(torch.bfloat16)))


def test_sharded_post_all_gather_semantics_give_the_full_dim_rmsnorm():
    """The sharded rms_norm_post_all_gather reduces its one-core stats tile over the whole row and scales by
    1/(number of stats tiles) = 1 (it reads a mean, the pre-all-gather stage's per-device E[x^2]): with the table row
    it must reproduce the gathered full-row RMSNorm on every device's span (ttsim-perf confirmed this scaling on the
    device kernels: max 4.9 bf16 ulps vs fp64, the same as the interleaved op; sum/tp was off by sqrt(8))."""
    g = torch.Generator().manual_seed(3)
    dim, tp, eps = 5120, 8, 1e-6
    e = (torch.randn(4, dim, generator=g) * 0.02).to(torch.bfloat16)
    gamma = 1.0 + 0.1 * torch.randn(dim, generator=g)
    tbl = tpc.mtp_enorm_stats_table(e).double()
    s = dim // tp
    full = e.double() * torch.rsqrt(e.double().square().mean(-1, keepdim=True) + eps) * gamma.double()
    mean = tbl.sum(-1, keepdim=True) * 1.0  # REDUCE_ROW over the stats tile x cinv = 1 / (1 stats tile)
    for d in range(tp):
        y_d = e[:, d * s : (d + 1) * s].double() * torch.rsqrt(mean + eps) * gamma[d * s : (d + 1) * s].double()
        assert torch.allclose(y_d, full[:, d * s : (d + 1) * s], rtol=2e-5, atol=1e-7)


def test_enorm_sharded_configs():
    x_mem, st_mem, pc = tpc.mtp_enorm_sharded_configs(640)
    assert x_mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and x_mem.buffer_type == ttnn.BufferType.L1
    assert list(x_mem.shard_spec.shape) == [32, 64] and x_mem.shard_spec.grid.num_cores() == tpc.MTP_ENORM_CORES == 10
    assert list(st_mem.shard_spec.shape) == [32, 32] and st_mem.shard_spec.grid.num_cores() == 1
    assert pc.block_w == 2 and pc.subblock_w == 2 and pc.block_h == 1
    grid = pc.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (10, 1)
    for bad in (64, 600, 320 + 32):
        assert tpc.mtp_enorm_sharded_configs(bad) is None


# ------------------------------------------------------------------------------------------------------------------ #
# fake device tensors and a recording ttnn
# ------------------------------------------------------------------------------------------------------------------ #
class _T:
    """Fake device tensor: shape, dtype, name, memory config."""

    def __init__(self, shape, name="t", dtype=None, mem=None):
        self.shape = tuple(shape)
        self.name = name
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self._mem = mem

    def memory_config(self):
        return self._mem

    def __repr__(self):
        return f"<{self.name}{list(self.shape)}>"


RES_MEM = "RES40"


def _memtag(mem):
    if isinstance(mem, str):
        return {RES_MEM: "res"}.get(mem, mem.lower())
    return "dram" if mem == ttnn.DRAM_MEMORY_CONFIG else "l1"


class _Rec:
    """Records (op, info) pairs; ``freed`` lists the names of deallocated tensors."""

    def __init__(self):
        self.log = []
        self.freed = []

    def ops(self):
        return [o for o, _ in self.log]


def _patch_ttnn(monkeypatch, rec):
    def dealloc(t, *a, **k):
        if t is not None:
            rec.freed.append(getattr(t, "name", repr(t)))

    def reshape(t, shape, padded=None):
        rec.log.append(("reshape", (t.name, tuple(shape))))
        return _T(tuple(shape), f"{t.name}.view")

    def to_mem(t, mem, *a, **k):
        rec.log.append(("to_memory_config", (t.name, mem)))
        return _T(t.shape, f"{t.name}@{_memtag(mem)}", mem=mem)

    def mesh_partition(t, dim, cluster_axis=None, memory_config=None):
        rec.log.append(("mesh_partition", (t.name, dim, cluster_axis)))
        return _T(t.shape[:-1] + (t.shape[-1] // 8,), f"{t.name}.span")

    def concat(ts, dim, memory_config=None):
        rec.log.append(("concat", (tuple(t.name for t in ts), dim, memory_config)))
        return _T(ts[0].shape[:-1] + (sum(t.shape[-1] for t in ts),), "cat")

    def linear(a, b, **kw):
        rec.log.append(("linear", (a.name, b, kw.get("program_config"), kw.get("memory_config"))))
        return _T(a.shape[:-1] + (5120,), "partial")

    def embedding(ids, w, layout=None, memory_config=None):
        rec.log.append(("embedding", (ids.name, w, layout)))
        return _T((ids.shape[0], ids.shape[1], 32), "st_rm")

    def pad(t, p, value=None):
        rec.log.append(("pad", (t.name, tuple(map(tuple, p)), value)))
        return _T((1, 1, 32, t.shape[-1]), "st_pad")

    def to_layout(t, layout, memory_config=None):
        rec.log.append(("to_layout", (t.name, layout)))
        return _T(t.shape, "st_tile")

    def post_ag(x, stats=None, epsilon=None, weight=None, program_config=None, compute_kernel_config=None,
                memory_config=None):
        rec.log.append(("rms_norm_post_all_gather", (x.name, stats.name, epsilon, weight, program_config)))
        return _T(x.shape, "en_sh")

    def ag(x, **kw):
        rec.log.append(("all_gather_async", (x.name, kw["dim"], kw["memory_config"], kw["num_links"])))
        return _T(x.shape[:-1] + (x.shape[-1] * 8,), f"{x.name}.gathered")

    def matmul(a, b, compute_kernel_config=None, **kw):
        rec.log.append(("matmul", (a.name, b.name)))
        return _T(a.shape[:-1] + (b.shape[-1],), "Hp")

    def copy(a, b):
        rec.log.append(("copy", (a.name, b.name)))

    monkeypatch.setattr(ttnn, "deallocate", dealloc)
    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn, "to_memory_config", to_mem)
    monkeypatch.setattr(ttnn, "mesh_partition", mesh_partition)
    monkeypatch.setattr(ttnn, "concat", concat)
    monkeypatch.setattr(ttnn, "linear", linear)
    monkeypatch.setattr(ttnn, "embedding", embedding)
    monkeypatch.setattr(ttnn, "pad", pad)
    monkeypatch.setattr(ttnn, "to_layout", to_layout)
    monkeypatch.setattr(ttnn, "rms_norm_post_all_gather", post_ag)
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", ag, raising=False)
    monkeypatch.setattr(ttnn, "matmul", matmul)
    monkeypatch.setattr(ttnn, "copy", copy)


class _FakeDAR:
    res_mem = RES_MEM
    num_links = 2
    topology = "ring"

    def __init__(self, rec):
        self.rec = rec

    def rms_norm(self, x, norm):
        self.rec.log.append(("dar.rms_norm", (x.name, norm.tag)))
        return _T(x.shape, f"norm_{norm.tag}", mem=RES_MEM)

    def final_norm(self, x, norm, output_memory_config=None):
        self.rec.log.append(("dar.final_norm", (x.name, norm.tag)))
        return _T(x.shape, "normed")

    def disarm(self):
        self.rec.log.append(("dar.disarm", ()))

    def all_reduce(self, partial, keep_fp32=False):
        self.rec.log.append(("dar.all_reduce", (partial.name,)))
        self.rec.freed.append(partial.name)  # the real all_reduce frees its partial
        return _T(partial.shape, "x0", mem=RES_MEM)


class _CCL:
    def get_and_cycle_ag_semaphore_handles(self):
        return "ag_sem"

    def get_and_cycle_barrier_semaphore_handle(self):
        return "bar_sem"


def _norm(tag):
    return types.SimpleNamespace(
        norm=types.SimpleNamespace(
            tag=tag, eps=1e-6, weight=f"{tag}.w", weight_distributed=f"{tag}.wd", compute_kernel_config_hifi2="HIFI2"
        )
    )


class _Decoder:
    def __init__(self, rec):
        self.rec = rec

    def forward(self, x, **kw):
        self.rec.log.append(("decoder.forward", (x.name, kw.get("mode"), kw.get("kv_only", False), kw["n_users"])))
        return None if kw.get("kv_only") else _T(x.shape, "block_out", mem=RES_MEM)


def _repl_head(monkeypatch, enorm="table", kv_only=False):
    """A bare MTP head set up like _init_repl_residual leaves it (no device)."""
    rec = _Rec()
    _patch_ttnn(monkeypatch, rec)
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 8
    m.args = types.SimpleNamespace(dim=5120)
    m.tt_ccl = _CCL()
    m._repl_dar = _FakeDAR(rec)
    m._repl_enorm = enorm
    m._enorm_stats = "STATS"
    m._enorm_cfg = ("XMEM", "STMEM", "ENORM_PC")
    m._fc_rp = "FC_RP"
    m._fc_rp_pc = "FC_RP_PC"
    m._fc_compute_cfg = "HIFI2"
    m.pre_fc_norm_embedding = _norm("enorm")
    m.pre_fc_norm_hidden = _norm("hnorm")
    m.head_norm = _norm("head")
    m.kv_only_maint = kv_only
    m.embd = lambda ids: rec.log.append(("embd", ids.name)) or _T((ids.shape[0], 1, 640), "emb")
    m.decoder = _Decoder(rec)
    m.repl_residual = True
    m._draft_logits = lambda normed, sharded: rec.log.append(("draft_logits", (normed.name, sharded))) or _T(
        (1, 1, 8, 31040), "logits"
    )
    return m, rec



# ------------------------------------------------------------------------------------------------------------------ #
# flag on: _forward_decode_repl
# ------------------------------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("fractured", [True, False])
def test_repl_step_table_op_sequence(monkeypatch, fractured):
    m, rec = _repl_head(monkeypatch, "table")
    hid = _T((1, 1, 8, 640 if fractured else 5120), "hid")
    tok = _T((8, 1), "tok", dtype=ttnn.uint32)
    logits, nxt = m._forward_decode_repl(hid, tok, "pos", "cos", "sin", "pt")
    h = "hid.gathered" if fractured else "hid"
    want = []
    if fractured:  # one exact gather of a fractured hidden; a replicated one is used as is
        want.append(("all_gather_async", ("hid", 3, ttnn.DRAM_MEMORY_CONFIG, 2)))
    want += [
        ("embd", "tok"),
        ("reshape", ("emb", (1, 1, 8, 640))),
        # hidden half: RES40 reshard -> local norm with the full gamma -> DRAM -> this device's span
        ("to_memory_config", (h, RES_MEM)),
        ("dar.rms_norm", (f"{h}@res", "hnorm")),
        ("to_memory_config", ("norm_hnorm", ttnn.DRAM_MEMORY_CONFIG)),
        ("mesh_partition", ("norm_hnorm@dram", 3, 1)),
        # embedding half (table): stats lookup (row-major, view, pad to a tile, tilize, view) + post-all-gather norm
        ("embedding", ("tok", "STATS", ttnn.ROW_MAJOR_LAYOUT)),
        ("reshape", ("st_rm", (1, 1, 8, 32))),
        ("pad", ("st_rm.view", ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0)),
        ("to_layout", ("st_pad", ttnn.TILE_LAYOUT)),
        ("reshape", ("st_tile", (1, 1, 8, 32))),
        ("to_memory_config", ("st_tile.view", "STMEM")),  # stats tile -> one core
        ("to_memory_config", ("emb.view", "XMEM")),  # this device's rows -> width-sharded over 10 cores
        ("rms_norm_post_all_gather", ("emb.view@xmem", "st_tile.view@stmem", 1e-6, "enorm.wd", "ENORM_PC")),
        ("to_memory_config", ("en_sh", ttnn.DRAM_MEMORY_CONFIG)),
        # [e span | h span] in L1 -> row-parallel fc -> ONE all-reduce -> replicated decoder -> head norm -> LM head
        ("concat", (("en_sh@dram", "norm_hnorm@dram.span"), -1, ttnn.L1_MEMORY_CONFIG)),
        ("linear", ("cat", "FC_RP", "FC_RP_PC", ttnn.L1_MEMORY_CONFIG)),
        ("dar.disarm", ()),  # nothing may be fused into the fc all-reduce
        ("dar.all_reduce", ("partial",)),
        ("decoder.forward", ("x0", "decode", False, 1)),
        ("dar.final_norm", ("block_out", "head")),
        ("draft_logits", ("normed", False)),
    ]
    assert rec.log == want
    assert logits.name == "logits" and nxt.name == "normed"  # the replicated head-norm output is the chain value
    # ownership: caller tensors and the returned chain value are never freed; every intermediate is
    for name in ("hid", "tok", "normed"):
        assert name not in rec.freed, name
    must = ["emb.view", "cat", "partial", "x0", "block_out", "en_sh@dram", "norm_hnorm@dram.span", "st_tile.view"]
    must += ["st_tile.view@stmem", "emb.view@xmem", "en_sh"]
    must += ["hid.gathered"] if fractured else []
    for name in must:
        assert name in rec.freed, name
    ccl = [o for o in rec.ops() if o in ("all_gather_async", "dar.all_reduce")]
    assert len(ccl) == (2 if fractured else 1)  # vs 8 AG/RS on the fractured path (5 norm gathers, 2 RS, + argmax)


def test_repl_step_gather_enorm(monkeypatch):
    m, rec = _repl_head(monkeypatch, "gather")
    hid = _T((1, 1, 8, 5120), "hid")
    m._forward_decode_repl(hid, _T((8, 1), "tok", dtype=ttnn.uint32), "pos", "cos", "sin", "pt")
    ops = rec.ops()
    assert "embedding" not in ops and "rms_norm_post_all_gather" not in ops
    # the embedding is gathered (one CCL), normed on RES40 with the full gamma, and sliced back to this device's span
    g = [e for e in rec.log if e[0] == "all_gather_async"]
    assert g == [("all_gather_async", ("emb.view", 3, ttnn.DRAM_MEMORY_CONFIG, 2))]
    norms = [e[1][1] for e in rec.log if e[0] == "dar.rms_norm"]
    assert norms == ["hnorm", "enorm"]
    assert ops.count("mesh_partition") == 2
    assert ops.count("dar.all_reduce") == 1


@pytest.mark.parametrize("kv_only", [True, False])
def test_repl_maintenance_step(monkeypatch, kv_only):
    """need_logits=False: no head norm / LM head; kv_only returns (None, None), otherwise the raw replicated block
    output (callers drop it)."""
    m, rec = _repl_head(monkeypatch, "table", kv_only=kv_only)
    hid = _T((1, 1, 32, 640), "feed")
    out = m._forward_decode_repl(
        hid, _T((32, 1), "rs_tok", dtype=ttnn.uint32), "p", "c", "s", "pt", need_logits=False, alias_kv_write=True,
        spec_n_users=8,
    )
    ops = rec.ops()
    assert "dar.final_norm" not in ops and "draft_logits" not in ops
    dec = [e for e in rec.log if e[0] == "decoder.forward"]
    assert dec == [("decoder.forward", ("x0", "decode", kv_only, 8))]
    assert "pad" not in ops  # 32 rows: whole tiles, no pad
    if kv_only:
        assert out == (None, None)
    else:
        assert out[0] is None and out[1].name == "block_out"
    assert "feed" not in rec.freed and "feed.gathered" in rec.freed


@pytest.mark.parametrize("width", [640, 5120])
def test_repl_wide_maintenance_forward_keeps_the_fractured_path(monkeypatch, width):
    """More than one tile of rows (an adaptive-K reseed of 64 rows) does not fit RES40: the class method runs on the
    fractured hidden (a replicated one is sliced back to this device's span first, and that slice is freed)."""
    m, rec = _repl_head(monkeypatch, "table")
    seen = {}

    def fractured(self, h, *a, **kw):
        seen["h"], seen["kw"] = h, kw
        return None, None

    monkeypatch.setattr(Qwen36MTP, "forward_decode", fractured)
    hid = _T((1, 1, 64, width), "feed")
    out = m._forward_decode_repl(hid, _T((64, 1), "tok", dtype=ttnn.uint32), "p", "c", "s", "pt", need_logits=False,
                                 alias_kv_write=True, spec_n_users=8)
    assert out == (None, None)
    assert seen["kw"]["need_logits"] is False and seen["kw"]["spec_n_users"] == 8 and seen["kw"]["alias_kv_write"]
    if width == 640:
        assert seen["h"] is hid and rec.ops() == []
    else:
        assert seen["h"].name == "feed.span" and rec.ops() == ["mesh_partition"] and rec.freed == ["feed.span"]
    assert "embedding" not in rec.ops() and "dar.all_reduce" not in rec.ops()


def test_installed_route_and_flag_off_class_method(monkeypatch):
    """_init_repl_residual installs _forward_decode_repl over forward_decode on the instance only."""
    m, _ = _repl_head(monkeypatch)
    assert "forward_decode" not in m.__dict__
    m.forward_decode = m._forward_decode_repl
    assert m.forward_decode.__func__ is Qwen36MTP._forward_decode_repl
    assert Qwen36MTP.forward_decode is not Qwen36MTP._forward_decode_repl
    bare = Qwen36MTP.__new__(Qwen36MTP)
    assert bare.forward_decode.__func__ is Qwen36MTP.forward_decode


# ------------------------------------------------------------------------------------------------------------------ #
# flag off: forward_decode's op sequence is today's (the _draft_logits move is verbatim)
# ------------------------------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("sharded_argmax,sharded_lm_head,ondev", [(True, False, False), (False, True, False), (False, False, True), (False, False, False)])
def test_flag_off_forward_decode_sequence(monkeypatch, sharded_argmax, sharded_lm_head, ondev):
    rec = _Rec()
    _patch_ttnn(monkeypatch, rec)
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 8
    m.args = types.SimpleNamespace(get_norm_config=lambda key, mode: {"output_mem_config": None, "key": key})
    m.embd = lambda ids: rec.log.append(("embd", ids.name)) or _T((ids.shape[0], 1, 640), "emb")
    m._fuse = lambda e, h, mode: rec.log.append(("fuse", (e.name, h.name))) or _T(e.shape, "fused")
    m.decoder = _Decoder(rec)
    m.head_norm = lambda x, mode=None, norm_config=None: rec.log.append(("head_norm", (x.name, norm_config["key"]))) or _T(
        x.shape[:-1] + (5120,), "normed"
    )
    m._sharded_argmax = sharded_argmax
    if ondev:
        m._ondev_argmax = True
    m.draft_lm_head_weight = "W"
    m._fp32_cfg = "FP32"
    m._lm_head_linear = lambda x, weight=None, **kw: rec.log.append(("lm_head_linear", (x.name, weight, tuple(sorted(kw))))) or _T(
        (1, 1, 8, 31040), "logits"
    )
    m._lm_head = lambda x, out_dtype=None, weight=None: rec.log.append(("lm_head", (x.name, out_dtype, weight))) or _T(
        (1, 1, 8, 248320), "logits"
    )
    hid = _T((1, 1, 8, 640), "hid")
    logits, nxt = m.forward_decode(hid, _T((8, 1), "tok", dtype=ttnn.uint32), "pos", "cos", "sin", "pt", sharded_lm_head)
    head = (
        ("lm_head_linear", ("normed", "W", ("compute_kernel_config", "dtype")))
        if sharded_argmax and not sharded_lm_head
        else ("lm_head_linear", ("normed", "W", ()))
        if (sharded_lm_head or ondev)
        else ("lm_head", ("normed", ttnn.float32, "W"))
    )
    assert rec.log == [
        ("embd", "tok"),
        ("reshape", ("emb", (1, 1, 8, 640))),
        ("fuse", ("emb.view", "hid")),
        ("decoder.forward", ("fused", "decode", False, 1)),
        ("head_norm", ("block_out", "lm_head")),
        ("mesh_partition", ("normed", 3, 1)),
        head,
    ]
    assert nxt.name == "normed.span" and logits.name == "logits"
    # freed in today's order: embedding view, fused, the raw block output, then normed after the LM head
    assert rec.freed == ["emb.view", "fused", "block_out", "normed"]
    assert "all_gather_async" not in rec.ops()


# ------------------------------------------------------------------------------------------------------------------ #
# _init_repl_residual
# ------------------------------------------------------------------------------------------------------------------ #
def _init_head(monkeypatch, dar):
    seen = {}
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 8
    m.mesh_device = "mesh"
    m.tt_ccl = "ccl"
    m.args = types.SimpleNamespace(dim=64, decode_grid_w=11, get_state_dict_prefix=lambda a, b: "")
    monkeypatch.setattr(tpc, "get_decode_allreduce", lambda args, ccl: dar)
    monkeypatch.setattr(tpc, "mtp_repl_fc_progcfg", lambda k, n, gw: ("PC", k, n, gw))
    monkeypatch.setattr(tpc, "mtp_enorm_sharded_configs", lambda width: ("X", "S", ("PC", width)))

    def as_tensor(t, preprocess=None, **kw):
        seen["fc_rp"] = (preprocess(t), kw)
        return "FC_RP"

    def from_torch(t, **kw):
        seen["stats"] = (t, kw)
        return "STATS"

    monkeypatch.setattr(ttnn, "as_tensor", as_tensor)
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda mesh, dim: ("shard", dim))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "rep")
    g = torch.Generator().manual_seed(4)
    sd = {
        "mtp.fc.weight": torch.randn(64, 128, generator=g).to(torch.bfloat16),
        "tok_embeddings.weight": torch.randn(50, 64, generator=g).to(torch.bfloat16),
    }
    return m, sd, seen


def test_init_without_decode_allreduce_stays_off(monkeypatch):
    m, sd, seen = _init_head(monkeypatch, None)
    m._init_repl_residual(sd, None)
    assert not getattr(m, "repl_residual", False)
    assert "forward_decode" not in m.__dict__ and not seen


@pytest.mark.parametrize("enorm", ["table", "gather"])
def test_init_builds_weights_and_routes(monkeypatch, tmp_path, enorm):
    monkeypatch.setenv("QWEN36_MTP_REPL_ENORM", enorm)
    dar = types.SimpleNamespace(name="DAR")
    m, sd, seen = _init_head(monkeypatch, dar)
    m._init_repl_residual(sd, tmp_path)
    assert m.repl_residual is True and m._repl_dar is dar and m._repl_enorm == enorm
    assert m.__dict__["forward_decode"].__func__ is Qwen36MTP._forward_decode_repl
    w, kw = seen["fc_rp"]
    perm = tpc.mtp_fc_rp_input_order(64, 8)
    assert torch.equal(w, sd["mtp.fc.weight"][:, perm].T.contiguous())
    assert kw["mesh_mapper"] == ("shard", 0) and kw["dtype"] == ttnn.bfloat8_b
    assert kw["cache_file_name"] == str(tmp_path / "fc_rp_tp8")
    assert m._fc_rp_pc == ("PC", 16, 64, 11)
    if enorm == "table":
        t, kw2 = seen["stats"]
        assert t.shape == (1, 1, 50, tpc.MTP_ENORM_STATS_W) and kw2["dtype"] == ttnn.bfloat16
        assert kw2["layout"] == ttnn.ROW_MAJOR_LAYOUT and kw2["mesh_mapper"] == "rep"
        assert torch.equal(t[0, 0], tpc.mtp_enorm_stats_table(sd["tok_embeddings.weight"]))
        assert m._enorm_stats == "STATS" and m._enorm_cfg == ("X", "S", ("PC", 8))
    else:
        assert "stats" not in seen and m._enorm_stats is None and m._enorm_cfg is None


def test_init_table_falls_back_to_gather_without_a_sharded_config(monkeypatch, tmp_path):
    dar = types.SimpleNamespace(name="DAR")
    m, sd, seen = _init_head(monkeypatch, dar)
    monkeypatch.setattr(tpc, "mtp_enorm_sharded_configs", lambda width: None)
    m._init_repl_residual(sd, tmp_path)
    assert m.repl_residual is True and m._repl_enorm == "gather" and m._enorm_stats is None and "stats" not in seen


# ------------------------------------------------------------------------------------------------------------------ #
# spec_engine._drafter_body: one feed gather, step-0 anchor from it (flag on); today's anchor copy (flag off)
# ------------------------------------------------------------------------------------------------------------------ #
class _Mtp:
    def __init__(self, rec, repl):
        self.rec = rec
        self.repl_residual = repl

    def replicate_hidden(self, x):
        self.rec.log.append(("replicate_hidden", x.name))
        return _T(x.shape[:-1] + (5120,), f"{x.name}.rep")

    def forward_decode(self, h, tok, pos, cos, sin, pt, need_logits=True, **kw):
        self.rec.log.append(("forward_decode", (h.name, need_logits)))
        if not need_logits:
            return None, None
        return _T((1, 1, 8, 31040), "logits"), _T((1, 1, 8, 5120), f"hn{len(self.rec.log)}")


def _engine(rec, repl, K=3):
    eng = types.SimpleNamespace(B=8, T=4, K=K, nb=4, _anchor_cc="CC", _dr_rs_pt="RSPT", _mtp_pt="MPT")
    eng.mtp = _Mtp(rec, repl)
    eng._hp_buf = _T((1, 1, 8, 640), "hp")
    d = dict(
        anchor_sel=_T((1, 1, 8, 32), "sel"),
        rs_tok="rs_tok",
        rs_pos="rs_pos",
        rs_cos="rs_cos",
        rs_sin="rs_sin",
        tok0=_T((8, 1), "tok0"),
        pos=["p"] * K,
        cos=["c"] * K,
        sin=["s"] * K,
        owned=[],
    )
    eng._dr_inputs = lambda b=None: d
    return eng


def test_drafter_body_wide_feed_keeps_the_anchor_copy(monkeypatch):
    """A feed wider than one tile of rows (adaptive K, 8 users x T=8) keeps today's anchor copy and fractured reseed
    even with the flag on: RES40 holds one tile of rows."""
    rec = _Rec()
    _patch_ttnn(monkeypatch, rec)
    monkeypatch.setattr(se, "draft_argmax", lambda mtp, logits: _T((1, 1, 8, 1), "idx"))
    eng = _engine(rec, True)
    se.MTPSpecEngine._drafter_body(eng, _T((1, 1, 64, 640), "feed"))
    assert "replicate_hidden" not in rec.ops()
    assert rec.log[:2] == [("matmul", ("sel", "feed")), ("copy", ("Hp", "hp"))]


@pytest.mark.parametrize("repl", [False, True])
def test_drafter_body(monkeypatch, repl):
    rec = _Rec()
    _patch_ttnn(monkeypatch, rec)
    monkeypatch.setattr(se, "draft_argmax", lambda mtp, logits: _T((1, 1, 8, 1), "idx"))
    eng = _engine(rec, repl)
    feed = _T((1, 1, 32, 640), "feed")
    outs = se.MTPSpecEngine._drafter_body(eng, feed)
    assert len(outs) == 3
    ops = rec.ops()
    fds = [e[1] for e in rec.log if e[0] == "forward_decode"]
    if repl:
        assert ops[:2] == ["replicate_hidden", "matmul"]
        assert rec.log[1] == ("matmul", ("sel", "feed.rep"))
        assert "copy" not in ops
        assert fds[0] == ("feed.rep", False)  # the reseed reads the gathered feed
        assert fds[1] == ("Hp", True)  # step 0 reads the replicated anchor rows
        assert "Hp" in rec.freed and "feed.rep" in rec.freed and "hp" not in rec.freed and "feed" not in rec.freed
    else:
        assert "replicate_hidden" not in ops
        assert rec.log[:2] == [("matmul", ("sel", "feed")), ("copy", ("Hp", "hp"))]
        assert fds[0] == ("feed", False)
        assert fds[1] == ("hp", True)
        assert "hp" not in rec.freed and "Hp" in rec.freed
    assert [f[1] for f in fds] == [False, True, True, True]
