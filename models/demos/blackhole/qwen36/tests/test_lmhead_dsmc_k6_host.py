# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K6 knobs of the DRAM-sharded LM head (QWEN36_LMHEAD_DSHARD=mc), host tests (no device; ttnn ops that would need
one are monkeypatched):

  QWEN36_LMHEAD_DSHARD_SCOPE      all (default) | draft | base: which head calls take the DRAM-sharded copy
  QWEN36_LMHEAD_DSHARD_OUT_PCN    logits tiles per core of the mc output (default: the 122-tile bank shard)
  QWEN36_LMHEAD_DSHARD_IBW        mc in0_block_w (default 2)
  QWEN36_LMHEAD_DSHARD_FP32_DEST  bf16 base calls keep each pass's whole K in fp32 Dest (default 0)

Checks: every knob defaults to the r5 behaviour (same plan, same program config, same compute config, same routing);
the mc layout / L1 model reproduces create_program_dram_sharded_multicore_spec's dataflow buffers for the Qwen3.8 TP8
head (K 160 x N 970 tiles, 8 banks x 122); the L1 check passes the production layouts and rejects the ones that would
clash at warm-up; the scope routes the drafter (explicit weight) and base (weight None) calls; the K6 knobs are
rejected for "exact".

Run: pytest models/demos/blackhole/qwen36/tests/test_lmhead_dsmc_k6_host.py -q
"""
import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

K = 5120
N_TP8 = 248320 // 8  # 31040 = 970 tiles
KB = 1024
K6_ENVS = (
    tpc.LMHEAD_DSHARD_SCOPE_ENV,
    tpc.LMHEAD_DSHARD_OUT_PCN_ENV,
    tpc.LMHEAD_DSHARD_IBW_ENV,
    tpc.LMHEAD_DSHARD_FP32_DEST_ENV,
)


@pytest.fixture
def clean_env(monkeypatch):
    extra = (
        tpc.LMHEAD_DSHARD_ENV,
        tpc.LMHEAD_DSHARD_READERS_ENV,
        "QWEN36_WEIGHT_PREFETCH",
        "QWEN36_WEIGHT_PREFETCH_GCB_KB",
    )
    for name in K6_ENVS + extra:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ------------------------------------------------------------------------------------------------ flags


def test_flag_names_and_defaults(clean_env):
    assert K6_ENVS == (
        "QWEN36_LMHEAD_DSHARD_SCOPE",
        "QWEN36_LMHEAD_DSHARD_OUT_PCN",
        "QWEN36_LMHEAD_DSHARD_IBW",
        "QWEN36_LMHEAD_DSHARD_FP32_DEST",
    )
    assert tpc.lm_head_dshard_scope() == "all"
    assert tpc.lm_head_dshard_out_pcn() is None
    assert tpc.lm_head_dshard_ibw() is None
    assert tpc.lm_head_dshard_fp32_dest() is False


@pytest.mark.parametrize("raw,want", [("draft", "draft"), ("BASE", "base"), (" all ", "all"), ("", "all")])
def test_scope_values(clean_env, raw, want):
    clean_env.setenv(tpc.LMHEAD_DSHARD_SCOPE_ENV, raw)
    assert tpc.lm_head_dshard_scope() == want


@pytest.mark.parametrize("raw", ["verify", "1", "drafter"])
def test_scope_invalid(clean_env, raw):
    clean_env.setenv(tpc.LMHEAD_DSHARD_SCOPE_ENV, raw)
    with pytest.raises(ValueError):
        tpc.lm_head_dshard_scope()


def test_int_and_bool_knobs(clean_env):
    clean_env.setenv(tpc.LMHEAD_DSHARD_OUT_PCN_ENV, "10")
    clean_env.setenv(tpc.LMHEAD_DSHARD_IBW_ENV, "8")
    clean_env.setenv(tpc.LMHEAD_DSHARD_FP32_DEST_ENV, "1")
    assert (tpc.lm_head_dshard_out_pcn(), tpc.lm_head_dshard_ibw(), tpc.lm_head_dshard_fp32_dest()) == (10, 8, True)
    for name in (tpc.LMHEAD_DSHARD_OUT_PCN_ENV, tpc.LMHEAD_DSHARD_IBW_ENV):
        clean_env.setenv(name, "0")
        with pytest.raises(ValueError):
            tpc._env_positive_int(name)
    clean_env.setenv(tpc.LMHEAD_DSHARD_FP32_DEST_ENV, "2")
    with pytest.raises(ValueError):
        tpc.lm_head_dshard_fp32_dest()
    clean_env.setenv(tpc.LMHEAD_DSHARD_FP32_DEST_ENV, "off")
    assert tpc.lm_head_dshard_fp32_dest() is False


def test_scope_takes_truth_table():
    take = tpc.lm_head_dshard_scope_takes
    assert take("all", draft=True) and take("all", draft=False)
    assert take("draft", draft=True) and not take("draft", draft=False)
    assert take("base", draft=False) and not take("base", draft=True)


# ------------------------------------------------------------------------------------------------ plan / layout


def test_plan_mc_defaults_are_r5():
    """No K6 knob: the r5 mc plan (bank shard as the logits shard, in0_block_w 2, qb2 reader rule)."""
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc")
    assert (p.k_tiles, p.n_tiles, p.bank_tiles, p.padded_tiles) == (160, 970, 122, 976)
    assert (p.readers, p.in0_block_w, p.per_core_N, p.out_cores) == (4, 2, 122, 8)
    assert (p.act_cores, p.act_shard_w) == (8, 640)
    # create_program_dram_sharded_multicore_spec: CN = min(W, Nbt), CK = 1, ncmax 31, P = 4 passes of NCP = 8 columns,
    # KB = 2, max_own = 31 (bank columns 0-30 | 30-61 | 61-91 | 91-122), 80 streamed blocks, 8 x 4 work items.
    assert p.mc_layout() == dict(cn=4, ck=1, p=4, ncp=8, kb=2, max_own=31, max_blocks=80, work=32)


def test_plan_exact_unchanged_and_rejects_k6():
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact")
    assert (p.readers, p.in0_block_w, p.per_core_N, p.bank_tiles) == (2, 2, 122, 122)
    for kw in (dict(out_pcn=10), dict(ibw=4)):
        with pytest.raises(ValueError):
            tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", **kw)
    with pytest.raises(ValueError):
        p.mc_layout()


def test_plan_out_pcn():
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", out_pcn=10)
    assert (p.per_core_N, p.out_cores, p.bank_tiles) == (10, 97, 122)  # 970 = 97 x 10: no partial shard
    assert tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", out_pcn=9).out_cores == 108  # partial last shard (7 tiles)
    assert tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", out_pcn=97).out_cores == 10
    # The weight layout does not move with the logits layout.
    assert p.mc_layout() == tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").mc_layout()


@pytest.mark.parametrize(
    "ibw,kb,blocks",
    [(1, 1, 160), (2, 2, 80), (3, 3, 54), (4, 4, 40), (8, 8, 20), (16, 16, 10)],
)
def test_plan_ibw(ibw, kb, blocks):
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", ibw=ibw)
    lay = p.mc_layout()
    assert (p.in0_block_w, lay["kb"], lay["max_blocks"]) == (ibw, kb, blocks)
    assert p.act_shard_w == 640  # 20 tiles per activation shard whatever the block width


@pytest.mark.parametrize(
    "readers,lay",
    [
        (2, dict(cn=2, ck=1, p=8, ncp=8, kb=2, max_own=61, max_blocks=80, work=16)),
        (3, dict(cn=3, ck=1, p=6, ncp=7, kb=2, max_own=41, max_blocks=80, work=24)),
        (8, dict(cn=8, ck=1, p=2, ncp=8, kb=2, max_own=16, max_blocks=80, work=64)),
    ],
)
def test_mc_layout_readers(readers, lay):
    assert tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", readers=readers).mc_layout() == lay


def test_mc_layout_rejects_factory_errors():
    with pytest.raises(ValueError):  # 8 banks x 64 workers: past the pipeline's one-byte role table
        tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", readers=64)
    with pytest.raises(ValueError):
        tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", readers=1)


@pytest.mark.parametrize("ibw,passk", [(1, 8), (2, 8), (3, 9), (4, 8), (8, 8), (16, 16)])
def test_mc_k_tiles_per_dest_pass(ibw, passk):
    """The "8 K tiles per Dest pass" of the r5 mc head (16-bit Dest + Float16_b L1 accumulator), and the whole K (160
    tiles, one rounding) with QWEN36_LMHEAD_DSHARD_FP32_DEST or for the drafter's fp32 call."""
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", ibw=ibw)
    assert p.mc_k_tiles_per_dest_pass() == passk
    assert p.mc_k_tiles_per_dest_pass(fp32_dest=True) == 160
    assert tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").mc_k_tiles_per_dest_pass() == 8


# ------------------------------------------------------------------------------------------------ L1 model


def test_mc_l1_bytes_default_layout():
    """W 4, KB 2: x = 160 bf16 tiles (327680), weights 2 streams x 3 blocks x 2 x 8 bfp8 tiles (104448), out staging
    31 tiles, Float16_b accumulator 31 tiles without fp32 Dest; logits shard 122 tiles on 8 cores."""
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc")
    x, w = 160 * 2048, 2 * 3 * 2 * 8 * 1088
    draft = p.mc_l1_bytes("float32", fp32_dest=True)  # the drafter's fp32 head: fp32 Dest, no L1 accumulator
    base = p.mc_l1_bytes("bfloat16")  # the base/verify bf16 head (r5 mc): 16-bit Dest + Float16_b accumulator
    base32 = p.mc_l1_bytes("bfloat16", fp32_dest=True)  # QWEN36_LMHEAD_DSHARD_FP32_DEST
    assert draft == dict(cb=x + w + 31 * 4096, act_shard=20 * 2048, out_shard=122 * 4096, total=1099776)
    assert base == dict(cb=x + w + 31 * 2048 * 2, act_shard=20 * 2048, out_shard=122 * 2048, total=849920)
    assert base32["cb"] == x + w + 31 * 2048 == 495616
    assert p.mc_l1_need() == draft  # the fp32 drafter call is the worst one
    # Two streams of depth 3 on Blackhole; one stream of depth 5 elsewhere.
    assert p.mc_l1_bytes("bfloat16", two_streams=False)["cb"] == x + 5 * 2 * 8 * 1088 + 31 * 2048 * 2


def test_mc_l1_out_pcn_frees_the_logits_shard():
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", out_pcn=10)
    d = p.mc_l1_bytes("float32", fp32_dest=True)
    assert d["out_shard"] == 10 * 4096 and d["total"] == 559104 + 40960 + 40960
    assert tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").mc_l1_need()["total"] - d["total"] == 112 * 4096


@pytest.mark.parametrize(
    "readers,cb", [(2, 327680 + 104448 + 61 * 4096), (3, 327680 + 2 * 3 * 2 * 7 * 1088 + 41 * 4096)]
)
def test_mc_l1_readers(readers, cb):
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", readers=readers)
    assert p.mc_l1_bytes("float32", fp32_dest=True)["cb"] == cb


def test_mc_l1_bank_constant():
    # 1536 KB - (MEM_MAP_END 41344 + 69 KB kernel config, 64 B aligned = 112000) - l1_small 24576.
    assert tpc.LMHEAD_DSHARD_MC_L1_BANK == 1436288
    assert tpc.LMHEAD_DSHARD_MC_L1_RESERVED == 32 * 5120 * 8 // 40 * 2  # the TP8 all-reduce staging buffer


def test_check_l1_mc(clean_env):
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").check_l1()  # r5 default: 1074 KB of 1338 KB
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", readers=2).check_l1()  # 1194 KB
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", out_pcn=10).check_l1(fp32_dest=True)
    # Weight prefetch "all": 6 GCBs x 72 KB beside the all-reduce buffer. The 8-core fp32 logits shard no longer fits
    # (it would clash at warm-up); the 97-core one does.
    clean_env.setenv("QWEN36_WEIGHT_PREFETCH", "all")
    with pytest.raises(ValueError, match="OUT_PCN"):
        tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").check_l1()
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", out_pcn=10).check_l1()
    clean_env.delenv("QWEN36_WEIGHT_PREFETCH")
    with pytest.raises(ValueError):
        tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", ibw=16, readers=2).check_l1(reserved=400 * KB)


# ------------------------------------------------------------------------------------------------ LMHeadDShard wiring


class _FakeMesh:
    def __init__(self, grid=(11, 10)):
        self._grid = grid

    def dram_grid_size(self):
        return types.SimpleNamespace(x=8, y=1)

    def compute_with_storage_grid_size(self):
        return types.SimpleNamespace(x=self._grid[0], y=self._grid[1])

    def arch(self):
        return "blackhole"


def _fake_weight():
    return types.SimpleNamespace(shape=(K, N_TP8), dtype=ttnn.bfloat8_b)


@pytest.fixture
def recorded(monkeypatch, clean_env):
    rec = {"progcfg": [], "ckc": []}
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mc: t)
    monkeypatch.setattr(ttnn, "create_sharded_memory_config", lambda **kw: ("act_memcfg", kw["shape"]))
    monkeypatch.setattr(
        ttnn, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", lambda **kw: rec["progcfg"].append(kw) or kw
    )
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda arch, **kw: rec["ckc"].append(kw) or kw)
    return rec


def test_dshard_mc_defaults_build_the_r5_program(recorded):
    h = tpc.LMHeadDShard(_fake_weight(), _FakeMesh(), "mc", None)
    assert recorded["progcfg"] == [
        dict(
            in0_block_w=2,
            per_core_M=1,
            per_core_N=122,
            fused_activation=None,
            num_workers_per_dram_bank=4,
            use_multicore_pipeline=True,
        )
    ]
    assert h.fp32_dest is False
    # Base call (bf16 out): HiFi2 (bf16 x bfp8), 16-bit Dest + packer L1 acc; drafter call (fp32 out): fp32 Dest.
    assert h.compute_config(ttnn.bfloat16, ttnn.bfloat16)["fp32_dest_acc_en"] is False
    assert recorded["ckc"][-1]["packer_l1_acc"] is True
    assert recorded["ckc"][-1]["math_fidelity"] == ttnn.MathFidelity.HiFi2
    assert h.compute_config(ttnn.bfloat16, ttnn.float32)["fp32_dest_acc_en"] is True
    assert recorded["ckc"][-1]["packer_l1_acc"] is False


def test_dshard_mc_k6_knobs(recorded):
    h = tpc.LMHeadDShard(_fake_weight(), _FakeMesh(), "mc", None, out_pcn=10, ibw=8, fp32_dest=True)
    cfg = recorded["progcfg"][-1]
    assert (cfg["per_core_N"], cfg["in0_block_w"], cfg["num_workers_per_dram_bank"]) == (10, 8, 4)
    assert h.plan.out_cores == 97
    ckc = h.compute_config(ttnn.bfloat16, ttnn.bfloat16)
    assert ckc["fp32_dest_acc_en"] is True and ckc["packer_l1_acc"] is False


def test_dshard_reads_k6_env(recorded):
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv(tpc.LMHEAD_DSHARD_OUT_PCN_ENV, "10")
        mp.setenv(tpc.LMHEAD_DSHARD_FP32_DEST_ENV, "1")
        h = tpc.LMHeadDShard(_fake_weight(), _FakeMesh(), "mc", 2)
        assert (recorded["progcfg"][-1]["per_core_N"], h.fp32_dest, h.plan.readers) == (10, True, 2)
        # exact rejects every K6 knob (it must keep the plain head's blocking, output layout and Dest mode).
        with pytest.raises(ValueError):
            tpc.LMHeadDShard(_fake_weight(), _FakeMesh(), "exact", None)
        mp.delenv(tpc.LMHEAD_DSHARD_OUT_PCN_ENV)
        with pytest.raises(ValueError):
            tpc.LMHeadDShard(_fake_weight(), _FakeMesh(), "exact", None)  # FP32_DEST still set
    finally:
        mp.undo()


def test_dshard_out_cores_must_fit_grid(recorded):
    with pytest.raises(ValueError, match="grid"):
        tpc.LMHeadDShard(_fake_weight(), _FakeMesh(grid=(8, 8)), "mc", None, out_pcn=10)  # 97 > 64
    tpc.LMHeadDShard(_fake_weight(), _FakeMesh(grid=(11, 10)), "mc", None, out_pcn=10)


# ------------------------------------------------------------------------------------------------ routing


BASE_W = object()
DRAFT_BFP4_W = object()


class _FakeDShard:
    def __init__(self):
        self.calls = []

    def applies(self, x):
        return True

    def linear(self, x, **kw):
        self.calls.append(kw)
        return "dshard"


def _route(monkeypatch, scope):
    monkeypatch.setattr(ttnn, "linear", lambda x, w, **kw: ("plain", w))
    fake = types.SimpleNamespace(lm_head_weight=BASE_W, _lm_head_dshard=_FakeDShard())
    if scope is not None:
        fake._lm_head_dshard_scope = scope
    base = Qwen36Model._lm_head_linear(fake, "x")  # verify / decode / prefill last row
    draft = Qwen36Model._lm_head_linear(fake, "x", weight=BASE_W, dtype=ttnn.float32)  # MTP drafter, shared head
    bfp4 = Qwen36Model._lm_head_linear(fake, "x", weight=DRAFT_BFP4_W, dtype=ttnn.float32)  # drafter's own copy
    return base, draft, bfp4


@pytest.mark.parametrize(
    "scope,want",
    [
        (None, ("dshard", "dshard")),  # a test double without the attribute: today's routing
        ("all", ("dshard", "dshard")),
        ("draft", (("plain", BASE_W), "dshard")),
        ("base", ("dshard", ("plain", BASE_W))),
    ],
)
def test_lm_head_linear_scope(monkeypatch, scope, want):
    base, draft, bfp4 = _route(monkeypatch, scope)
    assert (base, draft) == want
    assert bfp4 == ("plain", DRAFT_BFP4_W)  # never the BFP8 DRAM-sharded copy


def test_model_init_scope_attribute_default():
    """Qwen36Model.__init__ always sets _lm_head_dshard_scope ("all" unless QWEN36_LMHEAD_DSHARD is on)."""
    import inspect

    src = inspect.getsource(Qwen36Model.__init__)
    assert 'self._lm_head_dshard_scope = "all"' in src
    assert "self._lm_head_dshard_scope = _tpc.lm_head_dshard_scope()" in src
