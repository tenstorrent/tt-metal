# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only (no device is opened or enumerated): every Qwen3-32B long-prefill optimization applies only while the
prefill's total context (cached prefix + new tokens S) is <= QWEN_OPT_MAX_SEQ_LEN (40960). Beyond it, and in decode,
the model reads exactly what tt-metal main@8bdf164e6e5 reads.

- total <= cap: the gates and length-keyed configs equal the branch head before the cap (4a43f337532, HEAD_* below).
- total > cap: they equal main (MAIN_* below: no CP / seq-parallel path, main's matmul and SDPA configs, no ring CCL
  overrides, power-of-2 padding, main's warmup list above the cap), also for a prefix-cached prefill whose S alone
  is <= cap (PREFIX_CASES).
- RMSNorm eps: decode, every prefix-cached prefill and total > cap prefill use main's (the RMSNorm default 1e-5); a
  prefill with no cached prefix and total <= cap uses Qwen3's HF rms_norm_eps (1e-6). The prefill trace rule is HEAD's.
- Inline gates in the model code (llama_attention / llama_mlp / llama_model / llama_ccl) are checked structurally:
  every Qwen long-prefill condition also calls qwen_prefill_opt_ok (the total-context check).

Needs HF_MODEL (Qwen3-32B config only; dummy weights). ttnn.get_arch_name is patched to Wormhole (the optimizations
are Wormhole-only), so arch detection does not query the cluster."""

import ast
import inspect
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import ttnn

pytestmark = pytest.mark.host_only  # no device: these tests run anywhere

CAP = 40960  # the user's limit; QWEN_OPT_MAX_SEQ_LEN must equal it
SWEEP = [4096, 8192, 16384, 24576, 32768, 40960, 40961, 45000, 45056, 49152, 57344, 65536, 73728, 131072]
TT = Path(__file__).resolve().parents[1] / "tt"

_RS = dict(num_workers_per_link=6, chunks_per_sync=64, num_buffers_per_channel=4)
_AG = dict(num_workers_per_link=6)


def _mm(M, K, N, sh, sw, gx, gy):
    return dict(M=M, K=K, N=N, sh=sh, sw=sw, grid=(gx, gy))


# Branch head before the cap (4a43f337532), Wormhole, recorded host-only with the same readers as below.
HEAD_LE_CAP = {
    4096: dict(
        ring_sdpa=(256, 512, False),
        xqkv=_mm(8, 8, 8, 1, 8, 7, 8),
        wo=_mm(8, 8, 8, 4, 2, 7, 8),
        ff13=_mm(8, 8, 8, 4, 2, 7, 9),
        rs=None,
        ag=None,
        cp=False,
        fused=(False, False),
    ),
    8192: dict(
        ring_sdpa=(64, 512, True),
        xqkv=_mm(8, 8, 8, 1, 8, 7, 8),
        wo=_mm(8, 8, 8, 4, 2, 7, 8),
        ff13=_mm(8, 8, 8, 1, 8, 7, 9),
        rs=_RS,
        ag=_AG,
        cp=False,
        fused=(False, False),
    ),
    16384: dict(
        ring_sdpa=(128, 512, True),
        xqkv=_mm(8, 8, 8, 2, 2, 7, 10),
        wo=_mm(16, 8, 4, 2, 2, 7, 10),
        ff13=_mm(8, 8, 8, 1, 8, 7, 8),
        rs=_RS,
        ag=_AG,
        cp=True,
        fused=(False, False),
    ),
    24576: dict(
        ring_sdpa=(192, 512, True),
        xqkv=_mm(8, 8, 8, 2, 2, 7, 10),
        wo=_mm(16, 8, 4, 2, 2, 7, 10),
        ff13=_mm(8, 8, 8, 1, 8, 7, 8),
        rs=_RS,
        ag=_AG,
        cp=True,
        fused=(False, False),
    ),
    32768: dict(
        ring_sdpa=(256, 512, True),
        xqkv=_mm(8, 8, 8, 2, 2, 7, 10),
        wo=_mm(16, 8, 4, 2, 2, 7, 10),
        ff13=_mm(8, 8, 8, 1, 8, 7, 8),
        rs=_RS,
        ag=_AG,
        cp=True,
        fused=(True, True),
    ),
    40960: dict(
        ring_sdpa=(320, 640, True),
        xqkv=_mm(8, 8, 8, 2, 2, 7, 10),
        wo=_mm(16, 8, 4, 2, 2, 7, 10),
        ff13=_mm(8, 8, 8, 1, 8, 7, 8),
        rs=_RS,
        ag=_AG,
        cp=True,
        fused=(True, True),
    ),
}
# main@8bdf164e6e5 for every S > 4096 above the cap: the ring SDPA reads SDPA_PROGCFG (q256/k512, exp_approx off).
MAIN_GT_CAP = dict(
    ring_sdpa=(256, 512, False),
    xqkv=_mm(8, 8, 8, 1, 8, 7, 8),
    wo=_mm(8, 8, 8, 4, 2, 7, 8),
    ff13=_mm(8, 8, 8, 4, 2, 7, 8),
    rs=None,
    ag=None,
    cp=False,
    fused=(False, False),
)
MAIN_RMSNORM_EPS = 1e-05  # main builds every Qwen RMSNorm with the RMSNorm default
QWEN_HF_EPS = 1e-06


def main_padded_len(seq_len):  # main@8bdf164e6e5 generator.get_padded_prefill_len
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    return 2 ** (seq_len - 1).bit_length()


def main_warmup_lengths(max_seq_len):  # main@8bdf164e6e5 generator.get_prefill_warmup_sequence_lengths
    return [128] + [2**i for i in range(10, max_seq_len.bit_length()) if 2**i <= max_seq_len]


def head_padded_len(seq_len, m=8192):  # 4a43f337532 rule with Qwen's pad_multiple (no cap)
    p = main_padded_len(seq_len)
    return -(-seq_len // m) * m if p > 4096 else p


@pytest.fixture(autouse=True)
def ensure_devices():
    """Overrides the galaxy conftest's device-count check: these tests never touch a device."""


@pytest.fixture(scope="module")
def wormhole_arch():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
        yield


@pytest.fixture(scope="module")
def qwen_args(wormhole_arch):
    from models.demos.llama3_70b_galaxy.tests.qwen_host_model_args import _HostOnlyGalaxyMesh
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

    return TtQwenModelArgs(_HostOnlyGalaxyMesh(), max_batch_size=1, max_seq_len=131072, dummy_weights=True)


def _mmd(c):
    g = c.compute_with_storage_grid_size
    return _mm(c.M_block_size, c.K_block_size, c.N_block_size, c.subblock_h, c.subblock_w, g.x, g.y)


def _observed(args, S, cached=0):
    """What the model reads at prefill length S (batch 1) after `cached` prefix tokens, via the same lookups the
    model code uses. cached > 0 records the total context the way TtTransformer.forward does."""
    from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill as cp

    mc = args.model_config
    if cached:
        from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import set_prefill_total_len

        set_prefill_total_len(mc, cached + S)
    try:
        ring = mc.get("SDPA_RING_PROGCFG", mc["SDPA_PROGCFG"])(S, chunk_start_idx=0)  # llama_attention's lookup
        rs = {k: mc["QWEN_PREFILL_RING_RS_PARAMS"](S, k) for k in ("FF1", "FF3", "QKV", "WO", "FF2")}
        ag = {k: mc["QWEN_PREFILL_RING_AG_PARAMS"](S, k) for k in ("QKV", "FF2")}
        xqkv = _mmd(mc["XQKV_PREFILL_MINIMAL_PROGCFG"](S))
        wo = _mmd(mc["WO_PREFILL_MINIMAL_PROGCFG"](S))
        ff13 = _mmd(mc["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"](S))
    finally:
        mc.pop("QWEN_PREFILL_TOTAL_LEN", None)
    assert len({str(v) for v in rs.values()}) == 1 and len({str(v) for v in ag.values()}) == 1
    a = SimpleNamespace(is_qwen=True, is_blackhole=False, cluster_shape=(8, 4))
    model = SimpleNamespace(args=a, use_prefetcher=True, prefetcher_setup=object())
    cp_on = cp.cp_prefill_applies(a, True, object(), "prefill", 1, cached, S)
    if S % (32 * cp.EXIT_WINDOW) == 0:  # layer-major needs S/32 % 128 == 0; same verdict as cp_prefill_applies
        assert cp.layer_major_applies(model, 1, [S], [cached], object(), True) == cp_on
    return dict(
        ring_sdpa=(ring.q_chunk_size, ring.k_chunk_size, ring.exp_approx_mode),
        xqkv=xqkv,
        wo=wo,
        ff13=ff13,
        rs=rs["FF1"],
        ag=ag["QKV"],
        cp=cp_on,
        fused=(cp.fused_ag_enabled(S), cp.fused_ag_ff1_enabled(S)),
    )


@pytest.mark.parametrize("S", SWEEP)
def test_length_keyed_gates_and_configs(qwen_args, S):
    obs = _observed(qwen_args, S)
    if S <= CAP:
        assert obs == HEAD_LE_CAP[S], (S, obs)
    else:
        assert obs == MAIN_GT_CAP, (S, obs)


# Prefix-cached prefills: (cached, new S). The gates follow the total context cached + S (the KV extent).
PREFIX_CASES = [
    (32768, 16384),  # total 49152 > cap: every gate off, main's configs (even though S alone is <= cap)
    (24576, 24576),  # total 49152 > cap
    (45056, 4096),  # total 49152 > cap (S <= 4096: HEAD and main configs agree; the eps differs, see below)
    (8192, 32768),  # total 40960 <= cap: HEAD's values for S (no CP: it needs no prefix)
    (4096, 8192),  # total 12288 <= cap
]


@pytest.mark.parametrize("cached, S", PREFIX_CASES)
def test_prefix_cached_gates_follow_total_context(qwen_args, cached, S):
    obs = _observed(qwen_args, S, cached=cached)
    if cached + S > CAP:
        want = dict(MAIN_GT_CAP) if S > 4096 else dict(HEAD_LE_CAP[4096])
        want["fused"] = obs["fused"]  # fused AG is CP-only; CP is off with a prefix
    else:
        want = {**HEAD_LE_CAP[S], "cp": False}
    assert obs == want, (cached, S, obs)
    # the S-keyed lookup with no recorded total (direct callers) is unchanged
    assert _observed(qwen_args, S) == HEAD_LE_CAP[S]


@pytest.mark.parametrize("cached, S", PREFIX_CASES)
def test_prefix_cached_inline_gates_and_eps(cached, S):
    """The predicate every inline gate calls (qwen_prefill_opt_ok) follows the recorded total context; the prefill norm
    eps (q_norm / k_norm / DistributedNorm via prefill_norm) is main's for every prefix-cached prefill, whatever the
    total, so the warmup-captured prefix-cached (sp1) traces carry main's eps."""
    from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import (
        attach_prefill_opt_eps,
        prefill_norm,
        prefill_norm_eps,
        prefill_total_len,
        qwen_prefill_opt_ok,
        set_prefill_total_len,
    )

    mc = {}
    set_prefill_total_len(mc, cached + S, num_cached=cached)
    assert prefill_total_len(mc, S) == cached + S
    assert qwen_prefill_opt_ok(mc, S) == (cached + S <= CAP)
    norm = attach_prefill_opt_eps(_host_rmsnorm(), QWEN_HF_EPS)
    assert prefill_norm(norm, S, mc) is norm and prefill_norm_eps(norm, S, mc) == MAIN_RMSNORM_EPS
    # the same total without a cached prefix (as TtTransformer.forward records start_pos 0): Qwen's eps within the cap
    nc = {}
    set_prefill_total_len(nc, cached + S)
    assert prefill_norm_eps(norm, cached + S, nc) == (QWEN_HF_EPS if cached + S <= CAP else MAIN_RMSNORM_EPS)
    assert prefill_total_len({}, S) == S and prefill_total_len(None, S) == S  # nothing recorded: S
    assert prefill_norm_eps(norm, S) == QWEN_HF_EPS  # nothing recorded (op tests): S <= cap, no prefix


@pytest.mark.parametrize("cached, S", PREFIX_CASES)
def test_prefix_cached_distributed_norm_eps(wormhole_arch, monkeypatch, cached, S):
    from models.demos.llama3_70b_galaxy.tt import distributed_norm as dn_mod
    from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import set_prefill_total_len

    seen = []
    monkeypatch.setattr(dn_mod, "tt_distributed_rmsnorm", lambda *a, epsilon=None, **k: seen.append(epsilon))
    monkeypatch.setattr(dn_mod, "tt_sharded_distributed_rmsnorm", lambda *a, epsilon=None, **k: seen.append(epsilon))
    args = SimpleNamespace(
        is_qwen=True,
        norm_eps=QWEN_HF_EPS,
        qk_norm=True,
        dim=5120,
        blackhole_no_prefetcher=False,
        mesh_device=None,
        model_config={},
    )
    set_prefill_total_len(args.model_config, cached + S, num_cached=cached)
    dn = dn_mod.DistributedNorm(_host_rmsnorm(), args, use_sharded_decode=True)
    dn.forward(SimpleNamespace(shape=(1, 1, S, 1280)), None, "prefill", seq_len=S)  # decoder norms
    dn.forward(SimpleNamespace(shape=(1, 1, S, 1280)), None, "prefill")  # output norm after the forward
    dn.forward(SimpleNamespace(shape=(1, 1, 32, 1280)), None, "decode")
    assert seen == [MAIN_RMSNORM_EPS] * 3  # any cached prefix: main's eps, at totals <= and > cap
    # a prefill of the same total with no cached prefix keeps Qwen's eps within the cap
    seen.clear()
    set_prefill_total_len(args.model_config, cached + S)
    dn.forward(SimpleNamespace(shape=(1, 1, cached + S, 1280)), None, "prefill", seq_len=cached + S)
    assert seen == [QWEN_HF_EPS if cached + S <= CAP else MAIN_RMSNORM_EPS]


@pytest.fixture
def fake_prefill_stack(wormhole_arch, monkeypatch):
    """A Generator and a TtTransformer stand-in that run the real generator prefill entry points
    (_easy_trace_prefill replay, prefill_forward_single_user_text -> TtTransformer.ttnn_prefill_forward -> forward)
    with no device: the device-side pieces (trace replay, input staging, rot-mat slicing, layers) are host fakes.
    output_norm_eps() returns the eps the output norm of process_output_prefill* (model.norm(..., mode="prefill"))
    picks from the record the prefill left behind."""
    import torch

    import models.demos.llama3_70b_galaxy.tt.generator as gen_mod
    from models.demos.llama3_70b_galaxy.tt import distributed_norm as dn_mod
    from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer

    monkeypatch.setattr(gen_mod, "get_block_size", lambda kv_cache: 64)
    monkeypatch.setattr(gen_mod, "num_blocks_in_seq", lambda n, bs: -(-n // bs))
    monkeypatch.setattr(gen_mod, "_get_max_blocks_prefill", lambda kv_cache: 2048)
    monkeypatch.setattr(gen_mod, "_pad_or_create_page_table", lambda table, n: table)
    monkeypatch.setattr(ttnn, "concat", lambda *a, **k: "rot_slice_starts")
    monkeypatch.setattr(ttnn, "slice", lambda *a, **k: "rot_slice")
    seen = []
    monkeypatch.setattr(dn_mod, "tt_distributed_rmsnorm", lambda *a, epsilon=None, **k: seen.append(epsilon))

    mc = {}
    args = SimpleNamespace(
        is_qwen=True,
        norm_eps=QWEN_HF_EPS,
        qk_norm=True,
        dim=5120,
        blackhole_no_prefetcher=False,
        mesh_device=None,
        model_config=mc,
        cluster_shape=(8, 4),
        is_blackhole=False,
    )
    model = SimpleNamespace(
        args=args,
        model_config=mc,
        use_prefetcher=False,  # keeps CP / seq-parallel off: the fake has no layers
        prefetcher_setup=None,
        layers=[],
        mesh_device=None,
        tt_ccl=SimpleNamespace(close=lambda: None),
        _tt_slice_start_zeros_4=[0, 0, 0, 0],
        _tt_seq_len_buffer=None,
        norm=dn_mod.DistributedNorm(_host_rmsnorm(), args, use_sharded_decode=True),
        get_or_create_prefill_rot_mats=lambda: ("cos", "sin"),
        prepare_inputs_prefill=lambda tokens, **kw: (
            SimpleNamespace(shape=(1, 1, tokens.shape[-1], 1280)),  # embedded input
            "user_id",
            "page_table",
            "chunk_page_table",
            SimpleNamespace(shape=(1,)),  # chunk_start_idx device tensor
            "column_mask",
        ),
    )
    model.forward = TtTransformer.forward.__get__(model)
    model.ttnn_prefill_forward = TtTransformer.ttnn_prefill_forward.__get__(model)

    gen = gen_mod.Generator.__new__(gen_mod.Generator)
    gen.model = model
    gen.trace_id_prefill = defaultdict(lambda: None)
    gen.trace_inputs_prefill, gen.trace_output_prefill = defaultdict(lambda: None), defaultdict(lambda: None)
    gen._defer_trace_recording, gen._pending_prefill_traces = False, {}
    gen._prefill_forward_trace_text = lambda trace_id, inputs, out, tokens, *a, **k: SimpleNamespace(
        shape=(1, 1, tokens.shape[-1], 1280)
    )

    def traced(S, cached):
        gen.trace_id_prefill[f"{S}_1_{'sp1' if cached else 'sp0'}"] = 1  # captured earlier (warmup): replay only
        return gen._easy_trace_prefill(
            torch.zeros(1, S, dtype=torch.long),
            cached + S - 1,
            S,
            page_table=torch.zeros(1, (cached + S) // 64, dtype=torch.int32),
            kv_cache=[None],
            num_cached_tokens=cached,
        )

    def untraced(S, cached):
        return gen.prefill_forward_single_user_text(
            torch.zeros(1, S, dtype=torch.long),
            torch.zeros(1, (cached + S) // 64, dtype=torch.int32),
            0,
            cached + S - 1,
            kv_cache=[None],
            num_cached_tokens=cached,
        )

    def output_norm_eps(tt_out):
        seen.clear()
        model.norm(tt_out, res=None, mode="prefill")  # what process_output_prefill* runs first
        return seen[-1]

    yield SimpleNamespace(traced=traced, untraced=untraced, output_norm_eps=output_norm_eps, model_config=mc)
    gen.trace_id_prefill.clear()  # fake trace ids: nothing for Generator.__del__ to release


@pytest.mark.parametrize("path", ["traced", "untraced"])
def test_output_norm_eps_follows_the_prefill_just_run(fake_prefill_stack, path):
    """The output norm (process_output_prefill*, outside any trace) reads the record of the prefill that just ran,
    on the traced replay path (no forward) and the untraced one: no cached prefix and total <= cap -> Qwen's eps;
    any cached prefix or total > cap -> main's. Each case starts from the opposite stale record (e.g. warmup ending on
    a prefix-cached capture)."""
    from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import set_prefill_total_len

    run = getattr(fake_prefill_stack, path)
    cases = [(4096, 0, QWEN_HF_EPS), (4096, 128, MAIN_RMSNORM_EPS), (4096, 45056, MAIN_RMSNORM_EPS)]
    if path == "untraced":
        cases += [(32768, 0, QWEN_HF_EPS), (16384, 32768, MAIN_RMSNORM_EPS), (49152, 0, MAIN_RMSNORM_EPS)]
    for S, cached, want in cases:
        stale_cached = 0 if cached else 128
        set_prefill_total_len(fake_prefill_stack.model_config, stale_cached + 4096, num_cached=stale_cached)
        out = run(S, cached)
        assert fake_prefill_stack.output_norm_eps(out) == want, (path, S, cached)


def test_prefill_trace_rule_is_heads():
    """The prefill trace rule is HEAD's / main's: tracing is turned off only for untraced lengths and seeded prefill
    (the prefix-cached traces carry main's eps, valid at every total)."""
    gen_src = _src("generator.py")
    trace_off = sorted(
        ast.unparse(n.test)
        for n in ast.walk(ast.parse(gen_src))
        if isinstance(n, ast.If)
        and any(isinstance(s, ast.Assign) and getattr(s.targets[0], "id", None) == "user_enable_trace" for s in n.body)
    )
    assert trace_off == ["explicit_seeded_prefill", "prefill_seq_len not in self.model.tt_ccl.support_seqlens"]


def test_cap_constant_is_shared():
    from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill, qwen_model_config
    from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import QWEN_OPT_MAX_SEQ_LEN

    assert QWEN_OPT_MAX_SEQ_LEN == CAP
    assert qwen_cp_prefill.FUSED_AG_MAX_SEQ == QWEN_OPT_MAX_SEQ_LEN
    assert qwen_model_config.RING_SDPA_MAX_MEASURED_SEQ_LEN == QWEN_OPT_MAX_SEQ_LEN


def test_padding_matches_main_above_cap_and_head_below():
    from models.demos.llama3_70b_galaxy.tt.generator import get_padded_prefill_len

    m = 8192  # TtQwenModelArgs.prefill_pad_multiple
    for s in list(range(1, 2 * CAP, 7)) + list(range(CAP - 300, CAP + 300)) + [65536, 65537, 73728, 131072]:
        got = get_padded_prefill_len(s, m)
        if head_padded_len(s) <= CAP:
            assert got == head_padded_len(s), s  # unchanged below the cap
        else:
            assert got == main_padded_len(s), s  # main's power of 2 above it
        assert get_padded_prefill_len(s) == main_padded_len(s), s  # Llama (None) unchanged
    assert get_padded_prefill_len(40960, m) == 40960 and get_padded_prefill_len(40961, m) == 65536


def test_prefix_cached_padding_matches_main_above_cap():
    """A prefix-cached request whose total (cached + new) is above the cap pads its new tokens and decides the
    prefix-caching heuristic exactly as main (powers of 2), even when the new tokens alone are <= cap."""
    from models.demos.llama3_70b_galaxy.tt.generator import (
        _should_skip_prefix_caching,
        get_padded_prefill_len,
        prefill_pad_multiple_for,
    )

    def main_skip(seq_len, num_cached):  # main@8bdf164e6e5 _should_skip_prefix_caching
        if num_cached == 0:
            return False
        new, total = main_padded_len(seq_len - num_cached), main_padded_len(seq_len)
        if new >= total:
            return True
        r = new / total
        return False if r <= 0.125 else total > 32768 if r <= 0.25 else total > 16384 if r <= 0.5 else True

    for seq_len in range(CAP + 1, 3 * 65536, 997):
        for num_cached in range(128, seq_len, 4096):
            pm = prefill_pad_multiple_for(seq_len, 8192, num_cached)
            assert pm is None
            assert get_padded_prefill_len(seq_len - num_cached, pm) == main_padded_len(seq_len - num_cached)
            assert _should_skip_prefix_caching(seq_len, num_cached, 8192) == main_skip(seq_len, num_cached)
    assert prefill_pad_multiple_for(CAP, 8192) == 8192  # <= cap: unchanged
    # total context = cached + padded new tokens (the KV extent the model gates on)
    assert prefill_pad_multiple_for(32768 + 16384, 8192, 32768) is None  # 49152 > cap
    assert prefill_pad_multiple_for(16384 + 20000, 8192, 16384) == 8192  # 16384 + 24576 = 40960 <= cap
    assert prefill_pad_multiple_for(17024 + 23000, 8192, 17024) is None  # 17024 + 24576 > cap: main pads 32768
    assert get_padded_prefill_len(23000, prefill_pad_multiple_for(17024 + 23000, 8192, 17024)) == 32768


@pytest.mark.parametrize("max_seq_len", [4096, 20000, 40960, 45056, 65536, 131072])
def test_warmup_lengths(max_seq_len):
    from models.demos.llama3_70b_galaxy.tt.generator import get_padded_prefill_len, get_prefill_warmup_sequence_lengths

    warm = get_prefill_warmup_sequence_lengths(max_seq_len, 8192)
    assert [w for w in warm if w > CAP] == [w for w in main_warmup_lengths(max_seq_len) if w > CAP]
    last = min(max_seq_len, CAP)
    assert [w for w in warm if w <= CAP] == sorted(set(main_warmup_lengths(last)) | set(range(8192, last + 1, 8192)))
    # every bucket <= max_seq_len that a prompt of <= max_seq_len tokens can get is warmed (as main: a bucket above
    # max_seq_len, e.g. 65536 for max_seq_len 45056, is not)
    buckets = {get_padded_prefill_len(s, 8192) for s in list(range(1, max_seq_len + 1, 61)) + [max_seq_len]}
    assert {b for b in buckets if b <= max_seq_len} <= set(warm)


# --------------------------------------------------------------------------------------------------------------
# RMSNorm eps: decode and S > cap prefill -> main's; S <= cap prefill -> Qwen3's HF eps
# --------------------------------------------------------------------------------------------------------------
def _host_rmsnorm(dim=128):
    import torch

    from models.common.rmsnorm import RMSNorm

    norm = RMSNorm(device=None, dim=dim, state_dict={"n.weight": torch.ones(dim)}, weight_key="n", state_dict_prefix="")
    norm.weight_distributed = norm.weight  # host stand-in for the distributed copy DistributedNorm passes on
    return norm


def test_rmsnorm_default_is_main_eps():
    from models.common.rmsnorm import RMSNorm

    assert inspect.signature(RMSNorm.__init__).parameters["eps"].default == MAIN_RMSNORM_EPS


def test_qwen_hf_eps(qwen_args):
    assert qwen_args.norm_eps == QWEN_HF_EPS


@pytest.mark.parametrize("S", SWEEP)
def test_prefill_norm_selection(S):
    from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import attach_prefill_opt_eps, prefill_norm, prefill_norm_eps

    norm = _host_rmsnorm()
    attach_prefill_opt_eps(norm, QWEN_HF_EPS)
    assert norm.eps == MAIN_RMSNORM_EPS  # decode reads norm itself
    twin = prefill_norm(norm, S)
    assert prefill_norm_eps(norm, S) == twin.eps == (QWEN_HF_EPS if S <= CAP else MAIN_RMSNORM_EPS)
    assert twin.weight is norm.weight  # shared weights, no second copy
    plain = _host_rmsnorm()  # no twin attached (Llama, test stand-ins): always its own eps
    assert prefill_norm(plain, S) is plain


@pytest.mark.parametrize("S", [4096, 40960, 45056, 65536])
def test_distributed_norm_eps_per_mode(wormhole_arch, monkeypatch, S):
    """DistributedNorm (attention_norm, ff_norm, final norm) passes the eps the mode / S selects to the norm ops."""
    from models.demos.llama3_70b_galaxy.tt import distributed_norm as dn_mod

    seen = []

    def spy(*a, epsilon=None, **k):
        seen.append(epsilon)
        return "out", None

    monkeypatch.setattr(dn_mod, "tt_distributed_rmsnorm", spy)
    monkeypatch.setattr(dn_mod, "tt_sharded_distributed_rmsnorm", spy)
    args = SimpleNamespace(
        is_qwen=True, norm_eps=QWEN_HF_EPS, qk_norm=True, dim=5120, blackhole_no_prefetcher=False, mesh_device=None
    )
    dn = dn_mod.DistributedNorm(_host_rmsnorm(), args, use_sharded_decode=True)
    x = SimpleNamespace(shape=(1, 1, S, 1280))
    dn.forward(x, None, "decode")
    dn.forward(x, None, "prefill")  # final norm: S from the tensor
    dn.forward(SimpleNamespace(shape=(1, 1, S // 8, 1280)), None, "prefill", seq_len=S)  # decoder, seq-parallel
    want = QWEN_HF_EPS if S <= CAP else MAIN_RMSNORM_EPS
    assert seen == [MAIN_RMSNORM_EPS, want, want]
    llama = dn_mod.DistributedNorm(
        _host_rmsnorm(), SimpleNamespace(**{**vars(args), "is_qwen": False}), use_sharded_decode=True
    )
    seen.clear()
    llama.forward(x, None, "prefill")
    llama.forward(x, None, "decode")
    assert seen == [MAIN_RMSNORM_EPS, MAIN_RMSNORM_EPS]  # Llama unchanged


def _src(name):
    return (TT / name).read_text()


def test_norms_built_as_main():
    """attention_norm / ff_norm / final norm / q_norm / k_norm are built with main's arguments (no eps keyword):
    decode reads norm.eps, so decode is main's. q_norm / k_norm get their prefill twin in llama_attention."""
    for name in ("llama_attention.py", "llama_decoder.py", "llama_model.py"):
        for node in ast.walk(ast.parse(_src(name))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RMSNorm":
                kws = {k.arg for k in node.keywords}
                assert "eps" not in kws and None not in kws, (name, node.lineno)
    src = _src("llama_attention.py")
    assert "attach_prefill_opt_eps(self.q_norm, configuration.norm_eps)" in src
    assert "attach_prefill_opt_eps(self.k_norm, configuration.norm_eps)" in src


def test_attention_prefill_reads_eps_through_prefill_norm():
    """Prefill code of llama_attention / qwen_cp_prefill never reads q_norm / k_norm eps or calls them directly;
    decode code does (main's eps)."""
    tree = ast.parse(_src("llama_attention.py"))
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        body = ast.unparse(fn)
        direct = any(f"self.{n}.eps" in body or f"self.{n}(" in body for n in ("q_norm", "k_norm"))
        if fn.name == "forward_prefill" or fn.name.startswith(("_qwen_", "_k_prep")):
            assert not direct, fn.name
        elif fn.name.startswith("_apply_decode_qk_norm_"):  # the three decode q/k norm paths
            assert "self.q_norm" in body and "self.k_norm" in body and "prefill_norm" not in body, fn.name
    cp_src = _src("qwen_cp_prefill.py")
    assert ".q_norm.eps" not in cp_src and ".k_norm.eps" not in cp_src and "norm.norm.eps" not in cp_src


# --------------------------------------------------------------------------------------------------------------
# Inline gates (model forward code): every Qwen long-prefill condition also checks the cap
# --------------------------------------------------------------------------------------------------------------
def _names(node):
    return {getattr(n, "attr", None) or getattr(n, "id", None) for n in ast.walk(node)}


def _is_long_seq_compare(node):
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.ops[0], ast.Gt)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value in (4096, 8192)
    )


@pytest.mark.parametrize("name", ["llama_attention.py", "llama_mlp.py", "llama_model.py", "llama_decoder.py"])
def test_inline_qwen_long_prefill_gates_are_capped(name):
    gates = 0
    for node in ast.walk(ast.parse(_src(name))):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            if "is_qwen" in _names(node) and any(_is_long_seq_compare(v) for v in node.values):
                gates += 1
                # qwen_prefill_opt_ok: capped on the total context (cached prefix + S), not S alone
                assert "qwen_prefill_opt_ok" in _names(node), (name, node.lineno, ast.unparse(node))
    assert gates >= {"llama_attention.py": 1, "llama_mlp.py": 2, "llama_model.py": 1, "llama_decoder.py": 0}[name]


def test_ff2_gather_reduce_gate_is_capped():
    """llama_ccl line_all_reduce: the Qwen FF2 rank-ordered gather-reduce (main's FF2 path at every length) is kept
    above the traced lengths only for S <= cap; above the cap it is main's path again."""
    tree = ast.parse(_src("llama_ccl.py"))
    ifs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and any("_prefill_ff2_gather_reduce" in ast.unparse(b) for b in n.body)
    ]
    inner = [n for n in ifs if "total_len" in ast.unparse(n.test)]
    assert len(inner) == 1 and "qwen_prefill_opt_ok(self.model_config, total_len)" in ast.unparse(inner[0].test)
