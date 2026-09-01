# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder tests for zai-org/GLM-4.7-Flash on one Blackhole chip.

Same acceptance contract as the functional/fused suites (PCC >= 0.995 vs the
HF fp32 reference layer, router sub-ulp ties exempt per step but counted in
the aggregate; bf4-expert deployment arm at 0.99), exercised against
``OptimizedDecoder``. Extra coverage over the fused suite:

- a config audit that proves the optimized path is actually constructed
  (DRAM-sharded decode weights, LoFi decode kernels, tuned sparse configs) so
  these tests cannot silently pass against a functional fallback;
- bfloat8_b latent-cache arm (prefill fill-cast + decode update + flash read),
  the dtype the full-model 202k-context capacity projection relies on;
- traced decode + stress and batch-8/32 union-path coverage as in the fused
  suite, now over the sharded-layout decode path.

Synthetic-weight tests run the comparability arm (bf8 experts AND bf8
attention weights, bar 0.995) so they measure the optimized layout/config/
fidelity work against the same bars as the functional/fused suites; the
deployment dtype arm (bf4 experts + bf4 attention + bf8 cache) runs under
``real_weights`` (bar 0.99) where the real checkpoint's quantization behavior
is the relevant evidence (synthetic gaussian weights lose ~2x more to bf4
block quantization; see doc/optimized_decoder/work_log.md, OPT-012).
"""

import inspect

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tests.test_functional_decoder import (
    BF4_BAR,
    PCC_BAR,
    Harness,
    _assert_decode_steps,
)
from models.autoports.zai_org_glm_4_7_flash.tt import optimized_decoder as opt_module
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def cfg():
    return utils.hf_config()


def opt_harness(device, cfg, kind, attn_dtype=ttnn.bfloat8_b, mlp_dtype=ttnn.bfloat8_b, **kw):
    """attn_dtype/mlp_dtype: bf8 = comparability arm (synthetic tests);
    None = keep the class defaults (bf4 deployment policy, used by the
    real-weight deployment test)."""
    saved = (OptimizedDecoder.attn_weight_dtype, OptimizedDecoder.mlp_gateup_dtype, OptimizedDecoder.mlp_down_dtype)
    if attn_dtype is not None:
        OptimizedDecoder.attn_weight_dtype = attn_dtype
    if mlp_dtype is not None:
        OptimizedDecoder.mlp_gateup_dtype = mlp_dtype
        OptimizedDecoder.mlp_down_dtype = mlp_dtype
    try:
        return Harness(device, cfg, kind, decoder_cls=OptimizedDecoder, **kw)
    finally:
        (
            OptimizedDecoder.attn_weight_dtype,
            OptimizedDecoder.mlp_gateup_dtype,
            OptimizedDecoder.mlp_down_dtype,
        ) = saved


@pytest.fixture(scope="module")
def moe_synth(device, cfg):
    return opt_harness(device, cfg, "moe")


@pytest.fixture(scope="module")
def dense_synth(device, cfg):
    return opt_harness(device, cfg, "dense")


def harness_for(kind, moe_synth, dense_synth):
    return moe_synth if kind == "moe" else dense_synth


# --------------------------------------------------------------------- config audit


def test_optimized_path_constructed(moe_synth, dense_synth):
    """The optimized artifacts must actually exist on the built layers: this
    guards the whole suite against silently exercising an inherited
    functional/fused fallback path."""
    moe, dense = moe_synth.dec, dense_synth.dec
    # class defaults = deployment policy (bf4 attention + bf4 MLP); the synth
    # fixtures build the bf8 comparability arm.
    assert OptimizedDecoder.attn_weight_dtype == ttnn.bfloat4_b
    assert OptimizedDecoder.mlp_gateup_dtype == ttnn.bfloat4_b  # shared expert
    assert OptimizedDecoder.mlp_down_dtype == ttnn.bfloat4_b
    assert OptimizedDecoder.dense_mlp_dtype is None  # dense MLP stays bf8 (202k control evidence)
    for dec in (moe, dense):
        # decode DRAM-sharded weight copies
        for name in ("wqkv_a_ds", "wq_b_ds", "wo_ds"):
            w = getattr(dec, name)
            assert w.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED, name
            assert w.memory_config().buffer_type == ttnn.BufferType.DRAM, name
            assert w.dtype == ttnn.bfloat8_b, name  # comparability-arm fixture
        assert dec.shared_down_ds.dtype == ttnn.bfloat8_b if dec.layer_kind == "moe" else True
        # the flat interleaved wq_b was replaced by the DRAM-sharded copy
        assert not hasattr(dec, "wq_b")
        # LoFi decode kernel configs (policy must reach the constructor)
        assert dec.ck_attn.math_fidelity == ttnn.MathFidelity.LoFi, dec.ck_attn.math_fidelity
        assert dec.ck_mlp.math_fidelity == ttnn.MathFidelity.LoFi, dec.ck_mlp.math_fidelity
        assert dec.ck_expert.math_fidelity == ttnn.MathFidelity.LoFi, dec.ck_expert.math_fidelity
    assert moe.shared_down_ds.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert dense.dense_down_ds.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    # tuned sparse configs: indexed decode (osw=2 allowed) vs union (osw=1 only:
    # the non-indexed sparsity walk corrupts multi-group outputs at osw>1)
    assert moe.sparse_gu_pc.per_core_N == 2 and moe.sparse_gu_pc.in0_block_w == 8
    assert moe.sparse_gu_pc_union.out_subblock_w == 1
    assert moe.sparse_dn_pc_union.out_subblock_w == 1
    # prefill sparse configs route through the tuned geometries
    pf_gu = moe._sparse_pc(32, 2 * moe.moe_inter, moe.hidden)
    assert pf_gu.in0_block_w == 32 and pf_gu.out_subblock_w == 1
    pf_dn = moe._sparse_pc(1024, moe.hidden, moe.moe_inter)
    assert pf_dn.in0_block_w == 24 and pf_dn.out_subblock_w == 1
    print("optimized-path construction audit OK")


# --------------------------------------------------------------------- prefill PCC


@pytest.mark.parametrize(
    "kind,S",
    # moe: tiny non-aligned, page boundary, just-past-page, mid, exactly one
    # chunk, just-past-chunk, long non-divisible multi-chunk
    [("moe", s) for s in (17, 64, 65, 512, 1024, 1057, 3000)] + [("dense", s) for s in (17, 512, 3000)],
)
def test_prefill_pcc(moe_synth, dense_synth, kind, S):
    h = harness_for(kind, moe_synth, dense_synth)
    x = utils.synth_activations(h.cfg, h.layer_idx, S, seed=7)
    ref = utils.hf_forward(h.cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=S)
    got = h.prefill(x, cache, pt, seq_len=S)
    assert got.shape[0] == S, f"logical output length {got.shape[0]} != {S}"
    p = utils.pcc(ref[0], got[:S])
    print(f"[{kind}] optimized prefill S={S} PCC={p:.6f}")
    assert p >= PCC_BAR, f"prefill PCC {p:.6f} < {PCC_BAR}"
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- decode PCC


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_decode_pcc(moe_synth, dense_synth, kind):
    h = harness_for(kind, moe_synth, dense_synth)
    S, n_steps = 509, 8  # non-aligned prefill length; decode crosses page boundary at 512
    x = utils.synth_activations(h.cfg, h.layer_idx, S + n_steps, seed=7)
    ref = utils.hf_forward(h.cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=11)
    p_prefill = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    assert p_prefill >= PCC_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt)
    ttnn.deallocate(cache)


def test_decode_cache_content(moe_synth):
    """Paged latent cache bytes vs the exact fp32 linear reference through a
    permuted page table (the sharded-layout rewrites must not move bytes)."""
    h = moe_synth
    S = 200
    x = utils.synth_activations(h.cfg, h.layer_idx, S, seed=13)
    cache, pt, pt_torch = h.fresh_cache(seed=17)
    h.prefill(x, cache, pt, seq_len=S)
    cache_torch = ttnn.to_torch(cache).float()
    got = utils.gather_user_cache(cache_torch, pt_torch, 0, S, h.dec.paged_config.block_size)
    want = utils.torch_latent_cache_reference(h.cfg, h.sd, x[0])
    p = utils.pcc(want, got)
    print(f"optimized cache PCC={p:.6f}")
    assert p >= 0.999
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- bf8 latent cache arm


def test_bf8_cache_prefill_decode(device, cfg):
    """Deployment cache dtype: bfloat8_b latent cache end-to-end (prefill
    fill-cast, decode bf16-input paged_update_cache, flash prefill+decode
    reading the bf8 cache). This is the dtype the context contract's
    full-model 202k capacity projection assumes."""
    h = opt_harness(device, cfg, "moe")
    S, n_steps = 509, 4
    x = utils.synth_activations(cfg, h.layer_idx, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache = h.dec.allocate_kv_cache(dtype=ttnn.bfloat8_b)
    assert cache.dtype == ttnn.bfloat8_b
    pt_torch = utils.make_page_table(1, h.dec.paged_config.max_num_blocks, seed=61)
    pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[moe] optimized bf8-cache prefill S={S} PCC={p:.6f}")
    assert p >= PCC_BAR
    # cache content vs linear reference (bf8 quantization included)
    cache_torch = ttnn.to_torch(cache).float()
    got_rows = utils.gather_user_cache(cache_torch, pt_torch, 0, S, h.dec.paged_config.block_size)
    want = utils.torch_latent_cache_reference(cfg, h.sd, x[0, :S])
    p_cache = utils.pcc(want, got_rows)
    print(f"[moe] bf8 cache content PCC={p_cache:.6f}")
    assert p_cache >= 0.998
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt)
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- batch decode


def test_decode_batch_mixed_positions(device, cfg):
    """Batch-8 decode (union-sparsity path, osw=1 configs): users at different
    non-aligned positions, permuted pages."""
    B = 8
    lens = [33, 64, 96, 130, 200, 257, 300, 380]
    h = opt_harness(device, cfg, "moe", max_batch=B)
    xs = [utils.synth_activations(cfg, 1, L + 2, seed=100 + u) for u, L in enumerate(lens)]
    refs = [utils.hf_forward(cfg, h.hf_layer, x) for x in xs]
    cache, pt, _ = h.fresh_cache(batch=B, seed=23)
    for u, (L, x) in enumerate(zip(lens, xs)):
        p = utils.pcc(refs[u][0, :L], h.prefill(x, cache, pt, user_id=u, seq_len=L))
        assert p >= PCC_BAR, f"user {u} prefill PCC {p:.5f}"
    ties = [utils.router_tie_positions(cfg, h.hf_layer, x) for x in xs]
    for step in range(2):
        rows = torch.stack([xs[u][0, lens[u] + step] for u in range(B)])
        got = h.decode_step(rows, [lens[u] + step for u in range(B)], cache, pt)
        for u in range(B):
            p = utils.pcc(refs[u][0, lens[u] + step], got[u])
            if p < PCC_BAR:
                assert lens[u] + step in ties[u], f"user {u} step {step} PCC {p:.5f}"
            print(f"user {u} pos {lens[u]+step} PCC={p:.5f}")
    ttnn.deallocate(cache)


def test_decode_batch32(device, cfg):
    """Largest decode batch: 32 users; union path + two-group cache update
    under the optimized sharded layouts."""
    B = 32
    lens = [33 + 7 * u for u in range(B)]  # 33..250, mostly non-aligned
    h = opt_harness(device, cfg, "moe", max_batch=B)
    xs = [utils.synth_activations(cfg, 1, L + 1, seed=300 + u) for u, L in enumerate(lens)]
    refs = [utils.hf_forward(cfg, h.hf_layer, x) for x in xs]
    cache, pt, _ = h.fresh_cache(batch=B, seed=29)
    for u, (L, x) in enumerate(zip(lens, xs)):
        p = utils.pcc(refs[u][0, :L], h.prefill(x, cache, pt, user_id=u, seq_len=L))
        assert p >= PCC_BAR, f"user {u} prefill PCC {p:.5f}"
    rows = torch.stack([xs[u][0, lens[u]] for u in range(B)])
    got = h.decode_step(rows, lens, cache, pt)
    ok, tie_exempt = 0, 0
    for u in range(B):
        p = utils.pcc(refs[u][0, lens[u]], got[u])
        if p >= PCC_BAR:
            ok += 1
        else:
            ties = utils.router_tie_positions(cfg, h.hf_layer, xs[u])
            assert lens[u] in ties, f"user {u} PCC {p:.5f} not a tie"
            tie_exempt += 1
    print(f"optimized batch32 decode: {ok} users >= {PCC_BAR}, {tie_exempt} tie-exempt")
    assert ok >= B - 4
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- traced decode


def test_decode_traced_and_deterministic(device, cfg, moe_synth):
    """Decode via trace capture/replay (the whole optimized step is inside the
    trace: sharded norms, DRAM-sharded matmuls, on-device topk -> index list,
    embedding-table routing-weight pick): PCC per replay, plus bit-identical
    output when the same inputs are replayed."""
    h = moe_synth
    S, n_steps = 128, 4
    x = utils.synth_activations(cfg, h.layer_idx, S + n_steps + 1, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=31)
    assert utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S)) >= PCC_BAR

    def host_inputs(pos):
        return (
            ttnn.from_torch(
                x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(torch.tensor([pos], dtype=torch.int32)),
            ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32)),
        )

    hx, hp, hr = host_inputs(S)
    x_dev, pos_dev, rot_dev = hx.to(device), hp.to(device), hr.to(device)

    out_c = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.deallocate(out_c)  # compile pass (writes pos S)

    hx, hp, hr = host_inputs(S + 1)
    ttnn.copy_host_to_device_tensor(hx, x_dev)
    ttnn.copy_host_to_device_tensor(hp, pos_dev)
    ttnn.copy_host_to_device_tensor(hr, rot_dev)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_t = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ties = utils.router_tie_positions(cfg, h.hf_layer, x)
    got = None
    for i in range(1, n_steps):
        pos = S + i
        hx, hp, hr = host_inputs(pos)
        ttnn.copy_host_to_device_tensor(hx, x_dev)
        ttnn.copy_host_to_device_tensor(hp, pos_dev)
        ttnn.copy_host_to_device_tensor(hr, rot_dev)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(out_t).float()[0, 0, 0]
        p = utils.pcc(ref[0, pos], got)
        print(f"optimized traced replay pos={pos} PCC={p:.6f}")
        if p < PCC_BAR:
            assert pos in ties
    # determinism: replay identical inputs -> bit-identical output
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    got2 = ttnn.to_torch(out_t).float()[0, 0, 0]
    assert torch.equal(got, got2), "optimized traced decode not deterministic for identical inputs"
    ttnn.release_trace(device, tid)
    ttnn.deallocate(cache)


def test_decode_traced_stress(device, cfg, moe_synth):
    """Stress: 96 trace replays across changing positions/expert selections
    (three sweeps over 32 positions), checking PCC-or-tie at every step and
    bitwise repeatability at the final position."""
    h = moe_synth
    S, span = 96, 32
    x = utils.synth_activations(cfg, h.layer_idx, S + span + 1, seed=53)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=59)
    assert utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S)) >= PCC_BAR

    def host_inputs(pos):
        return (
            ttnn.from_torch(
                x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(torch.tensor([pos], dtype=torch.int32)),
            ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32)),
        )

    hx, hp, hr = host_inputs(S)
    x_dev, pos_dev, rot_dev = hx.to(device), hp.to(device), hr.to(device)
    out_c = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.deallocate(out_c)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_t = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ties = utils.router_tie_positions(cfg, h.hf_layer, x)
    below_bar = 0
    for sweep in range(3):
        for i in range(span):
            pos = S + i
            hx, hp, hr = host_inputs(pos)
            ttnn.copy_host_to_device_tensor(hx, x_dev)
            ttnn.copy_host_to_device_tensor(hp, pos_dev)
            ttnn.copy_host_to_device_tensor(hr, rot_dev)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = ttnn.to_torch(out_t).float()[0, 0, 0]
            p = utils.pcc(ref[0, pos], got)
            if p < PCC_BAR:
                below_bar += 1
                assert pos in ties, f"sweep {sweep} pos {pos} PCC {p:.5f} below bar and not a router tie"
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ref_out = ttnn.to_torch(out_t).float()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    assert torch.equal(ref_out, ttnn.to_torch(out_t).float()), "stress replays not deterministic"
    print(f"optimized traced stress: 96 replays OK ({below_bar} tie-exempt)")
    ttnn.release_trace(device, tid)
    ttnn.deallocate(cache)


def test_decode_traced_dense(device, cfg, dense_synth):
    """Dense-layer traced decode PCC + bitwise determinism (the moe traced
    tests share the attention path; this covers the dense MLP trace)."""
    h = dense_synth
    S, n_steps = 128, 3
    x = utils.synth_activations(cfg, h.layer_idx, S + n_steps + 1, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=67)
    assert utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S)) >= PCC_BAR

    def host_inputs(pos):
        return (
            ttnn.from_torch(
                x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(torch.tensor([pos], dtype=torch.int32)),
            ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32)),
        )

    hx, hp, hr = host_inputs(S)
    x_dev, pos_dev, rot_dev = hx.to(device), hp.to(device), hr.to(device)
    out_c = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.deallocate(out_c)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_t = h.dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    got = None
    for i in range(1, n_steps):
        pos = S + i
        hx, hp, hr = host_inputs(pos)
        ttnn.copy_host_to_device_tensor(hx, x_dev)
        ttnn.copy_host_to_device_tensor(hp, pos_dev)
        ttnn.copy_host_to_device_tensor(hr, rot_dev)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(out_t).float()[0, 0, 0]
        p = utils.pcc(ref[0, pos], got)
        print(f"optimized dense traced replay pos={pos} PCC={p:.6f}")
        assert p >= PCC_BAR  # no routing discreteness on the dense layer
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    assert torch.equal(got, ttnn.to_torch(out_t).float()[0, 0, 0]), "dense traced decode not deterministic"
    ttnn.release_trace(device, tid)
    ttnn.deallocate(cache)


def test_decode_batch8_bf8_cache(device, cfg):
    """Batch>1 union path crossed with the bf8 latent cache (two dtype axes
    that otherwise only meet at batch 1)."""
    B = 8
    lens = [33, 64, 96, 130, 200, 257, 300, 380]
    h = opt_harness(device, cfg, "moe", max_batch=B)
    xs = [utils.synth_activations(cfg, 1, L + 1, seed=100 + u) for u, L in enumerate(lens)]
    refs = [utils.hf_forward(cfg, h.hf_layer, x) for x in xs]
    cache = h.dec.allocate_kv_cache(dtype=ttnn.bfloat8_b)
    pt_torch = utils.make_page_table(B, h.dec.paged_config.max_num_blocks // B, seed=71)
    pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    for u, (L, x) in enumerate(zip(lens, xs)):
        p = utils.pcc(refs[u][0, :L], h.prefill(x, cache, pt, user_id=u, seq_len=L))
        assert p >= PCC_BAR, f"user {u} prefill PCC {p:.5f}"
    ties = [utils.router_tie_positions(cfg, h.hf_layer, x) for x in xs]
    rows = torch.stack([xs[u][0, lens[u]] for u in range(B)])
    got = h.decode_step(rows, lens, cache, pt)
    for u in range(B):
        p = utils.pcc(refs[u][0, lens[u]], got[u])
        if p < PCC_BAR:
            assert lens[u] in ties[u], f"user {u} PCC {p:.5f}"
        print(f"bf8-cache user {u} pos {lens[u]} PCC={p:.5f}")
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- determinism


def test_prefill_deterministic(moe_synth):
    h = moe_synth
    S = 200
    x = utils.synth_activations(h.cfg, h.layer_idx, S, seed=37)
    cache1, pt1, _ = h.fresh_cache(seed=41)
    out1 = h.prefill(x, cache1, pt1, seq_len=S)
    ttnn.deallocate(cache1)
    cache2, pt2, _ = h.fresh_cache(seed=41)
    out2 = h.prefill(x, cache2, pt2, seq_len=S)
    ttnn.deallocate(cache2)
    assert torch.equal(out1, out2), "optimized prefill not deterministic for identical inputs"


# --------------------------------------------------------------------- fallback audit


def test_runtime_no_host_fallback(moe_synth, monkeypatch):
    """No torch / from_torch / to_torch / as_tensor inside the optimized
    prefill or decode passes; the module imports torch only inside setup."""
    h = moe_synth
    S = 64
    x = utils.synth_activations(h.cfg, h.layer_idx, S + 1, seed=43)
    cache, pt, _ = h.fresh_cache(seed=43)
    x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=h.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    pos = torch.tensor([S], dtype=torch.int32)
    xd = ttnn.from_torch(
        x[:, S : S + 1].unsqueeze(0).permute(0, 2, 1, 3), device=h.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    cur = ttnn.from_torch(pos, device=h.device)
    rot = ttnn.from_torch(pos.unsqueeze(0).to(torch.uint32), device=h.device)

    def tripwire(name):
        def fn(*a, **k):
            raise AssertionError(f"host boundary {name} called inside a forward pass")

        return fn

    for name in ("from_torch", "to_torch", "as_tensor"):
        monkeypatch.setattr(ttnn, name, tripwire(f"ttnn.{name}"))
    out_p = h.dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
    out_d = h.dec.decode_forward(xd, kv_cache=cache, page_table=pt, cur_pos_tensor=cur, rot_idxs=rot)
    monkeypatch.undo()
    assert ttnn.to_torch(out_p).shape[2] == S
    assert ttnn.to_torch(out_d) is not None
    ttnn.deallocate(cache)

    # static audit: torch only imported inside setup-time functions
    src = inspect.getsource(opt_module)
    module_level_imports = [line for line in src.splitlines() if line.startswith("import ") or line.startswith("from ")]
    assert not any("torch" in line for line in module_level_imports), module_level_imports


# --------------------------------------------------------------------- real weights


@pytest.mark.real_weights
@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_real_weights_prefill_decode(device, cfg, kind):
    h = opt_harness(device, cfg, kind, real=True)
    S, n_steps = 512, 8
    x = utils.synth_activations(cfg, h.layer_idx, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=47)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[{kind}] optimized REAL weights prefill S={S} PCC={p:.6f}")
    assert p >= PCC_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt)
    ttnn.deallocate(cache)


@pytest.mark.real_weights
def test_expert_bf4_bf8cache_real(device, cfg):
    """Deployment arm end-to-end: bf4 routed experts + bf8 latent cache on
    real checkpoint weights (bar 0.99, doc/probe/README.md)."""
    h = opt_harness(device, cfg, "moe", attn_dtype=None, mlp_dtype=None, real=True, expert_dtype=ttnn.bfloat4_b)
    assert h.dec.experts_gate_up.dtype == ttnn.bfloat4_b
    assert h.dec.experts_down.dtype == ttnn.bfloat4_b
    assert h.dec.wqkv_a_ds.dtype == ttnn.bfloat4_b  # deployment attention arm
    assert h.dec.w_uk.dtype == ttnn.bfloat4_b and h.dec.w_uv_t.dtype == ttnn.bfloat4_b
    assert h.dec.shared_gate.dtype == ttnn.bfloat4_b and h.dec.shared_down_ds.dtype == ttnn.bfloat4_b
    S, n_steps = 512, 8
    x = utils.synth_activations(cfg, 1, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache = h.dec.allocate_kv_cache(dtype=ttnn.bfloat8_b)
    pt_torch = utils.make_page_table(1, h.dec.paged_config.max_num_blocks, seed=53)
    pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[moe] optimized REAL bf4 experts + bf8 cache prefill S={S} PCC={p:.6f}")
    assert p >= BF4_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt, bar=BF4_BAR)
    ttnn.deallocate(cache)
