# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder tests for zai-org/GLM-4.7-Flash on one Blackhole chip.

Same acceptance contract as the functional-decoder suite (PCC >= 0.995 vs the
HF fp32 reference layer, router sub-ulp ties exempt per step but counted in
the aggregate), exercised against ``FusedDecoder`` — the graph-fused
implementation. Extra coverage over the functional suite:

- fused-vs-functional equivalence on the routing-free dense layer (tight bar);
- traced-decode stress (many replays + mid-run PCC + bitwise determinism);
- repeated-prefill determinism across separately allocated caches.
"""

import inspect

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tests.test_functional_decoder import (
    BF4_BAR,
    CHUNK,
    MAX_CONTEXT,
    PCC_BAR,
    Harness,
    _assert_decode_steps,
)
from models.autoports.zai_org_glm_4_7_flash.tt import fused_decoder as fused_module
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import FunctionalDecoder
from models.autoports.zai_org_glm_4_7_flash.tt.fused_decoder import FusedDecoder


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def cfg():
    return utils.hf_config()


def fused_harness(device, cfg, kind, **kw):
    return Harness(device, cfg, kind, decoder_cls=FusedDecoder, **kw)


@pytest.fixture(scope="module")
def moe_synth(device, cfg):
    return fused_harness(device, cfg, "moe")


@pytest.fixture(scope="module")
def dense_synth(device, cfg):
    return fused_harness(device, cfg, "dense")


def harness_for(kind, moe_synth, dense_synth):
    return moe_synth if kind == "moe" else dense_synth


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
    print(f"[{kind}] fused prefill S={S} PCC={p:.6f}")
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
    permuted page table (the fused kv-path rewrites must not move bytes)."""
    h = moe_synth
    S = 200
    x = utils.synth_activations(h.cfg, h.layer_idx, S, seed=13)
    cache, pt, pt_torch = h.fresh_cache(seed=17)
    h.prefill(x, cache, pt, seq_len=S)
    cache_torch = ttnn.to_torch(cache).float()
    got = utils.gather_user_cache(cache_torch, pt_torch, 0, S, h.dec.paged_config.block_size)
    want = utils.torch_latent_cache_reference(h.cfg, h.sd, x[0])
    p = utils.pcc(want, got)
    print(f"fused cache PCC={p:.6f}")
    assert p >= 0.999
    ttnn.deallocate(cache)


def test_decode_batch_mixed_positions(device, cfg):
    """Batch-8 decode (union-sparsity path): users at different non-aligned
    positions, permuted pages."""
    B = 8
    lens = [33, 64, 96, 130, 200, 257, 300, 380]
    h = fused_harness(device, cfg, "moe", max_batch=B)
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
    """Largest decode batch: 32 users (tile-width limit of the decode row);
    exercises the union path and the two-group paged_update_cache split."""
    B = 32
    lens = [33 + 7 * u for u in range(B)]  # 33..250, mostly non-aligned
    h = fused_harness(device, cfg, "moe", max_batch=B)
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
    print(f"fused batch32 decode: {ok} users >= {PCC_BAR}, {tie_exempt} tie-exempt")
    assert ok >= B - 4
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- fused vs functional


def test_fused_matches_functional_dense(device, cfg):
    """Routing-free equivalence: the dense layer has no top-k discreteness, so
    fused and functional outputs must agree tightly (the rewrites only reorder
    exact linear algebra and change activation rounding sites)."""
    kind, S = "dense", 512
    layer_idx = utils.LAYER_KINDS[kind]
    sd = utils.synth_layer_state_dict(cfg, layer_idx)
    outs = {}
    for cls in (FunctionalDecoder, FusedDecoder):
        dec = cls.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=device,
            max_batch_size=1,
            max_context=MAX_CONTEXT,
            prefill_chunk_size=CHUNK,
        )
        x = utils.synth_activations(cfg, layer_idx, S + 2, seed=7)
        cache = dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(1, dec.paged_config.max_num_blocks, seed=3)
        pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
        prefill = ttnn.to_torch(out).float()[0, 0]
        pos = S
        xd = ttnn.from_torch(
            x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
        rot = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)
        od = dec.decode_forward(xd, kv_cache=cache, page_table=pt, cur_pos_tensor=cur, rot_idxs=rot)
        decode = ttnn.to_torch(od).float()[0, 0, 0]
        outs[cls.__name__] = (prefill, decode)
        ttnn.deallocate(cache)
    p_prefill = utils.pcc(outs["FunctionalDecoder"][0], outs["FusedDecoder"][0])
    p_decode = utils.pcc(outs["FunctionalDecoder"][1], outs["FusedDecoder"][1])
    print(f"fused-vs-functional dense: prefill PCC={p_prefill:.6f} decode PCC={p_decode:.6f}")
    assert p_prefill >= 0.9995
    assert p_decode >= 0.9995


# --------------------------------------------------------------------- traced decode


def test_decode_traced_and_deterministic(device, cfg, moe_synth):
    """Decode via trace capture/replay (indexed expert path is inside the
    trace, including the on-device topk -> index-list handoff): PCC per
    replay, plus bit-identical output when the same inputs are replayed."""
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
        print(f"fused traced replay pos={pos} PCC={p:.6f}")
        if p < PCC_BAR:
            assert pos in ties
    # determinism: replay identical inputs -> bit-identical output
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    got2 = ttnn.to_torch(out_t).float()[0, 0, 0]
    assert torch.equal(got, got2), "fused traced decode not deterministic for identical inputs"
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
    print(f"fused traced stress: 96 replays OK ({below_bar} tie-exempt)")
    ttnn.release_trace(device, tid)
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
    assert torch.equal(out1, out2), "fused prefill not deterministic for identical inputs"


# --------------------------------------------------------------------- fallback audit


def test_runtime_no_host_fallback(moe_synth, monkeypatch):
    """No torch / from_torch / to_torch / as_tensor inside the fused prefill
    or decode passes; the fused module imports torch only inside setup."""
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
    src = inspect.getsource(fused_module)
    module_level_imports = [line for line in src.splitlines() if line.startswith("import ") or line.startswith("from ")]
    assert not any("torch" in line for line in module_level_imports), module_level_imports


# --------------------------------------------------------------------- real weights


@pytest.mark.real_weights
@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_real_weights_prefill_decode(device, cfg, kind):
    h = fused_harness(device, cfg, kind, real=True)
    S, n_steps = 512, 8
    x = utils.synth_activations(cfg, h.layer_idx, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=47)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[{kind}] fused REAL weights prefill S={S} PCC={p:.6f}")
    assert p >= PCC_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt)
    ttnn.deallocate(cache)


@pytest.mark.real_weights
def test_expert_bf4_real(device, cfg):
    """Deployment-dtype arm: routed experts at bfloat4_b (mandatory from the
    full-model stage; see doc/probe/README.md). Layer-level bar 0.99."""
    h = fused_harness(device, cfg, "moe", real=True, expert_dtype=ttnn.bfloat4_b)
    assert h.dec.experts_gate_up.dtype == ttnn.bfloat4_b
    assert h.dec.experts_down.dtype == ttnn.bfloat4_b
    S, n_steps = 512, 8
    x = utils.synth_activations(cfg, 1, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=53)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[moe] fused REAL weights bf4 experts prefill S={S} PCC={p:.6f}")
    assert p >= BF4_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt, bar=BF4_BAR)
    ttnn.deallocate(cache)
