# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder tests for zai-org/GLM-4.7-Flash on one Blackhole chip.

Default tests use deterministic synthetic weights with real config shapes
(no HF download needed beyond tests/weight_stats.json). Tests marked
``real_weights`` load the actual checkpoint shards from the local HF snapshot.

PCC acceptance: >= 0.995 vs the HF fp32 reference layer (functional-decoder
default bar). Decode steps whose HF top-4 router selection is a sub-bf16-ulp
tie (see utils.router_tie_positions) are exempt from the per-step bar but
still counted in the aggregate.
"""

import inspect

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tt import functional_decoder as fd_module
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import FunctionalDecoder

PCC_BAR = 0.995
# Deployment-dtype arm: routed experts must be bfloat4_b from the full-model
# stage onward to fit 30.6B on one 32 GB chip (doc/probe/README.md). Measured
# full-layer real-weight PCC with bf4 experts is ~0.997 prefill / ~0.9975+
# decode; the bar for this arm is 0.99.
BF4_BAR = 0.99

MAX_CONTEXT = 4096
CHUNK = 1024


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def cfg():
    return utils.hf_config()


class Harness:
    """One layer kind + weight arm: TTNN decoder, HF reference layer, state dict."""

    def __init__(self, device, cfg, kind, *, real=False, expert_dtype=ttnn.bfloat8_b, max_batch=1):
        self.cfg = cfg
        self.kind = kind
        self.layer_idx = utils.LAYER_KINDS[kind]
        self.sd = (
            utils.load_real_layer_state_dict(cfg, self.layer_idx)
            if real
            else utils.synth_layer_state_dict(cfg, self.layer_idx)
        )
        self.dec = FunctionalDecoder.from_state_dict(
            self.sd,
            hf_config=cfg,
            layer_idx=self.layer_idx,
            mesh_device=device,
            max_batch_size=max_batch,
            max_context=MAX_CONTEXT,
            prefill_chunk_size=CHUNK,
            expert_dtype=expert_dtype,
        )
        self.hf_layer = utils.build_hf_layer(cfg, self.layer_idx, self.sd)
        self.device = device

    def fresh_cache(self, batch=1, seed=3):
        paged = self.dec.paged_config
        cache = self.dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(batch, paged.max_num_blocks // batch, seed=seed)
        pt = ttnn.from_torch(pt_torch, device=self.device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return cache, pt, pt_torch

    def prefill(self, x, cache, pt, user_id=0, seq_len=None):
        S = seq_len if seq_len is not None else x.shape[1]
        x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = self.dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=user_id, seq_len=S)
        res = ttnn.to_torch(out).float()[0, 0]
        ttnn.deallocate(out)
        ttnn.deallocate(x_tt)
        return res

    def decode_step(self, rows, positions, cache, pt):
        """rows: [B, H] token inputs, positions: list of int."""
        B = rows.shape[0]
        x_tt = ttnn.from_torch(rows[None, None], device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        pos = torch.tensor(positions, dtype=torch.int32)
        cur = ttnn.from_torch(pos, device=self.device)
        rot = ttnn.from_torch(pos.unsqueeze(0).to(torch.uint32), device=self.device)
        out = self.dec.decode_forward(x_tt, kv_cache=cache, page_table=pt, cur_pos_tensor=cur, rot_idxs=rot)
        res = ttnn.to_torch(out).float()[0, 0, :B]
        ttnn.deallocate(out)
        for t in (x_tt, cur, rot):
            ttnn.deallocate(t)
        return res


def _assert_decode_steps(h, ref, x, start, n_steps, cache, pt, bar=PCC_BAR):
    """Run n_steps single-user decode steps, assert per-step (tie-exempt) and
    aggregate PCC; returns [(pos, pcc)]."""
    ties = utils.router_tie_positions(h.cfg, h.hf_layer, x)
    per_step, gots, refs = [], [], []
    for i in range(n_steps):
        pos = start + i
        got = h.decode_step(x[0, pos : pos + 1], [pos], cache, pt)[0]
        p = utils.pcc(ref[0, pos], got)
        per_step.append((pos, p))
        gots.append(got)
        refs.append(ref[0, pos])
    agg = utils.pcc(torch.stack(refs), torch.stack(gots))
    print(f"[{h.kind}] decode steps: " + ", ".join(f"{pos}:{p:.5f}" for pos, p in per_step) + f" | agg={agg:.5f}")
    for pos, p in per_step:
        if p < bar:
            assert pos in ties, f"decode PCC {p:.5f} at pos {pos} below {bar} and not a router tie ({ties=})"
            print(f"  pos {pos} PCC {p:.5f} exempted: sub-ulp router tie (gap={ties[pos]:.2e})")
    assert agg >= bar, f"aggregate decode PCC {agg:.5f} < {bar}"
    return per_step


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def moe_synth(device, cfg):
    return Harness(device, cfg, "moe")


@pytest.fixture(scope="module")
def dense_synth(device, cfg):
    return Harness(device, cfg, "dense")


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
    print(f"[{kind}] prefill S={S} PCC={p:.6f}")
    assert p >= PCC_BAR, f"prefill PCC {p:.6f} < {PCC_BAR}"
    ttnn.deallocate(cache)


def test_prefill_rejects_overlong_input(moe_synth, expect_error):
    h = moe_synth
    x = utils.synth_activations(h.cfg, h.layer_idx, 128, seed=0)
    cache, pt, _ = h.fresh_cache()
    x_tt = ttnn.from_torch(x.unsqueeze(0), device=h.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(ValueError, "logical seq_len"):
        h.dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=17)
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
    """Paged latent cache bytes match the exact fp32 linear reference through a
    permuted page table (address/indexing check, not just attention output)."""
    h = moe_synth
    S = 200
    x = utils.synth_activations(h.cfg, h.layer_idx, S, seed=13)
    cache, pt, pt_torch = h.fresh_cache(seed=17)
    h.prefill(x, cache, pt, seq_len=S)
    cache_torch = ttnn.to_torch(cache).float()
    got = utils.gather_user_cache(cache_torch, pt_torch, 0, S, h.dec.paged_config.block_size)
    want = utils.torch_latent_cache_reference(h.cfg, h.sd, x[0])
    p = utils.pcc(want, got)
    print(f"cache PCC={p:.6f}")
    assert p >= 0.999
    ttnn.deallocate(cache)


def test_decode_batch_mixed_positions(device, cfg):
    """Batch-8 decode: users at different non-aligned positions, permuted pages."""
    B = 8
    lens = [33, 64, 96, 130, 200, 257, 300, 380]
    h = Harness(device, cfg, "moe", max_batch=B)
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
    """Largest decode batch: 32 users (tile-width limit of the decode row)."""
    B = 32
    lens = [33 + 7 * u for u in range(B)]  # 33..250, mostly non-aligned
    h = Harness(device, cfg, "moe", max_batch=B)
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
    print(f"batch32 decode: {ok} users >= {PCC_BAR}, {tie_exempt} tie-exempt")
    # Synthetic-weight router scores are worst-case clustered (sigmoid ~0.5);
    # measured sub-ulp tie rate is ~2-4% of tokens, so allow up to 4 of 32
    # tie-exempt users. Every exempted user is individually proven a tie above.
    assert ok >= B - 4
    ttnn.deallocate(cache)


# --------------------------------------------------------------------- traced decode


def test_decode_traced_and_deterministic(device, cfg, moe_synth):
    """Decode via ttnn trace capture/replay: PCC per replay, plus bit-identical
    output when the same inputs are replayed twice."""
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
        print(f"traced replay pos={pos} PCC={p:.6f}")
        if p < PCC_BAR:
            assert pos in ties
    # determinism: replay identical inputs -> bit-identical output
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    got2 = ttnn.to_torch(out_t).float()[0, 0, 0]
    assert torch.equal(got, got2), "traced decode not deterministic for identical inputs"
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
    assert torch.equal(out1, out2), "prefill not deterministic for identical inputs"


# --------------------------------------------------------------------- fallback audit


def test_runtime_no_host_fallback(moe_synth, monkeypatch):
    """No torch / from_torch / to_torch / as_tensor inside prefill or decode
    passes: host-boundary ttnn entry points are tripwired during the forwards,
    and the module file imports torch only inside setup-time functions."""
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
    src = inspect.getsource(fd_module)
    module_level_imports = [line for line in src.splitlines() if line.startswith("import ") or line.startswith("from ")]
    assert not any("torch" in line for line in module_level_imports), module_level_imports


# --------------------------------------------------------------------- real weights


@pytest.mark.real_weights
@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_real_weights_prefill_decode(device, cfg, kind):
    h = Harness(device, cfg, kind, real=True)
    S, n_steps = 512, 8
    x = utils.synth_activations(cfg, h.layer_idx, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=47)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[{kind}] REAL weights prefill S={S} PCC={p:.6f}")
    assert p >= PCC_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt)
    ttnn.deallocate(cache)


@pytest.mark.real_weights
def test_expert_bf4_real(device, cfg):
    """Deployment-dtype arm: routed experts at bfloat4_b (mandatory from the
    full-model stage; see doc/probe/README.md). Layer-level bar 0.99."""
    h = Harness(device, cfg, "moe", real=True, expert_dtype=ttnn.bfloat4_b)
    assert h.dec.experts_gate.dtype == ttnn.bfloat4_b
    assert h.dec.experts_down.dtype == ttnn.bfloat4_b
    S, n_steps = 512, 8
    x = utils.synth_activations(cfg, 1, S + n_steps, seed=7)
    ref = utils.hf_forward(cfg, h.hf_layer, x)
    cache, pt, _ = h.fresh_cache(seed=53)
    p = utils.pcc(ref[0, :S], h.prefill(x, cache, pt, seq_len=S))
    print(f"[moe] REAL weights bf4 experts prefill S={S} PCC={p:.6f}")
    assert p >= BF4_BAR
    _assert_decode_steps(h, ref, x, S, n_steps, cache, pt, bar=BF4_BAR)
    ttnn.deallocate(cache)
