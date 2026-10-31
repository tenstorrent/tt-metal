# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_LMHEAD_DSHARD: the DRAM-sharded LM head for decode-shaped head calls (tp_common.LMHeadDShard).

Host tests (no device): the env parsing and the shard arithmetic (Qwen3.8 TP8: 970 vocab tiles per device in
8 banks x 122 = 976 tiles, in0_block_w 2 like the plain head's auto 1D config; 2 readers/bank by default
because 1 reader/bank needs ~1.8 MB of L1 per storage core for the fp32 drafter head).

Device tests (one Blackhole chip, synthetic weights at the per-device TP shapes):
  * "exact" must reproduce the plain ``ttnn.linear(x, interleaved_weight)`` head BIT FOR BIT, for the base/verify
    call (bf16 out, default compute config) and for the drafter call (fp32 out, HiFi2 + fp32 Dest), at 1..32 rows;
  * "mc" (qb2 multi-core pipeline) is only checked for PCC and argmax agreement: it accumulates 8 K tiles per Dest
    pass, so it is not bit-identical by construction;
  * a microbenchmark prints the per-call device time of plain vs DRAM-sharded (incl. the reshard + copy-out).

K6 device tests (QWEN36_LMHEAD_DSHARD_OUT_PCN / _IBW / _FP32_DEST on the "mc" head; also runnable under ttsim-perf):
  * OUT_PCN keeps every logit's bits (it only re-lays the L1 logits shard);
  * numerics table of plain / mc / mc + FP32_DEST / the drafter's fp32 heads against an fp32 reference;
  * the drafter's fp32 mc call fits beside the all-reduce staging buffer with the production l1_small_size, eagerly
    and inside a trace, and an L1 ballast at the modelled margin +- 32 KB fits / is refused (LMHeadDShardPlan L1 model);
  * traced per-call time of plain and the mc variants (readers, in0_block_w, logits shard, FP32_DEST).

Run:
  pytest models/demos/blackhole/qwen36/tests/test_lmhead_dshard.py -k host -q                 # anywhere
  pytest models/demos/blackhole/qwen36/tests/test_lmhead_dshard.py -k device -v -s           # on a BH chip
"""
import time

import pytest
import torch

from models.demos.blackhole.qwen36.tt import tp_common as tpc

K = 5120
N_TP8 = 248320 // 8  # 31040 = 970 tiles
N_TP4 = 248320 // 4  # 62080 = 1940 tiles


# ---------------------------------------------------------------------------------------------- host


def test_host_mode_parsing(monkeypatch):
    monkeypatch.delenv(tpc.LMHEAD_DSHARD_ENV, raising=False)
    assert tpc.lm_head_dshard_mode() is None
    for raw, want in (("0", None), ("off", None), ("1", "exact"), ("exact", "exact"), ("MC", "mc")):
        monkeypatch.setenv(tpc.LMHEAD_DSHARD_ENV, raw)
        assert tpc.lm_head_dshard_mode() == want
    monkeypatch.setenv(tpc.LMHEAD_DSHARD_ENV, "2")
    with pytest.raises(ValueError):
        tpc.lm_head_dshard_mode()
    monkeypatch.delenv(tpc.LMHEAD_DSHARD_READERS_ENV, raising=False)
    assert tpc.lm_head_dshard_readers() is None
    monkeypatch.setenv(tpc.LMHEAD_DSHARD_READERS_ENV, "2")
    assert tpc.lm_head_dshard_readers() == 2


def test_host_readers_with_weight_prefetch(monkeypatch):
    """QWEN36_WEIGHT_PREFETCH on (its GCBs share L1) and READERS unset: "exact" takes 3 readers/bank; prefetch off,
    "mc", or an explicit READERS keep their old values."""
    from models.demos.blackhole.qwen36.tt import weight_prefetch

    monkeypatch.delenv(tpc.LMHEAD_DSHARD_READERS_ENV, raising=False)
    monkeypatch.delenv(weight_prefetch.ENV, raising=False)
    assert tpc.lm_head_dshard_readers("exact") is None
    for pf in ("1", "all", "gdn_in,down"):
        monkeypatch.setenv(weight_prefetch.ENV, pf)
        assert tpc.lm_head_dshard_readers("exact") == 3
        assert tpc.lm_head_dshard_readers("mc") is None
        assert tpc.lm_head_dshard_readers() is None
    monkeypatch.setenv(tpc.LMHEAD_DSHARD_READERS_ENV, "2")
    assert tpc.lm_head_dshard_readers("exact") == 2
    monkeypatch.setenv(weight_prefetch.ENV, "0")
    monkeypatch.delenv(tpc.LMHEAD_DSHARD_READERS_ENV)
    assert tpc.lm_head_dshard_readers("exact") is None
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", readers=3).check_l1()  # 3 readers fits the fp32 head


def test_host_plan_tp8_exact():
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact")
    assert (p.k_tiles, p.n_tiles) == (160, 970)
    assert (p.bank_tiles, p.padded_tiles) == (122, 976)  # the 976-tile padded vocab shard
    assert p.in0_block_w == 2  # = get_mcast_1d_config's K blocking for the plain head
    assert (p.readers, p.multicore) == (2, False)  # 1 reader/bank does not fit the fp32 head in L1
    assert (p.act_cores, p.act_shard_w, p.per_core_N) == (8, 640, 122)
    # Padding stays inside the last bank: 7 full banks + 116 real tiles.
    assert p.padded_tiles - p.n_tiles == 6 < p.bank_tiles


def test_host_plan_readers():
    p2 = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", readers=2)
    assert (p2.bank_tiles, p2.padded_tiles) == (122, 976)
    # Single-reader program with R readers needs bank_tiles == R * ceil(N / (banks * R)).
    assert p2.bank_tiles == 2 * -(-p2.n_tiles // (8 * 2))
    p3 = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", readers=3)
    assert (p3.bank_tiles, p3.padded_tiles) == (123, 984)
    assert p3.bank_tiles == 3 * -(-p3.n_tiles // (8 * 3))
    with pytest.raises(ValueError):
        tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", readers=4)


def test_host_plan_l1_fits():
    # Storage-core L1 (in1 + out/interm + in0 buffers + out shard + in shard) for the chosen readers, in bytes.
    budget = tpc.LMHEAD_DSHARD_L1_BUDGET
    assert budget <= 1400 * 1024
    for readers in (2, 3):
        p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", readers=readers)
        for out in ("bfloat16", "float32"):
            assert p.l1_bytes_per_core(out) < budget, (readers, out)
        p.check_l1()
    p2 = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact")
    # 2 readers: 61 tiles/worker -> in1 366 bfp8 tiles + 64 fp32 out tiles + 122-tile fp32 shard + 20-tile input.
    assert p2.l1_bytes_per_core("float32") == 366 * 1088 + 64 * 4096 + 4 * 2048 + 20 * 2048 + 122 * 4096
    # 1 reader/bank: ~1.85 MB for the fp32 drafter head -> rejected at init, before the weight copy is made.
    p1 = tpc.LMHeadDShardPlan(K, N_TP8, 8, "exact", readers=1)
    assert p1.l1_bytes_per_core("float32") > 1536 * 1024
    with pytest.raises(ValueError):
        p1.check_l1()
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").check_l1()  # mc: modelled since K6 (the r5 default layout fits)


def test_host_plan_mc_matches_qb2_rule():
    p = tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc")
    assert p.multicore and p.bank_tiles == 122 and p.readers == 4  # min(4, max(2, (122 + 7) // 8))
    assert tpc.dram_multicore_workers(8) == 2 and tpc.dram_multicore_workers(30) == 4
    with pytest.raises(ValueError):
        tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc", readers=1)


def test_host_plan_tp4_and_odd_k():
    p = tpc.LMHeadDShardPlan(K, N_TP4, 8, "exact")
    assert (p.n_tiles, p.bank_tiles, p.padded_tiles) == (1940, 244, 1952)  # 2 readers x 122
    # K with an odd tile count: the plain head blocks K by 1 tile, so must we.
    q = tpc.LMHeadDShardPlan(9 * 32, N_TP8, 3, "exact")  # 9 K tiles over 3 banks
    assert q.in0_block_w == 1 and q.act_shard_w == 3 * 32
    assert (q.bank_tiles, q.padded_tiles) == (324, 972)
    with pytest.raises(ValueError):  # 24 K tiles: 2-tile blocks do not fill 8 shards evenly
        tpc.LMHeadDShardPlan(24 * 32, N_TP8, 8, "exact")
    with pytest.raises(ValueError):
        tpc.LMHeadDShardPlan(K + 16, N_TP8, 8, "exact")


def test_host_default_fidelity():
    import ttnn

    assert tpc.lm_head_default_fidelity(ttnn.bfloat16, ttnn.bfloat8_b) == ttnn.MathFidelity.HiFi2
    assert tpc.lm_head_default_fidelity(ttnn.bfloat8_b, ttnn.bfloat8_b) == ttnn.MathFidelity.LoFi


# -------------------------------------------------------------------------------------------- device


def _setup(device, n, rows, seed=0):
    import ttnn

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(K, n, generator=g, dtype=torch.float32).mul_(0.02).to(torch.bfloat16)
    x = torch.randn(1, 1, rows, K, generator=g, dtype=torch.float32).to(torch.bfloat16)
    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return w_tt, x_tt


def _fp32_drafter_cfg(device):
    import ttnn

    # Qwen36MTP._fp32_cfg / Qwen36Model._lm_head's fp32 branch.
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )


@pytest.mark.parametrize("rows", [1, 4, 8, 32])
@pytest.mark.parametrize("readers", [2, 3])
@pytest.mark.parametrize("n", [N_TP8], ids=["tp8"])  # TP4 doubles the L1 logits shard (fp32: ~1 MB/core)
def test_device_exact_bit_identical(device, n, readers, rows):
    import ttnn

    w_tt, x_tt = _setup(device, n, rows)
    head = tpc.LMHeadDShard(w_tt, device, "exact", readers)
    assert head.applies(x_tt)

    # Base / verify call: no kwargs (bf16 out, ttnn's default compute config).
    ref = ttnn.to_torch(ttnn.linear(x_tt, w_tt))
    got_t = head.linear(x_tt)
    assert got_t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    got = ttnn.to_torch(got_t)
    assert got.shape == ref.shape == (1, 1, rows, n)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
        f"bf16 head not bit-identical: {(got != ref).sum().item()} of {got.numel()} differ, "
        f"max |d| {(got.float() - ref.float()).abs().max().item()}"
    )
    assert torch.equal(torch.argmax(got, dim=-1), torch.argmax(ref, dim=-1))

    # Drafter call: fp32 out, HiFi2 + fp32 Dest, no packer L1 acc.
    cfg = _fp32_drafter_cfg(device)
    ref32 = ttnn.to_torch(ttnn.linear(x_tt, w_tt, dtype=ttnn.float32, compute_kernel_config=cfg))
    got32 = ttnn.to_torch(head.linear(x_tt, dtype=ttnn.float32, compute_kernel_config=cfg))
    assert torch.equal(got32.view(torch.int32), ref32.view(torch.int32)), (
        f"fp32 drafter head not bit-identical: {(got32 != ref32).sum().item()} differ, "
        f"max |d| {(got32 - ref32).abs().max().item()}"
    )


@pytest.mark.parametrize("rows", [1, 32])
def test_device_mc_close(device, rows):
    import ttnn

    w_tt, x_tt = _setup(device, N_TP8, rows, seed=1)
    head = tpc.LMHeadDShard(w_tt, device, "mc")
    ref = ttnn.to_torch(ttnn.linear(x_tt, w_tt)).float()
    got = ttnn.to_torch(head.linear(x_tt)).float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
    agree = (torch.argmax(got, -1) == torch.argmax(ref, -1)).float().mean().item()
    print(f"mc rows={rows}: pcc={pcc:.6f} argmax agreement={agree:.3f} bit-equal={torch.equal(ref, got)}")
    # Both heads round to bf16 independently (16-bit Dest + Float16_b L1 accumulation, 2 vs 8 K tiles per pass); under
    # ttsim their PCC is 0.99989 at 8 rows with each ~0.016 RMS from fp32 (test_device_mc_numerics), so 0.9999 is a
    # coin flip. 0.9998 still catches a wrong layout or K order (those give PCC << 0.99).
    assert pcc > 0.9998


def _time_ms(device, fn, iters=200, warm=20):
    import ttnn

    for _ in range(warm):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / iters


@pytest.mark.parametrize("rows", [8, 32])
def test_device_perf(device, rows):
    """Eager per-call time (host dispatch included, so it is an upper bound; use tracy for device time)."""
    import ttnn

    w_tt, x_tt = _setup(device, N_TP8, rows, seed=2)
    plain = _time_ms(device, lambda: ttnn.linear(x_tt, w_tt))
    res = {"plain": plain}
    for mode, readers in (("exact", 2), ("exact", 3), ("mc", None)):
        head = tpc.LMHeadDShard(w_tt, device, mode, readers)
        res[f"{mode}/r{readers}"] = _time_ms(device, lambda: head.linear(x_tt))
        ttnn.deallocate(head.weight)
    print(f"LM head rows={rows} ms/call: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items()))


# ------------------------------------------------------------------------------------- K6 device tests
# QWEN36_LMHEAD_DSHARD=mc with the K6 knobs (OUT_PCN, IBW, FP32_DEST). One Blackhole chip, synthetic weights at the TP8
# per-device shape. Also runnable under ttsim-perf (TT_METAL_SIMULATOR), which never touches a device.

_DRAFT_ROWS = 8  # one draft step: 8 users
_VERIFY_ROWS = 32  # 8 users x (K=3 drafts + 1)


def _setup_with_ref(device, n, rows, seed):
    """_setup plus an fp32 torch reference on the exact operand values the device sees (bf16 x, BFP8-rounded weight).
    The weight is quantized on the host and its dequantized values come from that host tensor: no 169 MB device
    readback (which the simulator's completion queue does not take)."""
    import ttnn

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(K, n, generator=g, dtype=torch.float32).mul_(0.02).to(torch.bfloat16)
    x = torch.randn(1, 1, rows, K, generator=g, dtype=torch.float32).to(torch.bfloat16)
    w_host = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ref = x.float() @ ttnn.to_torch(w_host).float()
    w_tt = ttnn.to_device(w_host, device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return w_tt, x_tt, ref


def _err(got, ref):
    d = (got.float() - ref).flatten()
    return d.pow(2).mean().sqrt().item(), d.abs().max().item()


@pytest.mark.parametrize("rows", [_DRAFT_ROWS, _VERIFY_ROWS])
@pytest.mark.parametrize("out", ["bf16", "fp32"])
def test_device_mc_out_pcn_bit_identical(device, out, rows):
    """QWEN36_LMHEAD_DSHARD_OUT_PCN only re-lays the logits shard (97 cores x 10 tiles instead of 8 x 122): every logit
    must keep the default mc layout's bits, for the base call (bf16, default config) and the drafter call (fp32)."""
    import ttnn

    w_tt, x_tt = _setup(device, N_TP8, rows, seed=3)
    ref_head = tpc.LMHeadDShard(w_tt, device, "mc", out_pcn=None, ibw=None, fp32_dest=False)
    new_head = tpc.LMHeadDShard(w_tt, device, "mc", out_pcn=10, ibw=None, fp32_dest=False)
    assert new_head.plan.out_cores == 97
    kw = {} if out == "bf16" else dict(dtype=ttnn.float32, compute_kernel_config=_fp32_drafter_cfg(device))
    ref = ttnn.to_torch(ref_head.linear(x_tt, **kw))
    got = ttnn.to_torch(new_head.linear(x_tt, **kw))
    view = torch.int16 if out == "bf16" else torch.int32
    assert torch.equal(got.view(view), ref.view(view)), f"{(got != ref).sum().item()} logits differ"


@pytest.mark.parametrize("rows", [_DRAFT_ROWS, _VERIFY_ROWS])
def test_device_mc_numerics(device, rows):
    """Numerics of the head variants against an fp32 reference on the same operands (the "8 K tiles per Dest pass"
    question): plain 1D (16-bit Dest, 2 K tiles per pass + Float16_b L1 acc), mc (8 K tiles per 16-bit pass + L1 acc,
    rotated K start per column group), mc + FP32_DEST (whole K in fp32 Dest), and the drafter's fp32 heads. Prints the
    error table; asserts mc + FP32_DEST is at least as close to fp32 as the plain head and every variant keeps
    PCC >= 0.9999 with the plain head."""
    import ttnn

    w_tt, x_tt, ref = _setup_with_ref(device, N_TP8, rows, seed=4)
    cfg32 = _fp32_drafter_cfg(device)
    mc = tpc.LMHeadDShard(w_tt, device, "mc", out_pcn=10, ibw=None, fp32_dest=False)
    mcf = tpc.LMHeadDShard(w_tt, device, "mc", out_pcn=10, ibw=None, fp32_dest=True)
    outs = {
        "plain_bf16": ttnn.to_torch(ttnn.linear(x_tt, w_tt)),
        "mc_bf16": ttnn.to_torch(mc.linear(x_tt)),
        "mc_fp32dest_bf16": ttnn.to_torch(mcf.linear(x_tt)),
        "plain_fp32": ttnn.to_torch(ttnn.linear(x_tt, w_tt, dtype=ttnn.float32, compute_kernel_config=cfg32)),
        "mc_fp32": ttnn.to_torch(mc.linear(x_tt, dtype=ttnn.float32, compute_kernel_config=cfg32)),
    }
    am_ref = torch.argmax(ref, -1)
    res = {}
    for name, t in outs.items():
        rms, mx = _err(t, ref)
        base = outs["plain_bf16"] if name.endswith("bf16") else outs["plain_fp32"]
        pcc = torch.corrcoef(torch.stack([t.float().flatten(), base.float().flatten()]))[0, 1].item()
        res[name] = dict(
            rms=rms,
            max=mx,
            pcc_plain=pcc,
            same_bits_plain=(t == base).float().mean().item(),
            argmax_ref=(torch.argmax(t, -1) == am_ref).float().mean().item(),
            argmax_plain=(torch.argmax(t, -1) == torch.argmax(base, -1)).float().mean().item(),
        )
        print(f"[k6 numerics] rows={rows} {name}: " + " ".join(f"{k}={v:.6g}" for k, v in res[name].items()))
    # Two bf16 heads that round independently differ from each other by about their own error (ttsim, rows=8:
    # plain 0.0181 / mc 0.0154 / mc + FP32_DEST 0.0063 RMS vs fp32 on logits of std ~1.4, PCC with plain 0.99989), so
    # each variant is held to the fp32 reference, not to the plain head's bits.
    for name, r in res.items():
        assert r["pcc_plain"] > 0.9998, (name, r)
    assert res["mc_bf16"]["rms"] <= res["plain_bf16"]["rms"] * 1.05, res
    assert res["mc_fp32dest_bf16"]["rms"] <= res["plain_bf16"]["rms"] * 0.75, res
    assert res["mc_fp32"]["rms"] <= res["plain_fp32"]["rms"] * 1.02, res


def _dar_like_buffer(device):
    """The decode all-reduce's persistent staging buffer at TP8: [32, 5120 * 8] bf16 width-sharded over the 10x4 RES40
    grid (64 KB per core; like every L1 buffer it takes those 64 KB of address space on every core)."""
    import ttnn

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 3))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 5120 * 8 // 40], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.empty(
        [1, 1, 32, 5120 * 8], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )


def _l1_ballast(device, nbytes):
    """An L1 buffer of nbytes on one core (= nbytes of L1 address space on every core); None for 0."""
    import ttnn

    if nbytes <= 0:
        return None
    width = nbytes // (32 * 2) // 32 * 32
    g = device.compute_with_storage_grid_size()
    last = ttnn.CoreCoord(g.x - 1, g.y - 1)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(last, last)}), [32, width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.empty([1, 1, 32, width], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 8 << 20}], indirect=True)
@pytest.mark.parametrize("out_pcn", [None, 10])
def test_device_mc_l1_budget_in_trace(device, out_pcn):
    """The L1 the decode trace gives the mc head (QWEN36_LMHEAD_DSHARD=mc): with the production l1_small_size and the
    all-reduce staging buffer allocated, the drafter's fp32 call (the worst one) runs eagerly and inside a trace; an
    extra ballast of (model margin - 32 KB) still fits and (margin + 32 KB) is refused at enqueue (a circular-buffer /
    L1-buffer clash raises on the host, it does not hang), which checks LMHeadDShardPlan.mc_l1_bytes against the
    allocator to +-32 KB."""
    import ttnn

    w_tt, x_tt = _setup(device, N_TP8, _DRAFT_ROWS, seed=5)
    x_tt = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)
    dar = _dar_like_buffer(device)
    head = tpc.LMHeadDShard(w_tt, device, "mc", out_pcn=out_pcn, ibw=None, fp32_dest=False)
    cfg32 = _fp32_drafter_cfg(device)
    need = head.plan.mc_l1_need()["total"]
    margin = tpc.LMHEAD_DSHARD_MC_L1_BANK - tpc.LMHEAD_DSHARD_MC_L1_RESERVED - need
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    print(f"[k6 l1] out_pcn={out_pcn} need={need} margin={margin} l1_bank={view.total_bytes_per_bank}")

    def call():
        return head.linear(x_tt, dtype=ttnn.float32, compute_kernel_config=cfg32)

    ttnn.deallocate(call())  # eager (compiles)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = call()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
    ttnn.deallocate(out)
    fits = _l1_ballast(device, margin - 32 * 1024)
    ttnn.deallocate(call())
    if fits is not None:
        ttnn.deallocate(fits)
    over = _l1_ballast(device, margin + 32 * 1024)
    with pytest.raises(Exception):
        ttnn.deallocate(call())
    ttnn.deallocate(over)
    ttnn.deallocate(dar)


def _time_trace_us(device, fn, calls=8, reps=5):
    import ttnn

    ttnn.deallocate(fn())  # compile
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(calls)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) * 1e6 / (reps * calls)
    ttnn.release_trace(device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("rows", [_DRAFT_ROWS, _VERIFY_ROWS])
def test_device_mc_k6_perf(device, rows):
    """Traced per-call time (us, reshard + matmul + copy to DRAM, no host dispatch) of the plain head and the mc
    variants: readers W in {2, 3, 4}, in0_block_w in {2, 4, 8}, logits shard 122 (r5) or 10 tiles per core, bf16 base
    call (with / without FP32_DEST) and the drafter's fp32 call. Prints one line per variant (the device-phase sweep;
    pick the defaults from it)."""
    import ttnn

    w_tt, x_tt = _setup(device, N_TP8, rows, seed=6)
    x_tt = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)
    cfg32 = _fp32_drafter_cfg(device)
    res = {
        "plain bf16": _time_trace_us(device, lambda: ttnn.linear(x_tt, w_tt)),
        "plain fp32": _time_trace_us(
            device, lambda: ttnn.linear(x_tt, w_tt, dtype=ttnn.float32, compute_kernel_config=cfg32)
        ),
    }
    for readers in (4, 3, 2):
        for ibw in (2, 4, 8):
            for out_pcn in (None, 10):
                try:
                    h = tpc.LMHeadDShard(w_tt, device, "mc", readers, out_pcn=out_pcn, ibw=ibw, fp32_dest=False)
                    hf = tpc.LMHeadDShard(w_tt, device, "mc", readers, out_pcn=out_pcn, ibw=ibw, fp32_dest=True)
                except ValueError as e:
                    print(f"[k6 perf] rows={rows} W{readers} ibw{ibw} pcn{out_pcn}: skipped ({e})")
                    continue
                tag = f"mc W{readers} ibw{ibw} pcn{out_pcn or 122}"
                res[f"{tag} bf16"] = _time_trace_us(device, lambda: h.linear(x_tt))
                res[f"{tag} bf16+fp32dest"] = _time_trace_us(device, lambda: hf.linear(x_tt))
                res[f"{tag} fp32"] = _time_trace_us(
                    device, lambda: h.linear(x_tt, dtype=ttnn.float32, compute_kernel_config=cfg32)
                )
                ttnn.deallocate(h.weight)
                ttnn.deallocate(hf.weight)
    for k, v in res.items():
        print(f"[k6 perf] rows={rows} {k}: {v:.1f} us")
