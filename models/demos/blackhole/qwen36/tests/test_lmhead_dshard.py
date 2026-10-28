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
    tpc.LMHeadDShardPlan(K, N_TP8, 8, "mc").check_l1()  # mc: not modelled, never rejected


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
    assert pcc > 0.9999


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
