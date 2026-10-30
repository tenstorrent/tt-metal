# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device checks of ``custom_ops/sdpa_decode_splitfree`` (the split-independent paged decode attention): accuracy
against a torch reference over random paged caches and page tables, the bitwise identity of a row across the
padded batches and the grids, and the long-context timing against the stock op pinned (3 cores per head, the
Galaxy catalog's value) and unpinned.

Gated like ``tests/unit/test_attention.py::test_decode_sdpa_core_split_long_context``: ``GEMMA4_SDPA_TEST_CACHE``
must be set (the S of the timing table, e.g. 262144), so the long device run never starts by accident:

    TT_VISIBLE_DEVICES=<chip> GEMMA4_SDPA_TEST_CACHE=262144 pytest models/demos/gemma4/tests/test_sdpa_decode_splitfree.py
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import SdpaDecodeSplitfree

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
CACHE_ENV = "GEMMA4_SDPA_TEST_CACHE"
BLOCK = 64
PCC_MIN = 0.999

gated = pytest.mark.skipif(
    not os.environ.get(CACHE_ENV),
    reason=f"long-context device checks of sdpa_decode_splitfree: set {CACHE_ENV}=<tokens> to run them",
)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


class _Case:
    """One paged cache of ``S`` tokens on the device (random data, bfloat8_b), its torch mirror and the scratch."""

    def __init__(self, mesh_device, S, H_q, H_kv, D, dtype=ttnn.bfloat8_b, seed=0):
        torch.manual_seed(seed)
        self.device, self.S, self.H_q, self.H_kv, self.D = mesh_device, S, H_q, H_kv, D
        self.num_blocks = S // BLOCK
        self.k_cache = ttnn.from_torch(
            torch.randn(self.num_blocks, H_kv, BLOCK, D, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.from_torch(
            torch.randn(self.num_blocks, H_kv, BLOCK, D, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # the reference reads what the device reads (the bfp8 rounding)
        self.k_ref = ttnn.to_torch(self.k_cache).float()
        self.v_ref = ttnn.to_torch(self.v_cache).float()
        tiles = SdpaDecodeSplitfree.scratch_tiles(
            mesh_device, 32, [(H_kv, H_q // H_kv, D, 1024), (H_kv, H_q // H_kv, D, None)], S, kv_dtype=dtype
        )
        self.scratch = SdpaDecodeSplitfree.allocate_scratch(mesh_device, tiles)

    def inputs(self, B, cur_pos, tables, q):
        """Device tensors of one call: ``q [B, H_q, D]`` (the TILE layout pads the heads to 32 rows, as the
        model's Q arrives), ``cur_pos [B]``, ``tables [B, blocks]``."""
        tt_q = ttnn.from_torch(
            q.reshape(1, B, self.H_q, self.D), device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_pos = ttnn.from_torch(
            cur_pos.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        tt_tab = ttnn.from_torch(
            tables.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        return tt_q, tt_pos, tt_tab

    def run(self, tt_q, tt_pos, tt_tab, **kw):
        return SdpaDecodeSplitfree.op(tt_q, self.k_cache, self.v_cache, tt_pos, tt_tab, self.scratch, scale=1.0, **kw)

    def golden(self, q, cur_pos, tables, **kw):
        return SdpaDecodeSplitfree.golden(
            q.reshape(1, -1, self.H_q, self.D), self.k_ref, self.v_ref, tables, cur_pos, scale=1.0, **kw
        )

    def random_tables(self, B, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.stack([torch.randperm(self.num_blocks, generator=g) for _ in range(B)])


def _to_rows(out, B, H_q):
    return ttnn.to_torch(out)[0, :B, :H_q, :].contiguous()


def _model_compute_config(mesh_device):
    """The compute config the model's stock decode SDPA takes (tt/attention/decode.py): HiFi2 with fp32 dest
    accumulation (the op default leaves the dest in bf16, which the long-context scores cannot afford)."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _stock(case, tt_q, tt_pos, tt_tab, cap, compute_kernel_config):
    """The stock paged decode SDPA on the case's caches at ``cap`` cores per head."""
    grid = case.device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=cap,
    )
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q,
        case.k_cache,
        case.v_cache,
        cur_pos_tensor=tt_pos,
        page_table_tensor=tt_tab,
        scale=1.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_kernel_config,
    )


@gated
@pytest.mark.parametrize("S", [4096, 131072, 262144])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_matches_torch(mesh_device, device_params, S, reset_seeds):
    """(a) PCC > 0.999 against torch at S in {4K, 128K, 256K} for B in {1, 8, 32}: random caches, random page
    tables per row, random live positions (row 0 at S-1), some padding rows; one kv head, 4 q heads, head_dim 512
    (the Galaxy TP8 global layer)."""
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    for B in (1, 8, 32):
        q = torch.randn(B, 4, 512, dtype=torch.bfloat16)
        cur_pos = torch.randint(S // 2, S, (B,))
        cur_pos[0] = S - 1
        if B > 1:
            cur_pos[1] = 0  # a one-token row
            cur_pos[B - 1] = -1  # a padding row
        tables = case.random_tables(B, seed=B)
        out = case.run(*case.inputs(B, cur_pos, tables, q))
        got = _to_rows(out, B, 4)
        ref = case.golden(q, cur_pos, tables)[0]
        live = [b for b in range(B) if cur_pos[b] >= 0]
        pcc = _pcc(got[live], ref[live])
        logger.info(f"S={S} B={B}: pcc {pcc:.6f} over {len(live)} live rows")
        assert pcc > PCC_MIN, f"S={S} B={B}: pcc {pcc:.5f}"
        out.deallocate(True)


@gated
@pytest.mark.parametrize(
    "H_q, H_kv, window, modulo",
    [(4, 2, None, None), (4, 2, 1024, None), (2, 1, 1024, 2048)],
    ids=["gqa", "gqa-window", "window-ring"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_gqa_and_window(mesh_device, device_params, H_q, H_kv, window, modulo, reset_seeds):
    """Two KV groups (the face-line output path), the sliding window (the window mask, one chunk per unit) and
    the bounded ring (the virtual tile wrap before the page-table lookup) at S = 4096, head_dim 256."""
    S = 4096
    case = _Case(mesh_device, S, H_q=H_q, H_kv=H_kv, D=256)
    B = 8
    q = torch.randn(B, H_q, 256, dtype=torch.bfloat16)
    cur_pos = torch.tensor([S - 1, 0, 1023, 1024, 1025, 2500, -1, 3000])
    if modulo:
        # the ring holds modulo tokens: every row's positions wrap into its first modulo/BLOCK blocks
        tables = case.random_tables(B, seed=7)
    else:
        tables = case.random_tables(B, seed=7)
    kw = dict(num_kv_heads=H_kv, block_size=BLOCK, sliding_window=window, cache_position_modulo=modulo)
    out = case.run(*case.inputs(B, cur_pos, tables, q), **kw)
    got = _to_rows(out, B, H_q)
    ref = case.golden(q, cur_pos, tables, **kw)[0]
    live = [b for b in range(B) if cur_pos[b] >= 0]
    pcc = _pcc(got[live], ref[live])
    logger.info(f"H_q={H_q} H_kv={H_kv} window={window} ring={modulo}: pcc {pcc:.6f}")
    assert pcc > PCC_MIN, f"pcc {pcc:.5f}"


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_row_bits_independent_of_batch_and_grid(mesh_device, device_params, reset_seeds, monkeypatch):
    """(b) row 0's output bits are identical between the B=1 run and the B=8 / B=32 runs that carry the same row
    (other rows random and live), and between the device grid and a forced 8x8 grid."""
    S = int(os.environ.get(CACHE_ENV, "131072"))
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    q0 = torch.randn(4, 512, dtype=torch.bfloat16)
    table0 = case.random_tables(1, seed=11)[0]
    bits = {}
    for B in (1, 8, 32):
        q = torch.randn(B, 4, 512, dtype=torch.bfloat16)
        q[0] = q0
        cur_pos = torch.randint(0, S, (B,))
        cur_pos[0] = S - 1
        tables = case.random_tables(B, seed=100 + B)
        tables[0] = table0
        out = case.run(*case.inputs(B, cur_pos, tables, q))
        bits[B] = _to_rows(out, B, 4).view(torch.int16).clone()[0]
        out.deallocate(True)
    monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")
    out = case.run(*case.inputs(1, torch.tensor([S - 1]), table0.unsqueeze(0), q0.unsqueeze(0)))
    bits["8x8"] = _to_rows(out, 1, 4).view(torch.int16).clone()[0]
    monkeypatch.delenv("GEMMA4_TEST_GRID")
    ref = case.golden(q0.unsqueeze(0), torch.tensor([S - 1]), table0.unsqueeze(0))[0, 0]
    assert _pcc(bits[1].view(torch.bfloat16), ref) > PCC_MIN
    for key in (8, 32, "8x8"):
        assert torch.equal(bits[key], bits[1]), f"row 0 differs between the B=1 run and {key}"


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_timing_vs_stock(mesh_device, device_params, reset_seeds):
    """(c) ms per op at S = GEMMA4_SDPA_TEST_CACHE for the padded batches 1 / 8 / 32 with row 0 live at S-1 and
    the other rows padding (the lone-user scenario the pin punishes): the custom op against the stock op (the
    model's compute config) at the caps 3 (the Galaxy pin), 16 (the op default) and 32, every output's pcc
    against the torch golden logged. Asserted: the custom op meets PCC_MIN, the stock op at cap 3 gives row 0
    the same bytes in every padded batch (the pin's purpose, as the earlier core-split harness checked), the
    custom op does so without any pin, and -- loosely -- the custom op at B=1 runs within 2x of the stock at
    cap 16."""
    S = int(os.environ[CACHE_ENV])
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    ckc = _model_compute_config(mesh_device)
    q0 = torch.randn(4, 512, dtype=torch.bfloat16)
    table = torch.arange(case.num_blocks, dtype=torch.int32)
    ref0 = case.golden(q0.reshape(1, 1, 4, 512), torch.tensor([S - 1]), table.unsqueeze(0))[0, 0]
    times, bits = {}, {}
    reps = 10
    for B in (1, 8, 32):
        q = torch.zeros(B, 4, 512, dtype=torch.bfloat16)
        q[0] = q0
        cur_pos = torch.full((B,), -1)
        cur_pos[0] = S - 1
        tables = table.unsqueeze(0).repeat(B, 1)
        tt_q, tt_pos, tt_tab = case.inputs(B, cur_pos, tables, q)

        def custom():
            return case.run(tt_q, tt_pos, tt_tab)

        def stock(cap):
            return _stock(case, tt_q, tt_pos, tt_tab, cap, ckc)

        runs = {
            "custom": custom,
            "stock cap 3": lambda: stock(3),
            "stock cap 16": lambda: stock(16),
            "stock cap 32": lambda: stock(32),
        }
        outs = {}
        for name, fn in runs.items():
            o = fn()
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(reps):
                r = fn()
                r.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            times[(B, name)] = (time.perf_counter() - t0) * 1e3 / reps
            outs[name] = _to_rows(o, B, 4)[0].contiguous()
            o.deallocate(True)
        pccs = {name: _pcc(outs[name], ref0) for name in runs}
        for name in runs:
            logger.info(f"  S={S} B={B:2d} {name:13s}: {times[(B, name)]:7.2f} ms/op, pcc vs golden {pccs[name]:.6f}")
        assert (
            pccs["custom"] > PCC_MIN
        ), f"S={S} B={B}: custom pcc {pccs['custom']:.5f} (stock cap 16 {pccs['stock cap 16']:.5f})"
        bits[B] = {name: outs[name].view(torch.int16).clone() for name in ("custom", "stock cap 3")}
    for B in (8, 32):
        assert torch.equal(bits[B]["stock cap 3"], bits[1]["stock cap 3"]), f"stock cap 3: row 0 differs at B={B}"
        assert torch.equal(bits[B]["custom"], bits[1]["custom"]), f"custom: row 0 differs at B={B}"
    for B in (1, 8, 32):
        logger.info(
            f"S={S} B={B}: custom {times[(B, 'custom')]:.2f} ms vs stock cap 3 {times[(B, 'stock cap 3')]:.2f} / "
            f"cap 16 {times[(B, 'stock cap 16')]:.2f} / cap 32 {times[(B, 'stock cap 32')]:.2f} ms"
        )
    assert (
        times[(1, "custom")] <= 2.0 * times[(1, "stock cap 16")]
    ), f"custom B=1 {times[(1, 'custom')]:.2f} ms vs stock cap 16 {times[(1, 'stock cap 16')]:.2f} ms"


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_partials_vs_torch(mesh_device, device_params, reset_seeds):
    """(d) Where an error lives, at S = GEMMA4_SDPA_TEST_CACHE, B=1, 4 heads, head_dim 512, row 0 at S-1 on a
    random page table: the unit partials (m, l, O) read back from the scratch against a torch flash model of the
    same units on fp32 scores (m_dev - m_ref is the device's score error at the row max: a bf16 ulp is 0.5 there,
    a tf32 ulp 1/16), the combine (the device partials merged in torch against the output), the output against
    the golden (asserted at PCC_MIN), and the stock op against the golden under the model's compute config
    (HiFi2, fp32 dest accumulation) and the op default (bf16 dest), logged as the reference the stock reaches."""
    S = int(os.environ[CACHE_ENV])
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    q = torch.randn(1, 4, 512, dtype=torch.bfloat16)
    cur_pos = torch.tensor([S - 1])
    tables = case.random_tables(1, seed=1)
    tt_q, tt_pos, tt_tab = case.inputs(1, cur_pos, tables, q)
    out = case.run(tt_q, tt_pos, tt_tab)
    got = _to_rows(out, 1, 4)[0].float()
    ref = case.golden(q, cur_pos, tables)[0, 0]
    plan = SdpaDecodeSplitfree.plan(mesh_device, tt_q, case.k_cache, tt_pos, tt_tab)
    logger.info(f"S={S}: {plan.describe()}")
    blocks = tables[0].long()
    K = case.k_ref[blocks, 0].reshape(S, 512)
    V = case.v_ref[blocks, 0].reshape(S, 512)
    s = q[0].float() @ K.T  # [4, S] fp32 scores
    scratch = ttnn.to_torch(case.scratch)
    dm, lr, opcc, parts = [], [], [], []
    for u in plan.live_units(S - 1):
        page0 = plan.slot(0, 0, u) * plan.tiles_per_slot
        tile = lambda t: scratch[(page0 + t) * 32 : (page0 + t) * 32 + 4]  # the 4 head rows of page t
        m_dev, l_dev = tile(0)[:, 0], tile(1)[:, 0]
        O_dev = torch.cat([tile(2 + c) for c in range(16)], dim=1)
        su = s[:, u * plan.unit : (u + 1) * plan.unit]
        m_ref = su.amax(1)
        p = torch.exp(su - m_ref[:, None])
        O_ref = p @ V[u * plan.unit : (u + 1) * plan.unit]
        dm.append(m_dev - m_ref)
        lr.append(l_dev / p.sum(1))
        opcc.append(_pcc(O_dev, O_ref))
        parts.append((m_dev, l_dev, O_dev))
    dm, lr = torch.stack(dm), torch.stack(lr)
    logger.info(
        f"S={S}: {len(parts)} units; m_dev - m_ref: max |dm| {dm.abs().max():.4f}, mean {dm.mean():+.4f}, "
        f"rms {dm.pow(2).mean().sqrt():.4f}; l_dev / l_ref: min {lr.min():.4f} max {lr.max():.4f}; "
        f"pcc(O_dev, O_ref): min {min(opcc):.6f} mean {sum(opcc) / len(opcc):.6f}"
    )
    logger.info(f"S={S}: unit 0 dm per head {dm[0].tolist()}, l ratio {lr[0].tolist()}")
    m, l, O = parts[0]
    for mu, lu, Ou in parts[1:]:
        mn = torch.maximum(m, mu)
        a, au = torch.exp(m - mn), torch.exp(mu - mn)
        l, O, m = l * a + lu * au, O * a[:, None] + Ou * au[:, None], mn
    comb = O / l[:, None]
    logger.info(
        f"S={S}: pcc(out, golden) {_pcc(got, ref):.6f}; pcc(device partials combined in torch, golden) "
        f"{_pcc(comb, ref):.6f}; pcc(out, that) {_pcc(got, comb):.6f}; per-head pcc(out, golden) "
        f"{[round(_pcc(got[h], ref[h]), 5) for h in range(4)]}"
    )
    for name, ckc in (("model cfg (HiFi2, fp32 dest)", _model_compute_config(mesh_device)), ("op default", None)):
        for cap in (3, 16):
            o = _stock(case, tt_q, tt_pos, tt_tab, cap, ckc)
            logger.info(f"S={S}: stock cap {cap:2d} {name}: pcc vs golden {_pcc(_to_rows(o, 1, 4)[0], ref):.6f}")
            o.deallocate(True)
    assert _pcc(got, ref) > PCC_MIN, f"S={S}: pcc {_pcc(got, ref):.5f}"
