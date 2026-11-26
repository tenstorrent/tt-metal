# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shapes, torch references and packed-tile helpers shared by the gdn_decode_step tests.

The shapes are one Qwen3.8 GDN layer at TP=8 on one device: 6 value heads, 2 key heads, 128-wide heads.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, nrmse

NV, NK, DK, DV = 6, 2, 128, 128
KD, VD = NK * DK, NV * DV
C = 2 * KD + VD  # convolution channels [q | k | v]
AZ = C + VD  # qkvz_dim: first column of the a | b tile
W = AZ + 32  # projection row [q | k | v | z | a b], a | b padded to one tile
KV = (DK // 32) * (DV // 32)  # fp32 state pages per value head
HP = 4  # bf16 history pages per value head, one per convolution slot
SCALE = DK**-0.5
BMAX = 32
# (fast_mode, fp32_state); the model runs (True, True)
MODES = [(False, False), (True, False), (True, True)]
MODE_IDS = ["default", "fast", "fast_fp32_state"]
MODEL_MODE = (True, True)  # (fast_mode, fp32_state) as the model runs it
# One device per module; the trace region is for the tests that capture a trace.
DEVICE_PARAMS = {"trace_region_size": 16_000_000}
DETERMINISM_RUNS = 10
PERF_MARGIN = 0.05  # the device-time band of the perf tests, on both sides of the measured value
K_HEADS = 4  # KvBlocks: heads 0..3 in the K block, heads 4..5 and the whole history in the V block
FP32_PAGE, BF16_PAGE, BF8_PAGE = 4096, 2048, 1088
# NRMSE gates of assert_close_to_reference: the worst (row, value head) NRMSE against the torch reference measured on a
# P150 over the reference tests, plus headroom. The output's NRMSE depends on its dtype more than on the mode (measured
# worst 9.9e-3 for float32 outputs, 2.0e-2 for bfloat16 outputs); the state's on the mode (5.5e-3, 5.5e-3, 4.4e-3).
OUT_NRMSE = {torch.float32: 0.015, torch.bfloat16: 0.025}
STATE_NRMSE = {(False, False): 0.007, (True, False): 0.007, (True, True): 0.006}


def recurrence_reference(q, k, v, beta, g, h, w, l2_eps=1e-6, norm_eps=1e-6):
    """q, k: [NK, DK]; v: [NV, DV]; beta, g: [NV]; h: [NV, DK, DV]; w: [DV]. Returns (out [NV * DV], new h)."""
    rf = NV // NK
    q = q.repeat_interleave(rf, dim=0)
    k = k.repeat_interleave(rf, dim=0)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + l2_eps) * SCALE
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + l2_eps)
    h = h * torch.exp(g)[:, None, None]
    v_read = torch.einsum("hk,hkv->hv", kn, h)
    delta = beta[:, None] * (v - v_read)
    h = h + torch.einsum("hk,hv->hkv", kn, delta)
    o = torch.einsum("hk,hkv->hv", qn, h)
    on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + norm_eps) * w[None, :]
    return on.reshape(-1), h


def step_reference(hist, new_row, taps, dtb, nea, h, w):
    """One decode step from the projection row. hist: the 3 previous [C] rows, oldest first; new_row: [W]; taps: 4
    [C] rows, tap 0 oldest. Returns (gated output [NV * DV], new h, the 4 history rows after the shift)."""
    conv = hist[0] * taps[0] + hist[1] * taps[1] + hist[2] * taps[2] + new_row[:C] * taps[3]
    conv = torch.nn.functional.silu(conv)
    q, k, v = conv[:KD].reshape(NK, DK), conv[KD : 2 * KD].reshape(NK, DK), conv[2 * KD : C].reshape(NV, DV)
    z = new_row[C:AZ]
    a = new_row[AZ : AZ + NV]
    b = new_row[AZ + NV : AZ + 2 * NV]
    beta = torch.sigmoid(b)
    g = nea * torch.nn.functional.softplus(a + dtb, beta=1.0, threshold=20.0)
    out, h_new = recurrence_reference(q, k, v, beta, g, h, w)
    return out * torch.nn.functional.silu(z), h_new, [hist[0], hist[1], hist[2], new_row[:C]]


def pack_rows(rows, b=0, both=False):
    """4 [C] rows -> [NV, 4, 32, 32] packed head tiles of batch row b: channel chunk c of value head h's
    [q | k | v] in tile row 2c + (b & 1), or in both rows 2c and 2c + 1 (the taps and the paged history)."""
    rf = NV // NK
    out = torch.zeros(NV, 4, 32, 32, dtype=torch.bfloat16)
    for h in range(NV):
        hk = h // rf
        for j, r in enumerate(rows):
            r = r.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    r[hk * DK : (hk + 1) * DK],
                    r[KD + hk * DK : KD + (hk + 1) * DK],
                    r[2 * KD + h * DV : 2 * KD + (h + 1) * DV],
                ]
            ).reshape(-1, 32)
            n = chunks.shape[0]
            for par in (0, 1) if both else (b & 1,):
                out[h, j, par : 2 * n + par : 2, :] = chunks
    return out


def to_pooled(t, b):
    """Packed history of batch row b (chunk c in row 2c + (b & 1)) -> the paged layout (rows 2c and 2c + 1)."""
    p = b % 2
    out = t.clone()
    out[..., 1 - p :: 2, :] = t[..., p::2, :]
    return out


def from_pooled(t, b):
    """The paged layout -> the default layout of batch row b (row 2c + (b & 1), the other rows 0)."""
    p = b % 2
    out = torch.zeros_like(t)
    out[..., p::2, :] = t[..., p::2, :]
    return out


def tiles(x):
    """[..., R, Cc] -> [n, 32, 32]: the 32x32 tiles in device page order."""
    *_, r, cc = x.shape
    return x.reshape(-1, r // 32, 32, cc // 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def bits(x):
    """Bit pattern of a float tensor, so that equal NaNs compare equal."""
    return x.view({2: torch.int16, 4: torch.int32}[x.element_size()])


def assert_close_to_reference(name, fast_mode, fp32_state, out_ref, out, state_ref, state):
    """assert_accurate of GDN step outputs and states against the torch reference. out_ref, out: [rows, NV * DV];
    state_ref, state: [rows, NV, ...] (any per-head layout, the same in both). PCC gates each row. NRMSE gates each
    (row, value head), because PCC does not see a scale error and a whole-tensor NRMSE averages one bad head away."""
    out_nrmse = OUT_NRMSE[out.dtype]
    rows = out.shape[0]
    out_ref, out = (t.float().reshape(rows, NV, DV) for t in (out_ref, out))
    state_ref, state = state_ref.float(), state.float()
    logger.info(
        f"{name}: worst output NRMSE {nrmse(out_ref, out, (0, 1)).max():.3e}, "
        f"worst state NRMSE {nrmse(state_ref, state, (0, 1)).max():.3e} (per row and value head)"
    )
    assert_accurate(
        out_ref,
        out,
        name=f"{name} output",
        pcc_threshold=0.999,
        pcc_keep_dims=(0,),
        nrmse_threshold=out_nrmse,
        nrmse_keep_dims=(0, 1),
    )
    # the default kernels truncate the state to TF32 in the add, so their state tracks fp32 torch more closely
    assert_accurate(
        state_ref,
        state,
        name=f"{name} state",
        pcc_threshold=0.999 if fast_mode else 0.9999,
        pcc_keep_dims=(0,),
        nrmse_threshold=STATE_NRMSE[(fast_mode, fp32_state)],
        nrmse_keep_dims=(0, 1),
    )


def tile_dev(device, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def inputs(n_req, seed):
    """Per request: state [NV, DK, DV] and paged-layout history [NV, 4, 32, 32]; shared weights and constants."""
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen)
    rows = [[(0.5 * rn(C)).bfloat16() for _ in range(4)] for _ in range(n_req)]
    return {
        "h": (0.05 * rn(n_req, NV, DK, DV)).float(),
        "rows": rows,  # the history as [C] rows: slot 0 (oldest) .. slot 3
        "hist2": torch.stack([pack_rows(r, both=True) for r in rows]),
        "w": (1.0 + 0.1 * rn(DV)).bfloat16(),
        "tap_rows": [(0.3 * rn(C)).bfloat16() for _ in range(4)],
        "dtb": (0.1 * rn(NV)).bfloat16(),
        "nea": (-torch.exp(0.2 * rn(NV))).bfloat16(),
    }


def token_rows(b, gen):
    rows = (0.5 * torch.randn(b, W, generator=gen)).bfloat16()
    rows[:, AZ + 2 * NV :] = 0
    return rows


def consts_dev(device, inp):
    return {
        "dtb": tile_dev(device, inp["dtb"], ttnn.bfloat16),
        "nea": tile_dev(device, inp["nea"], ttnn.bfloat16),
        "w": tile_dev(device, inp["w"], ttnn.bfloat16),
        "taps": tile_dev(device, pack_rows(inp["tap_rows"], both=True), ttnn.bfloat16),
    }


def run(x, consts, state, hist, fast_mode, fp32_state, output_dtype=ttnn.bfloat16, key_dim=DK, value_dim=DV, **paged):
    """gdn_decode_step with the test shapes; `paged` passes state_page_idx, state_pool2 and conv_hist_pool2."""
    return ttnn.experimental.gdn.gdn_decode_step(
        x,
        consts["dtb"],
        consts["nea"],
        state,
        consts["w"],
        NV,
        NK,
        key_dim,
        value_dim,
        scale=SCALE,
        output_dtype=output_dtype,
        conv_hist=hist,
        conv_taps=consts["taps"],
        qkvz_dim=AZ,
        fast_mode=fast_mode,
        fp32_state=fp32_state,
        **paged,
    )


def default_step(device, x, consts, h_rows, hist2_rows, fast_mode, fp32_state):
    """The op without state_page_idx on [BMAX, ...] state and history, batch row b holding h_rows[b] and the
    paged-layout history hist2_rows[b] repacked for row b's parity. Returns (out, new states, new paged-layout
    histories)."""
    b_count = len(h_rows)
    h = torch.zeros(BMAX, NV, DK, DV)
    hist = torch.zeros(BMAX, NV, 4, 32, 32, dtype=torch.bfloat16)
    for b in range(b_count):
        h[b] = h_rows[b]
        hist[b] = from_pooled(hist2_rows[b], b)
    state_dev, hist_dev = tile_dev(device, h, ttnn.float32), tile_dev(device, hist, ttnn.bfloat16)
    out = ttnn.to_torch(run(x, consts, state_dev, hist_dev, fast_mode, fp32_state))
    h_t, hist_t = ttnn.to_torch(state_dev), ttnn.to_torch(hist_dev)
    return out, [h_t[b] for b in range(b_count)], [to_pooled(hist_t[b], b) for b in range(b_count)]


# ---- the layout of the 1024-token cache blocks of a TP=8 Qwen3.8 layer slot


def fp32_tile_raw(t):
    """[n, 32, 32] fp32 tiles -> [n, 1024] int32 in device byte order (four 16x16 faces)."""
    return t.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1024).view(torch.int32)


def raw_fp32_tile(raw):
    return raw.view(torch.float32).reshape(-1, 2, 2, 16, 16).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def bf16_tile_raw(t):
    return t.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1024).view(torch.int16)


class KvBlocks:
    """One layer slot of a TP=8 Qwen3.8 model: bfloat8_b K and V caches [N, 1, 1024, 256], one 1024-token block per
    request. A block holds 34,816 bytes per DRAM bank in each of K and V (8.5 fp32 pages), 64 whole fp32 pages per
    block in K: one request's GDN state (6 heads of 16 fp32 pages, 384 KiB) plus its history (24 bf16 pages, 48 KiB)
    does not fit K alone. Heads 0..3 go to the first whole fp32 page rows of the K block, heads 4..5 to the first ones
    of the V block, and the history to the last three bf16 page rows of the V block.

    Host content is kept as the pages of fp32 views of K and V (whole buffers for even N)."""

    def __init__(self, device, num_blocks, gen):
        assert num_blocks % 2 == 0
        self.device, self.n = device, num_blocks
        self.nb = device.dram_grid_size().x * device.dram_grid_size().y
        tiles_per_block = 1024 * 256 // 1024
        self.s = tiles_per_block // self.nb * BF8_PAGE  # bytes per bank per block
        self.hist_tiles = tiles_per_block * BF8_PAGE // BF16_PAGE
        self.hist_first = self.hist_tiles - -(-NV * HP // self.nb) * self.nb
        self.rec_pages = self.nb * (num_blocks * self.s // FP32_PAGE)
        bf8 = lambda: ttnn.from_torch(
            torch.zeros(num_blocks, 1, 1024, 256), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.k, self.v = bf8(), bf8()
        view_shape = [self.rec_pages // 8, 1, DK // 2, DV]
        check = self.s % FP32_PAGE == 0
        self.rec_k = ttnn.experimental.alias_view(self.k, view_shape, ttnn.float32, check_slices=check)
        self.rec_v = ttnn.experimental.alias_view(self.v, view_shape, ttnn.float32, check_slices=check)
        self.hist = ttnn.experimental.alias_view(self.v, [num_blocks, self.hist_tiles, 32, 32], ttnn.bfloat16)
        # random fp32 pages stand for the attention contents of K and V
        self.k_pages = torch.randn(self.rec_pages, 32, 32, generator=gen)
        self.v_pages = torch.randn(self.rec_pages, 32, 32, generator=gen)

    def first_page(self, block):
        """First whole fp32 page of `block` in either view: its first whole page row times the bank count."""
        return self.nb * -(-block * self.s // FP32_PAGE)

    def state_page(self, block, h):
        """(pool, first fp32 page) of head h of the request in `block`."""
        p = self.first_page(block)
        return (0, p + h * KV) if h < K_HEADS else (1, p + (h - K_HEADS) * KV)

    def hist_page(self, block, h):
        return block * self.hist_tiles + self.hist_first + h * HP

    def index_host(self, blocks):
        rows = [[self.state_page(b, h)[1] + (self.rec_pages if h >= K_HEADS else 0) for b in blocks] for h in range(NV)]
        rows += [[self.hist_page(b, h) for b in blocks] for h in range(NV)]
        return torch.tensor(rows, dtype=torch.int32)

    def _v_slot(self, q2):
        """bf16 page q2 of V -> (fp32 page, int16 offset) holding its bytes."""
        row = q2 // self.nb
        return q2 % self.nb + self.nb * (row // 2), (row % 2) * 1024

    def place(self, entries):
        """Host: for each (block, state [NV, DK, DV], paged history [NV, 4, 32, 32]), the request in the block holds
        that state and history."""
        for block, h_state, _ in entries:
            for h in range(NV):
                pool, page = self.state_page(block, h)
                (self.k_pages, self.v_pages)[pool][page : page + KV] = tiles(h_state[h])
        raw16 = fp32_tile_raw(self.v_pages).view(torch.int16)
        for block, _, hist2 in entries:
            for h in range(NV):
                for s, t in enumerate(bf16_tile_raw(hist2[h])):
                    q, off = self._v_slot(self.hist_page(block, h) + s)
                    raw16[q, off : off + 1024] = t
        self.v_pages = raw_fp32_tile(raw16.view(torch.int32))

    def write(self):
        for view, pages in ((self.rec_k, self.k_pages), (self.rec_v, self.v_pages)):
            host = pages.reshape(self.rec_pages // 8, 2, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 1, DK // 2, DV)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), view)

    def read(self):
        """Device -> (K fp32 pages, V fp32 pages, history bf16 pages [N * hist_tiles, 32, 32])."""
        k, v = (tiles(ttnn.to_torch(view)) for view in (self.rec_k, self.rec_v))
        return k, v, ttnn.to_torch(self.hist).reshape(-1, 32, 32)

    def owned(self, blocks):
        """Masks of the fp32 K and V pages and the bf16 history pages the requests in `blocks` own."""
        k_mask = torch.zeros(self.rec_pages, dtype=torch.bool)
        v_mask = torch.zeros(self.rec_pages, dtype=torch.bool)
        h_mask = torch.zeros(self.n * self.hist_tiles, dtype=torch.bool)
        for blk in blocks:
            for h in range(NV):
                pool, page = self.state_page(blk, h)
                (k_mask, v_mask)[pool][page : page + KV] = True
                hp = self.hist_page(blk, h)
                h_mask[hp : hp + HP] = True
                for q2 in range(hp, hp + HP):
                    v_mask[self._v_slot(q2)[0]] = True
        return k_mask, v_mask, h_mask

    def run(self, x, consts, idx):
        return run(x, consts, self.rec_k, self.hist, *MODEL_MODE, state_page_idx=idx, state_pool2=self.rec_v)


def kv_blocks_or_skip(device, num_blocks, gen):
    banks = device.dram_grid_size().x * device.dram_grid_size().y
    if (1024 * 256 // 1024) % banks:
        pytest.skip(f"a 1024-token block does not spread evenly over {banks} DRAM banks")
    return KvBlocks(device, num_blocks, gen)
