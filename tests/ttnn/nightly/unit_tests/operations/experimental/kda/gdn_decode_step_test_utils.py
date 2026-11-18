# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shapes, torch references and packed-tile helpers shared by the gdn_decode_step tests.

The shapes are one Qwen3.8 GDN layer at TP=8 on one device: 6 value heads, 2 key heads, 128-wide heads.
"""

import torch

import ttnn

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


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


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


def run(x, consts, state, hist, fast_mode, fp32_state, output_dtype=ttnn.bfloat16, **paged):
    """gdn_decode_step with the test shapes; `paged` passes state_page_idx, state_pool2 and conv_hist_pool2."""
    return ttnn.experimental.kda.gdn_decode_step(
        x,
        consts["dtb"],
        consts["nea"],
        state,
        consts["w"],
        NV,
        NK,
        DK,
        DV,
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
