# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of neighbor_pad_async's logical_w masking (LTX_VAE_FOLD_W_MASK=1).

Emulates the fused 2D [H, W] pad at stick granularity, mirroring the index math of the op's kernels
(local_copy_writer, minimal_default_reader/writer, phase2_w_reader, W fabric writer) and the host's
per-device num_valid_sticks. The per-device output must equal padding the input after zeroing the
W columns >= logical_w, which is what the LTX conv's mask multiply + neighbor_pad produced.
Run: python -m pytest --noconftest <this file>
"""

import pytest
import torch


def _w_valid_sticks(logical_w, w_index, w_in):
    # Mirrors the program factory.
    if logical_w == 0:
        return w_in
    off = w_index * w_in
    return min(logical_w - off, w_in) if logical_w > off else 0


def _emulate(x, rh, rw, ph, pw, mode, logical_h=0, logical_w=0, t_front_pad=0):
    """x: [O, Hg, Wg, C] (O = B*T outer rows). Returns {(r, c): [O + t_front_pad, H_out, W_out, C]}."""
    o, hg, wg, ch = x.shape
    h_in, w_in = hg // rh, wg // rw
    h_out, w_out = h_in + 2 * ph, w_in + 2 * pw
    zeros = mode == "zeros"
    tf_off = t_front_pad * h_out * w_out
    zero = torch.zeros(ch, dtype=x.dtype)
    nan = torch.full((ch,), float("nan"), dtype=x.dtype)
    inp = {
        (r, c): x[:, r * h_in : (r + 1) * h_in, c * w_in : (c + 1) * w_in].reshape(-1, ch)
        for r in range(rh)
        for c in range(rw)
    }
    # NaN-filled: a stick the kernels never write shows up as a mismatch.
    out = {k: nan.repeat((o + t_front_pad) * h_out * w_out, 1) for k in inp}

    for (r, c), src in inp.items():
        dst = out[(r, c)]
        valid = _w_valid_sticks(logical_w, c, w_in)
        # local_copy_writer: Phase A zero-fill of the T-front frames, Phase B interior copy.
        dst[:tf_off] = 0
        for outer in range(o):
            for t in range(h_in):
                masked = logical_h > 0 and r * h_in + t >= logical_h
                dst_id = (t + ph) * w_out + pw + outer * (w_out * h_out) + tf_off
                for it in range(w_in):
                    dst[dst_id + it] = zero if (masked or it >= valid) else src[(outer * h_in + t) * w_in + it]

    # H fabric: reader pushes sticks, writer self-pads or ships them to the H neighbor's output.
    for (r, c), src in inp.items():
        valid = _w_valid_sticks(logical_w, c, w_in)
        for d in (0, 1):
            first = r == (rh - 1 if d else 0)
            last = r == (0 if d else rh - 1)
            nbr = (r + (-1 if d else 1), c)
            for outer in range(o):
                cb = []
                base_in = outer * w_in * h_in
                if first:
                    if not zeros:
                        sid = base_in + (w_in * (h_in - 1) if d else 0)
                        cb += [src[sid + it] if it < valid else zero for it in range(w_in)]
                    else:
                        cb.append(zero)
                if not last:
                    for pad_id in range(ph, 0, -1):
                        sid = base_in + ((ph - pad_id) if d else (h_in - pad_id)) * w_in
                        cb += [src[sid + it] if it < valid else zero for it in range(w_in)]
                cb.reverse()
                base_out = outer * w_out * h_out + tf_off
                if first:
                    did = base_out + ((h_out - ph) * w_out + pw if d else pw)
                    if not zeros:
                        for it in range(w_in):
                            s = cb.pop()
                            for pad_id in range(ph):
                                out[(r, c)][did + it + pad_id * w_out] = s
                    else:
                        s = cb.pop()
                        for it in range(w_in):
                            for pad_id in range(ph):
                                out[(r, c)][did + it + pad_id * w_out] = s
                if not last:
                    for pad_id in range(ph):
                        did = base_out + ((h_out - (ph - pad_id)) * w_out if d else pad_id * w_out) + pw
                        for it in range(w_in):
                            out[nbr][did + it] = cb.pop()
                assert not cb

    # W fabric: phase2_w_reader Phase 1 (interior rows from INPUT) + Phase 2 (H-pad rows from OUTPUT),
    # W writer self-pads or ships each stick to the W neighbor's output.
    snap = {k: v.clone() for k, v in out.items()}  # Phase 2 reads OUTPUT after the H barrier
    for (r, c), src in inp.items():
        valid = _w_valid_sticks(logical_w, c, w_in)
        for d in (0, 1):
            first = c == (rw - 1 if d else 0)
            last = c == (0 if d else rw - 1)
            nbr = (r, c + (-1 if d else 1))
            rows = []  # (dst_row, pushed sticks)
            for t_abs in range(o + t_front_pad):
                t_front = t_abs < t_front_pad
                t_input = 0 if t_front else t_abs - t_front_pad
                for h in range(h_in):
                    h_masked = logical_h > 0 and r * h_in + h >= logical_h
                    base = (t_input * h_in + h) * w_in
                    cb = []
                    if first:
                        edge = w_in - 1 if d else 0
                        cb.append(zero if (t_front or h_masked or zeros or edge >= valid) else src[base + edge])
                    if not last:
                        for pad_id in range(pw, 0, -1):
                            col = (pw - pad_id) if d else (w_in - pad_id)
                            cb.append(zero if (t_front or h_masked or col >= valid) else src[base + col])
                    rows.append((t_abs * h_out + ph + h, cb))
            for t_abs in range(o + t_front_pad):
                t_front = t_abs < t_front_pad
                for orow in [t_abs * h_out + k for k in range(ph)] + [t_abs * h_out + ph + h_in + k for k in range(ph)]:
                    base = orow * w_out
                    cb = []
                    if first:
                        col = pw + w_in - 1 if d else pw
                        cb.append(zero if (t_front or zeros) else snap[(r, c)][base + col])
                    if not last:
                        for pad_id in range(pw, 0, -1):
                            col = pw + ((pw - pad_id) if d else (w_in - pad_id))
                            cb.append(zero if t_front else snap[(r, c)][base + col])
                    rows.append((orow, cb))
            for orow, cb in rows:
                cb.reverse()
                off = orow * w_out
                if first:
                    s = cb.pop()
                    did = off + (w_out - pw if d else 0)
                    for pad_id in range(pw):
                        out[(r, c)][did + pad_id] = s
                if not last:
                    for pad_id in range(pw):
                        out[nbr][off + ((w_out - (pw - pad_id)) if d else pad_id)] = cb.pop()
                assert not cb
    return {k: v.reshape(o + t_front_pad, h_out, w_out, ch) for k, v in out.items()}


def _pad_chunks(chunks, dim, p, mode):
    res = []
    for i, ch in enumerate(chunks):
        n = ch.shape[dim]
        if i > 0:
            left = chunks[i - 1].narrow(dim, chunks[i - 1].shape[dim] - p, p)
        else:
            left = (
                ch.narrow(dim, 0, 1).repeat_interleave(p, dim)
                if mode == "replicate"
                else torch.zeros_like(ch.narrow(dim, 0, p))
            )
        if i < len(chunks) - 1:
            right = chunks[i + 1].narrow(dim, 0, p)
        else:
            right = (
                ch.narrow(dim, n - 1, 1).repeat_interleave(p, dim)
                if mode == "replicate"
                else torch.zeros_like(ch.narrow(dim, 0, p))
            )
        res.append(torch.cat([left, ch, right], dim))
    return res


def _golden(x, rh, rw, ph, pw, mode, logical_h, logical_w, t_front_pad):
    """Zero rows >= logical_h and columns >= logical_w, then 2D pad (H first, then W), then T-front zeros."""
    m = x.clone()
    if logical_h:
        m[:, logical_h:] = 0
    if logical_w:
        m[:, :, logical_w:] = 0
    gold = {}
    for r, hc in enumerate(_pad_chunks(list(torch.chunk(m, rh, 1)), 1, ph, mode)):
        for c, wc in enumerate(_pad_chunks(list(torch.chunk(hc, rw, 2)), 2, pw, mode)):
            gold[(r, c)] = torch.cat([torch.zeros_like(wc[:t_front_pad]), wc], 0)
    return gold


CASES = [
    # (rh, rw, O, Hg, Wg, ph, pw, mode, logical_h, logical_w, t_front_pad)
    (2, 4, 3, 18, 32, 1, 1, "zeros", 0, 0, 0),  # no masking: emulator == plain pad
    (2, 4, 3, 18, 32, 1, 1, "zeros", 0, 30, 0),  # LTX 544x960 latent on 2x4: 2 pad cols on the last W chip
    (2, 4, 3, 18, 32, 1, 1, "zeros", 17, 30, 0),  # + H padding (logical_h)
    (2, 4, 3, 18, 32, 1, 1, "zeros", 0, 20, 0),  # pad spans 1.5 W chips: masked sticks cross a W halo
    (2, 4, 3, 18, 32, 1, 1, "zeros", 0, 16, 0),  # pad starts exactly on a W chip boundary
    (2, 4, 3, 18, 32, 1, 1, "zeros", 0, 1, 0),  # almost everything masked
    (2, 4, 2, 18, 32, 1, 1, "zeros", 17, 20, 2),  # + T-front zero frames
    (4, 8, 2, 36, 64, 1, 1, "zeros", 34, 60, 0),  # LTX 1080p latent on 4x8
    (2, 4, 2, 12, 32, 2, 2, "zeros", 0, 27, 0),  # wider halo (pad 2)
    (2, 4, 2, 18, 32, 1, 1, "replicate", 0, 30, 0),  # replicate edges read the masked column
    (2, 4, 2, 18, 32, 1, 1, "replicate", 0, 20, 0),
]


@pytest.mark.parametrize("rh, rw, o, hg, wg, ph, pw, mode, logical_h, logical_w, t_front_pad", CASES)
def test_neighbor_pad_logical_w_matches_masked_pad(rh, rw, o, hg, wg, ph, pw, mode, logical_h, logical_w, t_front_pad):
    torch.manual_seed(0)
    # Pad columns hold non-zero garbage (a previous conv's output), so a missed mask shows up.
    x = torch.randn(o, hg, wg, 4) + 3.0
    got = _emulate(x, rh, rw, ph, pw, mode, logical_h, logical_w, t_front_pad)
    gold = _golden(x, rh, rw, ph, pw, mode, logical_h, logical_w, t_front_pad)
    for k in gold:
        assert torch.equal(got[k], gold[k]), f"device {k}"


def test_unmasked_pad_columns_differ():
    """Without logical_w the pad columns leak into the halo/interior; the golden must then differ."""
    torch.manual_seed(0)
    x = torch.randn(2, 18, 32, 4) + 3.0
    got = _emulate(x, 2, 4, 1, 1, "zeros", 0, 0, 0)
    gold = _golden(x, 2, 4, 1, 1, "zeros", 0, 30, 0)
    assert not torch.equal(got[(0, 3)], gold[(0, 3)])
    assert torch.equal(got[(0, 0)], gold[(0, 0)])


@pytest.mark.parametrize("logical_w, w_in, rw", [(30, 8, 4), (60, 8, 8), (20, 8, 4), (960, 256, 4)])
def test_w_valid_sticks(logical_w, w_in, rw):
    valid = [_w_valid_sticks(logical_w, c, w_in) for c in range(rw)]
    assert sum(valid) == logical_w
    assert all(0 <= v <= w_in for v in valid)
    assert valid == sorted(valid, reverse=True)
