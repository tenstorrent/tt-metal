# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host side of image input (no device): the ragged n-gram hash with an image-span token mask vs the checkpoint's ``NgramHashState``, ``VisionSeq`` windows, and the
prompt expansion of ``vision_prompt.prepare_vl_inputs`` (span layout, token counts)."""

import io

import numpy as np
import torch
from PIL import Image

from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash
from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT
from models.demos.blackhole.deepseek_v41_flash.tt.vision import VisionArgs
from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionSeq


def _hash_state(batch, seq):
    from transformers import AutoTokenizer

    mod = R.load_model_module()
    args = R.model_args(batch, seq)
    layout = mod.EngramLayout.from_args(args)
    return mod.NgramHashState(args, layout, AutoTokenizer.from_pretrained(CKPT))


def test_ragged_hash_token_mask_matches_checkpoint():
    B, L = 3, 120
    ref, mine = _hash_state(B, 256), _hash_state(B, 256)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1000, 100000, (B, L), generator=g)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 30:70] = False  # an image span in the middle
    mask[1, 0:10] = False  # an image span at the start
    mask[2, 100:] = False  # a span reaching the end
    want = ref(ids, 0, mask)
    got = RaggedNgramHash(mine)(ids, torch.zeros(B, dtype=torch.long), token_mask=mask)
    assert torch.equal(want, got)
    # without a mask nothing changes
    assert torch.equal(ref(ids, 0, None), RaggedNgramHash(_hash_state(B, 256))(ids, torch.zeros(B, dtype=torch.long)))
    # windows: the cache carries the DEAD entries across chunks (second window starts inside the span)
    st = RaggedNgramHash(_hash_state(B, 256))
    a = st(ids[:, :50], torch.zeros(B, dtype=torch.long), token_mask=mask[:, :50])
    b = st(ids[:, 50:], torch.full((B,), 50), token_mask=mask[:, 50:])
    assert torch.equal(want, torch.cat([a, b], dim=1))


def test_vision_seq_windows():
    D = 8
    types = torch.full((40,), -1)
    types[5:15] = 1
    types[25:30] = 2
    rows = [
        (5, torch.arange(10 * D, dtype=torch.float32).reshape(10, D).to(torch.bfloat16)),
        (25, torch.ones(5, D, dtype=torch.bfloat16)),
    ]
    vs = VisionSeq(types, rows)
    m, e = vs.window(0, 40)
    assert (
        m.sum() == 15
        and torch.equal(e[5:15], rows[0][1])
        and torch.equal(e[25:30], rows[1][1])
        and e[m.logical_not()].abs().sum() == 0
    )
    m, e = vs.window(10, 20)  # straddles the first span's end and covers the second
    assert m[:5].all() and not m[5:15].any() and m[15:20].all()
    assert torch.equal(e[:5], rows[0][1][5:])
    m, e = vs.window(36, 8)  # beyond the prompt
    assert not m.any() and e.abs().sum() == 0
    try:
        VisionSeq(types, [(0, torch.zeros(5, D, dtype=torch.bfloat16))])
        raise RuntimeError("a span outside the image tokens must be rejected")
    except AssertionError:
        pass


def _png(w, h, seed=0):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(rng.randint(0, 255, (h, w, 3)).astype(np.uint8))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def test_prompt_expansion():
    from transformers import AutoTokenizer

    va = VisionArgs()
    tok = AutoTokenizer.from_pretrained(CKPT)
    ph = "<｜deepseek_image｜>"
    prompt = f"hello {ph} and {ph} bye"
    tokens, types, imgs = VP.prepare_vl_inputs(prompt, [{"data": _png(640, 480)}, {"data": _png(300, 900, 1)}], tok, va)
    assert len(tokens) == len(types) and len(imgs) == 2
    for im in imgs:
        n = im.types.numel()
        assert types[im.start : im.start + n] == im.types.tolist()
        assert all(t == va.image_token_id for t in tokens[im.start : im.start + n])
        assert im.types[0] == VP.IMAGE_START and im.types[-1] == VP.IMAGE_END
        assert n <= va.max_tokens
    # a wrong image count is an error
    try:
        VP.prepare_vl_inputs(prompt, [{"data": _png(64, 64)}], tok, va)
        raise RuntimeError("a wrong image count must be rejected")
    except ValueError:
        pass
    # the span length is predictable from the pixel size alone (scheduler / placeholder expansion)
    assert VP.image_span_len(640, 480, va) == imgs[0].types.numel()
    assert VP.image_span_len(300, 900, va) == imgs[1].types.numel()


def test_vendored_image_processor_matches_checkpoint():
    ref = VP.checkpoint_image_processor()
    if ref is None:
        return  # no checkpoint here
    va = VisionArgs()
    cargs = VP.checkpoint_args(va)
    for w, h, s in [(640, 480, 0), (300, 900, 1), (64, 64, 2), (3000, 200, 3), (1400, 1000, 4)]:
        rec = {"data": _png(w, h, s)}
        a, b = VP._ip.load_image(rec, cargs), ref.load_image(rec, cargs)
        assert torch.equal(a[0], b[0]) and a[1:] == b[1:], (w, h)
        im = Image.open(io.BytesIO(rec["data"]))
        c = VP.patchify_pil(im, va)
        assert torch.equal(c[0], b[0]) and c[1:] == b[1:], (w, h)


class _FakeVision:
    """Stands in for DSV41Vision: spans of random rows, counts the batches it is asked to encode."""

    def __init__(self):
        self.a = VisionArgs()
        self.batches = []

    def image_rows_many(self, items):
        self.batches.append(len(items))
        return [torch.randn(types.numel(), 8).to(torch.bfloat16) for _, _, _, types in items]


def _request_images(sizes, seed):
    g = torch.Generator().manual_seed(seed)
    pv = [torch.randn(h * w, 588, generator=g).to(torch.bfloat16) for h, w in sizes]
    grids = torch.tensor([[1, h, w] for h, w in sizes])
    return torch.cat(pv), grids, pv


def _tokens(image_id, grids, a, gap=5):
    toks, starts = [7] * gap, []
    for g in grids:
        n = -(-int(g[1]) // a.ratio) * (-(-int(g[2]) // a.ratio) + 1) + 2
        starts.append(len(toks))
        toks += [image_id] * n + [8] * gap
    return torch.tensor(toks), starts


def test_vision_book_batches_images_of_a_call_and_caches():
    from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionBook

    fv = _FakeVision()
    book = VisionBook(fv, 129264, cache_images=8)
    a = fv.a
    pv0, gr0, _ = _request_images([(9, 12), (6, 6)], 1)
    pv1, gr1, _ = _request_images([(9, 12)], 1)  # the same first image as request 0 (same seed): encoded once
    t0, s0 = _tokens(129264, gr0, a)
    t1, s1 = _tokens(129264, gr1, a)
    out = book.update_many(
        [(0, t0, len(t0), pv0, gr0, 0), (5, t1, len(t1), pv1, gr1, 0), (9, torch.tensor([1, 2, 3]), 3, None, None, 0)]
    )
    assert fv.batches == [2], fv.batches  # 3 images, 2 distinct: ONE call of the tower with 2 images
    assert out[9] is None
    for p, starts, grids in ((0, s0, gr0), (5, s1, gr1)):
        seq = out[p]
        assert [s for s, _ in seq.spans] == starts
        m, emb = seq.window(starts[0], 6)
        assert m.all() and torch.equal(emb, seq.spans[0][1][:6])
    assert torch.equal(out[0].spans[0][1], out[5].spans[0][1])  # shared rows
    # the same image again later: cache hit, no tower call
    out2 = book.update_many([(1, t1, len(t1), pv1, gr1, 0)])
    assert fv.batches == [2] and book.hits >= 1 and torch.equal(out2[1].spans[0][1], out[0].spans[0][1])
    # a continuation chunk that now contains the second image of request 0: only that image is encoded, the first one's rows are kept
    cut = s0[1] + 3
    book2 = VisionBook(_FakeVision(), 129264, cache_images=0)
    first = book2.update_many([(0, t0[: s0[1] - 1], s0[1] - 1, pv0, gr0, 0)])[0]
    assert len(first.spans) == 1 and book2.vision.batches == [1]
    rows_first = first.spans[0][1]
    try:
        book2.update_many([(0, t0[:cut], cut, pv0, gr0, s0[1] - 1)])
        raise RuntimeError("a span cut by the chunk end must be rejected")
    except ValueError:
        pass
    second = book2.update_many([(0, t0, len(t0), pv0, gr0, s0[1] - 1)])[0]
    assert book2.vision.batches == [1, 1] and len(second.spans) == 2 and second.spans[0][1] is rows_first
