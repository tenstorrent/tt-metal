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
