# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Windowed vision attention (`PaddingWindows`, 8+ frames at bucket 1280) vs the masked path.

`VisionModel.forward_hidden` replaces the dense key-padding mask with SDPA `cu_window_seqlens`
windows (`attention.padding_windows`) for WINDOWED_MIN_BATCH+ rows of bucket 1280. Both paths must
give the same valid-row output for the same input. The frames below have different valid lengths
(tile aligned and not, 2 padding rows up to 1199), so a window that crosses a frame boundary, ends
at the wrong row or includes padding keys changes the valid rows of at least one frame.

Padding rows: the windowed path gives padding queries their own window, so their outputs differ
from the masked path and must never be read. The tests check that (a) valid-row outputs do not
change when the padding rows of the input are overwritten with large noise and (b) the pooling
matrix, the only consumer of the encoder rows, has zero weight on every padding row.
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import download_rows, upload_rows
from models.demos.blackhole.embedding_gemma2.tt.vision.attention import (
    WINDOWED_MIN_BATCH,
    WINDOWED_SEQ_LENS,
    PaddingWindows,
    padding_windows,
    use_windowed_attention,
)
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_model import key_padding_mask

S = 1280
# per-frame patch grids (w, h), both multiples of the 3x3 pooling kernel; valid = w * h patches
FRAME_GRIDS = ((45, 24), (42, 30), (36, 27), (426, 3), (9, 9), (39, 30), (30, 21), (48, 24))
FRAME_VALID = [w * h for w, h in FRAME_GRIDS]  # 1080 1260 972 1278 81 1170 630 1152
N = max(FRAME_VALID)  # 1278 -> bucket 1280
PADDING_NOISE = 50.0


def synthetic_frames(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(pixel_values [B, N, 768], position_ids [B, N, 2]): frame b has a w_b x h_b prefix of valid patches."""
    B = len(FRAME_GRIDS)
    pid = torch.full((B, N, 2), -1, dtype=torch.int64)
    for b, (w, h) in enumerate(FRAME_GRIDS):
        i = torch.arange(w * h)
        pid[b, : w * h, 0] = i % w
        pid[b, : w * h, 1] = i // w
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, 768, generator=g), pid


def stub_encoder(vision_config) -> VisionEncoder:
    """Host-only encoder: `prepare_inputs` reads only `model.config`."""
    return VisionEncoder(SimpleNamespace(config=vision_config), device=None)


# ------------------------------------------------------------------------------ host: structure


def test_padding_windows_structure():
    cu = padding_windows(FRAME_VALID, S)
    assert cu.dtype == torch.int32
    expect = []
    for b, n in enumerate(FRAME_VALID):
        expect += [b * S, b * S + n]
    expect.append(len(FRAME_VALID) * S)
    assert cu.tolist() == expect
    assert len(cu) == 2 * len(FRAME_VALID) + 1
    assert bool((cu[1:] > cu[:-1]).all()), "every window must be non-empty"
    # window 2b is frame b's valid prefix, window 2b+1 its padding; no window crosses a frame boundary
    for b, n in enumerate(FRAME_VALID):
        assert (int(cu[2 * b]), int(cu[2 * b + 1]), int(cu[2 * b + 2])) == (b * S, b * S + n, (b + 1) * S)


@pytest.mark.parametrize("bad", (0, S, S + 1, -1))
def test_padding_windows_rejects_empty_window(bad, expect_error):
    valid = list(FRAME_VALID)
    valid[3] = bad
    with expect_error(ValueError, "row 3"):
        padding_windows(valid, S)


def test_use_windowed_attention_gate():
    assert WINDOWED_SEQ_LENS == (S,) and WINDOWED_MIN_BATCH == 8
    assert use_windowed_attention(8, S) and use_windowed_attention(9, S)
    assert not use_windowed_attention(7, S)
    assert not use_windowed_attention(8, 2560)


def test_prepare_inputs_windows_and_pool(vision_config):
    """The encoder builds the windows of the valid prefixes and never pools a padding row."""
    pv, pid = synthetic_frames()
    host = stub_encoder(vision_config).prepare_inputs(pv, pid)
    assert (host.batch, host.seq_len, host.num_valid) == (len(FRAME_GRIDS), S, FRAME_VALID)
    windows = ttnn.to_torch(host.tensors.windows).to(torch.int64).flatten()
    assert windows.tolist() == padding_windows(FRAME_VALID, S).tolist()
    pool = ttnn.to_torch(host.tensors.pool).float()  # [B, 1, S_out, S]
    for b, n in enumerate(FRAME_VALID):
        assert float(pool[b, 0, :, n:].abs().sum()) == 0.0, f"frame {b}: padding rows {n}..{S - 1} are pooled"
        assert float(pool[b, 0, :, :n].sum()) == n, f"frame {b}: a valid row is not pooled exactly once"


def test_prepare_inputs_windowed_needs_prefix(vision_config, expect_error):
    pv, pid = synthetic_frames()
    pid[2, 5] = -1  # a padding patch inside frame 2's valid prefix
    with expect_error(ValueError, "prefix"):
        stub_encoder(vision_config).prepare_inputs(pv, pid)


# ------------------------------------------------------------------------------ device: numerics


@pytest.mark.parametrize("layer", (0, 15))
def test_windowed_matches_masked(layer, eg2_device, tt_vision_encoder):
    enc = tt_vision_encoder
    cfg = enc.config
    attn = enc.model.layers[layer].self_attn
    pv, pid = synthetic_frames()
    host = enc.prepare_inputs(pv, pid)
    B = host.batch
    assert use_windowed_attention(B, host.seq_len)
    x = enc.to_device(host)

    g = torch.Generator().manual_seed(1 + layer)
    h = torch.randn(B, N, cfg.hidden_size, generator=g)
    h_noisy = h.clone()
    for b, n in enumerate(FRAME_VALID):
        h_noisy[b, n:] = PADDING_NOISE * torch.randn(N - n, cfg.hidden_size, generator=g)
        h[b, n:] = 0.0

    # windowed: what VisionModel.forward_hidden passes for 8+ frames at bucket 1280
    D = x.cos.shape[-1]
    w_cos, w_sin = (ttnn.reshape(t, (1, 1, B * S, D)) for t in (x.cos, x.sin))
    windows = PaddingWindows(x.windows)
    # masked: the dense bfp4 key-padding mask and head-expanded RoPE tables (every other shape)
    mask = key_padding_mask(x.keep)
    m_cos, m_sin = enc.model.rope.expand(x.cos, x.sin, cfg.num_heads)

    x_dev = upload_rows(eg2_device, h, S)
    x_noisy_dev = upload_rows(eg2_device, h_noisy, S)
    y_win = download_rows(attn(x_dev, windows, w_cos, w_sin), S)
    y_win_noisy = download_rows(attn(x_noisy_dev, windows, w_cos, w_sin), S)
    y_mask = download_rows(attn(x_dev, mask, m_cos, m_sin), S)
    y_mask_noisy = download_rows(attn(x_noisy_dev, mask, m_cos, m_sin), S)

    failures = []
    for b, n in enumerate(FRAME_VALID):
        value = pcc(y_mask[b, :n], y_win[b, :n])
        max_abs = float((y_mask[b, :n] - y_win[b, :n]).abs().max())
        logger.info(f"layer {layer} frame {b} (valid {n}): windowed vs masked max |diff| {max_abs:.4g}")
        if not log_result("vision_windowed_vs_masked_pcc", value, PCC_LAYER, layer=layer, frame=b, valid=n):
            failures.append(f"frame {b} (valid {n}): windowed vs masked PCC {value:.6f}")
        # Zero padding rows give zero v, so windowed padding queries (own window) output 0 while masked
        # padding queries average the valid v: the padding rows differ by design and are never pooled.
        pad_win = float(y_win[b, n:].abs().max())
        pad_mask = float(y_mask[b, n:].abs().max())
        if pad_win != 0.0 or pad_mask == 0.0:
            failures.append(
                f"frame {b}: padding rows windowed max {pad_win:.4g} (expect 0, own window), masked {pad_mask:.4g}"
            )
        # valid rows must not depend on the padding rows of their own frame or any other frame
        for name, clean, noisy in (("windowed", y_win, y_win_noisy), ("masked", y_mask, y_mask_noisy)):
            diff = float((clean[b, :n] - noisy[b, :n]).abs().max())
            logger.info(f"layer {layer} frame {b} (valid {n}): {name} padding-noise max |diff| {diff:.4g}")
            if diff != 0.0:
                failures.append(f"frame {b}: {name} valid rows change by {diff:.4g} when the padding rows change")
    assert not failures, "; ".join(failures)
