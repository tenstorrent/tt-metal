"""Vendored codec vs upstream (reference venv), chunked decode exactness, frame geometry."""
import numpy as np
import pytest
import torch

from models.autoports.fishaudio_s2_pro.config import SAMPLES_PER_FRAME
from models.autoports.fishaudio_s2_pro.tt.codec.codec_decoder import CPUCodec


@pytest.fixture(scope="module")
def codec(snapshot):
    torch.set_num_threads(4)
    return CPUCodec(snapshot)


def _codes(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.randint(0, 4096, (1, T), generator=g), torch.randint(0, 1024, (9, T), generator=g)])


def test_geometry(codec):
    wav = codec.decode(_codes(30))
    assert wav.shape == (30 * SAMPLES_PER_FRAME,) and np.abs(wav).max() <= 1.0


def test_matches_upstream(codec, snapshot, upstream):
    from fish_speech.models.dac.inference import load_model

    theirs = load_model("modded_dac_vq", f"{snapshot}/codec.pth", device="cpu").float()
    c = _codes(40)
    with torch.inference_mode():
        b = theirs.from_indices(c[None].clone())[0, 0].numpy()
    a = codec.decode(c)
    assert np.abs(a - np.clip(b, -1, 1)).max() < 1e-4
    wav = torch.from_numpy(b).view(1, 1, -1)
    with torch.inference_mode():
        ib, lb = theirs.encode(wav, torch.tensor([wav.shape[-1]]))
    ia = codec.encode(b)
    assert torch.equal(ia, ib[0, :, : int(lb[0])])


def test_chunked_matches_whole(codec):
    c = _codes(700, seed=5)
    whole = codec.decode(c)
    parts = list(codec.decode_chunked(c, chunk_frames=32, ctx_frames=512))
    cat = np.concatenate(parts)
    assert cat.shape == whole.shape
    # 8 window-128 transformer layers => ~512 frames of left context for the tail to converge; positions are absolute
    err = np.abs(cat - whole)
    assert err.max() < 1e-3, err.max()
    pcc = np.corrcoef(cat, whole)[0, 1]
    assert pcc > 0.99999, pcc
    # and the offset really matters: restarting positions at 0 is measurably worse
    worse = codec._decode_with_offset(c[:, 700 - 32 - 512 :], 0)[512 * SAMPLES_PER_FRAME :]
    assert np.abs(worse - whole[(700 - 32) * SAMPLES_PER_FRAME :]).max() > err.max()
