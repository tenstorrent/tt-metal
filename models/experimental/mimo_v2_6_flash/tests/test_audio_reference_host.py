# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio input reference (reference/audio.py) against the HF remote code and vLLM's processor: no device.

On real speech (reference/make_audio_clips.py: 5 s, 30 s and 65 s, the last over one 6000-frame segment):

* the mel equals vLLM 0.31.0's ``MiMoVLProcessor.preprocess_audio`` bit for bit, after vLLM's own resampling of the
  16 kHz file to 24 kHz, and our token count equals its ``audio_token_lens``;
* in fp32 the 20 RVQ codes are identical to the HF ``MiMoAudioTokenizer`` encoder's, and the [n, 4096]
  embeddings to ``MiMoAudioEncoder.get_audio_feature``'s; n == ``audio_token_lens``;
* in bf16 with bf16-rounded buffers (how HF's ``load_audio_tokenizer`` and vLLM load the tokenizer) the codes are
  identical to the HF module cast to bf16, so the reference also reproduces the GPU serving dtype path;
* fp64 gives the fp32 codes (the codes are not at a rounding tie in fp32), bf16 does not: the logged flip rates and
  embedding PCCs are why the host backend runs the tokenizer encoder in fp32.

The HF reference is ``modeling_mimo_v2.py`` imported from the checkpoint. Needs MIMO_MODEL_PATH with
``audio_tokenizer/`` and MIMO_AUDIO_CLIPS; the processor checks also need vllm and torchaudio.
"""

import importlib
import json
import math
import os
import sys
import types

import pytest
import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path

CLIPS = ("speech_5s", "speech_30s", "speech_65s")
THREADS = int(os.environ.get("MIMO_AUDIO_THREADS", "16"))


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def audio_model_path():
    path = default_model_path()
    if not os.path.exists(os.path.join(path, ref.AUDIO_TOKENIZER_DIR, "model.safetensors")):
        pytest.skip(f"no {ref.AUDIO_TOKENIZER_DIR}/model.safetensors under {path} (set MIMO_MODEL_PATH)")
    return path


def clips_dir():
    path = os.environ.get("MIMO_AUDIO_CLIPS", "/data/ttuser-agentx/mm/audio/clips")
    if not os.path.exists(os.path.join(path, "speech_5s.wav")):
        pytest.skip(f"speech clips not found in {path} (reference/make_audio_clips.py; set MIMO_AUDIO_CLIPS)")
    return path


def require_clips(names):
    """{name: path of its .wav} for every one of ``names``; skips when any is missing. A device test that quietly
    left out the clips it did not find passed on speech_5s alone, without its multi-segment (65 / 120 / 300 s) cases."""
    root = clips_dir()
    paths = {name: os.path.join(root, f"{name}.wav") for name in names}
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        pytest.skip(f"speech clips {missing} not found in {root} (reference/make_audio_clips.py; set MIMO_AUDIO_CLIPS)")
    return paths


def vllm_processor_or_none():
    """vLLM's MiMoVLProcessor (only its audio path is used) and vLLM's resampler, or None without vllm / torchaudio."""
    try:
        import torchaudio  # noqa: F401
        from vllm.multimodal.audio import resample_audio_torchaudio
        from vllm.transformers_utils.processors.mimo_v2_omni import MiMoVLProcessor
    except ImportError:
        return None
    pixels = dict(
        image_min_pixels=1, image_max_pixels=2, video_min_pixels=1, video_max_pixels=2, video_total_max_pixels=2
    )
    return MiMoVLProcessor(tokenizer=None, fps=1.0, **pixels), resample_audio_torchaudio


def load_wav_24k(path, resample):
    """The serving path for a file: soundfile decode to fp32 mono (vLLM load_audio), then vLLM's resampler."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return torch.from_numpy(resample(wav, orig_sr=sr, target_sr=24000)).float()


def hf_remote_module(model_path):
    """``modeling_mimo_v2.py`` from the checkpoint, imported as a package so its relative import resolves."""
    name = "mimo_hf_remote"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [model_path]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.modeling_mimo_v2")


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(THREADS)
    path = audio_model_path()
    clip_root = clips_dir()
    proc = vllm_processor_or_none()
    resample = proc[1] if proc is not None else None
    if resample is None:  # without vllm: torchaudio's resampler with the defaults vLLM uses
        taf = pytest.importorskip("torchaudio.functional")
        resample = lambda w, orig_sr, target_sr: taf.resample(torch.from_numpy(w), orig_sr, target_sr).numpy()
    cfg = ref.MiMoAudioConfig.from_model_path(path)
    weights = ref.load_audio_weights(path)
    wavs = {c: load_wav_24k(os.path.join(clip_root, f"{c}.wav"), resample) for c in CLIPS}
    mels = {c: ref.log_mel_spectrogram(w, cfg) for c, w in wavs.items()}
    return {"path": path, "cfg": cfg, "weights": weights, "wavs": wavs, "mels": mels, "proc": proc}


@pytest.fixture(scope="module")
def hf(setup):
    """HF modules in fp32: the tokenizer (``load_audio_tokenizer`` without the bf16 cast) and the audio encoder."""
    from safetensors.torch import load_file

    M = hf_remote_module(setup["path"])
    with open(os.path.join(setup["path"], ref.AUDIO_TOKENIZER_DIR, "config.json")) as f:
        tok_cfg = M.MiMoAudioTokenizerConfig(**json.load(f))
    tok_state = load_file(os.path.join(setup["path"], ref.AUDIO_TOKENIZER_DIR, "model.safetensors"))
    tok = M.MiMoAudioTokenizer(tok_cfg)
    missing, _ = tok.load_state_dict(tok_state, strict=False)
    assert not missing
    with open(os.path.join(setup["path"], "config.json")) as f:
        audio_cfg = M._as_namespace(json.load(f)["audio_config"])
    speech = M._build_speech_embeddings(audio_cfg)
    enc = M.MiMoAudioEncoder(audio_cfg)
    w = setup["weights"]
    speech.load_state_dict(
        {k[len("speech_embeddings.") :]: v for k, v in w.items() if k.startswith("speech_embeddings.")}
    )
    missing, unexpected = enc.load_state_dict(
        {k[len("audio_encoder.") :]: v for k, v in w.items() if k.startswith("audio_encoder.")}, strict=False
    )
    assert missing == ["input_local_transformer.embed_tokens.weight"] and not unexpected  # unused (inputs_embeds)
    return {
        "M": M,
        "tok": tok.float().eval(),
        "tok_cfg": tok_cfg,
        "tok_state": tok_state,
        "speech": speech.float().eval(),
        "enc": enc.float().eval(),
    }


def test_token_counts():
    """audio_token_lens (processor, whole mel) == ceil(code frames / 4) (model, per 6000-frame segment) for every
    length through 10 segments; 6.25 tokens per second of audio."""
    cfg = ref.MiMoAudioConfig()
    for T in range(1, 10 * cfg.segment_size + 7):
        assert ref.audio_token_len(T, cfg) == math.ceil(ref.num_codes(T, cfg) / cfg.group_size), T
    # 1 s: 101 mel frames -> 51 -> 26 codes -> 7 tokens; long clips approach 25 codes / 4 = 6.25 tokens per second.
    assert ref.audio_token_len(ref.num_mel_frames(24000)) == 7
    for seconds in (60, 300, 600):
        n = ref.audio_token_len(ref.num_mel_frames(seconds * 24000))
        assert n == math.ceil(seconds * 6.25 + 0.25), (seconds, n)


def test_mel_and_token_count_match_vllm_processor(setup):
    if setup["proc"] is None:
        pytest.skip("vllm or torchaudio not importable")
    proc = setup["proc"][0]
    for clip, wav in setup["wavs"].items():
        mel, tok_len = proc.preprocess_audio((wav, 24000))
        assert torch.equal(mel, setup["mels"][clip]), (clip, (mel - setup["mels"][clip]).abs().max())
        assert tok_len == ref.audio_token_len(mel.shape[0], setup["cfg"]), clip
        logger.info(f"{clip}: {wav.shape[0] / 24000:.2f} s, mel {tuple(mel.shape)} bit-identical, {tok_len} tokens")
    assert setup["mels"]["speech_65s"].shape[0] > setup["cfg"].segment_size  # the segmenting path is exercised


@pytest.mark.parametrize("clip", CLIPS)
def test_reference_matches_hf_fp32(setup, hf, clip):
    mel = setup["mels"][clip]
    with torch.no_grad():
        hf_codes = hf["M"].tokenize_audio_batch([mel], hf["tok"].encoder, segment_size=setup["cfg"].segment_size)[0]
        hf_emb = hf["enc"].get_audio_feature([mel], hf["speech"], hf["tok"].encoder)
    R = ref.MiMoAudioReference(setup["weights"], setup["cfg"])
    codes = R.codes([mel])[0]
    emb, lens = R.embed_codes([codes])
    assert codes.shape == (ref.num_codes(mel.shape[0], setup["cfg"]), 20)
    assert torch.equal(codes, hf_codes), (codes != hf_codes).any(1).nonzero().flatten()[:10]
    assert emb.shape == hf_emb.shape == (ref.audio_token_len(mel.shape[0], setup["cfg"]), 4096) and lens == [
        emb.shape[0]
    ]
    p, diff = pcc(emb, hf_emb), (emb - hf_emb).abs().max().item()
    logger.info(
        f"{clip}: {codes.shape[0]} code frames identical, {emb.shape[0]} rows, PCC {p:.8f}, max |diff| {diff:.3g}"
    )
    assert p >= 0.9999


@pytest.mark.parametrize("clip", ("speech_5s", "speech_65s"))
def test_reference_bf16_matches_hf_bf16(setup, hf, clip):
    """The GPU serving dtype path: tokenizer cast to bf16 (buffers too), RVQ back in fp32."""
    tok = hf["M"].MiMoAudioTokenizer(hf["tok_cfg"])
    tok.load_state_dict(hf["tok_state"], strict=False)
    tok = tok.to(torch.bfloat16).eval()
    mel = setup["mels"][clip]
    with torch.no_grad():
        hf_codes = hf["M"].tokenize_audio_batch([mel], tok.encoder, segment_size=setup["cfg"].segment_size)[0]
    enc = ref.AudioTokenizerEncoder(setup["weights"], setup["cfg"], torch.bfloat16, bf16_buffers=True)
    codes = ref.tokenize_audio_batch([mel], enc)[0]
    assert torch.equal(codes, hf_codes), (codes != hf_codes).any(1).float().mean()


def test_code_precision(setup):
    """fp64 reproduces the fp32 codes; bf16 (with or without bf16 buffers) flips many, mostly in late codebooks.
    Logs the flip rates and what they do to the embeddings (fp32 patch encoder on both sides)."""
    cfg, w = setup["cfg"], setup["weights"]
    patch = ref.AudioPatchEncoder(w, cfg, torch.float32)
    enc32 = ref.AudioTokenizerEncoder(w, cfg, torch.float32)
    enc64 = ref.AudioTokenizerEncoder(w, cfg, torch.float64)
    enc16 = ref.AudioTokenizerEncoder(w, cfg, torch.bfloat16, bf16_buffers=True)
    patch16 = ref.AudioPatchEncoder(w, cfg, torch.bfloat16)
    for clip in ("speech_5s", "speech_30s"):
        mel = setup["mels"][clip]
        c32 = ref.tokenize_audio_batch([mel], enc32)[0]
        assert torch.equal(ref.tokenize_audio_batch([mel], enc64)[0], c32), clip
        c16 = ref.tokenize_audio_batch([mel], enc16)[0]
        flips = c16 != c32
        e32 = patch(ref.pad_and_group_codes(c32, cfg))
        e16 = patch(ref.pad_and_group_codes(c16, cfg))
        ep = patch16(ref.pad_and_group_codes(c32, cfg))
        logger.info(
            f"{clip}: bf16 encoder flips {flips.any(1).float().mean():.3f} of the code frames (codebook 0: "
            f"{flips[:, 0].float().mean():.3f}, 19: {flips[:, -1].float().mean():.3f}); embeddings PCC "
            f"{pcc(e16, e32):.5f}. bf16 patch encoder on fp32 codes: PCC {pcc(ep, e32):.6f}"
        )
        assert flips.any(), "bf16 became code-exact: revisit the host backend's fp32 tokenizer encoder"
        assert pcc(e16, e32) > 0.95 and pcc(ep, e32) > 0.9995


def test_batching_changes_only_the_last_code_frame(setup):
    """Items encoded in one call vs alone (fp32): only an item's last code frame can differ (its last encoder
    frame sees batch-dependent padding when its length is odd)."""
    enc = ref.AudioTokenizerEncoder(setup["weights"], setup["cfg"], torch.float32)
    mels = [setup["mels"][c] for c in CLIPS]
    alone = [ref.tokenize_audio_batch([m], enc)[0] for m in mels]
    together = ref.tokenize_audio_batch(mels, enc)
    for clip, a, t in zip(CLIPS, alone, together):
        assert torch.equal(a[:-1], t[:-1]), clip
        logger.info(f"{clip}: last code frame {'differs' if not torch.equal(a[-1], t[-1]) else 'equal'} when batched")
