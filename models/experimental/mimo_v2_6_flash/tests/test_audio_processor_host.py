# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio tokenized in vLLM's front end (tt/audio/processor.py): no device.

Through vLLM 0.31.0's own renderer (the API server's input path, built from the checkpoint with the TT platform and
MiMoV2OmniForCausalLM, so the registered processor is ours), on LibriSpeech clips (16 kHz files, resampled by vLLM):

* with MIMO_AUDIO_TOKENIZER=frontend (MIMO_AUDIO_PATCH device or engine) each item reaches the engine as ``audio_codes`` int32
  [n, 20], without ``audio_features``, and the codes are the fp32 reference's for the item's mel (two items in one
  request: each tokenized on its own);
* with MIMO_AUDIO_PATCH=frontend it reaches the engine as ``audio_embeds`` bf16 [n_tokens, 4096]: bit for bit what
  the engine's host patch encoder makes from those codes, and tt/multimodal.encode_item hands them over unchanged
  (no encoder call);
* ``audio_token_lens`` and the placeholder range are unchanged (rows of the patch encoder's output);
* the engine side (``AudioCodesBackend.embed_audio_codes``, through tt/multimodal.encode_item) gives the embeddings
  today's engine-side path gives from the mel: bit-identical to the TEMPORARY HF host backend (fp32) with an fp32
  patch encoder, and to ``HostAudioBackend.embed_audio`` with the default bf16 one;
* by default (MIMO_AUDIO_TOKENIZER=device) it reaches the engine as ``audio_mel_bits``: the fp32 mel's int32 bits, exactly,
  with no encoder run in the front end, and tt/multimodal.encode_item hands the backend's ``embed_audio`` that mel;
* a repeated clip is a processor-cache hit: the same hash, and the front end does not encode it again.

Needs MIMO_MODEL_PATH with ``audio_tokenizer/``, MIMO_AUDIO_CLIPS, vllm and torchaudio.
"""

import os

import pytest
import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.tests.test_audio_reference_host import (
    audio_model_path,
    clips_dir,
    load_wav_24k,
    vllm_processor_or_none,
)

OMNI_CLASS = "models.experimental.mimo_v2_6_flash.tt.generator_vllm:MiMoV2OmniForCausalLM"
AUDIO_PLACEHOLDER = "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"


def build_renderer(model_path, max_model_len=32768):
    """vLLM's renderer for the checkpoint, as an API server builds it (no engine, no device)."""
    os.environ.setdefault("TT_MODEL_CLASS_OVERRIDES", f"TTMiMoV2OmniForCausalLM={OMNI_CLASS}")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers.registry import renderer_from_config

    config = EngineArgs(model=model_path, max_model_len=max_model_len, trust_remote_code=True).create_engine_config()
    return renderer_from_config(config)


def audio_prompt(renderer, n_items):
    return renderer.tokenizer.encode(AUDIO_PLACEHOLDER * n_items + "What is said?")


@pytest.fixture(scope="module")
def setup():
    path = audio_model_path()
    root = clips_dir()
    proc = vllm_processor_or_none()
    if proc is None:
        pytest.skip("vllm or torchaudio not importable")
    import soundfile as sf

    from models.experimental.mimo_v2_6_flash.tt.audio import processor as tt_processor

    renderer = build_renderer(path)
    assert isinstance(renderer.get_mm_processor(), tt_processor.MiMoV2OmniTTMultiModalProcessor)
    clips = {}
    for c in ("speech_5s", "speech_65s"):
        wav, sr = sf.read(os.path.join(root, f"{c}.wav"), dtype="float32")
        clips[c] = {
            "raw": (wav, sr),
            "mel": ref.log_mel_spectrogram(load_wav_24k(os.path.join(root, f"{c}.wav"), proc[1])),
        }
    return {"path": path, "renderer": renderer, "clips": clips, "module": tt_processor}


def process(renderer, items):
    """The API server's multimodal processing of a prompt with these audio items: vLLM's MultiModalInput."""
    prompt = audio_prompt(renderer, len(items))
    return renderer._process_multimodal(prompt, {"audio": list(items)}, None, None)


def _two_items(setup, frontend_patch):
    """One request with two audio items through the renderer, from an empty processor cache."""
    module = setup["module"]
    saved = module.FRONTEND_PATCH, module.DEVICE_TOKENIZER
    module.FRONTEND_PATCH, module.DEVICE_TOKENIZER = frontend_patch, False  # the front end's own tokenizer encoder
    try:
        setup["renderer"].clear_mm_cache()
        clips = setup["clips"]
        return process(setup["renderer"], [clips["speech_5s"]["raw"], clips["speech_65s"]["raw"]])
    finally:
        module.FRONTEND_PATCH, module.DEVICE_TOKENIZER = saved


@pytest.fixture(scope="module")
def front(setup):
    """MIMO_AUDIO_TOKENIZER=frontend, MIMO_AUDIO_PATCH=device or engine: the front end sends codes."""
    return _two_items(setup, False)


@pytest.fixture(scope="module")
def front_embeds(setup, front):
    """MIMO_AUDIO_PATCH=frontend: the front end sends embeddings."""
    return _two_items(setup, True)


def test_front_end_emits_codes(setup, front):
    clips = setup["clips"]
    cfg = ref.MiMoAudioConfig.from_model_path(setup["path"])
    enc = ref.AudioTokenizerEncoder(ref.load_audio_weights(setup["path"], patch=False), cfg, torch.float32)
    for k, clip in enumerate(("speech_5s", "speech_65s")):
        data = front["mm_kwargs"]["audio"][k].get_data()
        assert "audio_features" not in data and data["audio_codes"].dtype == torch.int32
        mel = clips[clip]["mel"]
        want = ref.tokenize_audio_batch([mel], enc)[0]
        assert torch.equal(data["audio_codes"].long(), want), clip
        n = ref.audio_token_len(mel.shape[0], cfg)
        assert int(data["audio_token_lens"]) == n == front["mm_placeholders"]["audio"][k].length
        # What the engine-side path tokenized: vLLM hands the processor's mel over in the model dtype.
        from_bf16 = ref.tokenize_audio_batch([mel.to(torch.bfloat16).float()], enc)[0]
        logger.info(
            f"{clip}: front end gave {tuple(data['audio_codes'].shape)} codes == reference (fp32 mel), {n} "
            f"placeholders; the bf16 mel the engine used to get changes {int((from_bf16 != want).any(1).sum())} frames"
        )


def test_engine_side_embeddings_unchanged(setup, front):
    """The same mel through today's engine-side paths, and as front-end codes through the engine's patch encoder."""
    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import AudioCodesBackend, HostAudioBackend

    clips, path = setup["clips"], setup["path"]
    hf = mm.HFHostBackend(path, dtype=torch.float32, modalities=("audio",))  # today's default engine-side path
    full = HostAudioBackend(path)
    codes_fp32 = HostAudioBackend(path, tokenizer=False, patch_dtype=torch.float32)
    engine = AudioCodesBackend(object(), path)  # bf16 patch encoder, as served
    for k, clip in enumerate(("speech_5s", "speech_65s")):
        mel, codes = clips[clip]["mel"], front["mm_kwargs"]["audio"][k].get_data()["audio_codes"]
        n = ref.audio_token_len(mel.shape[0])
        item = mm.MMItem("audio", clip, 10, n, None, {"audio_codes": codes, "audio_token_lens": torch.tensor(n)})
        served = mm.encode_item(engine, mm.parse_item(item))
        assert torch.equal(served, full.embed_audio(mel)), clip
        assert torch.equal(codes_fp32.embed_audio_codes(codes), hf.embed_audio(mel)), clip
        logger.info(f"{clip}: codes path == mel path (bf16 patch), and == the HF host backend (fp32 patch)")


def test_front_end_emits_embeds(setup, front, front_embeds):
    """The front end's patch encoder gives what the engine's gives from the same codes, and the engine uploads it."""
    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import AudioCodesBackend

    engine = AudioCodesBackend(object(), setup["path"])  # the engine's host patch encoder (bf16)
    no_encoder = object()  # an encoder backend without any method: audio_embeds must not need one
    for k, clip in enumerate(("speech_5s", "speech_65s")):
        data = front_embeds["mm_kwargs"]["audio"][k].get_data()
        codes = front["mm_kwargs"]["audio"][k].get_data()["audio_codes"]
        assert "audio_features" not in data and "audio_codes" not in data, sorted(data)
        embeds, n = data["audio_embeds"], int(data["audio_token_lens"])
        assert embeds.dtype == torch.bfloat16 and embeds.shape == (n, 4096), (embeds.dtype, embeds.shape)
        assert n == front_embeds["mm_placeholders"]["audio"][k].length
        assert torch.equal(embeds, engine.embed_audio_codes(codes)), clip
        item = mm.MMItem("audio", clip, 10, n, None, {"audio_embeds": embeds, "audio_token_lens": torch.tensor(n)})
        assert torch.equal(mm.encode_item(no_encoder, mm.parse_item(item)), embeds), clip
        logger.info(f"{clip}: front-end embeddings {tuple(embeds.shape)} == the engine's patch encoder on the codes")


def test_front_end_device_tokenizer_sends_the_mel(setup, monkeypatch):
    """MIMO_AUDIO_TOKENIZER=device: the item carries the fp32 mel bit for bit and the front end encodes nothing."""
    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm

    module, clips = setup["module"], setup["clips"]
    calls = []
    for name in ("tokenize_mels", "embed_mels"):
        monkeypatch.setattr(module, name, lambda mels, path: calls.append(len(mels)))
    monkeypatch.setattr(module, "DEVICE_TOKENIZER", True)
    setup["renderer"].clear_mm_cache()
    out = process(setup["renderer"], [clips["speech_5s"]["raw"], clips["speech_65s"]["raw"]])
    setup["renderer"].clear_mm_cache()
    assert calls == [], calls

    class Capture:
        def embed_audio(self, mel):
            self.mel = mel
            return torch.zeros(ref.audio_token_len(mel.shape[0]), 4096)

    for k, clip in enumerate(("speech_5s", "speech_65s")):
        data = out["mm_kwargs"]["audio"][k].get_data()
        assert sorted(f for f in data if f.startswith("audio_")) == ["audio_mel_bits", "audio_token_lens"], sorted(data)
        bits, mel = data["audio_mel_bits"], clips[clip]["mel"]
        assert bits.dtype == torch.int32 and torch.equal(bits.view(torch.float32), mel), clip
        n = int(data["audio_token_lens"])
        backend = Capture()
        item = mm.MMItem("audio", clip, 10, n, None, {"audio_mel_bits": bits, "audio_token_lens": torch.tensor(n)})
        mm.encode_item(backend, mm.parse_item(item))
        assert torch.equal(backend.mel, mel), clip
    logger.info("MIMO_AUDIO_TOKENIZER=device: the fp32 mels reach the engine bit for bit, nothing encoded in front")


def test_repeated_clip_is_a_cache_hit(setup, front_embeds, monkeypatch):
    clips, renderer, module = setup["clips"], setup["renderer"], setup["module"]
    calls = []
    for name in ("tokenize_mels", "embed_mels"):
        fn = getattr(module, name)
        monkeypatch.setattr(module, name, lambda mels, path, fn=fn: calls.append(len(mels)) or fn(mels, path))
    first = process(renderer, [clips["speech_5s"]["raw"]])  # encoded now, or still cached from an earlier test
    calls.clear()
    second = process(renderer, [clips["speech_5s"]["raw"]])
    assert first["mm_hashes"]["audio"] == second["mm_hashes"]["audio"] == front_embeds["mm_hashes"]["audio"][:1]
    assert calls == [], calls
    logger.info("repeated clip: same hash, not encoded again")
