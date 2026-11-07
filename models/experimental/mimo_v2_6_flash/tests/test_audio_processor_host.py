# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio tokenized in vLLM's front end (tt/audio/processor.py): no device.

Through vLLM 0.30.0's own renderer (the API server's input path, built from the checkpoint with the TT platform and
MiMoV2OmniForCausalLM, so the registered processor is ours), on LibriSpeech clips (16 kHz files, resampled by vLLM):

* each audio item reaches the engine as ``audio_codes`` int32 [n, 20], without ``audio_features``, and the codes are
  the fp32 reference's for the item's mel (two items in one request: each tokenized on its own);
* ``audio_token_lens`` and the placeholder range are unchanged (rows of the patch encoder's output);
* the engine side (``AudioCodesBackend.embed_audio_codes``, through tt/multimodal.encode_item) gives the embeddings
  today's engine-side path gives from the mel: bit-identical to the TEMPORARY HF host backend (fp32) with an fp32
  patch encoder, and to ``HostAudioBackend.embed_audio`` with the default bf16 one;
* a repeated clip is a processor-cache hit: the same hash, and the front end does not tokenize it again.

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


@pytest.fixture(scope="module")
def front(setup):
    """One request with two audio items through the renderer, from an empty processor cache."""
    setup["renderer"].clear_mm_cache()
    clips = setup["clips"]
    return process(setup["renderer"], [clips["speech_5s"]["raw"], clips["speech_65s"]["raw"]])


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


def test_repeated_clip_is_a_cache_hit(setup, front, monkeypatch):
    clips, renderer, module = setup["clips"], setup["renderer"], setup["module"]
    calls = []
    tokenize = module.tokenize_mels
    monkeypatch.setattr(module, "tokenize_mels", lambda mels, path: calls.append(len(mels)) or tokenize(mels, path))
    first = process(renderer, [clips["speech_5s"]["raw"]])  # processed by ``front`` already
    second = process(renderer, [clips["speech_5s"]["raw"]])
    assert first["mm_hashes"]["audio"] == second["mm_hashes"]["audio"] == front["mm_hashes"]["audio"][:1]
    assert calls == [], calls
    logger.info("repeated clip: same hash, not tokenized again")
