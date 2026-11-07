# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio tokenization in vLLM's front end: the upstream MiMo-V2-Omni processor plus the audio tokenizer encoder.

``MiMoV2OmniTTMultiModalProcessor`` runs vLLM 0.30.0's ``MiMoV2OmniMultiModalProcessor`` (placeholders, pixels, mels,
video timestamps; unchanged) and then replaces each audio item's ``audio_features`` mel [T, 128] by its RVQ codes
``audio_codes`` int32 [n_codes, 20] (25 Hz), computed in fp32 as the HF module computes them (reference/audio.py,
tt/audio/host_backend.py: bit-identical) from the fp32 mel (vLLM casts the processor's mel to bf16 before it leaves
the processor, which changed 8 of the 126 code frames of a 5 s clip; ``fp32_mels``). The engine then runs only the
patch encoder
(``AudioCodesBackend.embed_audio_codes``) inside prefill; the tokenizer encoder, 90-98 % of the audio work (1.6 s for
30 s of audio at 8 threads), leaves the engine process, where it held up every other request of the engine.

Where it runs: vLLM 0.30.0's renderer calls ``_process_multimodal`` (and so this processor) through
``make_async(..., executor=self._mm_executor)``, a single-worker ThreadPoolExecutor of the API server process, not on
its asyncio event loop. Torch kernels release the GIL, so the loop keeps streaming other requests' tokens: while a
120 s clip took 10.4 s there, a 10 ms text stream's largest gap was 20 ms (12 ms before; 10.9 s with the same work
on the loop), on bhg4 at load average ~37 (tests/perf/audio_frontend_stall.py). Multimodal items of one API server
queue behind each other on the one worker: a 5 s clip sent with that 120 s one finished after 10.7 s.
Each API server process (one per row engine, or per DP rank) loads its own copy of the tokenizer encoder (1.2 GB of
fp32 weights) on its first audio item, never at startup: an API server's OpenMP / oneDNN state must not be touched
before vLLM forks its engine-core processes (vllm/multimodal/audio.py).

Caching is unchanged: vLLM hashes an item's input data and processor kwargs, not its processed fields, and a cache
hit skips the processor entirely (``_cached_apply_hf_processor``), so a repeated clip is not tokenized again.

MIMO_AUDIO_FRONTEND_THREADS (default 8): torch threads of the tokenizer encoder in the front end (vLLM runs the
processor with OMP_NUM_THREADS threads, 1 when unset).
"""

import os
import threading

import torch
from loguru import logger
from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig

from ...reference import audio as ref
from .host_backend import HostAudioBackend

FRONTEND_THREADS = int(os.environ.get("MIMO_AUDIO_FRONTEND_THREADS", "8"))
_tokenizers = {}
_tokenizers_lock = threading.Lock()


def audio_checkpoint_path(model_config):
    """The local checkpoint directory holding ``audio_tokenizer/`` for vLLM's ``model_config``: the model path itself,
    its HF config's name, MIMO_MODEL_PATH, or the local HF hub snapshot of a repo id."""
    candidates = [getattr(model_config, "model", None)]
    hf_config = getattr(model_config, "hf_config", None)
    candidates.append(getattr(hf_config, "_name_or_path", None))
    candidates.append(os.environ.get("MIMO_MODEL_PATH"))
    for path in candidates:
        if path and os.path.exists(os.path.join(path, ref.AUDIO_TOKENIZER_DIR, "model.safetensors")):
            return path
    repo = candidates[0]
    if repo and not os.path.isabs(repo):
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo, revision=getattr(model_config, "revision", None), local_files_only=True)
            if os.path.exists(os.path.join(path, ref.AUDIO_TOKENIZER_DIR, "model.safetensors")):
                return path
        except Exception as e:  # noqa: BLE001 - fall through to the error below
            logger.warning(f"MiMo audio front end: no local snapshot of {repo}: {e}")
    raise RuntimeError(f"MiMo audio front end: no {ref.AUDIO_TOKENIZER_DIR}/model.safetensors under {candidates}")


def audio_tokenizer(model_path, num_threads=None):
    """This process's front-end tokenizer encoder for ``model_path`` (a tokenizer-only HostAudioBackend)."""
    with _tokenizers_lock:
        tok = _tokenizers.get(model_path)
        if tok is None:
            tok = HostAudioBackend(model_path, num_threads=num_threads or FRONTEND_THREADS, patch=False)
            _tokenizers[model_path] = tok
            logger.info(f"MiMo audio front end: tokenizer encoder loaded from {model_path}")
        return tok


def tokenize_mels(mels, model_path):
    """fp32 mels [T, 128] -> int32 [n_codes, 20] codes, one per item, each tokenized on its own (as the engine-side
    path did)."""
    tok = audio_tokenizer(model_path)
    return [tok.codes(m).to(torch.int32) for m in mels]


def fp32_mels(mels, waveforms, cfg):
    """The processor's mels in fp32. vLLM's ``call_hf_processor`` casts every floating output to the model dtype, so
    ``audio_features`` arrives in bf16; recompute each item's mel from its (resampled) waveform as
    ``MiMoVLProcessor.preprocess_audio`` does, and use it only if it rounds to the processor's bf16 mel (which also
    checks that items and waveforms line up). Otherwise keep the processor's mel (logged)."""
    if isinstance(mels, torch.Tensor):
        mels = list(mels) if mels.dim() == 3 else [mels]
    out = []
    for k, mel in enumerate(mels):
        mel32 = None
        if waveforms is not None and k < waveforms.get_count():
            wav = waveforms.get(k)
            if wav is not None:
                wav = torch.as_tensor(wav).float()
                mel32 = ref.log_mel_spectrogram(wav.mean(dim=0) if wav.dim() == 2 else wav, cfg)
        if mel32 is None or mel32.shape != mel.shape or not torch.equal(mel32.to(mel.dtype), mel):
            logger.warning(f"MiMo audio front end: audio item {k}: tokenizing the processor's {mel.dtype} mel")
            mel32 = mel.float()
        out.append(mel32)
    return out


class MiMoV2OmniTTMultiModalProcessor(MiMoV2OmniMultiModalProcessor):
    """The upstream processor, with each audio item's mel replaced by its RVQ codes (``audio_codes``)."""

    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs):
        out = super()._apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)
        if "audio_features" in out:
            if bool(torch.as_tensor(out.get("is_audio_tokenized", False)).any()):
                raise ValueError("pre-tokenized audio input is not served; send the audio itself")
            path = audio_checkpoint_path(self.info.ctx.model_config)
            mels = fp32_mels(
                out.pop("audio_features"), mm_items.get("audio"), ref.MiMoAudioConfig.from_model_path(path)
            )
            out["audio_codes"] = tokenize_mels(mels, path)
        return out

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        fields.pop("audio_features", None)
        fields["audio_codes"] = MultiModalFieldConfig.batched("audio")
        return fields
