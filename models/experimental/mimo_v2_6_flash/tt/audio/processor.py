# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio encoding in vLLM's front end: the upstream MiMo-V2-Omni processor plus the host audio encoders.

``MiMoV2OmniTTMultiModalProcessor`` runs vLLM 0.31.0's ``MiMoV2OmniMultiModalProcessor`` (placeholders, pixels, mels,
video timestamps; unchanged) and then replaces each audio item's ``audio_features`` mel [T, 128] by the fp32 mel
for the engine's device tokenizer encoder (MIMO_AUDIO_TOKENIZER=device, the default; below), or with
MIMO_AUDIO_TOKENIZER=frontend by its RVQ codes (25 Hz) computed here in fp32 as the HF module computes them
(reference/audio.py, tt/audio/host_backend.py: bit-identical) from the fp32 mel (vLLM casts the processor's mel to
bf16 before it leaves the processor, which changed 8 of the 126 code frames of a 5 s clip; ``fp32_mels``), or with
MIMO_AUDIO_PATCH=frontend by the LLM embeddings the patch encoder makes from them (below). The host tokenizer encoder
(1.6 s for 30 s of audio at 8 threads) runs here, never in the engine process, where it held up every other request
of the engine. Only a bit-identical fp32 computation keeps every code (its RVQ decisions sit at fp32 rounding level,
tests/perf/audio_codes_margin.py); the device encoder's few percent of changed frames cost no accuracy (below).

Where it runs: vLLM 0.31.0's renderer calls ``_process_multimodal`` (and so this processor) through
``make_async(..., executor=self._mm_executor)``, a single-worker ThreadPoolExecutor of the API server process, not on
its asyncio event loop. Torch kernels release the GIL, so the loop keeps streaming other requests' tokens: while a
120 s clip took 10.4 s there, a 10 ms text stream's largest gap was 20 ms (12 ms before; 10.9 s with the same work
on the loop), on bhg4 at load average ~37 (tests/perf/audio_frontend_stall.py). Multimodal items of one API server
queue behind each other on the one worker: a 5 s clip sent with that 120 s one finished after 10.7 s (9.7 s with the
aligned windowed attention, load average ~20). vLLM 0.31 offers no setting for more workers: ``BaseRenderer``
creates ``ThreadPoolExecutor(max_workers=1)`` and requires it ("must stay single-worker per #38418": the sender-side
processor cache must see items in the order the engine receives them). The tokenizer encoder is ~98 % of an item's
time here (30 s on bhg2: 1.63 s, of which resampling 6 ms, the upstream processor 17 ms, ``fp32_mels`` 15 ms), so
the queueing shrinks only with the encoder's own time: MIMO_AUDIO_FRONTEND_THREADS=16 takes 30 s / 120 s from 1.74
/ 8.0 s to 0.97 / 4.8 s on an idle host (tt/audio/host_backend.py), at the cost of 16 cores while it runs next to
the row engines' own CPU work (not measured under serving load).
Each API server process (one per row engine, or per DP rank) loads its own copy of the tokenizer encoder (1.2 GB of
fp32 weights) on its first audio item, never at startup: an API server's OpenMP / oneDNN state must not be touched
before vLLM forks its engine-core processes (vllm/multimodal/audio.py).

Caching is unchanged: vLLM hashes an item's input data and processor kwargs, not its processed fields, and a cache
hit skips the processor entirely (``_cached_apply_hf_processor``), so a repeated clip is not tokenized again.

MIMO_AUDIO_PATCH says where the patch encoder (codes -> LLM embeddings) runs; the API server and the engine read it:

* ``device`` (the default): on the engine's row (tt/audio/patch.py, TP=8): the item reaches the engine as
  ``audio_codes`` int32 [n_codes, 20] (or the mel, below) and the engine's ``AudioCodesBackend`` runs the device patch
  encoder: 15 / 11 / 29 / 60 ms for 5 / 30 / 120 / 300 s of audio on bhg2 (host embedding sum and transfers included;
  PCC 0.9998 against the host bf16 patch encoder, 0.9999 against fp32), where the host patch encoder takes 35 / 145 /
  610 ms of CPU (bf16, 8 threads, bhg4) during which the engine's row stands still. An engine without the device
  encoder (no mesh, or MIMO_MM_MODALITIES without audio) falls back to the host one. (Before the fixed prefill linear
  configs of 5c112e83ddb an engine with it failed its prefill batch-trace capture at startup: ttnn's matmul
  auto-config sizes blocks from the L1 free at the call.)
* ``engine``: codes, and the engine runs the host patch encoder (``AudioCodesBackend.embed_audio_codes``).
* ``frontend``: here, after the tokenizer encoder. The item reaches the engine as ``audio_embeds`` bf16
  [n_tokens, 4096], bit for bit what the engine's host patch encoder makes from the codes, and the engine only
  uploads it: the engine's row no longer waits for the host patch encoder, the item's processing here takes ~10 %
  longer. The processor cache and the engine transfer then hold 8 KiB per token instead of 80 B of codes per 25 Hz
  frame (a 30 s clip: 1.5 MB). It needs MIMO_AUDIO_TOKENIZER=frontend: the patch encoder here takes the codes of the
  tokenizer encoder here. With the device tokenizer (the default) the codes only exist on the engine's row, so the
  combination is refused at startup (``check_audio_places``), in the API server and in the engine; before that
  check it was accepted and silently ran the engine's host patch encoder, what ``frontend`` is there to avoid.

MIMO_AUDIO_TOKENIZER=device (the default) runs the tokenizer encoder on the engine's row too (tt/audio/tokenizer.py;
the API server and the engine read it): the item then reaches the engine as ``audio_mel_bits``, the fp32 mel as int32
bits [T, 128] (512 B per 10 ms frame), and this processor runs no encoder at all, so items no longer queue behind a
long clip here. The device encoder takes ~0.1 s for 30 s of audio where the host one takes ~1.6 s; its codes differ
from the fp32 reference in ~3.5-5 % of the frames (Blackhole matmul accumulation; tests/test_audio_tokenizer_tp8.py),
which does not cost accuracy. Serving A/B on bhg2 row 1 (one row engine, the release MTP config, full LibriSpeech
test-clean through tests/evals/run_eval.sh at concurrency 8): ``frontend`` (host patch encoder) WER 1.969 %, 1
degenerate output; ``device`` with the device patch encoder (integration 96920daecad, MIMO_MM_VIT_RESERVE_GB 0.35)
WER 1.793 %, 1 degenerate output (the same item), no burst, TT_METAL_TRACE_ALLOC_TRACKING clean. Audio TTFT (104
streamed requests at concurrency 8: LibriSpeech, MMAU, 30 s and 120 s clips): p50 4.54 -> 1.08 s, p95 6.62 -> 1.85 s
(30 s clip 5.86 -> 0.57 s). An earlier device run (before the verify-trace warm-up fix f4c26ced179) had 12 degenerate
generations in two bursts of in-flight requests (WER 2.557 %). MMAU test-mini (that earlier run, host patch encoder):
61.7 % frontend, 62.3 % device. ``frontend`` keeps the bit-identical host path. A video's audio track follows the
same setting (tt/video_audio.py: ``va_audio_mel_bits`` or ``va_audio_codes``).

MIMO_AUDIO_FRONTEND_THREADS (default 8): torch threads of the encoders in the front end (vLLM runs the processor
with OMP_NUM_THREADS threads, 1 when unset).
"""

import os
import threading

import torch
from loguru import logger
from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig

from ...reference import audio as ref
from .host_backend import HostAudioBackend, check_audio_places

FRONTEND_THREADS = int(os.environ.get("MIMO_AUDIO_FRONTEND_THREADS", "8"))
PATCH_PLACES = ("device", "frontend", "engine")
PATCH_WHERE = os.environ.get("MIMO_AUDIO_PATCH", "device")
if PATCH_WHERE not in PATCH_PLACES:
    raise ValueError(f"MIMO_AUDIO_PATCH={PATCH_WHERE!r}: one of {PATCH_PLACES}")
FRONTEND_PATCH = PATCH_WHERE == "frontend"
TOKENIZER_WHERE = os.environ.get("MIMO_AUDIO_TOKENIZER", "device")
if TOKENIZER_WHERE not in ("frontend", "device"):
    raise ValueError(f"MIMO_AUDIO_TOKENIZER={TOKENIZER_WHERE!r}: frontend or device")
DEVICE_TOKENIZER = TOKENIZER_WHERE == "device"
check_audio_places(PATCH_WHERE, TOKENIZER_WHERE)
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


def audio_tokenizer(model_path, num_threads=None, patch=None):
    """This process's front-end audio encoder for ``model_path``: a HostAudioBackend with the tokenizer encoder, and
    the patch encoder with ``patch`` (default FRONTEND_PATCH; loaded on the first call that asks for it)."""
    patch = FRONTEND_PATCH if patch is None else patch
    with _tokenizers_lock:
        tok = _tokenizers.get(model_path)
        if tok is None:
            tok = HostAudioBackend(model_path, num_threads=num_threads or FRONTEND_THREADS, patch=patch)
            _tokenizers[model_path] = tok
            logger.info(f"MiMo audio front end: tokenizer encoder{' + patch encoder' * patch} loaded from {model_path}")
        elif patch and tok.patch is None:
            tok.patch = HostAudioBackend(model_path, num_threads=tok.num_threads, tokenizer=False).patch
            logger.info(f"MiMo audio front end: patch encoder loaded from {model_path}")
        return tok


def tokenize_mels(mels, model_path):
    """fp32 mels [T, 128] -> int32 [n_codes, 20] codes, one per item, each tokenized on its own (as the engine-side
    path did)."""
    tok = audio_tokenizer(model_path, patch=False)
    return [tok.codes(m).to(torch.int32) for m in mels]


def embed_mels(mels, model_path):
    """fp32 mels [T, 128] -> bf16 [audio_token_len(T), 4096] LLM embeddings, one per item (HostAudioBackend's
    ``embed_audio``: the codes of ``tokenize_mels``, then the bf16 patch encoder)."""
    tok = audio_tokenizer(model_path, patch=True)
    return [tok.embed_audio(m) for m in mels]


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

    def _apply_hf_processor_main(self, mm_items, hf_kwargs):
        out = super()._apply_hf_processor_main(mm_items, hf_kwargs)
        if "audio_features" in out:
            if bool(torch.as_tensor(out.get("is_audio_tokenized", False)).any()):
                raise ValueError("pre-tokenized audio input is not served; send the audio itself")
            path = audio_checkpoint_path(self.info.ctx.model_config)
            mels = fp32_mels(
                out.pop("audio_features"), mm_items.get("audio"), ref.MiMoAudioConfig.from_model_path(path)
            )
            if DEVICE_TOKENIZER:  # the engine's row encodes it: the fp32 mel, exactly (int32 bits: never cast)
                out["audio_mel_bits"] = [m.contiguous().view(torch.int32) for m in mels]
            elif FRONTEND_PATCH:
                out["audio_embeds"] = embed_mels(mels, path)
            else:
                out["audio_codes"] = tokenize_mels(mels, path)
        return out

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        fields.pop("audio_features", None)
        fields["audio_codes"] = MultiModalFieldConfig.batched("audio")  # fields missing from the output are skipped
        fields["audio_embeds"] = MultiModalFieldConfig.batched("audio")
        fields["audio_mel_bits"] = MultiModalFieldConfig.batched("audio")
        return fields
