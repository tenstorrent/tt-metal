# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio input embeddings on the host CPU: the ``embed_audio`` part of the multimodal ``EncoderBackend``.

    backend = HostAudioBackend(model_path, num_threads=8)
    embeds = backend.embed_audio(mel)   # mel: the processor's audio_features item, fp32 [T, 128]
                                        # -> bf16 [audio_token_len(T), 4096], one row per <|audio_pad|> placeholder

The tokenizer encoder (mel -> 20 RVQ codes per 40 ms) runs in fp32, so the codes are the HF fp32 module's (bf16
would change a code in ~70 % of the frames: tests/test_audio_reference_host.py), with the reference's ops except
``is_causal`` attention on the full layers, which is bit-identical (tests/test_audio_host_backend_host.py). The patch
encoder (codes -> embeddings) runs in ``patch_dtype``: bf16 by default (PCC ~0.9999 against fp32; the codes are
not affected), fp32 for embeddings identical to the reference's.

Each item is encoded on its own, as vLLM does with one audio item per encoder batch (vLLM batches every audio item
of a step into one padded batch, which can change an item's last code frame; reference/audio.py).

``torch.set_num_threads`` is process-wide: ``embed_audio`` sets ``num_threads`` for its own duration and restores
the previous value, under a lock (calls from several threads run one at a time).
"""

import os
import threading

import torch
import torch.nn.functional as F

from models.experimental.mimo_v2_6_flash.reference import audio as ref


class CausalAudioTokenizerEncoder(ref.AudioTokenizerEncoder):
    """The reference encoder with the full-attention layers on SDPA's ``is_causal`` instead of HF's additive causal
    mask: the CPU flash kernel then skips the blocks above the diagonal instead of adding -inf to them, and the
    result is bit-identical (checked on 5 s to 600 s of speech). The windowed layers keep the masked dense SDPA:
    attending each query block only to its 128-key window was 2x cheaper there but not bit-identical (one code
    frame in 7421 changed on a 300 s clip), and the whole encoder only 10-18 % faster."""

    def _attention(self, q, k, v, window, masks):
        if window <= 0 or q.shape[2] <= window + 1:  # causal, or a window that covers the whole segment
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=q.shape[-1] ** -0.5)
        return super()._attention(q, k, v, window, masks)


class HostAudioBackend:
    """``embed_audio(mel)`` / ``embed_audio_codes(codes)`` -> bf16 [n_tokens, 4096] on the host.

    Args:
        model_path: the HF checkpoint directory (``config.json``, ``model.safetensors.index.json`` and the shards
            holding ``speech_embeddings.*`` / ``audio_encoder.*``, ``audio_tokenizer/``).
        num_threads: torch intra-op threads while embedding (the serving process needs the rest of the CPU).
        patch_dtype: dtype of the patch encoder (codes -> embeddings): ``torch.bfloat16`` (default) or
            ``torch.float32`` (embeddings bit-identical to the fp32 reference before the final bf16 cast).
        fast_attention: run the tokenizer encoder's full-attention layers with ``is_causal`` (bit-identical,
            ~5 % faster); False runs the reference's masked attention everywhere.
        tokenizer, patch: which halves to load. The engine needs only the patch encoder when the processor
            tokenizes in the front end (tt/audio/processor.py); the front end only the tokenizer encoder.
        weights: ``reference.audio.load_audio_weights(model_path)`` output, to share one load between instances.

    Host memory: ~1.7 GB of weights (308M tokenizer-encoder parameters in fp32, 236M patch-encoder parameters in
    bf16). Construction took 0.1 s on bhg4 with the safetensors files in the page cache.

    Latency of ``embed_audio`` on bhg4 (EPYC 9354P, shared host; tests/perf/audio_host_latency.py), 8 threads:
    5 s 0.35 s, 30 s 1.74 s, 120 s 8.9 s, 300 s 28.4 s, 600 s 62 s; the tokenizer encoder is 90-98 % of it.
    """

    def __init__(
        self,
        model_path,
        *,
        num_threads=8,
        patch_dtype=torch.bfloat16,
        fast_attention=True,
        tokenizer=True,
        patch=True,
        weights=None,
    ):
        if weights is None:
            weights = ref.load_audio_weights(model_path, tokenizer=tokenizer, patch=patch)
        self.cfg = ref.MiMoAudioConfig.from_model_path(model_path)
        self.num_threads = num_threads
        self.encoder = self.patch = None
        if tokenizer:
            encoder_cls = CausalAudioTokenizerEncoder if fast_attention else ref.AudioTokenizerEncoder
            self.encoder = encoder_cls(weights, self.cfg, torch.float32)
        if patch:
            self.patch = ref.AudioPatchEncoder(weights, self.cfg, patch_dtype)
        self._lock = threading.Lock()

    def num_tokens(self, mel_frames):
        """Rows ``embed_audio`` returns for a [mel_frames, 128] mel: the processor's ``audio_token_lens``."""
        return ref.audio_token_len(mel_frames, self.cfg)

    @staticmethod
    def _as_mel(mel, n_mels):
        mel = torch.as_tensor(mel)
        if mel.dim() == 3 and mel.shape[0] == 1:  # vLLM's batched field can hand over [1, T, 128]
            mel = mel[0]
        if mel.dim() != 2 or mel.shape[1] != n_mels or mel.shape[0] < 1:
            raise ValueError(f"audio_features must be [T, {n_mels}], got {tuple(mel.shape)}")
        return mel.float().contiguous()

    def _as_codes(self, codes):
        codes = torch.as_tensor(codes)
        if codes.dim() == 3 and codes.shape[0] == 1:
            codes = codes[0]
        sizes = torch.tensor(self.cfg.codebook_sizes)
        if (
            codes.dim() != 2
            or codes.shape[1] != self.cfg.audio_channels
            or codes.shape[0] < 1
            or codes.is_floating_point()
            or bool((codes < 0).any())
            or bool((codes >= sizes).any())
        ):
            raise ValueError(f"audio_codes must be [T, {self.cfg.audio_channels}] RVQ codes, got {tuple(codes.shape)}")
        return codes.long()

    @torch.inference_mode()
    def codes(self, mel):
        """[T, 128] mel -> RVQ codes [n_codes, 20] (int64, 25 Hz)."""
        mel = self._as_mel(mel, self.cfg.n_mels)
        with self._lock, _threads(self.num_threads):
            return ref.tokenize_audio_batch([mel], self.encoder)[0]

    @torch.inference_mode()
    def embed_audio_codes(self, codes):
        """[n_codes, 20] RVQ codes (the TT processor's ``audio_codes`` for one item) -> bf16 [ceil(n_codes / 4), 4096]."""
        codes = self._as_codes(codes)
        with self._lock, _threads(self.num_threads):
            out = self.patch(ref.pad_and_group_codes(codes, self.cfg))
        return out.to(torch.bfloat16).contiguous()

    @torch.inference_mode()
    def embed_audio(self, mel):
        """[T, 128] mel (the processor's ``audio_features`` for one item) -> bf16 [audio_token_len(T), 4096]."""
        mel = self._as_mel(mel, self.cfg.n_mels)
        with self._lock, _threads(self.num_threads):
            codes = ref.tokenize_audio_batch([mel], self.encoder)[0]
            out = self.patch(ref.pad_and_group_codes(codes, self.cfg))
        assert out.shape[0] == ref.audio_token_len(mel.shape[0], self.cfg)
        return out.to(torch.bfloat16).contiguous()


class AudioCodesBackend:
    """An encoder backend plus ``embed_audio_codes``: ``backend``'s methods for everything else, and the patch
    encoder (a patch-only HostAudioBackend, one per process and checkpoint, loaded on the first call) for the RVQ
    codes the TT processor (tt/audio/processor.py) computes in the front end."""

    _patch = {}
    _patch_lock = threading.Lock()

    def __init__(self, backend, model_path, num_threads=None):
        self.backend, self.model_path = backend, model_path
        self.num_threads = num_threads if num_threads is not None else int(os.environ.get("MIMO_AUDIO_THREADS", "8"))

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def embed_audio_codes(self, codes):
        with AudioCodesBackend._patch_lock:
            patch = AudioCodesBackend._patch.get(self.model_path)
            if patch is None:
                patch = HostAudioBackend(self.model_path, num_threads=self.num_threads, tokenizer=False)
                AudioCodesBackend._patch[self.model_path] = patch
        return patch.embed_audio_codes(codes)


class _threads:
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.prev = torch.get_num_threads()
        if self.n and self.n != self.prev:
            torch.set_num_threads(self.n)

    def __exit__(self, *exc):
        if torch.get_num_threads() != self.prev:
            torch.set_num_threads(self.prev)
