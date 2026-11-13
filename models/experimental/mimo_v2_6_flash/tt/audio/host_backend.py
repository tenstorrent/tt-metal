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


# Key block of PyTorch's CPU flash attention (aten/src/ATen/native/cpu/FlashAttentionKernel.cpp: kvSplitSize 512).
CPU_SDPA_KV_BLOCK = 512


def check_audio_places(patch, tokenizer):
    """Refuse MIMO_AUDIO_PATCH=frontend with MIMO_AUDIO_TOKENIZER=device. The front end's patch encoder takes the
    codes of the front end's tokenizer encoder; with the device tokenizer the item leaves the front end as a mel, the
    front end's patch encoder never runs, and the engine (which builds its device patch encoder only for
    MIMO_AUDIO_PATCH=device) would run the host patch encoder on its row without a word. Called at import by the
    front end (tt/audio/processor.py) and the engine (tt/generator_vllm.py), which read the same two variables."""
    if patch == "frontend" and tokenizer == "device":
        raise ValueError(
            "MIMO_AUDIO_PATCH=frontend needs MIMO_AUDIO_TOKENIZER=frontend: the front end's patch encoder takes the "
            "front end's codes. With MIMO_AUDIO_TOKENIZER=device (the default) set MIMO_AUDIO_PATCH to device or engine"
        )


class CausalAudioTokenizerEncoder(ref.AudioTokenizerEncoder):
    """The reference encoder with cheaper attention, bit-identical hidden states (checked on 5 s to 600 s of speech,
    tests/test_audio_host_backend_host.py):

    * Full-attention layers: SDPA's ``is_causal`` instead of HF's additive causal mask, so the CPU flash kernel
      skips the key blocks above the diagonal instead of adding -inf to them.
    * Windowed layers (``windowed_blocks``): the masked SDPA per 512-row query block, on the keys from the
      512-aligned key block holding the window's start up to the query block's end. The CPU flash kernel walks the
      keys in 512-key blocks from key 0; the dropped blocks are entirely masked for these queries, where the kernel
      adds nothing (its running max, sum and output are unchanged), and the kept blocks have the same bounds as in
      the full call, so every sum is taken in the same order. Unaligned bounds are not bit-identical (attending each
      query block only to its own 128-key window changed one code frame in 7421 on a 300 s clip). 10 % faster
      than the dense masked SDPA on 65 s to 600 s clips (a segment of 6000 mel frames has 3000 queries)."""

    windowed_blocks = True

    def _attention(self, q, k, v, window, masks):
        n = q.shape[2]
        if window <= 0 or n <= window + 1:  # causal, or a window that covers the whole segment
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=q.shape[-1] ** -0.5)
        if not self.windowed_blocks or n <= CPU_SDPA_KV_BLOCK:
            return super()._attention(q, k, v, window, masks)
        key = (n, window)
        if key not in masks:
            masks[key] = ref._attn_mask(n, self.cfg.encoder_causal, window, q.dtype)
        mask, out, B = masks[key], torch.empty_like(q), CPU_SDPA_KV_BLOCK
        for q0 in range(0, n, B):
            q1, k0 = min(n, q0 + B), max(0, q0 - window) // B * B
            out[:, :, q0:q1] = F.scaled_dot_product_attention(
                q[:, :, q0:q1], k[:, :, k0:q1], v[:, :, k0:q1], attn_mask=mask[q0:q1, k0:q1], scale=q.shape[-1] ** -0.5
            )
        return out


def as_codes(codes, cfg):
    """An item's ``audio_codes`` (vLLM may hand [1, T, 20]) -> int64 [T, 20], checked against the codebook sizes."""
    codes = torch.as_tensor(codes)
    if codes.dim() == 3 and codes.shape[0] == 1:
        codes = codes[0]
    sizes = torch.tensor(cfg.codebook_sizes)
    if (
        codes.dim() != 2
        or codes.shape[1] != cfg.audio_channels
        or codes.shape[0] < 1
        or codes.is_floating_point()
        or bool((codes < 0).any())
        or bool((codes >= sizes).any())
    ):
        raise ValueError(f"audio_codes must be [T, {cfg.audio_channels}] RVQ codes, got {tuple(codes.shape)}")
    return codes.long()


class HostAudioBackend:
    """``embed_audio(mel)`` / ``embed_audio_codes(codes)`` -> bf16 [n_tokens, 4096] on the host.

    Args:
        model_path: the HF checkpoint directory (``config.json``, ``model.safetensors.index.json`` and the shards
            holding ``speech_embeddings.*`` / ``audio_encoder.*``, ``audio_tokenizer/``).
        num_threads: torch intra-op threads while embedding (the serving process needs the rest of the CPU).
        patch_dtype: dtype of the patch encoder (codes -> embeddings): ``torch.bfloat16`` (default) or
            ``torch.float32`` (embeddings bit-identical to the fp32 reference before the final bf16 cast).
        fast_attention: the tokenizer encoder's cheaper, bit-identical attention (``CausalAudioTokenizerEncoder``:
            ``is_causal`` full layers, key-block-aligned windowed layers); False runs the reference's masked
            attention everywhere.
        tokenizer, patch: which halves to load. The engine needs only the patch encoder when the processor
            tokenizes in the front end (tt/audio/processor.py); the front end only the tokenizer encoder.
        weights: ``reference.audio.load_audio_weights(model_path)`` output, to share one load between instances.

    Host memory: ~1.7 GB of weights (308M tokenizer-encoder parameters in fp32, 236M patch-encoder parameters in
    bf16). Construction took 0.1 s on bhg4 with the safetensors files in the page cache.

    Latency of ``embed_audio`` on bhg4 (EPYC 9354P, shared host; tests/perf/audio_host_latency.py), 8 threads:
    5 s 0.35 s, 30 s 1.74 s, 120 s 8.9 s, 300 s 28.4 s, 600 s 62 s; the tokenizer encoder is 90-98 % of it. Its
    fp32 matmuls run at ~100 GFLOP/s per core (MKL sgemm at the AVX-512 peak: 790 GFLOP/s at 8 threads), 70 % of
    its time; attention ~20 %. By thread count on bhg2 (same CPU, lightly shared), key-block-aligned windowed layers
    vs dense: 30 s 1.74 / 0.97 / 0.72 / 0.85 s at 8 / 16 / 24 / 32 threads (dense windowed layers 1.85 / 1.17 /
    0.85 s at 8 / 16 / 24), 120 s 8.0 / 4.8 / 3.9 / 4.1 s (dense 8.7 / 5.8 at 8 / 16).
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
        return as_codes(codes, self.cfg)

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
    """An encoder backend plus ``embed_audio_codes``: ``backend``'s methods for everything else, and for the RVQ codes
    the TT processor (tt/audio/processor.py) computes in the front end the row's device patch encoder
    (``device_patch``, tt/audio/patch.py) or, without one, the host patch encoder (a patch-only HostAudioBackend, one
    per process and checkpoint, loaded on the first call)."""

    _patch = {}
    _tokenizer = {}
    _patch_lock = threading.Lock()

    def __init__(self, backend, model_path, num_threads=None, device_patch=None, device_tokenizer=None):
        self.backend, self.model_path = backend, model_path
        self.device_patch, self.device_tokenizer = device_patch, device_tokenizer
        self.num_threads = num_threads if num_threads is not None else int(os.environ.get("MIMO_AUDIO_THREADS", "8"))

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def __repr__(self):
        patch = "host" if self.device_patch is None else type(self.device_patch).__name__
        tok = "host" if self.device_tokenizer is None else type(self.device_tokenizer).__name__
        return f"AudioCodesBackend({self.backend!r}, tokenizer={tok}, patch={patch})"

    def embed_audio(self, mel):
        """A mel item (``audio_features``, or ``audio_mel_bits`` from a front end with MIMO_AUDIO_TOKENIZER=device): the
        row's device tokenizer encoder and the host RVQ, then ``embed_audio_codes``; without a device tokenizer the
        wrapped backend's ``embed_audio`` (the host encoders)."""
        if self.device_tokenizer is None:
            return self.backend.embed_audio(mel)
        return self.embed_audio_codes(self.audio_codes(mel))

    def audio_codes(self, mel):
        """fp32 mel [T, 128] -> int64 RVQ codes [n_codes, 20] (a video's audio track, ``va_audio_mel_bits``): the row's
        device tokenizer encoder with the RVQ on the host at ``num_threads`` (vLLM leaves an engine one torch thread
        unless OMP_NUM_THREADS is set, and a 27 min track has 41K code frames), else the fp32 host tokenizer encoder
        (one per process and checkpoint, loaded on the first call; the front end's codes, bit for bit)."""
        if self.device_tokenizer is None:
            with AudioCodesBackend._patch_lock:
                host = AudioCodesBackend._tokenizer.get(self.model_path)
                if host is None:
                    host = HostAudioBackend(self.model_path, num_threads=self.num_threads, patch=False)
                    AudioCodesBackend._tokenizer[self.model_path] = host
            return host.codes(mel)
        tok = self.device_tokenizer
        hidden = tok.hidden(HostAudioBackend._as_mel(mel, tok.cfg.n_mels))
        with _threads(self.num_threads):
            return tok.quantizer.quantize(hidden).t().contiguous()

    def embed_audio_codes(self, codes):
        if self.device_patch is not None:
            return self.device_patch.embed_audio_codes(codes)
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
