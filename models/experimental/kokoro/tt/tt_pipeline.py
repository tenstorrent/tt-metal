# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-native pipeline: reference KPipeline G2P + TTKModel on-device inference.

Usage::

    import ttnn
    from models.experimental.kokoro.reference.model import KModel
    from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig, TTKModel, preprocess_tt_kmodel
    from models.experimental.kokoro.tt.tt_pipeline import TTPipeline

    device = ttnn.open_device(device_id=0)

    ref_model = KModel(repo_id=KokoroConfig.repo_id, disable_complex=True).eval()
    params = preprocess_tt_kmodel(ref_model, device)
    tt_model = TTKModel(device, ref_model, params, use_torch_stft_fallback=True, use_torch_phase_fallback=True)

    pipe = TTPipeline(lang_code="a", device=device, model=tt_model)
    for result in pipe("Hello from Tenstorrent.", voice="af_heart"):
        audio = result.audio          # torch float32 CPU tensor, sample_rate 24 kHz
        print(result.graphemes, result.phonemes, audio.shape)

    ttnn.close_device(device)

Host-side torch ops in the inference path
------------------------------------------
The only torch operations that run between phoneme chunks during inference are:

* ``pack[len(ps) - 1]`` — single-row index on a small CPU float32 tensor (the voice
  style pack).  Constant-time, no NN computation.
* ``ref_s.cpu().float()`` / ``.unsqueeze(0)`` — dtype/shape normalisation on that row.

All neural-network computation runs on the TT device via :class:`TTKModel`.
:meth:`TTKModel.forward` handles its own input preprocessing (token IDs, attention mask,
keep mask), so no torch NN ops surface at the pipeline level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Union

import torch

import ttnn

from ..reference.pipeline import KPipeline
from .tt_kmodel import TTKModel


class TTPipeline:
    """TT-native pipeline mirroring :class:`~models.experimental.kokoro.reference.pipeline.KPipeline`.

    Delegates all G2P, text chunking, and voice-pack management to a *quiet*
    :class:`KPipeline` (``model=False``), then routes each phoneme chunk through
    :class:`TTKModel` for on-device synthesis.

    The interface is intentionally compatible with :class:`KPipeline`: the
    returned :class:`Result` objects expose the same ``.audio``, ``.phonemes``,
    ``.graphemes``, and ``.pred_dur`` attributes.
    """

    @dataclass
    class Result:
        """Per-chunk synthesis result (mirrors :class:`KPipeline.Result`)."""

        graphemes: str
        phonemes: str
        tokens: Optional[list] = None
        output: Optional[TTKModel.Output] = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[torch.FloatTensor]:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> Optional[torch.LongTensor]:
            return None if self.output is None else self.output.pred_dur

        # -- backward-compat iteration protocol (same as KPipeline.Result) --
        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self):
            return 3

    def __init__(
        self,
        lang_code: str,
        device: ttnn.Device,
        model: TTKModel,
        repo_id: Optional[str] = None,
        *,
        trf: bool = False,
        en_callable=None,
    ) -> None:
        """
        Args:
            lang_code: Language code (``"a"`` = American English, ``"b"`` = British
                English, etc.; see :data:`KPipeline.LANG_CODES`).
            device: Open TT device handle (used for TTKModel; not used directly here).
            model: Pre-constructed :class:`TTKModel` instance.
            repo_id: HuggingFace repo ID for voice-pack downloads.  Defaults to
                ``"hexgrad/Kokoro-82M"`` (same default as :class:`KPipeline`).
            trf: Use transformer-based G2P backend (passed to :class:`KPipeline`).
            en_callable: Custom English G2P callable (passed to :class:`KPipeline`).
        """
        self._g2p = KPipeline(
            lang_code=lang_code,
            repo_id=repo_id,
            model=False,
            trf=trf,
            en_callable=en_callable,
        )
        self.model = model
        self.device = device
        self.lang_code = self._g2p.lang_code

    # ------------------------------------------------------------------
    # Voice management (delegates to the internal quiet KPipeline)
    # ------------------------------------------------------------------

    def load_voice(
        self,
        voice: Union[str, torch.FloatTensor],
        delimiter: str = ",",
    ) -> torch.FloatTensor:
        """Load (and optionally blend) a voice pack.  See :meth:`KPipeline.load_voice`."""
        return self._g2p.load_voice(voice, delimiter=delimiter)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @staticmethod
    def infer(
        model: TTKModel,
        ps: str,
        ref_s: torch.FloatTensor,
        speed: Union[float, Callable[[int], float]] = 1.0,
    ) -> TTKModel.Output:
        """Run one phoneme chunk through TTKModel.

        Args:
            model: The TT model.
            ps: Phoneme string for this chunk.
            ref_s: ``[1, style_dim]`` style vector (CPU float32).  Selected from the
                voice pack by the caller as ``pack[len(ps) - 1]``.
            speed: Speech speed multiplier, or a callable that receives the phoneme
                count and returns a float.

        Returns:
            :class:`TTKModel.Output` with ``.audio`` as a CPU float32 tensor.
        """
        if callable(speed):
            speed = speed(len(ps))
        return model(phonemes=ps, ref_s=ref_s, speed=float(speed))

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable[[int], float]] = 1.0,
        split_pattern: Optional[str] = r"\n+",
    ) -> Generator["TTPipeline.Result", None, None]:
        """Synthesise ``text`` with ``voice`` and yield per-chunk results.

        Args:
            text: Input text (string or list of strings).
            voice: Voice name (e.g. ``"af_heart"``) or path to a ``.pt`` voice file.
            speed: Speech speed multiplier (or callable receiving phoneme count).
            split_pattern: Regex used to split text into segments (default ``r"\\n+"``).

        Yields:
            :class:`TTPipeline.Result` for each synthesised phoneme chunk.
        """
        if voice is None:
            raise ValueError('Specify a voice: pipeline(text, voice="af_heart")')

        pack = self._g2p.load_voice(voice)

        # Iterate over G2P chunks from the quiet pipeline.  Each chunk has
        # .phonemes populated and .output = None (no KModel inference).
        for chunk in self._g2p(text, voice=None, speed=speed, split_pattern=split_pattern):
            ps = chunk.phonemes
            if not ps:
                continue

            # --- voice pack style selection (only torch op in inference path) ---
            # pack shape: [N_styles, style_dim]; select row for this phoneme count.
            ref_s = pack[len(ps) - 1].cpu().float()
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)

            output = self.infer(self.model, ps, ref_s, speed)

            yield self.Result(
                graphemes=chunk.graphemes,
                phonemes=chunk.phonemes,
                tokens=chunk.tokens,
                output=output,
                text_index=getattr(chunk, "text_index", None),
            )
