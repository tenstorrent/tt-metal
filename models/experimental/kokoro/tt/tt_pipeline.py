"""
TTNN pipeline wrapper — thin layer over KPipeline that substitutes
the reference KModel with TTKModel.

The G2P, chunking, and voice loading logic is unchanged from KPipeline.
Only the model inference call uses TTNN.
"""

from typing import Callable, Generator, Optional, Union

import torch

from models.experimental.kokoro.reference.pipeline import KPipeline

from .kokoro_ttnn_model import KokoroTTNNModel


class TTKPipeline:
    """
    Language-aware pipeline that runs inference on the TTNN Kokoro model.

    Reuses all KPipeline utilities (G2P, chunking, voice loading) and
    only swaps the model.forward() call to use TTKModel.
    """

    def __init__(
        self,
        lang_code: str,
        ttnn_model: KokoroTTNNModel,
        repo_id: str = "hexgrad/Kokoro-82M",
        trf: bool = False,
        en_callable: Optional[Callable] = None,
    ):
        # Build a "quiet" KPipeline for G2P/chunking only (model=False)
        self._pipe = KPipeline(
            lang_code=lang_code,
            repo_id=repo_id,
            model=False,
            trf=trf,
            en_callable=en_callable,
        )
        self._ttnn_model = ttnn_model
        self._pipe.model = self  # so KPipeline.infer uses self.forward

    @property
    def voices(self):
        return self._pipe.voices

    def load_voice(self, voice, delimiter=","):
        return self._pipe.load_voice(voice, delimiter)

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
    ):
        return self._ttnn_model(phonemes, ref_s, speed=speed, return_output=return_output)

    # Make the pipeline callable with the same API as KPipeline
    def __call__(
        self,
        text,
        voice: Optional[str] = None,
        speed: Union[float, Callable] = 1,
        split_pattern: Optional[str] = r"\n+",
    ) -> Generator:
        return self._pipe(
            text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
            model=self,
        )
