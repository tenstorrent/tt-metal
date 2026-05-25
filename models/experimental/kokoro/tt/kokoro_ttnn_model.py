"""
Top-level TTNN Kokoro model class.

Provides a clean entry point:
  model = KokoroTTNNModel.from_pretrained(repo_id, device)
  audio = model(phonemes, ref_s)

This wraps TTKModel and handles device open/close if caller passes device_id.
"""

from typing import Optional

import torch
import ttnn

from .tt_model import TTKModel


class KokoroTTNNModel:
    """
    Top-level wrapper for the TTNN Kokoro 82M TTS model.

    Usage:
        device = ttnn.open_device(device_id=0)
        model  = KokoroTTNNModel.from_pretrained("hexgrad/Kokoro-82M", device)
        audio  = model(phonemes, ref_s)
        ttnn.close_device(device)
    """

    def __init__(self, tt_kmodel: TTKModel, device):
        self._model = tt_kmodel
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "hexgrad/Kokoro-82M",
        device=None,
        device_id: int = 0,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        disable_complex: bool = False,
    ) -> "KokoroTTNNModel":
        """
        Load from HuggingFace (or local path) and build TTNN model.

        Args:
            repo_id:        HuggingFace repo id (default hexgrad/Kokoro-82M).
            device:         Open ttnn device. If None, opens device_id.
            device_id:      TT device id to open if device is None.
            model_path:     Optional local path to .pth weights.
            config_path:    Optional local path to config.json.
            disable_complex: Use CustomSTFT instead of TorchSTFT.
        """
        import sys, os

        # Add reference to path so we can import KModel
        ref_dir = os.path.join(os.path.dirname(__file__), "..")
        if ref_dir not in sys.path:
            sys.path.insert(0, ref_dir)

        from models.experimental.kokoro.reference.model import KModel

        own_device = device is None
        if own_device:
            device = ttnn.open_device(device_id=device_id)

        kwargs = {}
        if model_path:
            kwargs["model"] = model_path
        if config_path:
            kwargs["config"] = config_path

        kmodel = KModel(
            repo_id=repo_id,
            disable_complex=disable_complex,
            **kwargs,
        ).eval()

        tt_kmodel = TTKModel.from_kmodel(kmodel, device)

        instance = cls(tt_kmodel, device)
        instance._own_device = own_device
        return instance

    @classmethod
    def from_kmodel(cls, kmodel, device) -> "KokoroTTNNModel":
        """Build from an already-loaded reference KModel."""
        tt_kmodel = TTKModel.from_kmodel(kmodel, device)
        instance = cls(tt_kmodel, device)
        instance._own_device = False
        return instance

    def __call__(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_output: bool = False,
    ):
        return self._model(phonemes, ref_s, speed=speed, return_output=return_output)

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ):
        return self._model.forward_with_tokens(input_ids, ref_s, speed)

    @property
    def vocab(self):
        return self._model.vocab

    @property
    def context_length(self):
        return self._model.context_length

    def close(self):
        """Close the TT device if this object opened it."""
        if getattr(self, "_own_device", False):
            ttnn.close_device(self.device)
