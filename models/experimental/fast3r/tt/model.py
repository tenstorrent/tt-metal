"""End-to-end tt-nn Fast3R: CPU patch_embed + device encoder trunk + device decoder trunk.

DPT head remains a follow-up: it adds many Conv2d/upsample ops that we're deferring
until the transformer trunks are stable.
"""
from __future__ import annotations

import torch
import ttnn

from models.experimental.fast3r.reference.model import Fast3RConfig
from models.experimental.fast3r.tt.decoder import TtDecoder
from models.experimental.fast3r.tt.encoder import TtEncoderBlocks


class TtFast3RTrunk:
    """Full encoder + decoder on device. Patch_embed remains on CPU for this iteration."""

    def __init__(self, device, cfg: Fast3RConfig, weights_path: str):
        self.cfg = cfg
        self.device = device
        self.encoder = TtEncoderBlocks(device, cfg, weights_path)
        self.decoder = TtDecoder(device, cfg, weights_path)

    def __call__(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        enc = self.encoder(tokens)
        dec = self.decoder(enc)
        enc.deallocate(True)
        return dec
