# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: MIT
"""Runtime facade for the CLI demo and the CPU baseline benchmark.

build_runtime() loads the config and canonical weights once and returns an
Esm2Runtime exposing one uniform surface:

    embed(input_ids, attention_mask) -> {"logits": [B,L,V], "hidden": [B,L,H]}

over either backend:

    device="cpu" -> reference_layers.Esm2Model (PyTorch FP32 twin)
    device="tt"  -> ttnn_backend.TtnnEsm2 (TTNN device backend)

The demo and baseline previously called ``TtnnEsm2.build(...)`` as if it were
a class factory and then an ``embed()`` method that no class in this package
defines; both flows failed with TypeError before any inference. Going through
this facade instantiates the appropriate backend and calls its supported
forward interface instead.
"""

from __future__ import annotations

import numpy as np
import torch


class Esm2Runtime:
    """Uniform embed() surface over the CPU reference or the TTNN backend."""

    def __init__(self, model, device_kind: str):
        self._model = model
        self.device_kind = device_kind

    @property
    def config(self):
        return self._model.config

    def embed(self, input_ids, attention_mask) -> dict:
        """input_ids/attention_mask: [B,L] int64 (numpy or torch).

        Returns {"logits": [B,L,V] fp32, "hidden": [B,L,H] fp32} numpy.
        """
        if self.device_kind == "tt":
            return self._model.forward(input_ids, attention_mask)  # already a numpy dict
        ids = torch.as_tensor(np.asarray(input_ids)).long()
        am = torch.as_tensor(np.asarray(attention_mask)).long()
        if ids.shape != am.shape:
            raise ValueError(f"input_ids {tuple(ids.shape)} != attention_mask {tuple(am.shape)}")
        with torch.no_grad():
            logits, hidden = self._model(ids, am)
        return {"logits": logits.numpy(), "hidden": hidden.numpy()}

    def close(self):
        close = getattr(self._model, "close", None)
        if close is not None:
            close()


def build_runtime(checkpoint: str, device: str = "cpu", precision: str = "fp32", device_id: int = 0) -> Esm2Runtime:
    """Load config + weights from `checkpoint` and build the requested backend."""
    if device not in ("cpu", "tt"):
        raise ValueError(f"device must be 'cpu' or 'tt', got {device!r}")
    if device == "cpu" and precision != "fp32":
        raise ValueError(f"the CPU reference runs FP32 only, got precision {precision!r}")

    from models.experimental.esm2.tt.esm2.config import Esm2TTConfig
    from models.experimental.esm2.tt.esm2.loader import load_canonical_weights

    config = Esm2TTConfig.from_json_file(str(checkpoint) + "/config.json")
    weights = load_canonical_weights(str(checkpoint), config)

    if device == "cpu":
        from models.experimental.esm2.tt.esm2.reference_layers import Esm2Model

        model = Esm2Model(config, weights)
        model.eval()
        return Esm2Runtime(model, "cpu")

    from models.experimental.esm2.tt.esm2.ttnn_backend import TtnnEsm2

    model = TtnnEsm2(config, weights, device=None, precision=precision, device_id=device_id).build()
    return Esm2Runtime(model, "tt")
