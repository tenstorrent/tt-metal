# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.tt.common import TorchModuleFallback


class TtnnGraniteTTMPatching:
    """Wraps the TinyTimeMixerPatchify (backbone.patching) layer.

    ``TinyTimeMixerPatchify.forward`` uses ``torch.Tensor.unfold`` which has no
    direct TTNN equivalent.  For Stage 1 the module is therefore always executed
    via ``TorchModuleFallback``.

    ``TinyTimeMixerPatchify.forward`` returns a plain ``torch.Tensor`` (not a
    namedtuple), so ``output_selector=None`` is passed to prevent
    ``TorchModuleFallback`` from calling ``extract_prediction_tensor`` on the
    result.
    """

    def __init__(self, *, parameters=None, config=None, torch_module=None):
        # parameters / config are accepted for API symmetry but are unused at
        # this stage because there is no TTNN patching implementation yet.
        self._fallback = TorchModuleFallback(torch_module) if torch_module is not None else None

    def __call__(self, history, *, device=None, **kwargs):
        if self._fallback is not None:
            # output_selector=None: the output is already a plain tensor, skip
            # the namedtuple extraction path inside TorchModuleFallback.
            return self._fallback(history, device=device, output_selector=None, **kwargs)
        return history
