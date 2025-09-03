# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from models.experimental.trocr.tt.trocr_decoder import TtTrOCRDecoder


class TtTrOCRDecoderWrapper(nn.Module):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config, base_address=None, state_dict=None, device=None):
        super().__init__()
        self.decoder = TtTrOCRDecoder(config, state_dict=state_dict, base_address=base_address, device=device)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
