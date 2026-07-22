# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.vllm_test_utils.generative_base import GenerativeTestModelBase


class DummyNoOpModel(GenerativeTestModelBase):
    """
    Dummy model class which does nothing for prefill and decode forward.
    Returns zero logits for host-side sampling and zero token IDs for
    device-side sampling. Used to measure the host overheads from vLLM.
    """

    model_capabilities = {
        "supports_sample_on_device": True,
    }

    def __init__(self, mesh_device, max_batch_size, vocab_size, **kwargs):
        # Accept arbitrary kwargs so the signature supports `vllm_config=...`
        # (used by vLLM interface checks) without impacting TT initialization.
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.decode_out = torch.zeros(max_batch_size, 1, vocab_size, dtype=torch.float32)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        vocab_size = hf_config.vocab_size
        return cls(mesh_device, max_batch_size, vocab_size)

    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens")
        if kwargs.get("sampling_params") is not None:
            return torch.zeros(tokens.shape[0], dtype=torch.int64, device=tokens.device)
        return torch.zeros(tokens.shape[0], 1, self.vocab_size, dtype=torch.float32)

    def decode_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens")
        assert tokens.shape[0] == self.max_batch_size, "Batch size mismatch"
        if kwargs.get("sampling_params") is not None:
            return torch.zeros(tokens.shape[0], dtype=torch.int64, device=tokens.device)
        return self.decode_out
