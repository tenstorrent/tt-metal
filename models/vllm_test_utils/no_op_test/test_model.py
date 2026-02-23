# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

from models.vllm_test_utils.generative_base import GenerativeTestModelBase


class DummyNoOpModel(GenerativeTestModelBase):
    """
    Dummy model class which does nothing for prefill and decode forward.
    Assumes sampling on device. Used to measure the host overheads from vLLM.
    """

    def __init__(self, mesh_device, max_batch_size, vocab_size, **kwargs):
        # Accept arbitrary kwargs so the signature supports `vllm_config=...`
        # (used by vLLM interface checks) without impacting TT initialization.
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.decode_out = torch.ones(max_batch_size, 1, dtype=torch.int32)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        vocab_size = hf_config.vocab_size
        return cls(mesh_device, max_batch_size, vocab_size)

    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens")
        # Run nothing for prefill forward in this dummy model
        return torch.ones(tokens.shape[0], 1, 1, dtype=torch.int32)

    def decode_forward(self, *args, **kwargs):
        # Run nothing for decode forward in this dummy model
        tokens = kwargs.get("tokens")
        assert tokens.shape[0] == self.max_batch_size, "Batch size mismatch"
        return self.decode_out

    def allocate_kv_cache(self, *args, **kwargs):
        return None

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass
