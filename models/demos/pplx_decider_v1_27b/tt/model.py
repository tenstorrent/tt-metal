# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.pplx_decider_v1_27b.tt.decision import answer, decision_probabilities, options, render_input_ids
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderModelArgs

KV_BLOCK_SIZE = 64
# The mesh must be opened with at least this trace region for capture_prefill_trace.
TRACE_REGION_SIZE = 1024 * 1024 * 1024


class PplxDecider:
    def __init__(self, mesh_device, args, model, max_seq_len):
        self.mesh_device = mesh_device
        self.args = args
        self.model = model
        self.tokenizer = args.tokenizer
        self.decision_config = args.decision_config
        self.max_seq_len = max_seq_len

        # 32-aligned block count is required by the flexible SDPA.
        num_blocks = math.ceil(((max_seq_len // KV_BLOCK_SIZE) + 8) / 32) * 32
        kv_shape = (num_blocks, args.n_local_kv_heads, KV_BLOCK_SIZE, args.head_dim)
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        self._page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1)

    @classmethod
    def from_pretrained(cls, mesh_device, max_seq_len=8192, hf_model=None):
        if hf_model is not None:
            os.environ["HF_MODEL"] = hf_model
        args = PplxDeciderModelArgs(mesh_device=mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
        state_dict = args.load_state_dict()
        model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=args.weight_cache_path())
        del state_dict
        return cls(mesh_device, args, model, max_seq_len)

    def capture_prefill_trace(self):
        """Trace full 2048-token chunks and precompile all masked buckets; prompts under 2048 tokens and the tail run eagerly."""
        self.model.capture_prefill_trace_chunked(self.mesh_device, self._page_table)

    def input_ids(self, state, question):
        return render_input_ids(self.tokenizer, state, question, self.decision_config.codes)

    def readout_logits(self, input_ids):
        """Readout logits for the last prompt token, fp32 [len(token_ids)]."""
        length = len(input_ids)
        if length == 0 or length > self.max_seq_len:
            raise ValueError(f"Prompt has {length} tokens; supported range is 1..{self.max_seq_len}.")
        tokens = torch.tensor([input_ids], dtype=torch.long)
        logits_dev = self.model.prefill_traced_chunked(tokens, self._page_table, actual_len=length)
        vocab = self.args.vocab_size
        logits = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        ttnn.deallocate(logits_dev)
        # The synthetic LM head is zero outside token_ids, so the readout rows are gathered there.
        token_ids = list(self.decision_config.token_ids)
        return logits.reshape(-1, vocab)[0].float()[token_ids]

    def predict(self, state, question, temperature=None):
        if temperature is None:
            temperature = self.decision_config.temperature
        logits = self.readout_logits(self.input_ids(state, question))
        probabilities = decision_probabilities(logits, len(options(question)[0]), temperature)
        return answer(question, probabilities.tolist())
