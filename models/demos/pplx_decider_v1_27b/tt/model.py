# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.pplx_decider_v1_27b.tt.bucket_trace import BucketPrefillTraces, bucket_sizes_for, max_seq_len_for
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
        self._two_d = tpc.is_2d_mesh(mesh_device)
        self.max_seq_len = max_seq_len_for(max_seq_len, self._two_d)
        if self.max_seq_len < max_seq_len:
            logger.info(
                f"2D mesh: max_seq_len capped at {self.max_seq_len}, the largest prefill bucket (no chunked prefill)"
            )
        self.bucket_traces = None

        # 32-aligned block count is required by the flexible SDPA.
        num_blocks = math.ceil(((self.max_seq_len // KV_BLOCK_SIZE) + 8) / 32) * 32
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
        """Trace full 2048-token chunks and precompile all masked buckets; prompts under 2048 tokens and the tail run eagerly.

        Unless PPLX_BUCKET_TRACE=0, also one fused prefill + readout trace per length bucket (bucket_trace.py), which
        serves every prompt up to the largest bucket; the chunk trace then only serves longer ones."""
        if os.environ.get("PPLX_BUCKET_TRACE", "1") == "0":
            if self._two_d:
                raise NotImplementedError("PPLX_BUCKET_TRACE=0 leaves only the chunk trace, which is 1xN-only")
            self.model.capture_prefill_trace_chunked(self.mesh_device, self._page_table)
            return
        # Nothing may be allocated or compiled once a trace is parked: prepare everything, then capture.
        # Only the buckets this max_seq_len reaches and the KV cache holds (page table length x block size).
        capacity = self._page_table.shape[-1] * KV_BLOCK_SIZE
        self.bucket_traces = BucketPrefillTraces(bucket_sizes_for(self.max_seq_len, capacity))
        self.bucket_traces.prepare(self.model, self.mesh_device, self._page_table)
        if self._two_d:
            # The chunk path's LM head and norms are not 2D-ready: bucket traces only.
            logger.info("2D mesh: capturing only the bucket prefill traces (the chunk trace path is 1xN-only)")
        else:
            self.model.prepare_prefill_trace_chunked(self.mesh_device, self._page_table)
            self.model.record_prefill_trace_chunked(self.mesh_device)
        self.bucket_traces.record(self.model, self.mesh_device)

    def input_ids(self, state, question):
        return render_input_ids(self.tokenizer, state, question, self.decision_config.codes)

    def readout_logits(self, input_ids, use_bucket_trace=True):
        """Readout logits for the last prompt token, fp32 [len(token_ids)]."""
        length = len(input_ids)
        if length == 0 or length > self.max_seq_len:
            where = " on a 2D mesh (no chunked prefill)" if self._two_d else ""
            raise ValueError(f"Prompt has {length} tokens; supported range is 1..{self.max_seq_len}{where}.")
        if use_bucket_trace and self.bucket_traces is not None:
            logits = self.bucket_traces.run(self.model, input_ids)
            if logits is not None:
                return logits
        if self._two_d:
            raise NotImplementedError(
                "a 2D mesh prefills only through the bucket traces (the chunk path is 1xN-only): "
                "call capture_prefill_trace() first, with PPLX_BUCKET_TRACE unset"
            )
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
