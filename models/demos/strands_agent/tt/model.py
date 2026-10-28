# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.strands_agent.tt.answers import build_answer
from models.demos.strands_agent.tt.head import load_config, load_head
from models.demos.strands_agent.tt.model_config import StrandsDeciderModelArgs
from models.demos.strands_agent.tt.prompting import EncodedQuestion, encode_request
from models.tt_transformers.tt.common import Mode

KV_BLOCK_SIZE = 64


class StrandsDecider:
    KV_BLOCK_SIZE = KV_BLOCK_SIZE

    def __init__(self, mesh_device, args, model):
        assert model.num_devices == 1, "strands_agent supports a single chip (1x1 mesh) only"
        self.mesh_device = mesh_device
        self.args = args
        self.model = model
        self.max_seq_len = args.max_seq_len

        # n_local_kv_heads is only set for TP meshes; on one chip all KV heads are local.
        # 32-aligned block count is required by the flexible SDPA.
        num_blocks = math.ceil(((self.max_seq_len // KV_BLOCK_SIZE) + 8) / 32) * 32
        kv_shape = (num_blocks, args.n_kv_heads, KV_BLOCK_SIZE, args.head_dim)
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        self.page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1)

        self.tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
        self.config = load_config(args.adapter_dir)
        self.head = load_head(args.adapter_dir, args.dim)

    @classmethod
    def from_pretrained(cls, mesh_device, max_seq_len=4096):
        args = StrandsDeciderModelArgs(mesh_device=mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
        state_dict = args.load_state_dict()
        model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=args.weight_cache_path())
        del state_dict
        return cls(mesh_device, args, model)

    def hidden_rows(self, input_ids: list[int], positions: list[int]) -> torch.Tensor:
        """Post-final-norm hidden rows of the given token positions, fp32 [len(positions), dim].

        One masked-bucket prefill call: lengths above 2048 round up to the next multiple of 128."""
        length = len(input_ids)
        if length == 0 or length > self.max_seq_len:
            raise ValueError(f"Prompt has {length} tokens; supported range is 1..{self.max_seq_len}.")
        assert all(0 <= p < length for p in positions), "position out of range"
        tokens = torch.tensor([input_ids], dtype=torch.long)
        _, hidden = self.model.prefill_masked_bucket(tokens, self.page_table, actual_len=length, return_hidden=True)
        bucket = hidden.shape[-2]
        count = len(positions)
        padded = -(-count // 32) * 32
        sel = torch.zeros(1, padded, bucket, dtype=torch.float32)
        for i, p in enumerate(positions):
            sel[0, i, p] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        rows = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        ttnn.deallocate(hidden)
        rows = ttnn.to_memory_config(rows, ttnn.DRAM_MEMORY_CONFIG)
        normed = self.model.norm(rows, mode=Mode.PREFILL)
        out = ttnn.to_torch(normed).reshape(padded, -1)[:count].float()
        if normed is not rows:
            ttnn.deallocate(normed)
        ttnn.deallocate(rows)
        return out

    def question_logits(self, q: EncodedQuestion):
        rows = self.hidden_rows(q.input_ids, [q.query_position, *q.option_positions])
        with torch.no_grad():
            logits = self.head(rows[:1], rows[None, 1:])[0]
        return logits, rows

    def evaluate(self, state, questions: dict) -> dict:
        encoded = encode_request(self.tokenizer, state, questions, self.config.max_length)
        answers, input_tokens = {}, 0
        for q in encoded:
            logits, _ = self.question_logits(q)
            answers[q.name] = build_answer(
                q, logits, self.config.temperature_by_kind, self.config.temperature, self.config.ordinal_smoothing
            )
            input_tokens += len(q.input_ids)
        return {"answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": len(encoded)}}
