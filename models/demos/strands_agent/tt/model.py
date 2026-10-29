# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.decode import recurrent_forward
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.strands_agent.tt.answers import build_answer
from models.demos.strands_agent.tt.head import load_config, load_head
from models.demos.strands_agent.tt.model_config import StrandsDeciderModelArgs
from models.demos.strands_agent.tt.prompting import EncodedQuestion, encode_request
from models.tt_transformers.tt.common import Mode

MAX_BATCH = 32  # prompts per forward (upstream max_batch)
# Padded tokens per forward (batch * bucket). Per-prompt cost stops falling near 4-8k tokens; larger forwards
# run slightly slower.
MAX_BATCH_TOKENS = 8192
# One forward on one chip costs about FWD_FIXED_MS + TOKEN_MS * batch * bucket (measured, S 128-4096, B 1-32).
FWD_FIXED_MS = 11.0
TOKEN_MS = 0.041


def plan_batches(lengths, max_batch=MAX_BATCH, max_tokens=MAX_BATCH_TOKENS) -> list[list[int]]:
    """Group prompt indices into forwards at the lowest estimated cost.

    Prompts are sorted by length. Each group is a run of that order padded to the bucket of its longest prompt,
    with len(group) <= max_batch and len(group) * bucket <= max_tokens (one prompt always fits)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    buckets = [Qwen36Model._mask_bucket_for(lengths[i]) for i in order]
    best = [0.0] + [math.inf] * len(order)
    start = [0] * (len(order) + 1)
    for j in range(1, len(order) + 1):
        for size in range(1, min(max_batch, j) + 1):
            if size > 1 and size * buckets[j - 1] > max_tokens:
                break
            cost = best[j - size] + FWD_FIXED_MS + TOKEN_MS * size * buckets[j - 1]
            if cost < best[j]:
                best[j], start[j] = cost, j - size
    groups, j = [], len(order)
    while j > 0:
        groups.append(order[start[j] : j])
        j = start[j]
    return groups[::-1]


class StrandsDecider:
    def __init__(self, mesh_device, args, model):
        assert model.num_devices == 1, "strands_agent supports a single chip (1x1 mesh) only"
        self.mesh_device = mesh_device
        self.args = args
        self.model = model
        self.max_seq_len = args.max_seq_len
        # Prefill only, so no KV cache: attention runs a plain causal SDPA over the batch.
        self.max_batch = MAX_BATCH
        self.max_batch_tokens = MAX_BATCH_TOKENS

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

    def hidden_rows_batch(self, seqs: list[list[int]], positions: list[list[int]]) -> list[torch.Tensor]:
        """Post-final-norm hidden rows of the given token positions of each prompt, fp32 [len(positions), dim].

        Prompts run in the forwards that plan_batches picks; each forward pads to the bucket of its longest prompt
        (128..2048, then multiples of 128)."""
        for ids, pos in zip(seqs, positions, strict=True):
            if not 1 <= len(ids) <= self.max_seq_len:
                raise ValueError(f"Prompt has {len(ids)} tokens; supported range is 1..{self.max_seq_len}.")
            assert all(0 <= p < len(ids) for p in pos), "position out of range"
        out = [None] * len(seqs)
        for group in plan_batches([len(ids) for ids in seqs], self.max_batch, self.max_batch_tokens):
            rows = self._forward_batch([seqs[i] for i in group], [positions[i] for i in group])
            for i, r in zip(group, rows):
                out[i] = r
        return out

    def _forward_batch(self, seqs, positions):
        """One forward over len(seqs) prompts, right-padded to one bucket S.

        Row-wise ops (embedding, norms, MLP) see one [1, B*S, dim] stream; attention and GDN see [B, S, dim].
        Attention, GDN and the conv are causal, so the padding never reaches a real row."""
        model = self.model
        B = len(seqs)
        S = Qwen36Model._mask_bucket_for(max(len(ids) for ids in seqs))
        dim = self.args.dim
        tokens = torch.zeros(B, S, dtype=torch.int32)
        for b, ids in enumerate(seqs):
            tokens[b, : len(ids)] = torch.tensor(ids, dtype=torch.int32)

        model._build_request_rope(tokens[:1, : len(seqs[0])], None)  # text only: plain 1D RoPE
        cos, sin = model.rope.get_prefill_rot_mats(0, S)
        tok = ttnn.from_torch(
            tokens.reshape(1, B * S), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )
        x = ttnn.to_memory_config(model.embd(tok), ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        valid = [len(ids) for ids in seqs]
        for layer in model.layers:
            # The single-chip prefill branch of Qwen36DecoderLayer.forward, with the batch folded into rows.
            attn_in = layer.attention_norm(x, mode=Mode.PREFILL, norm_config=None)
            attn_in = ttnn.reshape(attn_in, [B, S, dim])
            if layer.is_full_attention:
                # No chunk page table: plain causal SDPA over [B, H, S, D]; no KV cache write.
                attn_out = layer.attention.forward(attn_in, cos, sin)
                layer.attention.reset_cache()
            else:
                attn_out = recurrent_forward(
                    layer.attention,
                    attn_in,
                    mode="chunk",
                    chunk_size=layer.attention.long_prefill_chunk_size,
                    valid_len=valid,
                    stateless=True,
                )
            ttnn.deallocate(attn_in)
            h = ttnn.add(x, ttnn.reshape(attn_out, [1, B * S, dim]))
            ttnn.deallocate(attn_out)
            ttnn.deallocate(x)
            ff_in = layer.ffn_norm(h, mode=Mode.PREFILL, norm_config=None)
            ff_out = layer.feed_forward.forward(ff_in, mode="prefill")
            ttnn.deallocate(ff_in)
            x = ttnn.add(h, ff_out)
            ttnn.deallocate(h)
            ttnn.deallocate(ff_out)

        # Per-prompt one-hot row select: [B, P, S] @ [B, S, dim] -> [B, P, dim], then the final norm.
        count = max(len(pos) for pos in positions)
        padded = -(-count // 32) * 32
        sel = torch.zeros(B, padded, S, dtype=torch.float32)
        for b, pos in enumerate(positions):
            for i, p in enumerate(pos):
                sel[b, i, p] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        hidden = ttnn.reshape(x, [B, S, dim])
        rows = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        ttnn.deallocate(x)
        rows = ttnn.to_memory_config(ttnn.reshape(rows, [1, B * padded, dim]), ttnn.DRAM_MEMORY_CONFIG)
        normed = model.norm(rows, mode=Mode.PREFILL)
        out = ttnn.to_torch(normed).reshape(B, padded, -1).float()
        if normed is not rows:
            ttnn.deallocate(normed)
        ttnn.deallocate(rows)
        return [out[b, : len(pos)] for b, pos in enumerate(positions)]

    def question_logits_batch(self, qs: list[EncodedQuestion]):
        """(pointer logits, hidden rows) of each question; questions may come from different requests."""
        all_rows = self.hidden_rows_batch(
            [q.input_ids for q in qs], [[q.query_position, *q.option_positions] for q in qs]
        )
        out = []
        with torch.no_grad():
            for rows in all_rows:
                out.append((self.head(rows[:1], rows[None, 1:])[0], rows))
        return out

    def evaluate_batch(self, requests: list[tuple[str, dict]]) -> list[dict]:
        """evaluate() for several (state, questions) requests; all their questions share the batched forwards."""
        encoded = [
            encode_request(self.tokenizer, state, questions, self.config.max_length) for state, questions in requests
        ]
        flat = [q for qs in encoded for q in qs]
        logits = iter(self.question_logits_batch(flat))
        results = []
        for qs in encoded:
            answers = {}
            for q in qs:
                answers[q.name] = build_answer(
                    q,
                    next(logits)[0],
                    self.config.temperature_by_kind,
                    self.config.temperature,
                    self.config.ordinal_smoothing,
                )
            results.append(
                {
                    "answers": answers,
                    "usage": {"input_tokens": sum(len(q.input_ids) for q in qs), "output_tokens": len(qs)},
                }
            )
        return results

    def evaluate(self, state, questions: dict) -> dict:
        return self.evaluate_batch([(state, questions)])[0]
