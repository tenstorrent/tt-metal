# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from transformers import AutoTokenizer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.blackhole.qwen36.tt.attention.stateless_prefill import stateless_attention_prefill
from models.demos.blackhole.qwen36.tt.gdn.decode import recurrent_forward
from models.demos.blackhole.qwen36.tt.gdn.stateless_prefill import stateless_prefill_supported
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
# Forwards of these shapes run as traces: bucket S, rows padded up to one of TRACE_BATCHES, and at most TRACED_P
# select positions (query + options). Everything else runs eagerly.
TRACED_S = (128, 256, 512, 1024, 2048)
TRACE_BATCHES = (1, 2, 4, 8, 16, 32)
TRACED_P = 64


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
    def __init__(self, mesh_device, args, model, trace=True):
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

        self.trace = trace
        self._compiled = set()  # shape keys that have run eagerly
        self._traces = {}  # shape key -> (trace id, output tensor)
        self._trace_inputs = {}  # shape key -> persistent (tokens, select) device tensors
        self.trace_region_use = {}  # shape key -> trace region bytes
        if trace:
            for layer in model.layers:
                if not layer.is_full_attention:
                    assert stateless_prefill_supported(
                        layer.attention, "chunk", True
                    ), "traced forwards need the stateless GDN prefill (no valid_len masks)"
        # cos/sin of the traced buckets live for the whole run: a trace reads them and none may be allocated later.
        self._rope_cache = {}
        model._build_request_rope(torch.zeros(1, 1, dtype=torch.int32), None)
        for S in TRACED_S:
            self._rope_cache[S] = model.rope.get_prefill_rot_mats(0, S)

    @classmethod
    def from_pretrained(cls, mesh_device, max_seq_len=4096, trace=True):
        args = StrandsDeciderModelArgs(mesh_device=mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
        state_dict = args.load_state_dict()
        model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=args.weight_cache_path())
        del state_dict
        decider = cls(mesh_device, args, model, trace=trace)
        # One small forward builds the first-call caches (stateless GDN constants, fused attention weights), so
        # nothing persistent is allocated once a trace exists.
        decider.hidden_rows_batch([[1] * 128], [[127]])
        return decider

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

    def _rope(self, S):
        """cos/sin [1, S, head_dim] for plain 1D RoPE: the persistent ones for traced S, else built per call."""
        if S in self._rope_cache:
            return self._rope_cache[S]
        self.model._build_request_rope(torch.zeros(1, 1, dtype=torch.int32), None)  # text only: plain 1D RoPE
        return self.model.rope.get_prefill_rot_mats(0, S)

    def _shape(self, seqs, positions):
        """(B, S, P, traced): rows, bucket, select rows and whether the forward may run as a trace.

        Traced forwards pad B up to a power of two and P to TRACED_P, so a few shapes cover every group."""
        B = len(seqs)
        S = Qwen36Model._mask_bucket_for(max(len(ids) for ids in seqs))
        count = max(len(pos) for pos in positions)
        if self.trace and S in TRACED_S and count <= TRACED_P:
            Bp = next((b for b in TRACE_BATCHES if b >= B), None)
            if Bp is not None and Bp * S <= self.max_batch_tokens:
                return Bp, S, TRACED_P, True
        return B, S, -(-count // 32) * 32, False

    def _forward_batch(self, seqs, positions):
        """One forward over len(seqs) prompts, right-padded to one bucket S.

        Row-wise ops (embedding, norms, MLP) see one [1, B*S, dim] stream; attention and GDN see [B, S, dim].
        Attention, GDN and the conv are causal, so the padding never reaches a real row.

        A shape that has not run yet releases every trace and runs eagerly (it compiles its programs); a traced
        shape that has run is captured on its next call and replayed after that."""
        B, S, P, traced = self._shape(seqs, positions)
        tokens = torch.zeros(B, S, dtype=torch.int32)
        for b, ids in enumerate(seqs):
            tokens[b, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        sel = torch.zeros(B, P, S, dtype=torch.float32)
        for b, pos in enumerate(positions):
            for i, p in enumerate(pos):
                sel[b, i, p] = 1.0
        key = (B, S, P, traced)
        if key not in self._compiled:
            self.release_traces()
            valid = [len(ids) for ids in seqs] + [S] * (B - len(seqs))
            out = self._run_eager(tokens, sel, valid)
            self._compiled.add(key)
        elif traced:
            out = self._run_traced(key, tokens, sel)
        else:
            out = self._run_eager(tokens, sel, [len(ids) for ids in seqs])
        return [out[b, : len(pos)] for b, pos in enumerate(positions)]

    def _run_eager(self, tokens, sel, valid):
        B, S = tokens.shape
        tok = ttnn.from_torch(
            tokens.reshape(1, B * S), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )
        sel_tt = ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        cos, sin = self._rope(S)
        normed = self._body(tok, sel_tt, cos, sin, B, S, sel.shape[1], valid)
        ttnn.deallocate(tok)
        ttnn.deallocate(sel_tt)
        out = ttnn.to_torch(normed).reshape(B, sel.shape[1], -1).float()
        ttnn.deallocate(normed)
        return out

    def _body(self, tok, sel_tt, cos, sin, B, S, P, valid):
        """Device part of a forward: embedding, all layers, one-hot row select, final norm -> [1, B*P, dim].

        Reads only device tensors (tok, sel_tt, cos, sin), so a trace can capture it. Does not free them."""
        model = self.model
        dim = self.args.dim
        x = ttnn.to_memory_config(model.embd(tok), ttnn.DRAM_MEMORY_CONFIG)
        for layer in model.layers:
            # The single-chip prefill branch of Qwen36DecoderLayer.forward, with the batch folded into rows.
            attn_in = layer.attention_norm(x, mode=Mode.PREFILL, norm_config=None)
            attn_in = ttnn.reshape(attn_in, [B, S, dim])
            if layer.is_full_attention:
                # Stateless: one fused q|k|v|gate projection, heads built straight to [B, H, S, D], plain causal
                # SDPA; no KV cache is written, so there is nothing to reset.
                attn_out = stateless_attention_prefill(layer.attention, attn_in, cos, sin)
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
        assert x.dtype == sel_tt.dtype, f"hidden is {x.dtype}, one-hot select is {sel_tt.dtype}"
        rows = ttnn.matmul(sel_tt, ttnn.reshape(x, [B, S, dim]))
        ttnn.deallocate(x)
        rows = ttnn.to_memory_config(ttnn.reshape(rows, [1, B * P, dim]), ttnn.DRAM_MEMORY_CONFIG)
        normed = model.norm(rows, mode=Mode.PREFILL)
        if normed is not rows:
            ttnn.deallocate(rows)
        return normed

    def _run_traced(self, key, tokens, sel):
        B, S, P, _ = key
        mesh = self.mesh_device
        tokens_host = ttnn.from_torch(tokens.reshape(1, B * S), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        sel_host = ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        buf = self._trace_inputs.get(key)
        # The inputs are rewritten before each replay and the output is read right after it, so a replay of any
        # other trace may corrupt them in between: the tracker must not count them.
        with trace_allocation_tracker.corruptible_allocation_scope(mesh):
            if buf is None:
                buf = self._trace_inputs[key] = (
                    ttnn.from_torch(
                        tokens.reshape(1, B * S), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
                    ),
                    ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
                )
            else:
                ttnn.copy_host_to_device_tensor(tokens_host, buf[0])
                ttnn.copy_host_to_device_tensor(sel_host, buf[1])
            if key not in self._traces:
                cos, sin = self._rope(S)
                valid = [S] * B
                before = self._trace_bytes()
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = self._body(buf[0], buf[1], cos, sin, B, S, P, valid)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh)
                self._traces[key] = (trace_id, out)
                self.trace_region_use[key] = self._trace_bytes() - before
        trace_id, out = self._traces[key]
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        return ttnn.to_torch(out).reshape(B, P, -1).float()

    def _trace_bytes(self):
        view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        return view.total_bytes_allocated_per_bank * view.num_banks

    def release_traces(self):
        """Release every captured trace (a new program must compile only while none is parked)."""
        if self._traces:
            ttnn.synchronize_device(self.mesh_device)
        for trace_id, _ in self._traces.values():
            ttnn.release_trace(self.mesh_device, trace_id)
        self._traces.clear()

    def close(self):
        self.release_traces()

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
