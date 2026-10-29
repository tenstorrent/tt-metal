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
from models.demos.blackhole.qwen36.tt.prefill_matmul import swiglu_enabled
from models.demos.strands_agent.tt.answers import build_answer
from models.demos.strands_agent.tt.head import load_config, load_head
from models.demos.strands_agent.tt.model_config import StrandsDeciderModelArgs
from models.demos.strands_agent.tt.prompting import EncodedQuestion, encode_request
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat

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


_NORM_DIM = 2048
_NORM_GRID_X = 8
_NORM_MAX_ROWS = 512
_norm_configs = {}


def _norm_config(rows):
    cfg = _norm_configs.get(rows)
    if cfg is None:
        m_tiles = rows // 32
        gy = max(d for d in range(1, 9) if m_tiles % d == 0)
        block_h = m_tiles // gy
        mem = ttnn.create_sharded_memory_config(
            [rows // gy, _NORM_DIM // _NORM_GRID_X],
            core_grid=ttnn.CoreGrid(x=_NORM_GRID_X, y=gy),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(_NORM_GRID_X, gy),
            subblock_w=4,
            block_h=block_h,
            block_w=_NORM_DIM // 32 // _NORM_GRID_X,
            inplace=False,
        )
        cfg = _norm_configs[rows] = (mem, prog)
    return cfg


def _rms_norm(norm, x):
    """RMSNorm; block-sharded on an 8-wide grid for M <= 512 rows (the interleaved op uses one core per tile row)."""
    rows = x.shape[-2]
    if (
        rows > _NORM_MAX_ROWS
        or rows % 32 != 0
        or x.shape[-1] != _NORM_DIM
        or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        or x.memory_config().buffer_type != ttnn.BufferType.DRAM
    ):
        return norm(x, mode=Mode.PREFILL, norm_config=None)
    mem, prog = _norm_config(rows)
    xs = ttnn.to_memory_config(x, mem)
    y = ttnn.rms_norm(
        xs,
        epsilon=norm.eps,
        weight=norm.weight,
        program_config=prog,
        memory_config=mem,
        compute_kernel_config=norm.compute_kernel_config_hifi2,
    )
    ttnn.deallocate(xs)
    out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(y)
    return out


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
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for S in TRACED_S:
            self._rope_cache[S] = self._build_rope(S)

    def _build_rope(self, S):
        """cos/sin [1, 1, S, rotary_dim] bf16 TILE for plain 1D RoPE, in the rope-pair layout (cos[2j] = cos[2j + 1])."""
        model = self.model
        model._build_request_rope(torch.zeros(1, 1, dtype=torch.int32), None)  # text only: plain 1D RoPE
        cos_t, sin_t = model.rope.prefill_cos_sin_torch(0, S)  # [S, rd], HF layout cat([c, c])
        half = cos_t.shape[-1] // 2
        kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        return tuple(
            ttnn.from_torch(t[:, :half].repeat_interleave(2, dim=-1).reshape(1, 1, S, 2 * half).contiguous(), **kw)
            for t in (cos_t, sin_t)
        )

    @classmethod
    def from_pretrained(cls, mesh_device, max_seq_len=4096, trace=True):
        args = StrandsDeciderModelArgs(mesh_device=mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
        state_dict = args.load_state_dict()
        model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=args.weight_cache_path())
        del state_dict
        decider = cls(mesh_device, args, model, trace=trace)
        # One small forward builds the first-call caches (stateless GDN constants, fused attention weights), so
        # nothing persistent is allocated once a trace exists.
        if swiglu_enabled():
            for layer in model.layers:
                layer.feed_forward._gate_up_fused()  # fused SwiGLU gate/up weight, allocated before any trace
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
        """cos/sin [1, 1, S, rotary_dim] in the pair layout: the persistent ones for traced S, else built per call."""
        if S in self._rope_cache:
            return self._rope_cache[S]
        return self._build_rope(S)

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
        sel = torch.zeros(B, P, S, dtype=torch.bfloat16)  # one-hot row select (bf16 is what the device takes)
        counts = [len(pos) for pos in positions]
        rows_b = torch.repeat_interleave(torch.arange(len(positions)), torch.tensor(counts))
        rows_i = torch.cat([torch.arange(n) for n in counts])
        sel[rows_b, rows_i, torch.tensor([p for pos in positions for p in pos])] = 1.0
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
        return [out[b, : len(pos)].float() for b, pos in enumerate(positions)]

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
        out = ttnn.to_torch(normed).reshape(B, sel.shape[1], -1)
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
            attn_in = _rms_norm(layer.attention_norm, x)
            attn_in = ttnn.reshape(attn_in, [B, S, dim])
            if layer.is_full_attention:
                # Stateless: q|k|v and gate projections, heads built straight to [B, H, S, D], plain causal
                # SDPA; no KV cache is written, so there is nothing to reset.
                attn_out = stateless_attention_prefill(layer.attention, attn_in, cos, sin, self.trans_mat)
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
            ff_in = _rms_norm(layer.ffn_norm, h)
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
        normed = _rms_norm(model.norm, rows)
        if normed is not rows:
            ttnn.deallocate(rows)
        # Row-major, so the host readback is a plain copy (no untilize).
        out = ttnn.to_layout(normed, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(normed)
        return out

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
        return ttnn.to_torch(out).reshape(B, P, -1)

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
        with torch.no_grad():
            logits = self.head.forward_flat(all_rows)
        return list(zip(logits, all_rows))

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
