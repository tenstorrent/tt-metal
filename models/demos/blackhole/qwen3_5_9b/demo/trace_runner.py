# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace capture/replay for Qwen3.5 generation — the demo's --trace path.

A ttnn trace records a graph's op dispatch and the BUFFER ADDRESSES it reads/writes, then replays the
whole graph with a single host->device handoff. That hides per-op host dispatch latency, which is what
bounds decode (each token is one tiny step through 64 layers). Because a trace pins addresses, not
contents, replaying correctly takes two disciplines, both proven by tests/unit/test_*_trace.py and
reproduced here for the whole model:

  1. Persistent input buffers. The captured graph reads fixed device tensors, so every replayed step
     must COPY its new token / position / RoPE into those exact buffers (copy_host_to_device_tensor),
     never reallocate. The captured OUTPUT tensor handle is likewise the buffer replay writes into.

  2. Recurrent-state rewind around decode capture. compile + capture each run the decode step once,
     advancing the GDN conv/recurrent state twice off the post-prefill state. We snapshot that state
     before capture and restore it (in place, preserving the captured addresses) before the first real
     replay. The attention KV cache needs no rewind: the first replayed step rewrites the same position
     the capture run touched, and the prefill history below it is untouched.

Prefill is captured too (the user asked), but it is idempotent — forward_prefill recomputes from the
input and overwrites the KV/GDN state wholesale — so it needs no rewind. Note prefill trace buys little
(prefill is compute-bound, not dispatch-bound); decode is where replay pays off. Reported TTFT / tok/s
time the REPLAY only; the one-time compile+capture cost is returned separately as capture_s.

This reaches into the model's public pieces (embd / layers / norm / lm_head and each GDN layer's
conv_state / last_recurrent_state) exactly as the unit tests do, so tt/model.py stays a plain eager
bring-up with no trace machinery bolted on.
"""
import time

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.demo.demo import pad_prompt
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_prefill
from models.tt_transformers.tt.common import Mode


class TracedRunner:
    """Captures + replays tt/model.py's prefill and decode as ttnn traces for B=1 greedy generation."""

    def __init__(self, model):
        self.m = model
        self.device = model.device

    # ── Forward graphs over persistent buffers (mirror Qwen35Model.forward, fixed-address inputs) ──
    def _prefill_forward(self):
        """embed -> N layers (prefill) -> final norm -> last-real-position logits. Reads the fixed
        prompt + RoPE buffers; the captured return tensor is the logits buffer replay writes into."""
        m = self.m
        x = ttnn.unsqueeze_to_4D(m.embd(self._p_tok))
        for layer in m.layers:
            x = layer.forward_prefill(x, cos=self._p_cos, sin=self._p_sin, user_id=0)
        h = m.norm(x, mode=Mode.PREFILL)  # [1, 1, S, dim]
        last = h[:, :, self._idx : self._idx + 1, :]  # last REAL position (idx baked into the trace)
        return m.lm_head(last)  # [1, 1, 1, vocab]

    def _decode_forward(self):
        """embed -> N layers (decode) -> final norm -> on-device argmax id, one step. Reads the fixed
        token / position / RoPE buffers; GDN layers ignore position/RoPE and advance their own state.

        The greedy argmax is folded into the captured graph (argmax_on_device) so REPLAY returns the
        next-token id, not the logits — the per-token readback drops from the full 248K-wide row to one
        int (~15 ms/token at TP=4), with byte-identical tokens (same bf16 logits, same compare)."""
        m = self.m
        x = ttnn.unsqueeze_to_4D(m.embd(self._d_tok))
        for layer in m.layers:
            x = layer.forward_decode(x, position_tensor=self._d_pos, cos=self._d_cos, sin=self._d_sin)
        h = m.norm(x, mode=Mode.DECODE)  # [1, 1, 1, dim]
        return m.argmax_on_device(m.lm_head(h))  # [1, 1, 1, 1] uint32 id

    # ── Persistent input buffers ─────────────────────────────────────────────────────────────────
    def _repl(self):
        return ttnn.ReplicateTensorToMesh(self.device)

    def _alloc_prefill_io(self, prompt, seq_len):
        """Prompt token row + RoPE tables, all fixed for the whole generation (positions 0..S-1)."""
        self._p_tok = ttnn.from_torch(
            prompt.reshape(1, 1, 1, seq_len).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl(),
        )
        self._p_cos, self._p_sin = rot_mats_prefill(
            self.device, self.m.args.rope_head_dim, seq_len, self.m.args.rope_theta
        )

    def _alloc_decode_io(self):
        """The four per-step decode inputs, zero-initialized; _write_decode_io copies real values in."""
        repl = self._repl()
        rope_dim = self.m.args.rope_head_dim
        self._d_tok = ttnn.from_torch(
            torch.zeros(1, 1, 1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=repl,
        )
        self._d_pos = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=self.device, mesh_mapper=repl
        )
        zc = torch.zeros(1, 1, 1, rope_dim)
        self._d_cos = ttnn.from_torch(
            zc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=repl
        )
        self._d_sin = ttnn.from_torch(
            zc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=repl
        )

    def _write_decode_io(self, token_id, position):
        """Copy one step's token / position / decode-RoPE into the persistent decode buffers.

        Builds the values as host tensors and copy_host_to_device_tensor's them into the fixed device
        buffers the captured graph reads — the canonical trace-input refresh (see sentence_bert /
        tt_transformers generator). RoPE is rebuilt on host per step because it depends on the position.
        """
        repl = self._repl()
        tok_h = ttnn.from_torch(
            torch.tensor([[[[token_id]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=repl,
        )
        ttnn.copy_host_to_device_tensor(tok_h, self._d_tok)
        pos_h = ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, mesh_mapper=repl)
        ttnn.copy_host_to_device_tensor(pos_h, self._d_pos)
        # rot_mats_decode emits device tensors; we want host tensors to copy into the fixed buffers, so
        # build the same [1, 1, 1, rope_dim] cos/sin on host here (B=1) rather than route through it.
        rope_dim, theta = self.m.args.rope_head_dim, self.m.args.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        emb = torch.cat([torch.outer(torch.tensor([float(position)]), inv_freq)] * 2, dim=-1)  # [1, rope_dim]
        cos_h = ttnn.from_torch(
            emb.cos().reshape(1, 1, 1, rope_dim).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=repl,
        )
        sin_h = ttnn.from_torch(
            emb.sin().reshape(1, 1, 1, rope_dim).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=repl,
        )
        ttnn.copy_host_to_device_tensor(cos_h, self._d_cos)
        ttnn.copy_host_to_device_tensor(sin_h, self._d_sin)

    # ── GDN recurrent/conv state snapshot + restore (decode capture rewind) ────────────────────────
    def _gdn_layers(self):
        return [layer.attention for layer in self.m.layers if not layer.is_full_attention]

    def _snapshot_gdn(self):
        """Clone every GDN layer's conv + recurrent state (the post-prefill start state)."""
        return [(g, ttnn.clone(g.conv_state), ttnn.clone(g.last_recurrent_state)) for g in self._gdn_layers()]

    def _restore_gdn(self, snap):
        """Copy the snapshot back into the SAME buffers the trace captured (in place, addresses kept)."""
        for g, conv, rec in snap:
            ttnn.copy(conv, g.conv_state)
            ttnn.copy(rec, g.last_recurrent_state)
            ttnn.deallocate(conv)
            ttnn.deallocate(rec)

    # ── End-to-end traced greedy generation ───────────────────────────────────────────────────────
    def generate(self, prompt_ids, max_new_tokens, eos_token_id=None):
        """Greedy B=1 generation with traced prefill + traced decode.

        Returns (tokens, ttft_s, prefill_tok_s, decode_tok_s, capture_s). ttft / prefill_tok_s time the
        prefill REPLAY only (steady state); the one-time compile+capture cost is capture_s.
        """
        m = self.m
        prompt, valid_len, seq_len = pad_prompt(prompt_ids, m.args.max_seq_len, max_new_tokens)
        m.reset_state()

        # ── Prefill: capture once, replay (idempotent → no state rewind), read the next-token logit ──
        cap_t0 = time.time()
        self._alloc_prefill_io(prompt, seq_len)
        self._idx = valid_len - 1
        self._prefill_forward()  # compile (seeds KV + GDN state)
        p_tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        p_logits = self._prefill_forward()
        ttnn.end_trace_capture(self.device, p_tid, cq_id=0)
        capture_s = time.time() - cap_t0

        t0 = time.time()
        ttnn.execute_trace(self.device, p_tid, cq_id=0, blocking=True)
        prefill_logits = m._logits_to_torch(p_logits, n_rows=1).reshape(-1)
        ttft = time.time() - t0
        prefill_tok_s = seq_len / ttft  # prompt-ingestion rate over the padded length the replay ran
        ttnn.release_trace(self.device, p_tid)

        next_id = int(torch.argmax(prefill_logits).item())
        out = [next_id]
        if max_new_tokens == 1:
            return out, ttft, prefill_tok_s, float("nan"), capture_s

        # ── Decode: capture one step, rewind the GDN state the capture advanced, then replay per token ──
        pos = seq_len
        cap_t0 = time.time()
        self._alloc_decode_io()
        self._write_decode_io(next_id, pos)  # first decode step's inputs (token from prefill @ position S)
        snap = self._snapshot_gdn()  # post-prefill state
        self._decode_forward()  # compile (advances GDN state)
        d_tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        d_tok = self._decode_forward()  # captured output is the on-device argmax id, not the logits
        ttnn.end_trace_capture(self.device, d_tid, cq_id=0)
        self._restore_gdn(snap)  # rewind to post-prefill so the first replay starts where it should
        capture_s += time.time() - cap_t0

        t0 = time.time()
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_id == eos_token_id:
                break
            ttnn.execute_trace(self.device, d_tid, cq_id=0, blocking=True)  # decode(token @ pos) from current state
            # argmax already done on device (d_tok is the id); read one replica -> a single int, not the vocab row.
            next_id = int(ttnn.to_torch(ttnn.get_device_tensors(d_tok)[0]).item())
            out.append(next_id)
            pos += 1
            self._write_decode_io(next_id, pos)  # stage inputs for the NEXT replay
        tok_s = (len(out) - 1) / (time.time() - t0) if len(out) > 1 else float("nan")
        ttnn.release_trace(self.device, d_tid)
        return out, ttft, prefill_tok_s, tok_s, capture_s
