# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn Qwen3-1.7B decoder for Qwen3-ASR, built on tt_transformers.

The text decoder is a standard Qwen3 (validated: extracted checkpoint reproduces
golden logits PCC=1.0). We reuse `tt_transformers.tt.model.Transformer` verbatim and
only override prefill input prep so the prompt enters as pre-merged embeddings
(audio embeds spliced at audio-token positions) instead of token ids — the qwen3_vl
pattern, minus vision MRoPE (Qwen3-ASR uses plain 1D RoPE).

prefill (embeds) -> greedy decode loop (token ids) -> text.
"""
import os
import time

import torch

import ttnn
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.model import Transformer

# Host argmax was the PR default (wide vocab D2H was cheaper than ttnn.argmax on P150).
# Set QWEN3ASR_ONDEVICE_ARGMAX=0 to restore that path.
ONDEVICE_ARGMAX = os.environ.get("QWEN3ASR_ONDEVICE_ARGMAX", "1") == "1"
# Whisper-style decode trace: capture one AR step, replay it for every later token.
# Set QWEN3ASR_DECODE_TRACE=0 to keep the eager loop.
DECODE_TRACE = os.environ.get("QWEN3ASR_DECODE_TRACE", "1") == "1"
# In-graph token + pos: argmax copies into the decode token buffer and plus_one
# advances current_pos / RoPE idxs inside the captured graph (no per-step H2D).
# Set QWEN3ASR_INGRAPH_DECODE=0 to restore host restage every AR step.
INGRAPH_DECODE = os.environ.get("QWEN3ASR_INGRAPH_DECODE", "1") == "1"
# Overlap blocking EOS D2H with the next AR step (ViT/Whisper 2CQ). Requires
# open_device(..., num_command_queues=2). Set QWEN3ASR_2CQ=0 to disable.
USE_2CQ = os.environ.get("QWEN3ASR_2CQ", "1") == "1"


class Qwen3ASRDecoder(Transformer):
    def prepare_inputs_prefill_embeds(self, inputs_embeds, **kwargs):
        """inputs_embeds: torch (1, S, dim). Returns the same tuple as the base
        prepare_inputs_prefill but with the embedding step replaced by our embeds."""
        S = inputs_embeds.shape[-2]
        dummy = torch.zeros(1, S, dtype=torch.long)
        out = list(super().prepare_inputs_prefill(dummy, **kwargs))
        # Match self.embd on TP>1: column-shard hidden dim so DistributedNorm
        # all-gathers 1024+1024 -> 2048 before RMSNorm (replicate would look like 4096).
        mapper = (
            ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
            if self.args.num_devices > 1
            else ttnn.ReplicateTensorToMesh(self.mesh_device)
        )
        emb = ttnn.from_torch(
            inputs_embeds.reshape(1, 1, S, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        out[0] = emb
        return tuple(out)

    @torch.no_grad()
    def prefill_logits(self, inputs_embeds):
        """Run prefill on merged embeddings; return last-token logits (torch, vocab) and
        populate the internal KV cache for decoding. Pads the sequence to a multiple of
        128 (attention prefill requirement); causal masking makes the trailing pad
        positions invisible to the last real token."""
        S = inputs_embeds.shape[-2]
        last = S - 1
        # Always pad prefill to a multiple of 512 (the Blackhole prefill_len_cutoff), min 512.
        # The MLP reshapes x to [1, S//512, 512, -1] only when S >= 512, so a 256-pad prefill
        # takes a DIFFERENT (no-reshape) code path. Mixing 256-pad and >=512-pad prefills in
        # one long-lived process corrupts the model (every later request returns garbage/empty)
        # — likely a program-cache/KV-shape inconsistency across the two paths. Forcing every
        # prefill onto the >=512 reshape path keeps it consistent. (256 alone is fine; mixing
        # is not.) Caps single-shot at max_seq_len (2048 -> ~150s). Trailing pad is causal-masked.
        S_pad = ((S + 511) // 512) * 512
        if S_pad != S:
            inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, S_pad - S))
        # tt_transformers.Transformer.prepare_inputs_prefill returns 6 values
        # (tokens, rot_global, rot_local, page_table, chunk_page_table, chunk_start_idx);
        # we drive a single, non-paged, non-chunked prefill so only the first three matter.
        prefill_input, rot_g, rot_l = self.prepare_inputs_prefill_embeds(
            inputs_embeds, page_table=None, batch_size=1, user_id=0
        )[:3]
        get_last = (last // 32) * 32
        tt_logits = self.ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_g,
            rot_mats_local=rot_l,
            user_id=0,
            page_table=None,
            get_last_token=get_last,
            kv_cache=None,
            batch_size=1,
        )
        last_idx = last - get_last
        if ONDEVICE_ARGMAX and self.args.num_devices == 1:
            return self._argmax_token_device(tt_logits, seq_idx=last_idx), S
        # Prefill logits are vocab-sharded on N300; concat on host then argmax.
        tt_logits = ttnn.from_device(tt_logits)
        full = self.process_output_prefill(tt_logits, last_token_idx=last_idx)
        if ONDEVICE_ARGMAX:
            return int(full.float().argmax()), S
        return full.float(), S

    def _argmax_from_untilized(self, tt_logits, seq_idx=0):
        """Slice valid vocab and argmax on device. Returns the idx tensor (not a python int)."""
        if tt_logits.layout != ttnn.ROW_MAJOR_LAYOUT:
            tt_logits = ttnn.untilize(tt_logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vocab = int(self.vocab_size)
        row = ttnn.slice(tt_logits, [0, 0, seq_idx, 0], [1, 1, seq_idx + 1, vocab])
        idx = ttnn.argmax(row, dim=-1, keepdim=False)
        return idx, row, tt_logits

    def _read_token_id(self, idx):
        if self.args.num_devices > 1:
            return int(ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).reshape(-1)[0])
        return int(ttnn.to_torch(idx).reshape(-1)[0])

    def _argmax_token_device(self, tt_logits, seq_idx=0):
        """On-device argmax over vocab. Returns a python int; only the token id is read back."""
        idx, row, _ = self._argmax_from_untilized(tt_logits, seq_idx=seq_idx)
        tok = self._read_token_id(idx)
        ttnn.deallocate(idx)
        ttnn.deallocate(row)
        return tok

    def _host_decode_inputs(self, token_id, pos):
        tokens = torch.tensor([token_id], dtype=torch.long)
        current_pos = torch.tensor([pos], dtype=torch.int64)
        return self.prepare_decode_inputs_host(tokens, current_pos, page_table=None)

    def _decode_forward_argmax(self, device_inputs, feedback=False):
        tt_tokens, tt_pos, rope_idxs, tt_pt = device_inputs
        tt_out, _ = self.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=rope_idxs, page_table=tt_pt, kv_cache=None)
        idx, row, logits = self._argmax_from_untilized(tt_out, seq_idx=0)
        if feedback:
            self._write_token_feedback(idx, tt_tokens)
            self._increment_decode_positions_device(tt_pos, rope_idxs)
        return idx, row, logits

    def _write_token_feedback(self, idx, tt_tokens):
        """Copy the sampled id into the live decode token buffer (first of 32 padded slots)."""
        src = idx
        if src.dtype != ttnn.uint32:
            src = ttnn.typecast(src, dtype=ttnn.uint32)
        tgt_shape = list(tt_tokens.shape)
        src_shape = list(src.shape)
        if src_shape != tgt_shape:
            # Argmax of a single vocab row is rank-3/4 volume-1; the token buffer
            # is tile-padded to 32. Pad trailing ids with 0 (inactive slots).
            src = ttnn.reshape(src, [1, 1, 1, 1])
            pad = []
            for s, t in zip([1, 1, 1, 1], tgt_shape):
                if t < s:
                    raise RuntimeError(f"token feedback vs buffer {tgt_shape}")
                pad.append((0, t - s))
            if any(p[1] for p in pad):
                src = ttnn.pad(src, padding=pad, value=0)
            if list(src.shape) != tgt_shape:
                src = ttnn.reshape(src, tt_tokens.shape)
        ttnn.copy(src, tt_tokens)

    def _release_decode_trace(self):
        """Drop the captured decode graph so the encoder/prefill can allocate."""
        tid = getattr(self, "_decode_trace_id", None)
        if tid is not None:
            ttnn.release_trace(self.mesh_device, tid)
        self._decode_trace_id = None
        # Never deallocate _decode_trace_tok when it aliases the persistent token buffer.
        tok_buf = None
        if getattr(self, "_decode_trace_inputs", None) is not None:
            tok_buf = self._decode_trace_inputs[0]
        for name in ("_decode_trace_tok", "_decode_trace_row", "_decode_trace_logits"):
            t = getattr(self, name, None)
            if t is None or t is tok_buf:
                setattr(self, name, None)
                continue
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
            setattr(self, name, None)

    def _capture_decode_trace(self, token_id, pos):
        """Compile one AR step (once), then capture it. Replay uses the same device buffers."""
        host_inputs = self._host_decode_inputs(token_id, pos)
        feedback = INGRAPH_DECODE
        if not getattr(self, "_decode_kernels_compiled", False):
            compile_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
            idx, row, _logits = self._decode_forward_argmax(compile_inputs, feedback=feedback)
            self._alloc_read_slots(idx)
            # JIT copy + D2H before the trace is live so later CQ1 reads don't allocate.
            if self._read_slots:
                ttnn.copy(idx, self._read_slots[0])
                ttnn.copy(idx, self._read_slots[1])
                _ = ttnn.from_device(self._read_slots[0], blocking=True)
            if idx is not compile_inputs[0]:
                ttnn.deallocate(idx)
            ttnn.deallocate(row)
            ttnn.synchronize_device(self.mesh_device)
            self._decode_kernels_compiled = True

        if getattr(self, "_decode_trace_inputs", None) is None:
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
            self._decode_trace_inputs = device_inputs
        else:
            copy_host_to_device(host_tensors=host_inputs, device_tensors=self._decode_trace_inputs)
            device_inputs = self._decode_trace_inputs

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        idx, row, logits = self._decode_forward_argmax(device_inputs, feedback=feedback)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        self._decode_trace_id = trace_id
        self._decode_trace_tok = idx
        self._decode_trace_row = row
        self._decode_trace_logits = logits
        extra = " ingraph_token_pos" if feedback else ""
        print(f"[generate] decode trace captured{extra}", flush=True)
        if feedback:
            # Capture mutated token/pos (and KV at this pos). Restage this step's
            # inputs so the first execute_trace is the real AR step, same as Whisper.
            copy_host_to_device(host_tensors=host_inputs, device_tensors=device_inputs)

    def _alloc_read_slots(self, proto):
        """Ping-pong DRAM snapshots of the argmax id so CQ1 can D2H while CQ0 runs the next step."""
        if getattr(self, "_read_slots", None):
            return
        spec = getattr(proto, "spec", None)
        slots = []
        for _ in range(2):
            if spec is not None:
                slots.append(ttnn.allocate_tensor_on_device(spec, self.mesh_device))
            else:
                slots.append(
                    ttnn.allocate_tensor_on_device(
                        proto.shape,
                        proto.dtype,
                        proto.layout,
                        self.mesh_device,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
        self._read_slots = slots
        self._read_i = 0

    def _enqueue_traced_step(self):
        """CQ0: execute AR step + snapshot argmax. CQ1: async D2H of the snapshot."""
        ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=False)
        slot = self._read_slots[self._read_i % 2]
        ttnn.copy(self._decode_trace_tok, slot)
        op_ev = ttnn.record_event(self.mesh_device, 0)
        ttnn.wait_for_event(1, op_ev)
        host = ttnn.from_device(slot, blocking=False, cq_id=1)
        rd_ev = ttnn.record_event(self.mesh_device, 1)
        self._read_i += 1
        return host, rd_ev

    def _consume_token_host(self, host, ev):
        ttnn.event_synchronize(ev)
        if self.args.num_devices > 1:
            return int(ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[0])
        return int(ttnn.to_torch(host).reshape(-1)[0])

    def decode_token_traced(self, token_id, pos):
        """Replay the captured AR step. Captures on the first call of each generate()."""
        if getattr(self, "_decode_trace_id", None) is None:
            self._capture_decode_trace(token_id, pos)
        elif not INGRAPH_DECODE:
            host_inputs = self._host_decode_inputs(token_id, pos)
            copy_host_to_device(host_tensors=host_inputs, device_tensors=self._decode_trace_inputs)
        ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=True)
        return self._read_token_id(self._decode_trace_tok)

    def _decode_loop_2cq(self, nxt, pos, max_new_tokens, eos_id, out):
        """Speculate the next AR step on CQ0 while CQ1 reads back the previous token."""
        if getattr(self, "_decode_trace_id", None) is None:
            self._capture_decode_trace(nxt, pos)
        pending = None
        while len(out) < max_new_tokens:
            host, ev = self._enqueue_traced_step()
            if pending is not None:
                tok = self._consume_token_host(*pending)
                out.append(tok)
                if tok == eos_id:
                    return
            pending = (host, ev)
            pos += 1
        if pending is not None:
            tok = self._consume_token_host(*pending)
            out.append(tok)

    @torch.no_grad()
    def decode_token(self, token_id, pos):
        """One greedy decode step. token_id: int, pos: int (0-based position of this token).
        Returns next-token logits (torch, vocab), or a python token id if on-device argmax."""
        host_inputs = self._host_decode_inputs(token_id, pos)
        tt_tokens, tt_pos, rope_idxs, tt_pt = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        tt_out, _ = self.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=rope_idxs, page_table=tt_pt, kv_cache=None)
        if ONDEVICE_ARGMAX:
            return self._argmax_token_device(tt_out, seq_idx=0)
        tt_out = ttnn.from_device(tt_out)
        logits = self.process_output_decode(tt_out, B=1, S=1)
        return logits.float().reshape(-1)

    @torch.no_grad()
    def generate(self, inputs_embeds, max_new_tokens=64, eos_id=151645):
        use_trace = DECODE_TRACE and ONDEVICE_ARGMAX
        use_2cq = use_trace and INGRAPH_DECODE and USE_2CQ
        print(
            f"[generate] ondevice_argmax={ONDEVICE_ARGMAX} decode_trace={use_trace} "
            f"ingraph_decode={use_trace and INGRAPH_DECODE} 2cq={use_2cq}",
            flush=True,
        )
        t0 = time.time()
        prefill_out, S = self.prefill_logits(inputs_embeds)
        t_prefill = time.time() - t0
        nxt = int(prefill_out if ONDEVICE_ARGMAX else prefill_out.argmax())
        out = [nxt]
        pos = S
        t1 = time.time()
        try:
            if use_2cq:
                self._decode_loop_2cq(nxt, pos, max_new_tokens, eos_id, out)
            else:
                while len(out) < max_new_tokens and nxt != eos_id:
                    step = self.decode_token_traced(nxt, pos) if use_trace else self.decode_token(nxt, pos)
                    nxt = int(step if ONDEVICE_ARGMAX else step.argmax())
                    out.append(nxt)
                    pos += 1
        finally:
            t_decode = time.time() - t1
            if use_trace:
                self._release_decode_trace()
        if out and out[-1] == eos_id:
            out = out[:-1]
        ntok = max(len(out), 1)
        print(
            f"[generate] prefill={t_prefill:.3f}s decode={t_decode:.3f}s "
            f"({ntok} tok, {ntok / max(t_decode, 1e-9):.1f} tok/s)",
            flush=True,
        )
        return out
