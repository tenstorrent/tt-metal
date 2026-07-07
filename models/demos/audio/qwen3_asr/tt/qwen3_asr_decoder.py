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
import torch

import ttnn
from models.tt_transformers.tt.model import Transformer


class Qwen3ASRDecoder(Transformer):
    def prepare_inputs_prefill_embeds(self, inputs_embeds, **kwargs):
        """inputs_embeds: torch (1, S, dim). Returns the same tuple as the base
        prepare_inputs_prefill but with the embedding step replaced by our embeds."""
        S = inputs_embeds.shape[-2]
        dummy = torch.zeros(1, S, dtype=torch.long)
        out = list(super().prepare_inputs_prefill(dummy, **kwargs))
        emb = ttnn.from_torch(
            inputs_embeds.reshape(1, 1, S, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
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
        tt_logits = ttnn.from_device(tt_logits)
        full = self.process_output_prefill(tt_logits, last_token_idx=(last - get_last))
        return full.float(), S

    @torch.no_grad()
    def decode_token(self, token_id, pos):
        """One greedy decode step. token_id: int, pos: int (0-based position of this token).
        Returns next-token logits (torch, vocab)."""
        tokens = torch.tensor([token_id], dtype=torch.long)  # (B=1,)
        current_pos = torch.tensor([pos], dtype=torch.int64)  # (B=1,)
        tt_tokens, tt_pos, rope_idxs, tt_pt = self.prepare_inputs_decode(tokens, current_pos, None)
        tt_out, _ = self.ttnn_decode_forward(tt_tokens, tt_pos, rot_mat_idxs=rope_idxs, page_table=tt_pt, kv_cache=None)
        tt_out = ttnn.from_device(tt_out)
        logits = self.process_output_decode(tt_out, B=1, S=1)
        return logits.float().reshape(-1)

    @torch.no_grad()
    def generate(self, inputs_embeds, max_new_tokens=64, eos_id=151645):
        logits, S = self.prefill_logits(inputs_embeds)
        nxt = int(logits.argmax())
        out = [nxt]
        pos = S
        # Host-side argmax greedy decode. This is faster here than an on-device ttnn.argmax
        # over the 151936-wide vocab (the wide reduction kernel costs more than the logits
        # host transfer), and unlike a persistent decode trace it stays stable across the
        # mixed request shapes of a long-lived server.
        while len(out) < max_new_tokens and nxt != eos_id:
            nxt = int(self.decode_token(nxt, pos).argmax())
            out.append(nxt)
            pos += 1
        if out and out[-1] == eos_id:
            out = out[:-1]
        return out
