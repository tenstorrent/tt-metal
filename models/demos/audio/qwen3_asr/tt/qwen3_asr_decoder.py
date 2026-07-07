# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn Qwen3-1.7B decoder for Qwen3-ASR, built on tt_transformers.

The text decoder is a standard Qwen3 (validated: extracted checkpoint reproduces
golden logits PCC=1.0). We reuse `tt_transformers.tt.model.Transformer` verbatim; the
only model-level change is the qwen3_vl "tokens is actually embeddings" trick in
`prepare_inputs_prefill`, so the prompt enters as pre-merged embeddings (audio embeds
spliced at the audio-token positions) instead of token ids — the qwen3_vl pattern,
minus vision MRoPE (Qwen3-ASR uses plain 1D RoPE).

Both prefill and the greedy decode loop are driven through the shared
`tt_transformers.tt.generator.Generator` rather than bespoke plumbing:
  - prefill: `Generator.prefill_forward_single_user_text` (single-user, non-paged),
    which calls our embeds-aware `prepare_inputs_prefill` + the shared `ttnn_prefill_forward`.
  - decode : `Generator.decode_forward(enable_trace=False, ...)` + host argmax greedy.

Trace is intentionally left OFF: a persistent decode trace destabilized the long-lived
server across mixed prefill lengths (see README "Known limitations"). Generator makes
trace opt-in per call, so we keep the shared decode path without the instability.

prefill (embeds) -> greedy decode loop (token ids) -> text.
"""
import torch

import ttnn
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer


class Qwen3ASRDecoder(Transformer):
    @property
    def generator(self):
        """Lazily wrap this model in a stock `Generator` (single replica) used to drive
        prefill + decode. Constructed on first use so existing callers (which build the
        decoder exactly like any tt_transformers model) need no changes."""
        gen = getattr(self, "_generator", None)
        if gen is None:
            gen = self._generator = Generator([self], [self.args], self.mesh_device)
        return gen

    def prepare_inputs_prefill(self, tokens, **kwargs):
        """qwen3_vl-style embeds injection ("tokens is actually embeddings").

        `tokens` here is the pre-merged inputs_embeds (torch, (1, S, dim) or (S, dim)).
        We reuse the base `Transformer.prepare_inputs_prefill` for all the rot-mat /
        page-table preparation — driven by a throwaway id tensor — and only swap the
        embedding-lookup step for our embeddings, so this plugs straight into
        `Generator.prefill_forward_single_user_text`."""
        inputs_embeds = tokens if tokens.dim() == 3 else tokens.unsqueeze(0)
        S = inputs_embeds.shape[-2]
        # The base uses last_token_idx only for a seq-len assert; S already covers it.
        kwargs.pop("last_token_idx", None)
        dummy = torch.zeros(1, S, dtype=torch.long)
        out = list(super().prepare_inputs_prefill(dummy, **kwargs))
        out[0] = ttnn.from_torch(
            inputs_embeds.reshape(1, 1, S, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tuple(out)

    @torch.no_grad()
    def prefill_logits(self, inputs_embeds):
        """Run prefill on merged embeddings via the shared Generator single-user text
        path; return last-token logits (torch, vocab) and populate the internal KV cache
        for decoding. Pads the sequence to a multiple of 512, min 512 (the Blackhole
        prefill_len_cutoff / MLP reshape rule — see the comment on S_pad below); causal
        masking makes the trailing pad positions invisible to the last real token."""
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
        # Generator's single-user text prefill calls our embeds-aware prepare_inputs_prefill
        # + the shared ttnn_prefill_forward. Non-paged single-shot (page_table/kv_cache=None):
        # S_pad <= max_seq_len (2048) <= max_prefill_chunk_size, so it stays on the single-chunk
        # path (no paging required). It applies get_last_token=(last // 32) * 32 internally.
        tt_logits = self.generator.prefill_forward_single_user_text(
            inputs_embeds,
            page_table=None,
            user_id=0,
            last_token_idx=last,
            kv_cache=None,
            batch_size=1,
        )
        tt_logits = ttnn.from_device(tt_logits)
        get_last = (last // 32) * 32
        full = self.process_output_prefill(tt_logits, last_token_idx=(last - get_last))
        return full.float(), S

    @torch.no_grad()
    def generate(self, inputs_embeds, max_new_tokens=64, eos_id=151645):
        logits, S = self.prefill_logits(inputs_embeds)
        nxt = int(logits.argmax())
        out = [nxt]
        pos = S
        gen = self.generator
        # Host-side argmax greedy decode via the shared Generator with enable_trace=False.
        # Host argmax is faster here than an on-device ttnn.argmax over the 151936-wide vocab
        # (the wide reduction costs more than the logits host transfer), and a non-traced decode
        # stays stable across the mixed request shapes of a long-lived server (a persistent
        # decode trace did not — see README "Known limitations").
        while len(out) < max_new_tokens and nxt != eos_id:
            dl = gen.decode_forward(
                torch.tensor([[nxt]], dtype=torch.long),
                torch.tensor([pos]),
                page_table=None,
                kv_cache=None,
                enable_trace=False,
                read_from_device=True,
            )
            dl = (dl[0] if isinstance(dl, tuple) else dl).squeeze().float()
            nxt = int(dl.argmax())
            out.append(nxt)
            pos += 1
        if out and out[-1] == eos_id:
            out = out[:-1]
        return out
