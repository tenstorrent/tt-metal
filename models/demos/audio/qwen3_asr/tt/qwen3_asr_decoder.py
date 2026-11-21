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
import json
import os

import torch

import ttnn
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer

# <|im_end|>, the end-of-turn token Qwen3-ASR normally stops on. Used when the checkpoint has
# no readable generation_config.json.
DEFAULT_EOS_IDS = (151645,)


def load_eos_ids(*ckpt_dirs):
    """Stop-token ids from the first generation_config.json found in `ckpt_dirs` (the
    checkpoint lists both <|endoftext|> 151643 and <|im_end|> 151645). Falls back to
    DEFAULT_EOS_IDS when no config is readable or it has no eos_token_id."""
    for d in ckpt_dirs:
        if not d:
            continue
        try:
            with open(os.path.join(str(d), "generation_config.json")) as f:
                eos = json.load(f).get("eos_token_id")
        except (OSError, ValueError):
            continue
        if eos is None:
            continue
        ids = tuple(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
        if ids:
            return ids
    return DEFAULT_EOS_IDS


def decode_until_eos(first_token, step, max_new_tokens, eos_ids):
    """Greedy loop shared by `generate`: `step(token, offset)` returns the next token id.
    Stops after emitting any id in `eos_ids` or reaching `max_new_tokens`; the trailing stop
    token is stripped from the result."""
    eos_ids = frozenset(eos_ids)
    nxt = first_token
    out = [nxt]
    while len(out) < max_new_tokens and nxt not in eos_ids:
        nxt = step(nxt, len(out) - 1)
        out.append(nxt)
    if out and out[-1] in eos_ids:
        out = out[:-1]
    return out


class Qwen3ASRDecoder(Transformer):
    @property
    def eos_ids(self):
        """Stop ids from the checkpoint's generation_config.json (cached)."""
        ids = getattr(self, "_eos_ids", None)
        if ids is None:
            ids = self._eos_ids = load_eos_ids(getattr(self.args, "CKPT_DIR", None), os.environ.get("HF_MODEL"))
        return ids

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
        # tt_transformers' MLP reshapes prefill x to [1, S_pad//512, 512, -1] for S_pad >= 512, so
        # different padded lengths differ only in the batch dim -3 (512 -> [1,1,512,d], 1024 ->
        # [1,2,512,d]). A tt-metal program-cache collision (the prefill matmul hash doesn't cover
        # dim -3) then reuses the first bucket's program for the next: confirmed on device, a 512
        # prefill followed by a 1024 prefill TT_FATALs (1024 alone is fine). Real ASR prompts are
        # always <=512 tokens, so min-512 pins every request to the single [1,1,512,d] shape and
        # sidesteps the collision. Caps single-shot at max_seq_len (2048 -> ~150s); trailing pad is
        # causal-masked. See README "Known limitations" and docs/prefill_program_cache_collision_issue.md.
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
    def generate(self, inputs_embeds, max_new_tokens=64, eos_id=None):
        """Greedy decode. `eos_id` is an int or an iterable of stop ids; None uses the
        checkpoint's generation_config.json eos_token_id list (151645 only if absent)."""
        if eos_id is None:
            eos_ids = self.eos_ids
        elif isinstance(eos_id, int):
            eos_ids = (eos_id,)
        else:
            eos_ids = tuple(eos_id)
        logits, S = self.prefill_logits(inputs_embeds)
        gen = self.generator

        # Host-side argmax greedy decode via the shared Generator with enable_trace=False.
        # Host argmax is faster here than an on-device ttnn.argmax over the 151936-wide vocab
        # (the wide reduction costs more than the logits host transfer), and a non-traced decode
        # stays stable across the mixed request shapes of a long-lived server (a persistent
        # decode trace did not — see README "Known limitations").
        def step(tok, offset):
            dl = gen.decode_forward(
                torch.tensor([[tok]], dtype=torch.long),
                torch.tensor([S + offset]),
                page_table=None,
                kv_cache=None,
                enable_trace=False,
                read_from_device=True,
            )
            dl = (dl[0] if isinstance(dl, tuple) else dl).squeeze().float()
            return int(dl.argmax())

        return decode_until_eos(int(logits.argmax()), step, max_new_tokens, eos_ids)
