# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
)
from models.experimental.xtts.tt.xtts_gpt_block import (
    _mm_1d_config,
    _to_device,
    _to_device_w8,
    matmul_compute_config,
    sharded_decode_ln,
    sharded_prefill_ln,
)
from models.experimental.xtts.tt.xtts_gpt_stack import TtXttsGptStack

TILE = 32


def _to_device_rm(torch_tensor, device, memory_config=None):
    """Upload a torch tensor to device in row-major bfloat16."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
    )


class TtXttsGptModel(LightweightModule):
    def __init__(self, state_dict, device, num_layers=NUM_LAYERS):
        """Load GPT embeddings, stack, norms, and heads onto device."""
        super().__init__()
        self.device = device

        # Prefill tables in L1; mel (decode) tables stay DRAM.
        self.text_emb_weight = _to_device_rm(state_dict["gpt.text_embedding.weight"], device, ttnn.L1_MEMORY_CONFIG)
        self.mel_emb_weight = _to_device_rm(state_dict["gpt.mel_embedding.weight"], device)
        self.text_pos_weight = _to_device_rm(
            state_dict["gpt.text_pos_embedding.emb.weight"], device, ttnn.L1_MEMORY_CONFIG
        )
        self.mel_pos_weight = _to_device_rm(state_dict["gpt.mel_pos_embedding.emb.weight"], device)

        self.stack = TtXttsGptStack(state_dict, device, num_layers=num_layers)
        self.final_norm_weight = _to_device(state_dict["gpt.final_norm.weight"], device)
        self.final_norm_bias = _to_device(state_dict["gpt.final_norm.bias"], device)

        # nn.Linear [out, in] -> ttnn.linear [in, out].
        self.text_head_weight = _to_device(state_dict["gpt.text_head.weight"].t().contiguous(), device)
        self.text_head_bias = _to_device(state_dict["gpt.text_head.bias"], device)
        self.mel_head_weight = _to_device_w8(state_dict["gpt.mel_head.weight"].t().contiguous(), device)
        self.mel_head_bias = _to_device(state_dict["gpt.mel_head.bias"].reshape(1, -1), device)

    def _embed(self, ids, tok_weight, pos_weight):
        """Embed token ids with matching positional table."""
        seq = ids.shape[1]
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )
        pos_tt = ttnn.arange(0, seq, 1, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = ttnn.reshape(pos_tt, (1, seq))

        tok = ttnn.to_layout(ttnn.embedding(ids_tt, tok_weight), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ids_tt)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, pos_weight), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_tt)
        emb = ttnn.add(tok, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)
        return emb

    def forward(self, text_ids, mel_ids, cond_latents=None):
        """Run full GPT forward for text and mel logits."""
        text_len, mel_len = text_ids.shape[1], mel_ids.shape[1]

        text_emb = self._embed(text_ids, self.text_emb_weight, self.text_pos_weight)
        mel_emb = self._embed(mel_ids, self.mel_emb_weight, self.mel_pos_weight)

        parts, offset = [text_emb, mel_emb], 0
        if cond_latents is not None:
            parts = [cond_latents] + parts
            offset = cond_latents.shape[1]

        emb = ttnn.concat(parts, dim=1)
        ttnn.deallocate(text_emb)
        ttnn.deallocate(mel_emb)
        enc = self.stack(emb)
        if offset:
            enc_stripped = ttnn.slice(enc, [0, offset, 0], [enc.shape[0], enc.shape[1], HIDDEN_SIZE])
            ttnn.deallocate(enc)
            enc = enc_stripped
        enc_n = sharded_prefill_ln(enc, self.final_norm_weight, self.final_norm_bias, self.device)

        b = enc_n.shape[0]
        text_part = ttnn.slice(enc_n, [0, 0, 0], [b, text_len, HIDDEN_SIZE])
        mel_part = ttnn.slice(enc_n, [0, text_len, 0], [b, text_len + mel_len, HIDDEN_SIZE])
        ttnn.deallocate(enc_n)

        cc = matmul_compute_config(self.device)
        text_logits = ttnn.linear(text_part, self.text_head_weight, bias=self.text_head_bias, compute_kernel_config=cc)
        ttnn.deallocate(text_part)
        mel_logits = ttnn.linear(mel_part, self.mel_head_weight, bias=self.mel_head_bias, compute_kernel_config=cc)
        ttnn.deallocate(mel_part)
        return text_logits, mel_logits

    def prefill(self, text_ids, cond_latents, max_seq):
        """Prefill prompt into static KV and return the cache."""
        self.alloc_static_kv(max_seq)
        self.prompt_len = self.prefill_on_device(self.text_ids_to_device(text_ids), cond_latents)
        return self._static_kv

    def decode(self, token_id, mel_pos, kv):
        """Decode one mel token against the KV cache."""
        pos = self.prompt_len + mel_pos
        logits, latent = self.decode_on_device(
            self._pos_ids(token_id), self._pos_ids(mel_pos), self.cache_pos(pos), kv, write_idx=pos
        )
        return logits, latent, kv

    def init_static_decode(self, max_seq):
        """Initialize stack static buffers for decode."""
        self.max_seq = max_seq
        self.stack.init_static(max_seq)

    def set_text_padding(self, cond_len, real_len, padded_len):
        """Hide the text padding of the current prompt from decode attention.

        The prompt is [cond latents | text | STOP padding]. The padding only exists to pin a
        chunked take's prompt geometry to ONE captured trace; the model must not read it, or a
        short chunk keeps generating (it repeats itself, or drones) instead of emitting STOP.
        Prefill is causal and the padding sits last, so the real tokens' K/V are unaffected —
        masking these slots at decode makes a padded prompt exactly equal to an unpadded one.
        """
        self.stack.set_prompt_pad(cond_len + real_len, cond_len + padded_len)

    def _pos_ids(self, value):
        """Build a 1x1 uint32 id tensor on device."""
        return ttnn.from_torch(
            torch.tensor([[value]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )

    def cache_pos(self, value):
        """Build a broadcast cache-position tensor for decode."""
        return ttnn.from_torch(
            torch.full((1, 1, 1, self.max_seq), float(value), dtype=torch.float32),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32,
        )

    def decode_on_device(self, token_ids, mel_pos_ids, cache_pos, kv, write_idx=None):
        # RM emb + add, then one tilize (avoids two TilizeWithValPadding).
        """Run on-device decode step returning logits and latent."""
        tok = ttnn.embedding(token_ids, self.mel_emb_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        posn = ttnn.embedding(mel_pos_ids, self.mel_pos_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(
            ttnn.add(tok, posn, memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(tok)
        ttnn.deallocate(posn)
        hidden = self.stack.forward_decode(x, kv, cache_pos, write_idx=write_idx)
        latent = sharded_decode_ln(hidden, self.final_norm_weight, self.final_norm_bias, self.device)
        logits = ttnn.linear(
            latent,
            self.mel_head_weight,
            bias=self.mel_head_bias,
            program_config=_mm_1d_config(
                self.device, latent.shape[-2], latent.shape[-1], self.mel_head_weight.shape[-1]
            ),
            compute_kernel_config=matmul_compute_config(self.device),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return logits, latent

    def alloc_static_kv(self, max_seq):
        """Allocate static KV caches and text position table."""
        self.init_static_decode(max_seq)
        self._text_pos_full = ttnn.from_torch(
            torch.arange(max_seq, dtype=torch.int32).reshape(1, max_seq),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self._static_kv = []
        for _ in range(self.stack.num_layers):
            k = ttnn.from_torch(
                torch.zeros(1, NUM_HEADS, max_seq, HEAD_DIM),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            v = ttnn.from_torch(
                torch.zeros(1, NUM_HEADS, max_seq, HEAD_DIM),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            self._static_kv.append((k, v))
        return self._static_kv

    def text_ids_to_device(self, text_ids):
        """Upload text token ids to device as uint32."""
        return ttnn.from_torch(
            text_ids.to(torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _embed_dev(self, ids_tt, tok_weight, pos_weight):
        """Embed device ids using a sliced position table."""
        seq = ids_tt.shape[1]
        pos_tt = ttnn.slice(self._text_pos_full, [0, 0], [1, seq], memory_config=ttnn.L1_MEMORY_CONFIG)
        tok = ttnn.embedding(ids_tt, tok_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        pos = ttnn.embedding(pos_tt, pos_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_tt)
        emb = ttnn.to_layout(
            ttnn.add(tok, pos, memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)
        return emb

    def prefill_on_device(self, text_ids_tt, cond_latents):
        """Prefill on-device and fill static KV caches."""
        text_emb = self._embed_dev(text_ids_tt, self.text_emb_weight, self.text_pos_weight)
        prefix = ttnn.concat([cond_latents, text_emb], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(text_emb)
        prompt_ln, kv = self.stack.forward_prefill(prefix)
        ttnn.deallocate(prompt_ln)
        prompt_len = kv[0][0].shape[2]
        # fill_cache corrupts the first tile when prompt tiles is odd and heads=16 (measured at 9,
        # 11 and 13 tiles); pad +1 tile. Ceil, not floor: a prompt that is not tile-aligned still
        # occupies ceil(len/TILE) tiles, and flooring would miss the pad for e.g. 260 rows.
        prompt_tiles = -(-prompt_len // TILE)
        pad_row = prompt_tiles % 2 == 1 and prompt_len + TILE <= self.max_seq
        for i, (k, v) in enumerate(kv):
            kw = ttnn.pad(k, [(0, 0), (0, 0), (0, TILE), (0, 0)], value=0.0) if pad_row else k
            vw = ttnn.pad(v, [(0, 0), (0, 0), (0, TILE), (0, 0)], value=0.0) if pad_row else v
            ttnn.fill_cache(self._static_kv[i][0], kw, 0)
            ttnn.fill_cache(self._static_kv[i][1], vw, 0)
            if pad_row:
                ttnn.deallocate(kw)
                ttnn.deallocate(vw)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        return prompt_len
