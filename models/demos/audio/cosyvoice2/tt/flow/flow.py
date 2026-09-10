# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`CausalMaskedDiffWithXvec`: the outer module that ties the Conformer encoder
(`tt/flow/encoder.py`) and the CFM estimator (`tt/flow/decoder.py`) together --
speaker-embedding (xvec) projection, speech-token embedding, prompt splicing into
`conds`, and the final CFM call -- confirmed against real upstream source directly
(`cosyvoice/flow/flow.py`'s `CausalMaskedDiffWithXvec.inference`), specifically the
`finalize=True` branch (this phase's `streaming=False` scope throughout).

This is the one piece of this phase that is not "connect two already-validated
components" -- `conds` construction is a real, easy-to-get-wrong splice, verified
here against the literal real method body (quoted in full below) rather than
assumed from the shape of the CFM call:

    embedding = F.normalize(embedding, dim=1)
    embedding = self.spk_embed_affine_layer(embedding)          # xvec: normalize BEFORE the linear, not after

    token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
    mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
    token = self.input_embedding(torch.clamp(token, min=0)) * mask

    h, h_lengths = self.encoder(token, token_len, streaming=streaming)   # finalize=True branch
    mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
    h = self.encoder_proj(h)

    conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], ...)
    conds[:, :mel_len1] = prompt_feat                            # real prompt mel spliced in, rest left at 0
    conds = conds.transpose(1, 2)

    feat, _ = self.decoder(mu=h.transpose(1, 2).contiguous(), mask=..., spks=embedding,
                            cond=conds, n_timesteps=10, streaming=streaming)
    feat = feat[:, :, mel_len1:]                                 # only the NEWLY generated tail is returned

Three details this depends on, each verified against the quoted body above rather
than assumed:

* **`mel_len1` comes from `prompt_feat.shape[1]` directly, not from
  `token_mel_ratio * prompt_token_len`.** `mel_len2` is then whatever is left over
  (`h.shape[1] - mel_len1`) -- a subtraction, not a second multiplication. A
  well-formed call has `prompt_feat.shape[1] == token_mel_ratio * prompt_token_len`,
  but the method itself never assumes that equality, so this port doesn't either.
* **`conds`' un-spliced tail is exactly zero**, not the encoder's own `h` or
  anything derived from it -- the CFM's `cond` channel carries "known prompt audio,
  silence elsewhere," which is what lets a single CFM call generate a continuation
  that is acoustically consistent with the prompt.
* **The xvec affine layer's output is `spks` directly** (`self.output_size`-wide,
  80 here) -- there is no further normalisation after the linear, only before it.

`conds`' `.transpose(1, 2)` (channel-first, matching upstream's `[B, C, T]`
convention) is a no-op in this port: this package stays channels-last (`[N, L, C]`)
throughout, so `conds` is built directly in that shape and fed to
`TtCausalConditionalCFM` unchanged -- same note as tt/flow/decoder.py's module
docstring makes for `mu`/`x`/`cond`.

`input_embedding` (`nn.Embedding(vocab_size, input_size)`, mapping the same
speech-token vocabulary the LLM emits to this module's own 512-dim space) is a
SEPARATE, independently-learned table from `TtQwen2LM.speech_embedding` in
`tt/llm/qwen2lm.py` -- same vocabulary, different subsystem, different weights.
Reused here is only `TtSmallEmbedding` itself (a generic `ttnn.embedding` wrapper,
not LLM-specific despite living in that file), not any LLM state.

No CosyVoice2 checkpoint is available yet, so `input_embedding`/
`spk_embed_affine_layer`/`encoder_proj` are randomly initialised here, matching
every other component in this package.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

from ..llm.qwen2lm import TtSmallEmbedding
from .decoder import (
    CausalConditionalCFMRef,
    CausalConditionalDecoderRef,
    TtCausalConditionalCFM,
    TtCausalConditionalDecoder,
)
from .encoder import TtUpsampleConformerEncoder, UpsampleConformerEncoderRef

INPUT_SIZE = 512  # encoder's own d_model
OUTPUT_SIZE = 80  # mel channels
SPK_EMBED_DIM = 192  # real xvec width
VOCAB_SIZE = 6561  # speech_token_size, confirmed from cosyvoice2.yaml's flow.vocab_size
N_TIMESTEPS = 10  # hardcoded in the real inference() call, not a yaml/config parameter


# ---------------------------------------------------------------------------
# torch reference
# ---------------------------------------------------------------------------


class CausalMaskedDiffWithXvecRef(nn.Module):
    """`cosyvoice.flow.flow.CausalMaskedDiffWithXvec`, `inference`'s `finalize=True`
    branch only (`streaming=False` throughout this phase -- see module docstring)."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
        spk_embed_dim: int = SPK_EMBED_DIM,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.output_size = output_size
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = UpsampleConformerEncoderRef(d_model=input_size)
        self.encoder_proj = nn.Linear(input_size, output_size)
        estimator = CausalConditionalDecoderRef()
        self.decoder = CausalConditionalCFMRef(estimator)

    def inference(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """token/prompt_token: [1, N] speech token ids. prompt_feat: [1, mel_len1,
        80] real prompt mel features. embedding: [1, 192] raw xvec. Returns
        [1, mel_len2, 80] -- only the newly-generated tail (see module docstring
        for why `mel_len1`/`mel_len2` come from `prompt_feat.shape[1]` and
        subtraction, not `token_mel_ratio`).
        """
        assert token.shape[0] == 1
        spks = self.spk_embed_affine_layer(F.normalize(embedding, dim=1))

        full_token = torch.cat([prompt_token, token], dim=1)
        tok_emb = self.input_embedding(full_token.clamp(min=0))

        h = self.encoder(tok_emb)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        conds = torch.zeros(1, mel_len1 + mel_len2, self.output_size, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat

        mask = torch.ones(1, mel_len1 + mel_len2, 1, dtype=h.dtype)
        feat = self.decoder.forward(mu=h, mask=mask, n_timesteps=N_TIMESTEPS, spks=spks, cond=conds)
        feat = feat[:, mel_len1:, :]
        assert feat.shape[1] == mel_len2
        return feat


# ---------------------------------------------------------------------------
# TTNN port. [N, L, C] throughout -- `conds`' real-source `.transpose(1, 2)` is a
# no-op here, see module docstring.
# ---------------------------------------------------------------------------


def _linear_weight(device, weight: torch.Tensor, dtype):
    return ttnn.from_torch(
        weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def _bias(device, bias: torch.Tensor, dtype):
    return ttnn.from_torch(bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class TtCausalMaskedDiffWithXvec:
    def __init__(self, device, module: CausalMaskedDiffWithXvecRef, dtype=ttnn.bfloat16):
        self.device = device
        self.dtype = dtype
        self.output_size = module.output_size
        self.input_embedding = TtSmallEmbedding(device, module.input_embedding.weight, dtype=dtype)
        self.spk_w = _linear_weight(device, module.spk_embed_affine_layer.weight, dtype)
        self.spk_b = _bias(device, module.spk_embed_affine_layer.bias, dtype)
        self.encoder = TtUpsampleConformerEncoder(device, module.encoder, dtype=dtype)
        self.encoder_proj_w = _linear_weight(device, module.encoder_proj.weight, dtype)
        self.encoder_proj_b = _bias(device, module.encoder_proj.bias, dtype)
        self.decoder = TtCausalConditionalCFM(
            device,
            TtCausalConditionalDecoder(device, module.decoder.estimator, dtype=dtype),
            module.decoder.rand_noise,
            module.decoder,
        )

    def _xvec(self, embedding: torch.Tensor) -> torch.Tensor:
        """L2-normalise then project, on device -- the same
        normalize/sum/rsqrt/multiply idiom this bring-up already uses for a unit
        vector (SourceModuleHnNSF-adjacent code); returns a torch tensor (small,
        [1, 80]) since `TtCausalConditionalCFM.forward` takes its conditioning
        tensors as torch (see tt/flow/decoder.py)."""
        emb_dev = ttnn.from_torch(
            embedding.reshape(1, 1, -1).float(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        sq = ttnn.multiply(emb_dev, emb_dev)
        s = ttnn.sum(sq, dim=-1, keepdim=True)
        inv = ttnn.rsqrt(s)
        unit = ttnn.multiply(emb_dev, inv)
        spks = ttnn.linear(unit, self.spk_w, bias=self.spk_b)
        return ttnn.to_torch(spks).float().reshape(1, -1)

    def inference(
        self, token: torch.Tensor, prompt_token: torch.Tensor, prompt_feat: torch.Tensor, embedding: torch.Tensor
    ):
        """Same signature/semantics as `CausalMaskedDiffWithXvecRef.inference` --
        see its docstring. All host-facing args/return are torch tensors; device
        tensors stay internal."""
        assert token.shape[0] == 1
        spks = self._xvec(embedding)

        full_token = torch.cat([prompt_token, token], dim=1)
        ids_dev = ttnn.from_torch(
            full_token.reshape(1, 1, 1, -1).clamp(min=0).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        tok_emb_dev = self.input_embedding(ids_dev)

        h_dev = self.encoder(tok_emb_dev, full_token.shape[1], 1)
        t_len2 = h_dev.shape[1]
        mel_len1, mel_len2 = prompt_feat.shape[1], t_len2 - prompt_feat.shape[1]
        h_dev = ttnn.linear(h_dev, self.encoder_proj_w, bias=self.encoder_proj_b)
        mu = ttnn.to_torch(h_dev).float().reshape(1, t_len2, self.output_size)

        conds = torch.zeros(1, mel_len1 + mel_len2, self.output_size, dtype=mu.dtype)
        conds[:, :mel_len1] = prompt_feat
        mask = torch.ones(1, mel_len1 + mel_len2, 1, dtype=mu.dtype)

        feat = self.decoder.forward(mu, mask, N_TIMESTEPS, spks, conds)
        feat = feat[:, mel_len1:, :]
        assert feat.shape[1] == mel_len2
        return feat
