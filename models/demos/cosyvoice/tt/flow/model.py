# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MaskedDiffWithXvec: semantic tokens -> mel spectrogram, end to end on device.

The whole of `02_plan.md` P3 assembled:

    tokens ---------------> input_embedding (4096 x 512)
                        --> ConformerEncoder, 6 blocks, rel-pos attention
                        --> encoder_proj (512 -> 80)
                        --> InterpolateRegulator: token rate -> mel rate
    speaker embedding ----> L2 normalise, affine (192 -> 80)
    prompt mel -----------> cond, zero-padded to the full length
                        --> ConditionalCFM: 10 Euler steps of the UNet
                        --> drop the prompt frames

Every stage is a TTNN module. Nothing round-trips to the host between the token
IDs going in and the mel coming out.

The prompt is carried in three different ways at once, which is worth stating
because it looks redundant: its *tokens* are prepended to the semantic sequence
before the encoder, its *mel* is supplied as `cond` so the estimator can condition
on it directly, and its *length* decides where the generated mel starts. Only the
first `mel_len1` frames are dropped at the end.
"""
from __future__ import annotations

import torch

import ttnn

from .cfm import TtConditionalCFM
from .encoder import TtConformerEncoder, espnet_rel_positional_encoding
from .estimator import _linear
from .length_regulator import TtInterpolateRegulator


class TtMaskedDiffWithXvec:
    """The flow stage. Activations are channels-last `[B, T, C]` throughout."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16):
        self.device, self.dtype, self.meta = device, dtype, meta
        self.output_size = meta.get("output_size", 80)
        self.input_frame_rate = meta.get("input_frame_rate", 50)

        self.token_embedding = ttnn.from_torch(
            bag.tensor("input_embedding.weight"), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.encoder = TtConformerEncoder(device, bag.sub("encoder"), meta, dtype)
        self.proj_w, self.proj_b = _linear(device, bag, "encoder_proj", dtype)
        self.spk_w, self.spk_b = _linear(device, bag, "spk_embed_affine_layer", dtype)
        self.length_regulator = TtInterpolateRegulator(device, bag.sub("length_regulator"), self.output_size, dtype)
        self.decoder = TtConditionalCFM(
            device,
            bag.sub("decoder"),
            inference_cfg_rate=meta.get("inference_cfg_rate", 0.7),
            n_timesteps=meta.get("n_timesteps", 10),
            t_scheduler=meta.get("t_scheduler", "cosine"),
            dtype=dtype,
        )

    # ----------------------------------------------------------------------
    def speaker_embedding(self, embedding):
        """`F.normalize(e, dim=1)` then the affine. `[1, 1, 192]` -> `[1, 1, 80]`.

        `F.normalize` divides by `max(||e||, 1e-12)`; the rsqrt form here differs
        only when the norm underflows, which a real x-vector never does.
        """
        sq = ttnn.multiply(embedding, embedding)
        s = ttnn.sum(sq, dim=-1, keepdim=True)
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(s)
        ttnn.deallocate(s)
        unit = ttnn.multiply(embedding, inv)
        ttnn.deallocate(inv)
        out = ttnn.linear(unit, self.spk_w, bias=self.spk_b)
        ttnn.deallocate(unit)
        return out

    def encode_tokens(self, tokens):
        """Token IDs `[1, T]` -> `[1, T, 80]`, i.e. mu at *token* rate.

        `input_embedding(...) * mask` upstream: the mask is all-ones for a single
        utterance, so the product is dropped rather than carried.
        """
        emb = ttnn.embedding(tokens, self.token_embedding, layout=ttnn.TILE_LAYOUT)
        pos = espnet_rel_positional_encoding(emb.shape[1], self.meta["d_model"])
        pe = ttnn.from_torch(pos, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        h = self.encoder(emb, pe)
        ttnn.deallocate(emb)
        ttnn.deallocate(pe)
        out = ttnn.linear(h, self.proj_w, bias=self.proj_b)
        ttnn.deallocate(h)
        return out

    def regulate(self, h, token_len1: int, mel_len1: int, mel_len2: int):
        """Resample prompt and generated segments separately, then the conv stack.

        The two halves are resampled independently -- and the generated half is
        further split head/mid/tail -- so that the mel has a clean seam at
        `mel_len1`. That seam is what lets the prompt frames be dropped at the end
        without cutting into generated audio.
        """
        c = h.shape[-1]
        total = h.shape[1]
        if token_len1 == 0:
            # `sft` and `instruct` synthesise from a speaker id with no prompt audio,
            # so there is no first segment and no seam to protect. Taking the general
            # path anyway would slice a zero-length tensor and concat it, which
            # **segfaults inside `ttnn::concat`** rather than raising -- a crash the
            # caller's try/except cannot catch, several frames from the empty tensor
            # that caused it.
            cat = self.length_regulator.resample_split(h, total, mel_len2, self.input_frame_rate)
        else:
            x1 = ttnn.slice(h, [0, 0, 0], [1, token_len1, c])
            x2 = ttnn.slice(h, [0, token_len1, 0], [1, total, c])
            r1 = self.length_regulator.resample(x1, token_len1, mel_len1)
            r2 = self.length_regulator.resample_split(x2, total - token_len1, mel_len2, self.input_frame_rate)
            ttnn.deallocate(x1)
            ttnn.deallocate(x2)
            cat = ttnn.concat([r1, r2], dim=1)
            ttnn.deallocate(r1)
            ttnn.deallocate(r2)
        out = self.length_regulator(cat, mel_len1 + mel_len2)
        ttnn.deallocate(cat)
        return out

    def conditions(self, prompt_feat, mel_len1: int, mel_len2: int):
        """`cond` is the prompt mel followed by zeros for the part being generated."""
        zeros = ttnn.zeros(
            (1, mel_len2, self.output_size), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        if mel_len1 == 0:
            # No prompt mel at all (`sft`, `instruct`): the condition is all zeros, and
            # concatenating the empty `[1, 0, 80]` prompt onto it segfaults in
            # `ttnn::concat`. See `regulate` for the same guard on the token side.
            return zeros
        out = ttnn.concat([prompt_feat, zeros], dim=1)
        ttnn.deallocate(zeros)
        return out

    # ----------------------------------------------------------------------
    def inference(self, tokens, token_len1: int, mel_len1: int, mel_len2: int, prompt_feat, embedding, z):
        """tokens `[1, T]` uint32 (prompt tokens first); prompt_feat `[1, mel_len1, 80]`;
        embedding `[1, 1, 192]`; z `[1, mel_len1+mel_len2, 80]` -> `[1, mel_len2, 80]`.

        `z` is the initial noise. It is an argument rather than something drawn
        here because a device RNG cannot be aligned with torch's stream -- see
        `tt/flow/cfm.py`.
        """
        spks = self.speaker_embedding(embedding)
        mu = self.encode_tokens(tokens)
        mu = self.regulate(mu, token_len1, mel_len1, mel_len2)
        cond = self.conditions(prompt_feat, mel_len1, mel_len2)

        feat = self.decoder.solve_euler(z, mu, spks, cond)
        ttnn.deallocate(mu)
        ttnn.deallocate(cond)
        ttnn.deallocate(spks)

        total = mel_len1 + mel_len2
        if mel_len1 == 0:
            # Nothing to trim: with no prompt mel the whole solve is the output. The
            # slice below would be full-extent, and **a full-extent `ttnn.slice` is an
            # alias, not a copy** -- so the `deallocate` after it would free the tensor
            # being returned, and the caller would fault on "Tensor is not allocated"
            # one stage later. That aliasing is the same behaviour trace integration
            # ran into; here it only shows up in `sft` and `instruct`, the two modes
            # with no prompt audio.
            return feat
        out = ttnn.slice(feat, [0, mel_len1, 0], [1, total, self.output_size])
        ttnn.deallocate(feat)
        return out

    # ----------------------------------------------------------------------
    @staticmethod
    def mel_len_for(token_len: int, input_frame_rate: int = 50, sample_rate: int = 22050, hop: int = 256) -> int:
        """How many mel frames `token_len` semantic tokens become.

        Truncating, not rounding: `int(...)` upstream. At 50 Hz tokens and a 256
        hop at 22.05 kHz this is `token_len * 1.7226...`, so the fractional part is
        discarded on nearly every utterance.
        """
        return int(token_len / input_frame_rate * sample_rate / hop)

    @staticmethod
    def torch_speaker_embedding(embedding: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(torch.nn.functional.normalize(embedding, dim=1), w, b)
