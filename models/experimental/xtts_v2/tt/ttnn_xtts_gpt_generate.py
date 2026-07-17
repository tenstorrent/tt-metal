# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end greedy generation for the XTTS-v2 GPT (Block 3), host-orchestrated.

Wraps TTNNGPTDecoder (KV-cached decode on device) with the generation head kept on the
host (CPU): the mel_head (latent -> logits), mel_embedding + mel_pos (embed the sampled
code), matching the block boundary (transformer core on TT; embeddings/head on CPU).

Flow (host-driven loop, one device dispatch per token):
  prefill [prompt, START] into the KV cache
  repeat: latent = decode_step(x); logits = mel_head(latent); code = argmax(logits);
          stop if code == stop_token; else x = mel_embedding[code] + mel_pos[m]

`generate` free-runs on its own samples; `teacher_forced` replays given per-step inputs
(used to validate against the CPU reference without argmax-flip cascades).
"""

import torch
import ttnn

from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import TTNNGPTConfig
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTDecoder


class TTNNGPTGenerator:
    def __init__(
        self,
        device,
        parameters,
        heads,
        config: TTNNGPTConfig = None,
        max_seq: int = 256,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        start_token: int = 1024,
        stop_token: int = 1025,
    ):
        self.device = device
        self.decoder = TTNNGPTDecoder(device, parameters, config, math_fidelity=math_fidelity, max_seq=max_seq)
        # Generation head on host (CPU torch tensors).
        self.mel_emb = heads["mel_emb"]  # [1026,1024]
        self.mel_pos = heads["mel_pos"]  # [608,1024]
        self.mh_w = heads["mel_head_w"]  # [1026,1024]
        self.mh_b = heads["mel_head_b"]  # [1026]
        self.start_token = start_token
        self.stop_token = stop_token

    def _to_dev(self, t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _latent(self, emb_torch):  # [1,1,1024] torch -> [1,1,1024] torch
        lt = self.decoder.decode_step(self._to_dev(emb_torch))
        return ttnn.to_torch(lt).to(torch.float32)

    def _logits(self, latent):  # [1,1,1024] -> [1,1,1026]
        return latent @ self.mh_w.t() + self.mh_b

    def prefill(self, prompt_embeds):
        """Fill the KV cache with the prompt (token by token). [1,P,1024]."""
        self.decoder.reset()
        for p in range(prompt_embeds.shape[1]):
            self.decoder.decode_step(self._to_dev(prompt_embeds[:, p : p + 1, :].contiguous()))

    def generate(self, prompt_embeds, max_new=24):
        """Free-running greedy generation. Returns (codes list, latents [1,T,1024])."""
        self.prefill(prompt_embeds)
        codes, latents = [], []
        emb = (self.mel_emb[self.start_token] + self.mel_pos[0]).view(1, 1, -1)
        for m in range(max_new):
            latent = self._latent(emb)
            code = int(self._logits(latent).argmax(-1))
            codes.append(code)
            latents.append(latent)
            if code == self.stop_token:
                break
            emb = (self.mel_emb[code] + self.mel_pos[m + 1]).view(1, 1, -1)
        return codes, torch.cat(latents, dim=1)

    def teacher_forced(self, prompt_embeds, step_inputs):
        """Replay reference per-step inputs [1,T,1024]. Returns (logits [1,T,1026],
        latents [1,T,1024]) — isolates transformer+head numerics from sampling cascades."""
        self.prefill(prompt_embeds)
        logits, latents = [], []
        for m in range(step_inputs.shape[1]):
            latent = self._latent(step_inputs[:, m : m + 1, :].contiguous())
            latents.append(latent)
            logits.append(self._logits(latent))
        return torch.cat(logits, dim=1), torch.cat(latents, dim=1)
