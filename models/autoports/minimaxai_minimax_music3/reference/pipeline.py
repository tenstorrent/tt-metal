"""Music3Reference: the whole MiniMax Music 3 inference recipe in torch (CPU/fp32 by default), flattened from the
diffusers modular pipeline (encoders.py -> before_denoise.py -> denoise.py -> decoders.py). Same generator
consumption order as diffusers, so `generate(seed)` reproduces the diffusers pipeline's codes and audio.

Also provides `teacher_forced()` (replay golden codes, dump every intermediate the TT tests compare against) and
`denoise()` / `decode()` for the flow-matching + vocoder stages, which serve as the CPU parts of the serving path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from models.autoports.minimaxai_minimax_music3.config import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    AUDIO_VOCAB_SIZE,
    CHUNK_FRAMES,
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    DIT_GUIDANCE_SCALE,
    DIT_NUM_STEPS,
    FRAME_RATE,
    LATENT_CHANNELS,
    MAX_AUDIO_FRAMES,
    NUM_CODEBOOKS,
    OVERLAP_LATENT_LENGTH,
    Music3Config,
    chunk_starts,
)
from models.autoports.minimaxai_minimax_music3.reference.condition_encoder import Music3ConditionEncoder
from models.autoports.minimaxai_minimax_music3.reference.depth_decoder import Music3DepthDecoder
from models.autoports.minimaxai_minimax_music3.reference.dit import Music3DiT
from models.autoports.minimaxai_minimax_music3.reference.prompt import Music3Tokenizer, build_text_ids
from models.autoports.minimaxai_minimax_music3.reference.sampling import (
    full_vocab_mask,
    guided_c0_logits_full,
    guided_depth_logits,
    sample_top_k,
    slice_logits,
)
from models.autoports.minimaxai_minimax_music3.reference.vocoder import Music3Vocoder


def flow_schedule(num_steps: int = DIT_NUM_STEPS):
    """FlowMatchEulerDiscreteScheduler(invert_sigmas=True, shift=1, num_train_timesteps=1).set_timesteps(
    sigmas=linspace(1, 1/N, N)) -> (timesteps [N], sigmas [N+1]). t runs 0 -> (N-1)/N, terminal sigma 1.0."""
    sig = torch.from_numpy(np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32))
    sig = 1.0 - sig
    timesteps = sig.clone()
    sigmas = torch.cat([sig, torch.ones(1)])
    return timesteps, sigmas


def euler_step(
    latents: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor
) -> torch.Tensor:
    """scheduler.step(): fp32 update, cast back to the model dtype."""
    dtype = velocity.dtype
    prev = latents.to(torch.float32) + (sigma_next - sigma) * velocity.to(torch.float32)
    return prev.to(dtype)


class Music3Reference:
    def __init__(
        self,
        snapshot,
        dtype=torch.float32,
        device="cpu",
        *,
        load_llm: bool = True,
        load_dit: bool = True,
        load_vocoder: bool = True,
        llm_dtype=None,
        log=print,
    ):
        self.snapshot = Path(snapshot)
        self.cfg = Music3Config.from_snapshot(snapshot)
        self.dtype, self.device, self.log = dtype, device, log
        self.tok = Music3Tokenizer(snapshot)
        t0 = time.time()
        self.llm = self.depth = self.dit = self.vocoder = None
        if load_llm:
            from transformers import Qwen3ForCausalLM

            self.llm = (
                Qwen3ForCausalLM.from_pretrained(str(self.snapshot / "language_model"), torch_dtype=llm_dtype or dtype)
                .to(device)
                .eval()
            )
            self.depth = Music3DepthDecoder.load(snapshot, dtype=llm_dtype or dtype, device=device, cfg=self.cfg.depth)
        self.cond = Music3ConditionEncoder.load(snapshot, dtype=dtype, device=device, cfg=self.cfg.cond)
        if load_dit:
            self.dit = Music3DiT.load(snapshot, dtype=dtype, device=device, cfg=self.cfg.dit)
        if load_vocoder:
            self.vocoder = Music3Vocoder.load(snapshot, dtype=dtype, device=device, cfg=self.cfg.vocoder)
        log(
            f"Music3Reference loaded in {time.time() - t0:.1f}s (llm={load_llm} dit={load_dit} vocoder={load_vocoder}, dtype={dtype})"
        )

    # ------------------------------------------------------------------ prompt
    def text_ids(self, caption: str, lyrics: str) -> torch.Tensor:
        return build_text_ids(self.tok, caption, lyrics).to(self.device)

    # ------------------------------------------------------------------ autoregressive stage
    def embed_audio_frame(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """frame_codes [2, 8] -> [2, 1, D] (semantic embedding + summed residual embeddings, scaled by 8^-0.5)."""
        embed_tokens = self.llm.model.embed_tokens
        embeds = embed_tokens(frame_codes[:, :1] + AUDIO_CODE_OFFSET)
        offsets = (torch.arange(NUM_CODEBOOKS - 1, device=frame_codes.device) * AUDIO_VOCAB_SIZE).unsqueeze(0)
        extra = self.depth.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        return (embeds + extra.to(embeds.dtype)) * NUM_CODEBOOKS**-0.5

    def depth_codes(
        self,
        last_hidden: torch.Tensor,
        semantic_code: torch.Tensor,
        generator,
        forced: Optional[Sequence[int]] = None,
        record: Optional[dict] = None,
    ):
        """One frame of the local LLM. last_hidden [2, D] (cond, uncond), semantic_code [2].
        Returns (frame_codes [2, 8], depth_hidden [1, 7*D]). `forced` = codes c1..c7 to use instead of sampling."""
        d = self.depth
        sequence = [d.projection(last_hidden).unsqueeze(1)]
        code_embed = self.llm.model.embed_tokens(semantic_code + AUDIO_CODE_OFFSET)
        sequence.append(d.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, NUM_CODEBOOKS):
            hidden = d(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = d.audio_heads[index - 1](hidden)
            if record is not None:
                record.setdefault("depth_logits", []).append(logits.detach().float().cpu())
                record.setdefault("depth_hidden", []).append(hidden[:1].detach().float().cpu())
            if forced is not None:
                code = torch.tensor([int(forced[index - 1])], device=hidden.device).repeat(2)
            else:
                code = sample_top_k(guided_depth_logits(logits), generator).repeat(2)
            codes.append(code)
            if index < NUM_CODEBOOKS - 1:
                embed = d.audio_embeddings(code + (index - 1) * AUDIO_VOCAB_SIZE)
                sequence.append(d.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    @torch.no_grad()
    def semantic_generation(
        self,
        text_ids: torch.Tensor,
        max_frames: int,
        generator,
        forced_codes: Optional[torch.Tensor] = None,
        record: bool = False,
    ):
        """The autoregressive stage. Returns dict(frame_hiddens [1, F, 8D], codes_all [N, 8] (row 0 = the non-emitted
        pre-frame), ended (bool), + dumps when record=True: hidden_all [N(+1), 2, D], c0_logits [N(+1), 2, 16385],
        depth_logits [N, 7, 2, 1024], depth_hidden [N, 7, D]).
        forced_codes [N, 8]: teacher forcing (c0 + c1..c7 taken from the golden; nothing is sampled)."""
        llm = self.llm
        max_frames = min(int(max_frames), MAX_AUDIO_FRAMES)
        text_embeds = llm.model.embed_tokens(text_ids)
        out = llm.model(inputs_embeds=text_embeds, use_cache=True)
        past, last_hidden = out.past_key_values, out.last_hidden_state[:, -1]
        vocab_mask = full_vocab_mask(llm.config.vocab_size, device=text_ids.device)
        frame_hiddens, codes_all, ended = [], [], False
        dumps: Dict[str, list] = {"hidden_all": [], "c0_logits": [], "depth_logits": [], "depth_hidden": []}
        n_steps = max_frames + 1 if forced_codes is None else forced_codes.shape[0]
        for frame_index in range(n_steps):
            logits = llm.lm_head(last_hidden).float()
            if record:
                dumps["hidden_all"].append(last_hidden.detach().float().cpu())
                dumps["c0_logits"].append(slice_logits(logits).detach().cpu())
            if forced_codes is None:
                guided = guided_c0_logits_full(logits, vocab_mask)
                sampled = sample_top_k(guided, generator)
                if int(sampled.item()) == AUDIO_END_TOKEN_ID:
                    ended = True
                    break
                semantic_code = sampled - AUDIO_CODE_OFFSET
                forced = None
            else:
                row = forced_codes[frame_index]
                semantic_code = row[:1].to(text_ids.device)
                forced = row[1:].tolist()
            rec = {} if record else None
            frame_codes, depth_hidden = self.depth_codes(
                last_hidden, semantic_code.repeat(2), generator, forced=forced, record=rec
            )
            if record:
                dumps["depth_logits"].append(torch.stack(rec["depth_logits"]))  # [7, 2, 1024]
                dumps["depth_hidden"].append(torch.cat(rec["depth_hidden"]))  # [7, D]
            codes_all.append(frame_codes[0].detach().cpu())
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if forced_codes is None and len(frame_hiddens) >= max_frames:
                    break
            if forced_codes is not None and frame_index == n_steps - 1:
                break
            feedback = self.embed_audio_frame(frame_codes)
            out = llm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
            past, last_hidden = out.past_key_values, out.last_hidden_state[:, -1]
        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")
        res = {
            "frame_hiddens": torch.stack(frame_hiddens, dim=1),
            "codes_all": torch.stack(codes_all) if codes_all else torch.zeros(0, NUM_CODEBOOKS, dtype=torch.int64),
            "ended": ended,
            "prompt_len": int(text_ids.shape[1]),
        }
        if record:
            res.update({k: torch.stack(v) for k, v in dumps.items() if v})
        return res

    # ------------------------------------------------------------------ flow matching
    def condition_for(self, frame_hiddens: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return self.cond(frame_hiddens[:, start:end].to(self.device)).to(
            self.dit.dtype if self.dit is not None else self.dtype
        )

    @torch.no_grad()
    def denoise(
        self,
        frame_hiddens: torch.Tensor,
        generator,
        num_steps: int = DIT_NUM_STEPS,
        guidance_scale: float = DIT_GUIDANCE_SCALE,
        record_steps: Sequence[int] = (),
        dit_forward=None,
    ) -> dict:
        """before_denoise.py + denoise.py. Returns dict(latent_chunks [list of [1,128,L]], chunk_starts, and when
        record_steps: per-chunk dicts with condition / noise / timesteps / (t, latents_in, pred_cond, pred_uncond)).
        `dit_forward(latents [B,128,L], t [B], cond [B,L,2048]) -> velocity` lets a TT DiT drive the same loop."""
        dit_forward = dit_forward or (lambda x, t, c: self.dit(x, t, c))
        starts = chunk_starts(frame_hiddens.shape[1])
        timesteps, sigmas = flow_schedule(num_steps)
        latent_chunks, chunk_records = [], []
        prev_latent = prev_cond = None
        for k, start in enumerate(starts):
            end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
            condition = self.condition_for(frame_hiddens, start, end)
            overlap = 0
            if prev_latent is not None:
                overlap = min(prev_latent.shape[-1], condition.shape[1])
                condition[:, :overlap] = prev_cond[:, :overlap]
            latents = torch.randn(
                (1, LATENT_CHANNELS, condition.shape[1]), generator=generator, dtype=condition.dtype
            ).to(self.device)
            noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None
            rec = (
                {
                    "chunk_start": start,
                    "condition": condition.detach().cpu(),
                    "noise": latents.detach().cpu(),
                    "overlap": overlap,
                    "timesteps": timesteps.clone(),
                    "steps": [],
                }
                if record_steps
                else None
            )
            for i, t in enumerate(timesteps):
                if overlap > 0:
                    tv = t.to(latents.dtype)
                    latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * tv) * noise_prompt + tv * prev_latent[..., :overlap]
                timestep = t.expand(latents.shape[0]).to(latents.dtype).to(self.device)
                latents_in = latents.detach().clone() if rec is not None and i in record_steps else None
                pred_cond = dit_forward(latents, timestep, condition)
                pred_uncond = dit_forward(latents, timestep, torch.zeros_like(condition))
                velocity = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                if latents_in is not None:
                    rec["steps"].append(
                        {
                            "i": i,
                            "t": float(t),
                            "latents_in": latents_in.cpu(),
                            "pred_cond": pred_cond.detach().cpu(),
                            "pred_uncond": pred_uncond.detach().cpu(),
                        }
                    )
                latents = euler_step(latents, velocity, sigmas[i], sigmas[i + 1])
            if overlap > 0:
                latents[..., :overlap] = prev_latent[..., :overlap]
            os_ = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
            oe = max(os_, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
            prev_latent, prev_cond = latents[..., os_:oe], condition[:, os_:oe]
            latent_chunks.append(latents)
            if rec is not None:
                rec["latents_out"] = latents.detach().cpu()
                chunk_records.append(rec)
        return {"latent_chunks": latent_chunks, "chunk_starts": starts, "chunk_records": chunk_records}

    @torch.no_grad()
    def decode(self, latent_chunks: List[torch.Tensor], vocoder=None) -> torch.Tensor:
        """decoders.py: vocode each window, crop overlaps, stitch -> [1, 2, samples] float32 in [-1, 1] at 44.1 kHz."""
        vocoder = vocoder or self.vocoder
        hop = vocoder.hop_length
        n = len(latent_chunks)
        chunks = []
        for i, lat in enumerate(latent_chunks):
            wav = vocoder(lat.to(vocoder.dec_in_proj.weight.dtype).to(self.device))
            left = 0 if i == 0 else CROP_LEFT_LATENT * hop
            right = 0 if i == n - 1 else CROP_RIGHT_LATENT * hop
            chunks.append(wav[..., left : wav.shape[-1] - right])
        return torch.cat(chunks, dim=-1).float().clamp(-1.0, 1.0).cpu()

    # ------------------------------------------------------------------ end to end
    @torch.no_grad()
    def generate(
        self,
        caption: str,
        lyrics: str,
        *,
        audio_duration: float = 60.0,
        seed: int = 0,
        num_steps: int = DIT_NUM_STEPS,
        record: bool = False,
        record_steps: Sequence[int] = (),
    ) -> dict:
        generator = torch.Generator("cpu").manual_seed(int(seed))
        text_ids = self.text_ids(caption, lyrics)
        max_frames = min(int(audio_duration * FRAME_RATE), MAX_AUDIO_FRAMES)
        t0 = time.time()
        sem = self.semantic_generation(text_ids, max_frames, generator, record=record)
        t1 = time.time()
        den = self.denoise(sem["frame_hiddens"], generator, num_steps=num_steps, record_steps=record_steps)
        t2 = time.time()
        audio = self.decode(den["latent_chunks"])
        t3 = time.time()
        return {
            **sem,
            **den,
            "text_ids": text_ids.cpu(),
            "audio": audio,
            "sample_rate": self.cfg.vocoder.sampling_rate,
            "timing": {"semantic_s": t1 - t0, "denoise_s": t2 - t1, "decode_s": t3 - t2},
        }

    @torch.no_grad()
    def teacher_forced(self, text_ids: torch.Tensor, codes_all: torch.Tensor) -> dict:
        """Replay golden codes (N x 8, row 0 = pre-frame) and dump every intermediate. No sampling."""
        return self.semantic_generation(text_ids.to(self.device), 0, None, forced_codes=codes_all, record=True)
