# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The sampling loop below (vocabulary mask, CFG on the logits, restriction to the conditional row's
# top-k, top-k multinomial sampling, depth-code CFG, feedback embedding, frame bookkeeping) is a
# transcription of diffusers ``src/diffusers/modular_pipelines/minimax_music3/encoders.py``
# (``_sample_top_k``, ``_generate_depth_codes``, ``MiniMaxMusic3AutoregressiveStep``), Apache-2.0,
# Copyright 2026 The MiniMax Team and The HuggingFace Team.
"""MiniMax-Music3 autoregressive stage on one Blackhole chip: prompt -> per-frame hidden states.

``ARGenerator`` drives the stage-02 ``MusicLLM`` (Qwen3 backbone, traced decode) and the stage-03
``DepthDecoder`` (traced depth steps) exactly like diffusers' ``MiniMaxMusic3TokenizeStep`` +
``MiniMaxMusic3AutoregressiveStep``:

1. prompt + lyrics -> ``text_ids [2, L]`` (row 0 conditional, row 1 the CFG prompt);
2. prefill from the token embeddings;
3. per frame: CFG (scale 1.5) over the backbone logits restricted to the end token + 16384 semantic
   codes, top-50 restriction on the conditional row, top-50 multinomial sampling with a CPU
   ``torch.Generator``; end token -> stop; otherwise the depth decoder samples the seven residual
   codes (CFG 1.5, top-50) and the frame's hidden ``cat(last_hidden[:1], depth_hiddens)`` is
   emitted (frame 0 only advances past ``<|audio_start|>``);
4. feedback ``(embed(sem + offset) + sum_k audio_embeddings(c_k + (k - 1) * 1024)) * 8**-0.5``
   into the next decode step.

Host / device split (decided for this stage, see ``doc/ar_generator/README.md``): both traces run
on device; the feedback embedding (two table gathers of 4096 values) and the sampling run on the
host. The backbone trace slices the end-token / semantic-code window out of the logits on device so
only ``2 x 16389`` values are read back per frame, not the 200k vocabulary. Per frame the device
allocates nothing: all persistent buffers exist before the traces are captured (backbone decode
inputs first, then the depth trace buffers, then the backbone trace itself on the first decode).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.constants import (
    AR_CFG_SCALE,
    AR_CFG_TOP_K,
    AR_SAMPLING_TOP_K,
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    AUDIO_VOCAB_SIZE,
    LLM_BATCH,
    LLM_HIDDEN,
    MAX_AUDIO_FRAMES,
    NUM_CODEBOOKS,
    SEMANTIC_VOCAB_SIZE,
)
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DepthDecoder, DepthStepTrace
from models.autoports.minimaxai_minimax_music3.tt.llm import MusicLLM
from models.autoports.minimaxai_minimax_music3.tt.prompt import PromptEncoder

# The logits window the AR loop needs: [end token, ..., last semantic code].
WINDOW_START = AUDIO_END_TOKEN_ID
WINDOW_END = AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE
FRAME_HIDDEN = NUM_CODEBOOKS * LLM_HIDDEN  # 32768


def sample_top_k(logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """diffusers ``_sample_top_k``: top-50 (ties kept) multinomial sample per row, drawn on the generator's device.

    Literal transcription (the multinomial runs over the whole row, so the draw for a given seed depends on the
    row length); the AR loop uses :func:`sample_top_k_candidates` instead (stage 07)."""
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


def sample_top_k_candidates(logits: torch.Tensor, generator: Optional[torch.Generator]) -> int:
    """``_sample_top_k`` for one row with the same candidate set and probabilities, but the multinomial drawn over
    the candidates only (stage 07: the "final 50-way multinomial on host" of the stage prompt).

    Same semantics as :func:`sample_top_k`: ``nan_to_num``, every value >= the 50th largest is a candidate (ties
    kept), softmax over the candidates, one multinomial draw. Only the random-number consumption differs (one
    draw over <= 50 categories instead of one over the whole row), so the sampled sequences for a given seed differ
    from :func:`sample_top_k`'s while the distribution is identical. Returns the sampled column index."""
    values = torch.nan_to_num(logits.reshape(-1).float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k).values[-1]
    candidates = torch.nonzero(values >= threshold).reshape(-1)
    probs = torch.nan_to_num(F.softmax(values[candidates], dim=-1), nan=0.0)
    probs = probs / probs.sum().clamp_min(1e-12)
    pick = torch.multinomial(probs, 1, generator=generator)
    return int(candidates[pick].item())


class ARGenerator:
    """Prompt + lyrics -> ``frame_hiddens [1, F, 32768]`` and the sampled RVQ codes, on device.

    Args:
        llm: the stage-02 ``MusicLLM``. The generator installs the semantic logits window on it; a decode
            trace captured before that (a process that already ran ``MusicLLM.decode``) is released and
            re-captured with the window on the first frame.
        depth: the stage-03 ``DepthDecoder`` with its weights on device.
        tokenizer_dir: the checkpoint's ``tokenizer`` directory (default ``$MM3_WEIGHTS/tokenizer``).
        embed_weight / audio_embeddings: host copies of the backbone embedding table
            (``[200000, 4096]`` bf16) and the depth decoder's ``audio_embeddings`` (``[7168, 4096]``
            bf16); loaded from the safetensors when not given.
    """

    def __init__(
        self,
        llm: MusicLLM,
        depth: DepthDecoder,
        tokenizer_dir: Optional[str] = None,
        *,
        embed_weight: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
    ):
        self.llm = llm
        self.depth = depth
        self.batch = llm.max_batch_size
        assert self.batch == LLM_BATCH, self.batch
        self.encoder = PromptEncoder(tokenizer_dir)
        t0 = time.time()
        self.embed_weight = embed_weight if embed_weight is not None else R.load_embed_weight()
        self.audio_embeddings = audio_embeddings if audio_embeddings is not None else R.load_audio_embeddings()
        assert self.embed_weight.shape == (llm.vocab_size, LLM_HIDDEN), tuple(self.embed_weight.shape)
        logger.info(f"ARGenerator: host embedding tables ready in {time.time() - t0:.1f}s")

        # The traced decode returns only the tile-aligned semantic window of the logits. The window is part of the
        # captured graph, so a trace captured without it is released here and re-captured on the first frame.
        if llm.logits_window is None:
            if llm._trace_id is not None:
                logger.warning("ARGenerator: backbone trace captured without a logits window; releasing it")
                llm.release_trace()
            llm.set_logits_window(WINDOW_START, WINDOW_END)
        if tuple(llm.logits_window) != (WINDOW_START, WINDOW_END):
            raise ValueError(f"MusicLLM logits window {llm.logits_window} does not cover {(WINDOW_START, WINDOW_END)}")

        # Trace-lifetime order: backbone decode inputs, then the depth buffers + depth traces; the
        # backbone trace is captured on the first decode (its outputs are consumed before every
        # depth replay, so they may not overlap depth buffers that are alive across a replay -
        # the depth buffers are, because they were allocated first).
        llm.prepare_decode_inputs()
        self.depth_trace = DepthStepTrace(depth)

        # Vocabulary mask (diffusers ``vocab_mask``): only the 16384 semantic codes and the end token may be sampled.
        self.vocab_mask = torch.ones(llm.vocab_size, dtype=torch.bool)
        self.vocab_mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
        self.vocab_mask[AUDIO_END_TOKEN_ID] = False
        # The same mask on the read-back window (column i = vocabulary id WINDOW_START + i): the hot path works on
        # the 16389-wide window instead of a 200k-wide tensor (stage 07; identical candidate sets and probabilities,
        # see sample_top_k_candidates).
        self.window_mask = self.vocab_mask[WINDOW_START:WINDOW_END].clone()
        self._codebook_offsets = (torch.arange(NUM_CODEBOOKS - 1) * AUDIO_VOCAB_SIZE).unsqueeze(0)

    # ------------------------------------------------------------------ prompt
    def build_text_ids(self, prompt: str, lyrics: str) -> torch.LongTensor:
        """``MiniMaxMusic3TokenizeStep``: ``[2, L]`` token ids, row 1 = the CFG prompt (``ids[:, 1:-2] = 151654``)."""
        return self.encoder.encode(prompt, lyrics)

    # ------------------------------------------------------------------ pieces of the loop
    def _full_logits(self, window_logits: torch.Tensor) -> torch.Tensor:
        """Put the read-back window into a full-vocabulary fp32 tensor with every masked id at -inf.

        Diagnostics and tests only (``sample_top_k``, ``end_token_stats``, ``_semantic_rank``): the reference's
        full-length multinomial draws one random number per category, so reproducing its exact draws needs the
        full 200k-wide tensor. The generation hot path uses ``_guided_window`` + ``sample_top_k_candidates``
        instead (stage 07, doc/optimize/README.md decision 1: same candidate set and probabilities, cheaper draw).
        """
        full = torch.full((self.batch, self.llm.vocab_size), -float("inf"))
        full[:, WINDOW_START:WINDOW_END] = window_logits
        return full

    def _guided_logits(self, window_logits: torch.Tensor) -> torch.Tensor:
        """The reference's ``guided`` ``[1, vocab]``: masked CFG logits restricted to the conditional row's top-50."""
        logits = self._full_logits(window_logits).masked_fill(self.vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        # Restrict the guided distribution to the conditional branch's top candidates, then re-mask.
        threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        return guided.masked_fill(self.vocab_mask.unsqueeze(0), -float("inf"))

    def _guided_window(self, window_logits: torch.Tensor) -> torch.Tensor:
        """``_guided_logits`` on the window only: ``[W]`` fp32 guided logits, -inf outside the candidate set."""
        logits = window_logits.float().masked_fill(self.window_mask, -float("inf"))
        conditional, unconditional = logits[0], logits[1]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        threshold = torch.topk(conditional, AR_CFG_TOP_K).values[-1]
        return guided.masked_fill(conditional < threshold, -float("inf")).masked_fill(self.window_mask, -float("inf"))

    def _sample_semantic(self, window_logits: torch.Tensor, generator: torch.Generator) -> int:
        """The sampled vocabulary id (end token or ``AUDIO_CODE_OFFSET + code``)."""
        return WINDOW_START + sample_top_k_candidates(self._guided_window(window_logits), generator)

    def end_token_stats(self, window_logits: torch.Tensor) -> Dict[str, float]:
        """Diagnostics: rank of the end token in the conditional row and its probability under the sampling distribution."""
        conditional = self._full_logits(window_logits)[0].masked_fill(self.vocab_mask, -float("inf"))
        guided = self._guided_logits(window_logits)
        values = torch.nan_to_num(guided, nan=-1e9, posinf=1e9, neginf=-1e9)
        threshold = torch.topk(values, AR_SAMPLING_TOP_K, dim=-1).values[..., -1, None]
        probs = F.softmax(values.masked_fill(values < threshold, -float("inf")), dim=-1)[0]
        return {
            "rank_conditional": int((conditional > conditional[AUDIO_END_TOKEN_ID]).sum()),
            "prob_sampling": float(probs[AUDIO_END_TOKEN_ID]),
            "logit_gap_conditional": float(conditional.max() - conditional[AUDIO_END_TOKEN_ID]),
        }

    def _semantic_rank(self, window_logits: torch.Tensor, code: int) -> Dict[str, float]:
        """Diagnostics for teacher forcing: where the given code sits in the device's conditional / guided distributions."""
        logits = self._full_logits(window_logits).masked_fill(self.vocab_mask, -float("inf"))
        conditional, unconditional = logits[0], logits[1]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        token = code + AUDIO_CODE_OFFSET
        rank_cond = int((conditional > conditional[token]).sum())
        rank_guided = int((guided > guided[token]).sum())
        return {"rank_conditional": rank_cond, "rank_guided": rank_guided}

    def _feedback(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """diffusers ``_embed_audio_frame`` on the host tables, ``[2, 4096]`` bf16."""
        return R.embed_audio_frame(self.embed_weight, self.audio_embeddings, frame_codes)

    def _decode(self, feedback: torch.Tensor, position: int):
        """One backbone step -> (device hidden row tensor, host fp32 hidden [2, 4096], host fp32 window logits [2, W])."""
        # The full tiled logits output of the backbone trace is never read: it is allocated inside the backbone
        # capture, after the depth traces exist, so a depth replay may overwrite it - only the hidden (copied into
        # the depth seed buffer first) and the window (read back first) are consumed.
        pos = torch.full((self.batch,), position, dtype=torch.int64)
        return self.llm.decode_windowed(feedback, pos)

    def _depth_frame(
        self,
        hidden_seed,
        semantic_code: int,
        generator: torch.Generator,
        teacher: Optional[torch.Tensor],
    ):
        """``_generate_depth_codes``: the seven residual codes (sampled or teacher-forced) and ``[1, 7 * 4096]`` depth hiddens."""
        sem_embed = self.embed_weight[torch.full((self.batch,), semantic_code + AUDIO_CODE_OFFSET)]  # [2, 4096] bf16
        self.depth_trace.begin_frame(hidden_seed, sem_embed)
        codes = [semantic_code]
        hidden_parts: List[torch.Tensor] = []
        prev = None
        for index in range(1, NUM_CODEBOOKS):
            self.depth_trace.step(index, prev)
            hidden_parts.append(self.depth_trace.hidden_for(1))  # row 0 (conditional), like the reference's hidden[:1]
            if teacher is not None:
                code = int(teacher[index])
            else:
                logits = self.depth_trace.logits_for(index)  # [2, 1024] fp32
                conditional, unconditional = logits[0], logits[1]
                guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
                code = sample_top_k_candidates(guided, generator)
            codes.append(code)
            prev = torch.tensor([code, code])  # the sampled code is repeated for both CFG rows
        frame_codes = torch.tensor(codes, dtype=torch.int64).unsqueeze(0).expand(self.batch, -1)
        return frame_codes, torch.cat(hidden_parts, dim=-1)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        lyrics: str,
        *,
        max_frames: int,
        seed: int,
        teacher_codes: Optional[torch.Tensor] = None,
        teacher_frame0_codes: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        collect_end_token_stats: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        """Run the AR stage.

        Args:
            prompt, lyrics: the music description and the lyrics (``text_ids`` overrides them).
            max_frames: upper bound on emitted frames (25 per second; capped at 9000 like the reference).
            seed: seed of the CPU ``torch.Generator`` used for every top-k draw.
            teacher_codes: optional ``[F, 8]`` codes of the *emitted* frames (frame 1..F, the layout
                of the golden ``sampled_codes.pt``): no sampling, the loop feeds these back and stops
                after ``min(F, max_frames)`` frames.
            teacher_frame0_codes: the ``[8]`` codes of the un-emitted frame 0 (its feedback shapes
                every later state); when ``None`` under teacher forcing they are sampled with ``seed``.
            text_ids: pre-built ``[2, L]`` token ids (skips the tokenizer).
            collect_end_token_stats: also return per-frame ``end_token_stats`` (diagnostics, host cost).
            generator: an existing CPU ``torch.Generator`` to draw from instead of a fresh one seeded with
                ``seed`` (the pipeline threads one generator through the AR draws and the DiT noise, as diffusers does).
        Returns:
            ``frame_hiddens`` ``[1, F, 32768]`` fp32, ``codes`` ``[F, 8]`` (semantic + 7 residual codes of
            each emitted frame), ``frames`` = F, ``frame0_codes`` ``[8]``, ``stopped_by`` (``"end_token"``,
            ``"max_frames"``, ``"teacher_codes"`` or ``"context"``), ``text_ids``, ``timings``
            (host wall seconds per section and per frame) and ``semantic_ranks`` (teacher forcing only).
        """
        max_frames = min(int(max_frames), MAX_AUDIO_FRAMES)
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        if text_ids is None:
            text_ids = self.build_text_ids(prompt, lyrics)
        assert text_ids.shape[0] == self.batch, tuple(text_ids.shape)
        prompt_len = text_ids.shape[1]
        if teacher_codes is not None:
            teacher_codes = torch.as_tensor(teacher_codes, dtype=torch.int64)
            assert teacher_codes.dim() == 2 and teacher_codes.shape[1] == NUM_CODEBOOKS, tuple(teacher_codes.shape)
            max_frames = min(max_frames, teacher_codes.shape[0])
        if generator is None:
            generator = torch.Generator().manual_seed(int(seed))
        timings = {"prefill": 0.0, "llm_step": 0.0, "llm_steps": 0, "depth": 0.0, "host": 0.0, "per_frame": []}

        t0 = time.perf_counter()
        self.llm.reset_cache()
        hidden0, logits0 = self.llm.prefill(self.embed_weight[text_ids])
        timings["prefill"] = time.perf_counter() - t0
        hidden_seed = hidden0  # host [2, 4096]; later frames seed the depth decoder from the device tensor
        last_hidden = hidden0
        window_logits = logits0[:, WINDOW_START:WINDOW_END]

        frame_hiddens: List[torch.Tensor] = []
        codes_out: List[torch.Tensor] = []
        ranks: List[Dict[str, float]] = []
        end_stats: List[Dict[str, float]] = []
        frame0_codes = None
        stopped_by = "max_frames"
        # The first decode step only advances the state past <|audio_start|> and is not an emitted frame.
        for frame_index in range(max_frames + 1):
            tf = time.perf_counter()
            teacher = None
            if teacher_codes is not None:
                if frame_index == 0:
                    teacher = teacher_frame0_codes if teacher_frame0_codes is not None else None
                else:
                    teacher = teacher_codes[frame_index - 1]
            if teacher is not None:
                semantic_code = int(teacher[0])
                ranks.append(self._semantic_rank(window_logits, semantic_code))
            else:
                if collect_end_token_stats:
                    end_stats.append(self.end_token_stats(window_logits))
                sampled = self._sample_semantic(window_logits, generator)
                if sampled == AUDIO_END_TOKEN_ID:
                    stopped_by = "end_token"
                    break
                semantic_code = sampled - AUDIO_CODE_OFFSET
            th = time.perf_counter()
            timings["host"] += th - tf

            frame_codes, depth_hidden = self._depth_frame(hidden_seed, semantic_code, generator, teacher)
            td = time.perf_counter()
            timings["depth"] += td - th

            if frame_index == 0:
                frame0_codes = frame_codes[0].clone()
            else:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                codes_out.append(frame_codes[0].clone())
                if len(frame_hiddens) >= max_frames:
                    stopped_by = "teacher_codes" if teacher_codes is not None else "max_frames"
                    timings["per_frame"].append(time.perf_counter() - tf)
                    break
            position = prompt_len + frame_index
            if position >= self.llm.max_seq_len:
                stopped_by = "context"
                logger.warning(f"ARGenerator: context {self.llm.max_seq_len} reached after {len(frame_hiddens)} frames")
                break
            feedback = self._feedback(frame_codes)
            tfb = time.perf_counter()
            timings["host"] += tfb - td
            hidden_seed, last_hidden, window_logits = self._decode(feedback, position)
            tl = time.perf_counter()
            timings["llm_step"] += tl - tfb
            timings["llm_steps"] += 1
            timings["per_frame"].append(tl - tf)

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")
        out = {
            "frame_hiddens": torch.stack(frame_hiddens, dim=1),  # [1, F, 32768]
            "codes": torch.stack(codes_out, dim=0),  # [F, 8]
            "frames": len(frame_hiddens),
            "frame0_codes": frame0_codes,
            "stopped_by": stopped_by,
            "text_ids": text_ids,
            "prompt_len": prompt_len,
            "timings": timings,
            "decode_stats": dict(self.llm.decode_stats),
        }
        if ranks:
            out["semantic_ranks"] = ranks
        if collect_end_token_stats:
            out["end_token_stats"] = end_stats
        return out

    def release(self):
        self.depth_trace.release()
