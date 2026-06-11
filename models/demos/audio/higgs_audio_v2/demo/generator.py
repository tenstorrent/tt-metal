# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Higgs Audio v2 generation on TTNN.

The LLM backbone (prefill + autoregressive audio-token decode) runs on our TTNN
``HiggsAudioTTModel``; the HF ``HiggsAudioV2Processor`` is reused for I/O only —
chat templating, reference-audio encoding, and audio-token -> waveform decoding
(the DAC-based ``HiggsAudioV2TokenizerModel`` codec at /data/hf_cache/higgs/tokenizer).

Generation protocol (mirrors HF generate's audio_sequences format so we can hand
the result straight to ``processor.batch_decode``):
  1. Build input_ids from a chat conversation (ends at <|audio_out_bos|>).
  2. TTNN prefill over the prompt (text, + spliced reference-audio embeddings for
     voice cloning) -> first audio-row logits.
  3. Greedy-argmax each step through the delay-pattern state machine
     (tt/audio_decode.py), feeding the selected row back into ``decode_step_audio``
     until the all-EOS row (finished) or max_new_tokens.
  4. Stack rows as [1, T, 8] with a leading all-BOS row -> processor.batch_decode.
"""
from __future__ import annotations

import os
import pathlib

import torch
import ttnn
from loguru import logger

from models.demos.audio.higgs_audio_v2.tt.reference import HiggsAudioV2Config, load_higgs_v2_state_dict
from models.demos.audio.higgs_audio_v2.tt.model_args import HiggsModelArgs, BASE_TEXT_MODEL
from models.demos.audio.higgs_audio_v2.tt.model import HiggsAudioTTModel
from models.demos.audio.higgs_audio_v2.tt.precision_presets import build_precision
from models.demos.audio.higgs_audio_v2.tt.audio_decode import (
    initialize_delay_pattern_state,
    apply_delay_pattern_to_greedy_audio_tokens,
    apply_delay_pattern_to_selected_audio_tokens,
)
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup
from models.tt_transformers.tt.common import Mode


HIGGS_MODEL_DIR = "/data/hf_cache/higgs"
SAMPLING_RATE = 24000


class _DelayCfg:
    def __init__(self, args):
        self.audio_num_codebooks = args.audio_num_codebooks
        self.audio_stream_bos_id = args.audio_stream_bos_id
        self.audio_stream_eos_id = args.audio_stream_eos_id
        self.use_delay_pattern = True


class HiggsAudioTTSGenerator:
    def __init__(self, mesh_device, precision="performance", model_dir=HIGGS_MODEL_DIR):
        from transformers import AutoProcessor

        self.mesh_device = mesh_device
        self.model_dir = model_dir
        self.processor = AutoProcessor.from_pretrained(model_dir)

        higgs_cfg = HiggsAudioV2Config.from_json(pathlib.Path(model_dir) / "config.json")
        opt = build_precision(precision, higgs_cfg.num_hidden_layers, BASE_TEXT_MODEL)
        self.args = HiggsModelArgs(
            mesh_device=mesh_device, higgs_config=higgs_cfg, max_batch_size=1, max_seq_len=1024, optimizations=opt
        )
        self.cfg = _DelayCfg(self.args)
        self.K = self.args.audio_num_codebooks
        self.cb_size = self.args.audio_codebook_size

        _, state_dict = load_higgs_v2_state_dict(model_dir)
        tt_ccl = TT_CCL(mesh_device)
        RopeCls = HfRotarySetup if self.args.use_hf_rope else RotarySetup
        self.rope_setup = RopeCls(
            device=mesh_device, batch_size=self.args.max_batch_size, head_dim=self.args.head_dim,
            max_seq_len=self.args.max_seq_len, rope_theta=self.args.rope_theta, rope_scaling=self.args.rope_scaling,
            use_qk_fused=self.args.use_qk_fused, prefetcher=None,
        )
        self.model = HiggsAudioTTModel(
            args=self.args, mesh_device=mesh_device, tt_ccl=tt_ccl, state_dict=state_dict,
            transformation_mats=self.rope_setup.get_both_trans_mats(), dtype=ttnn.bfloat8_b,
        )
        logger.info(f"HiggsAudioTTSGenerator ready (precision={precision}, dim={self.args.dim})")

    # ---- prompt building -----------------------------------------------------
    def build_inputs(self, conversation):
        """Return (input_ids[S], audio_input_ids or None) for a chat conversation.

        Text-only conversations (TTS / multi-speaker) -> input_ids only.
        Conversations with an assistant 'audio' turn (voice cloning) -> also the
        reference audio_input_ids the processor encoded.
        """
        enc = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"][0].to(torch.int64)
        audio_in = enc.get("audio_input_ids", None)
        audio_mask = enc.get("audio_input_ids_mask", None)
        return input_ids, audio_in, audio_mask

    # ---- decode helpers ------------------------------------------------------
    def _rot_inputs(self, pos):
        cp = torch.tensor([pos], dtype=torch.int32)
        ridx = self.rope_setup.get_rot_idxs(cp, on_host=True)
        ridx = ttnn.to_device(ridx, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cp_tt = ttnn.from_torch(
            cp, device=self.mesh_device, dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
        )
        return cp_tt, self.rope_setup.get_rot_mats(ridx)

    @staticmethod
    def _sample_tokens(logits, temperature, top_k, top_p):
        """Per-codebook sampling. logits [K, cb_size] -> selected ids [K].

        temperature<=0 -> greedy argmax. Otherwise temperature + optional top-k
        + nucleus (top-p), sampled independently per codebook. BOS/EOS ids stay
        in the distribution so the stream can terminate naturally.
        """
        if temperature is None or temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits.float() / temperature
        if top_k and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p and top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
            remove = (cum - torch.softmax(s_logits, dim=-1)) > top_p  # always keep top-1
            s_logits = s_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, s_idx, s_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _ras(self, selected, logits, rows, win_len, max_rep):
        """Repetition-Aware Sampling (Higgs RAS, ras_win_len/ras_win_max_num_repeat).

        For each codebook whose sampled token already appears >= max_rep times in
        the last win_len generated frames (excluding BOS/EOS), resample it from the
        raw logits (no temperature) — breaks phrase-level repetition loops.
        """
        if win_len <= 0 or not rows:
            return selected
        window = torch.stack(rows[-win_len:], dim=0)  # [w, K]
        rep = (window == selected.unsqueeze(0))
        rep = rep & (window != self.cfg.audio_stream_bos_id) & (window != self.cfg.audio_stream_eos_id)
        repl = rep.sum(dim=0) >= max_rep  # [K]
        if bool(repl.any()):
            resampled = torch.multinomial(torch.softmax(logits.float(), dim=-1), 1).squeeze(-1)
            selected = torch.where(repl, resampled.to(selected.dtype), selected)
        return selected

    @torch.no_grad()
    def generate(self, conversation, max_new_tokens=750, temperature=1.0, top_k=50, top_p=0.95, seed=1234,
                 silence_patience=32, ras_win_len=7, ras_win_max_num_repeat=2):
        """Run prefill + free-running greedy audio decode.

        Returns audio_input_ids as a torch.LongTensor [1, T, K] in the delay-
        pattern format (leading all-BOS row, generated rows, trailing all-EOS
        row) ready for ``processor.batch_decode``.
        """
        input_ids, audio_in, audio_mask = self.build_inputs(conversation)
        if audio_in is not None:
            logger.info(f"reference audio present: {tuple(audio_in.shape)} (voice cloning)")
        S = input_ids.shape[0]

        # Prefill populates the KV cache; its audio-head output at the text
        # <|audio_out_bos|> position is NOT a content prediction (it's degenerate),
        # so we ignore it. Generation starts by feeding the all-BOS audio frame
        # (the canonical first audio row, == fixture row 0) at pos=S — exactly the
        # protocol the teacher-forced accuracy gate validated at 0.967.
        if seed is not None:
            torch.manual_seed(seed)
        greedy = (temperature is None) or (temperature <= 0)
        logger.info(f"decode: {'greedy' if greedy else f'sampling T={temperature} top_k={top_k} top_p={top_p}'} "
                    f"RAS(win={ras_win_len}, max_rep={ras_win_max_num_repeat})")
        _ = self.model.prefill_text(
            input_ids, self.rope_setup, audio_input_ids=audio_in, audio_input_ids_mask=audio_mask,
            audio_token_id=self.processor.audio_token_id,
        )
        if os.environ.get("HIGGS_DEBUG"):
            logger.info(f"[dbg] S={S} input_ids tail={input_ids[-6:].tolist()}")

        bos_row = torch.full((self.K,), self.cfg.audio_stream_bos_id, dtype=torch.long)
        num_delay, num_rem = initialize_delay_pattern_state(bos_row.view(self.K, 1), self.cfg)

        rows = []  # generated delay-pattern rows ([K] each), rows[0] follows the all-BOS frame
        cur_in = bos_row
        pos = S
        finished = False
        prev_row = None
        repeat = 0  # consecutive identical rows -> silence/degeneration
        for step in range(max_new_tokens):
            cp_tt, rot_mats = self._rot_inputs(pos)
            logits = self.model.decode_step_audio(cur_in.clamp(min=0, max=self.cb_size - 1), cp_tt, rot_mats)
            selected = self._sample_tokens(logits, temperature, top_k, top_p)
            selected = self._ras(selected, logits, rows, ras_win_len, ras_win_max_num_repeat)
            nxt, _active, num_delay, num_rem, finished = apply_delay_pattern_to_selected_audio_tokens(
                selected, self.cfg, num_delay, num_rem
            )
            nxt = nxt.to(torch.long)
            rows.append(nxt)
            if os.environ.get("HIGGS_DEBUG") and step < 14:
                logger.info(f"[dbg] step {step:2d} pos={pos} row={nxt.tolist()} "
                            f"nd={num_delay} nr={num_rem} fin={finished}")
            if finished:
                break
            # Early stop: the model loops on an identical row when it has run out
            # of content but failed to emit EOS (the silent tail). Drop the run.
            if prev_row is not None and torch.equal(nxt, prev_row):
                repeat += 1
                if repeat >= silence_patience:
                    del rows[-repeat:]  # drop the degenerate repeated run, keep first instance
                    logger.info(f"early stop at step {step}: dropped {repeat} repeated rows (no EOS)")
                    break
            else:
                repeat = 0
            prev_row = nxt
            cur_in = nxt
            pos += 1

        if not finished:
            logger.warning(f"no natural EOS (max_new_tokens / degeneration); appending EOS row")
            rows.append(torch.full((self.K,), self.cfg.audio_stream_eos_id, dtype=torch.long))

        # [1, T, K] with leading all-BOS row, as batch_decode expects.
        audio_seq = torch.stack([bos_row] + rows, dim=0).unsqueeze(0)
        logger.info(f"generated {len(rows)} audio rows (finished={finished}) -> audio_seq {tuple(audio_seq.shape)}")
        return audio_seq

    # ---- waveform output -----------------------------------------------------
    @staticmethod
    def _trim_silence(wf, sr=SAMPLING_RATE, rel_thresh=0.02, abs_thresh=0.004, pad_ms=120):
        """Trim leading/trailing near-silence from a 1D waveform.

        Threshold is the max of an absolute floor and a fraction of the clip's
        peak frame energy, so quiet clips aren't over-trimmed. A small pad is
        kept around the speech so it doesn't sound clipped.
        """
        x = wf.detach().float().flatten()
        fr = int(0.02 * sr)
        if x.numel() < 2 * fr:
            return wf
        nb = x.numel() // fr
        e = x[: nb * fr].reshape(nb, fr).pow(2).mean(-1).clamp_min(1e-12).sqrt()
        thr = max(abs_thresh, rel_thresh * float(e.max()))
        active = (e > thr).nonzero().flatten()
        if active.numel() == 0:
            return wf
        pad = int(pad_ms / 1000 * sr)
        start = max(0, int(active[0]) * fr - pad)
        end = min(x.numel(), (int(active[-1]) + 1) * fr + pad)
        return wf[start:end]

    def to_waveforms(self, audio_seq, trim=True):
        """audio_seq [1,T,K] -> list of 1D waveform tensors.

        Default uses the HF codec. With HIGGS_TTNN_CODEC=1, runs the ported
        TTNN DacDecoder (tt/codec.py) for the token->waveform step instead,
        replicating batch_decode's BOS/EOS trim + delay-pattern revert on host.
        """
        if os.environ.get("HIGGS_TTNN_CODEC") == "1":
            wfs = [self._ttnn_codec_decode(audio_seq)]
        else:
            wfs = self.processor.batch_decode(audio_seq)
        if trim:
            wfs = [self._trim_silence(w) for w in wfs]
        return wfs

    def _ttnn_codec_decode(self, audio_seq):
        from models.demos.audio.higgs_audio_v2.tt.codec import tt_decode

        p = self.processor
        bos, eos = p.audio_stream_bos_id, p.audio_stream_eos_id
        ids = audio_seq
        start = int((ids == bos).all(-1).nonzero()[-1, -1])
        ids = ids[:, start:]
        eos_rows = (ids == eos).all(-1).nonzero()
        end = int(eos_rows[eos_rows[:, 0] == 0, 1].min()) if (eos_rows[:, 0] == 0).any() else ids.shape[1]
        codes = p.revert_delay_pattern(ids[0, 1:end]).clip(0, bos - 1).transpose(0, 1).unsqueeze(0)  # [1,K,T]
        if not hasattr(self, "_ttdec"):
            from models.demos.audio.higgs_audio_v2.tt.codec import TtDacDecoder

            self._ttdec = TtDacDecoder(self.mesh_device, p.audio_tokenizer.acoustic_decoder)
        wf = tt_decode(self.mesh_device, p.audio_tokenizer, codes, ttdec=self._ttdec)
        return wf.squeeze().detach().cpu()

    def save(self, audio_seq, out_path):
        waveforms = self.to_waveforms(audio_seq)
        self.processor.save_audio(waveforms, str(out_path), sampling_rate=SAMPLING_RATE)
        wf = waveforms[0]
        dur = wf.numel() / SAMPLING_RATE
        logger.info(f"saved {out_path}  ({dur:.2f}s, {wf.numel()} samples @ {SAMPLING_RATE}Hz)")
        return out_path, dur
