# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.config import NUM_LATENTS
from models.experimental.xtts.reference.xtts_conditioning import chunk_wav
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_gpt_generate import MAX_AUDIO_TOKENS
from models.experimental.xtts.reference.xtts_hifi_decoder import LATENT_SCALE, SR_SCALE
from models.experimental.xtts.reference.xtts_hifigan import COND_CHANNELS
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning
from models.experimental.xtts.tt.xtts_full_decoder import TtXttsHifiDecoder
from models.experimental.xtts.tt.xtts_generator import TtTracedDecoder, TtXttsGenerator
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel
from models.experimental.xtts.tt.xtts_mel import TtConditioningMel


def _interp_len(frames):
    """Map mel frames to waveform samples after upsample scales."""
    return int(int(frames * LATENT_SCALE) * SR_SCALE)


class TtXtts(LightweightModule):
    def __init__(self, device, state_dict, ref_decoder_full):
        """Build conditioning, GPT, generator, and HiFi decoder."""
        super().__init__()
        self.device = device
        self.conditioning = TtXttsConditioning(state_dict, device)
        self.cond_mel_fe = TtConditioningMel(device, state_dict["mel_stats"].cpu())
        self.gpt = TtXttsGptModel(state_dict, device)
        self.generator = TtXttsGenerator(self.gpt)
        self.decoder = TtXttsHifiDecoder(device, ref_decoder_full)

    def _wav_chunk_to_device(self, chunk):
        """Upload a waveform chunk to device as float32."""
        return ttnn.from_torch(
            chunk.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.float32
        )

    def _style_from_mel(self, mel_dev):
        """Run conditioning encoder on device mel."""
        return self.conditioning.forward_dev(ttnn.typecast(mel_dev, ttnn.bfloat16))

    def _style_window(self, wav_dev):
        """Compute style latents for one waveform window."""
        mel = self.cond_mel_fe(wav_dev)
        mel_bf = ttnn.typecast(mel, ttnn.bfloat16)
        style = self.conditioning.forward_dev(mel_bf)
        # forward_dev keeps the input alive; drop mel chain once style is produced.
        for t in (mel_bf, mel):
            if t.is_allocated():
                ttnn.deallocate(t)
        out = ttnn.to_memory_config(style, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(style)
        return out

    def _style_mean(self, wav_devs):
        """Average style latents across waveform windows."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        acc = self._style_window(wav_devs[0])
        for w in wav_devs[1:]:
            part = self._style_window(w)
            nxt = ttnn.add(acc, part, memory_config=dram)
            ttnn.deallocate(part)
            ttnn.deallocate(acc)
            acc = nxt
        if len(wav_devs) > 1:
            mean = ttnn.multiply(acc, 1.0 / len(wav_devs), memory_config=dram)
            ttnn.deallocate(acc)
            acc = mean
        out = ttnn.permute(acc, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(acc)
        return out

    def _cond_latents(self, cond_wav):
        """Chunk conditioning wav and return mean style latents."""
        return self._style_mean([self._wav_chunk_to_device(c) for c in chunk_wav(cond_wav)])

    def _decode_wav(self, latents_tt, ref_wav_spk):
        """Decode GPT latents to waveform with reference speaker."""
        return self.decoder(latents_tt, ref_wav_spk)

    def inference(
        self,
        text_ids,
        cond_wav,
        ref_wav_spk,
        max_new_tokens=MAX_AUDIO_TOKENS,
        force_codes=None,
        temperature=0.0,
        top_k=0,
        repetition_penalty=1.0,
        top_p=1.0,
        min_new_tokens=0,
    ):
        """Run end-to-end TTS inference returning wav and codes."""
        cond_latents = self._cond_latents(cond_wav)
        if force_codes is not None:
            _, latents_tt = self.generator.latents_for_codes(text_ids, cond_latents, force_codes)
            codes = torch.tensor([force_codes], dtype=torch.long)
        else:
            codes, latents_tt = self.generator.generate(
                text_ids,
                cond_latents,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                min_new_tokens=min_new_tokens,
            )
        wav = self._decode_wav(latents_tt, ref_wav_spk)
        return wav, codes

    def inference_fully_traced(
        self,
        text_ids,
        cond_wav,
        ref_wav_spk,
        max_seq,
        max_new_tokens=MAX_AUDIO_TOKENS,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
        text_real_len=None,
    ):
        """Run fully traced setup, decode, and vocoder inference."""
        dev = self.device
        gpt = self.gpt
        t_all0 = time.perf_counter()

        wav_devs = [self._wav_chunk_to_device(c) for c in chunk_wav(cond_wav)]
        text_dev = gpt.text_ids_to_device(text_ids)
        gpt.alloc_static_kv(max_seq)
        # Padding exists only to tile-align the prompt; keep decode from attending to it.
        gpt.set_text_padding(
            NUM_LATENTS, text_ids.shape[1] if text_real_len is None else text_real_len, text_ids.shape[1]
        )

        def _setup():
            """Compute speaker emb and prefill for setup capture."""
            cl = self._style_mean(wav_devs)
            g = self.decoder.speaker_embedding(ref_wav_spk)
            return g, gpt.prefill_on_device(text_dev, cl)

        # Warmup allocs a throwaway speaker emb — free it before capture so L1 is clear for CBs.
        g_warm, _ = _setup()
        ttnn.synchronize_device(dev)
        if g_warm.is_allocated():
            ttnn.deallocate(g_warm)
        stid = ttnn.begin_trace_capture(dev, cq_id=0)
        g, prompt_len = _setup()
        ttnn.end_trace_capture(dev, stid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, stid, blocking=True)
        setup_replay_s = time.perf_counter() - t0
        ttnn.release_trace(dev, stid)
        # Setup inputs are no longer bound once the setup trace is released.
        for w in wav_devs:
            if w.is_allocated():
                ttnn.deallocate(w)
        if text_dev.is_allocated():
            ttnn.deallocate(text_dev)

        codes, latents, decode_replay_s, stopped = self.generator.generate_ondevice_traced(
            prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
        )

        # Warmup primes folded cond-bias (from_device is fatal inside a trace).
        lat_in = latents
        voc = self.decoder.decoder
        warm_wav = voc(ttnn.clone(lat_in), g)
        if warm_wav.is_allocated():
            ttnn.deallocate(warm_wav)
        ttnn.synchronize_device(dev)
        vtid = ttnn.begin_trace_capture(dev, cq_id=0)
        wav_dev = voc(ttnn.clone(lat_in), g)
        ttnn.end_trace_capture(dev, vtid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, vtid, blocking=True)
        vocoder_replay_s = time.perf_counter() - t0
        ttnn.release_trace(dev, vtid)
        # Release only after release_trace — never evict under a live trace.
        voc.generator.release_conditioning()
        voc.upsampler.release_cache()
        if lat_in.is_allocated():
            ttnn.deallocate(lat_in)
        if g.is_allocated():
            ttnn.deallocate(g)
        replay_s = setup_replay_s + decode_replay_s + vocoder_replay_s
        compile_s = max(0.0, time.perf_counter() - t_all0 - replay_s)
        perf = {
            "replay_s": replay_s,
            "compile_s": compile_s,
            "setup_replay_s": setup_replay_s,
            "decode_replay_s": decode_replay_s,
            "vocoder_replay_s": vocoder_replay_s,
            "stopped": stopped,
        }
        return wav_dev, codes, perf

    def traced_session(self, cond_wav, ref_wav_spk, text_len, max_seq, max_new_tokens, **sampling):
        """Create a reusable traced inference session."""
        return TtXttsTracedSession(self, cond_wav, ref_wav_spk, text_len, max_seq, max_new_tokens, **sampling)


class TtXttsTracedSession:
    def __init__(
        self,
        tt,
        cond_wav,
        ref_wav_spk,
        text_len,
        max_seq,
        max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
    ):
        """Capture setup, decode, and vocoder traces for reuse."""
        t0 = time.perf_counter()
        self.tt = tt
        self.device = dev = tt.device
        self.N = int(max_new_tokens)
        gpt = tt.gpt
        voc = tt.decoder.decoder

        # Allocate all persistent buffers before any capture (unsafe after an active trace).
        self.wav_devs = [tt._wav_chunk_to_device(c) for c in chunk_wav(cond_wav)]
        self.text_dev = gpt.text_ids_to_device(torch.zeros(1, text_len, dtype=torch.long))
        self.ref_wav_spk = ref_wav_spk
        gpt.alloc_static_kv(max_seq)
        prompt_len = NUM_LATENTS + text_len
        self.decoder = TtTracedDecoder(
            gpt,
            prompt_len,
            self.N,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            capture=False,
        )
        self.voc_in = ttnn.from_torch(
            torch.zeros(1, self.N, HIDDEN_SIZE), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.float32
        )
        self.g = ttnn.from_torch(
            torch.zeros(1, 1, COND_CHANNELS), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.float32
        )

        def _setup():
            """Run style mean, speaker emb, and GPT prefill."""
            cl = tt._style_mean(self.wav_devs)
            g = tt.decoder.speaker_embedding(ref_wav_spk)
            g_dram = ttnn.to_memory_config(g, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(g)
            ttnn.copy(g_dram, self.g)
            ttnn.deallocate(g_dram)
            return gpt.prefill_on_device(self.text_dev, cl)

        warm_prompt_len = _setup()
        assert warm_prompt_len == prompt_len, f"prefill gave prompt_len {warm_prompt_len}, expected {prompt_len}"
        self.decoder.warmup()
        warm_wav = voc(ttnn.clone(self.voc_in), self.g)
        if warm_wav.is_allocated():
            ttnn.deallocate(warm_wav)
        ttnn.synchronize_device(dev)

        self.setup_tid = ttnn.begin_trace_capture(dev, cq_id=0)
        _setup()
        ttnn.end_trace_capture(dev, self.setup_tid, cq_id=0)
        ttnn.synchronize_device(dev)
        self.decoder.capture()
        self.voc_tid = ttnn.begin_trace_capture(dev, cq_id=0)
        self.wav_dev = voc(ttnn.clone(self.voc_in), self.g)
        ttnn.end_trace_capture(dev, self.voc_tid, cq_id=0)
        ttnn.synchronize_device(dev)
        self.upsample = self.wav_dev.shape[-2] // _interp_len(self.N)
        self.compile_s = time.perf_counter() - t0

    def _samples_for(self, frames):
        """Convert latent frames to output sample count."""
        return _interp_len(frames) * self.upsample

    def run(self, text_ids, real_len=None):
        """Replay traced session for new text ids.

        ``real_len`` is this chunk's UNPADDED wrapped-token count. Chunks share one capture, so
        they are all padded to a common length; the padding is then masked out of decode
        attention, which is what lets a short chunk still emit STOP on time.
        """
        dev = self.device
        assert text_ids.shape[1] == self.text_dev.shape[1], (
            f"session captured for {self.text_dev.shape[1]} text tokens, got {text_ids.shape[1]} — "
            "every chunk must be padded to the same length"
        )
        text_tmp = self.tt.gpt.text_ids_to_device(text_ids)
        ttnn.copy(text_tmp, self.text_dev)
        if text_tmp.is_allocated():
            ttnn.deallocate(text_tmp)
        self.tt.gpt.set_text_padding(
            NUM_LATENTS, text_ids.shape[1] if real_len is None else real_len, text_ids.shape[1]
        )
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, self.setup_tid, blocking=True)
        setup_replay_s = time.perf_counter() - t0

        self.decoder.reset(redraw_noise=True)
        codes, lat_host, decode_replay_s = self.decoder.run()

        frames = lat_host.shape[1]
        padded = torch.zeros(1, self.N, HIDDEN_SIZE, dtype=torch.float32)
        padded[:, :frames, :] = lat_host.float()
        lat_tmp = ttnn.from_torch(padded, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.float32)
        ttnn.copy(lat_tmp, self.voc_in)
        if lat_tmp.is_allocated():
            ttnn.deallocate(lat_tmp)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, self.voc_tid, blocking=True)
        vocoder_replay_s = time.perf_counter() - t0

        wav = ttnn.to_torch(self.wav_dev).float().reshape(-1)[: self._samples_for(frames)]
        replay_s = setup_replay_s + decode_replay_s + vocoder_replay_s
        perf = {
            "replay_s": replay_s,
            "compile_s": 0.0,
            "setup_replay_s": setup_replay_s,
            "decode_replay_s": decode_replay_s,
            "vocoder_replay_s": vocoder_replay_s,
            "stopped": self.decoder.stopped,
        }
        return wav, codes, perf

    def close(self):
        """Release traces and deallocate session buffers."""
        for tid in (self.setup_tid, self.voc_tid):
            ttnn.release_trace(self.device, tid)
        self.decoder.release()
        voc = self.tt.decoder.decoder
        voc.generator.release_conditioning()
        voc.upsampler.release_cache()
        for t in (*self.wav_devs, self.text_dev, self.voc_in, self.g, getattr(self, "wav_dev", None)):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        self.wav_devs = []
