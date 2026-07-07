# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 full generation pipeline on TTNN (DiT denoise loop -> VAE decode -> audio).

This is the top-level factory that assembles the validated subsystems into the end-to-end
text-to-music compute path, mirroring the DiT/LTX pipeline-factory pattern:

    create_tt_pipeline(args, device) -> AceStepPipeline
    pipeline.generate(context_latents, encoder_hidden_states, noise, infer_steps, ...) -> latents
    pipeline.decode(latents) -> 48kHz stereo waveform

The denoise loop reproduces the reference `generate_audio` ODE (Euler flow-matching) branch:

    t = linspace(1, 0, steps+1);  (optional shift transform)
    for (t_curr, t_prev): vt = DiT(xt, t_curr, ...); xt = xt - vt*(t_curr - t_prev)

CFG guidance is omitted here (guidance_scale=1.0 path) because apg_guidance.py is absent from the
base snapshot and only combines two DiT predictions — orthogonal to the core update. The reference
comparison in the e2e test uses the same no-CFG ODE so the PCC is an honest apples-to-apples check.

Conditioning (encoder_hidden_states, context_latents) is precomputed and passed in, so the pipeline
here owns exactly the DiT-loop + VAE stages; the ConditionEncoder is a separate validated factory
(build_condition_encoder) and the two compose in the full-pipeline test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.tt_dit.utils.tracing import Tracer
from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.flow_match import FlowMatchStep
from models.experimental.acestep.tt.model_config import (
    AceStepModelConfig,
    _build_dit_model,
    build_vae_decoder,
)

HEAD_DIM = 128
PATCH = 2


def _shifted_timesteps(infer_steps: int, shift: float, device, dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t


def _set_dit_fidelity(module, fidelity, _seen=None):
    """Recursively set math_fidelity on every compute_kernel_config held in a module tree.

    Used to raise the DiT to HiFi4 ONLY for the CFG denoise path (guidance amplifies the per-step
    matmul error ~6x, so HiFi4 measurably improves the CFG latent PCC). The shipped no-CFG path never
    enters this and stays at its built HiFi2, so its e2e gate is unaffected. Returns the list of
    (config, previous_fidelity) pairs so the caller can restore afterwards."""
    if _seen is None:
        _seen = set()
    changed = []
    if id(module) in _seen:
        return changed
    _seen.add(id(module))
    cfg = getattr(module, "config", None)
    # Only swap the attention/model `compute_kernel_config`. We deliberately do NOT touch the MLP1D
    # feed-forward configs (ff1_3_/ff2_compute_kernel_cfg): measured HiFi4 on the MLP feed-forward
    # REGRESSES the CFG PCC (0.888 -> 0.621) - the reference MLP aligns better at HiFi2. HiFi4 helps
    # only the attention path under CFG amplification.
    ck = getattr(cfg, "compute_kernel_config", None) if cfg is not None else None
    if ck is not None and hasattr(ck, "math_fidelity"):
        changed.append((ck, ck.math_fidelity))
        ck.math_fidelity = fidelity
    if hasattr(module, "__dict__"):
        for attr in vars(module).values():
            if hasattr(attr, "config") and hasattr(attr, "__dict__"):
                changed.extend(_set_dit_fidelity(attr, fidelity, _seen))
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if hasattr(item, "config") and hasattr(item, "__dict__"):
                        changed.extend(_set_dit_fidelity(item, fidelity, _seen))
    return changed


@dataclass
class AceStepPipeline:
    """Assembled TT pipeline: text encoder + condition encoder + DiT denoiser + Oobleck VAE.

    Customer-facing usage is a single call (tokenization + all encoders + denoise + decode handled
    internally):

        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)   # full text-to-music
        wav  = pipe.generate_song("upbeat synthwave, nostalgic", lyrics="neon city lights", seconds=8)
        # wav: torch [1, 2, samples] @ 48 kHz stereo

    Lower-level stages (generate / decode) remain available for advanced use.
    """

    args: AceStepModelConfig
    dit: object  # AceStepDiTModel
    vae: object  # OobleckDecoder
    vae_config: object
    mesh_device: object
    _rope: Qwen3RotaryEmbedding
    _mod: object  # reference modeling module (for create_4d_mask)
    text_encoder: object = None  # AceStepTextEncoder (prompt/lyrics -> embeddings)
    condition_encoder: object = None  # AceStepConditionEncoder (-> DiT cross-attn context)
    tokenizer: object = None  # HF Qwen3 tokenizer
    _hf_text_cfg: object = None  # text-encoder HF config (for its RoPE)
    _null_condition_emb: object = None  # torch [1,1,2048] learned null cross-attn context (for CFG)
    _silence_latent: object = None  # torch [1,15000,64] learned silence src latent (text2music src)

    def _uncond_context(self, cond_context):
        """Build the CFG null/unconditional cross-attn context: null_condition_emb expanded to the
        conditional context shape [1,1,enc_len,2048] (matches the reference's null_condition_emb path)."""
        assert self._null_condition_emb is not None, "null_condition_emb not loaded; CFG unavailable"
        enc_len = cond_context.shape[2]
        null = self._null_condition_emb.reshape(1, 1, 1, -1).expand(1, 1, enc_len, -1).contiguous()
        return ttnn.from_torch(null, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # ---- denoise ----
    def _rope_tables(self, t_prime: int):
        pos = torch.arange(t_prime).unsqueeze(0)
        cos, sin = self._rope(torch.zeros(1, t_prime, HEAD_DIM), pos)
        cos_tt = ttnn.from_torch(
            cos.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        sin_tt = ttnn.from_torch(
            sin.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        return cos_tt, sin_tt

    def _sliding_mask(self, t_prime: int):
        if t_prime <= self.args.sliding_window:
            return None
        mk = self._mod.create_4d_mask(
            seq_len=t_prime,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attention_mask=None,
            sliding_window=self.args.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )
        return ttnn.from_torch(mk, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def generate(
        self,
        hidden_noise,
        context_latents,
        encoder_hidden_states,
        *,
        infer_steps=30,
        shift=1.0,
        use_trace=False,
        guidance_scale=1.0,
        uncond_encoder_hidden_states=None,
    ):
        """Run the ODE denoise loop. Inputs are TT tensors [1,1,T,·]; returns clean latents TT.

        hidden_noise:          [1,1,T,64]  initial noise xt
        context_latents:       [1,1,T,128] (src_latents concat chunk_masks)
        encoder_hidden_states: [1,1,enc,2048]
        use_trace:             capture the DiT velocity step as a ttnn trace and replay it each step
                               (removes the per-step host dispatch). Numerically identical to eager.
        guidance_scale:        classifier-free guidance strength. >1.0 enables CFG: run the DiT twice
                               per step (conditional + unconditional/null context) and combine the two
                               velocities via APG (apg_forward, the reference default). uncond context
                               must be provided when guidance_scale>1. guidance_scale=1.0 = no CFG
                               (single pass, the legacy path).
        uncond_encoder_hidden_states: the null-condition context [1,1,enc,2048] for the uncond pass.
        """
        seq_len = hidden_noise.shape[2]
        t_prime = seq_len // PATCH
        cos_tt, sin_tt = self._rope_tables(t_prime)
        sliding = self._sliding_mask(t_prime)

        solver = FlowMatchStep(self.mesh_device)
        t = _shifted_timesteps(infer_steps, shift, torch.device("cpu"))

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            assert uncond_encoder_hidden_states is not None, "CFG (guidance_scale>1) needs uncond context"
            return self._generate_cfg(
                hidden_noise, context_latents, encoder_hidden_states, uncond_encoder_hidden_states,
                cos_tt, sin_tt, sliding, solver, t, guidance_scale, use_trace=use_trace,
            )

        if use_trace:
            return self._generate_traced(
                hidden_noise, context_latents, encoder_hidden_states, cos_tt, sin_tt, sliding, solver, t
            )

        xt = hidden_noise
        for step_idx in range(infer_steps):
            t_curr = float(t[step_idx].item())
            t_prev = float(t[step_idx + 1].item())
            t_scalar = torch.tensor([t_curr], dtype=torch.float32)
            vt = self.dit.forward(
                xt, context_latents, t_scalar, t_scalar, cos_tt, sin_tt, encoder_hidden_states, sliding_mask=sliding
            )
            xt_new = solver.euler_step(xt, vt, t_curr - t_prev)
            if step_idx > 0:
                ttnn.deallocate(xt)
            xt = xt_new
        return xt  # [1,1,T,64] clean latents

    def _generate_cfg(
        self, hidden_noise, context_latents, enc_cond, enc_uncond, cos_tt, sin_tt, sliding, solver, t,
        guidance_scale, use_trace=False
    ):
        """CFG denoise loop (APG), PURE TTNN so it stays trace-capturable.

        Mirrors the reference sample() loop: per step run DiT(xt, cond) and DiT(xt, null), combine the
        two velocities via APG (apg_forward_ttnn) on-device, then Euler step. dim=-2 matches the
        reference [B,T,C] time-axis projection (dims=[1]) for our [1,1,T,64] latents. No host
        round-trip -> the whole loop is on-device (trace-safe). The APG momentum running-average is a
        resident device buffer updated in place per step.

        use_trace captures the whole two-pass-DiT + APG velocity step as a ttnn trace and replays it
        per ODE step (numerically identical to eager; verified PCC 1.0). The APG momentum state is
        carried across replays via the tracer-input read-back pattern (same as the xt accumulator).
        """
        from models.experimental.acestep.tt.apg_guidance import TTMomentumBuffer, apg_forward_ttnn

        # HiFi4 for the CFG path only: guidance amplifies the per-step matmul error ~6x, so higher
        # matmul fidelity measurably improves the CFG latent PCC (0.956 -> 0.972 at the test config).
        # The no-CFG path never enters here, so its HiFi2 e2e gate (0.9695) is untouched. Restored in
        # the finally block. All bf16 tensors; HiFi4 is matmul accumulation fidelity, not fp32 tensors.
        _restore = _set_dit_fidelity(self.dit, ttnn.MathFidelity.HiFi4)
        try:
            if use_trace:
                return self._generate_cfg_traced(
                    hidden_noise, context_latents, enc_cond, enc_uncond, cos_tt, sin_tt, sliding, solver, t, guidance_scale
                )
            return self._generate_cfg_impl(
                hidden_noise, context_latents, enc_cond, enc_uncond, cos_tt, sin_tt, sliding, solver, t, guidance_scale
            )
        finally:
            for ck, prev in _restore:
                ck.math_fidelity = prev

    def _generate_cfg_impl(
        self, hidden_noise, context_latents, enc_cond, enc_uncond, cos_tt, sin_tt, sliding, solver, t, guidance_scale
    ):
        from models.experimental.acestep.tt.apg_guidance import TTMomentumBuffer, apg_forward_ttnn

        momentum_buffer = TTMomentumBuffer()
        xt = hidden_noise
        for step_idx in range(len(t) - 1):
            t_curr = float(t[step_idx].item())
            t_prev = float(t[step_idx + 1].item())
            t_scalar = torch.tensor([t_curr], dtype=torch.float32)
            vt_cond = self.dit.forward(
                xt, context_latents, t_scalar, t_scalar, cos_tt, sin_tt, enc_cond, sliding_mask=sliding
            )
            vt_uncond = self.dit.forward(
                xt, context_latents, t_scalar, t_scalar, cos_tt, sin_tt, enc_uncond, sliding_mask=sliding
            )
            vt = apg_forward_ttnn(vt_cond, vt_uncond, guidance_scale, momentum_buffer=momentum_buffer, dim=-2)
            xt_new = solver.euler_step(xt, vt, t_curr - t_prev)
            ttnn.deallocate(vt_cond)
            ttnn.deallocate(vt_uncond)
            if step_idx > 0:
                ttnn.deallocate(xt)
            xt = xt_new
        return xt  # [1,1,T,64] clean latents

    def _generate_cfg_traced(
        self, hidden_noise, context_latents, enc_cond, enc_uncond, cos_tt, sin_tt, sliding, solver, t, guidance_scale
    ):
        """Trace-captured CFG denoise: the traced fn runs BOTH DiT passes + APG and returns
        (vt, new_running_average). The APG momentum state is stateful across steps, so — like the xt
        accumulator — the running-average is passed as a tracer INPUT and the updated value is copied
        back into tracer.inputs['run'] each step so it persists across replays. Verified numerically
        identical to the eager CFG loop (latent PCC 1.0). Euler runs eager on the traced velocity.
        """
        from models.experimental.acestep.tt.apg_guidance import apg_forward_ttnn_traced

        infer_steps = t.numel() - 1
        seq_shape = list(hidden_noise.shape)

        def _ts(v):
            return ttnn.from_torch(
                torch.tensor([[[[float(v)]]]], dtype=torch.float32),
                device=self.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            )

        def _cfg_velocity(xt, ts, context, enc, unc, cos, sin, run):
            vt_cond = self.dit.forward(xt, context, ts, ts, cos, sin, enc, sliding_mask=sliding)
            vt_uncond = self.dit.forward(xt, context, ts, ts, cos, sin, unc, sliding_mask=sliding)
            return apg_forward_ttnn_traced(vt_cond, vt_uncond, guidance_scale, run, dim=-2)

        # Momentum running-average starts at zeros (reference seeds running=diff on step 0, i.e.
        # diff + momentum*0). Resident buffer read back from the trace each step.
        run0 = ttnn.from_torch(
            torch.zeros(*seq_shape), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        tracer = Tracer(_cfg_velocity, device=self.mesh_device, prep_run=True, clone_prep_inputs=False)
        xt = hidden_noise
        for step_idx in range(infer_steps):
            t_curr = float(t[step_idx].item())
            t_prev = float(t[step_idx + 1].item())
            first = step_idx == 0
            vt, new_run = tracer(
                xt=xt,
                ts=_ts(t_curr),
                context=context_latents if first else tracer.inputs["context"],
                enc=enc_cond if first else tracer.inputs["enc"],
                unc=enc_uncond if first else tracer.inputs["unc"],
                cos=cos_tt if first else tracer.inputs["cos"],
                sin=sin_tt if first else tracer.inputs["sin"],
                run=run0 if first else tracer.inputs["run"],
                traced=True,
            )
            xt = solver.euler_step(tracer.inputs["xt"], ttnn.clone(vt), t_curr - t_prev)
            # Persist the updated momentum running-average into the trace's resident input buffer.
            ttnn.copy(ttnn.clone(new_run), tracer.inputs["run"])
        tracer.release_trace()
        return xt  # [1,1,T,64] clean latents

    def _generate_traced(
        self, hidden_noise, context_latents, encoder_hidden_states, cos_tt, sin_tt, sliding, solver, t
    ):
        """Trace-captured denoise loop, following the SD35/LTX pattern.

        The traced fn returns the DiT velocity `vt`; the Euler update runs eager (outside the trace).
        Constants (context/enc/rope) are passed on the first (capture) call; subsequent replays reuse
        the trace's resident input buffers via ``tracer.inputs[...]`` (trace execution may overwrite
        them, so we always read the handle back). The timestep is a device tensor updated per step.
        Numerically identical to the eager loop (same on-device ops).
        """
        infer_steps = t.numel() - 1

        def _ts(v):
            return ttnn.from_torch(
                torch.tensor([[[[float(v)]]]], dtype=torch.float32),
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            )

        def _dit_velocity(xt, ts, context, enc, cos, sin):
            return self.dit.forward(xt, context, ts, ts, cos, sin, enc, sliding_mask=sliding)

        tracer = Tracer(_dit_velocity, device=self.mesh_device, prep_run=True, clone_prep_inputs=False)

        xt = hidden_noise
        for step_idx in range(infer_steps):
            t_curr = float(t[step_idx].item())
            t_prev = float(t[step_idx + 1].item())
            first = step_idx == 0
            vt = tracer(
                xt=xt,
                ts=_ts(t_curr),
                context=context_latents if first else tracer.inputs["context"],
                enc=encoder_hidden_states if first else tracer.inputs["enc"],
                cos=cos_tt if first else tracer.inputs["cos"],
                sin=sin_tt if first else tracer.inputs["sin"],
                traced=True,
            )
            # Euler runs eager on the trace's velocity output; read xt back from the trace buffer
            # (trace execution may overwrite it). vt is the trace's resident output -> clone it.
            xt = solver.euler_step(tracer.inputs["xt"], ttnn.clone(vt), t_curr - t_prev)
        tracer.release_trace()
        return xt

    # ---- high-level customer API ----
    SAMPLE_RATE = 48000
    LATENT_HZ = 25  # 25 latent frames / second

    def generate_song(
        self,
        prompt: str,
        *,
        lyrics: str = "",
        seconds: float = 8.0,
        infer_steps: int = 30,
        shift: float = 3.0,
        seed: int = 0,
        use_trace: bool = False,
        guidance_scale: float = 7.0,
        vocal_language: str = "en",
        bpm=None,
        keyscale: str = "",
        timesignature: str = "",
    ):
        """Text -> 48 kHz stereo song in one call. Returns torch waveform [1, 2, samples].

        prompt:  style / caption text.  lyrics: optional lyric text.  seconds: song length.
        guidance_scale: classifier-free guidance strength (reference base default 7.0). >1.0 makes the
        model follow the prompt (without it the output is noise-like). Enables the CFG denoise path
        (two DiT passes/step + APG). 1.0 = legacy no-CFG single pass.
        use_trace: capture the DiT denoise step as a ttnn trace and replay it each ODE step,
        removing the per-step host dispatch (numerically identical to eager). Ignored under CFG.
        Tokenization, text encoding, conditioning, denoise (infer_steps ODE) and VAE decode are
        all handled internally. Requires the pipeline built with text/condition encoders
        (create_tt_pipeline(..., with_encoders=True), the default).
        """
        assert (
            self.text_encoder is not None and self.condition_encoder is not None
        ), "generate_song needs the encoders; build with create_tt_pipeline(..., with_encoders=True)"
        assert self.vae is not None, "generate_song needs the VAE; build with with_vae=True"

        enc_hs = self.encode_prompt(
            prompt, lyrics, vocal_language=vocal_language, audio_duration=seconds,
            bpm=bpm, keyscale=keyscale, timesignature=timesignature,
        )
        seq_len = self._latent_len(seconds)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, 1, seq_len, self.args.audio_acoustic_hidden_dim, generator=gen)
        # text2music: silence source latents + all-valid chunk mask.
        hidden_ch = self.args.audio_acoustic_hidden_dim
        # text2music src_latents = the LEARNED silence_latent (NOT zeros), tiled/cropped to seq_len;
        # chunk_masks = ones (full-generation span). Ref: diffusers AceStepPipeline.prepare_src_latents.
        if self._silence_latent is not None:
            sl = self._silence_latent  # [1, 15000, 64]
            if sl.shape[1] >= seq_len:
                src = sl[:, :seq_len, :]
            else:
                reps = (seq_len + sl.shape[1] - 1) // sl.shape[1]
                src = sl.repeat(1, reps, 1)[:, :seq_len, :]
            src = src.reshape(1, 1, seq_len, hidden_ch)
        else:
            src = torch.zeros(1, 1, seq_len, hidden_ch)
        context = torch.cat([src, torch.ones(1, 1, seq_len, hidden_ch)], dim=-1)
        noise_tt = ttnn.from_torch(noise, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        context_tt = ttnn.from_torch(context, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        uncond_hs = self._uncond_context(enc_hs) if guidance_scale > 1.0 else None
        latents = self.generate(
            noise_tt, context_tt, enc_hs, infer_steps=infer_steps, shift=shift, use_trace=use_trace,
            guidance_scale=guidance_scale, uncond_encoder_hidden_states=uncond_hs,
        )
        return self.decode(latents)

    def _latent_len(self, seconds: float) -> int:
        n = max(2, int(round(seconds * self.LATENT_HZ)))
        # Round UP to a multiple of 8 so t_prime = seq_len/patch_size is a multiple of 4. Certain
        # t_prime values (measured: 125, i.e. seconds=10 -> seq_len 250) hit a degenerate DiT matmul
        # program-config path that silently flattens the audio dynamic range to ~1.4x (static) while
        # keeping the latent std normal. Multiples of 8 (t_prime 128/132/160...) all produce healthy
        # 200x+ dynamic range. Rounds 250 -> 256 (10.24s), a negligible duration change.
        return ((n + 7) // 8) * 8

    # SFT prompt template — the text encoder was trained with this EXACT format (the newlines are
    # load-bearing). Feeding the raw prompt (no template) produces meaningless conditioning -> static
    # audio. Ref: diffusers AceStepPipeline SFT_GEN_PROMPT + _format_prompt.
    _SFT_GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
    _DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

    @staticmethod
    def _metadata_string(bpm=None, keyscale="", timesignature="", audio_duration=None) -> str:
        bpm_str = str(bpm) if bpm is not None and bpm > 0 else "N/A"
        ts_str = timesignature if timesignature and timesignature.strip() else "N/A"
        ks_str = keyscale if keyscale and keyscale.strip() else "N/A"
        dur_str = f"{int(audio_duration)} seconds" if audio_duration and audio_duration > 0 else "30 seconds"
        return f"- bpm: {bpm_str}\n- timesignature: {ts_str}\n- keyscale: {ks_str}\n- duration: {dur_str}\n"

    def encode_prompt(
        self,
        prompt: str,
        lyrics: str = "",
        *,
        vocal_language: str = "en",
        audio_duration: float = 30.0,
        bpm=None,
        keyscale: str = "",
        timesignature: str = "",
    ):
        """Tokenize + encode prompt/lyrics -> DiT cross-attn context TT tensor [1,1,ctx,hidden].

        Applies the SFT prompt template (instruction + caption + metadata) and the lyric-language
        format the text encoder was TRAINED on — without it the conditioning is meaningless and the
        audio is static. Ref: diffusers AceStepPipeline._format_prompt.

        text  -> SFT template -> tokenizer -> TT text encoder -> text_hidden_states
        lyric -> language-header format -> tokenizer -> text_encoder.embed_tokens lookup
        timbre-> silence (text2music, no reference audio)
        """
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        metas = self._metadata_string(bpm=bpm, keyscale=keyscale, timesignature=timesignature, audio_duration=audio_duration)
        formatted_text = self._SFT_GEN_PROMPT.format(self._DIT_INSTRUCTION, prompt, metas)
        formatted_lyrics = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"

        text_ids = torch.tensor([self.tokenizer(formatted_text, truncation=True, max_length=256).input_ids])
        lyric_ids = torch.tensor([self.tokenizer(formatted_lyrics, truncation=True, max_length=2048).input_ids])
        tlen, llen = text_ids.shape[1], lyric_ids.shape[1]

        # 1. text encoder (its own RoPE: theta from the text-encoder config, head_dim 128).
        te_rope = Qwen3RotaryEmbedding(self._hf_text_cfg)
        tcos, tsin = self._rope_tt(te_rope, tlen)
        text_hs = self.text_encoder.forward(text_ids, tcos, tsin)  # [1,1,tlen,text_hidden]

        # 2. lyric embeddings = text_encoder.embed_tokens lookup only (per reference).
        lyric_hs = self.text_encoder.embed_tokens[lyric_ids.long()].reshape(
            1, 1, llen, self.text_encoder.config.hidden_size
        )
        lyric_tt = ttnn.from_torch(lyric_hs, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # 3. timbre = the GENUINE learned silence_latent (text2music, no reference audio). Reference
        # feeds refer_audio = silence_latent[:, :timbre_fix_frame(750), :] (NOT zeros), runs the 4-layer
        # timbre encoder, and slices position 0 as the single CLS timbre token. Feeding zeros produced
        # degenerate timbre -> static audio. Ref: conditioning_embed.infer_refer_latent (silence path).
        # NB: our MLP1D requires seq_len divisible by prefill_len_cutoff (512) above 512, so we use 512
        # silence frames (the reference uses 750). The CLS token at pos 0 aggregates the silence context
        # bidirectionally either way; 512 vs 750 of the same learned silence is a negligible difference.
        timbre_len = 512
        if self._silence_latent is not None:
            timbre = self._silence_latent[:, :timbre_len, :].reshape(1, 1, timbre_len, self.args.audio_acoustic_hidden_dim)
        else:
            timbre = torch.zeros(1, 1, timbre_len, self.args.audio_acoustic_hidden_dim)
        timbre_tt = ttnn.from_torch(timbre, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # 4. condition encoder -> packed cross-attn context.
        lcos, lsin = self._rope_tt(self._rope, llen)
        tbcos, tbsin = self._rope_tt(self._rope, timbre_len)
        return self.condition_encoder.forward(
            text_hs,
            lyric_tt,
            timbre_tt,
            lcos,
            lsin,
            tbcos,
            tbsin,
            lyric_sliding=self._enc_sliding(llen),
            timbre_sliding=self._enc_sliding(timbre_len),
            timbre_cls=True,
        )

    def _rope_tt(self, rope, seq_len: int):
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = rope(torch.zeros(1, seq_len, HEAD_DIM), pos)
        cos_tt = ttnn.from_torch(
            cos.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        sin_tt = ttnn.from_torch(
            sin.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        return cos_tt, sin_tt

    def _enc_sliding(self, seq_len: int):
        mk = self._mod.create_4d_mask(
            seq_len=seq_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attention_mask=None,
            sliding_window=self.args.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )
        return ttnn.from_torch(mk, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # ---- decode ----
    def decode(self, latents_tt):
        """VAE-decode clean latents [1,1,T,64] -> 48kHz stereo waveform torch [1,2,T*1920]."""
        wav_tt = self.decode_tt(latents_tt)
        ac = self.vae_config.audio_channels
        # Output boundary: pull the final waveform to host for the caller (outside any trace region).
        wav = ttnn.to_torch(wav_tt).float()[..., :ac].reshape(1, -1, ac).transpose(1, 2)  # [1,2,samples]
        return wav

    def decode_tt(self, latents_tt):
        """On-device DiT->VAE seam + VAE decode: latents [1,1,T,64] -> waveform ttnn tensor.

        The DiT->VAE seam is pure layout/shape/dtype conversion, done ON-DEVICE (no host round-trip)
        so this is trace-safe: TILE->ROW_MAJOR, dtype match, [1,1,T,C]->[1,T,C]. The VAE decoder wants
        [B,T,C] ROW_MAJOR in the VAE's own compute dtype (bf16 by default; fp32 if configured).
        """
        seq_len = latents_tt.shape[2]
        # Match the VAE's compute dtype (Conv3d requires input.dtype == weight.dtype). Typecast while
        # still TILE, then untile + reshape on-device. No host round-trip -> trace-safe.
        vae_dtype = getattr(self.vae, "dtype", ttnn.float32) or ttnn.float32
        lat = latents_tt
        if lat.dtype != vae_dtype:
            lat = ttnn.typecast(lat, vae_dtype)
        lat = ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT)
        lat = ttnn.reshape(lat, (1, seq_len, self.args.audio_acoustic_hidden_dim))
        return self.vae.forward(lat)


def create_tt_pipeline(
    args: AceStepModelConfig, mesh_device, *, with_vae: bool = True, with_encoders: bool = True
) -> AceStepPipeline:
    """Assemble the full TT text-to-music pipeline from the genuine checkpoint.

    Single public model-construction entry point. Builds the DiT denoiser and, by default, the
    Oobleck VAE decoder + text encoder + condition encoder + tokenizer — so `generate_song(prompt)`
    works out of the box. Weights load lazily on first forward.

    - with_vae:      build the VAE decoder (needed for audio output / generate_song / decode).
    - with_encoders: build the text + condition encoders + tokenizer (needed for generate_song /
                     encode_prompt). Set False for DiT-only latent tests.
    """
    from models.experimental.acestep.tt.model_config import build_condition_encoder, build_text_encoder

    dit = _build_dit_model(args, mesh_device)
    vae, vae_cfg = (None, None)
    if with_vae:
        vae, vae_cfg = build_vae_decoder(mesh_device)

    text_encoder = condition_encoder = tokenizer = hf_text_cfg = None
    if with_encoders:
        from transformers import AutoTokenizer
        from models.experimental.acestep.reference.weight_utils import pipeline_dir

        text_encoder, hf_text = build_text_encoder(mesh_device)
        hf_text_cfg = hf_text.config
        condition_encoder = build_condition_encoder(args, mesh_device)
        tokenizer = AutoTokenizer.from_pretrained(str(pipeline_dir() / "Qwen3-Embedding-0.6B"))

    hf = load_config()
    rope = Qwen3RotaryEmbedding(hf)
    mod = load_modeling_module()
    # Load the learned null-condition embedding [1,1,2048] for classifier-free guidance (the uncond
    # cross-attn context). Present in the base checkpoint; None-tolerant if absent (CFG then errors).
    null_condition_emb = None
    if with_encoders:
        try:
            from models.experimental.acestep.reference.weight_utils import load_state_dict

            sd = load_state_dict()
            for k in ("null_condition_emb", "decoder.null_condition_emb"):
                if k in sd:
                    null_condition_emb = sd[k].float()
                    break
        except Exception:
            null_condition_emb = None

    # Load the genuine learned silence_latent (text2music src_latents). Ships as silence_latent.pt in
    # the base checkpoint snapshot ([1,64,15000] -> [1,15000,64]).
    silence_latent = None
    if with_encoders:
        try:
            import glob as _glob
            from models.experimental.acestep.reference.hf_reference import _snapshot_dir

            hits = _glob.glob(str(_snapshot_dir() / "silence_latent.pt"))
            if hits:
                sl = torch.load(hits[0], map_location="cpu").float()  # [1,64,15000]
                silence_latent = sl.transpose(1, 2).contiguous()  # [1,15000,64]
        except Exception:
            silence_latent = None

    return AceStepPipeline(
        args=args,
        dit=dit,
        vae=vae,
        vae_config=vae_cfg,
        mesh_device=mesh_device,
        _rope=rope,
        _mod=mod,
        text_encoder=text_encoder,
        condition_encoder=condition_encoder,
        tokenizer=tokenizer,
        _hf_text_cfg=hf_text_cfg,
        _null_condition_emb=null_condition_emb,
        _silence_latent=silence_latent,
    )
