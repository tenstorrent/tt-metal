# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE shared chained TTNN pipeline for `mistralai/Voxtral-4B-TTS-2603`: text -> 24 kHz speech.

Both `demo/` and `tests/e2e/` import and call the functions here, so a passing test guarantees a
working demo -- there is exactly one copy of the wiring.

`build_pipeline` builds three sections on the device, each from the HF reference's own weights:

  text      `text_stack.TextStack`        the 26-layer Mistral decoder, KV-cached; every layer is
                                          the fused `tt/modules/layer.py` block
  acoustic  `acoustic_stage.AcousticStage` the flow-matching sampler: one hidden state -> one frame
                                          of 37 codes (7 Euler steps with classifier-free guidance)
  vocode    `vocode_stage.VocodeStage`     the codec decoder: all codes -> the waveform, every row
                                          through `tt/modules/voxtral_t_t_s_audio_tokenizer.py`

`VoxtralTTSPipeline.run_text_to_speech` chains them: prefill the voiced prompt, then per frame one
acoustic step and one decode step fed the frame's own audio-token embedding, until every row emits
`end_audio`, then one vocode pass.

STAGES. The reference is a `MistralForCausalLM` subclass with `is_encoder_decoder=False`, which
gives `[prefill, decode]`; the model card's `pipeline_tag: text-to-speech` adds `[vocode]`.
`acoustic` is the fourth because `params.json` carries
`multimodal.audio_model_args.acoustic_transformer_args` as its OWN sub-config with its OWN
repeated stack. Each stage has a trace contract (`<stage>_trace_inputs/setup/step/items`) that the
perf test and `trace_capture_selftest` drive.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common

PIPELINE_STAGES = ["prefill", "decode", "acoustic", "vocode"]

# The pinned trace capacity for the sequence axis. The VARIABLE dim is the sequence length, whose
# bound is `config.max_position_embeddings` (128000) -- far past anything a trace region holds, so
# the stage pins it instead. 256 is tile-aligned and covers the real voiced speech request (the
# default voice's 147-token block + 18-token text + controls = 170 tokens; the longest preset, it_male,
# gives 191). A longer prompt is refused by `prefill_trace_setup`.
DEFAULT_TRACE_CAPACITY = int(os.environ.get("VOXTRAL_TRACE_C", "256"))

# Frames the `vocode` stage is captured at. The codec upsamples 8x, so 32 frames become 256 rows,
# inside the codec's prebuilt ALiBi mask.
DEFAULT_VOCODE_CAPACITY = int(os.environ.get("VOXTRAL_TRACE_VOCODE_C", "32"))

# Classifier-free guidance strength. `params.json` sets `p_uncond: 0.0` and ships no sampling
# alpha, and the reference takes it as a per-batch `[B]` argument rather than reading a config
# field, so the value is the caller's. 3.0 is this package's default and is applied IDENTICALLY to
# the TT chain and the golden, so it cannot tilt the comparison.
DEFAULT_CFG_ALPHA = float(os.environ.get("VOXTRAL_CFG_ALPHA", "3.0"))


def _tile_ceil(value: int) -> int:
    return -(-int(value) // 32) * 32


class VoxtralTTSPipeline:
    """The resident pipeline object: the text-to-speech chain plus the per-stage trace contract.

    Every repeated stack is reachable as a plain Python list (`self.stacks()`), and the HF reference
    stays on `.reference_model` -- it is ground truth for how many sections the model has and how
    deep each is.
    """

    PIPELINE_STAGES = PIPELINE_STAGES

    def __init__(
        self,
        device,
        hf_model,
        text=None,
        acoustic=None,
        vocode=None,
        batch=None,
        trace_capacity=None,
        vocode_capacity=None,
    ):
        self.device = device
        self.reference_model = hf_model
        self.config = hf_model.config
        self.text = text
        self.acoustic = acoustic
        self.vocode = vocode
        self.batch = common.DEFAULT_BATCH if batch is None else int(batch)
        self.tokenizer = common.load_tokenizer()
        self.trace_capacity = DEFAULT_TRACE_CAPACITY if trace_capacity is None else int(trace_capacity)
        self.vocode_capacity = DEFAULT_VOCODE_CAPACITY if vocode_capacity is None else int(vocode_capacity)
        self.n_special = common.n_audio_special_tokens(hf_model)
        self.stop_token_id = common.audio_stop_token_id(hf_model)
        self.n_acoustic = int(hf_model.acoustic_transformer.model_args.n_acoustic_codebook)
        self.n_codebooks = self.n_acoustic + 1
        self._stage_buffers: dict = {}
        self._traces: dict = {}
        # THE DEFAULT CFG WEIGHT IS BUILT HERE, NOT IN THE FORWARD. `torch.full` is host compute
        # wherever it runs, and the fully-on-device check watches the forward: building the
        # default there fired `aten.full.default` inside the observed region. It is a constant of
        # the pipeline, so it belongs with the weights.
        self._default_cfg_host = torch.full((self.batch,), DEFAULT_CFG_ALPHA)
        self._default_cfg_tt = ttnn.from_torch(
            self._default_cfg_host.reshape(self.batch, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    # ---- structure ---------------------------------------------------------------------

    def stacks(self) -> dict:
        """stage -> the repeated-block list that stage owns, for anything sizing or capping depth.

        `prefill` and `decode` share ONE stack: they are two phases of the same 26 layers, not two
        stacks. The reference is the authority on the section structure and is reachable at
        `.reference_model` (`model.layers` 26, `acoustic_transformer.layers` 3,
        `audio_tokenizer.decoder_blocks` 8).
        """
        out = {}
        if self.text is not None:
            out["prefill"] = self.text.blocks
            out["decode"] = self.text.blocks
        if self.acoustic is not None:
            out["acoustic"] = self.acoustic.blocks
        if self.vocode is not None:
            out["vocode"] = self.vocode.blocks
        return out

    # ---- helpers -----------------------------------------------------------------------

    def prepare_prompt(self, prompt):
        """THE ONE MARSHALLING POINT: the whole prompt onto the device, ONCE, before prefill.

        Shape and dtype prep, no compute. It is called exactly once per call, ahead of the
        prefill, and never from inside a decode loop -- the loop's next input is
        `vocode.audio_token_embedding(codes)`, device tensor to device tensor, so no token is ever
        rebuilt on the host or re-uploaded. Named for what it takes (a prompt) rather than for its
        dtype, because "upload some ids" is exactly the thing this pipeline must not be doing per
        step.
        """
        return ttnn.from_torch(
            prompt.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

    def _positions(self, start: int, length: int, batch: int):
        pos = torch.arange(start, start + length, dtype=torch.int32).unsqueeze(0).expand(batch, -1)
        return ttnn.from_torch(pos.contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def default_inputs(self, batch=None, seq_len=None):
        """The REAL task input: `batch` INDEPENDENT prompts from the model's own tokenizer.

        `[bos] + tekken_encode(text) + [begin_audio_token_id]` per row. The final
        `[BEGIN_AUDIO]` (id 25) is what tells the backbone to start emitting audio frames.
        """
        batch = self.batch if batch is None else int(batch)
        text_ids, texts = common.build_batch_inputs(batch=batch, seq_len=seq_len or common.DEFAULT_SEQ_LEN)
        rows, length = text_ids.shape
        # Allocated at its final width and filled, rather than grown by a concat: this is input
        # PREP either way, but the prompt's shape is a property of the call and writing it that
        # way says so.
        ids = torch.full((rows, length + 1), common.begin_audio_token_id(), dtype=text_ids.dtype)
        ids[:, :length] = text_ids
        return ids, texts

    def speech_inputs(self, texts=None, voice=None, batch=None):
        """The real TTS input for this pipeline's batch -- see module-level `speech_inputs`."""
        return speech_inputs(texts=texts, voice=voice, batch=self.batch if batch is None else batch)

    def stage_voice(self, audio_mask, voice_embedding, input_ids):
        """Upload the voice once, as persistent device constants, OUTSIDE the forward. `input_ids`
        (host) lets the prefill run the prompt prefix every row shares once instead of per row; the
        staged voice is then valid for exactly those ids."""
        return self.text.stage_voice(audio_mask, voice_embedding, input_ids=input_ids)

    def noise(self, max_frames: int, batch=None, seed: int = 0):
        """The flow-matching sampler's noise input, drawn ONCE on the host.

        The reference draws it inside `decode_one_frame`, which makes its output a function of the
        RNG; handing the same draw to both sides is what makes the comparison about the weights.
        """
        from models.demos.voxtral_4b_tts_2603.reference import golden

        return golden.draw_noise(self.batch if batch is None else int(batch), self.n_acoustic, max_frames, seed=seed)

    # ---- text to speech ----------------------------------------------------------------

    def run_text_to_speech(
        self,
        input_ids=None,
        x0=None,
        cfg_alpha=None,
        max_frames=None,
        collect=False,
        voice=None,
    ):
        """THE REAL TASK: tokenized text -> a 24 kHz waveform, all on device.

        An EXPLICIT chain over the three sections. Each stage is fed the PREVIOUS TT stage's real
        output; no reference tensor is ever spliced in at a joint, because that would hide exactly
        the wiring bugs this path exists to catch.

        The loop is the model's own iteration -- audio frames at 12.5 Hz -- and all `batch`
        samples ride the LEADING axis through it: ONE program per frame feeds every sample, and
        there is no python loop over samples anywhere.

        Stops on the model's OWN stop signal (`AudioSpecialTokens.end_audio`, read off the
        reference) once every row has emitted it, bounded by a derived safety cap.
        """
        if self.text is None or self.acoustic is None or self.vocode is None:
            raise RuntimeError("pipeline was built without its text, acoustic and vocode sections")

        if input_ids is None:
            input_ids, _ = self.default_inputs()
        batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        if max_frames is None:
            max_frames, _ = common.resolve_max_frames(self.reference_model)
        if int(max_frames) > self.vocode.max_frames:
            # Checked BEFORE the decode: the codec's mask is a build-time constant, so a longer run would
            # decode every frame and then fail in `vocode.decode`, losing the whole output.
            raise ValueError(
                f"max_frames {max_frames} exceeds the codec's {self.vocode.max_frames}-frame limit "
                f"({self.vocode.max_frames_provenance})"
            )
        if voice is not None and not torch.equal(voice["ids"], input_ids):
            # The shared-prefix prefill was laid out for the staged ids (how many leading tokens every
            # row shares); other texts would silently run on that layout.
            raise ValueError("`voice` was staged for different input_ids; call stage_voice with these ids")
        if x0 is None:
            x0 = self.noise(max_frames, batch=batch)

        # ONE UPLOAD, THEN SLICE ON DEVICE. Indexing the `[F, B, 36]` noise per frame on the host
        # (`x0[i]`) fires `aten.select.int` once per frame INSIDE the forward, which the
        # fully-on-device check counts as host compute. The whole block goes up once and each
        # frame's row comes off it with `ttnn.slice`.
        x0_all = ttnn.from_torch(
            x0.reshape(max_frames, batch, self.n_acoustic).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        x0_tt = [
            ttnn.reshape(
                ttnn.slice(x0_all, [i, 0, 0], [i + 1, batch, self.n_acoustic]),
                [batch, self.n_acoustic],
            )
            for i in range(max_frames)
        ]
        if cfg_alpha is None and batch == self.batch:
            # Both forms were built at __init__; naming them here calls nothing.
            cfg_alpha, cfg_tt = self._default_cfg_host, self._default_cfg_tt
        else:
            if cfg_alpha is None:
                cfg_alpha = torch.full((batch,), DEFAULT_CFG_ALPHA)
            cfg_tt = ttnn.from_torch(
                cfg_alpha.reshape(batch, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        self.text.reset_cache()
        ids_tt = self.prepare_prompt(input_ids)

        # --- prefill: the prompt through the 26-layer stack, seeding the KV cache.
        # NO EXPLICIT POSITIONS. A prefill's positions ARE 0..S-1, which is the rotary module's own
        # default branch -- and that branch slices a float32 table, where handing it the same
        # positions explicitly takes the `ttnn.embedding` GATHER instead, which `ttnn` requires to
        # be bfloat16. Rope was then the only bfloat16 term in a float32 residual stream and the
        # ~0.4% it costs compounds over 26 layers. It also keeps `torch.arange` out of the forward.
        if voice is None:
            prefill_hidden, llm_hidden = self.text.prefill(ids_tt)
        else:
            # THE VOICE GOES IN WHERE THE PLACEHOLDERS ARE, not in front of them. The prompt carries
            # a contiguous block of `[AUDIO]` ids whose EMBEDDINGS the speaker's voice replaces --
            # the token ids stay, only the rows change, so positions and the causal mask are the
            # prompt's own. `voice` was staged on device by `stage_voice` before this call, so the
            # substitution here is two device ops and no host compute.
            # Everything below runs from the PRODUCTION prefill -- the one the traced stages run,
            # whose last block computes only the rows read out. With `collect`, a whole-prompt
            # prefill runs first for the per-stage check's hidden state; the second re-seeds the cache.
            prefill_hidden = self.text.prefill_voiced(ids_tt, voice)[0] if collect else None
            _, llm_hidden = self.text.prefill_voiced(ids_tt, voice, need_hidden=False)

        frames, diagnostics = [], []
        # The stop test is accumulated ON DEVICE. `finished |= semantic == stop_id` in torch would
        # be host compute inside the forward, which the trace capture forbids; here the
        # comparison and the OR are ttnn ops over a resident `[batch, 1]` flag and only a single
        # reduced scalar crosses to the host, as a loop-control decision rather than as math.
        finished_flag = ttnn.zeros([batch, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        position = prompt_len
        stop_reason = f"max_frames={max_frames}"
        live_frames = ttnn.zeros([batch, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

        for step in range(max_frames):
            # --- acoustic: one frame of 37 codes from this hidden state
            probe = {} if collect else None
            codes = self.acoustic.decode_frame(llm_hidden, x0_tt[step], cfg_tt, probe=probe)
            frames.append(codes)
            if collect:
                diagnostics.append(
                    {
                        "llm_hidden": ttnn.to_torch(llm_hidden).to(torch.float32).clone(),
                        "x_final": ttnn.to_torch(probe["x_final"]).to(torch.float32).clone(),
                        "semantic_logits": ttnn.to_torch(probe["semantic_logits"]).to(torch.float32).clone(),
                    }
                )

            # Has every row emitted the stop token yet? Computed on device; one scalar crosses.
            # Inside a captured trace the loop runs at a FIXED capacity instead, so a
            # variable-length decode never makes the traced shapes dynamic.
            semantic = ttnn.typecast(
                ttnn.to_layout(ttnn.slice(codes, [0, 0], [batch, 1]), ttnn.TILE_LAYOUT), ttnn.float32
            )
            hit = ttnn.eq(semantic, float(self.stop_token_id))
            finished_flag = ttnn.logical_or(finished_flag, hit)
            # WHERE EACH ROW ENDED, counted on device: a row's end frame is the number of frames it
            # was still live for, read ONCE after the loop. Rows that end early keep generating until
            # the whole batch is done, and `trim_to_end` cuts each row back to its own end.
            #
            # ONE scalar crosses to the host per frame, straight off the 0-d reduction, and it is
            # LOOP CONTROL -- not a token feed. The codes never leave the device: the next input is
            # `vocode.audio_token_embedding(codes)` below. Indexing the flag on the host instead
            # (`[0]`, `.to(bool)`) fires `aten.select` / `aten._to_copy` inside the forward.
            live_frames = ttnn.add(live_frames, ttnn.rsub(finished_flag, 1.0))
            n_finished = int(ttnn.to_torch(ttnn.sum(finished_flag)))
            if n_finished >= batch:
                stop_reason = f"every row emitted end_audio (id {self.stop_token_id}) at frame {step}"
                break
            if step + 1 == max_frames:
                break

            # --- feed the frame back: the audio-token embedding is the next input, and one more
            #     decode step against the resident KV cache produces the next hidden state.
            embeds = self.vocode.audio_token_embedding(ttnn.reshape(codes, [batch, self.n_codebooks, 1]))
            llm_hidden = self.text.decode_step(embeds, position)
            position += 1

        # --- vocode: the whole code sequence -> the waveform
        framed = [ttnn.reshape(f, [batch, self.n_codebooks, 1]) for f in frames]
        # `ttnn.concat` of a single tensor is not worth asking for: a run that stops on the very
        # first frame is legal (every row emitting end_audio immediately) and must not crash here.
        codes_all = framed[0] if len(framed) == 1 else ttnn.concat(framed, dim=-1)
        waveform_tt = self.vocode.decode(codes_all)

        # Loop finished: one readback of the per-row counts. A row that never stopped reports -1.
        counts = ttnn.to_torch(live_frames).tolist()
        flags = ttnn.to_torch(finished_flag).tolist()
        end_frame = [int(round(c[0])) if f[0] > 0.5 else -1 for c, f in zip(counts, flags)]

        codes_host = ttnn.to_torch(codes_all).to(torch.int64)
        waveform = ttnn.to_torch(waveform_tt).to(torch.float32)
        return {
            "input_ids": input_ids,
            "codes": codes_host,
            "waveform": waveform,
            "prefill_hidden": None if prefill_hidden is None else ttnn.to_torch(prefill_hidden).to(torch.float32),
            "frames_decoded": int(codes_host.shape[-1]),
            "stop_reason": stop_reason,
            # Per-row length, so a caller can cut each sample at its OWN end instead of the batch's.
            "end_frame": [int(e) for e in end_frame],
            "sampling_rate": self.vocode.sampling_rate,
            "batch": batch,
            "x0": x0,
            "cfg_alpha": cfg_alpha,
            "max_frames": max_frames,
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------------------------
    # Trace contract -- one set of hooks per stage in PIPELINE_STAGES
    # ------------------------------------------------------------------------------------
    #
    # Each stage pins its VARIABLE dim (the sequence axis) to a fixed capacity C and pre-uploads
    # the padded input plus every shape-dependent constant into PERSISTENT device buffers OUTSIDE
    # the trace. The constant VALUES come from the HF reference itself, so they match the golden
    # exactly rather than being re-derived. `<stage>_trace_step()` is then one host-op-free
    # forward at the fixed shape that reads ONLY those buffers.

    def _reference_rope(self, capacity: int):
        """`(cos, sin)` for positions 0..C from the reference's OWN `rotary_emb`.

        Taking the values from the reference rather than re-deriving `inv_freq` is what keeps the
        traced constants bit-equal to the golden -- this checkpoint's `rope_theta` is 1e6 and the
        4.x/5.x spelling difference silently lands a re-derivation at 1e4.
        """
        rotary = self.reference_model.model.rotary_emb
        positions = torch.arange(capacity, dtype=torch.long).unsqueeze(0)
        probe = torch.zeros(1, capacity, 1, dtype=torch.float32)
        with torch.no_grad():
            cos, sin = rotary(probe, positions)
        return cos, sin

    def _reference_causal_mask(self, capacity: int, real_len: int, batch: int):
        """An additive causal mask that ALSO blocks the pad, so `[0:real_len]` is unchanged."""
        mask = torch.zeros(batch, 1, capacity, capacity, dtype=torch.float32)
        causal = torch.ones(capacity, capacity, dtype=torch.bool).triu(1)
        mask.masked_fill_(causal, float("-inf"))
        if real_len < capacity:
            mask[:, :, :, real_len:] = float("-inf")
        return mask

    # -- prefill ------------------------------------------------------------------------

    def prefill_trace_inputs(self):
        """ZERO-ARG. Exactly the argument `prefill_trace_setup` takes.

        The STANDARD, model-agnostic seam the perf engine calls to obtain a stage's inputs with no
        per-model knowledge: all the model-specific assembly lives here, behind this fixed name.
        The REAL speech request the e2e test and demo drive: the voiced prompt from the model's
        own tokenizer layout and the checkpoint's preset voice embedding.
        """
        input_ids, audio_mask, voice_embedding, _ = self.speech_inputs()
        return {"input_ids": input_ids, "audio_mask": audio_mask, "voice_embedding": voice_embedding}

    def prefill_trace_setup(self, inputs):
        input_ids = inputs["input_ids"]
        batch, real_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        if real_len > self.trace_capacity:
            raise ValueError(f"prompt is {real_len} tokens, past the pinned capacity {self.trace_capacity}")
        # Captured at the request's own tile-rounded length (the bucket `prefill_voiced` pads to on
        # the untraced path); the pinned capacity is only the ceiling a bucket may reach.
        capacity = _tile_ceil(real_len)

        padded = torch.zeros(batch, capacity, dtype=input_ids.dtype)
        padded[:, :real_len] = input_ids
        mask = torch.zeros(batch, capacity, dtype=torch.bool)
        mask[:, :real_len] = inputs["audio_mask"]
        cos, sin = self._reference_rope(capacity)

        self._stage_buffers["prefill"] = {
            "ids": self.prepare_prompt(padded),
            "voice": self.stage_voice(mask, inputs["voice_embedding"], input_ids=input_ids),
            "positions": self._positions(0, capacity, batch),
            "cos": ttnn.from_torch(
                cos.reshape(1, 1, capacity, -1).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            ),
            "sin": ttnn.from_torch(
                sin.reshape(1, 1, capacity, -1).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            ),
            "mask": ttnn.from_torch(
                self._reference_causal_mask(capacity, real_len, batch),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            ),
            "capacity": capacity,
            "real_len": real_len,
            "batch": batch,
        }
        return self._stage_buffers["prefill"]

    def prefill_trace_step(self):
        buf = self._stage_buffers["prefill"]
        _, last = self.text.prefill_voiced(buf["ids"], buf["voice"], real_len=buf["real_len"], need_hidden=False)
        return last

    def prefill_trace_items(self):
        """B x C: the 26 blocks process every prompt token of every sample."""
        buf = self._stage_buffers.get("prefill")
        batch = buf["batch"] if buf else self.batch
        capacity = buf["capacity"] if buf else self.trace_capacity
        return int(batch) * int(capacity)

    # -- decode -------------------------------------------------------------------------

    def decode_trace_inputs(self):
        """ZERO-ARG. Same shape as `prefill_trace_inputs` -- the decode stage is SEEDED by a
        prefill of the same prompt, then steps one token."""
        return self.prefill_trace_inputs()

    def decode_prefill(self, inputs):
        """Seed the resident self-attention KV cache for the whole prompt.

        There is no cross-attention to seed: the reference is decoder-only
        (`is_encoder_decoder=False`), so this is the self-attn cache alone.
        """
        capacity = self.trace_capacity
        input_ids = inputs["input_ids"]
        batch, real_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        self.text.reset_cache()
        ids_tt = self.prepare_prompt(input_ids)
        voice = self.stage_voice(inputs["audio_mask"], inputs["voice_embedding"], input_ids=input_ids)
        # Contiguous 0..S-1 -- the rotary module's float32 default branch; see run_text_to_speech.
        _, last = self.text.prefill_voiced(ids_tt, voice, need_hidden=False)
        self._stage_buffers["decode"] = {
            "llm_hidden": last,
            "position": real_len,
            "capacity": capacity,
            "batch": batch,
            "real_len": real_len,
        }
        return self._stage_buffers["decode"]

    def decode_trace_setup(self, inputs):
        buf = self.decode_prefill(inputs)
        batch = buf["batch"]
        # The step's input is an audio-token EMBEDDING, not a token id, so the persistent input
        # buffer is one frame's worth of codes put through the audio-token table OUTSIDE the
        # trace. Values come from the reference so the traced step matches the golden.
        frame = torch.full((batch, self.n_codebooks, 1), self.n_special, dtype=torch.int64)
        codes_tt = ttnn.from_torch(
            frame.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        buf["embeds"] = self.vocode.audio_token_embedding(codes_tt)
        return buf

    def decode_trace_step(self):
        buf = self._stage_buffers["decode"]
        return self.text.decode_step(buf["embeds"], buf["position"])

    def decode_trace_items(self):
        """B: one token per sample per step."""
        buf = self._stage_buffers.get("decode")
        return int(buf["batch"] if buf else self.batch)

    # -- acoustic ------------------------------------------------------------------------

    def acoustic_trace_inputs(self):
        """ZERO-ARG. The conditioning hidden state: the reference backbone's last hidden state for
        this package's own default prompts, and one draw of the sampler noise."""
        batch = self.batch
        input_ids, _ = self.default_inputs()
        with torch.no_grad():
            llm_hidden = self.reference_model.model(input_ids=input_ids).last_hidden_state[:, -1]
        llm_hidden = llm_hidden.to(torch.float32)
        x0 = self.noise(1, batch=batch)[0]
        return {
            "llm_hidden": llm_hidden,
            "x0": x0,
            "cfg_alpha": torch.full((batch,), DEFAULT_CFG_ALPHA),
        }

    def acoustic_trace_setup(self, inputs):
        batch = int(inputs["llm_hidden"].shape[0])
        self._stage_buffers["acoustic"] = {
            "llm_hidden": ttnn.from_torch(
                inputs["llm_hidden"].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
            "x0": ttnn.from_torch(
                inputs["x0"].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
            "cfg_alpha": ttnn.from_torch(
                inputs["cfg_alpha"].reshape(batch, 1).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            ),
            "batch": batch,
        }
        return self._stage_buffers["acoustic"]

    def acoustic_trace_step(self):
        buf = self._stage_buffers["acoustic"]
        return self.acoustic.decode_frame(buf["llm_hidden"], buf["x0"], buf["cfg_alpha"])

    def acoustic_trace_items(self):
        """`n_decoding_steps * 2 * B * 3`.

        One `_trace_step` is one FULL frame, and the 3 acoustic blocks see 3 token rows per
        CFG-doubled batch per Euler step. Priced at 1 item this stage would be handed a compute
        roof ~1300x too small and then reported as memory-bound when it is compute-bound.
        """
        buf = self._stage_buffers.get("acoustic")
        batch = int(buf["batch"] if buf else self.batch)
        return int(self.acoustic.n_steps) * 2 * batch * 3

    # -- vocode --------------------------------------------------------------------------

    def vocode_trace_inputs(self):
        """ZERO-ARG. A `[B, 37, C]` block of real audio codes at the pinned frame capacity.

        Assembled from the reference: the backbone's hidden state for this package's own prompts,
        through the reference's acoustic sampler, tiled up to the capacity. These are the same
        HF-or-local golden inputs the e2e PCC test and the demo drive the stage with.
        """
        from models.demos.voxtral_4b_tts_2603.reference import golden

        capacity = self.vocode_capacity
        input_ids, _ = self.default_inputs()
        batch = int(input_ids.shape[0])
        x0 = self.noise(1, batch=batch)
        with torch.no_grad():
            llm_hidden = self.reference_model.model(input_ids=input_ids).last_hidden_state[:, -1]
            frame, _ = golden.acoustic_frame(
                self.reference_model, llm_hidden, x0[0], torch.full((batch,), DEFAULT_CFG_ALPHA)
            )
        codes = frame.unsqueeze(-1).repeat(1, 1, capacity)
        return {"codes": codes}

    def vocode_trace_setup(self, inputs):
        codes = inputs["codes"]
        batch, rows, frames = (int(v) for v in codes.shape)
        capacity = self.vocode_capacity
        if frames > capacity:
            codes, frames = codes[:, :, :capacity], capacity
        if frames < capacity:
            # Held at the stage's FIXED trace capacity, last frame repeated into the tail. Built
            # by allocation + fill for the same reason as `default_inputs`; this runs OUTSIDE the
            # capture, where the contract puts the seeding of persistent buffers.
            padded = codes[:, :, -1:].repeat(1, 1, capacity)
            padded[:, :, :frames] = codes
            codes = padded
        self._stage_buffers["vocode"] = {
            "codes": ttnn.from_torch(
                codes.to(torch.int32).contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            ),
            "batch": batch,
            "capacity": capacity,
            "real_len": frames,
        }
        return self._stage_buffers["vocode"]

    def vocode_trace_step(self):
        return self.vocode.decode(self._stage_buffers["vocode"]["codes"])

    def vocode_trace_items(self):
        """`B * C * 8` -- the frame count at the decoder's OUTPUT rate.

        The four transformer groups process C, 2C, 4C and 8C frames (each transposed convolution
        doubles the rate), and 8C is what the last group and the output projection retire.
        Counting the C INPUT frames would undercount the repeated blocks by ~3.75x.
        """
        buf = self._stage_buffers.get("vocode")
        batch = int(buf["batch"] if buf else self.batch)
        capacity = int(buf["capacity"] if buf else self.vocode_capacity)
        return batch * capacity * 8


# ----------------------------------------------------------------------------------------
# The single build surface
# ----------------------------------------------------------------------------------------


def speech_inputs(texts=None, voice=None, batch=None):
    """THE REAL TTS INPUT, encoded exactly as the model's own tokenizer lays a speech request out.

    `[BOS] [BEGIN_AUDIO] [AUDIO]*N [NEXT_AUDIO_TEXT] <text> [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]`,
    with the `[AUDIO]` rows to be replaced by the speaker's voice embedding (the checkpoint repo
    ships 20 presets as `voice_embedding/<id>.pt`). Returns ``(input_ids, audio_mask, voice_embedding,
    texts)``, all host tensors -- this is ENCODING, done before the forward. Module-level so a
    caller can size the KV cache from the prompt BEFORE building the pipeline.
    """
    batch = common.DEFAULT_BATCH if batch is None else int(batch)
    texts = list(common.SPEECH_TEXTS[:batch]) if texts is None else list(texts)
    voice = common.DEFAULT_VOICE if voice is None else voice
    input_ids, audio_mask, emb = common.build_voice_prompt(texts, voice)
    return input_ids, audio_mask, emb, texts


def tts_kv_capacity(prompt_len: int, max_frames: int) -> int:
    """KV slots one speech request needs: the whole prompt, one per frame, and the shared-prefix layout.

    The shared-prefix prefill (`TextStack._stage_prefix`) stores the prefix tile-rounded (up to 31 slots
    of gap, which every decode position is offset by) and the per-row tail tile-rounded, so it needs up to
    2 x 32 slots more than prompt + frames. Without them `_stage_prefix` declines and the request takes the
    whole-prompt prefill, whose fused SwiGLU does not fit L1 at batch 32 for a long prompt.
    """
    return _tile_ceil(int(prompt_len) + int(max_frames) + 2 * ttnn.TILE_SIZE)


# Text tokens a speech request may carry beyond the voice block and its five control tokens; the
# package's 32 prompts use at most ~50.
TTS_TEXT_BUDGET = 96


def default_tts_kv_capacity(hf_model) -> int:
    """The resident KV size a build needs when no caller sized it: the LONGEST preset voice block
    (tekken.json's `voice_num_audio_tokens`), the text budget and the controls, plus the decode
    safety cap -- so the default build can serve any preset voice for the model's full horizon."""
    voices = common.available_voices()
    longest = max((int(v) for v in voices.values()), default=0)
    frames, _ = common.resolve_max_frames(hf_model)
    # No extra shared-prefix slack here: the text budget is far above the package's texts (18-51 tokens)
    # and already leaves it -- every preset fits the split with texts up to 94 tokens (the tightest is
    # casual_female) -- and every slot is attended on every decode step, so 64 more would cost ~1.4 ms a frame.
    # A longer prompt gets a clear error from `_stage_prefix` asking for `tts_kv_capacity(...)`.
    return _tile_ceil(longest + TTS_TEXT_BUDGET + 5 + frames)


def trim_to_end(result) -> list:
    """Each row's waveform cut at its OWN end: the frames BEFORE the one that emitted `end_audio`.

    The decode loop runs until EVERY row has stopped, so the batch is as long as its longest
    sentence; a shorter row's samples past its own stop are what the model produced after it
    should have stopped, and are not part of that row's output.
    """
    per_frame = int(result["waveform"].shape[-1] // max(1, result["frames_decoded"]))
    rows = []
    for i, end in enumerate(result["end_frame"]):
        wav = result["waveform"][i].reshape(-1)
        if end >= 0:
            wav = wav[: int(end) * per_frame]
        rows.append(wav)
    return rows


def _resolve_depth(name, stage_override, default, full, floor, reason):
    """One stack's depth from (its own override, the global `layers`, its full depth).

    `None` means EVERY layer -- never 0, which a builder reads as a zero-layer model. A cap below
    the stack's floor is clamped UP and PRINTED: a capped build must stay a MODEL, not a fragment.
    """
    value = stage_override if stage_override is not None else default
    if value is None:
        return full
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name}_layers={value} is not a depth; None means every layer")
    if value < floor:
        print(f"[voxtral_4b_tts_2603] {name}: clamping layers {value} -> {floor} ({reason})")
        value = floor
    return min(value, full)


def build_pipeline(
    device,
    model=None,
    kv_capacity=None,
    layers=None,
    prefill_layers=None,
    decode_layers=None,
    acoustic_layers=None,
    vocode_layers=None,
    batch=None,
    trace_capacity=None,
    vocode_capacity=None,
):
    """CONSTRUCT AND RETURN the resident pipeline object. Never runs the model.

    This is the SINGLE entry the perf test, `trace_capture_selftest`, the demo and the accuracy
    tests use to obtain the object carrying `PIPELINE_STAGES` and the per-stage trace hooks.

    `kv_capacity` is the resident KV cache length (default: `default_tts_kv_capacity`, enough for
    any preset voice at the full safety cap). `trace_capacity` / `vocode_capacity` override the
    pinned trace capacities (`DEFAULT_TRACE_CAPACITY`, `DEFAULT_VOCODE_CAPACITY`).

    `layers` is the DEFAULT depth for EVERY repeated block. Each stage that OWNS a stack also
    takes its own override, named after the stage:

        prefill_layers / decode_layers   the 26-layer text decoder (they SHARE it -- two phases of
                                         one stack, so setting both to different values is an
                                         error rather than a silent last-wins)
        acoustic_layers                  the 3 acoustic transformer blocks
        vocode_layers                    the codec's blocks-per-group (4 groups x 2)

    A single number cannot describe a three-section model: a profiler that sizes a coverage depth
    PER stack needs somewhere to put each one.
    """
    from models.demos.voxtral_4b_tts_2603.tt import acoustic_stage, text_stack, vocode_stage

    if prefill_layers is not None and decode_layers is not None and prefill_layers != decode_layers:
        raise ValueError(
            f"prefill_layers={prefill_layers} and decode_layers={decode_layers} disagree, but "
            "prefill and decode are two phases of ONE 26-layer stack -- set one of them"
        )

    hf_model = common.load_reference_model() if model is None else model

    text_depth = _resolve_depth(
        "text",
        decode_layers if decode_layers is not None else prefill_layers,
        layers,
        full=len(hf_model.model.layers),
        floor=1,
        reason="a stack needs at least one layer",
    )
    acoustic_depth = _resolve_depth(
        "acoustic",
        acoustic_layers,
        layers,
        full=len(hf_model.acoustic_transformer.layers),
        floor=1,
        reason="a stack needs at least one block",
    )
    vocode_depth = _resolve_depth(
        "vocode",
        vocode_layers,
        layers,
        full=max(len(b.layers) for b in hf_model.audio_tokenizer.decoder_blocks if hasattr(b, "layers")),
        floor=1,
        reason="the cap is blocks-per-group; all four groups must survive because each carries a "
        "different sliding window (2/4/8/16)",
    )

    if kv_capacity is None:
        kv_capacity = default_tts_kv_capacity(hf_model)
    # SIZED FOR THE WORKLOAD, not for a constant: any voice-conditioned prompt is longer than a
    # fixed 64-slot cache (the voice block alone is 67-218 tokens).
    text = text_stack.build_text_stack(device, hf_model, layers=text_depth, kv_capacity=kv_capacity)
    acoustic = acoustic_stage.build_acoustic_stage(device, hf_model, layers=acoustic_depth)
    vocode = vocode_stage.build_vocode_stage(device, hf_model, layers=vocode_depth)

    return VoxtralTTSPipeline(
        device,
        hf_model,
        text=text,
        acoustic=acoustic,
        vocode=vocode,
        batch=batch,
        trace_capacity=trace_capacity,
        vocode_capacity=vocode_capacity,
    )


# ----------------------------------------------------------------------------------------
# Selftest -- module-level, so it can run with no arguments in a fresh process
# ----------------------------------------------------------------------------------------
#
# Nothing under `tt/` opens a device; with no `device` the selftest borrows one from
# `device_session.py` at the package root.


def trace_capture_selftest(device=None, verbose: bool = True, pipe=None) -> bool:
    """Capture, replay and PCC-check ONE step of EACH stage in `PIPELINE_STAGES`.

    Stage traces must NOT co-reside: each is released before the next stage is captured, so the
    trace region only has to hold the LARGEST stage. A capture that fails is reported as a failed
    stage, not retried at a smaller capacity.
    """
    if device is None:
        from models.demos.voxtral_4b_tts_2603 import device_session

        with device_session.selftest_device() as opened:
            return trace_capture_selftest(opened, verbose=verbose, pipe=pipe)

    # REUSE THE CALLER'S PIPELINE WHEN THERE IS ONE: a second copy of the weights beside a caller's
    # live build is DRAM it does not need. Called with no pipeline, it builds its own.
    if pipe is None:
        pipe = build_pipeline(device)
    ok = True
    for stage in pipe.PIPELINE_STAGES:
        setup = getattr(pipe, f"{stage}_trace_setup")
        step = getattr(pipe, f"{stage}_trace_step")
        stage_inputs = getattr(pipe, f"{stage}_trace_inputs")
        items = getattr(pipe, f"{stage}_trace_items")

        setup(stage_inputs())
        reference = step()  # eager, outside the trace, as the comparison
        reference_host = ttnn.to_torch(reference).to(torch.float32).clone()

        trace_id = None
        capturing = False
        try:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            capturing = True
            captured = step()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            capturing = False
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            replay = ttnn.to_torch(captured).to(torch.float32)
            score = common.pcc(replay, reference_host)
            stage_ok = score >= 0.99
            if verbose:
                print(f"[trace] {stage:9s} items={items():<8d} replay PCC={score:.6f} {'ok' if stage_ok else 'FAIL'}")
            ok = ok and stage_ok
        except Exception as exc:  # noqa: BLE001 - a capture failure is a reportable result
            print(f"[trace] {stage:9s} CAPTURE FAILED: {type(exc).__name__}: {exc}")
            ok = False
        finally:
            if trace_id is not None:
                if capturing:  # a step that raised mid-capture left the capture open
                    ttnn.end_trace_capture(device, trace_id, cq_id=0)
                ttnn.release_trace(device, trace_id)
    return ok


if __name__ == "__main__":
    print("trace_capture_selftest ->", trace_capture_selftest())
