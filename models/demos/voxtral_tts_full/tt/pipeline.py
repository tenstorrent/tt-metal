# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN forward pass for Voxtral-TTS (text -> 24 kHz speech).

Both `demo/demo_tts.py` and `tests/e2e/test_e2e_tts.py` import `build_pipeline` and
`VoxtralTtsPipeline.run_tts` from here. There is no second copy of the wiring, so a green test
is a statement about the demo's code path.

THE CHAIN, as `VoxtralTtsForConditionalGeneration.forward` composes it:

    ids + voice preset ──[embed_prompt]──► inputs_embeds [1, P, 3072]
                                             │
      ┌──────────────────────────────────────┘
      │  [prefill]  tts_backbone ──► hidden [1, P, 3072];  h = hidden[:, -1]
      │                                │
      │  [decode]   flow_matching(h) ──┴──► codes [1, 37]      stop if codes[0,0] == end_audio_id
      │                                │
      │             embed_frame(codes) ──► emb [1, 1, 3072]
      │             inputs_embeds = concat(inputs_embeds, emb)
      └─────────────  decode stack ────► h = hidden[:, -1]      (loops back)
                                         │
         [vocode]  codec_decoder(frames [T, 37]) ──► waveform [1, 1, T*1920]

WHY THERE ARE TWO BACKBONE PATHS. The reference has two, and they are different methods:
`VoxtralTtsBackbone.forward` (the prefill) and `VoxtralTtsBackbone.prefill_then_step` (the
per-frame step). The graduated `_stubs/tts_backbone.py` ports the first; its own docstring and
`_stubs/decoder_layer.py`'s say the cache path is deliberately out of scope. So the prefill
stage runs the graduated `tts_backbone` (which imports `decoder_layer`, which imports
`attention`), and the decode stage runs `TtBackboneLeafLayer` -- the SAME `VoxtralDecoderLayer`
composition, built from the graduated LEAF ports `r_m_s_norm`, `attention` and `m_l_p`, over the
SAME staged weights (no second copy on device). Both paths are real, both feed the waveform.

Since the graduated `attention` port stages its RoPE table from position 0 and carries no KV
cache, the decode step re-runs the grown prefix causally and reads the last row. That is exactly
what the reference's own `forward` does -- it calls `prefill_then_step(inputs_embeds, emb)` with
a prefix that grows every frame -- and a causal prefill of `[prefix, emb]` at position P is
identical arithmetic to a cached step at position P.

HOST WORK. `encode_inputs` (tokenize, load the voice preset, build the index/mask tensors, draw
the ODE noise) runs on the host and returns device tensors; it is INPUT ENCODING and is excluded
from `host_op_selftest`'s observed region. `run_tts` from that point on is ttnn only. The single
exception is the scalar read of the semantic code that decides whether to stop -- host-side loop
control, which the reference does too and which `_stubs/codec_decoder.py` explicitly delegates to
the generation loop ("data-dependent control flow ... belongs to the host-side generation loop").
`run_tts(check_stop=False)` removes even that, and is what the on-device selftest observes.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import torch
import ttnn

from models.demos.voxtral_tts_full._stubs.attention import _mean_sum

_DEMO_ROOT = Path(__file__).resolve().parents[1]
_CAPTURED = _DEMO_ROOT / "_captured"
HF_MODEL_ID = "/localdev/lserbedzija/hf_models/voxtral-tts-full"

# --------------------------------------------------------------------------------------------
# Stage / stack declarations
# --------------------------------------------------------------------------------------------
# config.architectures = [VoxtralTtsForConditionalGeneration] -> a decoder-only AR model, so
# [prefill, decode]; config.modality_out = "audio" (task "text-to-speech") -> append [vocode].
# There is no is_encoder_decoder and no encoder sub-config, so there is no "encode" stage.
PIPELINE_STAGES = ["prefill", "decode", "vocode"]

# The seven graduated ports under `_stubs/`. All seven are routed into the chain above; see
# `e2e_plan.json::graduated_module_verification` for why the three the bring-up status file
# labels REUSE are graduated work products too.
GRADUATED_COMPONENTS = (
    "tts_backbone",
    "decoder_layer",
    "attention",
    "m_l_p",
    "r_m_s_norm",
    "flow_matching",
    "codec_decoder",
)

# Trace capacities. The variable dim of prefill/decode is the sequence axis, bounded by
# config.max_position_embeddings (2048) -- which is also exactly how many RoPE/mask positions
# the graduated attention port stages. 256 covers the 200-id default prompt plus its frames.
TRACE_SEQ_CAPACITY = 256
TRACE_FRAME_CAPACITY = 8
# A TRACE RECORDS THE COMMAND STREAM, NOT THE DATA, so this is sized by the pipeline's OP COUNT
# and not by the capacities above. Measured: the prefill stage commits 43 MB and the decode
# stage 44 MB, flat as C is halved from 256 to 32 -- because the same ttnn calls are issued
# either way, only over smaller tensors. The stages are captured and released one at a time, so
# the region only has to hold the largest of them; 256 MB is ~6x that, and it is 0.8% of this
# 32 GB part against a ~15 GB resident model.
TRACE_REGION_SIZE = 256 * 1024 * 1024

_AUDIO_TOKEN_ID = 24  # voxtral_pipeline_ref.AUDIO_TOKEN_ID
_DIM = 3072
_NUM_CODEBOOKS = 37
_N_ACOUSTIC = 36


# --------------------------------------------------------------------------------------------
# Reference-module access (SETUP ONLY -- never on the compute path)
# --------------------------------------------------------------------------------------------
_REF_PKG = "_voxtral_ref_pkg"


def _ref_module(name: str, model_id: str = HF_MODEL_ID):
    """Load one vendored reference module. They import only torch and stdlib.

    The modules import each other RELATIVELY (`from . import voxtral_backbone_ref`), so loading
    one by file path fails with "attempted relative import with no known parent package". They
    are therefore mounted under a synthetic package whose `__path__` is the model directory,
    which makes the relative imports resolve exactly as they do inside the model.
    """
    if _REF_PKG not in sys.modules:
        # The reference resolves its voices / params relative to this, and under
        # trust_remote_code `__file__` points at the module cache rather than the model.
        os.environ.setdefault("VOXTRAL_ASSETS", str(Path(model_id) / "assets"))
        pkg = types.ModuleType(_REF_PKG)
        pkg.__path__ = [str(model_id)]
        pkg.__package__ = _REF_PKG
        sys.modules[_REF_PKG] = pkg
    return importlib.import_module(f"{_REF_PKG}.{name}")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """`voxtral_common_ref.pcc` -- the reference's own definition of the accuracy metric."""
    a, b = a.detach().flatten().float(), b.detach().flatten().float()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def load_hf_model(model_id: str = HF_MODEL_ID, dtype=torch.float32):
    """The HF reference, in fp32 -- the reference's arithmetic is fp32 by construction."""
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
    )


# --------------------------------------------------------------------------------------------
# Invocation counting (Gate 2)
# --------------------------------------------------------------------------------------------
class _Counted:
    """Counts real invocations of a graduated stub without changing what it computes.

    Gate 2 asks whether each graduated module actually ran inside the forward, so the count is
    OBSERVED at the call rather than inferred from the stack depth. Attribute access falls
    through to the wrapped port, and every element of a stack is wrapped, so a structural walk
    still sees a plain list of same-typed objects with a `__dict__`.
    """

    def __init__(self, inner, component, counter):
        self.inner = inner
        self.component = component
        self.counter = counter

    def __call__(self, *args, **kwargs):
        self.counter[self.component] = self.counter.get(self.component, 0) + 1
        return self.inner(*args, **kwargs)

    def __getattr__(self, item):
        # Only reached for attributes this proxy does not own.
        return getattr(object.__getattribute__(self, "inner"), item)


# --------------------------------------------------------------------------------------------
# The decode-path layer: VoxtralDecoderLayer composed from the graduated LEAF ports
# --------------------------------------------------------------------------------------------
class TtBackboneLeafLayer:
    """`modeling_layers.VoxtralDecoderLayer.forward`, built from the graduated leaf ports.

        h = input_layernorm(x);  x = x + self_attn(h)
        h = post_attention_layernorm(x);  x = x + mlp(h)

    `attn` is the same `_stubs/attention.py` object the prefill path uses -- the weights are
    staged once and shared -- so this path costs only the two norm ports and the MLP port.
    """

    def __init__(self, attn_norm, attn, ffn_norm, mlp):
        self.attn_norm = attn_norm
        self.attn = attn
        self.ffn_norm = ffn_norm
        self.mlp = mlp

    def __call__(self, x):
        h = self.attn_norm(x)
        x = ttnn.add(x, self.attn(h, rope=True, causal=True))
        h = self.ffn_norm(x)
        return ttnn.add(x, self.mlp(h))


# --------------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------------
class VoxtralTtsPipeline:
    """Resident TTNN pipeline: build once, then `encode_inputs` -> `run_tts` per prompt."""

    def __init__(self, device, hf, stubs, tables, counter, depths, dtype=ttnn.bfloat16):
        self.device = device
        self.dtype = dtype
        # The HF reference stays reachable: it is ground truth for section structure (a walk
        # finds hf.backbone.layers = 26) and the golden helper needs it. It is NEVER called
        # from the forward path.
        self.hf = hf
        self.config = hf.config
        self.stages = list(PIPELINE_STAGES)
        self.invocations = counter
        self.depths = depths
        self.__dict__.update(stubs)   # backbone, decode_layers, flow, codec
        self.__dict__.update(tables)  # tok_table, audio_table, codebook_offsets, ...
        self._trace = {}

    # -- structure ---------------------------------------------------------------------
    @property
    def stacks(self):
        """Every repeated block, as plain same-typed lists -- what a structural walk sizes."""
        return {
            "backbone": self.backbone.layers,
            # Sliced to the depth the decode stage actually RUNS, not the depth built: the two
            # backbone paths share one staged stack, so the build is max(prefill, decode) deep
            # and reporting that would over-report this stack to anything sizing it.
            "backbone_decode": self.decode_layers[: self.depths["decode"]],
            "flow": self.flow.layers,
            "codec": [layer for stage in self.codec.stages for layer in stage],
        }

    def reset_counts(self):
        for name in GRADUATED_COMPONENTS:
            self.invocations[name] = 0

    # ============================================================================
    # INPUT ENCODING -- host side, deliberately OUTSIDE the observed forward
    # ============================================================================
    def encode_inputs(self, ids=None, text=None, voice=None, max_frames=8, seed=0):
        """Prompt ids + voice preset -> the device tensors `run_tts` consumes.

        This is the tokenizer / preset-loading half of the model, the analogue of feature
        extraction for an audio model. It fires host ops by design and is excluded from the
        on-device verdict; everything it returns already lives on the device.
        """
        pref = _ref_module("voxtral_pipeline_ref")
        cfg = self.config
        voice = voice or getattr(cfg, "default_voice", "neutral_male")

        if ids is None:
            if text is not None:
                tok = _ref_module("voxtral_tokenizer_ref").TekkenTokenizer()
                ids = torch.tensor(tok.build_prompt(text, voice), dtype=torch.long)
            else:
                ids = torch.tensor(list(cfg.default_prompt_ids), dtype=torch.long)
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)

        preset = pref.load_voice(voice, voice_dir=str(Path(HF_MODEL_ID) / "assets" / "voice_embedding"))
        mask = ids == _AUDIO_TOKEN_ID
        n_ph = int(mask.sum())
        if n_ph != preset.shape[0]:
            raise ValueError(
                f"prompt has {n_ph} audio placeholders but voice {voice!r} has {preset.shape[0]} rows. "
                f"The count is voice-specific -- re-tokenize the text for THIS voice."
            )

        P = int(ids.numel())
        # Row 0 of the staged voice table is dead, so a non-placeholder position reads zeros and
        # the two lookups can simply be added. `keep` zeroes the text lookup at placeholders.
        vidx = torch.zeros(P, dtype=torch.int32)
        vidx[mask] = torch.arange(1, n_ph + 1, dtype=torch.int32)
        keep = (~mask).float().reshape(1, P, 1).expand(1, P, _DIM).contiguous()

        # THE ODE'S INITIAL CONDITION, pinned so both sides integrate the same trajectory.
        #
        # A SEEDED GAUSSIAN IS THE DEFAULT BECAUSE ZERO IS DEGENERATE. Real inference draws
        # x_0 ~ N(0, 1) per frame (`voxtral_flow_ref.decode_frame`), so a seeded Gaussian is
        # both faithful and reproducible. Starting from zero -- the initial condition the
        # per-component PCC harness uses, and this block's staged default -- integrates from the
        # distribution's mean, and the trajectory it produces parks far more of the 36 acoustic
        # dimensions within arithmetic noise of an FSQ rounding boundary. Measured over 8
        # frames: zero start -> 111 flipped codes and waveform PCC 0.04; N(0,1) start -> 36 and
        # 0.98. `seed=None` selects the zero start for parity with the per-component harness.
        n_bank = int(max_frames) + 1
        if seed is None:
            bank = torch.zeros(n_bank, _N_ACOUSTIC)
        else:
            g = torch.Generator().manual_seed(int(seed))
            bank = torch.randn(n_bank, _N_ACOUSTIC, generator=g)

        dev = self.device

        def index_row(t):
            """A [1, P] index vector on device, in the layout `ttnn.embedding` wants."""
            return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

        def tiles(t, dtype=ttnn.bfloat16):
            torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
            return ttnn.from_torch(t.to(torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

        # The dead row is prepended ON DEVICE. `ttnn.embedding` needs row 0 to exist so a
        # non-placeholder position can read zeros, but building the padded table on the host
        # would allocate and copy a second full preset for the sake of one row of zeros.
        voice_table = ttnn.concat(
            [ttnn.zeros((1, _DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), tiles(preset.float())],
            dim=0,
        )
        return {
            "ids": ids,
            "voice": voice,
            "prompt_len": P,
            "max_frames": int(max_frames),
            "x0_bank_host": bank,
            "ids_dev": index_row(ids.to(torch.int32).reshape(1, P)),
            "vidx_dev": index_row(vidx.reshape(1, P)),
            "keep_dev": tiles(keep),
            "voice_dev": voice_table,
            # fp32: the ODE's initial condition enters the same rounding-boundary arithmetic the
            # flow block runs in (see its build dtype), so it is staged at that precision.
            "x0_dev": [tiles(bank[i : i + 1].float(), ttnn.float32) for i in range(n_bank)],
        }

    # ============================================================================
    # THE FORWARD PATH -- ttnn only
    # ============================================================================
    def embed_prompt(self, enc):
        """`voxtral_pipeline_ref.build_inputs_embeds`, on device.

        embeds = tok_embeddings[ids] * keep + voice_table[vidx]; the preset row substitution is
        a second lookup rather than a scatter, and `keep` zeroes the text lookup where a preset
        row takes over.
        """
        text = ttnn.embedding(enc["ids_dev"], self.tok_table, layout=ttnn.TILE_LAYOUT)
        voice = ttnn.embedding(enc["vidx_dev"], enc["voice_dev"], layout=ttnn.TILE_LAYOUT)
        # `ttnn.embedding` requires a bfloat16 table, so the two lookups happen at that
        # precision and the residual stream is lifted to its own afterwards.
        return self._cast(ttnn.add(ttnn.mul(text, enc["keep_dev"]), voice))

    def embed_frame(self, codes):
        """`voxtral_backbone_ref.embed_frame`, on device: sum of the 37 codebook lookups.

        Each codebook owns a slice of one flat table, so the code is shifted by that codebook's
        offset before the lookup. The shift is done in fp32 (exact to 2^24; semantic codes reach
        8193, which bf16 could not carry) and cast back to an index.
        """
        idx = ttnn.typecast(
            ttnn.add(ttnn.typecast(codes, ttnn.float32), self.codebook_offsets), ttnn.uint32
        )
        rows = ttnn.embedding(idx, self.audio_table, layout=ttnn.TILE_LAYOUT)  # [1, 37, 3072]
        # THE SUM IS TAKEN IN fp32, AND AS `mean * 37`. `ttnn.embedding` only accepts a bfloat16
        # table, but the checkpoint's embedding rows are themselves bfloat16-exact (the prompt
        # embedding reproduces the reference bit for bit), so the lookups lose nothing -- the
        # ACCUMULATION of 37 of them does, twice over. In the table's own precision it measured
        # 1.6e-3 relative against the reference; in fp32 through `ttnn.sum`, 4.2e-5; through the
        # model's `_mean_sum`, exact. This lands on the residual stream at exactly the position
        # Block 2 reads, so it is the least forgiving place in the pipeline to lose a digit.
        total = _mean_sum(ttnn.typecast(rows, ttnn.float32), dim=1)
        return self._cast(ttnn.reshape(total, (1, 1, _DIM)))

    def prefill(self, inputs_embeds):
        """[prefill] The graduated `tts_backbone` port (-> decoder_layer -> attention)."""
        return self.backbone(inputs_embeds)

    def decode_stack(self, inputs_embeds):
        """[decode] `VoxtralTtsBackbone.prefill_then_step`, from the graduated leaf ports."""
        x = inputs_embeds
        for layer in self.decode_layers[: self.depths["decode"]]:
            x = layer(x)
        return self.decode_final_norm(x)

    def _cast(self, x):
        """Lift a bfloat16 intermediate to the residual stream's precision (a no-op in bf16)."""
        return x if x.dtype == self.dtype else ttnn.typecast(x, self.dtype)

    @staticmethod
    def _last_row(hidden, seq_len):
        """hidden [1, S, 3072] -> h [1, 3072]: the only position Block 2 ever sees."""
        return ttnn.reshape(ttnn.slice(hidden, [0, seq_len - 1, 0], [1, seq_len, _DIM]), (1, _DIM))

    def run_tts(self, enc, max_frames=None, check_stop=True):
        """The chained forward pass. Real prompt in, real 24 kHz waveform out.

        Every stage is fed the previous TT stage's real device output; no reference tensor is
        injected at any joint.
        """
        max_frames = int(enc["max_frames"] if max_frames is None else max_frames)
        stop_id = int(getattr(self.config, "end_audio_id"))

        # ---- prefix embedding + [prefill] ------------------------------------------------
        embeds = self.embed_prompt(enc)
        seq_len = enc["prompt_len"]
        hidden = self.prefill(embeds)
        h = self._last_row(hidden, seq_len)

        prefill_hidden = hidden
        frames, step_hidden = [], []

        # ---- [decode] the autoregressive frame loop --------------------------------------
        for t in range(max_frames):
            codes = self.flow(h, x_0=enc["x0_dev"][t])  # [1, 37]

            if check_stop:
                # Host-side loop control, exactly as the reference's forward does it:
                #   if int(codes[0, 0]) == self.config.end_audio_id: break
                if int(ttnn.to_torch(codes).reshape(-1)[0]) == stop_id:
                    break
            frames.append(codes)

            emb = self.embed_frame(codes)
            embeds = ttnn.concat([embeds, emb], dim=1)
            seq_len += 1
            hidden = self.decode_stack(embeds)
            h = self._last_row(hidden, seq_len)
            step_hidden.append(h)

        if not frames:
            return {"waveform": None, "frames": None, "prefill_hidden": prefill_hidden, "step_hidden": []}

        # ---- [vocode] ---------------------------------------------------------------------
        all_codes = ttnn.concat(frames, dim=0)  # [T, 37]
        waveform = self.codec(all_codes)

        return {
            "waveform": waveform,
            "frames": all_codes,
            "prefill_hidden": prefill_hidden,
            "step_hidden": step_hidden,
            "n_frames": len(frames),
        }

    # ============================================================================
    # TRACE CONTRACT
    # ============================================================================
    def _pad_embeds(self, embeds, real_len, capacity):
        """Pin the sequence axis to `capacity`. Attention is causal, so rows [0:real_len] are
        unchanged by whatever sits after them -- the pad needs no separate mask."""
        if real_len >= capacity:
            return embeds
        pad = ttnn.zeros(
            (1, capacity - real_len, _DIM), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        return ttnn.concat([embeds, pad], dim=1)

    # ---- prefill ----------------------------------------------------------------------
    def prefill_trace_inputs(self):
        """ZERO-ARG. The real captured prompt embeds -- the same golden input the e2e test uses."""
        args = torch.load(_CAPTURED / "tts_backbone" / "args.pt", map_location="cpu", weights_only=False)
        return {"inputs_embeds": args[0].float()}

    def prefill_trace_setup(self, inputs):
        capacity = int(inputs.get("capacity", TRACE_SEQ_CAPACITY))
        embeds = inputs["inputs_embeds"]
        real_len = int(embeds.shape[1])
        dev = ttnn.from_torch(
            embeds.to(torch.float32 if self.dtype == ttnn.float32 else torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self._trace["prefill"] = {
            "embeds": self._pad_embeds(dev, real_len, capacity),
            "real_len": real_len,
            "capacity": capacity,
        }
        return self._trace["prefill"]

    def prefill_trace_step(self):
        """ONE host-op-free prefill at the pinned shape, reading only persistent buffers."""
        st = self._trace["prefill"]
        return self._last_row(self.prefill(st["embeds"]), st["real_len"])

    # ---- decode -----------------------------------------------------------------------
    def decode_trace_inputs(self):
        """ZERO-ARG. The captured prompt embeds, plus one real frame's CODES.

        The codes are handed over rather than an embedding of them: `decode_trace_setup` runs
        them through this pipeline's own on-device `embed_frame`, so what the traced step sees
        at the frame position is the tensor the real decode loop would put there -- not a host
        reimplementation of it via the reference's `embed_frame`.
        """
        args = torch.load(_CAPTURED / "tts_backbone" / "args.pt", map_location="cpu", weights_only=False)
        codes = torch.load(_CAPTURED / "flow_matching" / "output.pt", map_location="cpu", weights_only=False)
        return {
            "inputs_embeds": args[0].float(),
            "codes": torch.as_tensor(codes).reshape(-1)[:_NUM_CODEBOOKS].reshape(1, _NUM_CODEBOOKS),
        }

    def decode_trace_setup(self, inputs):
        capacity = int(inputs.get("capacity", TRACE_SEQ_CAPACITY))
        embeds = inputs["inputs_embeds"]
        prompt_len = int(embeds.shape[1])
        prompt = ttnn.from_torch(
            embeds.to(torch.float32 if self.dtype == ttnn.float32 else torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        # The frame is embedded and appended ON DEVICE, exactly as `run_tts` does it.
        codes = ttnn.from_torch(
            inputs["codes"].to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        grown = ttnn.concat([prompt, self.embed_frame(codes)], dim=1)
        real_len = prompt_len + 1
        self._trace["decode"] = {
            "embeds": self._pad_embeds(grown, real_len, capacity),
            "real_len": real_len,
            "capacity": capacity,
        }
        return self._trace["decode"]

    def decode_trace_step(self):
        st = self._trace["decode"]
        return self._last_row(self.decode_stack(st["embeds"]), st["real_len"])

    # -- the AR decode contract (resident KV seeding). This backbone port carries no KV cache
    # (see the module docstring), so the resident state IS the grown prefix: decode_prefill
    # pins it, decode_step reads it and never recomputes the prompt embedding.
    def decode_prefill(self, inputs):
        return self.decode_trace_setup(inputs)

    def decode_step(self):
        return self.decode_trace_step()

    # ---- vocode -----------------------------------------------------------------------
    def vocode_trace_inputs(self):
        """ZERO-ARG. The real code frames the capture recorded for the codec block."""
        args = torch.load(_CAPTURED / "codec_decoder" / "args.pt", map_location="cpu", weights_only=False)
        return {"codes": torch.as_tensor(args[0])}

    def vocode_trace_setup(self, inputs):
        capacity = int(inputs.get("capacity", TRACE_FRAME_CAPACITY))
        codes = inputs["codes"]
        if codes.dim() == 3:  # [1, 37, T] -> [T, 37]
            codes = codes[0].t()
        real_len = int(codes.shape[0])
        capacity = max(capacity, real_len)
        staged = ttnn.from_torch(
            codes.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        if real_len < capacity:
            # Pin the frame axis to C ON DEVICE by repeating the last frame. The codec is causal,
            # so the first `real_len` frames of the output are unchanged by what follows them.
            last = ttnn.slice(staged, [real_len - 1, 0], [real_len, _NUM_CODEBOOKS])
            staged = ttnn.concat([staged] + [last] * (capacity - real_len), dim=0)
        self._trace["vocode"] = {
            "codes": staged,
            "real_len": real_len,
            "capacity": capacity,
        }
        return self._trace["vocode"]

    def vocode_trace_step(self):
        return self.codec(self._trace["vocode"]["codes"])

    # ============================================================================
    # SELF-TESTS
    # ============================================================================
    def host_op_selftest(self, max_frames=2):
        """AUTHORITATIVE fully-on-device check.

        Input encoding and the one-time weight build happen OUTSIDE the observed region; the
        model math -- prefix embedding, prefill, every decode step, the frame embedding and the
        vocoder -- happens INSIDE it. ttnn ops do not dispatch through torch, so a truly
        on-device forward fires zero host aten ops.

        `check_stop=False`: the stop test reads one scalar back to decide whether to keep
        looping. That is host-side control flow, not model math (the reference makes the same
        decision on the host, and the codec port explicitly leaves it to the generation loop),
        and reading it would put a tensor copy inside the observed region.
        """
        from scripts.tt_hw_planner import host_op_observer

        enc = self.encode_inputs(max_frames=max_frames)
        with host_op_observer.observe_host_ops() as ops:
            self.run_tts(enc, max_frames=max_frames, check_stop=False)
        return host_op_observer.verdict(ops)

    def trace_capture_selftest(self, device=None):
        """Capture ONE step per stage, execute it, check it against the eager output, release.

        Stage traces must not co-reside, so each is released before the next stage is captured.
        """
        device = device or self.device
        ok = True
        for stage in PIPELINE_STAGES:
            capacity = TRACE_FRAME_CAPACITY if stage == "vocode" else TRACE_SEQ_CAPACITY
            inputs = getattr(self, f"{stage}_trace_inputs")()
            inputs["capacity"] = capacity
            getattr(self, f"{stage}_trace_setup")(inputs)
            step = getattr(self, f"{stage}_trace_step")

            reference = ttnn.to_torch(step()).float()  # eager, outside the trace
            tid = None
            try:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = step()
                ttnn.end_trace_capture(device, tid, cq_id=0)
            except RuntimeError as exc:
                # THE FAILED CAPTURE MUST BE RELEASED. Its buffers are already charged against
                # the trace region, so leaving it makes the NEXT stage overflow too and report
                # a larger, meaningless shortfall -- the failure then looks like it belongs to
                # a stage that would have captured fine on its own.
                #
                # There is deliberately no shrink-and-retry here. Retrying at a smaller C is
                # the obvious remedy and it does not work: a trace records the command stream,
                # so its size follows the OP COUNT, which does not change with the sequence
                # capacity. Measured, the commit stayed at ~44 MB from C=256 down to C=32. If
                # this fires, the trace region is too small -- raise TRACE_REGION_SIZE.
                if tid is not None:
                    try:
                        ttnn.release_trace(device, tid)
                    except RuntimeError:
                        pass
                print(f"[trace] {stage}: capture FAILED at C={capacity}: {exc}")
                ok = False
                continue

            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = ttnn.to_torch(out).float()
            p = pcc(got, reference)
            ttnn.release_trace(device, tid)
            print(f"[trace] {stage}: captured at C={capacity}, execute_trace PCC={p:.6f}")
            ok = ok and p >= 0.99
        return ok


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------
def _stub(name):
    import importlib

    return importlib.import_module(f"models.demos.voxtral_tts_full._stubs.{name}")


def _stage_vector(device, t, dtype=ttnn.bfloat16):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    return ttnn.from_torch(
        t.detach().float().contiguous().to(torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def build_pipeline(device, model=None, layers=None, **kwargs):
    """CONSTRUCT AND RETURN the resident pipeline object. Does not run it.

    This is the single build surface: the demo, the e2e test and the perf harness all obtain
    the measurement object from here.

    `layers` is the DEFAULT depth for every repeated stack; None means every layer (never 0).
    Each stage that owns a stack takes its own override so a multi-section model is not forced
    to one number:

        prefill_layers  -> the backbone depth the prefill stage runs
        decode_layers   -> the backbone depth the decode stage runs
        vocode_layers   -> the per-stage transformer depth of the codec stack
        flow_layers     -> the flow stack (the decode stage owns TWO stacks -- backbone and
                           flow -- and one number cannot describe both)

    Embeddings, norms and heads are always built in full, so a capped build still exercises
    every distinct op the full model runs, just fewer times.

    Demo kwargs (text, prompt, voice, language, ...) are accepted and ignored: the resident
    build derives its shapes from the config, not from a prompt.
    """
    hf = model if model is not None else load_hf_model()

    # BACKBONE PRECISION, and why fp32 is the DEFAULT here rather than an option of last resort.
    # The AR loop is a feedback system whose output is QUANTISED onto 21 FSQ levels, so an error
    # in the hidden state does not stay small: it flips a code, which swaps a whole learned row of
    # the audio embedding table (|d|/|r| = 0.335 between adjacent codes), which moves the frame
    # embedding fed back in ~5%, which moves the next hidden state further. bfloat16 lands the
    # 26-layer stack at PCC 0.998 per decode step -- enough for the semantic argmax (exact on
    # every frame either way) but not for the acoustic rounding. fp32 lands it at 0.99999, which
    # is the difference between the first frames' acoustic codes being right and being wrong.
    # 26 layers in fp32 is ~12 GB, which this 32 GB Blackhole carries comfortably.
    backbone_dtype = kwargs.get("backbone_dtype", None) or ttnn.float32

    n_backbone = len(hf.backbone.layers)

    def depth(override, total):
        value = kwargs.get(override, None)
        if value is None:
            value = layers
        return total if value is None else max(1, min(int(value), total))

    d_prefill = depth("prefill_layers", n_backbone)
    d_decode = depth("decode_layers", n_backbone)
    d_flow = depth("flow_layers", 3)
    d_vocode = depth("vocode_layers", 2)
    n_built = max(d_prefill, d_decode)

    counter = {name: 0 for name in GRADUATED_COMPONENTS}
    attn_mod, mlp_mod, norm_mod = _stub("attention"), _stub("m_l_p"), _stub("r_m_s_norm")
    layer_mod, backbone_mod = _stub("decoder_layer"), _stub("tts_backbone")
    flow_mod, codec_mod = _stub("flow_matching"), _stub("codec_decoder")

    # ---- one staging of the backbone weights, shared by BOTH backbone paths ---------------
    attentions, decode_layers, prefill_layers = [], [], []
    for i in range(n_built):
        src = hf.backbone.layers[i]
        attn = _Counted(
            attn_mod.TtVoxtralAttention.build(device, src.self_attn, dtype=backbone_dtype), "attention", counter
        )
        mlp = _Counted(mlp_mod.build(device, src.mlp, dtype=backbone_dtype), "m_l_p", counter)
        n1 = _Counted(norm_mod.build(device, src.input_layernorm, dtype=backbone_dtype), "r_m_s_norm", counter)
        n2 = _Counted(
            norm_mod.build(device, src.post_attention_layernorm, dtype=backbone_dtype), "r_m_s_norm", counter
        )
        attentions.append(attn)

        # decode path: the leaf ports, in VoxtralDecoderLayer's order
        decode_layers.append(TtBackboneLeafLayer(n1, attn, n2, mlp))

        # prefill path: the graduated composite port, constructed over the SAME staged tensors
        # (the norm vectors are 3072 floats each -- restaging them costs 12 KB per layer).
        weights = {
            "attn_norm": _stage_vector(device, src.input_layernorm.weight, backbone_dtype),
            "ffn_norm": _stage_vector(device, src.post_attention_layernorm.weight, backbone_dtype),
            "gate": mlp.inner.gate,
            "down": mlp.inner.down,
            "up": mlp.inner.up,
        }
        prefill_layers.append(
            _Counted(
                layer_mod.TtVoxtralDecoderLayer(attn, weights, float(getattr(src.input_layernorm, "eps", 1e-5))),
                "decoder_layer",
                counter,
            )
        )

    backbone = _Counted(
        backbone_mod.TtVoxtralTtsBackbone(
            prefill_layers[:d_prefill],
            _stage_vector(device, hf.backbone.norm.weight, backbone_dtype),
            float(getattr(hf.backbone.norm, "eps", 1e-5)),
        ),
        "tts_backbone",
        counter,
    )
    decode_final_norm = _Counted(
        norm_mod.build(device, hf.backbone.norm, dtype=backbone_dtype), "r_m_s_norm", counter
    )

    # ---- blocks 2 and 3 --------------------------------------------------------------------
    # Block 2 in fp32: its 36 acoustic floats are ROUNDED onto 21 FSQ levels, and a few
    # dimensions per frame sit within 5e-4 of a rounding boundary -- inside bfloat16's own error,
    # so those codes flip on arithmetic noise. A flipped code shifts a latent by a full 1/20th of
    # its range and the waveform PCC collapses rather than degrades. 390M parameters, so the
    # extra ~0.8 GB is affordable here where it would not be for the 3.4B backbone.
    flow = _Counted(
        flow_mod.TtVoxtralFlowMatching.build(device, hf.flow, n_steps=7, dtype=ttnn.float32),
        "flow_matching",
        counter,
    )
    if d_flow < 3:
        flow.inner.layers = flow.inner.layers[:d_flow]
    codec = _Counted(codec_mod.TtVoxtralCodecDecoder.build(device, hf.codec), "codec_decoder", counter)
    if d_vocode < 2:
        codec.inner.stages = [stage[:d_vocode] for stage in codec.inner.stages]

    # ---- embedding tables (the prefix embedding runs on device) ----------------------------
    tables = {
        # ttnn.embedding takes a bfloat16 table only; the lookups are lifted after the fact.
        "tok_table": _stage_vector(device, hf.backbone.tok_embeddings),
        "audio_table": _stage_vector(device, hf.backbone.audio_embeddings),
        "codebook_offsets": ttnn.from_torch(
            _ref_module("voxtral_common_ref").codebook_offsets().reshape(1, _NUM_CODEBOOKS).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        "decode_final_norm": decode_final_norm,
    }

    stubs = {"backbone": backbone, "decode_layers": decode_layers, "flow": flow, "codec": codec}
    depths = {"prefill": d_prefill, "decode": d_decode, "flow": d_flow, "vocode": d_vocode}
    pipe = VoxtralTtsPipeline(device, hf, stubs, tables, counter, depths, dtype=backbone_dtype)
    pipe.reset_counts()
    return pipe


# --------------------------------------------------------------------------------------------
# Standalone probe entry points (no pytest fixture in that process -- see selftest_runtime.py)
# --------------------------------------------------------------------------------------------
def host_op_selftest():
    from models.demos.voxtral_tts_full.selftest_runtime import standalone_device

    with standalone_device(TRACE_REGION_SIZE) as device:
        return build_pipeline(device).host_op_selftest()


def trace_capture_selftest(device=None):
    if device is not None:
        return build_pipeline(device).trace_capture_selftest(device)
    from models.demos.voxtral_tts_full.selftest_runtime import standalone_device

    with standalone_device(TRACE_REGION_SIZE) as dev:
        return build_pipeline(dev).trace_capture_selftest(dev)
