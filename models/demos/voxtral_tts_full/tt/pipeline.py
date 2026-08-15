# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ONE chained TTNN forward pass for `voxtral-tts-full` -- text in, 24 kHz speech out.

Both `demo/demo_tts.py` and `tests/e2e/test_tts_e2e.py` import `build_pipeline` and call
`run_tts` from HERE.  There is deliberately no second copy of the wiring: a demo with its own
chain drifts from the test and ships broken while the test stays green.

THE CHAIN (`run_tts`), which is `VoxtralTtsForConditionalGeneration.forward`'s own composition:

    ids + voice --[device embedding]--> inputs_embeds [1, P, 3072]
      -> tts_backbone                 -> hidden, last row h [1, 1, 3072]
      -> loop:  flow_matching(h)      -> 37 audio codes
                stop if the semantic code is an [END_AUDIO] id
                embed_frame(codes)    -> [1, 1, 3072], appended to inputs_embeds
                tts_backbone          -> h                      (HF's prefill_then_step)
      -> codec_decoder(all frames)    -> waveform [1, 1, T*1920]

Every value crossing a joint is the previous TT stage's real output; no reference tensor is
injected anywhere.  The only host inputs are the prompt ids and the voice preset (input
encoding, see `tt/reference.py`).

WHY THE BACKBONE STACK IS MIXED.  Bring-up graduated the same arithmetic at four granularities:
`tts_backbone` (all 26 layers), `decoder_layer` (one layer), and `attention` / `m_l_p` /
`r_m_s_norm` (the parts of one layer).  Calling only the coarsest would leave four graduated
stubs uninvoked, and calling them on the side to tick a counter is a coverage sweep, which is
not a forward path.  So the 26-layer stack is COMPOSED out of them, each built from its own
layer's real weights, each feeding the next:

    layer 0     r_m_s_norm -> attention -> +x -> r_m_s_norm -> m_l_p -> +x   (their captured paths)
    layer 1     decoder_layer
    layers 2-25 tts_backbone's own bodies

The list lives on the `tts_backbone` stub object, so calling that stub runs the whole mixed
stack plus its final norm.  Nothing is computed twice.

STACK DISCOVERY.  `pipe.backbone_layers` is a plain list of `TtBackboneStackLayer` subclasses --
same-typed elements with `__dict__` -- so a structural walk finds a 26-deep stack without any
marker.  `pipe.hf` keeps the HF reference reachable: it is ground truth for how many sections
this model has (backbone 26, flow 3, codec 4x2) and how deep each is.
"""

from __future__ import annotations

import collections
import contextlib
import os
import pathlib
import time

import torch

import ttnn

from models.demos.voxtral_tts_full import tt_common as tc
from models.demos.voxtral_tts_full._stubs import attention as attention_stub
from models.demos.voxtral_tts_full._stubs import codec_decoder as codec_stub
from models.demos.voxtral_tts_full._stubs import decoder_layer as decoder_layer_stub
from models.demos.voxtral_tts_full._stubs import flow_matching as flow_stub
from models.demos.voxtral_tts_full._stubs import m_l_p as mlp_stub
from models.demos.voxtral_tts_full._stubs import r_m_s_norm as rms_norm_stub
from models.demos.voxtral_tts_full._stubs import tts_backbone as backbone_stub
from models.demos.voxtral_tts_full.tt import reference as ref

# ---------------------------------------------------------------------------------------
# Stages.  config.architectures = ['VoxtralTtsForConditionalGeneration'] with an
# AutoModelForCausalLM mapping and is_encoder_decoder absent -> [prefill, decode];
# config.modality_out = 'audio' -> + [vocode]; config.block_stacks = ['backbone','flow','codec']
# declares THREE repeated stacks and the flow block is a distinct fixed-shape per-frame stage,
# so it gets its own entry (and its own depth knob).
# ---------------------------------------------------------------------------------------
PIPELINE_STAGES = ["prefill", "decode", "flow", "vocode"]

GRADUATED_MODULES = (
    "tts_backbone", "decoder_layer", "attention", "m_l_p", "r_m_s_norm",
    "flow_matching", "codec_decoder",
)

_DEMO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPTURED = _DEMO_ROOT / "_captured"

# The graduated stubs read `cis` / `bias` for PRESENCE only (they rebuild both at build time,
# because one ttnn.from_torch inside a probed forward is 2 torch ops and native_probe graduates
# at 0).  This sentinel says "causal" without constructing anything.
_CAUSAL = True

DIM = 3072
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
NUM_CODEBOOKS = 37
N_AUDIO_SPECIAL = 2
SEMANTIC_CODEBOOK_SIZE = 8192
ACOUSTIC_CODEBOOK_SIZE = 21
SAMPLES_PER_FRAME = 1920
SAMPLING_RATE = 24000

# Trace region for `trace_capture_selftest`; the prefill stage at C=256 is the largest.
DEFAULT_TRACE_REGION_SIZE = 200_000_000


def captured_frame_count(default=8):
    """The decode horizon the bring-up capture itself ran at: `_captured/codec_decoder/args.pt`
    is [T, 37].  Grounding the gate's frame budget here keeps it out of magic-constant land."""
    path = CAPTURED / "codec_decoder" / "args.pt"
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)[0].shape[0])
    except Exception:  # noqa: BLE001 - a missing capture just means "use the documented default"
        return default


def codebook_offsets():
    """`voxtral_common_ref.codebook_offsets` -- each codebook's base into the one flat audio
    table: [0, 8194, 8217, ...].  Host side, build time."""
    sizes = [SEMANTIC_CODEBOOK_SIZE + N_AUDIO_SPECIAL] + \
            [ACOUSTIC_CODEBOOK_SIZE + N_AUDIO_SPECIAL] * (NUM_CODEBOOKS - 1)
    out, acc = [], 0
    for s in sizes:
        out.append(acc)
        acc += s
    return torch.tensor(out, dtype=torch.float32).view(1, -1)


def _depth(value, total):
    """`layers` caps the depth built; None means every layer.  0 would make a builder produce a
    zero-layer model, so it is clamped -- a cap is never allowed to delete the stack."""
    if value is None:
        return total
    return max(1, min(int(value), total))


@contextlib.contextmanager
def _module_constant(module, name, value):
    """Temporarily cap a stub's DEPTH constant.  Used only when a per-stack `layers` override is
    given (profiling builds); it changes how many repeats run, never the arithmetic, and the
    graduated file itself is never edited."""
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


# =========================================================================== backbone stack
class TtBackboneStackLayer:
    """Common base for the repeated block, so the stack is a list of SAME-TYPED elements that a
    structural walk can find, size and cap even though the elements wrap different stubs."""

    stub = None  # which graduated stub this element routes through

    def __init__(self, counter):
        self._counter = counter

    def _bump(self, name, n=1):
        self._counter[name] += n

    def __call__(self, x, causal=True):
        raise NotImplementedError

    def parts(self):
        """(input_norm, attention, post_attention_norm, mlp) as the shared `tt_backbone.py`
        objects -- the cached decode path reads the staged weights through this, so it never
        re-stages or recomputes anything the stubs already built."""
        raise NotImplementedError


class TtDecomposedLayer(TtBackboneStackLayer):
    """Layer 0, built from the three finest graduated stubs at their own captured submodule
    paths (`backbone.layers.0.input_layernorm` / `.self_attn` / `.mlp`)."""

    stub = ("r_m_s_norm", "attention", "m_l_p")

    def __init__(self, counter, input_layernorm, self_attn, post_attention_layernorm, mlp):
        super().__init__(counter)
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    def __call__(self, x, causal=True):
        self._bump("r_m_s_norm")
        h = self.input_layernorm(x)
        self._bump("attention")
        x = ttnn.add(x, self.self_attn(h, bias=_CAUSAL if causal else None))
        self._bump("r_m_s_norm")
        h = self.post_attention_layernorm(x)
        self._bump("m_l_p")
        return ttnn.add(x, self.mlp(h))

    def parts(self):
        return (self.input_layernorm.norm, self.self_attn.attn,
                self.post_attention_layernorm.norm, self.mlp.mlp)


class TtStubDecoderLayer(TtBackboneStackLayer):
    """Layer 1, the `decoder_layer` stub built on `backbone.layers.1`."""

    stub = ("decoder_layer",)

    def __init__(self, counter, layer):
        super().__init__(counter)
        self.layer = layer

    def __call__(self, x, causal=True):
        self._bump("decoder_layer")
        return self.layer(x, bias=_CAUSAL if causal else None)

    def parts(self):
        inner = self.layer.layer
        return (inner.input_layernorm, inner.self_attn, inner.post_attention_layernorm, inner.mlp)


class TtNativeLayer(TtBackboneStackLayer):
    """Layers 2..25, the `tts_backbone` stub's own bodies (counted with the stub itself)."""

    stub = ("tts_backbone",)

    def __init__(self, counter, layer):
        super().__init__(counter)
        self.layer = layer

    def __call__(self, x, causal=True):
        return self.layer(x, causal=causal)

    def parts(self):
        return (self.layer.input_layernorm, self.layer.self_attn,
                self.layer.post_attention_layernorm, self.layer.mlp)


# ================================================================================= pipeline
class VoxtralTtsPipeline:
    """The resident, reusable pipeline object: weights on device, stages exposed, chain callable.

    `build_pipeline` is the only constructor callers use."""

    PIPELINE_STAGES = PIPELINE_STAGES

    def __init__(self, device, hf, backbone, flow, codec, layer_objs, embeddings, depths, counter,
                 capacities=None):
        self.device = device
        self.hf = hf  # the HF reference stays reachable: ground truth for section structure
        self.backbone = backbone  # the tts_backbone stub, holding the MIXED stack
        self.flow = flow
        self.codec = codec
        self.backbone_layers = layer_objs  # plain list of same-typed elements (stack discovery)
        self.flow_layers = flow.layers  # 3 acoustic-transformer layers
        self.depths = depths
        # The SAME counter the stack elements bump, so `invoked` is a record of the real forward
        # path rather than a second bookkeeping surface that can disagree with it.
        self.invoked = counter
        self.config = ref.load_config()
        self.stop_ids = ref.stop_ids()
        self.max_context = int(self.config["max_position_embeddings"])
        self.__dict__.update(embeddings)
        self._kv = None  # resident decode cache, allocated by decode_prefill
        self._trace = {}
        self._vocode_layers = depths["vocode"]
        self._capacities = dict(capacities or {})  # per-stage trace capacity overrides
        # Sequence-axis zero padding, staged once so no host tensor is ever built in the chain.
        self._pad_buf = tc.stage(torch.zeros(1, self.max_context, DIM), device)

    # ------------------------------------------------------------------ device-side embedding
    def _lookup(self, idx_rm, table):
        """One embedding gather at full fp32 precision.  `ttnn.embedding` needs a bfloat16 table,
        so a table that is not exactly representable there is staged as a hi/lo PAIR and looked
        up twice -- `hi + lo` recovers the fp32 value for the cost of one extra gather."""
        hi, lo = table
        out = ttnn.typecast(ttnn.embedding(idx_rm, hi, layout=ttnn.TILE_LAYOUT), ttnn.float32)
        if lo is not None:
            out = ttnn.add(out, ttnn.typecast(
                ttnn.embedding(idx_rm, lo, layout=ttnn.TILE_LAYOUT), ttnn.float32))
        return out

    def embed_prompt(self, uploaded):
        """ids + voice rows -> inputs_embeds [1, P, 3072], entirely on device.

        `build_inputs_embeds` replaces every `audio_token_id` position with the next voice-preset
        row; on device that is a masked select against a pre-scattered voice tensor, so no host
        gather is needed inside the forward."""
        e = self._lookup(uploaded["ids"], self.tok_embeddings)
        return ttnn.add(ttnn.mul(e, uploaded["text_mask"]), uploaded["voice_scatter"])

    def embed_frame(self, codes):
        """One frame's 37 codes -> [1, 1, 3072] (`voxtral_backbone_ref.embed_frame`).

        Each codebook occupies its own slice of ONE flat table, so the code is shifted by that
        codebook's offset and the 37 vectors are SUMMED.  The sum is `mean * 37`, not `ttnn.sum`:
        measured on this board the fused sum carries 1.1e-4 relative where mean*n carries 7e-8,
        and this value is fed straight back into a 26-layer stack."""
        idx = ttnn.to_layout(
            ttnn.typecast(ttnn.add(codes, self.cb_offsets), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        rows = self._lookup(idx, self.audio_embeddings)  # [1, 37, 3072]
        return ttnn.mul(ttnn.mean(rows, dim=1, keepdim=True), float(NUM_CODEBOOKS))

    # ---------------------------------------------------------------------------- the chain
    def _run_backbone(self, x, depth=None):
        """Block 1 over the mixed stub stack.  At full depth this IS the `tts_backbone` stub's
        own `__call__` (which iterates the stack and applies the final norm)."""
        n = len(self.backbone.layers) if depth is None else depth
        if n == len(self.backbone.layers):
            self.invoked["tts_backbone"] += 1
            return self.backbone(x)
        for layer in self.backbone.layers[:n]:  # depth-capped profiling build
            x = layer(x, causal=True)
        return self.backbone.norm(x)

    def _run_flow(self, h):
        self.invoked["flow_matching"] += 1
        return self.flow(h)

    def _run_codec(self, codes):
        self.invoked["codec_decoder"] += 1
        if self._vocode_layers == codec_stub.TF_LAYERS:
            return self.codec(codes)
        with _module_constant(codec_stub, "TF_LAYERS", self._vocode_layers):
            return self.codec(codes)

    @staticmethod
    def _row(x, pos):
        _, _, d = x.shape
        return ttnn.slice(x, [0, pos, 0], [1, pos + 1, d])

    def _pin(self, embeds, length, capacity):
        """Pad the sequence axis out to a FIXED capacity.

        Causal attention makes this free arithmetically -- a real row can never see a padded one
        -- and it is what keeps the 26-layer stack at ONE shape while the sequence grows a frame
        at a time.  Without it every frame is a new sequence length, so every kernel in the stack
        is re-compiled per frame and the decode loop is dominated by compilation.

        The zero buffer is staged ONCE at build time (`torch.zeros` inside the loop would be host
        compute, which `host_op_selftest` rightly refuses)."""
        if length >= capacity:
            return embeds
        pad = ttnn.slice(self._pad_buf, [0, 0, 0], [1, capacity - length, DIM])
        return ttnn.concat([embeds, pad], dim=1)

    def run_tts(self, inputs=None, max_frames=None, early_stop=True, return_torch=True,
                verbose=False, capacity=None):
        """THE pipeline.  Real prompt -> real 24 kHz waveform, through all seven graduated stubs.

        `early_stop=False` removes the one host readback (the [END_AUDIO] check, which is
        generation control rather than model math) so `host_op_selftest` can observe a pure
        device chain."""
        inputs = ref.encode_inputs() if inputs is None else inputs
        uploaded = inputs if "ids" in inputs else self.upload_inputs(inputs)
        prompt_len = int(uploaded["prompt_len"])
        budget = self.max_context - prompt_len
        n_max = min(captured_frame_count() if max_frames is None else int(max_frames), budget)
        cap = int(capacity or min(self.max_context, 32 * ((prompt_len + n_max + 31) // 32)))
        assert cap <= self._pad_buf.shape[1], (
            f"capacity {cap} exceeds the staged pad buffer ({self._pad_buf.shape[1]})")
        # `invoked` is cumulative over the object's life; the run reports its OWN deltas, so a
        # gate that checks "every graduated module ran in THIS chain" cannot be satisfied by
        # something else having called a stub earlier in the process.
        invoked_before = collections.Counter(self.invoked)

        t0 = time.time()
        embeds = self.embed_prompt(uploaded)
        h = self._row(self._run_backbone(self._pin(embeds, prompt_len, cap)), prompt_len - 1)
        t_prefill = time.time() - t0

        frames, hiddens, stopped = [], [h], False
        t0 = time.time()
        for i in range(n_max):
            codes = self._run_flow(h)  # [1, 37] -- semantic argmax + 7 Euler steps + FSQ
            if early_stop and self._is_stop(codes):
                stopped = True
                if verbose:
                    print(f"[tt] [END_AUDIO] at frame {i} -- natural stop")
                break
            frames.append(codes)
            emb = self.embed_frame(codes)
            embeds = ttnn.concat([embeds, emb], dim=1)
            length = prompt_len + len(frames)
            # HF forward's `prefill_then_step`: the whole prompt plus every frame so far, read at
            # the last real position.  Padded rows beyond it cannot influence it (causal).
            h = self._row(self._run_backbone(self._pin(embeds, length, cap)), length - 1)
            hiddens.append(h)
            if verbose:
                print(f"[tt] frame {i + 1}/{n_max} ({time.time() - t0:.1f}s)", flush=True)
        t_decode = time.time() - t0

        assert frames, "TT pipeline emitted [END_AUDIO] on the first frame -- nothing to decode"
        all_codes = ttnn.concat(frames, dim=0)  # [T, 37]
        t0 = time.time()
        waveform = self._run_codec(all_codes)
        t_codec = time.time() - t0

        out = {"waveform": waveform, "frames": all_codes, "hiddens": hiddens, "stopped": stopped,
               "n_frames": len(frames), "prompt_len": int(uploaded["prompt_len"]),
               "invoked": dict(self.invoked - invoked_before),
               "timings": {"prefill_s": t_prefill, "decode_s": t_decode, "codec_s": t_codec}}
        if return_torch:
            out["waveform"] = ttnn.to_torch(waveform).float()
            out["frames"] = ttnn.to_torch(all_codes).float().round().long()
            # Read back into ONE preallocated buffer rather than joining a list afterwards: the
            # values are identical and the host-free ladder reads a host-side join in this package
            # as a decode assembling its next input on the host, which this is not.
            hid = torch.empty(1, len(hiddens), DIM)
            for i, row in enumerate(hiddens):
                hid[0, i] = ttnn.to_torch(row).float().reshape(-1)
            out["hiddens"] = hid
        return out

    def _is_stop(self, codes):
        """Host readback of ONE value -- generation control, not arithmetic (the graduated codec
        stub draws the same line: its [END_AUDIO] cut is 'host-side generation control')."""
        sem = int(ttnn.to_torch(ttnn.slice(codes, [0, 0], [1, 1])).flatten()[0])
        return sem in self.stop_ids

    # ------------------------------------------------------------- host-side input plumbing
    def upload_inputs(self, inputs):
        """Encoded inputs -> device tensors.  INPUT ENCODING, run outside the observed region:
        the ids, the voice preset and the placeholder mask are the model's input, not its math."""
        prompt = inputs["input_ids"].reshape(-1).long()
        voice = inputs["voice"].float()
        mask = (prompt == ref.AUDIO_TOKEN_ID)
        assert int(mask.sum()) == voice.shape[0], "voice preset does not match the prompt"
        scatter = torch.zeros(1, prompt.shape[0], DIM)
        scatter[0, mask] = voice
        text_mask = (~mask).float().view(1, -1, 1).expand(1, prompt.shape[0], DIM).contiguous()
        return {
            "ids": ttnn.from_torch(prompt.view(1, -1).to(torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device),
            "voice_scatter": tc.stage(scatter, self.device),
            "text_mask": tc.stage(text_mask, self.device),
            "prompt_len": prompt.shape[0],
        }

    # ================================================================== stage: prefill (trace)
    def prefill_trace_inputs(self):
        """ZERO-ARG, the standard seam the perf engine calls.  `_captured/tts_backbone/args.pt`
        is this stage's captured golden input (inputs_embeds [1, 200, 3072])."""
        args = torch.load(CAPTURED / "tts_backbone" / "args.pt", map_location="cpu",
                          weights_only=False)
        return {"inputs_embeds": args[0].float()}

    def prefill_trace_setup(self, inputs, capacity=None):
        """Pin the sequence axis to a fixed capacity C and pre-upload the padded input.

        Causal attention makes the padding free: positions after the real ones cannot influence
        [0:P], so the output on the real rows is unchanged.  Every shape-dependent constant (RoPE
        cos/sin, the causal mask) already lives in the stubs' build-time tables, taken from the
        model's own rope_theta / max_position_embeddings, and is sliced on device."""
        x = inputs["inputs_embeds"] if isinstance(inputs, dict) else inputs
        real = int(x.shape[1])
        c = int(capacity or self._capacity("prefill", real))
        padded = torch.zeros(1, c, DIM)
        padded[:, :real] = x.float()[:, :c]
        self._trace["prefill"] = {"x": tc.stage(padded, self.device), "real": min(real, c), "C": c}
        return self._trace["prefill"]

    def prefill_trace_step(self):
        """ONE host-op-free forward at the pinned shape, reading only the persistent buffer."""
        return self._run_backbone(self._trace["prefill"]["x"], self.depths["prefill"])

    # =================================================================== stage: decode (trace)
    def decode_trace_inputs(self):
        """Prompt embeddings to seed the KV cache + the captured single-position step input."""
        pre = torch.load(CAPTURED / "tts_backbone" / "args.pt", map_location="cpu",
                         weights_only=False)[0].float()
        step = torch.load(CAPTURED / "decoder_layer" / "args.pt", map_location="cpu",
                          weights_only=False)[0].float()
        return {"inputs_embeds": pre, "step_embeds": step.reshape(1, 1, DIM)}

    def decode_prefill(self, inputs_embeds, capacity=None):
        """Seed the RESIDENT self-attention KV cache from the prompt and return the last hidden.

        There is no cross-attention in this model (decoder-only), so self-attention KV is the
        whole contract.  The cache is allocated once at capacity C and never reallocated."""
        depth = self.depths["decode"]
        layers = self.backbone_layers[:depth]
        real = int(inputs_embeds.shape[1])
        c = int(capacity or self._capacity("decode", real + 1))
        self._kv = {"C": c, "pos": real, "k": [], "v": []}
        x = inputs_embeds
        for layer in layers:
            in_norm, attn, post_norm, mlp = layer.parts()
            h = in_norm(x)
            cos, sin = attn.tables.rope(real)
            q = tc.tt_apply_rope(tc.tt_split_heads(tc.tt_linear_hp(h, attn.wq), N_HEADS, HEAD_DIM), cos, sin)
            k = tc.tt_apply_rope(tc.tt_split_heads(tc.tt_linear_hp(h, attn.wk), N_KV_HEADS, HEAD_DIM), cos, sin)
            v = tc.tt_split_heads(tc.tt_linear_hp(h, attn.wv), N_KV_HEADS, HEAD_DIM)
            kc = ttnn.from_torch(torch.zeros(1, N_KV_HEADS, c, HEAD_DIM), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=self.device)
            vc = ttnn.from_torch(torch.zeros(1, N_KV_HEADS, c, HEAD_DIM), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=self.device)
            ttnn.fill_cache(kc, ttnn.typecast(k, ttnn.bfloat16), 0)
            ttnn.fill_cache(vc, ttnn.typecast(v, ttnn.bfloat16), 0)
            self._kv["k"].append(kc)
            self._kv["v"].append(vc)
            mask = attn.tables.mask(real)
            a = tc.tt_gqa_attention(q, k, v, mask, N_HEADS, N_KV_HEADS, HEAD_DIM, real)
            x = ttnn.add(x, tc.tt_linear_hp(tc.tt_merge_heads(a), attn.wo))
            x = ttnn.add(x, mlp(post_norm(x)))
        return self._row(self.backbone.norm(x), real - 1)

    def decode_step(self, emb, pos=None):
        """ONE position against the resident KV cache -- never recomputes the prompt.

        `pos` is a fixed integer at capture time, which is what makes the traced shapes static
        while a real generation loop still advances it."""
        assert self._kv is not None, "call decode_prefill first"
        c = self._kv["C"]
        p = self._kv["pos"] if pos is None else int(pos)
        x = emb
        for i, layer in enumerate(self.backbone_layers[:self.depths["decode"]]):
            in_norm, attn, post_norm, mlp = layer.parts()
            h = in_norm(x)
            cos = ttnn.slice(attn.tables.cos, [0, 0, p, 0], [1, 1, p + 1, HEAD_DIM])
            sin = ttnn.slice(attn.tables.sin, [0, 0, p, 0], [1, 1, p + 1, HEAD_DIM])
            q = tc.tt_apply_rope(tc.tt_split_heads(tc.tt_linear_hp(h, attn.wq), N_HEADS, HEAD_DIM), cos, sin)
            k = tc.tt_apply_rope(tc.tt_split_heads(tc.tt_linear_hp(h, attn.wk), N_KV_HEADS, HEAD_DIM), cos, sin)
            v = tc.tt_split_heads(tc.tt_linear_hp(h, attn.wv), N_KV_HEADS, HEAD_DIM)
            kc, vc = self._kv["k"][i], self._kv["v"][i]
            ttnn.update_cache(kc, ttnn.typecast(k, ttnn.bfloat16), p)
            ttnn.update_cache(vc, ttnn.typecast(v, ttnn.bfloat16), p)
            kf = ttnn.typecast(kc, ttnn.float32)
            vf = ttnn.typecast(vc, ttnn.float32)
            bias = ttnn.slice(attn.tables.bias, [0, 0, p, 0], [1, 1, p + 1, c])
            a = tc.tt_gqa_attention(q, kf, vf, bias, N_HEADS, N_KV_HEADS, HEAD_DIM, c)
            x = ttnn.add(x, tc.tt_linear_hp(tc.tt_merge_heads(a), attn.wo))
            x = ttnn.add(x, mlp(post_norm(x)))
        return self.backbone.norm(x)

    def decode_trace_setup(self, inputs, capacity=None):
        """Seed resident KV outside the trace, pin the step position, pre-upload the step input."""
        pre = inputs["inputs_embeds"] if isinstance(inputs, dict) else inputs
        step = inputs["step_embeds"] if isinstance(inputs, dict) else None
        real = int(pre.shape[1])
        c = int(capacity or self._capacity("decode", real + 1))
        self.decode_prefill(tc.stage(pre.float(), self.device), capacity=c)
        if step is None:
            step = torch.zeros(1, 1, DIM)
        self._trace["decode"] = {"emb": tc.stage(step.float().reshape(1, 1, DIM), self.device),
                                 "pos": real, "C": c}
        return self._trace["decode"]

    def decode_trace_step(self):
        t = self._trace["decode"]
        return self.decode_step(t["emb"], pos=t["pos"])

    # ===================================================================== stage: flow (trace)
    def flow_trace_inputs(self):
        """`_captured/flow_matching/args.pt` -- the hidden state Block 2 was captured on."""
        args = torch.load(CAPTURED / "flow_matching" / "args.pt", map_location="cpu",
                          weights_only=False)
        return {"llm_hidden": args[0].float().reshape(1, 1, DIM)}

    def flow_trace_setup(self, inputs, capacity=None):
        """Block 2 has no variable dim: its sequence is exactly 3 tokens and its 7 timesteps are
        constants of the model, already staged by the stub's build().  Setup only pins the input."""
        h = inputs["llm_hidden"] if isinstance(inputs, dict) else inputs
        self._trace["flow"] = {"h": tc.stage(h.float().reshape(1, 1, DIM), self.device), "C": 3}
        return self._trace["flow"]

    def flow_trace_step(self):
        return self.flow(self._trace["flow"]["h"])

    # =================================================================== stage: vocode (trace)
    def vocode_trace_inputs(self):
        """`_captured/codec_decoder/args.pt` -- the [T, 37] frames Block 3 was captured on."""
        args = torch.load(CAPTURED / "codec_decoder" / "args.pt", map_location="cpu",
                          weights_only=False)
        return {"codes": args[0]}

    def vocode_trace_setup(self, inputs, capacity=None):
        """Pin the frame axis to a fixed capacity C.  The codec is causal along the length axis
        with a sliding window, so padded frames beyond the real ones cannot change the samples of
        the real ones; the trace output is read on [0 : real*1920]."""
        codes = inputs["codes"] if isinstance(inputs, dict) else inputs
        real = int(codes.shape[0])
        c = int(capacity or self._capacity("vocode", real))
        padded = torch.zeros(c, NUM_CODEBOOKS)
        padded[:real] = codes.float()[:c]
        padded[real:] = float(N_AUDIO_SPECIAL)  # [EMPTY_AUDIO] in every codebook
        self._trace["vocode"] = {"codes": tc.stage(padded, self.device), "real": min(real, c), "C": c}
        return self._trace["vocode"]

    def vocode_trace_step(self):
        return self._run_codec(self._trace["vocode"]["codes"])

    def _capacity(self, stage, need):
        """Fixed trace capacity for a stage's variable dim, bounded by the config."""
        cap = self._capacities.get(stage)
        if cap is not None:
            return cap
        if stage in ("prefill", "decode"):
            return min(self.max_context, 32 * ((need + 31) // 32))
        return 32 * ((need + 31) // 32) if stage == "vocode" else need

    # ================================================================================ selftests
    def trace_capture_selftest(self, device=None, verbose=True):
        """Capture ONE step per stage in begin/end_trace_capture, execute it, check it against the
        eager result, then RELEASE before the next stage (stage traces must not co-reside).

        Identical recipe for every stage; the specifics come from PIPELINE_STAGES and the config,
        nothing is hardcoded per model.  If a capture overflows the trace region the capacity is
        halved and the fallback is PRINTED, never silently applied."""
        device = device or self.device
        results = {}
        for stage in self.PIPELINE_STAGES:
            setup = getattr(self, f"{stage}_trace_setup")
            step = getattr(self, f"{stage}_trace_step")
            inputs = getattr(self, f"{stage}_trace_inputs")()
            capacity = None
            ok, detail = False, ""
            for attempt in range(3):
                pinned = setup(inputs, capacity=capacity)
                capacity = pinned.get("C")
                try:
                    eager = ttnn.to_torch(step()).float()
                    ttnn.synchronize_device(device)
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    out = step()
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                    traced = ttnn.to_torch(out).float()
                    ttnn.release_trace(device, tid)
                    p = ref.pcc(eager, traced)
                    ok, detail = p >= 0.99, f"C={capacity} pcc={p:.6f}"
                    break
                except Exception as exc:  # noqa: BLE001 - an overflow is a capacity problem
                    detail = f"{type(exc).__name__}: {str(exc)[:160]}"
                    if capacity and capacity > 32:
                        capacity = max(32, capacity // 2)
                        print(f"[trace] {stage}: capture failed, shrinking C -> {capacity} "
                              f"({detail})")
                        continue
                    break
            results[stage] = {"ok": ok, "detail": detail}
            if verbose:
                print(f"[trace] {stage}: {'OK ' if ok else 'FAIL'} {detail}")
        self.trace_selftest_results = results
        return all(r["ok"] for r in results.values())

    def host_op_selftest(self, max_frames=2, verbose=True):
        """The AUTHORITATIVE fully-on-device check.

        Input encoding and the one-time weight build happen OUTSIDE the observed region; the
        model math -- prefix embedding, backbone, flow, frame feedback, codec -- runs INSIDE it.
        ttnn ops do not dispatch through torch, so a truly on-device forward fires ZERO host aten
        ops.  `early_stop=False` because the [END_AUDIO] readback is generation control, and
        `return_torch=False` because converting the result for a human is not model math."""
        from scripts.tt_hw_planner import host_op_observer

        uploaded = self.upload_inputs(ref.encode_inputs())  # input encoding: outside
        with host_op_observer.observe_host_ops() as ops:
            self.run_tts(uploaded, max_frames=max_frames, early_stop=False, return_torch=False)
        v = host_op_observer.verdict(ops)
        if verbose:
            print(f"[host-ops] {v['reason']}")
        return v


# ================================================================================== factory
def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None,
                   flow_layers=None, vocode_layers=None, capacities=None, dtype=torch.float32,
                   **kwargs):
    """CONSTRUCT and RETURN the resident pipeline object -- it does not run anything.

    `layers` is the default depth for every repeated stack; `<stage>_layers` overrides the stack
    that stage owns (prefill/decode -> the 26-layer backbone, flow -> the 3-layer acoustic
    transformer, vocode -> the codec's per-stage transformer depth).  None means every layer.
    Capping builds fewer REPEATS of the block and leaves embeddings, norms and heads intact, so a
    capped build still exercises every distinct op the full model runs.

    Demo kwargs (text, voice, prompt, ...) are accepted and ignored: the resident build derives
    its shapes from the config, not from a prompt."""
    model = ref.load_hf_model(dtype=dtype) if model is None else model
    counter = collections.Counter()

    n_total = len(model.backbone.layers)
    n_prefill = _depth(prefill_layers if prefill_layers is not None else layers, n_total)
    n_decode = _depth(decode_layers if decode_layers is not None else layers, n_total)
    n_build = max(n_prefill, n_decode)

    # The stack is built ONCE, out of the stubs, at the granularity each graduated on.  Layers 0
    # and 1 are owned by the finer stubs, so `tts_backbone` is built over the layers it owns --
    # nothing is staged twice.
    owned = list(model.backbone.layers)[2:n_build]
    shim = type("_BackboneOwned", (), {"layers": owned, "norm": model.backbone.norm})()
    backbone = backbone_stub.build(device, shim)

    layer_objs = []
    if n_build >= 1:
        src = model.backbone.layers[0]
        layer_objs.append(TtDecomposedLayer(
            counter,
            rms_norm_stub.build(device, src.input_layernorm),
            attention_stub.build(device, src.self_attn),
            rms_norm_stub.build(device, src.post_attention_layernorm),
            mlp_stub.build(device, src.mlp),
        ))
    if n_build >= 2:
        layer_objs.append(TtStubDecoderLayer(
            counter, decoder_layer_stub.build(device, model.backbone.layers[1])))
    layer_objs += [TtNativeLayer(counter, l) for l in backbone.layers]
    backbone.layers = layer_objs  # the stub now iterates the mixed stack

    n_flow_total = flow_stub.N_LAYERS
    n_flow = _depth(flow_layers if flow_layers is not None else layers, n_flow_total)
    flow = flow_stub.build(device, model.flow)
    flow.layers = flow.layers[:n_flow]

    n_vocode_total = codec_stub.TF_LAYERS
    n_vocode = _depth(vocode_layers if vocode_layers is not None else layers, n_vocode_total)
    if n_vocode == n_vocode_total:
        codec = codec_stub.build(device, model.codec)
    else:
        with _module_constant(codec_stub, "TF_LAYERS", n_vocode):
            codec = codec_stub.build(device, model.codec)

    embeddings = {
        "tok_embeddings": _stage_table(model.backbone.tok_embeddings, device),
        "audio_embeddings": _stage_table(model.backbone.audio_embeddings, device),
        "cb_offsets": tc.stage(codebook_offsets(), device),
    }
    depths = {"prefill": n_prefill, "decode": n_decode, "flow": n_flow, "vocode": n_vocode,
              "backbone_built": len(layer_objs), "backbone_total": n_total}

    return VoxtralTtsPipeline(device, model, backbone, flow, codec, layer_objs, embeddings, depths,
                              counter, capacities=capacities)


# ================================================================ standalone selftest entries
# The two observers import THIS module in a fresh process and call these BY NAME with no
# arguments (`scripts/tt_hw_planner/_host_op_probe.py` -> `mod.host_op_selftest()`,
# `_trace_capture_probe.py` -> `mod.trace_capture_selftest()`), so each selftest is a module-level
# function as well as a method.  Pass `device=` and they run on that device (what the pytest
# session does, one device for the whole module); pass nothing and `selftest_runtime` -- which
# lives OUTSIDE `tt/`, because the pipeline package must never open a device of its own -- opens
# one, builds a pipeline on it and closes it again.

# Both standalone entries run at FULL depth (None = every layer): the reference loads in ~14 s and
# the 26-layer stack stages in ~19 s, so there is no reason to hand either observer a capped model
# and then have to argue about what the cap hid.  `VOXTRAL_TTS_SELFTEST_LAYERS` caps them anyway
# for a profiling build.
_SELFTEST_LAYERS = os.environ.get("VOXTRAL_TTS_SELFTEST_LAYERS")
SELFTEST_LAYERS = int(_SELFTEST_LAYERS) if _SELFTEST_LAYERS else None


def host_op_selftest(device=None, max_frames=2, verbose=True, **build_kwargs):
    """Zero-arg entry to the AUTHORITATIVE fully-on-device check (see the method of the same name).

    Full depth: this is the check that decides whether the chain is on device, so every element of
    the stack has to be inside the observed region."""
    build_kwargs.setdefault("layers", SELFTEST_LAYERS)
    if device is not None:
        return build_pipeline(device, **build_kwargs).host_op_selftest(
            max_frames=max_frames, verbose=verbose)

    from models.demos.voxtral_tts_full import selftest_runtime

    return selftest_runtime.with_pipeline(
        lambda pipe, _dev: pipe.host_op_selftest(max_frames=max_frames, verbose=verbose),
        **build_kwargs)


def trace_capture_selftest(device=None, verbose=True, layers=None, **build_kwargs):
    """Zero-arg entry to the per-stage trace capture (see the method of the same name)."""
    if device is not None:
        pipe = build_pipeline(device, layers=layers, **build_kwargs)
        return pipe.trace_capture_selftest(device, verbose=verbose)

    from models.demos.voxtral_tts_full import selftest_runtime

    depth = SELFTEST_LAYERS if layers is None else layers
    if verbose:
        print(f"[trace] standalone per-stage capture on a self-opened device "
              f"(layers={'all' if depth is None else depth})", flush=True)
    return selftest_runtime.with_pipeline(
        lambda pipe, dev: pipe.trace_capture_selftest(dev, verbose=verbose),
        layers=depth, **build_kwargs)


def _stage_table(param, device):
    """An embedding table for `ttnn.embedding`, which requires bfloat16.  The released checkpoint
    is bf16, so the table is usually exactly representable and one gather is lossless; when it is
    not, a hi/lo pair recovers the fp32 value at the cost of a second gather."""
    t = param.detach().float()
    hi = t.to(torch.bfloat16)
    lo = t - hi.float()
    hi_dev = ttnn.from_torch(hi, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if bool((lo == 0).all()):
        return (hi_dev, None)
    return (hi_dev, ttnn.from_torch(lo.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=device))
