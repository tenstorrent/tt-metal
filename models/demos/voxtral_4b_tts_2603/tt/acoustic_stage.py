# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ACOUSTIC stage of `mistralai/Voxtral-4B-TTS-2603`: one text hidden state -> one audio frame.

This is the flow-matching sampler. Per frame it produces 37 integer codes for each of the B
samples:

    semantic_logit = semantic_codebook_output(llm_hidden)     # reads the RAW backbone hidden state
    semantic_logit[:, empty_audio] = -inf; semantic_logit[:, n_special + 8192:] = -inf
    semantic_code  = argmax(semantic_logit, -1)               # [B, 1]

    x = x_0                                                   # host-drawn noise, [B, 36]
    for i in 0..6:                                            # 7 Euler steps
        v_all = velocity_field(cat([x, x]), llm_proj_cfg, t_proj_i)   # 2B rows: cond ; uncond
        v     = cfg_alpha * v_all[:B] + (1 - cfg_alpha) * v_all[B:]
        x     = x + v * dt_i
    codes = round(((clamp(x, -1, 1) + 1) / 2) * 20) + n_special
    codes[semantic_code == end_audio] = empty_audio + n_special
    frame = cat([semantic_code, codes], -1)                   # [B, 37] uint32

Everything above happens ON DEVICE in ttnn -- including the argmax over 8194 masked logits, the
clamp, the round and the +2 offset. `x_0` is an INPUT: the reference draws it inside
`decode_one_frame`, so a port that drew its own could never be compared. The orchestrator draws it
once on the host and hands the same tensor to this stage and to the torch golden.

THE VELOCITY FIELD IS BIDIRECTIONAL, NON-CAUSAL AND RoPE-FREE. Its sequence is literally three
tokens -- `[input_projection(x_t), time_projection(time_embedding(t)), llm_projection(llm_hidden)]`
-- and only token 0 is read out. Each row attends only to its own three tokens (an additive mask). The `rope_theta: 10000.0` in the acoustic sub-config is dead.

ONE BODY FOR EVERY ROW. The velocity field is `tt/modules/flow_matching_audio_transformer`; the
CFG-doubled batch (2B rows) goes through it in one call per Euler step, on its compact layout
(token t of every row in rows t*2B..) when 2B is a multiple of 32.

NUMERICS. float32 activations against bfloat16 weights, HiFi4 with `fp32_dest_acc_en`, and the
attention softmax written out in float32 (SDPA rejects float32). This stage ends in two quantizations -- an argmax over 8194
logits and a round onto 21 levels 0.1 apart in x -- and a single flipped code makes that frame's
audio diverge, so the accuracy has to come from the numerics rather than from a looser threshold.

CONSTANTS ARE PERSISTENT BUFFERS. The timestep rows, the CFG zero-conditioning rows, the
precomputed time projections and the semantic logit mask are all created once (at build
time for the batch the caller declares) and reused, so nothing in `decode_frame` allocates and
the stage can be captured in a trace.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.modules import flow_matching_audio_transformer

_TILE = 32

# Additive mask value for the masked semantic logits. The reference writes -inf; a finite but
# unreachable value keeps the argmax identical without putting inf into a tensor that is also
# PCC-compared.
_LOGIT_NEG = -1.0e30


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


# --------------------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------------------


class AcousticStage:
    """The resident acoustic sampler. Built by `build_acoustic_stage`; never opens a device."""

    def __init__(
        self,
        device,
        field,
        n_layers,
        n_steps,
        n_acoustic,
        dim,
        semantic_out,
        levels,
        n_special,
        end_audio_id,
        empty_audio_id,
        semantic_codebook_size,
        timesteps,
    ):
        self.device = device
        self.field = field  # flow_matching_audio_transformer: (llm, x_t, t) -> (velocity, semantic logits)
        self.n_layers = int(n_layers)
        self.n_steps = int(n_steps)
        self.n_acoustic = int(n_acoustic)
        self.dim = int(dim)
        self.semantic_out = int(semantic_out)
        self.levels = int(levels)
        self.n_special = int(n_special)
        self.end_audio_id = int(end_audio_id)
        self.empty_audio_id = int(empty_audio_id)
        self.semantic_codebook_size = int(semantic_codebook_size)
        self.timesteps = [float(t) for t in timesteps]
        self.dts = [self.timesteps[i + 1] - self.timesteps[i] for i in range(self.n_steps)]
        self._consts = {}

    # ---------------------------------------------------------------- persistent constants

    def _const(self, key, make):
        buf = self._consts.get(key)
        if buf is None:
            buf = make()
            self._consts[key] = buf
        return buf

    def _zeros(self, rows):
        """The CFG zero-conditioning rows. `llm_projection` has no bias, so these project to 0."""
        return self._const(
            ("zeros", rows), lambda: _from_torch(torch.zeros(rows, self.dim), self.device, dtype=ttnn.float32)
        )

    def _ones(self, rows):
        return self._const(("ones", rows), lambda: _from_torch(torch.ones(rows, 1), self.device, dtype=ttnn.float32))

    def _t_rows(self, rows, step):
        """`t_i` broadcast to `[rows, 1]` -- the timestep the reference reads out of `_timesteps`."""
        return self._const(
            ("t", rows, step),
            lambda: _from_torch(
                torch.full((rows, 1), self.timesteps[step], dtype=torch.float32), self.device, dtype=ttnn.float32
            ),
        )

    def _t_proj(self, rows, step):
        """`time_projection(time_embedding(t_step))` for `rows` rows -- a CONSTANT, since the
        timesteps are fixed; computed once per (rows, step) instead of every frame. Only the
        compact (tile-aligned) layout consumes it."""
        if rows % _TILE:
            return None
        return self._const(("t_proj", rows, step), lambda: self.field.time_projection(self._t_rows(rows, step)))

    def _logit_mask(self, rows):
        """Additive mask for `empty_audio` and the unused tail of the padded semantic head."""

        def make():
            m = torch.zeros(rows, self.semantic_out)
            m[:, self.empty_audio_id] = _LOGIT_NEG
            m[:, self.n_special + self.semantic_codebook_size :] = _LOGIT_NEG
            return _from_torch(m, self.device, dtype=ttnn.float32)

        return self._const(("logit_mask", rows), make)

    def _empty_codes(self, rows):
        """`empty_audio + n_special` on every acoustic slot -- the frame a finished row emits."""
        return self._const(
            ("empty_codes", rows),
            lambda: _from_torch(
                torch.full((rows, self.n_acoustic), float(self.empty_audio_id + self.n_special)),
                self.device,
                dtype=ttnn.float32,
            ),
        )

    def prebuild(self, batch):
        """Create every per-batch constant now, so the first `decode_frame` allocates nothing."""
        batch = int(batch)
        if batch < 1:
            return
        self._logit_mask(batch)
        self._empty_codes(batch)
        self._zeros(batch)
        self._ones(batch)
        for i in range(self.n_steps):
            self._t_rows(2 * batch, i)
            self._t_proj(2 * batch, i)

    def _fp32(self, tensor, rows, width):
        t = tensor
        if len(list(t.shape)) != 2 or int(t.shape[0]) != rows:
            t = ttnn.reshape(t, [rows, width])
        if t.dtype != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32)
        return t

    # ---------------------------------------------------------------- public API

    def decode_frame(self, llm_hidden, x0, cfg_alpha, probe=None):
        """One audio frame: `llm_hidden [B, 3072]`, `x0 [B, 36]`, `cfg_alpha [B]` -> `[B, 37]` uint32.

        Runs the reference's 7 Euler steps with classifier-free guidance, then the reference's own
        discretization (clamp / rescale / round / +n_special, with finished rows forced to
        `empty_audio + n_special`), and concatenates the semantic code in front. Every step is on
        device; nothing comes back to the host.

        `probe`, if a dict, receives the two CONTINUOUS tensors the discretization rounds --
        `x_final` and `semantic_logits` -- as device tensors. They are the quantities a numeric
        comparison against the reference can actually be made on (the codes downstream of them are
        integers on a 0.1-spaced grid). Nothing is copied to the host here; the caller decides.
        """
        n = int(llm_hidden.shape[0])
        llm_hidden = self._fp32(llm_hidden, n, self.dim)
        x = self._fp32(x0, n, self.n_acoustic)
        alpha = self._fp32(cfg_alpha, n, 1)
        one_minus = ttnn.subtract(self._ones(n), alpha)
        # CFG doubles the BATCH axis: [cond rows ; zero-conditioned rows].
        llm_cfg = ttnn.concat([llm_hidden, self._zeros(n)], dim=0)
        # Per-frame: the llm projection + semantic head, computed at step 0 and reused.
        cache = {}
        sem = None
        for i in range(self.n_steps):
            v_all, sem_all = self.field(
                llm_cfg,
                x_t=ttnn.concat([x, x], dim=0),
                t=self._t_rows(2 * n, i),
                step_cache=cache,
                t_proj=self._t_proj(2 * n, i),
            )
            v_cond = ttnn.slice(v_all, [0, 0], [n, self.n_acoustic])
            v_unc = ttnn.slice(v_all, [n, 0], [2 * n, self.n_acoustic])
            v = ttnn.add(ttnn.multiply(v_cond, alpha), ttnn.multiply(v_unc, one_minus))
            x = ttnn.add(x, ttnn.multiply(v, self.dts[i]))
            if sem is None:
                # The semantic head reads only `llm_hidden`, so it is the same at every step:
                # keep the conditional rows of the first step's output and never recompute it.
                sem = ttnn.slice(sem_all, [0, 0], [n, self.semantic_out])

        if probe is not None:
            probe["x_final"] = x
            probe["semantic_logits"] = sem
        return self.discretize(x, sem)

    def discretize(self, x, semantic_logits):
        """`x [B, 36]` + raw semantic logits -> the frame `[B, 37]` uint32, entirely on device."""
        batch = int(x.shape[0])
        scale = 0.5 * float(self.levels - 1)

        codes = ttnn.add(
            ttnn.round(ttnn.multiply(ttnn.add(ttnn.clamp(x, -1.0, 1.0), 1.0), scale)),
            float(self.n_special),
        )

        masked = ttnn.add(semantic_logits, self._logit_mask(batch))
        semantic_code = ttnn.argmax(ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)

        code_f = ttnn.typecast(ttnn.to_layout(semantic_code, ttnn.TILE_LAYOUT), ttnn.float32)
        finished = ttnn.repeat(ttnn.eq(code_f, float(self.end_audio_id)), [1, self.n_acoustic])
        codes = ttnn.where(finished, self._empty_codes(batch), codes)

        acoustic = ttnn.to_layout(ttnn.typecast(codes, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.concat([semantic_code, acoustic], dim=-1)

    def __repr__(self):
        return f"AcousticStage(n_layers={self.n_layers}, n_steps={self.n_steps}, n_acoustic={self.n_acoustic})"


def build_acoustic_stage(device, hf_model, layers=None, batch=common.DEFAULT_BATCH):
    """Build the resident acoustic stage on `device` from `hf_model.acoustic_transformer`.

    `layers` caps the 3 acoustic blocks (`None` means all of them). `batch` is the batch the caller
    intends to drive -- it only pre-creates the persistent constants, and a different batch still
    works (its constants are made on first use).

    Never opens a device. `torch` is used here, at build time, to stage weights; nothing reachable
    from `decode_frame` touches torch or an `hf_model` submodule.
    """
    at = hf_model.acoustic_transformer
    args = at.acoustic_transformer_args
    full_depth = len(at.layers_ids)
    n_layers = full_depth if layers is None else min(int(layers), full_depth)
    if n_layers < 1:
        raise ValueError(f"acoustic layers={layers}: need at least one block")
    rows_hint = int(batch) if batch else 0

    # The batch hint pre-creates the tile pad for the CFG-doubled row count `decode_frame` feeds it.
    field = flow_matching_audio_transformer.build(
        device, at, batch=(2 * rows_hint if rows_hint else None), layers=n_layers
    )
    stage = AcousticStage(
        device=device,
        field=field,
        n_layers=n_layers,
        n_steps=int(args.n_decoding_steps),
        n_acoustic=int(at.model_args.n_acoustic_codebook),
        dim=int(args.dim),
        semantic_out=int(at.semantic_codebook_output.out_features),
        levels=int(at.acoustic_embeddings_levels),
        n_special=common.n_audio_special_tokens(hf_model),
        end_audio_id=common.audio_stop_token_id(hf_model),
        empty_audio_id=common.audio_empty_token_id(hf_model),
        semantic_codebook_size=int(at.model_args.semantic_codebook_size),
        timesteps=at._timesteps.detach().to(torch.float32).tolist(),
    )
    stage.prebuild(rows_hint)
    return stage
