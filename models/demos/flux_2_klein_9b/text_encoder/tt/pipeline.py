# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN forward pass for FLUX.2-klein-9B's text encoder.

Both demo entrypoints and both e2e tests import `build_pipeline` from here and
call the same `run_*` methods, so a green test and a working demo are the same
fact rather than two hopes.

The chain is an explicit Python function over the graduated bring-up stubs in
`models/tt_transformers/demo/flux_2_klein_9b_text_encoder/_stubs/`. No HF module
is ever called inside it; HF appears only in `model_ref.py` (golden values) and
in `*_trace_setup` (seeding fixed constants before capture).

    Call 1  text -> text        token_embed -> rotary_embedding -> encoder_stack
                                (36 x layer/decoder_layer{attention{r_m_s_norm},
                                 mlp/m_l_p}) -> decoder_head -> ttnn.argmax -> loop
    Call 2  text -> embedding   the same chain, stopped before decoder_head

Topology is TP=8 (mesh 1x8), the layout the components graduated at: q/k/v and
gate/up column-parallel, o_proj and down row-parallel with one all_reduce each,
lm_head column-parallel over vocab with an all_gather, and every lookup table and
norm gamma replicated. The KV cache is indexed by this chip's OWN kv heads, so it
inherits that split for free.
"""
from __future__ import annotations

import contextlib
import importlib
import os
from typing import Dict, List, Optional

import torch

import ttnn

from . import model_ref

# --------------------------------------------------------------------------
# Stage list. Derived from Source A: config.json `architectures` is
# ["Qwen3ForCausalLM"] and the config declares no encoder-decoder sub-configs,
# so the phases are the decoder-only pair. No speech output -> no vocode stage.
# --------------------------------------------------------------------------
PIPELINE_STAGES = ["prefill", "decode"]

_STUB_PKG = "models.demos.flux_2_klein_9b.text_encoder._stubs"

# Every graduated component, and the stub module that owns its body. `mlp` and
# `m_l_p` re-export ONE TtMLP; `layer` and `decoder_layer` re-export ONE
# TtDecoderLayer -- they are two discovery passes' names for one module of the
# checkpoint, which is why routing them separately would create drift, not
# coverage.
GRADUATED_COMPONENTS = (
    "token_embed",
    "r_m_s_norm",
    "rotary_embedding",
    "attention",
    "m_l_p",
    "mlp",
    "decoder_layer",
    "layer",
    "encoder_stack",
    "decoder_head",
)

TRACE_PREFILL_C = int(os.environ.get("TT_TRACE_PREFILL_C", "128"))
TRACE_DECODE_C = int(os.environ.get("TT_TRACE_DECODE_C", "256"))


def _stub(name: str):
    return importlib.import_module(f"{_STUB_PKG}.{name}")


def _stub_class(name: str):
    """The class each component name routes to (aliases resolve to one class).

    A stub that DEFINES its body wins on that body; only an alias module -- one
    that defines none and re-exports exactly one -- resolves to the imported class.
    Picking the first `Tt*` name in the module namespace would be wrong: several
    stubs import another stub's body (attention imports TtRMSNorm, decoder_layer
    imports TtAttention and TtMLP), so the import would shadow the real one.
    """
    mod = _stub(name)
    tt_classes = [v for v in vars(mod).values() if isinstance(v, type) and v.__name__.startswith("Tt")]
    own = [v for v in tt_classes if getattr(v, "__module__", "") == mod.__name__]
    if len(own) == 1:
        return own[0]
    if own:
        raise RuntimeError(f"stub {name} defines {len(own)} bodies: {[c.__name__ for c in own]}")
    if len(tt_classes) == 1:
        return tt_classes[0]  # an alias module: exactly one re-exported body
    raise RuntimeError(f"cannot resolve the body of stub {name} from {[c.__name__ for c in tt_classes]}")


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _num_devices(device) -> int:
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


# ==========================================================================
#  Gate-2 instrumentation: observe the REAL forward, never drive it.
# ==========================================================================


@contextlib.contextmanager
def graduated_invocation_probe():
    """Count `__call__`s of the graduated bodies during whatever runs inside.

    This OBSERVES the real chain -- it does not call anything itself, so a
    component only shows up if the pipeline's own forward reached it. Alias names
    share a class and therefore share a count, which is the honest answer: they
    are one module of the checkpoint.
    """
    classes = {}
    for name in GRADUATED_COMPONENTS:
        classes.setdefault(_stub_class(name), []).append(name)

    counts: Dict[str, int] = {name: 0 for name in GRADUATED_COMPONENTS}
    originals = {}
    for cls, names in classes.items():
        originals[cls] = cls.__call__

        def make(cls=cls, names=names, orig=cls.__call__):
            def counted(self, *a, **kw):
                for n in names:
                    counts[n] += 1
                return orig(self, *a, **kw)

            return counted

        cls.__call__ = make()
    try:
        yield counts
    finally:
        for cls, orig in originals.items():
            cls.__call__ = orig


# ==========================================================================
#  The pipeline
# ==========================================================================


class Flux2Klein9BTextEncoderPipeline:
    """Resident TT pipeline: built once, then called for either task head."""

    def __init__(
        self,
        device,
        hf_model,
        layers: Optional[int] = None,
        prefill_layers: Optional[int] = None,
        decode_layers: Optional[int] = None,
        kv_capacity: Optional[int] = None,
        batch: int = 1,
    ) -> None:
        if batch != 1:
            raise RuntimeError("this text-encoder pipeline is built for batch=1 (the demo/gate shape)")

        self.device = device
        self.hf_model = hf_model  # reference/ground truth, NEVER called in the forward
        self.config = hf_model.config
        self.num_devices = _num_devices(device)
        self.batch = batch
        self.hidden_size = int(self.config.hidden_size)
        self.head_dim = int(self.config.head_dim)
        self.vocab_size = int(self.config.vocab_size)
        self.max_position_embeddings = int(self.config.max_position_embeddings)

        depth = self._resolve_depth(layers, prefill_layers, decode_layers)
        self.n_layers = depth

        embed_mod, rope_mod = _stub("token_embed"), _stub("rotary_embedding")
        stack_mod, head_mod = _stub("encoder_stack"), _stub("decoder_head")

        # --- the graduated bodies, built from the reference's own weights.
        self.token_embed = embed_mod.build(device, hf_model.model.embed_tokens)
        self.rotary_embedding = rope_mod.build(device, hf_model.model.rotary_emb)
        self.encoder_stack = stack_mod.build(device, hf_model.model, layers=depth)
        self.decoder_head = head_mod.build(device, hf_model.lm_head)

        # A plain list of same-typed elements: the repeated stack, discoverable by
        # a walk over the object this module's factory returns.
        self.layers = self.encoder_stack.layers

        # --- resident decode state, allocated ONCE so no step allocates on host.
        self.kv_capacity = int(kv_capacity or TRACE_DECODE_C)
        self.kv_caches = self.encoder_stack.allocate_kv_caches(self.batch, self.kv_capacity)
        self._cur_pos = ttnn.from_torch(
            torch.zeros(self.batch, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate(),
        )
        self._next_id = ttnn.from_torch(
            torch.zeros(self.batch, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate(),
        )
        self._trace_state: Dict[str, dict] = {}

    # ------------------------------------------------------------- plumbing

    def _resolve_depth(self, layers, prefill_layers, decode_layers) -> int:
        """`layers` is the default depth; each stage that owns a stack may override.

        This model has exactly ONE repeated stack (`model.layers`, 36 x
        Qwen3DecoderLayer) and BOTH stages run it -- prefill fills the KV cache the
        decode step then reads. Two different depths would mean two different
        models, so a genuine conflict is refused rather than silently resolved.
        """
        overrides = {"prefill": prefill_layers, "decode": decode_layers}
        stated = {k: int(v) for k, v in overrides.items() if v is not None}
        if len(set(stated.values())) > 1:
            raise RuntimeError(
                f"prefill and decode share one stack (model.layers), so their depths cannot "
                f"differ; got {stated}. Pass a single `layers=` instead."
            )
        depth = next(iter(stated.values()), None)
        if depth is None:
            depth = self.config.num_hidden_layers if layers is None else int(layers)
        if depth < 1:
            raise RuntimeError(f"layers must be >= 1 (None = every layer); got {depth}")
        return min(depth, int(self.config.num_hidden_layers))

    def _replicate(self):
        if self.num_devices <= 1:
            return None
        return ttnn.ReplicateTensorToMesh(self.device)

    def _to_torch(self, tensor, rows: int = 1):
        """Read a REPLICATED device tensor back (every chip holds the same answer)."""
        if self.num_devices <= 1:
            return ttnn.to_torch(tensor)
        full = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return full[:rows]

    def _upload_ids(self, tokens: torch.Tensor):
        """Upload a PROMPT (setup, once per call). The decode loop never comes here:
        its next token is already a device tensor produced by the on-device argmax."""
        return ttnn.from_torch(
            tokens.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate(),
        )

    def _upload_positions(self, positions: torch.Tensor):
        """float32 on purpose: positions run to max_position_embeddings (40960),
        which bf16 cannot represent as exact integers past 256."""
        return ttnn.from_torch(
            positions.to(torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate(),
        )

    def _upload_rope(self, cos: torch.Tensor, sin: torch.Tensor):
        out = []
        for t in (cos, sin):
            out.append(
                ttnn.from_torch(
                    t.reshape(1, 1, -1, self.head_dim).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self._replicate(),
                )
            )
        return out[0], out[1]

    # ---------------------------------------------------- graduated stage ops

    def _rope_on_device(self, hidden, position_ids_tt):
        """The graduated `rotary_embedding` body, on device, then cast to the
        activation dtype so the attention ports can use it without a host hop."""
        cos, sin = self.rotary_embedding(hidden, position_ids=position_ids_tt)
        return ttnn.typecast(cos, ttnn.bfloat16), ttnn.typecast(sin, ttnn.bfloat16)

    def _embed(self, ids_tt):
        return self.token_embed(ids_tt)

    # ============================== the chain ==============================

    def run_prefill(self, ids_tt, position_ids_tt, *, seed_kv: bool = False):
        """token_embed -> rotary_embedding -> encoder_stack. Returns [1, S, H]."""
        hidden = self._embed(ids_tt)
        cos, sin = self._rope_on_device(hidden, position_ids_tt)
        return self.encoder_stack(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
            is_causal=True,
            kv_caches=self.kv_caches if seed_kv else None,
            mode="prefill",
        )

    def _logits_at(self, hidden, index: int, seq_len: int):
        """decoder_head on ONE position of the encoded stream -> [1, 1, 1, vocab]."""
        h4 = ttnn.reshape(hidden, (1, 1, seq_len, self.hidden_size))
        last = ttnn.slice(h4, (0, 0, index, 0), (1, 1, index + 1, self.hidden_size))
        return self.decoder_head(last)

    def decode_prefill(self, ids_tt, position_ids_tt, seq_len: int, real_len: int):
        """Seed the resident KV cache and emit the first next-token logits.

        Right-padding is safe precisely because the stack runs CAUSALLY: a padded
        position can only be attended to by later padded positions, so the encoded
        stream on [0:real_len] -- and the cache rows under it -- are unchanged.
        """
        hidden = self.run_prefill(ids_tt, position_ids_tt, seed_kv=True)
        logits = self._logits_at(hidden, real_len - 1, seq_len)
        self._set_position(real_len)
        return logits

    def decode_step(self):
        """ONE autoregressive token: token_embed -> rotary_embedding -> encoder_stack
        over the RESIDENT KV cache -> decoder_head.

        The input is whatever token is resident in `_next_id` -- i.e. the previous
        step's own on-device argmax -- and the cursor advances on device, so the loop
        never needs a reference value to keep going.
        """
        hidden = self._embed(self._next_id)
        hidden = ttnn.reshape(hidden, (1, 1, self.batch, self.hidden_size))
        cos, sin = self._rope_on_device(hidden, self._position_f())
        x = self.encoder_stack(
            hidden,
            position_embeddings=(cos, sin),
            kv_caches=self.kv_caches,
            cur_pos=self._cur_pos,
            mode="decode",
        )
        logits = self.decoder_head(x)
        ttnn.plus_one(self._cur_pos)
        return logits

    # --------------------------------------------------------- decode state

    def _set_position(self, position: int):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.full((self.batch,), int(position), dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._replicate(),
            ),
            self._cur_pos,
        )

    def _position_f(self):
        """The current position as the float32 [1, 1] the RoPE table generator
        wants -- derived ON DEVICE from the int cursor, so there is one cursor."""
        p = ttnn.to_layout(self._cur_pos, ttnn.TILE_LAYOUT)
        return ttnn.reshape(ttnn.typecast(p, ttnn.float32), (self.batch, 1))

    def _read_next_id(self) -> int:
        """Read back the token the device just chose -- for the OUTPUT text and the
        stop check only. It is not how the next step is fed: `decode_step` consumes
        the device tensor `_next_id` directly, so nothing on host closes the loop."""
        return int(self._to_torch(self._next_id, rows=self.batch).flatten()[0])

    # ============================ task heads ===============================

    def run_prompt_encoding(self, prompt: Optional[str] = None, input_ids: Optional[torch.Tensor] = None):
        """Call 2 -- text -> the prompt embedding Flux2Transformer2DModel consumes."""
        ids = input_ids if input_ids is not None else model_ref.encode_prompt(prompt or model_ref.DEFAULT_PROMPT)
        real_len = int(ids.shape[-1])
        padded, positions = self._pad_inputs(ids)
        hidden = self.run_prefill(self._upload_ids(padded), self._upload_positions(positions))
        out = self._to_torch(hidden, rows=1).reshape(1, -1, self.hidden_size)
        return out[:, :real_len, :].float()

    def run_text_generation(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        collect_logits: bool = True,
    ):
        """Call 1 -- text -> text, one explicit autoregressive chain.

        Decoding stops on the model's OWN end tokens (generation_config's
        eos_token_id). `max_new_tokens` is only the safety bound.
        """
        ids = input_ids if input_ids is not None else model_ref.encode_prompt(prompt or model_ref.DEFAULT_PROMPT)
        real_len = int(ids.shape[-1])
        horizon = max_new_tokens or model_ref.resolve_max_new_tokens(self.hf_model, real_len)
        horizon = min(horizon, self.kv_capacity - real_len)
        if horizon < 1:
            raise RuntimeError(f"kv_capacity={self.kv_capacity} leaves no room for a prompt of {real_len} tokens")
        stop_ids = set(model_ref.stop_token_ids(self.hf_model))

        padded, positions = self._pad_inputs(ids)
        logits = self.decode_prefill(
            self._upload_ids(padded), self._upload_positions(positions), int(padded.shape[-1]), real_len
        )

        token_ids: List[int] = []
        step_logits: List[torch.Tensor] = []
        for _ in range(horizon):
            if collect_logits:
                step_logits.append(self._to_torch(logits, rows=1).reshape(-1)[: self.vocab_size].float())
            # The argmax of THIS step's logits becomes the next step's input token --
            # the joint is the TT tensor itself, never a reference value.
            self._next_id = ttnn.reshape(ttnn.argmax(logits, dim=-1, keepdim=True), (self.batch, 1))
            token = self._read_next_id()
            token_ids.append(token)
            if token in stop_ids:
                break
            logits = self.decode_step()

        return {
            "token_ids": token_ids,
            "text": model_ref.decode_tokens(token_ids),
            "step_logits": torch.stack(step_logits) if step_logits else torch.empty(0),
            "prompt_len": real_len,
            "horizon": horizon,
        }

    def _pad_inputs(self, ids: torch.Tensor):
        """Right-pad to a whole number of tiles; positions keep counting."""
        real_len = int(ids.shape[-1])
        padded_len = _ceil_to(real_len, ttnn.TILE_SIZE)
        pad_id = int(getattr(self.hf_model.generation_config, "pad_token_id", 0) or 0)
        padded = torch.full((1, padded_len), pad_id, dtype=torch.long)
        padded[0, :real_len] = ids[0, :real_len]
        positions = torch.arange(padded_len, dtype=torch.long).unsqueeze(0)
        return padded, positions

    # ======================================================================
    #  Command 3 -- the trace contract, one pair of hooks per PIPELINE_STAGE.
    # ======================================================================

    @property
    def trace_capacity(self) -> Dict[str, int]:
        if not hasattr(self, "_trace_capacity"):
            self._trace_capacity = {
                "prefill": min(TRACE_PREFILL_C, self.max_position_embeddings),
                "decode": self.kv_capacity,
            }
        return self._trace_capacity

    # ------------------------------------------------------------- inputs

    def _captured_position_ids(self, seq_len: int) -> torch.Tensor:
        """Positions in the convention the capture pass recorded.

        `_captured/decoder_layer/kwargs.pt` holds the golden `position_ids` for the
        captured 8-token window; it is `arange(S)` unsqueezed, so a longer window
        follows the same rule. Read rather than assumed, so a checkpoint that
        numbered positions differently would be caught here.
        """
        captured = _captured_tensor("decoder_layer", "kwargs", "position_ids")
        base = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        if captured is not None:
            n = int(captured.shape[-1])
            if not torch.equal(captured.flatten().long(), torch.arange(n, dtype=torch.long)):
                raise RuntimeError(
                    f"captured position_ids are not arange ({captured.flatten().tolist()}); "
                    "the trace seam would be assembling a different convention than the golden"
                )
        return base

    def prefill_trace_inputs(self):
        """ZERO-ARG. Exactly the value `prefill_trace_setup` takes."""
        ids = model_ref.encode_prompt(model_ref.DEFAULT_PROMPT)
        return {
            "input_ids": ids,
            "position_ids": self._captured_position_ids(int(ids.shape[-1])),
        }

    def decode_trace_inputs(self):
        """ZERO-ARG. Exactly the value `decode_trace_setup` takes."""
        return self.prefill_trace_inputs()

    def prefill_trace_items(self) -> int:
        """One traced prefill retires the whole pinned window, batch included."""
        return self.batch * self.trace_capacity["prefill"]

    def decode_trace_items(self) -> int:
        """A recurring step retires one token per batch element."""
        return self.batch

    # -------------------------------------------------------------- setup

    def prefill_trace_setup(self, inputs):
        """Pin the sequence axis to C and pre-upload everything shape-dependent."""
        capacity = int(self.trace_capacity["prefill"])
        ids = inputs["input_ids"]
        real_len = min(int(ids.shape[-1]), capacity)
        pad_id = int(getattr(self.hf_model.generation_config, "pad_token_id", 0) or 0)

        padded = torch.full((self.batch, capacity), pad_id, dtype=torch.long)
        padded[0, :real_len] = ids[0, :real_len]
        positions = torch.arange(capacity, dtype=torch.long).unsqueeze(0)

        # RoPE tables come from the REFERENCE's own rotary_emb, so the traced shape
        # carries the golden's values. Masking is structural: the stack runs causally,
        # so a padded position cannot influence [0:real_len].
        cos, sin = model_ref.hf_rope_tables(self.hf_model, positions)
        cos_tt, sin_tt = self._upload_rope(cos, sin)

        self._trace_state["prefill"] = {
            "ids": self._upload_ids(padded),
            "cos": cos_tt,
            "sin": sin_tt,
            "capacity": capacity,
            "real_len": real_len,
        }
        return self._trace_state["prefill"]

    def decode_trace_setup(self, inputs):
        """Seed the resident self-attn KV, then pin the step's single position."""
        capacity = int(self.trace_capacity["decode"])
        if capacity != self.kv_capacity:
            self.kv_caches = self.encoder_stack.allocate_kv_caches(self.batch, capacity)
            self.kv_capacity = capacity

        ids = inputs["input_ids"]
        real_len = int(ids.shape[-1])
        if real_len >= capacity:
            raise RuntimeError(f"decode capacity {capacity} leaves no room for a {real_len}-token prefix")
        padded, positions = self._pad_inputs(ids)

        logits = self.decode_prefill(
            self._upload_ids(padded), self._upload_positions(positions), int(padded.shape[-1]), real_len
        )
        self._next_id = ttnn.reshape(ttnn.argmax(logits, dim=-1, keepdim=True), (self.batch, 1))

        step_positions = torch.full((self.batch, 1), float(real_len)).long()
        cos, sin = model_ref.hf_rope_tables(self.hf_model, step_positions)
        cos_tt, sin_tt = self._upload_rope(cos, sin)

        self._trace_state["decode"] = {
            "cos": cos_tt,
            "sin": sin_tt,
            "capacity": capacity,
            "position": real_len,
        }
        return self._trace_state["decode"]

    # --------------------------------------------------------------- steps

    def prefill_trace_step(self):
        """ONE host-op-free prefill at the pinned shape, over persistent buffers."""
        st = self._trace_state["prefill"]
        hidden = self.token_embed(st["ids"])
        encoded = self.encoder_stack(
            hidden,
            position_embeddings=(st["cos"], st["sin"]),
            attention_mask=None,
            is_causal=True,
            mode="prefill",
        )
        return self._logits_at(encoded, st["real_len"] - 1, st["capacity"])

    def decode_trace_step(self):
        """ONE host-op-free token step. Reads the resident KV; never recomputes it,
        and never advances the cursor, so repeated execution is idempotent."""
        st = self._trace_state["decode"]
        hidden = self.token_embed(self._next_id)
        hidden = ttnn.reshape(hidden, (1, 1, self.batch, self.hidden_size))
        x = self.encoder_stack(
            hidden,
            position_embeddings=(st["cos"], st["sin"]),
            kv_caches=self.kv_caches,
            cur_pos=self._cur_pos,
            mode="decode",
        )
        logits = self.decoder_head(x)
        ttnn.argmax(logits, dim=-1, keepdim=True)  # sampling stays on device, inside the step
        return logits

    # ===================== self-checks the harness calls ====================

    def trace_capture_selftest(self, device, pcc_target: float = 0.99, verbose: bool = True):
        """Capture, execute and release ONE trace per stage, in isolation."""
        ok_all = True
        report = {}
        for stage in PIPELINE_STAGES:
            setup = getattr(self, f"{stage}_trace_setup")
            step = getattr(self, f"{stage}_trace_step")
            inputs = getattr(self, f"{stage}_trace_inputs")()

            while True:
                setup(inputs)
                reference = self._to_torch(step(), rows=1).float()
                trace_id = None
                try:
                    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                    captured = step()
                    ttnn.end_trace_capture(device, trace_id, cq_id=0)
                except Exception as exc:  # noqa: BLE001 - region overflow is recoverable
                    if trace_id is not None:
                        with contextlib.suppress(Exception):
                            ttnn.release_trace(device, trace_id)
                    shrunk = self.trace_capacity[stage] // 2
                    floor = ttnn.TILE_SIZE if stage == "prefill" else 2 * ttnn.TILE_SIZE
                    if shrunk < floor:
                        print(
                            f"[trace] stage={stage} could not capture at any C: {type(exc).__name__}: {exc}", flush=True
                        )
                        report[stage] = {"captured": False, "reason": str(exc)}
                        ok_all = False
                        break
                    print(
                        f"[trace] stage={stage} FALLBACK: trace region overflowed at "
                        f"C={self.trace_capacity[stage]}, shrinking to C={shrunk} "
                        f"({type(exc).__name__})",
                        flush=True,
                    )
                    self.trace_capacity[stage] = shrunk
                    continue

                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                got = self._to_torch(captured, rows=1).float()
                value = model_ref.pcc(reference, got)
                ttnn.release_trace(device, trace_id)  # stage traces must not co-reside
                stage_ok = value >= pcc_target
                ok_all = ok_all and stage_ok
                report[stage] = {
                    "captured": True,
                    "C": self.trace_capacity[stage],
                    "pcc": value,
                    "items": getattr(self, f"{stage}_trace_items")(),
                }
                if verbose:
                    print(
                        f"[trace] stage={stage} C={self.trace_capacity[stage]} "
                        f"items={report[stage]['items']} trace_pcc={value}",
                        flush=True,
                    )
                break

        self.trace_report = report
        return ok_all

    def host_op_selftest(self, steps: int = 2, prompt: Optional[str] = None):
        """The authoritative fully-on-device check.

        Tokenization, the position tensors and the one-time weight build all happen
        OUTSIDE the observed region; everything from encoded inputs to task output --
        the prefix embedding included -- happens inside it. A ttnn op never dispatches
        through torch, so a truly on-device forward fires zero host aten ops.
        """
        try:
            from scripts.tt_hw_planner import host_op_observer
        except ImportError as exc:  # pragma: no cover - bring-up-only hook
            raise RuntimeError(
                "host_op_selftest() needs scripts/tt_hw_planner, which ships "
                "with the bring-up tool rather than with this model. It is a "
                "bring-up verification hook, not part of the model path."
            ) from exc

        ids = model_ref.encode_prompt(prompt or model_ref.DEFAULT_PROMPT)
        real_len = int(ids.shape[-1])
        padded, positions = self._pad_inputs(ids)
        padded_len = int(padded.shape[-1])
        ids_tt = self._upload_ids(padded)
        pos_tt = self._upload_positions(positions)
        self._set_position(real_len)

        all_ops: List[str] = []
        per_task = {}

        # Call 2 -- text -> prompt embedding.
        with host_op_observer.observe_host_ops() as ops:
            self.run_prefill(ids_tt, pos_tt)
        per_task["prompt_encoding"] = host_op_observer.verdict(list(ops))
        all_ops.extend(ops)

        # Call 1 -- text -> text, prefill + real decode steps, sampling on device.
        with host_op_observer.observe_host_ops() as ops:
            hidden = self.run_prefill(ids_tt, pos_tt, seed_kv=True)
            logits = self._logits_at(hidden, real_len - 1, padded_len)
            for _ in range(steps):
                self._next_id = ttnn.reshape(ttnn.argmax(logits, dim=-1, keepdim=True), (self.batch, 1))
                logits = self.decode_step()
        per_task["text_generation"] = host_op_observer.verdict(list(ops))
        all_ops.extend(ops)

        result = host_op_observer.verdict(all_ops)
        result["per_task"] = per_task
        return result


# ==========================================================================
#  Captured-golden access (Source B) -- used by the trace seams.
# ==========================================================================

_CAPTURED_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tt_transformers",
    "demo",
    "flux_2_klein_9b_text_encoder",
    "_captured",
)


def _captured_tensor(component: str, bundle: str, key=None):
    path = os.path.join(_CAPTURED_ROOT, component, f"{bundle}.pt")
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, weights_only=False)
    if key is None:
        return blob
    if isinstance(blob, dict):
        return blob.get(key)
    return None


# ==========================================================================
#  The single build surface.
# ==========================================================================


def build_pipeline(
    device,
    model=None,
    layers: Optional[int] = None,
    prefill_layers: Optional[int] = None,
    decode_layers: Optional[int] = None,
    **kwargs,
):
    """CONSTRUCT and RETURN the resident pipeline object -- never run it.

    This is the one entry the demos, the e2e tests and the perf harness all use, so
    there is a single build surface. Demo kwargs (`prompt`, `text`, `language`, ...)
    are accepted and ignored: the resident build takes its shapes from the config,
    not from a prompt.

    `layers` caps the depth of the repeated stack (`model.layers`); None means every
    layer. `prefill_layers` / `decode_layers` are the per-stage overrides named after
    the entries of PIPELINE_STAGES -- this model has one stack that both stages run,
    so they must agree with each other.
    """
    if layers is None and os.environ.get("TT_PERF_LAYERS"):
        layers = int(os.environ["TT_PERF_LAYERS"])
    kv_capacity = kwargs.pop("kv_capacity", None)
    batch = kwargs.pop("batch", 1)
    hf_model = model if model is not None else model_ref.load_hf_model(torch.float32)
    return Flux2Klein9BTextEncoderPipeline(
        device,
        hf_model,
        layers=layers,
        prefill_layers=prefill_layers,
        decode_layers=decode_layers,
        kv_capacity=kv_capacity,
        batch=batch,
    )


# ==========================================================================
#  Standalone (no-pytest) self-tests.
#
#  The host-op observer and the trace probe import this module in a bare
#  subprocess and call these ZERO-ARG entry points. They are thin: they borrow a
#  mesh from `../selftest.py` (the only device owner in this demo outside the
#  pytest fixture and the demo entrypoints -- `tt/` itself never opens one),
#  build the pipeline on it exactly as any caller would, and hand off to the
#  methods the pytest gates already run.
# ==========================================================================


def _selftest_pipeline(device, layers=None):
    from ..selftest import SELFTEST_LAYERS

    return build_pipeline(device, layers=SELFTEST_LAYERS if layers is None else layers)


def host_op_selftest(steps: int = 2, layers=None, device=None):
    """Zero-arg: prove the forward fires no host aten ops. See the method."""
    if device is not None:
        return _selftest_pipeline(device, layers).host_op_selftest(steps=steps)

    from ..selftest import own_a_mesh

    with own_a_mesh() as dev:
        return _selftest_pipeline(dev, layers).host_op_selftest(steps=steps)


def trace_capture_selftest(layers=None, device=None):
    """Zero-arg: capture / execute / release a real trace per stage. See the method."""
    if device is not None:
        return _selftest_pipeline(device, layers).trace_capture_selftest(device)

    from ..selftest import own_a_mesh

    with own_a_mesh() as dev:
        return _selftest_pipeline(dev, layers).trace_capture_selftest(dev)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(f"host_op_selftest : {host_op_selftest()}")
    print(f"trace_capture    : {trace_capture_selftest()}")
