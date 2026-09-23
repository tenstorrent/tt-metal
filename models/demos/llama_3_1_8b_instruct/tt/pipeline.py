# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE shared chained TTNN pipeline for `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`.

`demo/demo_text_generation.py` and `tests/e2e/test_e2e_pipeline.py` BOTH import and
call `run_text_generation()` from this module, so a green test is a working demo:
there is exactly one copy of the wiring.

Task head (Call 1) — text -> text, causal LM
--------------------------------------------
`config.json` declares `architectures: ["LlamaForCausalLM"]`, `is_encoder_decoder`
is absent/false and there are no sub-configs, so the model has ONE task head and
the pipeline has two stages: `PIPELINE_STAGES = ["prefill", "decode"]`.

The chain (every stage is fed the previous TT stage's REAL output — no reference
tensor is ever injected at a joint):

    input_ids (HF tokenizer)
      -> ttnn.embedding                       [authored here; no graduated stub exists]
      -> rotary_embedding      GRADUATED      -> (cos, sin) on device
      -> decoder_layer x 32    GRADUATED      -> each invoking
             attention         GRADUATED      (TP=4: ShardTensorToMesh + all_reduce)
             m_l_p             GRADUATED      (TP=4: ShardTensorToMesh + all_reduce)
      -> r_m_s_norm            GRADUATED      (the model's final norm)
      -> ttnn.linear lm_head                  [authored here]
      -> ttnn.argmax                          greedy sample, ON DEVICE
      -> the sampled id feeds straight back into ttnn.embedding (no host round-trip)

All five graduated modules from `bringup_status.json` are on that path.

Placement — TP=4 x DP=1 on a 1x4 Blackhole mesh
-----------------------------------------------
Attention, MLP and the decoder layer keep the sharding they graduated with
(`ShardTensorToMesh` + `ttnn.experimental.all_reduce_async` on cluster axis 1).
Embeddings, norms, RoPE tables and the lm_head are replicated — that is their own
graduated scheme (a per-element scale or a lookup table has no axis to split on),
not a downgrade of a shardable one.

Memory: the resident bf16 weights are ~5.7 GB per chip (embed 1.05 + lm_head 1.05 +
32 x 218M params / TP4 = 3.5), against the REGISTERED 28.6 GB usable per chip once a
CCL axis is in play. That figure is an estimate from parameter counts, not a
measurement by this run. The full 32-layer model is therefore resident; `layers`
exists only to make profiling cheap.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch

import ttnn
from models.demos.llama_3_1_8b_instruct._stubs.decoder_layer import TtLlamaDecoderLayer
from models.demos.llama_3_1_8b_instruct._stubs.r_m_s_norm import TtLlamaRMSNorm
from models.demos.llama_3_1_8b_instruct._stubs.rotary_embedding import TtLlamaRotaryEmbedding
from models.demos.llama_3_1_8b_instruct.tt import _invocation

HF_MODEL_ID = "/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct"

DEMO_DIR = Path(__file__).resolve().parents[1]
CAPTURED_DIR = DEMO_DIR / "_captured"
GOLDEN_DIR = CAPTURED_DIR / "_e2e_golden"

PIPELINE_STAGES = ["prefill", "decode"]

# Every module listed in bringup_status.json that carries a
# `_stubs/<name>.py.last_good_{native,sharded}` snapshot. The e2e test re-derives
# this set from disk and asserts the pipeline routed exactly it (nothing wasted).
GRADUATED_MODULES = ("attention", "decoder_layer", "m_l_p", "r_m_s_norm", "rotary_embedding")

# Sequence-axis capacity. The config bound is max_position_embeddings = 131072; a
# run pins it to a fixed, far smaller C so shapes are static (trace-capturable) and
# the KV cache is a fixed allocation.
DEFAULT_MAX_SEQ_LEN = 512
TRACE_PREFILL_CAPACITY = 128
TRACE_REGION_SIZE = 90 * 1024 * 1024

# On-device gate horizon. Decoding is STOP-TOKEN driven (config eos_token_id); this
# is the safety cap that bounds a non-terminating run, and it is handed IDENTICALLY
# to the HF golden so both sides are compared over the same length.
GATE_MAX_NEW_TOKENS = 40

DEFAULT_PROMPT = "What is the capital of France? Answer in one short sentence."

TILE = 32


# ---------------------------------------------------------------------------
# HF setup helpers (SOURCE A) — setup / reference only, never the forward path
# ---------------------------------------------------------------------------
def load_tokenizer(model_id: str = HF_MODEL_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def load_hf_model(model_id: str = HF_MODEL_ID, dtype=torch.float32):
    """The HF reference. Used for (a) weight extraction at build time and
    (b) computing the golden inside `hf_reference_text_generation`. It is never
    called from the TT forward path."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    return model


def encode_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Real HF input construction: the instruct chat template."""
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):  # transformers>=5 returns a BatchEncoding
        ids = ids["input_ids"]
    return torch.as_tensor(ids).to(torch.int64).reshape(1, -1)


def eos_token_ids(hf_model) -> list[int]:
    raw = getattr(hf_model.generation_config, "eos_token_id", None)
    if raw is None:
        raw = getattr(hf_model.config, "eos_token_id", None)
    if raw is None:
        return []
    return [int(raw)] if isinstance(raw, int) else [int(x) for x in raw]


# ---------------------------------------------------------------------------
# Small ttnn helpers
# ---------------------------------------------------------------------------
def _num_devices(device) -> int:
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _to_device(device, host, dtype, layout):
    kwargs = dict(dtype=dtype, layout=layout, device=device)
    if _num_devices(device) > 1:
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(host, **kwargs)


def _first_shard(device, tensor) -> torch.Tensor:
    """Bring a REPLICATED device tensor back to host, taking chip 0's copy."""
    n = _num_devices(device)
    if n > 1:
        full = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        return full[: full.shape[0] // n]
    return ttnn.to_torch(tensor)


def _pad_to_tile(n: int) -> int:
    return int(math.ceil(n / TILE) * TILE)


# ---------------------------------------------------------------------------
# The pipeline object
# ---------------------------------------------------------------------------
class LlamaTTPipeline:
    """Resident TTNN Llama-3.1-8B. Built once, then prefilled/decoded many times.

    The repeated block is held as a plain Python list of same-typed
    `TtLlamaDecoderLayer` objects (`self.layers`) so tooling can discover, size and
    cap the stack, and the HF reference stays reachable as `self.hf` — it is the
    ground truth for how many sections the model has and how deep each is.
    """

    def __init__(
        self,
        mesh_device,
        hf_model,
        tokenizer=None,
        layers=None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        batch: int = 1,
    ):
        self.mesh_device = mesh_device
        self.hf = hf_model  # reference: structure ground truth + golden helper
        self.tokenizer = tokenizer
        self.config = hf_model.config
        self.num_devices = _num_devices(mesh_device)
        self.batch = batch
        self.max_seq_len = int(max_seq_len)
        self.hidden_size = int(self.config.hidden_size)
        self.vocab_size = int(self.config.vocab_size)

        body = hf_model.model
        total_layers = len(body.layers)
        n_layers = total_layers if layers is None else max(1, min(int(layers), total_layers))
        self.n_layers = n_layers
        self.total_layers = total_layers

        # --- Embedding (replicated; ROW_MAJOR is what ttnn.embedding wants) ---
        emb = body.embed_tokens.weight.detach().to(torch.bfloat16).contiguous()
        self.embed_weight = _to_device(mesh_device, emb, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

        # --- GRADUATED: rotary embedding (replicated table) ---
        self.rope = TtLlamaRotaryEmbedding(mesh_device, body.rotary_emb)

        # --- GRADUATED: the decoder stack (each layer builds the graduated
        #     attention + m_l_p internally, sharded across TP=4) ---
        self.layers = [TtLlamaDecoderLayer(mesh_device, body.layers[i]) for i in range(n_layers)]

        # --- GRADUATED: the model's final RMSNorm, feeding the lm_head ---
        self.norm = TtLlamaRMSNorm(mesh_device, body.norm)

        # --- lm_head (replicated) ---
        lm_w = hf_model.lm_head.weight.detach().to(torch.float32).transpose(0, 1).contiguous().to(torch.bfloat16)
        self.lm_head_weight = _to_device(mesh_device, lm_w, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        self.compute_kernel_config = self.layers[0].compute_kernel_config if self.layers else None

        # --- Resident KV cache: capacity C on the sequence axis ---
        for layer in self.layers:
            layer.allocate_kv_cache(self.max_seq_len, batch)

        # Position tensors are pure setup constants; they are cached so the decode
        # loop itself performs no host->device marshalling (host_op_selftest).
        self._pos_f32: dict[tuple, ttnn.Tensor] = {}
        self._pos_i32: dict[int, ttnn.Tensor] = {}
        self._trace = {}

        self.tp = self.layers[0].tp if self.layers else 1
        self.sharded_weight_count = sum(
            1 for layer in self.layers for _ in range(5) if layer.tp > 1
        )  # wqkv, wo, gate, up, down per layer
        self.collectives_per_layer = 2 if self.tp > 1 else 0

    # ------------------------------------------------------------------
    # Structure / introspection
    # ------------------------------------------------------------------
    @property
    def stacks(self) -> dict:
        """The model's repeated blocks. Llama-3.1 has exactly ONE (the text decoder);
        both stages own it. The HF reference agrees: len(hf.model.layers)."""
        return {"decoder": self.layers}

    def describe(self) -> dict:
        return {
            "stages": list(PIPELINE_STAGES),
            "layers_built": self.n_layers,
            "layers_total": self.total_layers,
            "tp": self.tp,
            "mesh": list(getattr(self.mesh_device, "shape", [1, self.num_devices])),
            "max_seq_len": self.max_seq_len,
            "graduated_modules": list(GRADUATED_MODULES),
        }

    # ------------------------------------------------------------------
    # Setup constants (created OUTSIDE any observed/traced forward)
    # ------------------------------------------------------------------
    def positions_f32(self, start: int, length: int) -> ttnn.Tensor:
        """position_ids as a float32 TILE tensor — the ttnn form the graduated
        rotary_embedding stub consumes without any host marshalling."""
        key = (start, length)
        t = self._pos_f32.get(key)
        if t is None:
            host = torch.arange(start, start + length, dtype=torch.float32).reshape(1, length)
            t = _to_device(self.mesh_device, host, ttnn.float32, ttnn.TILE_LAYOUT)
            self._pos_f32[key] = t
        return t

    def cur_pos(self, pos: int) -> ttnn.Tensor:
        """The int32 [batch] cache slot tensor the decode ops index with."""
        t = self._pos_i32.get(pos)
        if t is None:
            host = torch.full((self.batch,), int(pos), dtype=torch.int32)
            t = _to_device(self.mesh_device, host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self._pos_i32[pos] = t
        return t

    def ids_to_device(self, ids: torch.Tensor) -> ttnn.Tensor:
        return _to_device(self.mesh_device, ids.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def warmup_constants(self, prefill_len: int, steps: int) -> None:
        """Pre-create every position constant a run of this shape will ask for, so
        the forward itself is free of host->device marshalling."""
        self.positions_f32(0, prefill_len)
        for i in range(prefill_len, prefill_len + steps + 1):
            self.positions_f32(i, 1)
            self.cur_pos(i)

    # ------------------------------------------------------------------
    # THE CHAIN — stage 1: prefill
    # ------------------------------------------------------------------
    def prefill(self, ids_tt: ttnn.Tensor, seq_len: int, last_index: int, return_all: bool = False) -> ttnn.Tensor:
        """Run the padded prompt through the whole stack, seed the resident KV cache,
        and return logits: position `last_index` only, or -- with `return_all` -- every
        position, which is what the prefill-parity check compares to HF.  Pure ttnn."""
        h = ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        h = ttnn.reshape(h, (self.batch, 1, seq_len, self.hidden_size))

        # GRADUATED rotary_embedding -> the (cos, sin) every layer below consumes
        cos, sin = self.rope(h, position_ids=self.positions_f32(0, seq_len))

        for layer in self.layers:  # GRADUATED decoder_layer (-> attention, m_l_p)
            h = layer(h, position_embeddings=(cos, sin), mode="prefill")

        h = self.norm(h)  # GRADUATED r_m_s_norm — the model's final norm
        if return_all:
            logits = ttnn.linear(
                h,
                self.lm_head_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(h)
            return logits
        h_last = ttnn.slice(h, [0, 0, last_index, 0], [self.batch, 1, last_index + 1, self.hidden_size])
        ttnn.deallocate(h)
        logits = ttnn.linear(
            h_last,
            self.lm_head_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h_last)
        return logits

    # AR decode contract: seeding the resident KV is exactly the prefill above.
    decode_prefill = prefill

    # ------------------------------------------------------------------
    # THE CHAIN — stage 2: decode (one token, reads the resident KV)
    # ------------------------------------------------------------------
    def decode_step(self, token_tt: ttnn.Tensor, pos: int) -> ttnn.Tensor:
        """One autoregressive step. Never recomputes the prefix: every layer reads
        its resident KV cache and appends this token to it.  Pure ttnn."""
        h = ttnn.embedding(token_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        h = ttnn.reshape(h, (1, 1, self.batch, self.hidden_size))

        cos, sin = self.rope(h, position_ids=self.positions_f32(pos, 1))  # GRADUATED
        cur = self.cur_pos(pos)

        for layer in self.layers:  # GRADUATED decoder_layer (-> attention, m_l_p)
            h = layer(h, position_embeddings=(cos, sin), mode="decode", cur_pos=cur)

        h = self.norm(h)  # GRADUATED r_m_s_norm
        logits = ttnn.linear(
            h,
            self.lm_head_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h)
        return logits

    @staticmethod
    def sample(logits: ttnn.Tensor) -> ttnn.Tensor:
        """Greedy sampling ON DEVICE. Returns the id shaped [1, 1] so it can be fed
        straight back into ttnn.embedding without a host round-trip."""
        tok = ttnn.argmax(logits, dim=-1, keepdim=True)
        return ttnn.reshape(tok, (1, 1))

    # ------------------------------------------------------------------
    # Command 3 — per-stage trace contract
    # ------------------------------------------------------------------
    def prefill_trace_inputs(self) -> dict:
        """ZERO-ARG. The exact argument `prefill_trace_setup` takes, assembled from
        the captured reference inputs the e2e test / demo use."""
        return {"input_ids": _load_or_make_reference_input_ids(self.tokenizer)}

    decode_trace_inputs = prefill_trace_inputs

    def prefill_trace_items(self) -> int:
        """One traced prefill retires C token positions."""
        return int(self._trace.get("prefill_capacity", TRACE_PREFILL_CAPACITY))

    def decode_trace_items(self) -> int:
        """One traced decode step retires one token (batch = 1)."""
        return int(self.batch)

    def _hf_rope(self, seq_len: int, start: int = 0):
        """cos/sin taken FROM THE HF REFERENCE itself, so the pinned trace constants
        match the golden exactly."""
        body = self.hf.model
        pos = torch.arange(start, start + seq_len, dtype=torch.int64).reshape(1, seq_len)
        dummy = torch.zeros(1, seq_len, self.hidden_size, dtype=next(body.parameters()).dtype)
        with torch.no_grad():
            cos, sin = body.rotary_emb(dummy, pos)
        return cos.to(torch.float32), sin.to(torch.float32)

    def prefill_trace_setup(self, inputs: dict, capacity: int | None = None) -> None:
        """Pin the sequence axis to a fixed capacity C and pre-upload the padded
        input plus every shape-dependent constant into PERSISTENT device buffers."""
        capacity = int(capacity or TRACE_PREFILL_CAPACITY)
        capacity = min(_pad_to_tile(capacity), self.max_seq_len)
        ids = inputs["input_ids"].reshape(1, -1)
        real_len = min(int(ids.shape[-1]), capacity)
        padded = torch.zeros(1, capacity, dtype=torch.int64)
        padded[0, :real_len] = ids[0, :real_len]

        cos, sin = self._hf_rope(capacity)
        self._trace["prefill_capacity"] = capacity
        self._trace["prefill_real_len"] = real_len
        self._trace["prefill_ids"] = self.ids_to_device(padded)
        self._trace["prefill_cos"] = _to_device(self.mesh_device, cos, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._trace["prefill_sin"] = _to_device(self.mesh_device, sin, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Causal masking is the SDPA kernel's own is_causal flag, so padded positions
        # past real_len can never influence output[0:real_len].

    def prefill_trace_step(self) -> ttnn.Tensor:
        """ONE host-op-free prefill at the pinned shape, reading only persistent buffers."""
        capacity = self._trace["prefill_capacity"]
        h = ttnn.embedding(self._trace["prefill_ids"], self.embed_weight, layout=ttnn.TILE_LAYOUT)
        h = ttnn.reshape(h, (self.batch, 1, capacity, self.hidden_size))
        cos, sin = self._trace["prefill_cos"], self._trace["prefill_sin"]
        for layer in self.layers:
            h = layer(h, position_embeddings=(cos, sin), mode="prefill")
        h = self.norm(h)
        last = self._trace["prefill_real_len"] - 1
        h_last = ttnn.slice(h, [0, 0, last, 0], [self.batch, 1, last + 1, self.hidden_size])
        ttnn.deallocate(h)
        logits = ttnn.linear(
            h_last,
            self.lm_head_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h_last)
        return logits

    def decode_trace_setup(self, inputs: dict, capacity: int | None = None) -> None:
        """Seed the resident KV from the real prompt, then pin the decode step's
        constants (token buffer, RoPE row, cache slot) as persistent buffers."""
        if "prefill_ids" not in self._trace:
            self.prefill_trace_setup(inputs, capacity)
        # Seed the resident self-attention KV once, OUTSIDE the trace.
        logits = self.prefill_trace_step()
        tok = self.sample(logits)
        ttnn.deallocate(logits)
        pos = self._trace["prefill_real_len"]
        cos, sin = self._hf_rope(1, start=pos)
        self._trace["decode_pos"] = pos
        self._trace["decode_token"] = tok
        self._trace["decode_cos"] = _to_device(self.mesh_device, cos, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._trace["decode_sin"] = _to_device(self.mesh_device, sin, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._trace["decode_cur"] = self.cur_pos(pos)

    def decode_trace_step(self) -> ttnn.Tensor:
        """ONE host-op-free decode step reading only persistent buffers."""
        h = ttnn.embedding(self._trace["decode_token"], self.embed_weight, layout=ttnn.TILE_LAYOUT)
        h = ttnn.reshape(h, (1, 1, self.batch, self.hidden_size))
        cos, sin = self._trace["decode_cos"], self._trace["decode_sin"]
        cur = self._trace["decode_cur"]
        for layer in self.layers:
            h = layer(h, position_embeddings=(cos, sin), mode="decode", cur_pos=cur)
        h = self.norm(h)
        logits = ttnn.linear(
            h,
            self.lm_head_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h)
        return logits


# ---------------------------------------------------------------------------
# Module-level factory — the SINGLE build surface
# ---------------------------------------------------------------------------
def build_pipeline(
    device,
    model=None,
    layers=None,
    prefill_layers=None,
    decode_layers=None,
    tokenizer=None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    batch: int = 1,
    **kwargs,
) -> LlamaTTPipeline:
    """CONSTRUCT AND RETURN the resident pipeline object (it does not run it).

    `layers` caps the depth of every repeated block; None means every layer (never
    0). Llama-3.1-8B has exactly ONE repeated stack — the 32-layer text decoder —
    and both stages own it, so `prefill_layers` / `decode_layers` are accepted for
    call-signature compatibility and must agree with each other. Everything else
    (embedding, RoPE, final norm, lm_head) is always built, so a capped build still
    exercises every distinct op the full model runs, just fewer times.

    Extra demo kwargs (prompt, max_new_tokens, ...) are accepted and ignored: the
    resident build takes its shapes from the config, not from a prompt.
    """
    per_stage = [v for v in (prefill_layers, decode_layers) if v is not None]
    if per_stage:
        if len(set(per_stage)) > 1:
            raise ValueError(
                "prefill_layers and decode_layers address the SAME decoder stack in this "
                f"model and must agree (got prefill={prefill_layers}, decode={decode_layers})"
            )
        layers = per_stage[0]
    if layers is not None and int(layers) <= 0:
        raise ValueError("layers must be >= 1 (None means every layer); a 0-layer model cannot run")

    env_layers = os.environ.get("TT_PERF_LAYERS")
    if layers is None and env_layers:
        layers = int(env_layers)

    if model is None:
        model = load_hf_model()
    if tokenizer is None:
        try:
            tokenizer = load_tokenizer()
        except Exception:  # noqa: BLE001 — a tokenizer is only needed for text I/O
            tokenizer = None
    return LlamaTTPipeline(device, model, tokenizer=tokenizer, layers=layers, max_seq_len=max_seq_len, batch=batch)


# ---------------------------------------------------------------------------
# THE SHARED TASK ENTRY — demo/ and tests/e2e/ both call THIS
# ---------------------------------------------------------------------------
def run_text_generation(
    pipe: LlamaTTPipeline,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = GATE_MAX_NEW_TOKENS,
    input_ids: torch.Tensor | None = None,
    check_stop: bool = True,
    collect_logits: bool = True,
    verbose: bool = False,
) -> dict:
    """Call 1: text -> text.

    Explicit chain over the graduated stubs; no HF orchestration, no torch compute.
    Decoding is STOP-TOKEN driven (config `eos_token_id`) with `max_new_tokens` as
    the safety cap — the same rule and the same cap the HF golden is given.
    """
    if input_ids is None:
        assert pipe.tokenizer is not None, "a tokenizer is required to encode a prompt"
        input_ids = encode_prompt(pipe.tokenizer, prompt)
    input_ids = input_ids.reshape(1, -1)
    real_len = int(input_ids.shape[-1])
    padded_len = _pad_to_tile(real_len)
    assert padded_len + max_new_tokens <= pipe.max_seq_len, (
        f"prompt {real_len} (padded {padded_len}) + {max_new_tokens} new tokens exceeds "
        f"the pinned sequence capacity {pipe.max_seq_len}"
    )

    pipe.warmup_constants(real_len, max_new_tokens)

    padded = torch.zeros(1, padded_len, dtype=torch.int64)
    padded[0, :real_len] = input_ids[0]
    ids_tt = pipe.ids_to_device(padded)

    stops = set(eos_token_ids(pipe.hf))
    new_ids: list[int] = []
    step_logits: list[torch.Tensor] = []

    # --- stage 1: prefill (also seeds the resident KV cache) ---
    logits = pipe.prefill(ids_tt, padded_len, real_len - 1)
    ttnn.deallocate(ids_tt)
    token = pipe.sample(logits)

    for step in range(max_new_tokens):
        if collect_logits:
            step_logits.append(_first_shard(pipe.mesh_device, logits).reshape(-1)[: pipe.vocab_size].float())
        ttnn.deallocate(logits)
        tid = int(_first_shard(pipe.mesh_device, token).reshape(-1)[0])
        new_ids.append(tid)
        if verbose and pipe.tokenizer is not None and tid not in stops:
            print(pipe.tokenizer.decode([tid]), end="", flush=True)
        if (check_stop and tid in stops) or step == max_new_tokens - 1:
            break
        # --- stage 2: decode — the sampled id feeds the next step directly ---
        logits = pipe.decode_step(token, real_len + step)
        ttnn.deallocate(token)
        token = pipe.sample(logits)
    ttnn.deallocate(token)

    text = pipe.tokenizer.decode(new_ids, skip_special_tokens=True) if pipe.tokenizer else ""
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "new_ids": new_ids,
        "text": text,
        "step_logits": torch.stack(step_logits) if step_logits else torch.empty(0),
        "invoked": _invocation.snapshot(),
    }


# ---------------------------------------------------------------------------
# HF golden (SOURCE A) — reference only, kept out of the TT forward path
# ---------------------------------------------------------------------------
def hf_reference_text_generation(
    hf_model,
    tokenizer,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = GATE_MAX_NEW_TOKENS,
    input_ids: torch.Tensor | None = None,
    cache: bool = True,
) -> dict:
    """`model.generate()` — the golden. Greedy so it is deterministic, and given the
    SAME eos set and the SAME cap as the TT side."""
    if input_ids is None:
        input_ids = encode_prompt(tokenizer, prompt)
    input_ids = input_ids.reshape(1, -1)

    key = f"{abs(hash((prompt, int(max_new_tokens), tuple(input_ids[0].tolist()))))}"
    path = GOLDEN_DIR / f"golden_{key}.pt"
    if cache and path.exists():
        blob = torch.load(path, weights_only=False)
        return blob

    with torch.no_grad():
        out = hf_model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=eos_token_ids(hf_model)[0],
        )
    new_ids = out.sequences[0, input_ids.shape[-1] :].tolist()
    step_logits = torch.stack([s[0].float() for s in out.scores])
    blob = {
        "prompt": prompt,
        "input_ids": input_ids,
        "new_ids": new_ids,
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "step_logits": step_logits,
    }
    if cache:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)
        torch.save(input_ids, GOLDEN_DIR / "input_ids.pt")
    return blob


def _load_or_make_reference_input_ids(tokenizer=None) -> torch.Tensor:
    """The captured golden input the e2e test and demo run on."""
    path = GOLDEN_DIR / "input_ids.pt"
    if path.exists():
        return torch.load(path, weights_only=False).reshape(1, -1)
    tok = tokenizer or load_tokenizer()
    ids = encode_prompt(tok, DEFAULT_PROMPT)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ids, path)
    return ids


# ---------------------------------------------------------------------------
# Command 3 — selftests
# ---------------------------------------------------------------------------
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(a.float(), b.float(), 0.0)
    try:
        return float(pcc)
    except (TypeError, ValueError):
        return float(str(pcc).split()[-1])


def trace_capture_selftest(device=None, pipe: LlamaTTPipeline | None = None, capacity: int | None = None) -> bool:
    """For EACH stage in PIPELINE_STAGES: capture ONE step, execute the trace, PCC it
    against the eager reference, then RELEASE the trace before the next stage.

    The device must be opened with `trace_region_size=TRACE_REGION_SIZE`. Callable
    with NO arguments (that is how the trace probe invokes it): this module never
    opens a device itself, so the standalone entry borrows the demo's own opener
    (`mesh_harness`, outside `tt/`) and closes it again on the way out.
    """
    owned_device = None
    if device is None:
        device = getattr(pipe, "mesh_device", None)
    if device is None:
        from models.demos.llama_3_1_8b_instruct.mesh_harness import open_mesh

        device = owned_device = open_mesh(trace_region_size=TRACE_REGION_SIZE)
    try:
        return _trace_capture_stages(device, pipe, capacity)
    finally:
        if owned_device is not None:
            from models.demos.llama_3_1_8b_instruct.mesh_harness import close_mesh

            close_mesh(owned_device)


def _trace_capture_stages(device, pipe: LlamaTTPipeline | None, capacity: int | None) -> bool:
    """The per-stage capture/replay/PCC loop of `trace_capture_selftest`, split out so
    device ownership and the measurement stay separate concerns."""
    if pipe is None:
        layers = int(os.environ["TT_TRACE_LAYERS"]) if os.environ.get("TT_TRACE_LAYERS") else None
        pipe = build_pipeline(device, layers=layers)
    capacity = int(capacity or TRACE_PREFILL_CAPACITY)
    ok = True
    for stage in PIPELINE_STAGES:
        setup = getattr(pipe, f"{stage}_trace_setup")
        step = getattr(pipe, f"{stage}_trace_step")
        inputs = getattr(pipe, f"{stage}_trace_inputs")()
        items = getattr(pipe, f"{stage}_trace_items")()

        cap = capacity
        while True:
            setup(inputs, cap)
            reference = _first_shard(device, step()).float()  # eager, outside the trace
            try:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = step()
                ttnn.end_trace_capture(device, tid, cq_id=0)
            except Exception as exc:  # noqa: BLE001
                if cap <= TILE:
                    print(f"[trace] stage={stage} FAILED to capture even at C={cap}: {exc}")
                    return False
                cap //= 2
                print(f"[trace] stage={stage} capture overflowed; FALLING BACK to C={cap}")
                continue
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = _first_shard(device, out).float()
            pcc = _pcc(reference, got)
            ttnn.release_trace(device, tid)
            print(f"[trace] stage={stage} C={cap} items={items} host_op_free_capture=True PCC={pcc:.6f}")
            ok = ok and pcc >= 0.99
            break
    return ok


def host_op_selftest(pipe: LlamaTTPipeline | None = None, device=None, steps: int = 2) -> dict:
    """AUTHORITATIVE fully-on-device check.

    Input ENCODING (tokenize) and the one-time weight/constant build happen OUTSIDE
    the observed region; the model math — embedding, RoPE, all 32 layers, final
    norm, lm_head and sampling — happens INSIDE it. ttnn ops do not dispatch through
    torch, so a truly on-device forward fires ZERO host aten ops.

    Callable with NO arguments (that is how the observer probe invokes it). This
    module never opens a device — the pipeline always runs on the `device` passed
    into `build_pipeline`. When the caller supplies neither, the standalone entry
    borrows the demo's own opener (`mesh_harness`, outside `tt/`) and closes it
    again, so there is still exactly one device open at a time.
    """
    owned_device = None
    if pipe is None:
        if device is None:
            from models.demos.llama_3_1_8b_instruct.mesh_harness import close_mesh, open_mesh

            device = owned_device = open_mesh(trace_region_size=TRACE_REGION_SIZE)
        layers = int(os.environ["TT_HOSTOP_LAYERS"]) if os.environ.get("TT_HOSTOP_LAYERS") else None
        pipe = build_pipeline(device, layers=layers)
    try:
        return _host_op_selftest_observe(pipe, steps)
    finally:
        if owned_device is not None:
            close_mesh(owned_device)


def _host_op_selftest_observe(pipe: LlamaTTPipeline, steps: int) -> dict:
    """Observed region of `host_op_selftest`, split out so device ownership and the
    measurement stay separate concerns."""
    from scripts.tt_hw_planner import host_op_observer

    # --- OUTSIDE the observed region: encode + stage every constant ---
    ids = _load_or_make_reference_input_ids(pipe.tokenizer).reshape(1, -1)
    real_len = int(ids.shape[-1])
    padded_len = _pad_to_tile(real_len)
    padded = torch.zeros(1, padded_len, dtype=torch.int64)
    padded[0, :real_len] = ids[0]
    ids_tt = pipe.ids_to_device(padded)
    pipe.warmup_constants(real_len, steps + 1)

    def forward():
        logits = pipe.prefill(ids_tt, padded_len, real_len - 1)
        token = pipe.sample(logits)
        ttnn.deallocate(logits)
        for i in range(steps):
            logits = pipe.decode_step(token, real_len + i)
            ttnn.deallocate(token)
            token = pipe.sample(logits)
            ttnn.deallocate(logits)
        ttnn.deallocate(token)

    forward()  # warm the program cache outside the observed region
    with host_op_observer.observe_host_ops() as ops:
        forward()
    return host_op_observer.verdict(ops)
