# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""THE one shared end-to-end TTNN pipeline for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone` (`MistralForCausalLM`).

Both demos and both e2e tests import the chain from here — there is no second
copy of the wiring anywhere in the package, so a green test is a working demo.

WHAT THE CHECKPOINT IS
    Despite the `tts-backbone` name the checkpoint registers exactly ONE task
    head: causal language modelling (`architectures: [MistralForCausalLM]`,
    `is_encoder_decoder: False`, no audio/codec sub-config, no vocoder). So the
    pipeline has the two stages a decoder-only LM has — `prefill` and
    `decode` — and no vocode stage to fake.

THE EXPLICIT CHAIN (no HF orchestration, no `generate()`, no torch compute)
    prefill  (Call 2 `causal_lm_logits`, and the seed for Call 1):
        tokens_tt -> ttnn.embedding
                  -> TtRotaryEmbedding(position_ids)            [graduated]
                  -> 26 x TtDecoderLayer(mask, (cos,sin), KV)   [graduated ->
                     TtAttention, TtMLP, TtRMSNorm x2]
                  -> TtRMSNorm final norm                       [graduated]
                  -> ttnn.linear(lm_head)
    decode   (Call 1 `text_generation`), ONE resident-state step per token:
        the previous step's OWN ttnn.argmax token -> ttnn.embedding
                  -> TtRotaryEmbedding(resident position)        [graduated]
                  -> 26 x TtDecoderLayer(resident KV, resident cache index)
                  -> TtRMSNorm -> ttnn.linear -> ttnn.argmax
                  -> the token is written back into the resident [1,1] input
                     buffer and both position buffers advance ON DEVICE.

    Every one of the five graduated modules is inside that chain with its output
    feeding downstream compute; nothing is "touched" for a counter.

WHY IT IS TRACE-CAPTURABLE
    * every weight is staged ONCE in `__init__` and stays resident for the whole
      run (the KV buffers too),
    * a decode step reads ONLY resident device tensors: the [1,1] token buffer,
      the [1,1] rotary position, the [1] int32 cache index and the per-layer KV,
    * the token never round-trips through the host: `ttnn.argmax` feeds
      `ttnn.embedding` directly, and both position buffers advance with
      `ttnn.add` / `ttnn.copy` in place,
    * so shapes are constant every step and `decode_step` is host-op free.

HF USAGE IS SETUP/REFERENCE ONLY
    `load_hf_model` (weights + config reads), and `_hf_reference_logits` /
    `_hf_reference_generate`, which compute the PCC golden. The TT chain above
    never calls an HF module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_import_paths() -> Path:
    """Make `import ttnn` resolve to the built package, then return the repo root.

    tt-metal's editable install publishes three sys.path entries (repo root,
    `<repo>/ttnn`, `<repo>/tools`; see `setup.py::ttnn-custom.pth`). This
    checkout was built in place without that install, and with only the repo
    root on the path `import ttnn` finds the same-named SOURCE directory and
    yields an empty namespace package. Re-publishing the same three entries here
    keeps this module importable from a bare subprocess — which is how the
    planner's trace/host-op probes import it.
    """
    repo = Path(__file__).resolve().parents[4]
    for extra in (repo, repo / "ttnn", repo / "tools"):
        entry = str(extra)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    shadowed = sys.modules.get("ttnn")
    if shadowed is not None and not hasattr(shadowed, "argmax"):
        for name in [n for n in list(sys.modules) if n == "ttnn" or n.startswith("ttnn.")]:
            del sys.modules[name]
    return repo


REPO_ROOT = _bootstrap_import_paths()

import torch  # noqa: E402
import ttnn  # noqa: E402

from models.common.utility_functions import comp_pcc  # noqa: E402
from models.demos.voxtral_tts_backbone._stubs.decoder_layer import TtDecoderLayer  # noqa: E402
from models.demos.voxtral_tts_backbone._stubs.r_m_s_norm import TtRMSNorm  # noqa: E402
from models.demos.voxtral_tts_backbone._stubs.rotary_embedding import TtRotaryEmbedding  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[1]
HF_MODEL_ID = os.environ.get("E2E_MODEL_ID") or "/localdev/lserbedzija/hf_models/voxtral-tts-backbone"

#: The two stages a decoder-only causal LM has. Both own the same repeated stack.
PIPELINE_STAGES = ["prefill", "decode"]

TILE = 32
#: KV / trace capacity C. Fixed at build time so traced shapes never move.
DEFAULT_CAPACITY = 128
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."
#: Safety cap on the decode horizon. `generation_config` carries neither
#: `max_new_tokens` nor `max_length`, and `max_position_embeddings` is 128000 --
#: not a runnable bound -- so this value is chosen here for lack of any model
#: signal. It is a CAP, not the stop rule: `eos_token_id` is the stop rule and it
#: is applied identically to the TT chain and to the HF golden.
FALLBACK_MAX_NEW_TOKENS = 48
MASK_NEG = -1.0e9


# ---------------------------------------------------------------------------
# setup helpers (torch/HF allowed here: nothing below runs in the TT chain)
# ---------------------------------------------------------------------------
def load_hf_model(model_id: str = None, dtype=torch.bfloat16):
    """SETUP ONLY: the HF checkpoint we read weights/config from, and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or HF_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    return model, tokenizer


def _pad_to_tile(n: int) -> int:
    return ((int(n) + TILE - 1) // TILE) * TILE


def captured_inputs_path() -> Path:
    """Where the e2e test persists the pipeline-level golden prompt tokens."""
    return DEMO_DIR / "_captured" / "e2e_pipeline" / "input_ids.pt"


def persist_captured_inputs(prompt_tokens) -> Path:
    """Record the golden pipeline inputs so the zero-arg `*_trace_inputs()` hooks
    can rebuild them with no per-model knowledge."""
    path = captured_inputs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_tokens.detach().to(torch.int64).cpu(), path)
    return path


def _resolve_depth(layers, prefill_layers, decode_layers):
    """This model has ONE repeated stack shared by prefill and decode, so the
    per-stage overrides alias that stack's depth. Two conflicting non-None values
    are an error rather than a silent pick."""
    given = {"prefill_layers": prefill_layers, "decode_layers": decode_layers}
    named = {k: int(v) for k, v in given.items() if v is not None}
    if len(set(named.values())) > 1:
        raise ValueError(
            "prefill_layers/decode_layers disagree (%s) but this model has ONE repeated stack shared by "
            "both stages; pass a single depth via `layers=`" % named
        )
    if named:
        return next(iter(named.values()))
    return None if layers is None else int(layers)


def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs):
    """THE single build surface: returns the resident pipeline object.

    It builds and returns; it never runs generation. `layers` caps the depth of
    the one repeated stack (None = all 26, never 0); embeddings, both per-layer
    norms, the final norm and the LM head are always built, so a capped build
    still exercises every distinct op the full model runs. `prefill_layers` /
    `decode_layers` (the PIPELINE_STAGES-derived names) alias that same depth.
    Unknown kwargs (prompt, text, ...) are accepted and ignored so the demos can
    forward their own argv.
    """
    depth = _resolve_depth(layers, prefill_layers, decode_layers)
    tokenizer = kwargs.pop("tokenizer", None)
    model_id = kwargs.pop("model_id", None)
    capacity = int(kwargs.pop("capacity", DEFAULT_CAPACITY))
    if model is None:
        model, hf_tokenizer = load_hf_model(model_id)
        tokenizer = tokenizer or hf_tokenizer
    elif tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id or HF_MODEL_ID)
    return VoxtralTtsBackbonePipeline(device, model, tokenizer, depth=depth, capacity=capacity)


class VoxtralTtsBackbonePipeline:
    """The resident TT pipeline: PIPELINE_STAGES, the AR decode contract
    (`decode_prefill` / `decode_step`), the per-stage trace hooks and the two
    task-head entrypoints (`run_prefill_logits`, `run_generate`)."""

    PIPELINE_STAGES = PIPELINE_STAGES

    def __init__(self, device, hf_model, tokenizer=None, depth=None, capacity=DEFAULT_CAPACITY):
        self.device = device
        # The HF module is kept reachable as the REFERENCE (config reads, golden
        # helpers, and so a structural walk can find hf.model.layers as ground
        # truth). It is never called from the TT chain.
        self.hf = hf_model
        self.reference = hf_model
        self.tokenizer = tokenizer

        config = hf_model.config
        self.config = config
        self.generation_config = getattr(hf_model, "generation_config", None)
        self.hidden_size = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)
        self.n_heads = int(config.num_attention_heads)
        self.n_kv_heads = int(getattr(config, "num_key_value_heads", None) or self.n_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or self.hidden_size // self.n_heads)
        self.full_depth = int(config.num_hidden_layers)
        self.depth = self.full_depth if depth is None else max(1, min(int(depth), self.full_depth))
        self.capacity = min(int(capacity), int(getattr(config, "max_position_embeddings", capacity)))
        self.stop_token_ids = self._stop_token_ids()

        try:
            self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        except Exception:  # noqa: BLE001 - accuracy tuning is best-effort
            self.compute_kernel_config = None

        # The LM head gets its OWN compute config, matched to its bfloat4_b weight.
        # HiFi4 runs four math passes to preserve mantissa bits a bf4_b operand does not
        # carry, so on this one matmul it buys nothing and costs cycles; the guideline
        # policy for a bf8b/bf4b matmul is LoFi with fp32_dest_acc_en=False (which also
        # unlocks the larger subblocks) and packer_l1_acc=True. Scoped to the head alone:
        # the per-layer weights are still bf16 and keep the HiFi4 config above, and the
        # norms must never be walked down this far.
        try:
            self.lm_head_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        except Exception:  # noqa: BLE001 - accuracy tuning is best-effort
            self.lm_head_compute_kernel_config = self.compute_kernel_config

        # --- the repeated stack: a plain list of same-typed blocks ------------
        self.layers = [TtDecoderLayer.build(device, hf_model.model.layers[i]) for i in range(self.depth)]
        self.final_norm = TtRMSNorm.build(device, hf_model.model.norm)
        self.rotary = TtRotaryEmbedding.build(device, hf_model.model.rotary_emb)

        # --- glue weights (not graduated components, but still pure ttnn) ----
        embed_weight = hf_model.model.embed_tokens.weight
        self.w_embed = ttnn.from_torch(
            embed_weight.detach().to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        # tie_word_embeddings=True, so this is the embedding matrix transposed.
        #
        # bfloat8_b, not bfloat16: decode's LM head is DRAM-BANDWIDTH bound, streaming this
        # whole [3072, 131072] weight (~805 MB at bf16) to produce ONE token, so its cost is
        # bytes/bandwidth and halving the stored format halves the op. Grid and shard levers
        # cannot touch that -- they redistribute the read, they do not shrink it. The head is
        # a single projection with no depth to compound error through, and the token it feeds
        # comes from an argmax, which cares only about the ARGMAX of the logits, not their
        # exact values.
        self.w_lm_head = ttnn.from_torch(
            hf_model.lm_head.weight.detach().to(torch.bfloat16).transpose(0, 1).contiguous(),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # --- resident KV, one pair per layer, [1, n_kv, C, head_dim] ---------
        cache_host = torch.zeros(1, self.n_kv_heads, self.capacity, self.head_dim, dtype=torch.bfloat16)
        self.k_caches = [
            ttnn.from_torch(cache_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(self.depth)
        ]
        self.v_caches = [
            ttnn.from_torch(cache_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(self.depth)
        ]

        # --- resident decode buffers ----------------------------------------
        # Allocated ONCE here, so their addresses are pinned for the whole run
        # (a captured trace refers to these exact buffers). `decode_prefill`
        # only writes them, on device.
        self.token_buffer = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.position_buffer = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.cache_index = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        #: Rows the decode rotary table is replicated to: the head count the
        #: blocks rotate on the row axis, rounded up to the tile it lands in.
        self._rope_rows = max(_pad_to_tile(self.n_heads), _pad_to_tile(1))
        self._staged = None
        self._pinned = {}
        #: The untilized logits row the last `_greedy_token` scanned.
        self._flat_logits = None

    # ---------------------------------------------------------------- config
    def _stop_token_ids(self):
        """The authoritative stop ids: generation_config first, then config.

        (The tokenizer's own `eos_token_id` disagrees with both for this
        checkpoint; `generate()` obeys generation_config, so that is what both
        sides of the comparison must use.)"""
        found = []
        for source in (self.generation_config, self.config):
            value = getattr(source, "eos_token_id", None)
            if isinstance(value, (list, tuple)):
                found.extend(int(v) for v in value)
            elif value is not None:
                found.append(int(value))
        return tuple(dict.fromkeys(found))

    def decode_horizon(self, prompt_len: int, max_new_tokens=None) -> int:
        """How many tokens the decode path produces, grounded in the model.

        Priority: an explicit request, then `generation_config.max_new_tokens` /
        `max_length - prompt_len`, then the documented fallback cap. Always
        bounded by the KV capacity so a run cannot outgrow its resident cache.
        The STOP RULE itself is `stop_token_ids`, applied to both sides by
        `common_stop_length`.
        """
        wanted = None if max_new_tokens is None else int(max_new_tokens)
        if wanted is None:
            wanted = getattr(self.generation_config, "max_new_tokens", None)
        if wanted is None:
            max_length = getattr(self.generation_config, "max_length", None)
            if max_length:
                wanted = int(max_length) - int(prompt_len)
        if wanted is None:
            wanted = int(os.environ.get("TT_E2E_MAX_NEW_TOKENS", FALLBACK_MAX_NEW_TOKENS))
        room = self.capacity - int(prompt_len)
        if room < 1:
            raise ValueError(
                "prompt (%d) does not leave room to decode inside capacity C=%d" % (prompt_len, self.capacity)
            )
        return max(1, min(int(wanted), room))

    def common_stop_length(self, tt_tokens, reference_tokens) -> int:
        """The model-grounded comparison length: the first stop token on EITHER
        side truncates both, so the golden is never forced to a length the TT
        side invented."""
        limit = min(len(tt_tokens), len(reference_tokens))

        def cut(sequence):
            for i, token in enumerate(sequence[:limit]):
                if int(token) in self.stop_token_ids:
                    return i + 1
            return limit

        return max(1, min(cut(tt_tokens), cut(reference_tokens)))

    # ----------------------------------------------------------------- setup
    def stage_inputs(self, prompt: str = None, input_ids=None, seq_len: int = None) -> dict:
        """Tokenize (if needed) and upload everything the chain reads.

        SETUP, not the forward path: this is the only place host->device staging
        happens for an inference, and it happens once per prompt.
        """
        if input_ids is None:
            if self.tokenizer is None:
                raise RuntimeError("stage_inputs needs a tokenizer or explicit input_ids")
            encoded = self.tokenizer(DEFAULT_PROMPT if prompt is None else prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
        prompt_tokens = input_ids.detach().to(torch.int64).reshape(1, -1)
        prompt_len = int(prompt_tokens.shape[1])
        seq_len = _pad_to_tile(prompt_len) if seq_len is None else int(seq_len)
        if seq_len % TILE or seq_len < prompt_len or seq_len > self.capacity:
            raise ValueError(
                "seq_len %d must be a multiple of %d, >= prompt_len %d and <= capacity %d"
                % (seq_len, TILE, prompt_len, self.capacity)
            )

        host_tokens = torch.zeros(1, seq_len, dtype=torch.int32)
        host_tokens[0, :prompt_len] = prompt_tokens[0].to(torch.int32)
        tokens_tt = ttnn.from_torch(
            host_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        # The additive causal mask MistralModel feeds its layers. Right-padding
        # needs no extra column masking: a real row never attends a later
        # column, and the pad rows' own outputs are dropped by the [:prompt_len]
        # slice. The pad KV rows are overwritten by decode before they are ever
        # inside an attention window.
        blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        host_mask = torch.zeros(seq_len, seq_len, dtype=torch.float32).masked_fill_(blocked, MASK_NEG)
        mask_tt = ttnn.from_torch(
            host_mask.view(1, 1, seq_len, seq_len).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        positions_tt = ttnn.from_torch(
            torch.arange(seq_len, dtype=torch.float32).view(1, seq_len),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        # Seeds for the resident decode buffers: the first generated token sits
        # at position `prompt_len`.
        seed_position = ttnn.from_torch(
            torch.full((1, 1), float(prompt_len), dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        seed_cache_index = ttnn.from_torch(
            torch.full((1,), prompt_len, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self._staged = {
            "prompt_tokens": prompt_tokens,
            "prompt_len": prompt_len,
            "seq_len": seq_len,
            "tokens_tt": tokens_tt,
            "mask_tt": mask_tt,
            "positions_tt": positions_tt,
            "seed_position": seed_position,
            "seed_cache_index": seed_cache_index,
        }
        return self._staged

    def _as_staged(self, inputs) -> dict:
        """Accept the staged dict, `{"input_ids": ...}`, a raw id tensor, or
        None (reuse / build from the default prompt). Staging is setup, so the
        host-free hot path is entered with an already-staged dict."""
        if isinstance(inputs, dict) and "tokens_tt" in inputs:
            self._staged = inputs
            return inputs
        if isinstance(inputs, dict):
            return self.stage_inputs(prompt=inputs.get("prompt"), input_ids=inputs.get("input_ids"))
        if isinstance(inputs, torch.Tensor):
            return self.stage_inputs(input_ids=inputs)
        if isinstance(inputs, str):
            return self.stage_inputs(prompt=inputs)
        if inputs is None and self._staged is not None:
            return self._staged
        return self.stage_inputs()

    # ------------------------------------------------------- TT: prefill (2)
    def run_prefill_logits(self, inputs=None):
        """Call 2 `causal_lm_logits`, and the prefill stage of Call 1.

        Pure ttnn: logits for every prompt position, [1, S, vocab]. Also seeds
        the resident KV for rows [0:S) via the graduated attention body's
        optional cache kwargs.
        """
        staged = self._as_staged(inputs)
        hidden = ttnn.embedding(
            staged["tokens_tt"], self.w_embed, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        cos, sin = self.rotary(position_ids=staged["positions_tt"])
        for index in range(len(self.layers)):
            hidden = self.layers[index](
                hidden,
                attention_mask=staged["mask_tt"],
                position_embeddings=(cos, sin),
                kv_cache=(self.k_caches[index], self.v_caches[index]),
                cache_fill=True,
            )
        hidden = self.final_norm(hidden)
        return self._lm_head(hidden)

    def _lm_head(self, hidden):
        if self.lm_head_compute_kernel_config is not None:
            return ttnn.linear(hidden, self.w_lm_head, compute_kernel_config=self.lm_head_compute_kernel_config)
        return ttnn.linear(hidden, self.w_lm_head)

    def _greedy_token(self, logits):
        """On-device greedy sample over the vocab, on the MULTI-CORE argmax path.

        `ttnn.argmax` picks its parallel path from the INPUT LAYOUT, not from a
        flag: `uses_multicore_path` (argmax_device_operation.cpp) bails out to
        the single-core kernel for anything that is not ROW_MAJOR, so feeding it
        the TILE tensor `ttnn.linear` produces puts the whole 131072-wide vocab
        scan on ONE core. TILE layout also pads this [1, 1, vocab] decode row up
        to the 32-row tile height, so that one core reads ~32x the bytes the
        reduction actually needs.

        Untilizing first fixes both: the scan fans out across the grid and reads
        the unpadded row. Output contract is unchanged either way -- argmax
        returns UINT32 ROW_MAJOR -- so the token still feeds `ttnn.embedding`
        straight back with no host round-trip.

        The untilized row is also stashed: it is the same values as `logits` in
        1/32nd of the bytes (no tile padding), which is what the traced decode
        loop copies out per step.
        """
        self._flat_logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.argmax(self._flat_logits, dim=-1)

    def _prompt_tail_logits(self, logits, prompt_len: int):
        """The last REAL prompt position — the one that predicts the first new
        token (rows at and after `prompt_len` are padding)."""
        return ttnn.slice(logits, [0, prompt_len - 1, 0], [1, prompt_len, self.vocab_size])

    def unpadded_logits(self, logits, prompt_len: int):
        """The real prompt window of a prefill result, [1, prompt_len, vocab]."""
        return ttnn.slice(logits, [0, 0, 0], [1, prompt_len, self.vocab_size])

    # -------------------------------------------- TT: AR decode contract (1)
    def decode_prefill(self, inputs=None):
        """Seed the resident state once: run the prefill chain (which fills the
        resident self-attention KV), pick the first token on device, and point
        the resident position/cache-index buffers at `prompt_len`.

        Returns the state dict. Host-op free when handed an already-staged
        input: every write below is a device-to-device copy.
        """
        staged = self._as_staged(inputs)
        logits = self.run_prefill_logits(staged)
        tail_logits = self._prompt_tail_logits(logits, staged["prompt_len"])
        first_token = self._greedy_token(tail_logits)
        ttnn.copy(first_token, self.token_buffer)
        ttnn.copy(staged["seed_position"], self.position_buffer)
        ttnn.copy(staged["seed_cache_index"], self.cache_index)
        state = {
            "token": self.token_buffer,
            "position": self.position_buffer,
            "cache_index": self.cache_index,
            "kv_cache": (self.k_caches, self.v_caches),
            "prompt_len": staged["prompt_len"],
            "seq_len": staged["seq_len"],
            "first_token": first_token,
            "first_logits": tail_logits,
            "prefill_logits": logits,
        }
        self._state = state
        return state

    def decode_step(self):
        """ONE fixed-shape, host-op-free decode token.

        Reads only resident device tensors, returns `(logits, token)` and leaves
        the state advanced for the next call. This is the unit the trace
        captures and the unit the perf engine replays.
        """
        cos, sin = self.rotary(position_ids=self.position_buffer)
        # The decode blocks lay their heads out on the ROW axis (see
        # `_stubs/attention.py::_decode_native`), so the rotation walks cos/sin
        # row-wise instead of broadcasting one row across a head dimension.
        # Every head is at the same position, so every row it needs is the same
        # row -- replicate it up the tile here, ONCE for the whole step, rather
        # than 26 times inside the blocks. A block whose operands the layout
        # does not fit ignores this and reads row 0, which is unchanged.
        cos = ttnn.repeat(cos, [1, self._rope_rows, 1])
        sin = ttnn.repeat(sin, [1, self._rope_rows, 1])
        hidden = ttnn.embedding(self.token_buffer, self.w_embed, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for index in range(len(self.layers)):
            hidden = self.layers[index](
                hidden,
                attention_mask=None,
                position_embeddings=(cos, sin),
                kv_cache=(self.k_caches[index], self.v_caches[index]),
                cache_pos_tensor=self.cache_index,
            )
        hidden = self.final_norm(hidden)
        logits = self._lm_head(hidden)
        next_token = self._greedy_token(logits)
        # On-device token feed + position advance: nothing leaves the device.
        # cache_index stays on the two-op `copy(add(x, 1), x)` form on purpose:
        # it is ROW_MAJOR (paged_update_cache takes its row index that way) and
        # ttnn eltwise refuses a preallocated output for row-major inputs
        # ("Optional output tensor with Row Major input is not supported"), so
        # the in-place one-op form position_buffer uses is not available here.
        ttnn.copy(next_token, self.token_buffer)
        ttnn.add(self.position_buffer, 1.0, output_tensor=self.position_buffer)
        ttnn.copy(ttnn.add(self.cache_index, 1), self.cache_index)
        return logits, next_token

    def run_generate(self, inputs=None, max_new_tokens=None):
        """Call 1 `text_generation`: prefill, then free-running greedy decode.

        Each step consumes the PREVIOUS TT step's own argmax token out of the
        resident buffer — no reference token is injected at any joint. Returns
        device tensors; the caller reads them back once, after the run.

        The decode step is CAPTURED ONCE and replayed per token when the device
        was opened with a trace region: the step is already host-op free and
        fixed-shape (that is what `decode_step` is built for), so re-issuing its
        several hundred ops from Python every token is pure host overhead —
        94% of this call's ttnn dispatches. A device without a trace region
        falls back to the eager loop, which is the same math.
        """
        staged = self._as_staged(inputs)
        max_new_tokens = self.decode_horizon(staged["prompt_len"], max_new_tokens)
        state = self.decode_prefill(staged)
        step_logits = [state["first_logits"]]
        tokens = [state["first_token"]]
        remaining = max_new_tokens - 1

        if remaining > 0:
            # One eager step first: it compiles the kernels the capture records,
            # and it is a REAL token, so nothing is thrown away.
            logits, next_token = self.decode_step()
            step_logits.append(logits)
            tokens.append(next_token)
            remaining -= 1

        replay = self._capture_decode_step(remaining) if remaining > 0 else None
        for step in range(remaining):
            if replay is None:
                logits, next_token = self.decode_step()
            else:
                trace_id, logits_buffer, token_buffer, logit_slots, token_slots = replay
                ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)
                # Every replay writes the SAME buffers, so each step's outputs
                # are copied out before the next one overwrites them. The copy
                # is of the UNTILIZED row, not the tile tensor: same values in
                # 1/32nd of the bytes. It goes into a slot allocated BEFORE the
                # capture -- allocating here instead hands back addresses the
                # trace's own freed intermediates still get written to, which
                # silently corrupts earlier steps.
                logits, next_token = logit_slots[step], token_slots[step]
                ttnn.copy(logits_buffer, logits)
                ttnn.copy(token_buffer, next_token)
            step_logits.append(logits)
            tokens.append(next_token)
        if replay is not None:
            ttnn.release_trace(self.device, replay[0])
        return {
            "tokens": tokens,
            "step_logits": step_logits,
            "prompt_len": staged["prompt_len"],
            "max_new_tokens": max_new_tokens,
        }

    def _capture_decode_step(self, slots: int):
        """Capture one decode step, or None if this device has no trace region.

        Capture RECORDS without executing, so the resident state the next replay
        reads is exactly the state this call was entered with.

        The per-step destination buffers are allocated FIRST, before the capture
        allocates anything, so no slot can land on a trace-internal address.
        """
        trace_id = None
        try:
            logit_slots = [
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, self.vocab_size]),
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                    self.device,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(slots)
            ]
            token_slots = [
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
                )
                for _ in range(slots)
            ]
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            _, next_token = self.decode_step()
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            return trace_id, self._flat_logits, next_token, logit_slots, token_slots
        except Exception:  # noqa: BLE001 - no/too-small trace region: eager loop
            # A capture that began must be closed even on the failure path, or
            # the device is left recording and every later op disappears into a
            # trace nobody replays.
            if trace_id is not None:
                try:
                    ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
                try:
                    ttnn.release_trace(self.device, trace_id)
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
            return None

    # -------------------------------------------------------- readback (host)
    def token_ids(self, tokens) -> list:
        """Read the generated ids back — once, after the chain has finished."""
        return [int(ttnn.to_torch(token).reshape(-1)[0]) for token in tokens]

    def stacked_logits(self, step_logits):
        """The per-step next-token logits as one [steps, vocab] torch tensor."""
        rows = [ttnn.to_torch(logits).float().reshape(-1, self.vocab_size)[-1] for logits in step_logits]
        return torch.stack(rows)

    # ------------------------------------------------------- trace: prefill
    def prefill_trace_inputs(self) -> dict:
        """ZERO-ARG: exactly what `prefill_trace_setup` takes."""
        return {"input_ids": self._golden_prompt_tokens()}

    def prefill_trace_setup(self, inputs):
        """Pin the sequence axis to capacity C and pre-upload every buffer the
        forward reads (padded tokens, additive mask, rotary positions, resident
        KV), so the captured step touches nothing on the host."""
        staged = self.stage_inputs(input_ids=inputs["input_ids"], seq_len=self.capacity)
        self._pinned["prefill"] = staged
        return staged

    def prefill_trace_step(self):
        """ONE host-op-free prefill forward at the fixed [1, C, hidden] shape."""
        return self.run_prefill_logits(self._pinned["prefill"])

    # -------------------------------------------------------- trace: decode
    def decode_trace_inputs(self) -> dict:
        """ZERO-ARG: exactly what `decode_trace_setup` takes."""
        return {"input_ids": self._golden_prompt_tokens()}

    def decode_trace_setup(self, inputs):
        """Seed the resident self-KV at capacity C and point the resident
        buffers at the first decode position, then hand the pinned step over."""
        staged = self.stage_inputs(input_ids=inputs["input_ids"], seq_len=self.capacity)
        self.decode_prefill(staged)
        self._pinned["decode"] = staged
        return staged

    def decode_trace_step(self):
        """ONE host-op-free decode token off the resident buffers."""
        return self.decode_step()[0]

    def _golden_prompt_tokens(self):
        path = captured_inputs_path()
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False).reshape(1, -1)
        encoded = self.tokenizer(DEFAULT_PROMPT, return_tensors="pt")
        return encoded["input_ids"]


# ---------------------------------------------------------------------------
# HF reference helpers — the ONLY place HF forward code runs
# ---------------------------------------------------------------------------
class _hf_depth:
    """Cap the reference at the same depth the TT build used, so `--layers N`
    compares like with like."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.saved = None

    def __enter__(self):
        pipe = self.pipeline
        if pipe.depth >= pipe.full_depth:
            return pipe.hf
        inner = pipe.hf.model
        self.saved = (inner.layers, int(pipe.hf.config.num_hidden_layers))
        inner.layers = torch.nn.ModuleList(list(inner.layers)[: pipe.depth])
        pipe.hf.config.num_hidden_layers = pipe.depth
        return pipe.hf

    def __exit__(self, *exc):
        if self.saved is not None:
            self.pipeline.hf.model.layers, self.pipeline.hf.config.num_hidden_layers = self.saved
            self.saved = None
        return False


def _hf_reference_logits(pipeline, staged):
    """GOLDEN for Call 2: the CausalLM head's own teacher-forced logits."""
    with torch.no_grad(), _hf_depth(pipeline) as hf:
        return hf(input_ids=staged["prompt_tokens"], use_cache=False).logits.float()


def _hf_reference_generate(pipeline, staged, max_new_tokens):
    """GOLDEN for Call 1: greedy `generate()` under the SAME stop rule and the
    SAME horizon cap the TT chain used."""
    with torch.no_grad(), _hf_depth(pipeline) as hf:
        out = hf.generate(
            input_ids=staged["prompt_tokens"],
            attention_mask=torch.ones_like(staged["prompt_tokens"]),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=list(pipeline.stop_token_ids) or None,
            pad_token_id=(pipeline.stop_token_ids or (0,))[0],
        )
    generated = out.sequences[0, staged["prompt_tokens"].shape[1] :]
    scores = torch.stack([score[0].float() for score in out.scores])
    return [int(t) for t in generated], scores


# ---------------------------------------------------------------------------
# tool-facing selftests (invoked with ZERO arguments by the planner probes)
# ---------------------------------------------------------------------------
def _selftest_layers():
    value = os.environ.get("TT_E2E_SELFTEST_LAYERS")
    return int(value) if value else None


#: Filled in by `trace_capture_selftest` with what the last capture measured, so
#: a caller can record the evidence without re-running the capture.
TRACE_REPORT = {}
TRACE_REPLAYS = 5


def trace_capture_selftest(device=None):
    """Per stage in PIPELINE_STAGES: setup -> capture ONE step -> execute_trace
    -> PCC against the eager step -> release before the next stage.

    True only if EVERY stage captured host-free and matched. Stage traces are
    released before the next stage so they never co-reside.
    """
    import time

    from models.demos.voxtral_tts_backbone.selftest_device import close_selftest_device, open_selftest_device

    owned = device is None
    if owned:
        device = open_selftest_device()
    try:
        pipeline = build_pipeline(device, layers=_selftest_layers())
        every_stage_ok = True
        TRACE_REPORT.clear()
        TRACE_REPORT["capacity"] = pipeline.capacity
        TRACE_REPORT["depth"] = pipeline.depth
        TRACE_REPORT["stages"] = {}
        for stage in PIPELINE_STAGES:
            inputs = getattr(pipeline, "%s_trace_inputs" % stage)()
            setup = getattr(pipeline, "%s_trace_setup" % stage)
            step = getattr(pipeline, "%s_trace_step" % stage)
            setup(inputs)
            eager = ttnn.to_torch(step()).float()
            # Re-seed so the captured step starts from the same state the eager
            # one did (a decode step advances the resident buffers).
            setup(inputs)
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            captured = step()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            replayed = ttnn.to_torch(captured).float()
            ok, pcc = comp_pcc(eager, replayed, 0.99)
            started = time.perf_counter()
            for _ in range(TRACE_REPLAYS):
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            replay_ms = (time.perf_counter() - started) * 1000.0 / TRACE_REPLAYS
            ttnn.release_trace(device, trace_id)
            TRACE_REPORT["stages"][stage] = {
                "captured": True,
                "replay_matches_eager": bool(ok),
                "replay_corr": float(pcc),
                "replay_ms": round(replay_ms, 3),
                "step": "%s_trace_step" % stage,
            }
            print(
                "[voxtral-e2e] trace stage=%s captured=1 replay_match=%s corr=%s replay=%.3fms"
                % (stage, bool(ok), pcc, replay_ms),
                flush=True,
            )
            every_stage_ok = every_stage_ok and bool(ok)
        TRACE_REPORT["trace_ready"] = every_stage_ok
        return every_stage_ok
    finally:
        if owned:
            close_selftest_device(device)


def host_op_selftest(device=None):
    """Run BOTH task heads' forwards under the host-op observer.

    Tokenization, the one-time weight build and the final readback stay OUTSIDE
    the observed region; the whole staged-inputs -> output math (the prompt
    embedding, all 26 layers, the LM head and greedy sampling) is INSIDE it.
    """
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    from models.demos.voxtral_tts_backbone.selftest_device import close_selftest_device, open_selftest_device

    owned = device is None
    if owned:
        device = open_selftest_device()
    try:
        pipeline = build_pipeline(device, layers=_selftest_layers())
        staged = pipeline.stage_inputs(DEFAULT_PROMPT)
        steps = int(os.environ.get("TT_E2E_OBSERVED_STEPS", 3))
        with observe_host_ops() as ops:
            # Call 2: teacher-forced logits over the whole prompt.
            prefill_logits = pipeline.run_prefill_logits(staged)
            # Call 1: seed the resident state and free-run the decode chain.
            pipeline.decode_prefill(staged)
            for _ in range(steps):
                pipeline.decode_step()
        ttnn.synchronize_device(device)
        result = verdict(list(ops))
        print(
            "[voxtral-e2e] host-op observer: on_device=%s n_host_ops=%d (%s)"
            % (result.get("on_device"), result.get("n_host_ops", 0), result.get("reason")),
            flush=True,
        )
        del prefill_logits
        return result
    finally:
        if owned:
            close_selftest_device(device)


if __name__ == "__main__":  # pragma: no cover - manual smoke entry
    print(host_op_selftest())
    print(trace_capture_selftest())
