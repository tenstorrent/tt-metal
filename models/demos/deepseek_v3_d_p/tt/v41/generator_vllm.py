# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash for vllm-tt-plugin (harness): the V4.1 prefill and the decode step behind the plugin's
model contract (``initialize_vllm_model``, ``allocate_kv_cache``, ``prefill_forward``, ``decode_forward``,
``read_decode_output``, ``warmup_model_prefill`` / ``warmup_model_decode``; ``models/demos/deepseek_v3/tt/generator_vllm.py``
is the precedent).

Requests and state. The plugin hands prefill and decode rows with vLLM's block tables; this model keeps its own
caches and ignores the KV tensors it allocates for the plugin, so a request is identified by the first block of its
block table (block 0 is vLLM's null block, never a request's). A prefill runs into one of ``TT_V41_PREFILL_SLOTS``
prefill states (``V41PrefillState``; traced per state and prompt length, else eager) and joins the request's decode
slot (``TtV41BatchDecoder``: ``max_batch_size`` slots in a pool sharded over the mesh rows; the slot of that block,
or the least recently used one); a decode call steps every row's slot at once (one replayed capture) and reads the
gathered logits once. Every row is text, single-chunk prefill (``start_pos`` 0), host sampling (the logits of the
full vocabulary come to the host). Prompts need at least ``SLIDING_WINDOW`` tokens (the decode step starts after a
full window) and prompt plus output at most ``max_model_len``.

Environment: ``TT_V41_CHECKPOINT`` (the pinned snapshot; ``weights.resolve_checkpoint``), ``TT_V41_WEIGHT_CACHE``
(the device MoE cache, ``tests/v41/weight_cache.py``), ``TT_V41_ENGRAM_MMAP`` (the Engram tables mapped from disk).
``TT_V41_CHUNK`` (default 8192) is the prefill chunk; ``max_model_len`` is rounded up to a multiple of it.
``TT_V41_LAYERS`` (default ``all``) selects the layer schedule, for checks on shallower models.
"""

import os
import time
from collections import OrderedDict

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.decode import ROWS
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer, V41PrefillTrace
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

WARMUP_PROMPT = 255  # tokens of the warm-up prompt (a full window; odd, so the ratio-2 carry path runs first)
WARMUP_DECODE_STEPS = 8  # at most; done once the step is captured and replayed once
PREFILL_SLOTS = "TT_V41_PREFILL_SLOTS"  # prefill states (default 2): requests prefill through them into the pool
# Traced prefill: ``TT_V41_TRACE_PREFILL_LENGTHS`` lists prompt lengths (tokens, comma-separated) whose prefill is
# captured per slot at warm-up (the trace phase) and replayed for every prompt of exactly that length; other
# lengths run eagerly, or are captured on their first request when ``TT_V41_TRACE_PREFILL_ON_MISS=1``.
TRACE_PREFILL_LENGTHS = "TT_V41_TRACE_PREFILL_LENGTHS"
TRACE_PREFILL_ON_MISS = "TT_V41_TRACE_PREFILL_ON_MISS"


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _layers(spec: str) -> list[int]:
    """``all`` or a layer list such as ``0,2-13,20-39`` (a harness knob: shallower models for checks)."""
    if spec.strip() == "all":
        return list(range(C.NUM_LAYERS))
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return sorted(out)


class DeepseekV41ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_sample_on_device": False,
        "supports_async_decode": False,
    }

    def __init__(
        self,
        mesh_device,
        model: TtV41Transformer,
        decoder: TtV41BatchDecoder,
        tokenizer,
        max_batch_size: int,
        prefill_slots: int = 2,
        vllm_config=None,
    ):
        """``vllm_config``: unused; vLLM's registry classifies a model as generative by the shape of ``__init__``,
        ``embed_input_ids``, ``forward`` and ``compute_logits`` (interfaces_base.py), none of which the TT plugin
        calls: it runs ``prefill_forward`` / ``decode_forward``."""
        self.mesh_device, self.model, self.decoder, self.tokenizer = mesh_device, model, decoder, tokenizer
        self.config = model.config
        self.max_batch_size = max_batch_size
        assert decoder.num_slots == max_batch_size, (decoder.num_slots, max_batch_size)
        # decode slots: first block id -> pool slot, least recently used first; the free slots
        self.slots: OrderedDict[int, int] = OrderedDict()
        self.free: list[int] = list(range(max_batch_size))
        # prefill states, in use round-robin (a request's rows are in its decode slot once its prefill returns)
        self.prefill_states: list[V41PrefillState] = [model.new_state() for _ in range(prefill_slots)]
        self.next_prefill_state = 0
        self.warmed_up = False
        # traced prefill per (slot, prompt length): the fixed input buffers are shared by every trace
        self.prefill_traces: dict[tuple[int, int], V41PrefillTrace] = {}
        self.prefill_buffers: dict = {}
        self.prefill_traced = False
        spec = os.environ.get(TRACE_PREFILL_LENGTHS, "").strip()
        self.trace_lengths = sorted({int(x) for x in spec.split(",") if x.strip()})
        self.trace_on_miss = os.environ.get(TRACE_PREFILL_ON_MISS, "0") == "1"
        bad = [n for n in self.trace_lengths if n + ROWS > model.max_seq_len or n < C.SLIDING_WINDOW]
        assert not bad, f"{TRACE_PREFILL_LENGTHS} {bad}: outside [{C.SLIDING_WINDOW}, max_seq_len - {ROWS}]"

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("the TT plugin runs prefill_forward / decode_forward")

    def forward(self, input_ids, positions):
        raise NotImplementedError("the TT plugin runs prefill_forward / decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("the TT plugin runs prefill_forward / decode_forward")

    # --- construction (the plugin's loader) --------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        assert tt_data_parallel == 1, f"tt_data_parallel {tt_data_parallel}: one request pool on the whole mesh"
        ckpt = resolve_checkpoint()
        assert ckpt is not None, "the V4.1 checkpoint is not available (TT_V41_CHECKPOINT or the HF cache)"
        chunk = int(os.environ.get("TT_V41_CHUNK", 8192))
        max_seq = _round_up(max_seq_len, chunk)
        layers = _layers(os.environ.get("TT_V41_LAYERS", "all"))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ckpt.root)  # the snapshot holds the tokenizer
        t0 = time.time()
        spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
        root = weight_cache_dir(spec, mesh_device.shape)
        engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
        top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
        model = TtV41Transformer(
            mesh_device,
            C,
            layers,
            lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
            top["embed.weight"],
            top["norm.weight"],
            top["head.weight"],
            max_seq_len=max_seq,
            chunk=chunk,
            engram=engram,
            engram_hash=engram_hash,
            weight_cache_path=root,
        )
        sp = tuple(mesh_device.shape)[0]
        assert (
            max_batch_size % sp == 0
        ), f"max_num_seqs {max_batch_size}: the pool holds the same slots per mesh row ({sp})"
        decoder = TtV41BatchDecoder(model, root, users_per_rank=max_batch_size // sp, trace=True)
        prefill_slots = int(os.environ.get(PREFILL_SLOTS, 2))
        logger.info(
            f"DeepSeek-V4.1-Flash: {len(layers)} layers built in {time.time() - t0:.0f} s; max_seq_len {max_seq} "
            f"(chunk {chunk}), {max_batch_size} decode slots ({max_batch_size // sp} per mesh row), {prefill_slots} "
            f"prefill states, mesh {tuple(mesh_device.shape)}"
        )
        return cls(mesh_device, model, decoder, tokenizer, max_batch_size, prefill_slots)

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """vLLM's KV block pool: every slot can hold a full context (the plugin's blocks are bookkeeping here)."""
        assert max_model_len is not None and max_num_seqs is not None
        return int(max_model_len) * int(max_num_seqs)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """The plugin's per-layer KV tensors: not used (the request slots hold the caches); placeholders."""
        logger.info(
            f"DeepSeek-V4.1-Flash: vLLM KV cache {kv_cache_shape} x {num_layers} layers not allocated (slot states)"
        )
        return [torch.zeros(1) for _ in range(num_layers)]

    # --- requests -------------------------------------------------------------------------------------------------
    @staticmethod
    def _request_key(page_table, row: int) -> int:
        key = int(page_table[row, 0])
        assert key != 0, f"row {row}: block table starts with the null block"
        return key

    def _slot_for_prefill(self, key: int) -> int:
        if key in self.slots:
            slot = self.slots.pop(key)
        elif self.free:
            slot = self.free.pop()
        else:
            _, slot = self.slots.popitem(last=False)  # the least recently used request is done or preempted
        self.slots[key] = slot
        return slot

    def _slot_for_decode(self, key: int) -> int:
        slot = self.slots.pop(key)
        self.slots[key] = slot  # most recently used
        return slot

    def _prefill_state(self) -> V41PrefillState:
        state = self.prefill_states[self.next_prefill_state]
        self.next_prefill_state = (self.next_prefill_state + 1) % len(self.prefill_states)
        return state

    def prefill_forward(self, **kwargs) -> torch.Tensor:
        """``tokens`` [B, S] (padded), ``prompt_lens`` [B], ``page_table`` [B, blocks], ``start_pos`` zeros ->
        fp32 logits ``[B, 1, vocab]`` of each prompt's last token (host sampling reads row -1)."""
        tokens, lengths, page_table = kwargs["tokens"], kwargs["prompt_lens"], kwargs["page_table"]
        start_pos = kwargs.get("start_pos")
        assert start_pos is None or all(int(x) == 0 for x in start_pos), f"chunked or cached prefill: {start_pos}"
        assert "sampling_params" not in kwargs or kwargs["sampling_params"] is None, "host sampling only"
        out = torch.zeros(tokens.shape[0], 1, self.config.VOCAB_SIZE, dtype=torch.float32)
        for i in range(tokens.shape[0]):
            n = int(lengths[i])
            if n == 0:
                continue
            assert (
                n >= self.config.SLIDING_WINDOW
            ), f"prompt of {n} tokens: decode needs a full window ({self.config.SLIDING_WINDOW})"
            assert (
                n + ROWS <= self.model.max_seq_len
            ), f"prompt of {n} tokens beyond max_seq_len {self.model.max_seq_len}"
            prompt = tokens[i, :n].to(torch.int64)
            slot = self._slot_for_prefill(self._request_key(page_table, i))
            state = self._prefill_state()
            t0 = time.time()
            trace = self.prefill_traces.get((id(state), n))
            if trace is None and self.trace_on_miss:
                trace = self._capture_prefill(state, n)
                logger.info(f"DeepSeek-V4.1-Flash: prefill of {n} tokens captured in {time.time() - t0:.1f} s")
                t0 = time.time()
            if trace is not None:
                logits, _ = trace.run(prompt)
            else:
                logits = self.model.prefill_into(prompt, state)
            t1 = time.time()
            self.decoder.join(slot, prompt, state)
            mode = "replay" if trace is not None else "eager"
            phases = "" if trace is None else " " + " ".join(f"{k} {v:.3f}" for k, v in trace.last.items())
            logger.info(
                f"DeepSeek-V4.1-Flash: prefill of {n} tokens in {time.time() - t0:.2f} s ({mode};{phases} "
                f"join {time.time() - t1:.3f})"
            )
            out[i, 0] = logits[-1]
        return out

    def _capture_prefill(self, state: V41PrefillState, n: int) -> V41PrefillTrace:
        """Capture the prefill of ``n``-token prompts into ``state`` (the slot's own cache tensors, so the slot's
        decode captures read what the replay wrote). The first capture at a length compiles its kernels (two
        untraced runs); later slots at that length need one untraced run for the steady-state resources."""
        first = not any(k[1] == n for k in self.prefill_traces)
        trace = V41PrefillTrace(
            self.model,
            self._warmup_prompt(n),
            1,
            state=state,
            buffers=self.prefill_buffers,
            warmup=2 if first else 1,
        )
        self.prefill_traces[(id(state), n)] = trace
        return trace

    def decode_forward(self, **kwargs) -> torch.Tensor:
        """``tokens`` [B, 1], ``start_pos`` [B], ``page_table`` [B, blocks] -> fp32 logits ``[B, 1, vocab]``; rows
        whose block table starts with the null block are idle. One step of the pool for every active row."""
        tokens, positions, page_table = kwargs["tokens"], kwargs["start_pos"], kwargs["page_table"]
        assert kwargs.get("sampling_params") is None, "host sampling only"
        out = torch.zeros(tokens.shape[0], 1, self.config.VOCAB_SIZE, dtype=torch.float32)
        step, rows = {}, {}
        for i in range(tokens.shape[0]):
            if int(page_table[i, 0]) == 0:
                continue
            slot = self._slot_for_decode(self._request_key(page_table, i))
            p = int(positions[i])
            assert p == self.decoder.position(
                slot
            ), f"row {i}: vLLM position {p}, slot {slot} at {self.decoder.position(slot)}"
            step[slot] = int(tokens[i, 0])
            rows[slot] = i
        if not step:
            return out
        t0 = time.time()
        logits = self.decoder.decode(step)
        for slot, i in rows.items():
            out[i, 0] = logits[slot]
        last = self.decoder.last
        logger.debug(f"DeepSeek-V4.1-Flash: decode step of {len(step)} rows in {time.time() - t0:.3f} s {last}")
        return out

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out

    # --- warm-up (the plugin's model_runner.warmup_model) ---------------------------------------------------------------
    def warmup_model_prefill(self, enable_trace: bool, **kwargs) -> None:
        """Compile the prefill (one prompt per prefill state, each joining every decode slot) on the first call;
        the second (trace) call, which the plugin makes after the decode capture, captures the prefill of every
        ``TT_V41_TRACE_PREFILL_LENGTHS`` length on every prefill state (the captures' persistent tensors are
        consumed right after each replay, so they may share addresses with the decode capture's intermediates; the
        pool and the prefill states predate every capture). Other lengths keep running eagerly."""
        if enable_trace:
            if self.prefill_traced:
                return
            for n in self.trace_lengths:
                for state in self.prefill_states:
                    t0 = time.time()
                    self._capture_prefill(state, n)
                    logger.info(
                        f"DeepSeek-V4.1-Flash warm-up: prefill of {n} tokens captured in {time.time() - t0:.1f} s"
                    )
            self.prefill_traced = True
            return
        if self.warmed_up:
            return
        self._warmup_join()

    def _warmup_join(self) -> None:
        """Prefill the warm-up prompt into every prefill state and join every decode slot from them."""
        prompt = self._warmup_prompt()
        for k, state in enumerate(self.prefill_states):
            t0 = time.time()
            self.model.prefill_into(prompt, state)
            logger.info(f"DeepSeek-V4.1-Flash warm-up: prefill of {prompt.numel()} tokens in {time.time() - t0:.1f} s")
        t0 = time.time()
        for slot in range(self.max_batch_size):
            self.decoder.join(slot, prompt, self.prefill_states[slot % len(self.prefill_states)])
        logger.info(f"DeepSeek-V4.1-Flash warm-up: {self.max_batch_size} slots joined in {time.time() - t0:.1f} s")

    def warmup_model_decode(self, enable_trace: bool, **kwargs) -> None:
        """Run the decode step on every slot until it is captured and replayed once (the first call compiles, the
        capture follows; the second call finds it captured)."""
        if self.warmed_up:
            return
        if not self.decoder.slots:  # the prefill warm-up did not run
            self._warmup_join()
        prompt = self._warmup_prompt()
        tokens = {slot: int(prompt[-1]) for slot in range(self.max_batch_size)}
        replays = 0
        for step in range(WARMUP_DECODE_STEPS):
            t0 = time.time()
            logits = self.decoder.decode(tokens)
            tokens = {slot: int(logits[slot].argmax()) for slot in tokens}
            mode = self.decoder.last.get("mode")
            replays += mode == "replay"
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: decode step {step} ({mode}) in {time.time() - t0:.2f} s {self.decoder.last}"
            )
            if self.decoder.capture is not None and replays >= 1:
                break
        assert self.decoder.capture is not None, "the decode step was not captured"
        self.warmed_up = True

    def _warmup_prompt(self, n: int = WARMUP_PROMPT) -> torch.Tensor:
        """The first ``n`` tokens of the text the tests use (``oracle.text_tokens``, with this tokenizer), the text
        repeated when ``n`` outruns it."""
        import bz2

        book = bz2.open(orc._HERE.parents[3] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2", "rt").read()
        text = book[book.index("It was the best of times") :]
        text = (text * (12 * n // len(text) + 1))[: 12 * n]
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n]
        assert len(ids) == n, f"warm-up text yields {len(ids)} tokens, {n} needed"
        return torch.tensor(ids, dtype=torch.int64)
