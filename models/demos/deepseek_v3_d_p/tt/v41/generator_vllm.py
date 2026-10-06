# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash for vllm-tt-plugin (harness): the V4.1 prefill and the decode step behind the plugin's
model contract (``initialize_vllm_model``, ``allocate_kv_cache``, ``prefill_forward``, ``decode_forward``,
``read_decode_output``, ``warmup_model_prefill`` / ``warmup_model_decode``; ``models/demos/deepseek_v3/tt/generator_vllm.py``
is the precedent).

Requests and state. The plugin hands prefill and decode rows with vLLM's block tables; this model keeps its own
caches (one ``V41PrefillState`` per request slot, ``max_batch_size`` of them, each with its decode captures) and
ignores the KV tensors it allocates for the plugin, so a request is identified by the first block of its block table
(block 0 is vLLM's null block, never a request's): a prefill takes the slot of that block or the least recently used
free one, a decode row finds its slot by it. Every row is text, single-chunk prefill (``start_pos`` 0), host sampling
(the logits of the full vocabulary come to the host); prefill runs eagerly (``TtV41Transformer.prefill_into``), decode
replays the slot's captures (``TtV41Decoder``, ``trace=True``). Prompts need at least ``SLIDING_WINDOW`` tokens (the
decode step starts after a full window) and prompt plus output at most ``max_model_len - 32``.

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
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.decode import ROWS, TtV41Decoder
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

WARMUP_PROMPT = 255  # tokens of the warm-up prompt (a full window; odd, so the ratio-2 carry path runs first)
WARMUP_DECODE_STEPS = 16  # at most; the slot is done once both parities are captured and replayed once


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
        decoder: TtV41Decoder,
        tokenizer,
        max_batch_size: int,
        vllm_config=None,
    ):
        """``vllm_config``: unused; vLLM's registry classifies a model as generative by the shape of ``__init__``,
        ``embed_input_ids``, ``forward`` and ``compute_logits`` (interfaces_base.py), none of which the TT plugin
        calls: it runs ``prefill_forward`` / ``decode_forward``."""
        self.mesh_device, self.model, self.decoder, self.tokenizer = mesh_device, model, decoder, tokenizer
        self.config = model.config
        self.max_batch_size = max_batch_size
        # request slots: first block id -> state, least recently used first
        self.slots: OrderedDict[int, V41PrefillState] = OrderedDict()
        self.free: list[V41PrefillState] = [model.new_state() for _ in range(max_batch_size)]
        for state in self.free:
            decoder.prepare(state)  # every slot's buffers before the first capture (TtV41Decoder.prepare)
        self.warmed_up = False

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
        decoder = TtV41Decoder(model, root, trace=True)
        logger.info(
            f"DeepSeek-V4.1-Flash: {len(layers)} layers built in {time.time() - t0:.0f} s; max_seq_len {max_seq} "
            f"(chunk {chunk}), {max_batch_size} request slots, mesh {tuple(mesh_device.shape)}"
        )
        return cls(mesh_device, model, decoder, tokenizer, max_batch_size)

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

    def _slot_for_prefill(self, key: int) -> V41PrefillState:
        if key in self.slots:
            state = self.slots.pop(key)
        elif self.free:
            state = self.free.pop()
        else:
            _, state = self.slots.popitem(last=False)  # the least recently used request is done or preempted
        self.slots[key] = state
        return state

    def _slot_for_decode(self, key: int) -> V41PrefillState:
        state = self.slots.pop(key)
        self.slots[key] = state  # most recently used
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
            state = self._slot_for_prefill(self._request_key(page_table, i))
            t0 = time.time()
            logits = self.model.prefill_into(prompt, state)
            self.decoder.begin(prompt, state)
            logger.info(f"DeepSeek-V4.1-Flash: prefill of {n} tokens in {time.time() - t0:.2f} s")
            out[i, 0] = logits[-1]
        return out

    def decode_forward(self, **kwargs) -> torch.Tensor:
        """``tokens`` [B, 1], ``start_pos`` [B], ``page_table`` [B, blocks] -> fp32 logits ``[B, 1, vocab]``; rows
        whose block table starts with the null block are idle."""
        tokens, positions, page_table = kwargs["tokens"], kwargs["start_pos"], kwargs["page_table"]
        assert kwargs.get("sampling_params") is None, "host sampling only"
        out = torch.zeros(tokens.shape[0], 1, self.config.VOCAB_SIZE, dtype=torch.float32)
        for i in range(tokens.shape[0]):
            if int(page_table[i, 0]) == 0:
                continue
            state = self._slot_for_decode(self._request_key(page_table, i))
            p = int(positions[i])
            assert p == state.start, f"row {i}: vLLM position {p}, state at {state.start}"
            out[i, 0] = self.decoder.decode(int(tokens[i, 0]), state)
        return out

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out

    # --- warm-up (the plugin's model_runner.warmup_model) ---------------------------------------------------------------
    def warmup_model_prefill(self, enable_trace: bool, **kwargs) -> None:
        """Compile the prefill (one prompt per slot) on the first call; the second (trace) call is a no-op: prefill
        runs eagerly."""
        if self.warmed_up:
            return
        prompt = self._warmup_prompt()
        for state in self.free:
            t0 = time.time()
            self.model.prefill_into(prompt, state)
            self.decoder.begin(prompt, state)
            logger.info(f"DeepSeek-V4.1-Flash warm-up: prefill of {prompt.numel()} tokens in {time.time() - t0:.1f} s")

    def warmup_model_decode(self, enable_trace: bool, **kwargs) -> None:
        """Run the decode step on every slot until its two parities are captured (the first call compiles, the
        captures follow; the second call finds every slot captured)."""
        if self.warmed_up:
            return
        prompt = self._warmup_prompt()
        for state in self.free:
            if state.start == 0:  # the prefill warm-up did not run
                self.model.prefill_into(prompt, state)
                self.decoder.begin(prompt, state)
            token, replays = int(prompt[-1]), 0
            for step in range(WARMUP_DECODE_STEPS):
                t0 = time.time()
                token = int(self.decoder.decode(token, state).argmax())
                mode = self.decoder.last.get("mode")
                replays += mode == "replay"
                logger.info(f"DeepSeek-V4.1-Flash warm-up: decode step {step} ({mode}) in {time.time() - t0:.2f} s")
                if self.decoder.traced(state) and replays >= 1:
                    break
            assert self.decoder.traced(state), "the decode step was not captured"
        self.warmed_up = True

    def _warmup_prompt(self) -> torch.Tensor:
        """The first WARMUP_PROMPT tokens of the text the tests use (``oracle.text_tokens``, with this tokenizer)."""
        import bz2

        book = bz2.open(orc._HERE.parents[3] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2", "rt").read()
        text = book[book.index("It was the best of times") :][: 12 * WARMUP_PROMPT]
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:WARMUP_PROMPT]
        return torch.tensor(ids, dtype=torch.int64)
