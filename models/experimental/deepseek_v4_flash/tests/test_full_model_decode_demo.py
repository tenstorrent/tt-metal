# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode demo on the full ttnn ``DeepSeekV4Model``.

Builds the whole model once via :class:`DeepSeekV4Model`. There is no dedicated
prefill: a chat prompt is "prefilled" by replaying the decode step once per
prompt token (at ascending absolute positions), seeding every layer's sliding
K=V + compressor cache in place, then generation continues one token per step
(``S = 1``) against that cache. Traced decode uses a paged KV cache (one
session, 32-row blocks) so this demo shares the cache layout with the
multi-user and serving paths. The RoPE tables are produced once for the
maximum length; each decode step slices the single position row(s) it needs.

The test has three deployment variants:

* ``tp1_8chip``: eight single-chip pipeline stages on an 8-chip mesh.
* ``tp4_8chip``: two 1x4 tensor-parallel stages on the same 8-chip mesh.
* ``tp4_32chip``: the same two 1x4 stages on a 32-chip Galaxy (24 chips idle).

Attention uses q_a/kv, replicated full-width on every rank of a
stage; head-sharded SDPA, batched local-group O_A and row-parallel O_B.
MoE shards the intermediate dimension and all-reduces its output. The DRISC
prefetcher stays on (same as TP1) for every projection that still fits the
shared GCB.

All weights live on device in ``bfloat4_b``. Set ``DEEPSEEK_V4_CACHE_DIR`` to
reuse the converted ttnn weight tiles across runs, and optionally cap the stack
with ``DEEPSEEK_V4_DECODE_LAYERS=N`` for bring-up.

Run it (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    DEEPSEEK_V4_MAX_NEW_TOKENS=16 pytest -s \\
      models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py

Set ``DEEPSEEK_V4_START_POS`` to begin prefill at a non-zero absolute position,
for example ``DEEPSEEK_V4_START_POS=1024``.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.encoding_dsv4 import encode_messages
from models.experimental.deepseek_v4_flash.tt.common import _region
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)

_DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731")
_DEFAULT_TEXT = "Tell me the name of the top 10 movies of all time. Also list out the top 10 worst movies of all time. Give me details of why you choose those movies. Try to make your response as humours as possible."
if int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "4096")) < 10:
    _DEFAULT_TEXT = "I"
_WEIGHT_DTYPE = ttnn.bfloat4_b
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")
_PAGE_BLOCK_SIZE = 32


def _pad_to_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _w(loader: DeepseekV4WeightLoader, name: str):
    """Lazy (dequantized) fetch -> thunk (a populated tile cache skips the read)."""
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _build_rope(config, max_seq: int) -> dict:
    """YaRN RoPE tables (cos/sin halves) spanning ``max_seq`` for every layer family.

    ``win[cr]`` holds one windowed table per distinct compress-rate (CSA / HCA
    layers); decode slices the rows it needs from the max-length tables.
    """
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    dummy = torch.zeros(1, max_seq, 1, dtype=torch.float32)
    rotary = M.DeepseekV4RotaryEmbedding(config).to(torch.float32)

    def half(layer_type: str, position_ids: torch.Tensor):
        cos, sin = rotary(dummy, position_ids=position_ids, layer_type=layer_type)
        return cos[0].contiguous(), sin[0].contiguous()

    positions = torch.arange(max_seq).unsqueeze(0)
    rope = {
        "main": half("main", positions),
        "compress": half("compress", positions),
        "win": {},
    }
    for cr in sorted({int(v) for v in config.compress_rates.values()}):
        win_pos = (torch.arange(max_seq // cr) * cr).unsqueeze(0)
        rope["win"][cr] = half("compress", win_pos)
    return rope


def _first_mesh_copy(tensor: ttnn.Tensor, device: ttnn.MeshDevice) -> torch.Tensor:
    """Read rank zero from a replicated mesh tensor."""
    copies = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    return copies[: tensor.shape[0]]


def _construct_model(
    mesh_device,
    prefetcher: contextlib.ExitStack,
    *,
    tp_size: int = 1,
    system_config=None,
    use_prefetcher=None,
    loader=None,
    config=None,
):
    """Build ``DeepSeekV4Model`` + ``lm_head`` and attach the prefetcher session.

    Shared by the single-user decode demo and the multi-user paged demo so the two
    tests cannot drift on construction (mesh profile, TP layout, weight cache, DRISC
    session). ``loader`` / ``config`` are reused when the caller already opened them
    to tokenize; otherwise they are created here.
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    if loader is None:
        loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    if config is None:
        config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
        config._attn_implementation = "eager"

    max_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    top_cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
    model = DeepSeekV4Model(
        config,
        loader,
        mesh_device,
        cache=top_cache,
        weight_dtype=_WEIGHT_DTYPE,
        max_layers=max_layers,
        use_submeshes=True,
        system_config=system_config,
        use_prefetcher=use_prefetcher,
        tp_size=tp_size,
    )
    lm_head = Linear(
        _w(loader, "lm_head.weight"),
        model.last_device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )
    logger.info(f"built DeepSeekV4Model with {model.num_layers}/{config.num_hidden_layers} layers")
    # One prefetcher session for the whole run rather than one per step: starting the DRISC
    # senders is not free, and each GCB's ring state carries from one step to the next. The
    # caller owns the stack so the session also covers the generation loop. A no-op when the
    # model was built without the prefetcher.
    prefetcher.enter_context(model.prefetcher_session())
    # Registered after the session so it unwinds first (LIFO): stopping the traced-decode
    # replay thread releases the model before the DRISC senders stop. Without it the
    # thread's closure keeps the model -- and every ttnn tensor in it -- alive until
    # interpreter shutdown, where nanobind reports the whole graph as leaked.
    prefetcher.callback(model.shutdown)
    logger.info(f"tensor prefetcher: {'on' if model.use_prefetcher else 'off'}")
    return model, lm_head, loader, config


def _assert_decode_parallelism(model: DeepSeekV4Model, tp_size: int) -> None:
    """The TP / pipeline layout both decode demos pin."""
    assert model.tp_size == tp_size
    assert model.num_submeshes == (2 if tp_size == 4 else 8)
    assert model.pipeline_devices == model.num_submeshes * tp_size
    assert all(layer.self_attn.tp_size == tp_size for layer in model.layers)
    assert all(layer.mlp.tp_size == tp_size for layer in model.layers)
    assert all(layer.mlp.experts.tp_size == tp_size for layer in model.layers)
    if tp_size > 1:
        attn = model.layers[0].self_attn
        assert attn.qkv_tp_strategy == "replicated", "TP4 keeps q_a and kv replicated"
        assert not attn.q_a_proj.keep_weights_in_l1
        assert not attn.kv_proj.keep_weights_in_l1
        assert not attn.q_a_proj.partial_width_sharded
        assert not attn.kv_proj.partial_width_sharded
        assert attn.kv_proj.num_inputB_cores == 16
        assert attn.q_b_proj.N == attn.num_heads * attn.head_dim // tp_size
    logger.info(
        f"parallelism: {model.num_submeshes} pipeline stages x TP{tp_size} " f"({model.pipeline_devices} chips)"
    )


def _tokenize_chat(tokenizer, text: str) -> list[int]:
    """Chat-template a single user turn the way both decode demos do."""
    prompt = encode_messages([{"role": "user", "content": text}], "chat")
    # ``encode_messages`` includes DeepSeek's required BOS token explicitly.
    return list(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _traced_max_seq(config, needed: int) -> int:
    """Round ``needed`` the way the single-user demo does, honoring ``DEEPSEEK_V4_MAX_SEQ``.

    The fixed compressor buffers tile cleanly into windows only if the capacity is a
    multiple of every compress-rate.
    """
    max_seq = _pad_to_tile(needed)
    max_seq = max(int(os.environ.get("DEEPSEEK_V4_MAX_SEQ", max_seq)), max_seq)
    crs = {int(v) for v in config.compress_rates.values()}
    step = math.lcm(32, *crs) if crs else 32
    return ((max_seq + step - 1) // step) * step


def _build_and_prefill(
    mesh_device,
    text: str,
    prefetcher: contextlib.ExitStack,
    system_config=None,
    *,
    tp_size: int = 1,
):
    """Build the full ttnn model, prepare the static traced-decode buffers, and
    prefill ``text`` one token at a time. Returns the populated state shared by
    the decode demo and the max-perf measurement tests.

    ``system_config`` pins the tuning profile; ``None`` lets the model pick the one
    matching the open mesh's device count."""
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    max_new_tokens = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "2560"))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id
    start_pos = int(os.environ.get("DEEPSEEK_V4_START_POS", "0"))
    if start_pos < 0:
        raise ValueError(f"DEEPSEEK_V4_START_POS must be nonnegative, got {start_pos}")

    # Wrap the user input in the V4 chat template, tokenize, and build the RoPE
    # tables for the longest sequence we might decode (prompt + new tokens).
    # ``DEEPSEEK_V4_TRACED_DECODE``: replay one captured ttnn trace per submesh per
    # step (fixed-size in-place caches) instead of the host-bound eager decode.
    traced = os.environ.get("DEEPSEEK_V4_TRACED_DECODE", "1") not in ("0", "", "false", "False")

    prompt_ids: list[int] = _tokenize_chat(tokenizer, text)
    real_len = len(prompt_ids)
    needed = start_pos + real_len + max_new_tokens
    if traced:
        max_seq = round_context(
            _traced_max_seq(config, needed),
            set(config.compress_rates.values()),
            _PAGE_BLOCK_SIZE,
        )
    else:
        max_seq = _pad_to_tile(needed)
        max_seq = max(int(os.environ.get("DEEPSEEK_V4_MAX_SEQ", max_seq)), max_seq)
    rope = _build_rope(config, max_seq)

    model, lm_head, loader, config = _construct_model(
        mesh_device,
        prefetcher,
        tp_size=tp_size,
        system_config=system_config,
        use_prefetcher=None,
        loader=loader,
        config=config,
    )

    # --- prefill the prompt by replaying decode one token at a time --------- #
    # There is no dedicated prefill: each prompt token is fed at its absolute
    # position through the (eager or traced) decode path, filling the in-place
    # caches exactly as a full-sequence prefill would. The logits after the final
    # prompt token give the first generated token. Traced decode allocates a
    # one-session paged KV pool here (lm_head folded into the last submesh's
    # trace); the first prefill step captures the traces.
    if traced:
        model.prepare_static_decode(
            rope,
            max_seq,
            lm_head=lm_head,
            num_sessions=1,
            total_tokens=max_seq,
            block_size=_PAGE_BLOCK_SIZE,
        )
        model.activate_session(model.open_session())
        logger.info(
            f"traced decode: paged KV (1 session, {_PAGE_BLOCK_SIZE}-row blocks); "
            "trace captured on first prefill step"
        )
    else:
        model.reset_caches(max_seq)

    next_id = pad_id
    for prompt_idx in range(real_len):
        pos = start_pos + prompt_idx
        if traced:
            # [1, 1, vocab], lm_head in-trace and read back off the D2H socket
            logits = model.decode_traced(prompt_ids[prompt_idx], pos).reshape(1, -1).float()
        else:
            hidden = model.decode(prompt_ids[prompt_idx], pos, rope)  # [1, 1, D]
            with _region("LM_HEAD"):
                logits = _first_mesh_copy(lm_head(hidden), model.last_device).reshape(1, -1).float()
        next_id = int(logits[0].argmax().item())
    logger.info(f"prefill ({real_len} tokens at pos {start_pos}) -> token id {next_id} {tokenizer.decode([next_id])!r}")
    if traced:
        logger.info(f"pool usage after prefill: {model.session_usage()}")

    return {
        "model": model,
        "lm_head": lm_head,
        "tokenizer": tokenizer,
        "config": config,
        "rope": rope,
        "prompt_ids": prompt_ids,
        "real_len": real_len,
        "start_pos": start_pos,
        "max_seq": max_seq,
        "max_new_tokens": max_new_tokens,
        "eos_id": config.eos_token_id,
        "next_id": next_id,
        "traced": traced,
        "tp_size": tp_size,
    }


class _InterruptFlag:
    """Record SIGINT instead of raising it, for the duration of the generation loop.

    The default handler raises ``KeyboardInterrupt`` at the first bytecode after the
    signal arrives. For a Ctrl-C during a decode step that bytecode is on the return
    path of the blocking D2H socket read: the step's output has already been taken off
    the socket by then, but the loop's ``read`` counter has not advanced, so the unwind
    at the end of the test would ask the socket for one output more than any trace will
    send and park in ``read_decoded_output`` forever -- with the sockets in a byte state
    a further Ctrl-C cannot break (that read spins in C).

    Recording the signal lets the step finish (packet, output and both counters), and
    the loop raises ``KeyboardInterrupt`` itself at the top of the next iteration, where
    the counters and the socket agree. ``retire_replay`` runs before :meth:`restore`, so
    the unwind itself cannot be cut in half either; the posted window bounds it.
    """

    def __init__(self) -> None:
        self.hit = False
        self._previous = None
        self._installed = False

    def install(self) -> None:
        self._previous = signal.signal(signal.SIGINT, self._record)
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self._installed = False
            signal.signal(signal.SIGINT, self._previous)

    def _record(self, signum, frame) -> None:
        self.hit = True


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)  # heavy: bf4 conversion of every expert + many decode steps
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2}],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize(
    "mesh_device,tp_size",
    [
        pytest.param((8, 1), 1, id="tp1_8chip"),
        # TP4 opens the mesh directly in the 1x4-stage shape so no ``mesh.reshape``
        # runs — on an 8-chip P150 host, reshaping (8, 1) -> (2, 4) has been seen
        # to leave submesh 1 with a downgraded per-device compute grid, which
        # breaks the single-user hyperconnection's width-sharded layout.
        pytest.param((2, 4), 4, id="tp4_8chip"),
        pytest.param((8, 4), 4, id="tp4_32chip"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_full_model_decode_demo(mesh_device, reset_seeds, text: str, tp_size: int) -> None:
    import time

    # The prefetcher session spans prefill and generation both, so it is opened inside
    # ``_build_and_prefill`` (once the model exists) against this stack.
    with contextlib.ExitStack() as prefetcher:
        state = _build_and_prefill(mesh_device, text, prefetcher, tp_size=tp_size)
        model, lm_head, tokenizer = state["model"], state["lm_head"], state["tokenizer"]
        rope, prompt_ids, real_len = state["rope"], state["prompt_ids"], state["real_len"]
        start_pos = state["start_pos"]
        max_seq, max_new_tokens, eos_id = state["max_seq"], state["max_new_tokens"], state["eos_id"]
        traced, next_id = state["traced"], state["next_id"]
        generated: list[int] = [next_id]
        _assert_decode_parallelism(model, tp_size)
        if traced:
            assert model.paged, "traced decode must use the paged KV layout"

        # Each step feeds the previously generated token at its absolute position and
        # reads back the single-token logits (no recompute over the prior context).
        decode_tokens = 0
        decode_time = 0.0
        logger.info(f"max_new_tokens: {max_new_tokens}")
        n_ahead = min(max_new_tokens, max(0, max_seq - (start_pos + real_len)))
        gen_positions = [start_pos + real_len + step - 1 for step in range(1, n_ahead + 1)]
        if traced and gen_positions:
            # Grow the paged pool for the whole generation before any replay: page
            # tables are device tensors the traces read, so rewriting them from the
            # replay thread while earlier steps are in flight would race.
            model.ensure_session_capacity(gen_positions[-1])

        # Traces are posted in a bounded window rather than for the whole generation
        # (the CLI's ``_DECODE_REPLAY_AHEAD``). A posted ``execute_trace`` has to be
        # handed one H2D packet and have its full output read off the D2H socket before
        # the pipeline can unwind, so the window is also the most an interrupt, an EOS
        # or an error can leave to drain. Posting every position up front would make
        # Ctrl-C retire the whole rest of the reply instead of returning.
        replay_ahead = 32
        posted = 0  # gen_positions handed to replay_traced_ahead
        fed = 0  # packets written to the H2D socket
        read = 0  # outputs read back off the D2H socket

        def refill_replay(step: int) -> None:
            """Keep one window of traces posted: the current step and the ones after."""
            nonlocal posted
            if not traced:
                return
            want = min(len(gen_positions), step - 1 + replay_ahead)
            if want > posted:
                model.replay_traced_ahead(gen_positions[posted:want])
                posted = want

        def retire_replay() -> None:
            """Feed and read every posted-but-unretired step, one step in flight.

            ``read`` counts outputs already taken off the D2H socket and ``fed`` the
            packets already pushed, so a step interrupted between its write and its read
            is neither re-fed nor skipped. Interleaving the two matters: the D2H FIFO is
            one page deep, so feeding the whole window before reading any of it would
            park the sender kernel and wedge the queue.
            """
            nonlocal fed, read
            if not traced:
                return
            dummy = next_id if next_id is not None else 0
            for i in range(read, posted):
                if i >= fed:
                    model.write_step_packet(dummy, gen_positions[i])
                    fed += 1
                model.read_decoded_output()
                read += 1

        # Ctrl-C is recorded rather than raised while this loop runs (see
        # ``_InterruptFlag``): raised on the return path of the blocking D2H read it
        # would leave ``read`` one step behind the socket, and the retire below would
        # then block forever asking for an output no trace will send.
        interrupt = _InterruptFlag()
        interrupt.install()
        try:
            for step, pos in enumerate(gen_positions, start=1):
                if interrupt.hit:
                    # Safe point: the step before this one is written, read and counted,
                    # so the socket and the counters agree for the retire below.
                    logger.info("interrupted; unwinding the posted replays")
                    raise KeyboardInterrupt
                if next_id == eos_id:
                    logger.info("hit EOS; stopping")
                    break
                # Top the window up before feeding, so the device is already parked on
                # in-trace recv for this step and the next ones.
                refill_replay(step)
                t0 = time.perf_counter()
                if traced:
                    model.write_step_packet(next_id, pos)
                    fed += 1
                    logits = model.read_decoded_output().reshape(1, -1).float()
                    read += 1
                else:
                    hidden = model.decode(next_id, pos, rope)  # [1, 1, D]
                    with _region("LM_HEAD"):
                        logits = (
                            _first_mesh_copy(lm_head(hidden), model.last_device).reshape(1, -1).float()
                        )  # forces device sync
                next_id = int(logits[0].argmax().item())
                decode_time += time.perf_counter() - t0
                decode_tokens += 1
                generated.append(next_id)
                logger.info(f"step {step:3d} (pos {pos:4d}): token id {next_id} {tokenizer.decode([next_id])!r}")

                # Running decode throughput, reported every 10 generated tokens.
                if decode_tokens % 64 == 0:
                    logger.info(
                        f"decode throughput: {decode_tokens / decode_time:.2f} tok/s "
                        f"({decode_tokens} tokens in {decode_time:.2f}s)"
                    )
                    decode_tokens = 0
                    decode_time = 0.0
        finally:
            # Retire before restoring the handler, so the unwind cannot be cut in half
            # by a second Ctrl-C. An early EOS (or an error) leaves the posted
            # execute_traces parked on in-trace recv; dummy-feed and read exactly those
            # so the sockets and the replay thread can unwind. Discard the logits --
            # they are not part of the reply.
            try:
                retire_replay()
            finally:
                interrupt.restore()

    if decode_tokens:
        logger.info(
            f"decode throughput (final): {decode_tokens / decode_time:.2f} tok/s "
            f"({decode_tokens} tokens in {decode_time:.2f}s)"
        )

    assert generated, "no tokens were generated"
    logger.info(f"PROMPT    : {tokenizer.decode(prompt_ids)!r}")
    logger.info(f"GENERATED : {tokenizer.decode(generated)!r}  ({len(generated)} tokens)")
    if traced:
        logger.info(f"pool usage after generation: {model.session_usage()}")
