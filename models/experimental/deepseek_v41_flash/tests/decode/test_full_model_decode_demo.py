# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode demo on the full ttnn ``DeepSeekV41Model``.

Builds the whole model once via :class:`DeepSeekV41Model`. There is no dedicated
prefill: a chat prompt is "prefilled" by replaying the decode step once per
prompt token (at ascending absolute positions), seeding every layer's sliding
ring and the shared compressed KV in place, then generation continues one token
per step (``S = 1``) against that cache. The RoPE tables are produced once for
the maximum length; each decode step rebuilds its row on device from the position.

The deployment opens the full 8x4 mesh and carves ``DEEPSEEK_V4_NUM_STAGES`` (default eight) 1x4 stages from it.

Attention uses q_a/kv, replicated full-width on every rank of a stage; head-sharded
SDPA, batched local-group O_A and row-parallel O_B. MoE shards the intermediate
dimension and all-reduces its output. The DRISC prefetcher streams through four
rings per stage device: the shared 64-receiver ring (q_b, o_a, o_b, shared-expert
down), q_a's 32-receiver ring, kv's 16-receiver ring and the router gate's
8-receiver ring. The hyper-connection ``fn`` (large-K), the V4.1 compressor and
indexer, and the per-rank shared-expert gate/up copy DRAM -> L1.

The final collapse uses the last layer's FFN ``pre_mix``; there is no ``hc_head``.
Engram layers 1 and 14 read host n-gram tables (~101 GB each) and skip the test
when that RAM is not free.

All attention, routed-expert and ``lm_head`` weights live on device in ``bfloat4_b``.
Set ``DEEPSEEK_V4_CACHE_DIR`` to reuse the converted ttnn weight tiles across runs,
and optionally cap the stack with ``DEEPSEEK_V4_DECODE_LAYERS=N`` for bring-up.

Run it (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    DEEPSEEK_V4_MAX_NEW_TOKENS=16 pytest -s \\
      models/experimental/deepseek_v41_flash/tests/decode/test_full_model_decode_demo.py

Set ``DEEPSEEK_V4_START_POS`` to begin prefill at a non-zero absolute position,
for example ``DEEPSEEK_V4_START_POS=1024``. Engram hashes then have no earlier
tokens, so a non-zero start only matches a run that never built an Engram layer.

Set ``DEEPSEEK_V4_MEMORY_REPORTS=1`` to write each program's L1 / DRAM block map as it compiles
(``ttnn.device.EnableMemoryReports``), for tracking down circular-buffer / L1 clashes.

``DEEPSEEK_V4_TEMPERATURE`` (default 0, greedy) samples each token at that temperature, on the host from the
step's logits, or with ``DEEPSEEK_V4_DEVICE_SAMPLING=1`` on the output stage, which then sends only the token
(:class:`~models.experimental.deepseek_v41_flash.tt.lm_head.DeepSeekV41DeviceSampler`).
``DEEPSEEK_V4_SEED`` (default 1234) seeds either sampler.
"""

from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.reference import load_reference
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    dequantized,
    load_config,
    pipeline_stage_count,
)
from models.experimental.deepseek_v41_flash.tt.decode.decode_prefetch import ENGRAM_WKV_GCB, O_B_GCB, SHARED_DOWN_GCB
from models.experimental.deepseek_v41_flash.tt.engram_lookup import engram_ram_needed
from models.experimental.deepseek_v41_flash.tt.lm_head import DeepSeekV41LMHead
from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model
from models.experimental.deepseek_v4_flash.encoding_dsv4 import encode_messages
from models.experimental.deepseek_v4_flash.tt.common import _profile
from models.experimental.deepseek_v4_flash.tt.decode.attention import CSA_INDEX_BLOCK_SIZE
from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import (
    DECODE_GCB_GROUP,
    KV_GCB,
    Q_A_GCB,
)
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)

# The mesh stays (8, 4). DEEPSEEK_V4_NUM_STAGES (default 8) is how many 1x4 submeshes the model creates.
_NUM_STAGES = pipeline_stage_count()
# Chat-templated this is exactly 128 tokens (= the sliding window), so the same prompt can be
# fed through a prefill model or replayed through decode.
_DEFAULT_TEXT = (
    "Tell me the name of the top 10 movies of all time. Also list out the top 10 worst movies of all time. "
    "Give me details of why you choose those movies. Try to make your response as humours as possible. "
    "For context, I am putting together a light-hearted movie night for my friends, who love classic dramas, "
    "science fiction, and over-the-top comedies alike. Keep every entry short, mention the director and the "
    "release year, and end with a one-line joke about each film. Please format the answer as two clearly "
    "numbered lists, one for the best and one for the worst."
)
if int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "4096")) < 10:
    _DEFAULT_TEXT = "I"
_WEIGHT_DTYPE = ttnn.bfloat4_b
# Accepted by prepare_static_decode for the V4 call shape. The index pool uses CSA_INDEX_BLOCK_SIZE.
_PAGE_BLOCK_SIZE = 32
_ENGRAM_RAM_MARGIN = 1.25
_DEVICE_SAMPLING = os.environ.get("DEEPSEEK_V4_DEVICE_SAMPLING", "0") not in ("0", "", "false", "False")
_TEMPERATURE = float(os.environ.get("DEEPSEEK_V4_TEMPERATURE", "0"))
_SEED = int(os.environ.get("DEEPSEEK_V4_SEED", "1234"))


def _pad_to_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


def _next_token(out: torch.Tensor, generator: torch.Generator) -> int:
    """The token of one step's :meth:`read_decoded_output`: the device's pick (int32, token first), or from the
    logits on the host -- argmax, or a draw at ``_TEMPERATURE``."""
    if out.dtype == torch.int32:
        return int(out.reshape(-1)[0])
    logits = out.reshape(-1).float()
    if _TEMPERATURE == 0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits / _TEMPERATURE, dim=-1), 1, generator=generator))


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _available_bytes() -> int:
    with open("/proc/meminfo") as meminfo:
        return next(int(line.split()[1]) * 1024 for line in meminfo if line.startswith("MemAvailable:"))


def _build_rope(config, snapshot_dir, max_seq: int) -> dict:
    """``(cos_half, sin_half)`` ``[max_seq, Rd/2]`` for the sliding and compressed RoPE families.

    Sliding attention disables YaRN. Compressed KV uses the checkpoint's YaRN settings.
    """
    ref = load_reference(snapshot_dir)
    scaling = config.rope_scaling
    dim = config.qk_rope_head_dim

    def tables(original_seq_len: int, base: float):
        freqs = ref.precompute_freqs_cis(
            dim,
            max_seq,
            original_seq_len,
            base,
            scaling["factor"],
            scaling["beta_fast"],
            scaling["beta_slow"],
        )
        return freqs.real.contiguous(), freqs.imag.contiguous()

    return {
        "main": tables(0, config.rope_theta),
        "compress": tables(scaling["original_max_position_embeddings"], config.compress_rope_theta),
        "win": {},
    }


def _construct_model(
    mesh_device,
    prefetcher: contextlib.ExitStack,
    *,
    tp_size: int = 4,
    system_config=None,
    loader=None,
    config=None,
    num_stages: int | None = None,
    submeshes=None,
    use_prefetcher=None,
    layer_submesh_ids=None,
    verify_rows: int = 1,
):
    """Build ``DeepSeekV41Model`` + ``lm_head`` and attach the prefetcher session (``verify_rows``: see the model)."""
    if loader is None:
        loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    if config is None:
        config = load_config(loader.snapshot_dir)

    max_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    top_cache = WeightCache(CACHE_DIR) if CACHE_DIR else None
    model = DeepSeekV41Model(
        config,
        loader,
        mesh_device,
        cache=top_cache,
        weight_dtype=_WEIGHT_DTYPE,
        max_layers=max_layers,
        use_submeshes=True,
        system_config=system_config,
        tp_size=tp_size,
        num_stages=num_stages,
        submeshes=submeshes,
        use_prefetcher=use_prefetcher,
        layer_submesh_ids=layer_submesh_ids,
        verify_rows=verify_rows,
    )
    lm_head = DeepSeekV41LMHead(
        dequantized(loader, "lm_head.weight", translate=True),
        model.last_device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )
    logger.info(f"built DeepSeekV41Model with {model.num_layers}/{config.num_hidden_layers} layers")
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


def _assert_decode_parallelism(model: DeepSeekV41Model, tp_size: int, num_stages: int | None = None) -> None:
    """The TP / pipeline layout this demo pins."""
    if num_stages is None:
        num_stages = pipeline_stage_count()
    assert model.tp_size == tp_size
    assert model.num_submeshes == num_stages
    assert model.pipeline_devices == model.num_submeshes * tp_size
    assert all(layer.self_attn.tp_size == tp_size for layer in model.layers)
    assert all(layer.mlp.tp_size == tp_size for layer in model.layers)
    assert all(layer.mlp.experts.tp_size == tp_size for layer in model.layers)
    attn = model.layers[0].self_attn
    assert attn.qkv_tp_strategy == "replicated", "TP4 keeps q_a and kv replicated"
    assert not attn.q_a_proj.partial_width_sharded
    assert not attn.kv_proj.partial_width_sharded
    assert attn.kv_proj.num_inputB_cores == 16
    assert attn.q_b_proj.N == attn.num_heads * attn.head_dim // tp_size
    logger.info(f"parallelism: {model.num_submeshes} pipeline stages x TP{tp_size} ({model.pipeline_devices} chips)")


def _assert_prefetch_rings(model: DeepSeekV41Model) -> None:
    """The GCB set the prefetched decode streams through, pinned per stage device.

    Each device holds at most the shared ring plus the q_a, kv, o_b and shared-down rings, and
    a device with an Engram layer also its wkv ring -- six, all of the DRISC senders' ~six
    state slots. The hyper-connection ``fn`` reads its weight from L1, so it is on no ring.
    """
    if not model.use_prefetcher:
        return
    allowed = set(DECODE_GCB_GROUP) | {Q_A_GCB, KV_GCB, O_B_GCB, SHARED_DOWN_GCB, ENGRAM_WKV_GCB}
    for _, buffers in model._prefetch_buffers_by_device.values():
        assert set(buffers) <= allowed, f"unexpected prefetch rings {sorted(set(buffers) - allowed)}"
        rings = {id(gcb) for gcb in buffers.values()}
        limit = 6 if ENGRAM_WKV_GCB in buffers else 5
        assert len(rings) <= limit, f"{len(rings)} GCBs on one device, expected at most {limit}"
    for layer in model.layers:
        q_a_ring = layer.self_attn.q_a_proj.global_cb
        assert q_a_ring is not None, "q_a must stream through the prefetcher"
        assert not hasattr(layer.attn_hc.fn, "global_cb"), "attention hyper-connection fn must not use a GCB"
        assert not hasattr(layer.ffn_hc.fn, "global_cb"), "FFN hyper-connection fn must not use a GCB"
        proj = getattr(layer.mlp.gate, "gate", None)
        if proj is not None and hasattr(proj, "dram_sharded_weight"):
            assert (
                not proj.use_prefetcher and proj.dram_sharded_weight
            ), "the router gate has no ring and must read its DRAM ND-sharded weight inside matmul_decode"
        if model.tp_size > 1 and layer.mlp.shared_experts is not None:
            shared = layer.mlp.shared_experts
            assert (
                not shared.gate_proj.use_prefetcher and not shared.up_proj.use_prefetcher
            ), "the per-rank shared-expert gate/up cut matches no ring"
            assert (
                shared.gate_proj.dram_sharded_weight and shared.up_proj.dram_sharded_weight
            ), "shared-expert gate/up must read their DRAM ND-sharded weights inside matmul_decode"
    logger.info(
        f"prefetch rings: {len(model._prefetch_buffers_by_device)} device(s), hyper-connection fn off the rings"
    )


def _tokenize_chat(tokenizer, text: str, thinking_mode: str = "chat", reasoning_effort: str | None = None) -> list[int]:
    """Chat-template a single user turn the way the V4 decode demo does."""
    prompt = encode_messages([{"role": "user", "content": text}], thinking_mode, reasoning_effort=reasoning_effort)
    # ``encode_messages`` includes DeepSeek's required BOS token explicitly.
    return list(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _traced_max_seq(config, needed: int, num_layers: int) -> int:
    """Round ``needed`` up so every compressed group tiles into index blocks.

    ``DEEPSEEK_V4_MAX_SEQ`` raises the floor.
    """
    max_seq = _pad_to_tile(needed)
    max_seq = max(int(os.environ.get("DEEPSEEK_V4_MAX_SEQ", max_seq)), max_seq)
    ratios = {int(r) for r in config.compress_ratios[:num_layers] if int(r)}
    if not ratios:
        return max_seq
    return round_context(max_seq, ratios, CSA_INDEX_BLOCK_SIZE)


def _build_and_prefill(
    mesh_device,
    text: str,
    prefetcher: contextlib.ExitStack,
    system_config=None,
    *,
    tp_size: int = 4,
    num_stages: int | None = None,
):
    """Build the full ttnn model, prepare the static traced-decode buffers, and
    prefill ``text`` one token at a time."""
    from transformers import AutoTokenizer

    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    max_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    engram_ids = [i for i in config.engram_layer_ids if i < max_layers]
    if engram_ids:
        nbytes = engram_ram_needed(loader, engram_ids)
        if _available_bytes() < _ENGRAM_RAM_MARGIN * nbytes:
            pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables of layers {engram_ids}")

    max_new_tokens = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "2560"))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    start_pos = int(os.environ.get("DEEPSEEK_V4_START_POS", "0"))
    if start_pos < 0:
        raise ValueError(f"DEEPSEEK_V4_START_POS must be nonnegative, got {start_pos}")
    if start_pos and engram_ids:
        logger.warning(
            f"DEEPSEEK_V4_START_POS={start_pos} with Engram layers {engram_ids}: the hash has no tokens before it"
        )

    prompt_ids: list[int] = _tokenize_chat(tokenizer, text)
    real_len = len(prompt_ids)
    needed = start_pos + real_len + max_new_tokens
    max_seq = _traced_max_seq(config, needed, max_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)

    model, lm_head, loader, config = _construct_model(
        mesh_device,
        prefetcher,
        tp_size=tp_size,
        system_config=system_config,
        loader=loader,
        config=config,
        num_stages=num_stages,
    )

    # --- prefill the prompt by replaying decode one token at a time --------- #
    # There is no dedicated prefill: each prompt token is fed at its absolute
    # position through the traced decode path, filling the rings and the shared
    # compressed KV exactly as a full-sequence prefill would. The logits after the
    # final prompt token give the first generated token. lm_head is folded into the
    # last submesh's trace; the first prefill step captures the traces.
    model.prepare_static_decode(
        rope,
        max_seq,
        lm_head=lm_head,
        num_sessions=1,
        total_tokens=max_seq,
        block_size=_PAGE_BLOCK_SIZE,
        device_sampling=_DEVICE_SAMPLING,
    )
    generator = torch.Generator().manual_seed(_SEED)
    if model.device_sampler is not None:
        model.device_sampler.set_sampling(_TEMPERATURE, _SEED)
    logger.info(f"sampling: {'device' if _DEVICE_SAMPLING else 'host'}, temperature {_TEMPERATURE}, seed {_SEED}")
    if model.context_limit is not None and model.context_limit < max_seq:
        logger.warning(f"context capped at {model.context_limit} tokens by the compressed KV")
        max_seq = model.context_limit
    model.activate_session(model.open_session())
    logger.info(
        f"traced decode: shared compressed KV (1 session, {CSA_INDEX_BLOCK_SIZE}-row index blocks); "
        "trace captured on first prefill step"
    )

    next_id = pad_id
    for prompt_idx in range(real_len):
        pos = start_pos + prompt_idx
        # [1, 1, vocab] logits or the device's token, lm_head in-trace and read back off the D2H socket
        next_id = _next_token(model.decode_traced(prompt_ids[prompt_idx], pos), generator)
        # decode_traced returns once the step's output is read, so no replay is parked on a recv here.
        for sm in model.submeshes_io:
            _profile(sm["device"])
    logger.info(f"prefill ({real_len} tokens at pos {start_pos}) -> token id {next_id} {tokenizer.decode([next_id])!r}")
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
        "eos_id": tokenizer.eos_token_id,
        "next_id": next_id,
        "tp_size": tp_size,
        "generator": generator,
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


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)  # heavy: bf4 conversion of every expert + many decode steps
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            # The CCL semaphores and width_sharded_all_reduce's scratch live as long as their cached programs.
            # Without L1_SMALL they land in L1 wherever the first layer's temporaries leave the allocator, and
            # cap the circular buffers of every later program on the stage (fused_experts clashes at 40 layers).
            "l1_small_size": int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "8192")),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize(
    "mesh_device,tp_size,num_stages",
    [
        # TP4 opens the full Galaxy as (8, 4) so no reshape changes the per-device grid.
        # DEEPSEEK_V4_NUM_STAGES (default 8) submeshes are created; the other rows stay idle.
        pytest.param((8, 4), 4, _NUM_STAGES, id=f"tp4_{_NUM_STAGES}stage"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_full_model_decode_demo(mesh_device, reset_seeds, text: str, tp_size: int, num_stages: int) -> None:
    import time

    # Per-program L1 / DRAM block maps, written as each program compiles (before its circular-buffer
    # check), under generated/reports/tt_metal/.
    if os.environ.get("DEEPSEEK_V4_MEMORY_REPORTS"):
        ttnn.device.EnableMemoryReports()

    # The prefetcher session spans prefill and generation both, so it is opened inside
    # ``_build_and_prefill`` (once the model exists) against this stack.
    with contextlib.ExitStack() as prefetcher:
        state = _build_and_prefill(
            mesh_device,
            text,
            prefetcher,
            tp_size=tp_size,
            num_stages=num_stages,
        )
        model, tokenizer = state["model"], state["tokenizer"]
        prompt_ids, real_len = state["prompt_ids"], state["real_len"]
        start_pos = state["start_pos"]
        max_seq, max_new_tokens, eos_id = state["max_seq"], state["max_new_tokens"], state["eos_id"]
        next_id = state["next_id"]
        generated: list[int] = [next_id]
        _assert_decode_parallelism(model, tp_size, num_stages)
        _assert_prefetch_rings(model)
        assert model.paged, "traced decode must use the shared compressed-KV pools"

        # Each step feeds the previously generated token at its absolute position and
        # reads back the single-token logits (no recompute over the prior context).
        decode_tokens = 0
        decode_time = 0.0
        logger.info(f"max_new_tokens: {max_new_tokens}")
        n_ahead = min(max_new_tokens, max(0, max_seq - (start_pos + real_len)))
        gen_positions = [start_pos + real_len + step - 1 for step in range(1, n_ahead + 1)]
        if gen_positions:
            # Grow past the whole generation before any replay: the traces read fixed
            # buffers, and a later ensure must not race a step already in flight.
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
        # Under the device profiler its DRAM buffers have to be read before they fill, and a read syncs every
        # submesh, which deadlocks against replays parked on in-trace recv. So the window stops at each
        # ``replay_ahead`` boundary instead of sliding: once that block is read back, nothing is in flight.
        profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"

        def refill_replay(step: int) -> None:
            """Keep one window of traces posted: the current step and the ones after."""
            nonlocal posted
            want = min(len(gen_positions), step - 1 + replay_ahead)
            if profiling:
                want = min(want, ((step - 1) // replay_ahead + 1) * replay_ahead)
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
                if profiling and step > 1 and (step - 1) % replay_ahead == 0:
                    assert read == posted, f"profiler read with {posted - read} replay(s) still in flight"
                    for sm in model.submeshes_io:
                        _profile(sm["device"])
                # Top the window up before feeding, so the device is already parked on
                # in-trace recv for this step and the next ones.
                refill_replay(step)
                t0 = time.perf_counter()
                model.write_step_packet(next_id, pos)
                fed += 1
                out = model.read_decoded_output()
                read += 1
                next_id = _next_token(out, state["generator"])
                decode_time += time.perf_counter() - t0
                decode_tokens += 1
                generated.append(next_id)

                # Running decode throughput, reported every 64 generated tokens.
                if decode_tokens % 64 == 0:
                    logger.info(
                        f"decode throughput: {decode_tokens / decode_time:.2f} tok/s "
                        f"({decode_tokens} tokens in {decode_time:.2f}s) (pos {pos})"
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
    logger.info(f"pool usage after generation: {model.session_usage()}")
