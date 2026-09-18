# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the Llama-3.1-8B prefill model and its chunk loop (tt-blaze#4148).

Graded on **hidden states**, not logits: this model has no LM head (prefill's product is the
populated KV cache), so the comparison runs embedding -> layers -> final norm against the same span
of the torch reference.

Device cases:

  1. ``tp8-1x8`` — eight chips, TP=8, SP=1. Production TP width and a one-shot prefill. SP=1 means
     the cache-read path is not reachable (a second chunk at SP=1 raises by design — plain causal
     SDPA cannot express a Q offset), so this case is *one* chunk.

  2. ``sp4-4x2`` — 4x2, TP=2 and SP=4: the real chunked prefill. Two chunks, the second reading the
     first out of the block-cyclic cache through the ring-joint SDPA. This is the case that proves
     the chunk loop, and it is graded against a **single-shot** reference run: chunking is only
     correct if it does not change the answer, so the reference deliberately does not chunk.

The model is built with few layers and a small vocab. At the real 32 layers / 128256 vocab the
comparison would be dominated by loading a 16 GB checkpoint, and every dimension that matters for
correctness (head counts, the TP/SP split, the residual layout, the chunk geometry) is unchanged by
either. ``test_real_checkpoint_hidden_states`` covers the real weights and is opt-in.

Run (eight-chip loudbox):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_model_vs_ref.py
"""

from __future__ import annotations

import gc
import getpass
import os
import subprocess
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.reference.model import Llama31Model, build_hf_cos_sin
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig, _quantize_like

EMB_DIM = Llama31_8BConfig.EMB_SIZE
N_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
PCC_REQUIRED = 0.99

# The per-layer KV-vs-golden floor (tt-blaze#4150) is deliberately looser than PCC_REQUIRED, and it
# is a different measurement rather than a weaker one. Everything else here grades a device tensor
# against a same-process torch reference over one or two layers; this grades a bfloat8_b cache,
# written by 32 layers of bf16 compute, against an fp32 CPU golden at 2048 tokens. The error is
# accumulated, not local.
#
# The full-model bar, same 0.99 as every other cell in this file. Measured on a 4x8 galaxy with the
# real checkpoint, deterministically: worst K 0.9990, worst V 0.9959 (layer 28), best 0.99997.
#
# This floor was 0.95 for a while on the theory that the bfloat8_b KV cache set the ceiling. That
# was wrong, and the way it was wrong is worth recording. Round-tripping the fp32 golden through
# bfloat8_b and scoring it against itself puts the cache's own quantisation ceiling at 0.99996 --
# the dtype costs nothing. The real limit was bf16 accumulation in the QKV/O projection matmuls
# (``attention.COMPUTE_KERNEL_CONFIG_PROJECTIONS``): harmless for K, whose outputs run |max| 10-20,
# and severe for V at 0.5-2.5, where a small result from large operands means cancellation and
# cancellation turns accumulator rounding into large relative error. Enabling fp32 accumulation on
# those two matmuls moved worst V from 0.9666 to 0.9959 and worst K from 0.9899 to 0.9990.
#
# The lesson for the next person tempted to lower a floor: the uniformity of the error was the tell.
# It sat at the same level across all 8 KV heads and all token blocks, which is accumulator
# precision, not a wiring fault -- and "the dtype must be the limit" is a hypothesis that is cheap
# to actually test before relaxing a gate.
PCC_REQUIRED_KV_GOLDEN = PCC_REQUIRED

# Small enough to build and load in seconds; see the module docstring for why this is not a
# weakening of the test.
TEST_LAYERS = 2
TEST_VOCAB = 2048


def _reference_state_dict(reference: Llama31Model) -> dict:
    """The reference's parameters under their HuggingFace checkpoint names."""
    own = dict(reference.named_parameters())
    return {hf_key: own[own_key].detach() for hf_key, own_key in reference.hf_key_map().items()}


def _reference_hidden_states(reference: Llama31Model, token_ids: torch.Tensor) -> torch.Tensor:
    """Embedding -> layers -> final norm, i.e. the reference forward stopping short of the LM head.

    Spelled out instead of calling ``reference(...)`` because that returns logits, and the device
    model has no LM head to compare them against.
    """
    x = reference.embed_tokens(token_ids)
    cos, sin = build_hf_cos_sin(torch.arange(token_ids.shape[-1]))
    for layer in reference.layers:
        x, _ = layer(x, cos, sin)
    return reference.norm(x)


def _build(mesh_device, *, chunk_size, max_seq_len, num_layers=TEST_LAYERS, vocab_size=TEST_VOCAB, state_dict=None):
    rows, cols = mesh_device.shape
    config = TtPrefillRuntimeConfig(
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        mesh_shape=(rows, cols),
        num_layers=num_layers,
        num_users=1,
        tp_axis=1,
        vocab_size=vocab_size,
    )
    runtime = TtPrefillRuntime(mesh_device=mesh_device, config=config, state_dict=state_dict)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp_axis=config.sp_axis,
        num_users=1,
        chunk_size=chunk_size,
        num_kv_heads_per_chip=N_KV_HEADS // config.tp_factor,
    )
    return runtime, kv_cache, config


# =====================================================================================
# Device-free
# =====================================================================================
def test_runtime_module_is_import_light():
    """The runtime is what the adapter imports, so it is the one that must stay cheap."""
    forbidden = ("safetensors", "transformers", "models.demos.llama_3p1_8b_d_p.reference.model")
    probe = (
        "import sys;"
        "import models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime;"
        f"print(','.join(m for m in {forbidden!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"tt.tt_prefill_runtime import pulled in {out}"


def test_runtime_config_rejects_bad_chunk_geometry(expect_error):
    """The three chunk-geometry rules are enforced at construction.

    None of them fails loudly at run time: a chunk slab that is not a whole number of DRAM shards,
    or a cache that is not an exact number of chunks, puts tokens in the wrong bank or on the wrong
    chip and surfaces much later as a KV PCC failure at some interior position.
    """
    # chunk_size must be a multiple of sp * 32 -> 4x2 mesh has sp=4, so 4*32=128.
    with expect_error(ValueError, "must be a multiple of"):
        TtPrefillRuntimeConfig(max_seq_len=1024, chunk_size=96, mesh_shape=(4, 2))
    # max_seq_len must be an exact number of chunks.
    with expect_error(ValueError, "must be a multiple of chunk_size"):
        TtPrefillRuntimeConfig(max_seq_len=1000, chunk_size=128, mesh_shape=(4, 2))
    with expect_error(ValueError, "tp_axis"):
        TtPrefillRuntimeConfig(max_seq_len=1024, chunk_size=128, mesh_shape=(4, 2), tp_axis=2)

    # And a valid one derives the geometry it will be driven with.
    config = TtPrefillRuntimeConfig(max_seq_len=1024, chunk_size=256, mesh_shape=(4, 2))
    assert (config.sp_factor, config.tp_factor, config.sp_axis) == (4, 2, 0)
    assert config.chunk_size_local == 64


# =====================================================================================
# Device PCC
# =====================================================================================
@pytest.mark.parametrize(
    "mesh_device, device_params, chunk_size, prompt_len",
    [
        # SP=1: one chunk only. A second chunk at SP=1 raises by design (no chunk-position-aware
        # single-device SDPA), so the prompt is exactly one chunk here.
        pytest.param((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, 256, 256, id="tp8-1x8-one-chunk"),
        # SP=4: the real chunked path. 512 tokens over 256-token chunks = 2 chunks, so chunk 1 reads
        # chunk 0 back out of the block-cyclic cache.
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, 256, 512, id="sp4-4x2-two-chunks"),
        # SP=2: the same chunked path on the smallest mesh that has both a TP and an SP dimension.
        # Worth keeping alongside the 4x2 case for two reasons: it is the cheapest configuration
        # that can regress the block-cyclic cache read at all, and four chips is a QuietBox, so the
        # chunked path stays testable without an eight-chip box.
        pytest.param((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, 256, 512, id="sp2-2x2-two-chunks"),
        # The production geometry, and the only shape a Galaxy will open (see
        # ``galaxy_torus_xy_device_params``): 4x8 is SP=4 x TP=8, i.e. the full 32 chips, which is
        # also ``TtPrefillRuntimeConfig``'s default mesh. Chunked, so the Galaxy exercises the
        # block-cyclic cache read rather than just a single chunk.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), 256, 512, id="galaxy-sp4-tp8-4x8-two-chunks"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_model_vs_ref(mesh_device, device_params, chunk_size, prompt_len, reset_seeds):
    """Prefill model hidden states vs the torch reference, PCC >= 0.99."""
    torch.manual_seed(0)
    max_seq_len = 1024

    reference = Llama31Model(num_layers=TEST_LAYERS, vocab_size=TEST_VOCAB).eval()
    with torch.no_grad():
        # Default-initialised norms are all ones, which makes gamma a no-op and would hide a
        # dropped or misplaced norm weight anywhere in the stack.
        for layer in reference.layers:
            layer.input_layernorm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)
            layer.post_attention_layernorm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)
        reference.norm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)

    token_ids = torch.randint(0, TEST_VOCAB, (1, prompt_len))
    with torch.no_grad():
        torch_hidden = _reference_hidden_states(reference, token_ids)

    runtime, kv_cache, config = _build(
        mesh_device, chunk_size=chunk_size, max_seq_len=max_seq_len, state_dict=_reference_state_dict(reference)
    )
    num_chunks = prompt_len // chunk_size
    logger.info(
        f"mesh={tuple(mesh_device.shape)} tp={config.tp_factor} sp={config.sp_factor} "
        f"prompt={prompt_len} in {num_chunks} chunk(s) of {chunk_size} ({config.chunk_size_local}/chip)"
    )

    tt_hidden = runtime.prefill_prompt(token_ids[0].tolist(), kv_cache, return_hidden_states=True)

    # ``tt_hidden`` is the LAST chunk's hidden states: SP-sharded on the sequence across the mesh
    # rows, and replicated across the TP columns (the final norm all-gathered the hidden dim).
    # Rows concat on the sequence dim; the column concat just stacks tp identical replicas, so keep
    # the first. ConcatMesh2dToTensor only concatenates — it has no "take one replica" mode.
    got = ttnn.to_torch(
        tt_hidden, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(2, -1))
    )
    got = got[..., :EMB_DIM].reshape(1, chunk_size, EMB_DIM).to(torch.float32)

    # The last chunk covers the prompt's last chunk_size positions.
    want = torch_hidden[:, -chunk_size:]
    assert not torch.isnan(got).any(), "NaN in model hidden states"
    assert not torch.isinf(got).any(), "Inf in model hidden states"

    passing, pcc = comp_pcc(want, got, PCC_REQUIRED)
    logger.info(f"model hidden-state PCC ({num_chunks} chunk(s), tp={config.tp_factor}, sp={config.sp_factor}): {pcc}")
    assert passing, f"model PCC {pcc} below {PCC_REQUIRED}"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2")],
    indirect=["mesh_device", "device_params"],
)
def test_layer_completion_sink_fires_once_per_layer_per_chunk(mesh_device, device_params, reset_seeds):
    """The per-layer sink fires once per layer per chunk, with GLOBAL layer indices.

    This is the seam the disaggregated pipeline hangs KV migration off, and it keys on
    ``request_id * num_layers + layer_idx``. Two things about that key are pinned here:

    * the layer index is **global**, not rank-local — otherwise every rank's local layer k lands on
      the same key and all but one rank's completion is dropped as a duplicate;
    * the request id is **constant across chunks**, because a multi-chunk prompt is one request.
      A chunk index here would announce each chunk as a separate request and scatter one request's
      KV across several keys.

    Both failures are silent, which is why they are asserted rather than reasoned about.
    """
    torch.manual_seed(0)
    chunk_size, prompt_len, first_layer_idx = 256, 512, 8

    rows, cols = mesh_device.shape
    config = TtPrefillRuntimeConfig(
        max_seq_len=1024,
        chunk_size=chunk_size,
        mesh_shape=(rows, cols),
        num_layers=TEST_LAYERS,
        first_layer_idx=first_layer_idx,
        vocab_size=TEST_VOCAB,
    )
    runtime = TtPrefillRuntime(mesh_device=mesh_device, config=config)
    kv_cache = allocate_kv_cache(
        mesh_device,
        # Rank-local: this rank runs TEST_LAYERS layers, so it gets TEST_LAYERS slots per user even
        # though its layers are globally 8 and 9. The sink still reports the global indices.
        num_layers=TEST_LAYERS,
        max_seq_len=1024,
        sp_axis=config.sp_axis,
        num_users=1,
        chunk_size=chunk_size,
        num_kv_heads_per_chip=N_KV_HEADS // config.tp_factor,
    )

    seen = []
    runtime.set_layer_completion_sink(lambda layer_idx, request_id: seen.append((request_id, layer_idx)))
    runtime.prefill_prompt(torch.randint(0, TEST_VOCAB, (prompt_len,)).tolist(), kv_cache, request_id=7)

    num_chunks = prompt_len // chunk_size
    expected = [(7, first_layer_idx + offset) for _ in range(num_chunks) for offset in range(TEST_LAYERS)]
    assert seen == expected, f"sink fired {seen}, expected {expected}"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2")],
    indirect=["mesh_device", "device_params"],
)
def test_prefill_chunk_rejects_a_sub_tile_resume(mesh_device, device_params, expect_error, reset_seeds):
    """A continuation must resume on a 32-token tile boundary — but need not be chunk-aligned.

    32 is the real constraint: the KV writer derives its tile offset by dividing the offset by 32,
    and the indexed-RoPE reader and ring SDPA assert the same. A sub-tile offset does not fail in
    the kernel, so it is refused here where the caller's offset is named.
    """
    chunk_size = 256
    runtime, kv_cache, config = _build(mesh_device, chunk_size=chunk_size, max_seq_len=1024)
    tokens = runtime.make_chunk_input([0] * chunk_size)

    with expect_error(ValueError, "must be a multiple of 32"):
        runtime.prefill_chunk(tokens, kv_cache, slot_id=0, actual_start=8, actual_end=8 + chunk_size)
    with expect_error(ValueError, "out of range"):
        runtime.prefill_chunk(tokens, kv_cache, slot_id=3, actual_start=0, actual_end=chunk_size)
    with expect_error(ValueError, "past the per-user cache"):
        runtime.prefill_chunk(tokens, kv_cache, slot_id=0, actual_start=1024, actual_end=1024 + chunk_size)
    ttnn.deallocate(tokens)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_chunk_input_matches_the_engine_h2d_geometry(mesh_device, device_params, reset_seeds):
    """What ``make_chunk_input`` builds is spec-identical to what the engine's H2D socket delivers.

    ``prefill_chunk`` consumes the socket payload as given, so the two producers have to agree on
    shape, dtype, layout and sharding. They are written independently — ``make_global_spec`` and
    ``H2D_MAPPER_CONFIG`` live in the shared runner, this one in the model — and nothing else
    compares them: a mismatch would not fail here, it would fail only under a live socket, which is
    the most expensive place to find it.

    (The issue asks for this against ``MockPrefillRunner``. No such symbol exists in this repo —
    the engine's device-free half is ``prefill_producer.py`` driving a real ``prefill_runner.py``
    over a socket — so the geometry those two agree on is what gets pinned instead.)
    """
    from models.demos.common.prefill.runners.prefill_runner import H2D_MAPPER_CONFIG
    from models.demos.common.prefill.runners.runner_utils import make_global_spec

    chunk_size = 256
    runtime, _kv_cache, config = _build(mesh_device, chunk_size=chunk_size, max_seq_len=1024)
    tokens = runtime.make_chunk_input(list(range(chunk_size)))

    # make_global_spec is the *global* geometry (sp, 1, chunk/sp); a sharded tensor reports its
    # per-chip shape. They agree exactly when sharding dim 0 across the sp rows turns the former
    # into the latter, which is the contract being pinned.
    want = make_global_spec(tuple(mesh_device.shape), chunk_size)
    sp = config.sp_factor
    want_global = tuple(want.shape)
    assert want_global == (sp, 1, chunk_size // sp), f"engine spec {want_global} is not sp-major"
    assert tuple(tokens.shape) == (
        1,
        1,
        chunk_size // sp,
    ), f"per-chip shape {tuple(tokens.shape)} is not the sp-shard of the engine's {want_global}"
    assert tokens.dtype == want.dtype == ttnn.uint32
    assert tokens.layout == want.layout == ttnn.ROW_MAJOR_LAYOUT

    # The engine shards dim 0 over the first mesh axis and replicates over the second; this model
    # must put its SP axis in the same place or every chip gets the wrong tokens.
    placements = H2D_MAPPER_CONFIG.placements
    assert config.sp_axis == 0, (
        f"the engine's H2D mapper shards dim 0 on mesh axis 0 ({placements}), so sp_axis must be 0, "
        f"got {config.sp_axis}"
    )
    ttnn.deallocate(tokens)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_non_chunk_aligned_continuation_matches_the_reference(mesh_device, device_params, reset_seeds):
    """#4148's non-identity case: a continuation resuming off a chunk boundary.

    The trap this covers: the block-cyclic map that deals a chunk's tokens to the SP rows is the
    plain contiguous split **exactly** when ``actual_start % chunk_size == 0``. Every single-turn
    request starts at 0, so a runtime that assumes prompt order passes the entire suite and then
    misplaces every token of the first multi-turn continuation — which resumes at
    ``aligned_resume_length``, a multiple of 32 and only incidentally of ``chunk_size``.

    ``first_len`` is chosen so the resume offset is 32-aligned and *not* chunk-aligned, and so that
    it does not land on a ``chunk_size / sp`` boundary either — that sub-case is a pure rotation,
    while an offset inside a chip's own slab additionally rotates the boundary chip, which is the
    half a naive rotation gets wrong.

    Graded on the cache rather than the hidden states because the cache is what prefill produces:
    misplaced tokens can leave the last chunk's hidden states plausible while the cache decode
    inherits is scrambled.
    """
    from models.demos.llama_3p1_8b_d_p.reference.model import to_meta_frame
    from models.demos.llama_3p1_8b_d_p.tt.kv_cache import aligned_resume_length

    torch.manual_seed(0)
    chunk_size, max_seq_len = 256, 1024
    sp = mesh_device.shape[0]
    first_len = 160  # 32-aligned; not a multiple of chunk_size (256) nor of chunk_size/sp (64)
    resume = aligned_resume_length(first_len)
    assert (
        resume % 32 == 0 and resume % chunk_size and resume % (chunk_size // sp)
    ), f"resume={resume} does not exercise the rotated branch at sp={sp}"
    total = resume + chunk_size

    reference = Llama31Model(num_layers=TEST_LAYERS, vocab_size=TEST_VOCAB).eval()
    with torch.no_grad():
        # Default norms are all ones, which would hide a dropped gamma anywhere in the stack.
        for layer in reference.layers:
            layer.input_layernorm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)
            layer.post_attention_layernorm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)
        reference.norm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)

    token_ids = torch.randint(0, TEST_VOCAB, (1, total))

    # Reference KV for the whole sequence in one shot: the device has to reproduce this regardless
    # of how the prompt was split into turns.
    with torch.no_grad():
        _, past_kvs = reference(token_ids, start_pos=0, past_kvs=None, return_kv=True)

    runtime, kv_cache, config = _build(
        mesh_device,
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        state_dict=_reference_state_dict(reference),
    )

    # Turn 1, then the continuation from a non-chunk-aligned prefix.
    runtime.prefill_prompt(token_ids[0, :first_len].tolist(), kv_cache, start_pos=0)
    runtime.prefill_prompt(token_ids[0, resume:total].tolist(), kv_cache, start_pos=resume)

    got_k, got_v = runtime.read_slot_kv(kv_cache, 0, num_tokens=total)

    pccs = []
    for layer, (ref_k, ref_v) in enumerate(past_kvs):
        # The device stores K post-RoPE in the Meta-interleaved frame; the reference keeps HF.
        want_k = _quantize_like(to_meta_frame(ref_k)[0, :, :total], ttnn.bfloat8_b, mesh_device)
        want_v = _quantize_like(ref_v[0, :, :total], ttnn.bfloat8_b, mesh_device)
        for name, want, got in (("k", want_k, got_k[layer]), ("v", want_v, got_v[layer])):
            _, pcc = comp_pcc(want, got, PCC_REQUIRED)
            logger.info(f"continuation layer {layer} {name} PCC: {pcc}")
            pccs.append(pcc)

    assert min(pccs) >= PCC_REQUIRED, (
        f"KV PCC {min(pccs)} after a continuation resuming at {resume} (chunk_size={chunk_size}); "
        f"the chunk's tokens were dealt to the wrong SP rows"
    )


# =====================================================================================
# Validation read-back (#4150 / #4152)
# =====================================================================================
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card"),
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kv_readback_inverts_the_blockcyclic_write(mesh_device, device_params, reset_seeds):
    """``read_slot_kv``'s host inverse agrees with the writer kernel's walk, across two chunks.

    The read-back is what grades the full-model run, so it has to be wrong-able independently of the
    model: here the cache is written directly with known per-position vectors, so a mismatch means
    the inverse walk disagrees with the kernel and nothing else.

    Two chunks specifically. The block-cyclic mapping is contiguous-per-row for chunk 0 only (see
    ``test_blockcyclic_first_chunk_covers_exactly_the_first_chunk`` in the KV suite), so an inverse
    that merely un-shards would pass a one-chunk test and scramble every continuation.
    """
    from models.demos.llama_3p1_8b_d_p.tt.kv_cache import write_kv_chunk

    torch.manual_seed(0)
    chunk_size, max_seq_len, num_tokens = 128, 256, 256
    runtime, kv_cache, config = _build(mesh_device, chunk_size=chunk_size, max_seq_len=max_seq_len, num_layers=1)

    heads = kv_cache.num_kv_heads_per_chip * config.tp_factor
    head_dim = Llama31_8BConfig.HEAD_DIM
    # A distinct random vector per (head, absolute position): unique enough that any misplacement is
    # a mismatch, and unlike a position-stamped integer it survives bfloat8_b without aliasing its
    # neighbours (bf8 cannot hold 2047 exactly, so stamping would blur adjacent positions together).
    ref_k = torch.randn(1, heads, num_tokens, head_dim)
    ref_v = torch.randn(1, heads, num_tokens, head_dim)

    shard_dims = [None, None]
    shard_dims[config.sp_axis] = 2  # sequence over the SP rows
    shard_dims[config.tp_axis] = 1  # KV heads over the TP columns
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=config.mesh_shape, dims=tuple(shard_dims))

    for start in range(0, num_tokens, chunk_size):
        sl = slice(start, start + chunk_size)

        def to_dev(t):
            return ttnn.from_torch(
                t[:, :, sl], device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )

        tt_k, tt_v = to_dev(ref_k), to_dev(ref_v)
        write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=0, kv_actual=start, sp_axis=config.sp_axis)
    ttnn.synchronize_device(mesh_device)

    got_k, got_v = runtime.read_slot_kv(kv_cache, 0, num_tokens=num_tokens)
    assert got_k.shape == (1, heads, num_tokens, head_dim), f"unexpected read-back shape {tuple(got_k.shape)}"

    for name, ref, got in (("k", ref_k, got_k), ("v", ref_v, got_v)):
        want = _quantize_like(ref[0], ttnn.bfloat8_b, mesh_device)
        moved = [p for p in range(num_tokens) if not torch.allclose(want[:, p], got[0, :, p], atol=1e-2)]
        assert not moved, (
            f"{name}: {len(moved)} position(s) read back from the wrong cache row, first few "
            f"{moved[:8]} — the host inverse walk disagrees with the writer kernel"
        )
    logger.info(f"KV read-back round-trip exact over {num_tokens} positions, {heads} head(s)")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card")],
    indirect=["mesh_device", "device_params"],
)
def test_kv_pcc_check_refuses_a_trace_it_could_not_read(mesh_device, device_params, tmp_path, expect_error):
    """A read-back that compared nothing fails (#4152), rather than reporting a vacuous pass.

    The failure mode this guards is a green galaxy job that validated zero layers because
    ``PREFILL_TRACE_DIR`` pointed somewhere empty — the most expensive kind of false pass, since it
    is the one that makes everything downstream look verified.
    """
    runtime, kv_cache, _ = _build(mesh_device, chunk_size=128, max_seq_len=256, num_layers=1)

    with expect_error(FileNotFoundError, "does not exist"):
        runtime.kv_cache_pcc_check(kv_cache, tmp_path, num_tokens=128)

    (tmp_path / "kv_cache").mkdir()  # present but empty: no layer_*.safetensors to compare
    with expect_error(FileNotFoundError, "golden layer 0 missing"):
        runtime.kv_cache_pcc_check(kv_cache, tmp_path, num_tokens=128)


@pytest.mark.skipif(
    os.getenv("LLAMA31_8B_REAL_WEIGHTS") != "1",
    reason="opt-in: reads the real 16 GB checkpoint (set LLAMA31_8B_REAL_WEIGHTS=1)",
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        # #4150: the full model on one Galaxy, at the production 4x8 (SP=4 x TP=8). This arm is the
        # acceptance run -- 32 layers, real weights, chunked, on the geometry that ships.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_real_checkpoint_hidden_states(mesh_device, device_params, reset_seeds):
    """The full 32-layer model on real weights, chunked, vs the torch reference.

    Opt-in because it reads the whole checkpoint. This is the test that would catch a weight-name
    or frame error that random weights cannot: with random weights, a wrongly-permuted q/k
    projection is still *a* valid projection, so PCC stays high.
    """
    from models.demos.llama_3p1_8b_d_p.reference.model import load_hf_state_dict

    torch.manual_seed(0)
    chunk_size, prompt_len, max_seq_len = 256, 512, 1024

    hf_state_dict = load_hf_state_dict()
    reference = Llama31Model().eval()
    reference.load_hf_state_dict(hf_state_dict)

    token_ids = torch.randint(0, 32000, (1, prompt_len))
    with torch.no_grad():
        torch_hidden = _reference_hidden_states(reference, token_ids)

    # Drop the 32-layer torch model before building the device one. Two full-size copies of
    # Llama-3.1-8B plus the staging tensors do not fit in host RAM on these nodes, and the way it
    # fails is a silent stall in weight upload rather than a MemoryError.
    del reference
    gc.collect()

    runtime, kv_cache, config = _build(
        mesh_device,
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        vocab_size=Llama31_8BConfig.VOCAB_SIZE,
        state_dict=hf_state_dict,
    )
    del hf_state_dict
    gc.collect()
    tt_hidden = runtime.prefill_prompt(token_ids[0].tolist(), kv_cache, return_hidden_states=True)

    got = ttnn.to_torch(
        tt_hidden, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(2, -1))
    )
    got = got[..., :EMB_DIM].reshape(1, -1, EMB_DIM)[:, -chunk_size:].to(torch.float32)

    passing, pcc = comp_pcc(torch_hidden[:, -chunk_size:], got, PCC_REQUIRED)
    logger.info(f"real-checkpoint 32-layer hidden-state PCC: {pcc}")
    assert passing, f"real-checkpoint PCC {pcc} below {PCC_REQUIRED}"


@pytest.mark.skipif(
    os.getenv("LLAMA31_8B_REAL_WEIGHTS") != "1",
    reason="opt-in: reads the real 16 GB checkpoint (set LLAMA31_8B_REAL_WEIGHTS=1)",
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        # #4150's acceptance geometry: SP=4 x TP=8 on one Galaxy, 32 layers, 2 chunks of 1024.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_real_checkpoint_kv_pcc_vs_golden(mesh_device, device_params, reset_seeds):
    """#4150: 32 layers, real weights, 2 chunks, per-layer KV PCC against the CPU-generated golden.

    This is the acceptance run, and it grades a different thing than
    ``test_real_checkpoint_hidden_states``: hidden states are the model's *output*, while the cache
    is its *product* — prefill is headless, so a cache written to the wrong rows, in the wrong RoPE
    frame, or with V converted as if it were K can leave the hidden states intact and still hand
    decode garbage. Only a per-layer cache comparison sees that.

    The golden's tokens come from the trace, not from a fresh random draw, because the golden KV is
    only meaningful for the prompt it was generated from.
    """
    import json
    from pathlib import Path

    from models.demos.llama_3p1_8b_d_p.reference.model import load_hf_state_dict
    from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter

    trace_dir = Path(os.environ.get("PREFILL_TRACE_DIR") or Llama31PrefillAdapter.prefill_trace_default)
    if not (trace_dir / "metadata.json").exists():
        pytest.skip(f"no golden trace at {trace_dir}; generate with scripts/generate_golden_kv_cache.py")
    # Present but unreadable is a failure, not a skip, and it needs saying out loud: a trace on the
    # shared store belongs to whoever generated it, `save_file` writes 0600, and CI runs as another
    # user. Left to safetensors this surfaces as "No such file or directory" naming a file that is
    # sitting right there, which reads as a missing golden or a wrong mount.
    layer_0 = trace_dir / "kv_cache" / "layer_0.safetensors"
    if layer_0.exists() and not os.access(layer_0, os.R_OK):
        raise AssertionError(f"golden at {trace_dir} is not readable by {getpass.getuser()}; chmod -R a+rX it")

    metadata = json.loads((trace_dir / "metadata.json").read_text())
    # The two fields a consumer must not guess. A golden in the HF frame would fail on K and pass on
    # V, which looks like a RoPE bug rather than a mis-generated golden.
    assert metadata["rope_frame"] == "meta", f"golden is in the {metadata['rope_frame']} frame; decode reads meta"
    assert metadata["num_layers"] == Llama31_8BConfig.NUM_LAYERS, "golden is not a full 32-layer trace"

    chunk_size, max_seq_len = 1024, 2048
    token_ids = list(metadata["token_ids"])[:max_seq_len]
    num_tokens = len(token_ids)
    assert num_tokens == max_seq_len, f"golden has {num_tokens} tokens, this arm needs {max_seq_len}"

    runtime, kv_cache, _ = _build(
        mesh_device,
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        vocab_size=Llama31_8BConfig.VOCAB_SIZE,
        state_dict=load_hf_state_dict(),
    )
    runtime.prefill_prompt(token_ids, kv_cache)

    results = runtime.kv_cache_pcc_check(kv_cache, trace_dir, num_tokens=num_tokens, min_pcc=PCC_REQUIRED_KV_GOLDEN)
    assert len(results) == Llama31_8BConfig.NUM_LAYERS, f"graded {len(results)} layers, not all 32"
    worst = min(min(pair) for pair in results.values())
    logger.info(f"#4150 per-layer KV PCC: {len(results)} layers, worst {worst:.6f}")

    # A wiring bug in one layer reads as an outlier against its neighbours rather than as a low
    # floor, and a floor alone cannot see that. Depth-accumulated error is smooth, so also require
    # each layer to sit near the median. 0.01 against a measured spread of 0.9959..0.99997: tight
    # enough to catch one broken layer hiding above the floor, loose enough not to trip on the
    # smooth depth trend. Was 0.03 when the floor was 0.95 and the spread was ten times wider.
    per_layer = {layer: min(pair) for layer, pair in results.items()}
    median = sorted(per_layer.values())[len(per_layer) // 2]
    outliers = {layer: pcc for layer, pcc in per_layer.items() if median - pcc > 0.01}
    assert not outliers, f"layer(s) {outliers} sit far below the median {median:.6f}; suspect that layer"
