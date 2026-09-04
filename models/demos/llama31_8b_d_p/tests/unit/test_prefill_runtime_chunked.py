# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-RUNTIME` — `tt/tt_prefill_runtime.py` satisfies the engine's §2 runtime contract.

The chunked-prefill test of P7's choosing (`BRINGUP_RECIPE.md` P7 step 4 leaves the second test
file open). It gates the *contract*, which `G-CHUNK` does not: `G-CHUNK` drives the KV-cache
producer through its public functions because the packed cache needs `TP == num_key_value_heads`
(`R-027`), so nothing else in this package ever constructs a `TtPrefillRuntime` or checks that the
engine will find the attributes it reads. An adapter is one line in P10 only if this holds now.

What it asserts, in the order the engine does it
(`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` §2):

1. `runtime.mesh_device` and `runtime.config` exist, and `config` exposes **exactly** the five
   documented names — `chunk_size`, `max_seq_len`, `first_layer_idx`, `is_first_rank`,
   `is_last_rank`. `chunk_size` is a property aliasing `default_chunk_size` (`DEC-054`); the
   template has only the latter and bridges it in its adapter, which is a rename waiting to drift.
2. The three shape constraints the engine documents: `MAX_SEQ_LEN % CHUNK_SIZE == 0`,
   `CHUNK_SIZE % (SP*32) == 0`, and `kv_actual_global % 32 == 0`.
3. `make_chunk_input(token_ids)` -> a device tensor of the right dtype/layout/shape, on both the
   first-rank (token ids) and non-first-rank (activation placeholder) paths.
4. `compile` / `prefill_chunk` / `set_layer_ack_channel` / `kv_migration_base_address` /
   `build_kv_chunk_table` all exist with the documented signatures.
5. Every refusal is loud and names its cause: the two single-device blockers (`DEC-056`, `R-027`,
   `R-028`), a half-enabled SP config, a raw-dict `hf_config` (the `rope_theta` trap, `R-014`), a
   mesh-shape/device mismatch, and a chunk size that does not tile the cache.

**Negative control** (`test_the_contract_check_can_fail`): the same attribute audit run against an
object that is missing one of the five names must FAIL. Without it, an audit built from
`getattr(..., default)` would pass against anything, including a runtime with no config at all.

**Input distribution / reference dtype:** neither applies — no PCC is measured here. The weights
are random because this gate is about interfaces and refusals; every number in P7 comes from
`G-CHUNK` / `G-GOLDEN` on real weights.

Run::

    pytest models/demos/llama31_8b_d_p/tests/unit/test_prefill_runtime_chunked.py -x -q
"""

from __future__ import annotations

import inspect

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, llama_config_dims
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config
from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import (
    TtPrefillRuntime,
    TtPrefillRuntimeConfig,
    resolve_chunk_sizes,
)

# The five names `ADDING_A_PREFILL_MODEL.md` §2 says the engine reads off `runtime.config`.
ENGINE_CONFIG_FIELDS = ("chunk_size", "max_seq_len", "first_layer_idx", "is_first_rank", "is_last_rank")
# The runtime methods the engine calls, with the arguments it passes.
ENGINE_METHODS = {
    "compile": ("kv_cache",),
    "make_chunk_input": ("token_ids",),
    "prefill_chunk": ("input_tensor", "kv_cache", "slot_id", "actual_start", "actual_end", "request_id"),
}

CHUNK = 128
MAX_SEQ = 512


def _random_one_layer_state(dims):
    """One decoder layer plus the embedding / final norm — the minimum `Model` will build."""
    hidden, vocab = dims["hidden_size"], dims["vocab_size"]
    n_q, n_kv, head_dim = dims["num_attention_heads"], dims["num_key_value_heads"], dims["head_dim"]
    inter = dims["intermediate_size"]
    gen = torch.Generator().manual_seed(0)
    return {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, generator=gen) * 0.02,
        "model.norm.weight": torch.randn(hidden, generator=gen) * 0.1 + 1.0,
        "model.layers.0.input_layernorm.weight": torch.randn(hidden, generator=gen) * 0.1 + 1.0,
        "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden, generator=gen) * 0.1 + 1.0,
        "model.layers.0.self_attn.q_proj.weight": torch.randn(n_q * head_dim, hidden, generator=gen) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(n_kv * head_dim, hidden, generator=gen) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(n_kv * head_dim, hidden, generator=gen) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, n_q * head_dim, generator=gen) * 0.02,
        "model.layers.0.mlp.gate_proj.weight": torch.randn(inter, hidden, generator=gen) * 0.02,
        "model.layers.0.mlp.up_proj.weight": torch.randn(inter, hidden, generator=gen) * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, inter, generator=gen) * 0.02,
    }


def _audit_engine_contract(runtime) -> list:
    """Return the list of contract violations; `[]` means the engine will find what it reads.

    Deliberately built from `hasattr` + `inspect.signature` rather than `getattr(..., default)`:
    a default would make the audit pass against anything, which is what
    `test_the_contract_check_can_fail` exists to prove it does not.
    """
    problems = []
    if not hasattr(runtime, "mesh_device"):
        problems.append("runtime.mesh_device missing")
    if not hasattr(runtime, "config"):
        problems.append("runtime.config missing")
        return problems
    for name in ENGINE_CONFIG_FIELDS:
        if not hasattr(runtime.config, name):
            problems.append(f"runtime.config.{name} missing")
    for method, params in ENGINE_METHODS.items():
        fn = getattr(runtime, method, None)
        if fn is None or not callable(fn):
            problems.append(f"runtime.{method} missing")
            continue
        sig = inspect.signature(fn).parameters
        for param in params:
            if param not in sig:
                problems.append(f"runtime.{method} does not accept {param!r}")
    return problems


# =====================================================================================
# Host-only: the config surface and the shape constraints
# =====================================================================================
def test_config_exposes_the_engine_contract_names():
    """`chunk_size` aliases `default_chunk_size`, and all five documented names resolve."""
    config = TtPrefillRuntimeConfig(num_layers=32, max_seq_len=MAX_SEQ, default_chunk_size=CHUNK)
    for name in ENGINE_CONFIG_FIELDS:
        assert hasattr(config, name), f"config.{name} is missing (ADDING_A_PREFILL_MODEL.md §2)"
    assert config.chunk_size == config.default_chunk_size == CHUNK
    assert (config.is_first_rank, config.is_last_rank, config.first_layer_idx) == (True, True, 0)
    # DEC-055: the engine owns the cache, so the default must not allocate one.
    assert config.owns_kv_cache is False, "owns_kv_cache must default to False (the engine owns it)"
    config_48 = TtPrefillRuntimeConfig(num_layers=32, max_seq_len=MAX_SEQ, mesh_shape=(4, 8))
    assert (config_48.sp_factor, config_48.tp_factor) == (4, 8)
    logger.info(
        f"[G-RUNTIME] config surface OK: chunk_size={config.chunk_size} (alias of default_chunk_size), "
        f"max_seq_len={config.max_seq_len}, ranks=({config.is_first_rank}, {config.is_last_rank}), "
        f"first_layer_idx={config.first_layer_idx}, owns_kv_cache={config.owns_kv_cache}; "
        f"(4,8) -> sp={config_48.sp_factor} tp={config_48.tp_factor}"
    )


def test_resolve_chunk_sizes_orders_and_refuses(expect_error):
    """Largest first, deduped, and every size must divide `max_seq_len`."""
    assert resolve_chunk_sizes(128, (512, 128), 1024) == (512, 128)
    assert resolve_chunk_sizes(256, (), 1024) == (256,)
    with expect_error(ValueError, "must be a multiple of every supported chunk size"):
        resolve_chunk_sizes(300, (), 1024)
    with expect_error(ValueError, "does not divide it"):
        resolve_chunk_sizes(128, (384,), 1024)
    logger.info("[G-RUNTIME] resolve_chunk_sizes: largest-first, deduped, non-dividing sizes refused")


@pytest.mark.parametrize("mesh_shape", [(1, 1), (1, 8), (4, 8)], ids=["1x1", "1x8", "4x8"])
def test_the_three_documented_shape_constraints(mesh_shape, expect_error):
    """`MAX % CHUNK == 0`, `CHUNK % (SP*32) == 0`, `kv_actual % 32 == 0` — host arithmetic only.

    The third is `update_padded_kv_cache`'s own assert and is re-stated in `prefill_chunk`; the
    first two are checked at construction so a bad config fails before 8 B parameters are loaded.
    Parametrised over the three meshes because `CHUNK % (SP*32)` is the only one that depends on
    the mesh, and at SP=4 it becomes `CHUNK % 128 == 0` — the constraint `00_MODEL_CARD.md` §4.4
    records for the target.
    """
    sp = mesh_shape[0]
    good = ttnn.TILE_SIZE * sp
    assert MAX_SEQ % CHUNK == 0
    assert CHUNK % good == 0, f"CHUNK={CHUNK} must be a multiple of TILE_SIZE*sp={good}"
    config = TtPrefillRuntimeConfig(num_layers=1, max_seq_len=MAX_SEQ, mesh_shape=mesh_shape, default_chunk_size=CHUNK)
    assert config.sp_factor == sp
    # A size that tiles the cache but not the per-chip tile grid: SP=4 rejects 32, SP=1 accepts it.
    tiny = ttnn.TILE_SIZE
    assert (tiny % good == 0) == (sp == 1)
    with expect_error(ValueError, "does not divide it"):
        resolve_chunk_sizes(CHUNK, (MAX_SEQ - CHUNK,), MAX_SEQ)
    logger.info(
        f"[G-RUNTIME] {mesh_shape}: sp={sp}, CHUNK={CHUNK} % (TILE_SIZE*sp={good}) == 0, "
        f"MAX_SEQ={MAX_SEQ} % CHUNK == 0"
    )


def test_the_contract_check_can_fail():
    """NEGATIVE CONTROL: the audit must reject an object missing a contract name.

    `_audit_engine_contract` is the only thing standing between P10 and a renamed attribute, so an
    audit that cannot fail is worse than none — it would report a clean PASS for a runtime the
    engine cannot drive.
    """

    class MissingChunkSize:
        mesh_device = object()

        class config:  # noqa: N801 - a stand-in, not a real class name
            max_seq_len = MAX_SEQ
            first_layer_idx = 0
            is_first_rank = True
            is_last_rank = True

        def compile(self, kv_cache=None):
            pass

        def make_chunk_input(self, token_ids):
            pass

        def prefill_chunk(self, input_tensor, kv_cache=None, *, slot_id, actual_start, actual_end, request_id=0):
            pass

    problems = _audit_engine_contract(MissingChunkSize())
    assert problems == ["runtime.config.chunk_size missing"], f"audit did not catch the omission: {problems}"

    class NoConfig:
        mesh_device = object()

    assert _audit_engine_contract(NoConfig()) == ["runtime.config missing"]
    logger.info("[G-RUNTIME] negative control OK: the contract audit rejects a missing name")


def test_hf_config_must_be_the_normalised_object(expect_error):
    """A raw dict or a bare `transformers` config is refused — the `rope_theta` trap (`R-014`).

    On transformers 5.12.1 `LlamaConfig.rope_theta` raises `AttributeError` and
    `getattr(cfg, "rope_theta", DEFAULT)` **succeeds with the default**, so a runtime that accepted
    either would build a RoPE that is wrong at every position with no exception anywhere. The
    check is at construction, before any weight is read.
    """
    config = TtPrefillRuntimeConfig(num_layers=1, max_seq_len=MAX_SEQ, mesh_shape=(1, 1), default_chunk_size=CHUNK)
    dims = llama_config_dims()
    with expect_error(AssertionError, "pass the LlamaHFConfig from"):
        TtPrefillRuntime(None, dims, {}, config)
    # And the normalised object really does carry theta, non-None, at Llama-3.1's value.
    hf_config = llama_hf_config(dims)
    assert float(hf_config.rope_theta) == 500000.0, f"rope_theta is {hf_config.rope_theta}, expected 500000.0"
    logger.info(
        f"[G-RUNTIME] a raw config dict is refused at construction; LlamaHFConfig.rope_theta = "
        f"{hf_config.rope_theta} (a getattr default of 10000.0 would be silently wrong)"
    )


# =====================================================================================
# Device: build the runtime, exercise the contract, check every refusal
# =====================================================================================
@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_runtime_satisfies_the_engine_contract(mesh_device, reset_seeds, expect_error):
    """Build a 1-layer runtime on one card and drive the whole documented surface."""
    dims = llama_config_dims()
    hf_config = llama_hf_config(dims)
    mesh_shape = tuple(mesh_device.shape)
    config = TtPrefillRuntimeConfig(
        num_layers=1,
        max_seq_len=MAX_SEQ,
        mesh_shape=mesh_shape,
        default_chunk_size=CHUNK,
        additional_chunk_sizes=(MAX_SEQ,),  # the one-shot size, which must also tile the cache
        num_users=1,
    )
    runtime = TtPrefillRuntime(mesh_device, hf_config, _random_one_layer_state(dims), config)

    problems = _audit_engine_contract(runtime)
    assert problems == [], f"engine contract violations: {problems}"
    assert runtime.mesh_device is mesh_device
    assert runtime.chunk_sizes == (MAX_SEQ, CHUNK), f"chunk sizes {runtime.chunk_sizes} not largest-first"
    assert set(runtime.rope_indexed) == {CHUNK, MAX_SEQ}, "one indexed rope table per supported size"
    for size, mats in runtime.rope_indexed.items():
        assert tuple(mats[0].shape) == (1, 1, MAX_SEQ // config.sp_factor, hf_config.head_dim), size
    assert runtime.kv_cache is None, "owns_kv_cache=False must not allocate a cache (DEC-055)"

    # --- make_chunk_input, both ranks ---
    tokens = runtime.make_chunk_input(list(range(CHUNK)))
    assert tuple(tokens.shape) == (1, 1, 1, CHUNK // config.sp_factor), tuple(tokens.shape)
    assert tokens.dtype == ttnn.uint32 and tokens.layout == ttnn.ROW_MAJOR_LAYOUT
    embedded = runtime.model.embedding(tokens)
    assert tuple(embedded.shape) == (1, 1, CHUNK // config.sp_factor, hf_config.hidden_size)
    embedded.deallocate(True)
    tokens.deallocate(True)

    big = runtime.make_chunk_input(list(range(MAX_SEQ)), MAX_SEQ)
    assert tuple(big.shape) == (1, 1, 1, MAX_SEQ // config.sp_factor)
    big.deallocate(True)

    with expect_error(AssertionError, "must be exactly chunk_size"):
        runtime.make_chunk_input(list(range(CHUNK - 1)))
    with expect_error(AssertionError, "not one of this runtime's supported sizes"):
        runtime.make_chunk_input([0] * 64, 64)

    runtime.config.is_first_rank = False
    placeholder = runtime.make_chunk_input([])
    assert tuple(placeholder.shape) == (1, 1, CHUNK // config.sp_factor, hf_config.hidden_size)
    assert placeholder.dtype == ttnn.bfloat16 and placeholder.layout == ttnn.TILE_LAYOUT
    placeholder.deallocate(True)
    runtime.config.is_first_rank = True

    # --- the refusals ---
    kv_cache = allocate_kv_cache(
        mesh_device, num_layers=1, max_seq_len=MAX_SEQ, num_users=1, head_dim=hf_config.head_dim
    )
    # R-028 / DEC-056: chunk > 0 needs the cache-read attention. Checked BEFORE the cache, so this
    # message wins over the TP one even though both apply on this mesh.
    with expect_error(NotImplementedError, "needs the cache-read attention"):
        runtime.prefill_chunk(
            runtime.make_chunk_input([0] * CHUNK), kv_cache, slot_id=0, actual_start=CHUNK, actual_end=2 * CHUNK
        )
    with expect_error(AssertionError, "must be tile-aligned"):
        runtime.prefill_chunk(
            runtime.make_chunk_input([0] * CHUNK), kv_cache, slot_id=0, actual_start=17, actual_end=17 + CHUNK
        )
    with expect_error(AssertionError, "slot_id 3 out of range"):
        runtime.prefill_chunk(
            runtime.make_chunk_input([0] * CHUNK), kv_cache, slot_id=3, actual_start=0, actual_end=CHUNK
        )
    with expect_error(AssertionError, "exceeds the per-user cache capacity"):
        runtime.prefill_chunk(
            runtime.make_chunk_input([0] * CHUNK), kv_cache, slot_id=0, actual_start=MAX_SEQ, actual_end=MAX_SEQ + 1
        )
    # R-027: the packed cache is one KV head per chip, so TP must equal num_key_value_heads.
    with expect_error(AssertionError, "ONE KV head per chip"):
        runtime.prefill_chunk(
            runtime.make_chunk_input([0] * CHUNK), kv_cache, slot_id=0, actual_start=0, actual_end=CHUNK
        )
    with expect_error(AssertionError, "ONE KV head per chip"):
        runtime.compile(kv_cache)
    with expect_error(AssertionError, "does not own a KV cache"):
        runtime.prefill_chunk(runtime.make_chunk_input([0] * CHUNK), None, slot_id=0, actual_start=0, actual_end=CHUNK)
    # P7 asserted here that `build_kv_chunk_table` refused with "P10's deliverable". **P10 landed it**
    # (`tt/runners/kv_chunk_table.py`, `R-030` closed), so that refusal is correctly gone and this is
    # replaced by its opposite — the same move P8 made when it implemented the `dense_sp` stub.
    # Two things are asserted instead, and together they keep the hook under test on a mesh that
    # cannot build a real table:
    #   1. it no longer raises the P10 placeholder (a regression to a stub would fail here), and
    #   2. it now gets far enough to resolve the cache, where `(1,1)`'s TP=1 hits R-027's refusal.
    # The table's real behaviour is gated on the `(4,8)` galaxy by `G-KV-TABLE`
    # (`tests/unit/test_kv_chunk_table.py`), bit-exactly.
    with expect_error(AssertionError, "ONE KV head per chip"):
        runtime.build_kv_chunk_table(kv_cache, "/tmp/does-not-matter")
    from models.demos.llama31_8b_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

    assert callable(
        build_and_serialize_kv_chunk_table
    ), "the module build_kv_chunk_table forwards to is missing; it stopped being a stub in P10"
    # DEC-109: the multi-rank merge is not implemented and must RAISE rather than publish a table
    # that addresses one rank's DRAM under every rank's layer ids (R-040).
    with expect_error(NotImplementedError, "R-040"):
        runtime.build_kv_chunk_table(kv_cache, "/tmp/does-not-matter", first_layer_idx=16)
    with expect_error(NotImplementedError, "set_layer_ack_channel"):
        runtime.prefill_chunk(
            runtime.make_chunk_input([0] * CHUNK),
            kv_cache,
            slot_id=0,
            actual_start=0,
            actual_end=CHUNK,
            d2h_service=object(),
        )
    with expect_error(AssertionError, "call compile\\(\\) before set_layer_ack_channel"):
        runtime.set_layer_ack_channel(object())

    logger.info(
        f"[G-RUNTIME] contract satisfied on {mesh_shape}: 5/5 config names, 3/3 engine methods, "
        f"chunk sizes {runtime.chunk_sizes}, {len(runtime.rope_indexed)} indexed rope tables; "
        f"9 refusals all loud and named"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_construction_refuses_half_enabled_and_mismatched_configs(mesh_device, expect_error):
    """SP on a 1-row mesh, and a config/device mesh-shape mismatch.

    Both are cheap to get wrong and expensive to notice: a half-enabled SP config raises inside the
    forward, **after** the KV cache has been written, so a caller that swallowed the exception would
    be left with a half-populated cache and no output.

    **Changed in P8.** This test used to assert a third refusal, `sequence_parallel=True` while
    `tt/attention/dense_sp.dense_sp_attention` was "still the P5 stub". P8 implemented it, so that
    refusal is *correctly* gone and asserting it would now fail. What replaces it is not another
    refusal but its opposite, asserted below: `_dense_sp_is_implemented()` — the probe
    `TtPrefillRuntime` uses to decide whether SP is available (`DEC-056`) — must now return **True**.
    That keeps the probe itself under test: it works by calling the function with no arguments and
    distinguishing `NotImplementedError` (stub) from `TypeError` (a real signature), so a port that
    kept a `*args, **kwargs` signature would leave it stuck on `False` and silently disable SP.
    """
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import _dense_sp_is_implemented

    hf_config = llama_hf_config(llama_config_dims())
    assert _dense_sp_is_implemented(), (
        "the dense_sp stub probe still reports NOT implemented after P8's port; "
        "TtPrefillRuntime._chunked_read_supported would then refuse every chunk past the first"
    )
    with expect_error(AssertionError, "there is no sequence axis to shard"):
        TtPrefillRuntime(
            mesh_device,
            hf_config,
            {},
            TtPrefillRuntimeConfig(
                num_layers=1,
                max_seq_len=MAX_SEQ,
                mesh_shape=tuple(mesh_device.shape),
                default_chunk_size=CHUNK,
                sequence_parallel=True,
            ),
        )
    with expect_error(AssertionError, "but the open device is"):
        TtPrefillRuntime(
            mesh_device,
            hf_config,
            {},
            TtPrefillRuntimeConfig(num_layers=1, max_seq_len=MAX_SEQ, mesh_shape=(4, 8), default_chunk_size=CHUNK),
        )
    logger.info(
        "[G-RUNTIME] the dense_sp probe reports IMPLEMENTED (P8); half-enabled SP (sp=1) and "
        "mesh-shape mismatches are still refused at construction"
    )
