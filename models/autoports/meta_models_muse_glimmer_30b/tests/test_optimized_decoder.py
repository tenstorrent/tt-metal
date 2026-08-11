# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Muse-Glimmer-30B TTNN *optimized* decoder layer.

This is the stage-02 twin of ``test_functional_decoder.py``: the same coverage, the same
test names, the same 0.995 PCC bar, against
:class:`~models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder.OptimizedDecoder`.
Both suites can therefore be diffed test-for-test, which is the point — the optimized layer
changes precision, memory layout and program configs, and nothing else.

Two precision policies appear here, for one measured reason:

* :data:`DEFAULT_POLICY` (``bfp4_all``) is what the layer ships with. On **real checkpoint
  weights** it measures prefill PCC 0.9986 (sliding_rope) / 0.9990 (full_nope) and decode
  0.9970 / 0.9984 — comfortably above 0.995. On **synthetic** weights the same code
  measures ~0.961-0.970, because BFP4's shared-exponent block quantisation is far lossier
  on i.i.d. Gaussian weight blocks than on the checkpoint's own strongly correlated ones.
  That is the ``$optimize`` skill's OPT-012 finding: a weights discrepancy, not an
  implementation defect.
* :data:`STRUCTURAL_POLICY` (``bfp8_all_lofi``) is the *same code path* at BFP8 weights,
  which measures 0.9987-0.9989 prefill and 0.9981-0.9987 decode on synthetic weights. Every
  structural test (page tables, block sizes, non-aligned lengths, ragged slots, batching,
  window enforcement, continued prefill, short prefills) uses it and asserts the real 0.995
  bar, so a page-table, chunking or layout bug still fails the suite at full strength on
  cheap synthetic weights.

Acceptance for the shipped BFP4 policy comes from the real-weight tests
(``test_real_weights_prefill_decode``, ``test_real_weights_non_aligned_and_traced``,
``test_stress_repeated_prefill_decode``), all of which assert 0.995.
``test_synthetic_weight_precision_discrepancy`` records the synthetic gap itself so the
artifact JSON carries the evidence rather than a comment.

Run the fast suite:

    pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py \
        -m "not real_weights"
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    DECODE_SDPA_K_CHUNK,
    DEFAULT_POLICY,
    POLICIES,
    PREFILL_CHUNK_SIZE,
    PREFILL_SDPA_K_CHUNK,
    PREFILL_SDPA_Q_CHUNK,
    MuseGlimmerDecoderConfig,
    OptimizedDecoder,
    _round_up,
)

#: The acceptance bar, identical to the functional stage's.
PCC_BAR = 0.995

#: Policy used by every synthetic-weight structural test. Same code, same layouts, same
#: program-config selection as the default policy — only the weight dtype differs — so a
#: structural failure is still caught at the full 0.995 bar (see the module docstring).
STRUCTURAL_POLICY = "bfp8_all_lofi"

#: Diagnostic-only floor for BFP4 policies measured on *synthetic* weights. Never an
#: acceptance gate: see ``test_synthetic_weight_precision_discrepancy``.
SYNTHETIC_BFP4_DIAGNOSTIC_FLOOR = 0.95

#: How far a long-context PCC may fall below the same policy's PCC at 1000 tokens before the
#: policy counts as length-dependent, on **real** weights. Tight on purpose: measured drift
#: over 1000 -> 12345 tokens is 0.0002 for the shipped policy
#: (``doc/optimized_decoder/pcc/length_dependence.json``).
LENGTH_INDEPENDENCE_DELTA = 0.005

#: The same allowance on **synthetic** weights, where BFP4's shared-exponent quantisation of
#: i.i.d. Gaussian blocks does drift with length on the `full_attention` kind: the last query
#: block measures 0.9576 / 0.9555 / 0.9498 at 1000 / 8192 / 32768 tokens, against
#: 0.99149 / 0.99131 / 0.99097 for BFP8 attention weights. The real-weight control above
#: shows the effect is a property of the synthetic distribution, not of the length, so this
#: test asserts the *shape* (no worse than the measured synthetic drift) rather than an
#: absolute floor that would mean nothing for BFP4 on synthetic weights.
SYNTHETIC_LENGTH_DELTA = 0.02

#: The policies whose synthetic-weight PCC is recorded as OPT-012 evidence.
DIAGNOSTIC_POLICIES = ("bfp8_all_lofi", "bfp8_attn_bfp4_mlp", "bfp4_all")

KIND_IDS = ["sliding_rope", "full_nope"]

DEFAULT_BLOCK = 64


# --------------------------------------------------------------------------- helpers


def _blocks_for(total_tokens: int, block_size: int) -> int:
    return _round_up(total_tokens, block_size) // block_size


def _alloc_paged(decoder, mesh_device, *, batch, total_tokens, block_size, page_seed, pool_multiplier=3):
    """Allocate a paged cache from a *pool* larger than needed, with a shuffled page table.

    Identical to the functional suite's helper, deliberately: the shuffle plus the larger
    pool means logical block ``i`` almost never maps to physical block ``i``, so an
    address/indexing bug cannot pass by accident. The width comes from
    ``decoder.blocks_per_seq`` because the decode SDPA rounds its K extent up to the K
    chunk and reads the page table over the *rounded* extent.
    """
    blocks_per_seq = decoder.blocks_per_seq(total_tokens)
    total_blocks = max(blocks_per_seq * batch * pool_multiplier, blocks_per_seq * batch + 3)
    page_table = U.build_page_table(
        batch=batch, blocks_per_seq=blocks_per_seq, total_blocks=total_blocks, seed=page_seed
    )
    assert page_table.min() >= 0
    assert len(set(page_table.reshape(-1).tolist())) == page_table.numel(), "page table must not alias blocks"
    kv_cache = decoder.allocate_kv_cache(
        batch_size=batch, max_seq_len=blocks_per_seq * block_size, num_blocks=total_blocks
    )
    return kv_cache, page_table, U.page_table_to_device(page_table, mesh_device)


def _ref_prefill_batch(ref, hidden, *, cache_len, backend="eager"):
    """Reference prefill plus zero-padded K/V buffers long enough for later decode steps."""
    ref_out, ref_k, ref_v = ref.prefill(hidden, backend=backend)
    batch, kv_heads, seq, head_dim = ref_k.shape
    k_full = torch.zeros(batch, kv_heads, cache_len, head_dim, dtype=ref_k.dtype)
    v_full = torch.zeros_like(k_full)
    k_full[:, :, :seq] = ref_k
    v_full[:, :, :seq] = ref_v
    return ref_out, k_full, v_full


def _decode_once(decoder, hidden_d, *, kv_cache, page_table_tt, position, mesh_device):
    """One device decode step at a single position, returned as ``[batch, 1, hidden]`` torch."""
    current_pos, rope_idxs = U.position_tensors([position], mesh_device)
    out = decoder.decode_forward(
        U.decode_input(hidden_d, mesh_device),
        kv_cache=kv_cache,
        page_table=page_table_tt,
        current_pos=current_pos,
        rope_idxs=rope_idxs,
    )
    return U.decode_output(out)


def _policy_uses_bfp4(name: str) -> bool:
    policy = POLICIES[name]
    return ttnn.bfloat4_b in (
        policy.attn_weight_dtype,
        policy.mlp_weight_dtype,
        policy.mlp_down_weight_dtype,
    )


# --------------------------------------------------------------------------- fixtures


OPT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "doc" / "optimized_decoder"

#: Files whose contents decide the numbers this suite records. ``tests/conftest.py``
#: fingerprints the *functional* stage's files; a record carrying that fingerprint would
#: not pin the optimized layer at all, so this suite computes its own.
FINGERPRINTED_SOURCES = (
    "models/autoports/meta_models_muse_glimmer_30b/tt/optimized_decoder.py",
    "models/autoports/meta_models_muse_glimmer_30b/tt/functional_decoder.py",
    "models/autoports/meta_models_muse_glimmer_30b/reference/hf_reference.py",
    "models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py",
    "models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder_perf.py",
    "models/autoports/meta_models_muse_glimmer_30b/tests/decoder_test_utils.py",
    "models/autoports/meta_models_muse_glimmer_30b/tests/conftest.py",
)


@pytest.fixture(scope="session")
def code_fingerprint():
    """Provenance for an optimized-stage PCC record: what code produced it, and when."""
    repo_root = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    for relative in FINGERPRINTED_SOURCES:
        digest.update((repo_root / relative).read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # pragma: no cover - git absent
        commit = ""
    return {
        "code_sha256": digest.hexdigest()[:16],
        "git_head": commit,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@pytest.fixture(scope="session")
def opt_pcc_record(code_fingerprint):
    """Collect this suite's PCCs into the *optimized* stage artifact.

    ``tests/conftest.py`` owns a ``pcc_record`` fixture that writes into
    ``doc/functional_decoder/pcc/pcc_results.json``. Both suites share that conftest, so
    without this override the optimized numbers would be merged into (and would overwrite
    same-named records in) the functional stage's committed evidence.
    """
    records: list[dict] = []
    yield records
    if not records:
        return
    path = OPT_ARTIFACT_DIR / "pcc" / "pcc_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text())["records"] if path.is_file() else []
    by_key = {(r["test"], r["metric"]): r for r in existing}
    for record in records:
        by_key[(record["test"], record["metric"])] = record
    path.write_text(
        json.dumps(
            {
                "current_code_sha256": code_fingerprint["code_sha256"],
                "records": sorted(by_key.values(), key=lambda r: (r["test"], r["metric"])),
            },
            indent=2,
        )
        + "\n"
    )


@pytest.fixture
def record_pcc(opt_pcc_record, request, code_fingerprint):
    """Override of the conftest fixture so records land in the optimized-stage artifact."""

    def _record(metric: str, value: float, *, threshold: float = PCC_BAR, **extra):
        entry = {"test": request.node.name, "metric": metric, "pcc": float(value), "threshold": threshold}
        entry.update(extra)
        entry.update(code_fingerprint)
        opt_pcc_record.append(entry)
        return value

    return _record


@pytest.fixture(scope="module")
def optimized_decoder_cache(mg_mesh_device):
    """Module-scoped store of built optimized layers (one upload is ~0.5-1 GB of weights)."""
    cache: dict = {}
    yield cache
    cache.clear()


@pytest.fixture
def build_optimized_decoder(mg_mesh_device, text_config, optimized_decoder_cache):
    """``(layer_idx, state_dict, precision=..., block_size=..., ...) -> OptimizedDecoder``.

    Cached on every argument that changes the uploaded weights or the derived program
    configs, so a test that only varies the sequence length reuses the layer. ``tag`` keys
    the state dict, so real- and synthetic-weight layers never alias.
    """

    def _build(
        layer_idx,
        state_dict,
        *,
        tag="synthetic",
        precision=STRUCTURAL_POLICY,
        block_size=DEFAULT_BLOCK,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
        **kwargs,
    ):
        key = (
            layer_idx,
            tag,
            str(precision),
            block_size,
            prefill_chunk_size,
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
        )
        if key not in optimized_decoder_cache:
            optimized_decoder_cache[key] = OptimizedDecoder.from_state_dict(
                state_dict,
                hf_config=text_config,
                layer_idx=layer_idx,
                mesh_device=mg_mesh_device,
                precision=precision,
                block_size=block_size,
                prefill_chunk_size=prefill_chunk_size,
                **kwargs,
            )
        return optimized_decoder_cache[key]

    return _build


# ------------------------------------------------------------------- host-only tests


@pytest.mark.parametrize(
    "length,cap,expected",
    [
        (8192, 256, 256),  # the measured-fast configuration is kept when it divides
        (131072, 256, 256),
        (2080, 512, 32),  # 2080 = 4*512 + 32: a non-dividing q_chunk HANGS the SDPA op
        (3008, 256, 64),
        (128, 256, 128),  # short chunk: falls back to a divisor, never a padded 256
        (10240, 128, 128),
        (10240, 256, 256),
    ],
)
def test_sdpa_chunk_divides_length(length, cap, expected):
    """SDPA chunk sizes must divide the sequence length.

    The optimized layer inherits the functional stage's chunk selector unchanged, and the
    reason it exists is bug 7 of the functional ``work_log.md``:
    ``scaled_dot_product_attention`` with a ``q_chunk_size`` that does not divide the Q
    length does not error, it hangs the device. Re-asserted here so a "harmless" tidy-up of
    the optimized selector cannot reintroduce the hang.
    """
    chunk = OptimizedDecoder._sdpa_chunk_for_length(length, cap)
    assert chunk == expected
    assert length % chunk == 0
    assert chunk <= cap


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [3000, 8256, 12345, 16448, 131072])
def test_chunked_sdpa_k_chunk_stays_inside_the_page_table(seq_len, block_size):
    """The k-chunk-padded K extent must fit the page table, for every chunk of every length.

    Regression for the bug ``$autofix`` found in the functional stage (work_log section 9):
    ``chunked_scaled_dot_product_attention`` validates only
    ``kv_length >= q_len + chunk_start_idx``, but its program factory rounds the K extent up
    to ``k_chunk_size`` and the reader consumes one page-table entry per ``block_size`` of
    that *rounded* extent with no bound check
    (``sdpa/device/kernels/dataflow/dataflow_common.hpp``). Overrunning reads the page-table
    stick's 32-byte alignment padding as block ids and then reads K/V from outside the cache
    buffer — measured as layer PCC 0.7345 instead of 0.9998, and separately as a device hang.

    This walks the optimized layer's real chunking (its own ``blocks_per_seq``, its own
    ``prefill_chunk_size``) for a range of lengths and page block sizes and asserts the
    selected ``k_chunk`` never overruns.
    """
    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    decoder.prefill_sdpa_q_chunk = PREFILL_SDPA_Q_CHUNK
    decoder.prefill_sdpa_k_chunk = PREFILL_SDPA_K_CHUNK
    decoder.block_size = block_size
    decoder.decode_sdpa_k_chunk = DECODE_SDPA_K_CHUNK

    padded_seq = _round_up(seq_len, 32)
    # What a caller allocates: enough blocks for the prompt plus one decode step.
    blocks_per_seq = decoder.blocks_per_seq(seq_len + 1)
    kv_length = blocks_per_seq * block_size

    chunk_start = 0
    while chunk_start < padded_seq:
        chunk_len = min(PREFILL_CHUNK_SIZE, padded_seq - chunk_start)
        q_chunk, k_chunk = decoder._chunked_sdpa_chunk_sizes(chunk_len, chunk_start, kv_length)
        padded_k_extent = _round_up(chunk_start + chunk_len, k_chunk)
        assert padded_k_extent <= kv_length, (
            f"seq_len={seq_len} block_size={block_size} chunk[{chunk_start}, "
            f"{chunk_start + chunk_len}): k_chunk={k_chunk} rounds the K extent up to "
            f"{padded_k_extent}, past the {kv_length}-token page table"
        )
        assert q_chunk <= PREFILL_SDPA_Q_CHUNK and k_chunk <= PREFILL_SDPA_K_CHUNK
        assert chunk_start % q_chunk == 0 and chunk_start % k_chunk == 0
        assert chunk_len % q_chunk == 0
        chunk_start += chunk_len


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [777, 1000, 3000, 8256, 131072])
def test_decode_page_table_capacity_covers_the_k_chunk_rounding(seq_len, block_size, expect_error):
    """The decode SDPA's k-chunk rounding must not walk past the page table either.

    ``paged_scaled_dot_product_attention_decode`` rounds its K extent up to ``k_chunk_size``
    (``rt_args_common.hpp``: ``valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size)``) and the
    reader resolves the rounded extent through the unbounded ``page_table_ptr[virtual_block]``,
    exactly like the chunked prefill op. ``cur_pos`` is a device tensor, so the guard has to
    hold for *every* position the page table can address — which is what ``blocks_per_seq``
    rounding up to a whole number of K chunks buys. The optimized layer keeps the guard, and
    keeps it loud: an under-sized page table is refused, not silently read out of bounds.
    """
    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    decoder.decode_sdpa_k_chunk = DECODE_SDPA_K_CHUNK
    decoder.block_size = block_size

    blocks = decoder.blocks_per_seq(seq_len + 1)
    capacity = blocks * block_size
    assert capacity >= seq_len + 1
    # Every addressable position, not just the ones the tests happen to decode at.
    for position in (0, 1, seq_len - 1, seq_len, capacity - 1):
        assert _round_up(position + 1, DECODE_SDPA_K_CHUNK) <= capacity, (
            f"seq_len={seq_len} block_size={block_size} capacity={capacity}: decoding at "
            f"position {position} rounds the K extent to "
            f"{_round_up(position + 1, DECODE_SDPA_K_CHUNK)}, past the page table"
        )
    decoder._check_decode_page_table_capacity(capacity)

    unaligned = _blocks_for(seq_len + 1, block_size) * block_size
    if unaligned % DECODE_SDPA_K_CHUNK:
        with expect_error(ValueError, "whole number of decode K chunks"):
            decoder._check_decode_page_table_capacity(unaligned)


@pytest.mark.parametrize("batch,expected_cores", [(1, 32), (16, 32), (17, 110), (32, 110)])
def test_decode_sdpa_grid_covers_kv_heads(batch, expected_cores, kinds, text_config, expect_error):
    """The decode SDPA grid must give every (user, KV head) pair its own core.

    Regression for bug 8 of the functional ``work_log.md``: a grid with
    ``cores < batch * num_key_value_heads`` passes the op's own validation and silently
    returns wrong results (batch 32 on 8x4 measured PCC 0.7176). The optimized layer shrinks
    the *default* decode grid to the swept 8x4, which makes the escalation rule load-bearing
    rather than decorative. Exercised without a device by driving the grid selection directly.
    """
    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    decoder.config = MuseGlimmerDecoderConfig.from_hf_config(text_config, kinds["sliding_rope"].layer_idx)
    decoder.device_grid = (11, 10)
    decoder.decode_sdpa_core_grid = (8, 4)
    decoder.decode_sdpa_k_chunk = DECODE_SDPA_K_CHUNK
    decoder.sdpa_core_grid = (11, 10)

    program_config = decoder._decode_sdpa_program_config(batch)
    grid = program_config.compute_with_storage_grid_size
    cores = grid.x * grid.y
    assert cores == expected_cores
    assert cores >= batch * text_config.num_key_value_heads

    # Beyond the grid's capacity the layer must fail loudly rather than mis-compute.
    with expect_error(ValueError, "needs"):
        decoder._decode_sdpa_program_config(56)


def test_source_has_no_runtime_torch():
    """Static audit: torch / host-transfer calls only exist inside the setup boundary.

    ``from_state_dict`` (weight conversion, RoPE table construction) and
    ``allocate_kv_cache`` (empty cache allocation) are the two documented setup entry points
    of the optimized layer, exactly as in the functional one; every other function in
    ``optimized_decoder.py`` must be pure device code. The optimized layer adds host-side
    *arithmetic* (L1 budgeting, program-config selection) in ``__init__``, which is why this
    audit is worth repeating: none of it may reach for torch.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt import optimized_decoder as module

    setup_functions = {"from_state_dict", "allocate_kv_cache"}
    forbidden_ttnn = {"from_torch", "to_torch", "as_tensor", "to_torch_and_close"}
    tree = ast.parse(inspect.getsource(module))
    offenders: list[str] = []

    def visit(node, in_setup: bool):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_setup = in_setup or node.name in setup_functions
        if not in_setup:
            if isinstance(node, ast.Name) and node.id == "torch":
                offenders.append(f"line {node.lineno}: reference to torch")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "torch":
                    offenders.append(f"line {node.lineno}: torch.{node.attr}")
                if node.value.id == "ttnn" and node.attr in forbidden_ttnn:
                    offenders.append(f"line {node.lineno}: ttnn.{node.attr}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "torch":
                        offenders.append(f"line {node.lineno}: import torch")
        for child in ast.iter_child_nodes(node):
            visit(child, in_setup)

    visit(tree, False)
    assert not offenders, "runtime host fallback in optimized_decoder.py:\n" + "\n".join(offenders)


# ------------------------------------------------------------------- device tests


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize(
    "seq_len",
    [
        32,  # smallest tile-aligned smoke length
        100,  # non-aligned, below one tile row of pages
        1000,  # non-aligned, below the sliding window
        2048,  # exactly the sliding-window boundary
        2080,  # just across the sliding-window boundary
        3000,  # non-aligned, past the window
        pytest.param(8192, marks=pytest.mark.slow),  # exactly the prefill chunk boundary
        pytest.param(8256, marks=pytest.mark.slow),  # just across the prefill chunk boundary
        pytest.param(12345, marks=pytest.mark.slow),  # divisible by neither tile, page, window nor chunk
    ],
)
def test_paged_prefill_decode_pcc(
    kind_id,
    seq_len,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Paged prefill + paged decode PCC across tile/page/window/chunk boundaries.

    Also checks the paged K/V cache contents (un-paged through the shuffled page table) and
    that decode reads the cache the prefill wrote. The optimized layer stores the cache at
    BFP8 rather than BF16, so the cache comparison is a real check of that choice and not a
    formality.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    decode_steps = 2
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=13)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + decode_steps)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + decode_steps, block_size=block_size, page_seed=101
    )
    hidden_tt = U.prefill_input(hidden, mg_mesh_device)
    out_tt = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
    out = U.prefill_output(out_tt)
    assert out.shape == (1, seq_len, text_config.hidden_size)
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill", prefill_pcc, seq_len=seq_len, kind=kind_id, policy=STRUCTURAL_POLICY)
    assert prefill_pcc >= PCC_BAR, f"prefill PCC {prefill_pcc}"

    k_dev = U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=seq_len)
    v_dev = U.read_paged_cache(kv_cache[1], page_table, block_size=block_size, seq_len=seq_len)
    k_pcc = U.pcc(ref_k[:, :, :seq_len], k_dev)
    v_pcc = U.pcc(ref_v[:, :, :seq_len], v_dev)
    record_pcc("prefill_k_cache", k_pcc, seq_len=seq_len, kind=kind_id, policy=STRUCTURAL_POLICY)
    record_pcc("prefill_v_cache", v_pcc, seq_len=seq_len, kind=kind_id, policy=STRUCTURAL_POLICY)
    assert k_pcc >= PCC_BAR and v_pcc >= PCC_BAR, f"cache PCC k={k_pcc} v={v_pcc}"

    for step in range(decode_steps):
        position = seq_len + step
        hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=200 + step)
        ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([position]))
        out_d = _decode_once(
            decoder,
            hidden_d,
            kv_cache=kv_cache,
            page_table_tt=page_table_tt,
            position=position,
            mesh_device=mg_mesh_device,
        )
        decode_pcc = U.pcc(ref_d, out_d)
        record_pcc(
            f"decode_step{step}",
            decode_pcc,
            seq_len=seq_len,
            position=position,
            kind=kind_id,
            policy=STRUCTURAL_POLICY,
        )
        assert decode_pcc >= PCC_BAR, f"decode PCC {decode_pcc} at position {position}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_page_block_sizes(
    kind_id,
    block_size,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Page-table handling is independent of the block size (32/64/128 tokens per page).

    The block size also feeds ``blocks_per_seq``'s lcm rounding against the 64-token decode
    K chunk, so this is the device-side twin of
    ``test_decode_page_table_capacity_covers_the_k_chunk_rounding``.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    seq_len = 777
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=17)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=block_size
    )
    out = U.prefill_output(
        decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    )
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill", prefill_pcc, block_size=block_size, kind=kind_id, policy=STRUCTURAL_POLICY)
    assert prefill_pcc >= PCC_BAR

    hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=18)
    ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([seq_len]))
    out_d = _decode_once(
        decoder,
        hidden_d,
        kv_cache=kv_cache,
        page_table_tt=page_table_tt,
        position=seq_len,
        mesh_device=mg_mesh_device,
    )
    decode_pcc = U.pcc(ref_d, out_d)
    record_pcc("decode", decode_pcc, block_size=block_size, kind=kind_id, policy=STRUCTURAL_POLICY)
    assert decode_pcc >= PCC_BAR


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("batch", [4, 32])
def test_batched_prefill_and_decode(
    kind_id,
    batch,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Batched paged prefill (one ``paged_fill_cache`` per user slot) and batched decode.

    Batch 32 is the case that decides the decode SDPA grid escalation (32 users x 2 KV heads
    = 64 cores, past the swept 8x4 default) and the width-sharded decode residual's per-user
    shard layout, so a per-user PCC is asserted, not just the aggregate.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    seq_len = 300
    hidden = R.unit_rms_hidden_states((batch, seq_len, text_config.hidden_size), seed=23)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=batch, total_tokens=seq_len + 1, block_size=block_size, page_seed=7 + batch
    )
    out = U.prefill_output(
        decoder.prefill_forward(
            U.prefill_input(hidden, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            user_ids=list(range(batch)),
        )
    )
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc("prefill", prefill_pcc, batch=batch, kind=kind_id, policy=STRUCTURAL_POLICY)
    assert prefill_pcc >= PCC_BAR
    for b in range(batch):
        per_user = U.pcc(ref_out[b], out[b])
        assert per_user >= PCC_BAR, f"user {b} prefill PCC {per_user}"

    hidden_d = R.unit_rms_hidden_states((batch, 1, text_config.hidden_size), seed=24)
    positions = torch.full((batch,), seq_len, dtype=torch.long)
    ref_d = ref.decode(hidden_d, ref_k, ref_v, positions)
    current_pos, rope_idxs = U.position_tensors(positions.tolist(), mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    decode_pcc = U.pcc(ref_d, out_d)
    record_pcc("decode", decode_pcc, batch=batch, kind=kind_id, policy=STRUCTURAL_POLICY)
    assert decode_pcc >= PCC_BAR
    for b in range(batch):
        per_user = U.pcc(ref_d[b], out_d[b])
        assert per_user >= PCC_BAR, f"user {b} decode PCC {per_user}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_ragged_slots_and_current_positions(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Per-user prefill into permuted cache slots, then batched decode at ragged positions.

    Each user gets a different (non-aligned) prompt length and is written to a *shuffled*
    cache slot, so a wrong ``batch_idx`` / page-table row or a wrong ``current_pos`` entry
    shows up as a per-user PCC failure. The sliding kind spans lengths on both sides of the
    2048 window. This is also the test that would catch the optimized decode's per-user
    cos/sin gather being sharded onto a tidy rectangle instead of Q's own shard set — which
    rotates users by each other's positions and is invisible at batch 1.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    prompt_lens = [37, 1000, 2049, 3111]
    slots = [3, 0, 2, 1]  # user u prefills into cache slot slots[u]
    batch = len(prompt_lens)
    max_tokens = max(prompt_lens) + 1

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=batch, total_tokens=max_tokens, block_size=block_size, page_seed=55
    )

    ref_k = torch.zeros(batch, text_config.num_key_value_heads, max_tokens, text_config.head_dim)
    ref_v = torch.zeros_like(ref_k)
    for user, (prompt_len, slot) in enumerate(zip(prompt_lens, slots)):
        hidden = R.unit_rms_hidden_states((1, prompt_len, text_config.hidden_size), seed=300 + user)
        out_ref, k_ref, v_ref = ref.prefill(hidden, backend="eager")
        ref_k[slot, :, :prompt_len] = k_ref[0]
        ref_v[slot, :, :prompt_len] = v_ref[0]
        out = U.prefill_output(
            decoder.prefill_forward(
                U.prefill_input(hidden, mg_mesh_device),
                kv_cache=kv_cache,
                page_table=page_table_tt,
                user_ids=[slot],
            )
        )
        per_user = U.pcc(out_ref, out)
        record_pcc(
            f"prefill_user{user}_slot{slot}",
            per_user,
            seq_len=prompt_len,
            kind=kind_id,
            policy=STRUCTURAL_POLICY,
        )
        assert per_user >= PCC_BAR, f"user {user} (slot {slot}) prefill PCC {per_user}"

    # Batched decode: slot s continues the prompt that was written into slot s.
    slot_lens = {slot: prompt_len for prompt_len, slot in zip(prompt_lens, slots)}
    positions = torch.tensor([slot_lens[s] for s in range(batch)], dtype=torch.long)
    hidden_d = torch.cat(
        [R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=400 + s) for s in range(batch)], dim=0
    )
    ref_d = ref.decode(hidden_d, ref_k, ref_v, positions)
    current_pos, rope_idxs = U.position_tensors(positions.tolist(), mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    for slot in range(batch):
        per_user = U.pcc(ref_d[slot], out_d[slot])
        record_pcc(
            f"decode_slot{slot}",
            per_user,
            position=int(positions[slot]),
            kind=kind_id,
            policy=STRUCTURAL_POLICY,
        )
        assert per_user >= PCC_BAR, f"slot {slot} decode PCC {per_user} at position {int(positions[slot])}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_short_prefill_lengths(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Sub-tile prompt lengths (1, 7, 31) — the bottom of the advertised ``[1, 131072]`` range.

    These are also the lengths where the optimized prefill's row folding degenerates: fewer
    rows than one ``prefill_matmul_cutoff`` block, so the 2D program config runs at
    ``per_core_M`` 1 with a single padded tile row.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    for seq_len in (1, 7, 31):
        hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=800 + seq_len)
        ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=seq_len + 1,
            block_size=block_size,
            page_seed=800 + seq_len,
        )
        out = U.prefill_output(
            decoder.prefill_forward(
                U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
            )
        )
        assert out.shape == (1, seq_len, text_config.hidden_size)
        prefill_pcc = U.pcc(ref_out, out)
        record_pcc(f"prefill_len{seq_len}", prefill_pcc, seq_len=seq_len, kind=kind_id, policy=STRUCTURAL_POLICY)
        assert prefill_pcc >= PCC_BAR, f"prefill PCC {prefill_pcc} at seq_len {seq_len}"

        hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=900 + seq_len)
        ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([seq_len]))
        decode_pcc = U.pcc(
            ref_d,
            _decode_once(
                decoder,
                hidden_d,
                kv_cache=kv_cache,
                page_table_tt=page_table_tt,
                position=seq_len,
                mesh_device=mg_mesh_device,
            ),
        )
        record_pcc(f"decode_after_len{seq_len}", decode_pcc, position=seq_len, kind=kind_id, policy=STRUCTURAL_POLICY)
        assert decode_pcc >= PCC_BAR, f"decode PCC {decode_pcc} after seq_len {seq_len}"
        for cache in kv_cache:
            cache.deallocate(True)


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_batched_multichunk_prefill_shared_pool(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Batched prefill that crosses the internal chunk boundary, out of a shared slot pool.

    This is the combination the single-user and short-batched tests miss:

    * ``seq_len`` > ``prefill_chunk_size``, so the second chunk fills the cache at
      ``first_block > 0`` **and** the sliding overlap-trim path runs, both with batch > 1;
    * the page table has more rows than the batch and ``user_ids`` is not the identity, so
      ``_rows_page_table``'s multi-row gather + concat branch is exercised (the chunked SDPA
      needs exactly the batch's rows, in batch order);
    * a non-default page block size with batch > 1.

    The functional suite tightens the bar to 0.9995 here because this configuration is how
    the chunked-SDPA page-table overrun was found. That number is not carried over: the
    optimized layer runs a different precision policy, so its absolute PCC level is a
    different (and unmeasured) constant. The overrun manifests as a collapse to ~0.73, which
    the 0.995 bar catches with room to spare.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = 128
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    batch, seq_len = 2, 8256  # 8256 = 8192 + 64: two chunks, second one only 64 tokens long
    slots = [3, 1]  # non-identity rows out of a 4-row pool
    pool_rows = 4

    hidden = R.unit_rms_hidden_states((batch, seq_len, text_config.hidden_size), seed=1717)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1, backend="sdpa")

    blocks_per_seq = decoder.blocks_per_seq(seq_len + 1)
    page_table = U.build_page_table(
        batch=pool_rows, blocks_per_seq=blocks_per_seq, total_blocks=blocks_per_seq * pool_rows + 5, seed=1717
    )
    kv_cache = decoder.allocate_kv_cache(
        batch_size=pool_rows,
        max_seq_len=blocks_per_seq * block_size,
        num_blocks=blocks_per_seq * pool_rows + 5,
    )
    page_table_tt = U.page_table_to_device(page_table, mg_mesh_device)
    assert page_table.shape[0] > batch, "the point of this test is a pool wider than the batch"

    out = U.prefill_output(
        decoder.prefill_forward(
            U.prefill_input(hidden, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            user_ids=slots,
        )
    )
    for index in range(batch):
        per_user = U.pcc(ref_out[index], out[index])
        record_pcc(
            f"prefill_batch{index}_slot{slots[index]}",
            per_user,
            seq_len=seq_len,
            kind=kind_id,
            block_size=block_size,
            policy=STRUCTURAL_POLICY,
        )
        assert per_user >= PCC_BAR, f"batch row {index} (slot {slots[index]}) prefill PCC {per_user}"

    # Each row must have landed in *its* slot: compare the un-paged cache slot-by-slot.
    k_dev = U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=seq_len)
    for index, slot in enumerate(slots):
        cache_pcc = U.pcc(ref_k[index, :, :seq_len], k_dev[slot])
        record_pcc(f"k_cache_slot{slot}", cache_pcc, seq_len=seq_len, kind=kind_id, policy=STRUCTURAL_POLICY)
        assert cache_pcc >= PCC_BAR, f"slot {slot} K cache PCC {cache_pcc}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_continued_prefill_contract(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
    expect_error,
):
    """``start_pos > 0``: correct on full-attention layers, refused on sliding layers.

    A continued segment's first ``sliding_window`` positions need K/V that live in the paged
    cache, and the windowed prefill SDPA reads its prefix from the input tensor instead, so a
    sliding layer must refuse rather than return a truncated-window answer. Full-attention
    layers read the whole prefix from the cache via the chunked SDPA, so they are exact — and
    that is checked against a single-shot prefill of the same prompt. The optimized layer
    keeps both halves of this contract verbatim.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    first, second = 1024, 512
    total = first + second
    hidden = R.unit_rms_hidden_states((1, total, text_config.hidden_size), seed=1919)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=total + 1, block_size=block_size, page_seed=1919
    )
    decoder.prefill_forward(
        U.prefill_input(hidden[:, :first], mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
    ).deallocate(True)

    if kind.sliding_window:
        with expect_error(ValueError, "start_pos"):
            decoder.prefill_forward(
                U.prefill_input(hidden[:, first:], mg_mesh_device),
                kv_cache=kv_cache,
                page_table=page_table_tt,
                start_pos=first,
            )
        return

    ref_out, _, _ = ref.prefill(hidden, backend="eager", keep_kv=False)
    out = U.prefill_output(
        decoder.prefill_forward(
            U.prefill_input(hidden[:, first:], mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            start_pos=first,
        )
    )
    continued_pcc = U.pcc(ref_out[:, first:], out)
    record_pcc(
        "continued_prefill", continued_pcc, seq_len=second, start_pos=first, kind=kind_id, policy=STRUCTURAL_POLICY
    )
    assert continued_pcc >= PCC_BAR, f"continued prefill PCC {continued_pcc}"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_sliding_window_is_enforced(
    kind_id, kinds, text_config, synthetic_state_dicts, build_optimized_decoder, mg_mesh_device
):
    """A sliding layer must ignore tokens outside its window; a full layer must not.

    Perturbing the cache at a position that is ``> sliding_window`` behind the decode
    position changes the full-attention output and leaves the sliding output untouched. This
    distinguishes a correct window from "window silently ignored" — a failure mode the
    optimized decode path can reintroduce, because it passes ``sliding_window_size`` into a
    differently configured decode SDPA (8x4 grid, 64-token K chunk).
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=STRUCTURAL_POLICY, block_size=block_size)

    seq_len = 3072  # > 2048 window, so early positions fall outside it
    position = seq_len
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=31)
    hidden_far = hidden.clone()
    hidden_far[:, :512] += 4.0  # positions 0..511 are >2048 behind `position`

    outputs = []
    for prompt in (hidden, hidden_far):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=71
        )
        decoder.prefill_forward(U.prefill_input(prompt, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
        hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=32)
        outputs.append(
            _decode_once(
                decoder,
                hidden_d,
                kv_cache=kv_cache,
                page_table_tt=page_table_tt,
                position=position,
                mesh_device=mg_mesh_device,
            )
        )
        for cache in kv_cache:
            cache.deallocate(True)

    delta = (outputs[0] - outputs[1]).abs().max().item()
    scale = outputs[0].abs().max().item()
    if kind.sliding_window:
        assert delta <= 1e-3 * scale, f"sliding layer reacted to out-of-window tokens (delta {delta}, scale {scale})"
    else:
        assert delta > 1e-2 * scale, f"full-attention layer ignored distant tokens (delta {delta}, scale {scale})"


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_determinism_repeated_inputs(
    kind_id, kinds, text_config, synthetic_state_dicts, build_optimized_decoder, mg_mesh_device
):
    """Identical inputs produce bit-identical prefill and decode outputs.

    Run on the shipped :data:`DEFAULT_POLICY`, because this test makes no accuracy claim —
    it pins the *shipped* configuration's reproducibility, which the sharded decode path
    could break through a layout- or grid-dependent reduction order.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=DEFAULT_POLICY, block_size=block_size)

    seq_len = 1000
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=41)
    hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=42)

    prefill_runs, decode_runs = [], []
    for _ in range(3):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=91
        )
        prefill_runs.append(
            U.prefill_output(
                decoder.prefill_forward(
                    U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
                )
            )
        )
        decode_runs.append(
            _decode_once(
                decoder,
                hidden_d,
                kv_cache=kv_cache,
                page_table_tt=page_table_tt,
                position=seq_len,
                mesh_device=mg_mesh_device,
            )
        )
        for cache in kv_cache:
            cache.deallocate(True)

    for repeat in range(1, 3):
        assert torch.equal(prefill_runs[0], prefill_runs[repeat]), "prefill is not deterministic"
        assert torch.equal(decode_runs[0], decode_runs[repeat]), "decode is not deterministic"


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize(
    "weights",
    [
        "synthetic",
        pytest.param("real", marks=pytest.mark.real_weights),
    ],
)
def test_traced_decode_pcc(
    weights,
    kind_id,
    kinds,
    text_config,
    build_reference,
    synthetic_state_dicts,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Decode runs under ``ttnn`` traced execution and PCC is measured from the replay.

    Trace inputs (hidden state, ``current_pos``, ``rope_idxs``) are pre-allocated device
    tensors; replay only copies new contents into them, so no tensor is allocated after
    capture and the cache/page-table buffer addresses stay baked into the trace. The
    optimized decode ends on an L1 width-sharded output, and the replayed tensor is read
    through the same ``ttnn.to_torch`` path as the untraced one — which is the check that
    the residual shard survives capture.

    Two variants: the synthetic-weight one runs :data:`STRUCTURAL_POLICY` and proves the
    traced *implementation* is exact at the 0.995 bar; the real-weight one runs the shipped
    :data:`DEFAULT_POLICY` and is the acceptance measurement.
    """
    kind = kinds[kind_id]
    real = weights == "real"
    policy = DEFAULT_POLICY if real else STRUCTURAL_POLICY
    seq_len = 512 if real else 1000
    steps = 3

    if real:
        state_dict = R.load_real_layer_state_dict(kind.layer_idx)
        hidden = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(seq_len))
        hidden_steps = [
            R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(1, offset=seq_len + s))
            for s in range(steps)
        ]
    else:
        state_dict = synthetic_state_dicts[kind.layer_idx]
        hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=51)
        hidden_steps = [R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=500 + s) for s in range(steps)]

    ref = build_reference(kind.layer_idx, state_dict, tag=weights)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, tag=weights, precision=policy, block_size=block_size)

    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + steps)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + steps, block_size=block_size, page_seed=61
    )
    decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    del ref_out

    positions = [seq_len + s for s in range(steps)]

    # Persistent trace inputs, written in place before every replay.
    x_dev = U.decode_input(hidden_steps[0], mg_mesh_device)
    pos_dev, rope_dev = U.position_tensors([positions[0]], mg_mesh_device)

    # Warmup (compiles kernels and fills the program cache) and capture both execute the
    # step, and a decode step is idempotent in the cache: it writes the same K/V to the same
    # slot for the same input and position. So the replayed step 0 still sees exactly the
    # post-prefill cache state plus its own (identical) write.
    decoder.decode_forward(x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev)
    ttnn.synchronize_device(mg_mesh_device)

    trace_id = ttnn.begin_trace_capture(mg_mesh_device, cq_id=0)
    out_dev = decoder.decode_forward(
        x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev
    )
    ttnn.end_trace_capture(mg_mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mg_mesh_device)

    try:
        for step in range(steps):
            host_x = U.host_decode_input(hidden_steps[step], mg_mesh_device)
            host_pos, host_rope = U.position_tensors([positions[step]], mg_mesh_device, device=False)
            ttnn.copy_host_to_device_tensor(host_x, x_dev)
            ttnn.copy_host_to_device_tensor(host_pos, pos_dev)
            ttnn.copy_host_to_device_tensor(host_rope, rope_dev)
            ttnn.execute_trace(mg_mesh_device, trace_id, cq_id=0, blocking=True)
            traced_out = U.decode_output(out_dev)

            ref_d = ref.decode(hidden_steps[step], ref_k, ref_v, torch.tensor([positions[step]]))
            traced_pcc = U.pcc(ref_d, traced_out)
            record_pcc(
                f"traced_decode_step{step}",
                traced_pcc,
                position=positions[step],
                kind=kind_id,
                weights=weights,
                policy=policy,
            )
            assert traced_pcc >= PCC_BAR, f"traced decode PCC {traced_pcc} at position {positions[step]}"
    finally:
        ttnn.release_trace(mg_mesh_device, trace_id)


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_no_runtime_host_fallback(
    kind_id, kinds, text_config, synthetic_state_dicts, build_optimized_decoder, mg_mesh_device
):
    """A single prefill and decode pass must not touch torch or host<->device transfer.

    ``TorchFunctionMode`` catches *any* torch op, and the ttnn conversion entry points are
    replaced with tripwires, so a hidden host fallback in the layer or in a helper it calls
    fails the test instead of quietly costing latency. Run on the shipped
    :data:`DEFAULT_POLICY`: the optimized layer computes shard specs, L1 budgets and program
    configs at setup time precisely so that none of that arithmetic runs per step.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=DEFAULT_POLICY, block_size=block_size)

    seq_len = 1024
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=71)
    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=81
    )
    hidden_tt = U.prefill_input(hidden, mg_mesh_device)
    hidden_d_tt = U.decode_input(R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=72), mg_mesh_device)
    current_pos, rope_idxs = U.position_tensors([seq_len], mg_mesh_device)

    class NoTorchOps(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"torch op {getattr(func, '__name__', func)} called inside a measured pass")

    tripwires = {}
    for name in ("from_torch", "to_torch", "as_tensor"):
        tripwires[name] = getattr(ttnn, name)

        def _tripwire(*args, _name=name, **kwargs):
            raise AssertionError(f"ttnn.{_name} called inside a measured pass")

        setattr(ttnn, name, _tripwire)
    try:
        with NoTorchOps():
            out = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
            out_d = decoder.decode_forward(
                hidden_d_tt,
                kv_cache=kv_cache,
                page_table=page_table_tt,
                current_pos=current_pos,
                rope_idxs=rope_idxs,
            )
            ttnn.synchronize_device(mg_mesh_device)
    finally:
        for name, original in tripwires.items():
            setattr(ttnn, name, original)

    assert out.shape[-2] == seq_len
    assert out_d.shape[-2] == 1
    # The decode output must come back on the layer-to-layer residual contract, not on the
    # interleaved layout the functional layer returned: stacking layers needs no reshard.
    out_memory_config = out_d.memory_config()
    assert out_memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert out_memory_config.buffer_type == ttnn.BufferType.L1


@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_synthetic_weight_precision_discrepancy(
    kind_id,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """Diagnostic: record every candidate policy's PCC on *synthetic* weights (OPT-012).

    This test is coverage, not an acceptance gate. Its purpose is to put the measured
    synthetic-vs-real gap in the artifact JSON:

    * the BFP8 policies clear the real 0.995 bar on synthetic weights, so they are asserted
      against :data:`PCC_BAR` — that is what establishes the implementation is exact, since
      the arithmetic, layouts and program-config selection are identical across policies;
    * the BFP4 policies do not, measuring 0.960-0.970 (`bfp4_all`) and 0.991-0.993 (`bfp8_attn_bfp4_mlp`) on synthetic weights while measuring
      0.9970-0.9990 on the real checkpoint. They are asserted only against the explicitly
      named :data:`SYNTHETIC_BFP4_DIAGNOSTIC_FLOOR` (0.95), which exists to catch a genuine
      regression (a page-table or layout bug collapses PCC to ~0.7) without pretending the
      synthetic number is an acceptance measurement.

    Acceptance for the shipped BFP4 policy is ``test_real_weights_prefill_decode``,
    ``test_real_weights_non_aligned_and_traced`` and ``test_stress_repeated_prefill_decode``,
    all at the full 0.995 bar. Nothing here is marked xfail: every case must pass its stated
    floor.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK

    seq_len = 512
    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=1201)
    hidden_d = R.unit_rms_hidden_states((1, 1, text_config.hidden_size), seed=1202)
    ref_out, ref_k_base, ref_v_base = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)
    ref_d = ref.decode(hidden_d, ref_k_base.clone(), ref_v_base.clone(), torch.tensor([seq_len]))

    for policy_name in DIAGNOSTIC_POLICIES:
        floor = SYNTHETIC_BFP4_DIAGNOSTIC_FLOOR if _policy_uses_bfp4(policy_name) else PCC_BAR
        decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=policy_name, block_size=block_size)
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=1200
        )
        out = U.prefill_output(
            decoder.prefill_forward(
                U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
            )
        )
        prefill_pcc = U.pcc(ref_out, out)
        record_pcc(
            f"synthetic_prefill_{policy_name}",
            prefill_pcc,
            threshold=floor,
            seq_len=seq_len,
            kind=kind_id,
            weights="synthetic",
            policy=policy_name,
            diagnostic=True,
        )

        out_d = _decode_once(
            decoder,
            hidden_d,
            kv_cache=kv_cache,
            page_table_tt=page_table_tt,
            position=seq_len,
            mesh_device=mg_mesh_device,
        )
        decode_pcc = U.pcc(ref_d, out_d)
        record_pcc(
            f"synthetic_decode_{policy_name}",
            decode_pcc,
            threshold=floor,
            position=seq_len,
            kind=kind_id,
            weights="synthetic",
            policy=policy_name,
            diagnostic=True,
        )
        for cache in kv_cache:
            cache.deallocate(True)

        assert prefill_pcc >= floor, (
            f"policy {policy_name}: synthetic prefill PCC {prefill_pcc} below its documented " f"floor {floor}"
        )
        assert decode_pcc >= floor, (
            f"policy {policy_name}: synthetic decode PCC {decode_pcc} below its documented " f"floor {floor}"
        )


def test_optimized_beats_functional_topology(
    kinds, text_config, synthetic_state_dicts, build_optimized_decoder, mg_mesh_device
):
    """The shipped configuration really is the optimized topology, not the functional one.

    ``config_summary()`` is what the stage report quotes, so it is asserted rather than
    trusted: DRAM-sharded decode matmuls for every projection role, an L1 width-sharded
    decode residual that tiles the hidden size exactly, BFP4 weights and a BFP8 KV cache.
    The contrast with the functional layer — which has no decode residual contract at all
    and returns DRAM-interleaved activations — is checked on the class, with no device work.

    Needs the mesh device only because the decode program configs are derived from the real
    DRAM bank count and compute grid; it reuses the cached layer the other default-policy
    tests build.
    """
    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import FunctionalDecoder

    assert not hasattr(FunctionalDecoder, "decode_residual_memory_config"), (
        "the functional layer gained a decode residual contract; this test's premise "
        "(optimized-only sharded residual) needs revisiting"
    )

    kind = kinds["sliding_rope"]
    decoder = build_optimized_decoder(
        kind.layer_idx,
        synthetic_state_dicts[kind.layer_idx],
        precision=DEFAULT_POLICY,
        block_size=DEFAULT_BLOCK,
    )
    summary = decoder.config_summary()

    assert summary["precision_policy"] == DEFAULT_POLICY
    assert summary["decode"]["matmul_family"] == "dram_sharded"
    for role in ("qkv", "attn_gate", "wo", "mlp_gate_up", "mlp_down"):
        program_config = summary["decode"]["program_configs"][role]
        assert (
            program_config["type"] == "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig"
        ), f"decode role {role} is not a DRAM-sharded matmul: {program_config['type']}"

    dtypes = summary["dtypes"]
    assert "BFLOAT4_B" in dtypes["attn_weights"].upper(), dtypes["attn_weights"]
    assert "BFLOAT4_B" in dtypes["mlp_gate_up_weights"].upper(), dtypes["mlp_gate_up_weights"]
    assert "BFLOAT4_B" in dtypes["mlp_down_weights"].upper(), dtypes["mlp_down_weights"]
    assert "BFLOAT8_B" in dtypes["kv_cache"].upper(), dtypes["kv_cache"]

    decode = summary["decode"]
    assert decode["residual_cores"] * decode["residual_shard_width"] == text_config.hidden_size
    # The knobs the stage selected with measured evidence. Pinning them here means a silent
    # revert fails a test rather than only changing a number in a perf artifact.
    assert decode["mlp_working_cores"] == 52, decode["mlp_working_cores"]
    assert decode["sdpa_output_l1"] is True
    assert decode["rope_pad_to_tile"] is True
    assert decode["packer_l1_acc"] is True
    assert decode["fp32_dest_acc_en"] is False
    prefill = summary["prefill"]
    assert prefill["in0_block_w_cap"] == 26, prefill["in0_block_w_cap"]
    assert prefill["out_subblock_h"] == 1, prefill["out_subblock_h"]
    assert prefill["out_subblock_max"] == 8, prefill["out_subblock_max"]
    assert prefill["matmul_cutoff"] == 512, prefill["matmul_cutoff"]
    assert prefill["weight_memory"] == "dram_sharded"
    memory_config = decoder.decode_residual_memory_config
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert memory_config.buffer_type == ttnn.BufferType.L1


@pytest.mark.real_weights
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_real_weights_prefill_decode(
    kind_id, kinds, text_config, build_reference, build_optimized_decoder, mg_mesh_device, record_pcc
):
    """Real checkpoint weights and real activations on the shipped policy, end to end.

    The layer input is the checkpoint's own embedding rows for real tokenizer output, run
    through the *real* preceding layers (``stacked_layer_input``), so both the weights and
    the activation distribution are the model's own — not synthetic. This is the acceptance
    measurement for :data:`DEFAULT_POLICY` (measured 0.9986/0.9990 prefill and 0.9970/0.9984
    decode against the 0.995 bar).
    """
    kind = kinds[kind_id]
    state_dict = R.load_real_layer_state_dict(kind.layer_idx)
    assert set(state_dict) == set(R.LAYER_PARAM_NAMES)
    ref = build_reference(kind.layer_idx, state_dict, tag="real")
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(
        kind.layer_idx, state_dict, tag="real", precision=DEFAULT_POLICY, block_size=block_size
    )

    seq_len = 512
    token_ids = R.real_token_ids(seq_len)
    hidden = R.stacked_layer_input(text_config, kind.layer_idx, token_ids)
    assert hidden.shape == (1, seq_len, text_config.hidden_size)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=block_size, page_seed=909
    )
    out = U.prefill_output(
        decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    )
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc(
        "prefill_real_weights", prefill_pcc, seq_len=seq_len, kind=kind_id, weights="real", policy=DEFAULT_POLICY
    )
    assert prefill_pcc >= PCC_BAR, f"real-weight prefill PCC {prefill_pcc}"

    k_dev = U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=seq_len)
    cache_pcc = U.pcc(ref_k[:, :, :seq_len], k_dev)
    record_pcc(
        "prefill_k_cache_real_weights",
        cache_pcc,
        seq_len=seq_len,
        kind=kind_id,
        weights="real",
        policy=DEFAULT_POLICY,
    )
    assert cache_pcc >= PCC_BAR

    hidden_d = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(1, offset=seq_len))
    ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([seq_len]))
    out_d = _decode_once(
        decoder,
        hidden_d,
        kv_cache=kv_cache,
        page_table_tt=page_table_tt,
        position=seq_len,
        mesh_device=mg_mesh_device,
    )
    decode_pcc = U.pcc(ref_d, out_d)
    record_pcc("decode_real_weights", decode_pcc, position=seq_len, kind=kind_id, weights="real", policy=DEFAULT_POLICY)
    assert decode_pcc >= PCC_BAR, f"real-weight decode PCC {decode_pcc}"


@pytest.mark.real_weights
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_real_weights_non_aligned_and_traced(
    kind_id, kinds, text_config, build_reference, build_optimized_decoder, mg_mesh_device, record_pcc
):
    """Same-contract real-weight coverage for the paths the synthetic tests hold (OPT-012).

    Because the shipped BFP4 policy cannot be accepted on synthetic weights, the structural
    paths that matter most must also be measured *on real weights at the same 0.995 bar*,
    or the acceptance evidence would only cover a tile-aligned single-shot prefill. This
    test carries that requirement:

    * a non-tile-aligned prompt length (777: divisible by neither the tile height nor the
      64-token page block), so the prefill pad/trim path is exercised on real weights;
    * the paged prefill -> paged decode transition on a shuffled page table, so decode reads
      the cache the prefill wrote;
    * a 3-step traced decode replay, so the shipped policy's numbers are the ones the
      traced runtime actually produces.
    """
    kind = kinds[kind_id]
    state_dict = R.load_real_layer_state_dict(kind.layer_idx)
    ref = build_reference(kind.layer_idx, state_dict, tag="real")
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(
        kind.layer_idx, state_dict, tag="real", precision=DEFAULT_POLICY, block_size=block_size
    )

    seq_len = 777  # non-tile-aligned, non-page-aligned
    steps = 3
    hidden = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(seq_len))
    assert hidden.shape == (1, seq_len, text_config.hidden_size)
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + steps)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=1, total_tokens=seq_len + steps, block_size=block_size, page_seed=7771
    )
    out = U.prefill_output(
        decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    )
    assert out.shape == (1, seq_len, text_config.hidden_size)
    prefill_pcc = U.pcc(ref_out, out)
    record_pcc(
        "prefill_real_weights_non_aligned",
        prefill_pcc,
        seq_len=seq_len,
        kind=kind_id,
        weights="real",
        policy=DEFAULT_POLICY,
    )
    assert prefill_pcc >= PCC_BAR, f"real-weight non-aligned prefill PCC {prefill_pcc}"

    hidden_steps = [
        R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(1, offset=seq_len + s))
        for s in range(steps)
    ]
    positions = [seq_len + s for s in range(steps)]

    # Untraced first step: the paged prefill -> paged decode transition itself.
    ref_first = ref.decode(hidden_steps[0], ref_k, ref_v, torch.tensor([positions[0]]))
    first_out = _decode_once(
        decoder,
        hidden_steps[0],
        kv_cache=kv_cache,
        page_table_tt=page_table_tt,
        position=positions[0],
        mesh_device=mg_mesh_device,
    )
    transition_pcc = U.pcc(ref_first, first_out)
    record_pcc(
        "decode_real_weights_non_aligned",
        transition_pcc,
        position=positions[0],
        kind=kind_id,
        weights="real",
        policy=DEFAULT_POLICY,
    )
    assert transition_pcc >= PCC_BAR, f"real-weight decode PCC {transition_pcc} at position {positions[0]}"

    # Traced replay over all three steps. Step 0 repeats the untraced step, which is
    # idempotent in the cache (same input, same position, same K/V written to the same slot).
    x_dev = U.decode_input(hidden_steps[0], mg_mesh_device)
    pos_dev, rope_dev = U.position_tensors([positions[0]], mg_mesh_device)
    decoder.decode_forward(x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev)
    ttnn.synchronize_device(mg_mesh_device)

    trace_id = ttnn.begin_trace_capture(mg_mesh_device, cq_id=0)
    out_dev = decoder.decode_forward(
        x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev
    )
    ttnn.end_trace_capture(mg_mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mg_mesh_device)

    try:
        for step in range(steps):
            host_x = U.host_decode_input(hidden_steps[step], mg_mesh_device)
            host_pos, host_rope = U.position_tensors([positions[step]], mg_mesh_device, device=False)
            ttnn.copy_host_to_device_tensor(host_x, x_dev)
            ttnn.copy_host_to_device_tensor(host_pos, pos_dev)
            ttnn.copy_host_to_device_tensor(host_rope, rope_dev)
            ttnn.execute_trace(mg_mesh_device, trace_id, cq_id=0, blocking=True)
            traced_out = U.decode_output(out_dev)

            if step == 0:
                ref_d = ref_first  # the untraced step above already advanced the reference cache
            else:
                ref_d = ref.decode(hidden_steps[step], ref_k, ref_v, torch.tensor([positions[step]]))
            traced_pcc = U.pcc(ref_d, traced_out)
            record_pcc(
                f"traced_decode_real_weights_step{step}",
                traced_pcc,
                position=positions[step],
                kind=kind_id,
                weights="real",
                policy=DEFAULT_POLICY,
            )
            assert traced_pcc >= PCC_BAR, f"traced real-weight decode PCC {traced_pcc} at {positions[step]}"
    finally:
        ttnn.release_trace(mg_mesh_device, trace_id)


@pytest.mark.slow
@pytest.mark.real_weights
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_stress_repeated_prefill_decode(
    kind_id, kinds, text_config, build_reference, build_optimized_decoder, mg_mesh_device, record_pcc
):
    """Three full prefill + 8-step decode cycles on the shipped policy stay on the bar.

    A single pass cannot catch state that leaks between cycles: a cache buffer that is not
    fully rewritten, an L1 shard that survives into the next allocation, or a program-cache
    entry that binds a stale address. Each cycle allocates a fresh paged cache and page
    table, and every prefill and every decode step is measured against the same golden, so a
    drift shows up as a PCC failure at a specific step rather than as a vague flake.

    Real weights, because :data:`DEFAULT_POLICY` is only accepted against real weights (see
    the module docstring and ``test_synthetic_weight_precision_discrepancy``). Device output
    is additionally required to be bit-identical across cycles.
    """
    kind = kinds[kind_id]
    state_dict = R.load_real_layer_state_dict(kind.layer_idx)
    ref = build_reference(kind.layer_idx, state_dict, tag="real")
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(
        kind.layer_idx, state_dict, tag="real", precision=DEFAULT_POLICY, block_size=block_size
    )

    seq_len, steps, cycles = 512, 8, 3
    hidden = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(seq_len))
    hidden_steps = [
        R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(1, offset=seq_len + s))
        for s in range(steps)
    ]
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + steps)

    # The reference decode chain is computed once (it mutates its K/V buffers in place, so
    # it is inherently sequential) and reused as the golden for every cycle.
    golden_decode = [
        ref.decode(hidden_steps[step], ref_k, ref_v, torch.tensor([seq_len + step])) for step in range(steps)
    ]

    first_prefill = None
    first_decode: list[torch.Tensor] = []
    for cycle in range(cycles):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=seq_len + steps,
            block_size=block_size,
            page_seed=2200 + cycle,
        )
        out = U.prefill_output(
            decoder.prefill_forward(
                U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
            )
        )
        prefill_pcc = U.pcc(ref_out, out)
        record_pcc(
            f"stress_prefill_cycle{cycle}",
            prefill_pcc,
            seq_len=seq_len,
            kind=kind_id,
            weights="real",
            policy=DEFAULT_POLICY,
        )
        assert prefill_pcc >= PCC_BAR, f"cycle {cycle} prefill PCC {prefill_pcc}"

        for step in range(steps):
            out_d = _decode_once(
                decoder,
                hidden_steps[step],
                kv_cache=kv_cache,
                page_table_tt=page_table_tt,
                position=seq_len + step,
                mesh_device=mg_mesh_device,
            )
            decode_pcc = U.pcc(golden_decode[step], out_d)
            record_pcc(
                f"stress_decode_cycle{cycle}_step{step}",
                decode_pcc,
                position=seq_len + step,
                kind=kind_id,
                weights="real",
                policy=DEFAULT_POLICY,
            )
            assert decode_pcc >= PCC_BAR, f"cycle {cycle} decode PCC {decode_pcc} at step {step}"
            if cycle == 0:
                first_decode.append(out_d)
            else:
                assert torch.equal(first_decode[step], out_d), f"cycle {cycle} decode step {step} drifted"

        if cycle == 0:
            first_prefill = out
        else:
            assert torch.equal(first_prefill, out), f"cycle {cycle} prefill drifted from cycle 0"

        for cache in kv_cache:
            cache.deallocate(True)
        page_table_tt.deallocate(True)


@pytest.mark.long_context
@pytest.mark.parametrize("policy", [STRUCTURAL_POLICY, DEFAULT_POLICY])
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_full_context_prefill_and_decode(
    kind_id,
    policy,
    kinds,
    text_config,
    synthetic_state_dicts,
    build_reference,
    build_optimized_decoder,
    mg_mesh_device,
    record_pcc,
):
    """The full advertised context on the *optimized* path: 131072, 131071, decode at 131071.

    The functional stage proved the 131072-token contract with a streaming PCC over every
    query position. This stage rewrote the prefill matmul path underneath it - the row
    fold and its zero padding, the L1-budget-derived program configs and the
    DRAM-width-sharded weight grid - so ``doc/context_contract.json`` must not keep
    advertising 131072 on stage-01 evidence alone.

    What is different at 131072 rather than 8192 is the *number* of chunks (16 instead of
    1) and how far into the page table and the RoPE table they reach, not the fold: each
    ``_prefill_linear`` call still sees one 8192-row chunk. So the reference is driven with
    a block filter that keeps the first, a middle and the last query block of each prompt -
    including the block that lands on the padded tail - rather than every block, which
    keeps one 131072-token host reference per layer kind affordable. The K/V cache is
    compared over the same blocks, and decode runs at the last addressable position on the
    cache the 131071-token prefill wrote.

    Coverage is recorded on every PCC record so the artifact says exactly what was
    measured; ``test_paged_prefill_decode_pcc`` still covers every position at the shorter
    lengths.
    """
    kind = kinds[kind_id]
    state_dict = synthetic_state_dicts[kind.layer_idx]
    ref = build_reference(kind.layer_idx, state_dict)
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, precision=policy, block_size=block_size)
    # The shipped BFP4 policy is measured here too, not only the structural BFP8 one: the
    # advertised 131072 context has to be validated at the precision the layer ships with.
    # On synthetic weights BFP4 sits far below 0.995 by construction (see
    # test_synthetic_weight_precision_discrepancy), so the absolute bar is policy-dependent -
    # and an absolute bar is not the interesting question for BFP4 anyway. The question is
    # whether BFP4 *degrades with length*, so the BFP4 run additionally measures the same
    # policy at 1000 tokens and requires the 131072-token PCC to be within
    # LENGTH_INDEPENDENCE_DELTA of it. That is the length-dependence check the absolute bar
    # cannot make.
    # Only the structural BFP8 policy gets an absolute bar here; the BFP4 metrics are
    # floored at their own 1000-token measurement below (`short_floor`), which is a real
    # number rather than 0.0 and is what gets stamped on each record.
    bar = PCC_BAR

    max_context = text_config.max_position_embeddings
    hidden_size = text_config.hidden_size
    hidden = R.unit_rms_hidden_states((1, max_context, hidden_size), seed=97)

    # For the shipped BFP4 policy an absolute bar on synthetic weights would be arbitrary
    # (see test_synthetic_weight_precision_discrepancy), so every metric is instead floored
    # at its own 1000-token measurement minus SYNTHETIC_LENGTH_DELTA. That floor is a real
    # number, it is what gets stamped on the record, and it catches exactly the failure an
    # absolute bar cannot: accuracy that degrades as the prefix grows.
    short_floor = {}
    if policy != STRUCTURAL_POLICY:
        short_len = 1000
        short_out, short_k, short_v = ref.prefill(hidden[:, :short_len], backend="eager")
        short_kv, short_pt, short_pt_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=short_len + 2,
            block_size=block_size,
            page_seed=771,
        )
        short_dev = decoder.prefill_forward(
            U.prefill_input(hidden[:, :short_len], mg_mesh_device), kv_cache=short_kv, page_table=short_pt_tt
        )
        short_metrics = {
            "prefill": U.pcc(short_out, U.prefill_output(short_dev)),
            "k_cache": U.pcc(
                short_k, U.read_paged_cache(short_kv[0], short_pt, block_size=block_size, seq_len=short_len)
            ),
        }
        short_dev.deallocate(True)
        short_hidden_d = R.unit_rms_hidden_states((1, 1, hidden_size), seed=98)
        cache_len = short_len + 2
        k_pad = torch.zeros(1, short_k.shape[1], cache_len, short_k.shape[3], dtype=short_k.dtype)
        v_pad = torch.zeros_like(k_pad)
        k_pad[:, :, :short_len] = short_k
        v_pad[:, :, :short_len] = short_v
        short_ref_d = ref.decode(short_hidden_d, k_pad, v_pad, torch.tensor([short_len]))
        short_pos, short_rope = U.position_tensors([short_len], mg_mesh_device)
        short_out_d = decoder.decode_forward(
            U.decode_input(short_hidden_d, mg_mesh_device),
            kv_cache=short_kv,
            page_table=short_pt_tt,
            current_pos=short_pos,
            rope_idxs=short_rope,
        )
        short_metrics["decode"] = U.pcc(short_ref_d, U.decode_output(short_out_d))
        short_out_d.deallocate(True)
        for tensor in (short_kv[0], short_kv[1], short_pt_tt):
            tensor.deallocate(True)
        for metric, value in short_metrics.items():
            short_floor[metric] = value - SYNTHETIC_LENGTH_DELTA
            record_pcc(
                f"{metric}_len{short_len}_{policy}_length_reference",
                value,
                seq_len=short_len,
                kind=kind_id,
                policy=policy,
                threshold=bar,
                coverage="full; the length-independence reference for the 131072 run",
            )

    def floor_for(metric):
        """The bar this metric is actually asserted against, and stamped with."""
        return short_floor.get(metric, bar)

    for prefill_len in (max_context, max_context - 1):
        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=max_context,
            block_size=block_size,
            page_seed=1000 + prefill_len % 7,
            pool_multiplier=1,
        )
        hidden_tt = U.prefill_input(hidden[:, :prefill_len], mg_mesh_device)
        out_dev = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
        assert out_dev.shape[-2] == prefill_len
        hidden_tt.deallocate(True)

        # Keep the first, a middle and the last reference query block. The reference picks
        # its own block size from the length; ask it for one that divides both lengths.
        q_chunk = 8192
        kept = {0, (prefill_len // 2 // q_chunk) * q_chunk, ((prefill_len - 1) // q_chunk) * q_chunk}

        stats = R.StreamingPCC()
        blocks = []

        def on_chunk(start, end, out_chunk, _stats=stats, _blocks=blocks, _limit=prefill_len):
            end = min(end, _limit)
            if start >= _limit or out_chunk is None:
                return
            device_slice = ttnn.slice(out_dev, [0, 0, start, 0], [1, 1, end, hidden_size])
            _stats.update(out_chunk[:, : end - start], U.prefill_output(device_slice))
            device_slice.deallocate(True)
            _blocks.append((start, end))

        # Drive the reference with the full 131072 rows even when the device prompt is
        # 131071: attention is causal, so the reference output at position p is identical
        # for both lengths, and this leaves ref_k/ref_v long enough to hold the decode
        # step at position 131071. The streaming comparison is clamped to prefill_len.
        _, ref_k, ref_v = ref.prefill(
            hidden,
            q_chunk=q_chunk,
            q_chunk_filter=lambda start, end: start in kept,
            on_chunk=on_chunk,
            backend="sdpa",
        )
        assert blocks, "no reference query block was computed"
        assert blocks[-1][1] == prefill_len, f"the last block was not covered: {blocks[-1]}"
        covered = sum(end - start for start, end in blocks)
        pcc_value = stats.pcc
        record_pcc(
            f"prefill_len{prefill_len}_{policy}",
            pcc_value,
            seq_len=prefill_len,
            kind=kind_id,
            policy=policy,
            threshold=floor_for("prefill"),
            coverage=f"{covered}/{prefill_len} positions in {len(blocks)} blocks",
        )
        assert pcc_value >= floor_for("prefill"), (
            f"prefill PCC {pcc_value} at seq_len {prefill_len} ({policy}) is below " f"{floor_for('prefill')}"
        )

        cache_pcc = U.pcc(
            ref_k[:, :, :q_chunk],
            U.read_paged_cache(kv_cache[0], page_table, block_size=block_size, seq_len=q_chunk),
        )
        record_pcc(
            f"prefill_k_cache_len{prefill_len}_{policy}",
            cache_pcc,
            seq_len=prefill_len,
            kind=kind_id,
            policy=policy,
            threshold=floor_for("k_cache"),
            coverage=f"first {q_chunk} positions",
        )
        assert cache_pcc >= floor_for("k_cache"), (
            f"K-cache PCC {cache_pcc} at seq_len {prefill_len} ({policy}) is below " f"{floor_for('k_cache')}"
        )
        out_dev.deallocate(True)

        if prefill_len == max_context - 1:
            position = max_context - 1
            ref_k[:, :, position] = 0
            ref_v[:, :, position] = 0
            hidden_d = R.unit_rms_hidden_states((1, 1, hidden_size), seed=98)
            ref_d = ref.decode(hidden_d, ref_k, ref_v, torch.tensor([position]))
            current_pos, rope_idxs = U.position_tensors([position], mg_mesh_device)
            out_d = U.decode_output(
                decoder.decode_forward(
                    U.decode_input(hidden_d, mg_mesh_device),
                    kv_cache=kv_cache,
                    page_table=page_table_tt,
                    current_pos=current_pos,
                    rope_idxs=rope_idxs,
                )
            )
            decode_pcc = U.pcc(ref_d, out_d)
            record_pcc(
                f"decode_max_position_{policy}",
                decode_pcc,
                position=position,
                kind=kind_id,
                policy=policy,
                threshold=floor_for("decode"),
            )
            assert decode_pcc >= floor_for("decode"), (
                f"decode PCC {decode_pcc} at position {position} ({policy}) is below " f"{floor_for('decode')}"
            )

        del ref_k, ref_v
        for cache in kv_cache:
            cache.deallocate(True)
        page_table_tt.deallocate(True)


@pytest.mark.real_weights
@pytest.mark.parametrize("kind_id", KIND_IDS)
def test_real_weights_batched(
    kind_id, kinds, text_config, build_reference, build_optimized_decoder, mg_mesh_device, record_pcc
):
    """Batch > 1 on real weights at the shipped BFP4 policy.

    OPT-012 lists larger batch among the conditions a synthetic-only precision failure has
    to be re-checked under on real weights. The batched paths themselves are
    policy-independent and covered at BFP8 by ``test_batched_prefill_and_decode``; what
    this adds is the shipped policy's real-weight accuracy when four users share a prefill
    and a decode step, which is the case the batch-32 structural test cannot speak to.
    """
    kind = kinds[kind_id]
    state_dict = R.load_real_layer_state_dict(kind.layer_idx)
    ref = build_reference(kind.layer_idx, state_dict, tag="real")
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(
        kind.layer_idx, state_dict, tag="real", precision=DEFAULT_POLICY, block_size=block_size
    )

    batch, seq_len = 4, 512
    hidden = torch.cat(
        [
            R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(seq_len, offset=user * seq_len))
            for user in range(batch)
        ],
        dim=0,
    )
    ref_out, ref_k, ref_v = _ref_prefill_batch(ref, hidden, cache_len=seq_len + 1)

    kv_cache, page_table, page_table_tt = _alloc_paged(
        decoder, mg_mesh_device, batch=batch, total_tokens=seq_len + 1, block_size=block_size, page_seed=515
    )
    out = U.prefill_output(
        decoder.prefill_forward(U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt)
    )
    for user in range(batch):
        user_pcc = U.pcc(ref_out[user], out[user])
        record_pcc(
            f"prefill_real_weights_batch{batch}_user{user}",
            user_pcc,
            seq_len=seq_len,
            kind=kind_id,
            weights="real",
            policy=DEFAULT_POLICY,
        )
        assert user_pcc >= PCC_BAR, f"user {user} prefill PCC {user_pcc}"

    hidden_d = torch.cat(
        [
            R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(1, offset=(user + 1) * seq_len))
            for user in range(batch)
        ],
        dim=0,
    )
    positions = torch.tensor([seq_len] * batch)
    ref_d = ref.decode(hidden_d, ref_k, ref_v, positions)
    current_pos, rope_idxs = U.position_tensors(positions.tolist(), mg_mesh_device)
    out_d = U.decode_output(
        decoder.decode_forward(
            U.decode_input(hidden_d, mg_mesh_device),
            kv_cache=kv_cache,
            page_table=page_table_tt,
            current_pos=current_pos,
            rope_idxs=rope_idxs,
        )
    )
    for user in range(batch):
        user_pcc = U.pcc(ref_d[user], out_d[user])
        record_pcc(
            f"decode_real_weights_batch{batch}_user{user}",
            user_pcc,
            position=seq_len,
            kind=kind_id,
            weights="real",
            policy=DEFAULT_POLICY,
        )
        assert user_pcc >= PCC_BAR, f"user {user} decode PCC {user_pcc}"


@pytest.mark.real_weights
@pytest.mark.slow
@pytest.mark.parametrize("policy", [DEFAULT_POLICY, "bfp8_attn_bfp4_mlp"])
def test_real_weights_length_independence(
    kinds, text_config, build_reference, build_optimized_decoder, mg_mesh_device, record_pcc, policy
):
    """The shipped precision policy must not lose accuracy as the context grows.

    Weight quantisation error is length-independent by construction, but a
    ``full_attention`` layer attends over the *whole* paged prefix, so error in the cached
    K/V compounds with the number of keys in the softmax. On **synthetic** weights that is
    visible: the last query block of a ``full_nope`` prefill measures 0.9576 / 0.9555 /
    0.9498 at 1000 / 8192 / 32768 tokens under ``bfp4_all``, while ``bfp8_attn_bfp4_mlp``
    stays at 0.99149 / 0.99131 / 0.99097 — i.e. it is BFP4 *attention* weights, feeding a
    long-prefix softmax through the cache, and not the MLP.

    This test is the control that decides whether that matters for the shipped policy. It
    runs the same probe on **real checkpoint weights and real activations**, at 1000 and
    12345 tokens, on the ``full_nope`` kind (the only kind that reads a growing prefix; a
    sliding layer never sees more than its 2048-token window). Both lengths must clear the
    0.995 bar, and the long length must be within :data:`LENGTH_INDEPENDENCE_DELTA` of the
    short one.

    Measured drift for the shipped policy is 0.0002 over that range
    (``doc/optimized_decoder/pcc/length_dependence.json``), against 0.017 on synthetic
    weights — the same real-versus-synthetic gap the precision policy was selected on.
    ``full_nope`` is layer 3, so this replays three real host layers per length.
    """
    kind = kinds["full_nope"]
    state_dict = R.load_real_layer_state_dict(kind.layer_idx)
    ref = build_reference(kind.layer_idx, state_dict, tag="real")
    block_size = DEFAULT_BLOCK
    decoder = build_optimized_decoder(kind.layer_idx, state_dict, tag="real", precision=policy, block_size=block_size)

    q_chunk = 8192
    measured = {}
    for length in (1000, 12345):
        hidden = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(length))
        # Only the last query block: it has the longest prefix, so it is the worst case for
        # anything that compounds over the cached K/V.
        last_start = ((length - 1) // q_chunk) * q_chunk if length > q_chunk else 0
        golden = {}

        def on_chunk(start, end, out_chunk, _golden=golden, _limit=length):
            if out_chunk is not None:
                _golden[start] = (min(end, _limit), out_chunk)

        chunk = q_chunk if length > q_chunk else length - (length % 32)
        ref.prefill(
            hidden,
            q_chunk=chunk,
            q_chunk_filter=lambda start, end, _s=last_start: start == _s,
            on_chunk=on_chunk,
            backend="sdpa",
        )
        end, ref_block = golden[last_start]

        kv_cache, page_table, page_table_tt = _alloc_paged(
            decoder,
            mg_mesh_device,
            batch=1,
            total_tokens=length + 2,
            block_size=block_size,
            page_seed=97,
            pool_multiplier=1,
        )
        out = decoder.prefill_forward(
            U.prefill_input(hidden, mg_mesh_device), kv_cache=kv_cache, page_table=page_table_tt
        )
        device_block = ttnn.slice(out, [0, 0, last_start, 0], [1, 1, end, text_config.hidden_size])
        pcc_value = U.pcc(ref_block[:, : end - last_start], U.prefill_output(device_block))
        device_block.deallocate(True)
        out.deallocate(True)
        for tensor in (kv_cache[0], kv_cache[1], page_table_tt):
            tensor.deallocate(True)

        measured[length] = pcc_value
        record_pcc(
            f"prefill_real_weights_last_block_len{length}_{policy}",
            pcc_value,
            seq_len=length,
            kind="full_nope",
            weights="real",
            policy=policy,
            coverage=f"query block [{last_start}, {end})",
        )
        assert pcc_value >= PCC_BAR, f"{policy} real-weight PCC {pcc_value} at {length} tokens"

    drift = measured[1000] - measured[12345]
    record_pcc(
        f"length_drift_1000_to_12345_{policy}",
        1.0 - drift,
        kind="full_nope",
        weights="real",
        policy=policy,
        coverage="1.0 minus the PCC drift, so a higher value is better",
    )
    assert drift <= LENGTH_INDEPENDENCE_DELTA, (
        f"{policy} loses {drift:.6f} PCC between 1000 and 12345 real-weight tokens, more than "
        f"{LENGTH_INDEPENDENCE_DELTA}"
    )
