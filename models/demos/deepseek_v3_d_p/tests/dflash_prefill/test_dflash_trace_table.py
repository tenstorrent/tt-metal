# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: DFlash drafter context-KV read back **through the migration KV chunk address table** vs the golden
55k trace.

Three gates already exist and each covers exactly two of the three things that must be right:

  * ``test_kv_cache_table.py::test_dflash_kv_cache_mock`` — the per-head address table plus an
    every-32-token ``read_device_chunk`` readback, against a ``torch.randn`` MOCK cache. Table and
    readback, but no dflash run and no trace.
  * ``test_dflash.py::test_dflash_pcc`` — a real drafter run PCC'd against the HF reference, read back with
    our OWN host layout math (``ttnn.to_torch`` + ``blockcyclic_positions``). Run and reference, no table.
  * ``tt/runners/prefill_kv_validation.py::kv_cache_pcc_check`` — the runner's KV gate, also host layout
    math. Reference, no table.

Nothing PCCs REAL device KV read through the TABLE against a golden. That is the gap this test closes, and
it is the combination migration actually depends on: the table is what the migration worker reads DRAM
through, so an address-math error is invisible to every gate above and fatal in serving.

Inputs, both READ-only (never copied — the ``/mnt`` golden trees are read-only in CI):

  * hidden-state taps, ``$DFLASH_TRACE_DIR`` (default: the Kimi adapter's ``test_prefill_trace_default``) —
    ``decoder_io/decoder_output_layer_{1,12,24,35,47,58}/rows_*.safetensors``, bf16 ``[56320, 7168]`` each.
    Concatenated on the feature axis they are the drafter's ``target_feature_size`` (6 * 7168 = 43008) input,
    i.e. the REAL verifier residual stream in place of ``test_dflash_pcc``'s ``torch.randn``.
  * golden context K/V, ``$DFLASH_GOLDEN_KV_DIR`` (default ``.../golden/dflash_context_kv_55k``) —
    ``{k,v}_cache.safetensors``, bf16 ``[6, 8, 56320, 128]``, axes ``(draft_layer, kv_head, seq, head_dim)``.
    Those axes ARE the address table's ``(layer, config -> head, position)`` key plus head_dim, so the golden
    indexes straight against the readback: ``read_device_chunk`` returns chunks by NATURAL position (the
    table performs the block-cyclic un-rotation internally), so no host ``blockcyclic_positions`` is needed
    on this path at all.

Which reference gates what, and why they differ. Measured host-side at ctx_len 5120 — the trace taps fed
through the in-process HF drafter, compared to this golden, per (layer, head), min over all 48:

    V   0.999996          K   0.053   (whole-tensor per layer: 0.0725 .. 0.2130)

So golden **V** is an independent, rope-free oracle and is hard-gated here. Golden **K** is reported but NOT
gated: the golden's K does not agree with what the checkpoint's own ``rope_parameters`` block produces
(``deepseek_yarn``, theta 50000, factor 64, original_max_position_embeddings 4096 — which is what
``conftest.normalize_rope_config`` hands the reference), while its V, which never touches RoPE, agrees to
1e-5 through the same code path. That is the A/B; this test does not assert anything about what produced the
file. K's hard gate is instead the in-process HF reference built from the SAME trace taps, which covers the
whole K pipeline including RoPE and k_norm.

    DFLASH_HF_MODEL=/mnt/models/Kimi-K2.6-DFlash MESH_DEVICE=8x4 \
    pytest models/demos/deepseek_v3_d_p/tests/dflash_prefill/test_dflash_trace_table.py -svv
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.tests.dflash_prefill.test_dflash import (
    _FABRIC_2D,
    CHUNK_GLOBAL,
    MAX_SEQ_LEN,
    PCC_THRESHOLD,
    _read_cache_natural,
)
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.tt_dflash_drafter import TtDFlashDrafter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6 import KimiK26Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import _dram_chunk_size_bytes
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_dflash_kv_cache,
    populate_kv_chunk_address_table_dflash,
)
from models.demos.deepseek_v3_d_p.utils.test_utils import read_sharded_rows
from tests.ttnn.utils_for_testing import comp_pcc

TRACE_DIR_ENV = "DFLASH_TRACE_DIR"
GOLDEN_KV_ENV = "DFLASH_GOLDEN_KV_DIR"
# Mirrors ``HF_ENV`` in this directory's conftest.py. Inlined rather than imported on purpose: importing that
# conftest by dotted path pulls in transformers + the vendored reference modeling (~120 s), and pytest then
# imports the same file again under its own module identity — i.e. ~4 minutes of COLLECTION time added to
# every leg that merely collects this directory, in exchange for one string. If it ever drifts, the only
# symptom is a skip reason naming the wrong variable.
HF_ENV = "DFLASH_HF_MODEL"

# The drafter golden lives beside the verifier's structured traces but is a separate artifact with its own
# layout (whole-tensor safetensors, not row-sharded), so it gets its own env var rather than riding
# PREFILL_TRACE_DIR — a CI leg may point that at a DeepSeek trace, which has no decoder_io/ at all.
GOLDEN_KV_DEFAULT = "/mnt/models/deepseek-prefill-cache/golden/dflash_context_kv_55k"

# Host-side agreement required between the golden and the in-process HF reference BEFORE any device work.
# Measured 0.999996 for V at ctx 5120 (see the module docstring); this fails fast and unambiguously if the
# trace dir and the golden dir are ever paired with different runs.
GOLDEN_V_VS_REF_THRESHOLD = 0.9999

# Golden K is compared and logged but not gated — see the module docstring. Set this to a float to promote
# the golden-K comparison to a hard gate once the RoPE-base question is settled upstream.
GOLDEN_K_PCC_THRESHOLD = None


# Resolved at IMPORT time so a missing input skips at COLLECTION. Deferring these checks into the test body
# does not work: the ``mesh_device`` fixture is set up first and would bring up the 8x4 galaxy under
# FABRIC_2D only to skip immediately afterwards.
TRACE_DIR = Path(os.environ.get(TRACE_DIR_ENV, KimiK26Adapter.test_prefill_trace_default))
GOLDEN_DIR = Path(os.environ.get(GOLDEN_KV_ENV, GOLDEN_KV_DEFAULT))


def _missing_inputs() -> list:
    """Which of the three required inputs are absent, as a human-readable list for the skip reason."""
    missing = []
    if not (TRACE_DIR / "decoder_io").is_dir():
        # decoder_io/ specifically, not just the dir: a DeepSeek trace tree exists but carries no taps, so
        # pointing DFLASH_TRACE_DIR at one has to skip rather than fail deep inside read_sharded_rows.
        missing.append(f"${TRACE_DIR_ENV} (hidden-state taps; want {TRACE_DIR}/decoder_io/)")
    if not all((GOLDEN_DIR / f"{n}_cache.safetensors").exists() for n in ("k", "v")):
        missing.append(f"${GOLDEN_KV_ENV} (drafter golden; want {GOLDEN_DIR}/{{k,v}}_cache.safetensors)")
    hf = os.environ.get(HF_ENV)
    if not hf or not Path(hf).exists():
        # Mirrors the hf_drafter fixture's own skip; repeated here only to move it earlier than mesh_device.
        missing.append(f"${HF_ENV} (drafter checkpoint, for the in-process reference)")
    return missing


_MISSING_INPUTS = _missing_inputs()


def _load_trace_ctx(trace_dir: Path, target_layer_ids, total_len: int, hidden_size: int) -> torch.Tensor:
    """The drafter's context input from the verifier trace: ``[1, total_len, len(tids) * hidden_size]`` fp32.

    Concatenation ORDER is the contract — ``TtDFlashDrafter.tap`` and the HF reference both slice feature
    block ``j`` as ``[j * H : (j+1) * H]`` for the ``j``-th entry of ``target_layer_ids``, so the taps must be
    concatenated in that same order. A permuted concat still produces a plausible cache; only the golden can
    see it (the in-process reference is fed the identical tensor, so it is blind to tap order by
    construction). That is the single most valuable thing the golden adds over ``test_dflash_pcc``.
    """
    blocks = []
    for tid in target_layer_ids:
        key = f"decoder_output_layer_{tid}"
        rows = read_sharded_rows(trace_dir / "decoder_io" / key, key, 0, total_len)
        assert rows.shape == (
            total_len,
            hidden_size,
        ), f"{key}: got {tuple(rows.shape)}, want {(total_len, hidden_size)}"
        blocks.append(rows)
    ctx = torch.cat(blocks, dim=-1).unsqueeze(0)
    logger.info(f"trace taps {list(target_layer_ids)} -> ctx {tuple(ctx.shape)} from {trace_dir}")
    return ctx


def _load_golden_kv(golden_dir: Path, total_len: int, num_layers: int, num_kv_heads: int, head_dim: int):
    """Golden context K and V as ``[num_layers, num_kv_heads, total_len, head_dim]`` fp32.

    The shape assert is the axis-mapping check: the golden's ``(draft_layer, kv_head, seq, head_dim)`` must
    line up with ``(config_id - base, layer, position)`` exactly, so if the artifact ever changes axis order
    this fails here instead of producing a uniformly bad PCC that reads like a device bug.
    """
    out = []
    for name in ("k_cache", "v_cache"):
        # Presence is a collection-time skip (see _missing_inputs), so by here the files exist.
        with safe_open(golden_dir / f"{name}.safetensors", framework="pt") as f:
            sl = f.get_slice(name)
            shape = list(sl.get_shape())
            assert shape[0] == num_layers and shape[1] == num_kv_heads and shape[3] == head_dim, (
                f"{name} shape {shape} does not match (draft_layer={num_layers}, kv_head={num_kv_heads}, "
                f"seq, head_dim={head_dim}) — the golden's axes must be (layer, head, seq, head_dim)"
            )
            assert shape[2] >= total_len, f"{name} has only {shape[2]} positions, need {total_len}"
            out.append(sl[:, :, :total_len, :].to(torch.float32))
    logger.info(f"golden K/V {tuple(out[0].shape)} from {golden_dir}")
    return out[0], out[1]


def _read_cache_via_table(lookup_table, base_config_id, num_layers, num_kv_heads, total_len, head_dim, slot=0):
    """Read a drafter K or V cache back ENTIRELY through the address table.

    Returns ``[num_layers, num_kv_heads, total_len, head_dim]`` fp32 in NATURAL token order. Natural, not
    rotated: ``populate_kv_chunk_address_table_dflash`` registers chip ``row``'s local block
    ``seq_chunk * blocks_per_chunk_local + j`` under position ``seq_chunk * chunk_size_global +
    row * tokens_per_chunk_local + j * 32`` — i.e. the table already encodes the writer's block-cyclic
    staircase, so this path never touches ``blockcyclic_positions``. It is therefore an INDEPENDENT inverse
    of the same permutation ``_read_cache_natural`` computes on the host, which is what makes comparing the
    two (below) worth doing.
    """
    out = torch.empty(num_layers, num_kv_heads, total_len, head_dim, dtype=torch.float32)
    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]
    for layer in range(num_layers):
        for head_idx in range(num_kv_heads):
            for position in range(0, total_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                raw_bytes = lookup_table.read_device_chunk(
                    layer=layer, position=position, slot=slot, config_id=base_config_id + head_idx
                )
                chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
                chunk = ttnn.to_torch(chunk_tt).to(torch.float32)
                out[layer, head_idx, position : position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK] = chunk[0, 0]
    return out


def _pcc_per_head(tag: str, expected: torch.Tensor, actual: torch.Tensor, threshold, num_layers: int) -> float:
    """PCC every (layer, head) slice separately and return the minimum; assert when ``threshold`` is a float.

    Per-head, not whole-tensor: PCC dilutes as 1 - (fraction wrong), so one wrong head out of 48 lands at
    ~0.99 on the whole tensor and hides under any threshold worth setting. A per-head split also localizes
    the fault — all heads of one TP column bad means the device-group/column mapping, one head per chip bad
    means the ``head_idx % heads_per_chip`` term, one layer bad means the head/layer shard stride.
    """
    worst, worst_at = 1.0, None
    for i in range(num_layers):
        heads = []
        for h in range(expected.shape[1]):
            _, pcc = comp_pcc(expected[i, h], actual[i, h])
            heads.append(pcc)
            if pcc < worst:
                worst, worst_at = pcc, (i, h)
        _, whole = comp_pcc(expected[i], actual[i])
        logger.info(
            f"  {tag} layer {i}: whole={whole:.6f}  per-head min={min(heads):.6f} (head {heads.index(min(heads))}) "
            f"max={max(heads):.6f}"
        )
        if threshold is not None:
            for h, pcc in enumerate(heads):
                assert pcc > threshold, f"{tag} layer {i} head {h}: PCC {pcc:.6f} <= {threshold}"
    logger.info(f"  --> {tag} min over all (layer, head) = {worst:.6f} at {worst_at}")
    return worst


# pytest.ini caps every test at timeout=300, which does not fit FABRIC_2D 8x4 bring-up + a pretrained
# safetensors load + an fp32 HF drafter forward + ~15k-31k read_device_chunk calls (~0.37 ms each: 6 s at
# ctx 5120, 11 s at 10240). Disabled as the sibling table tests and test_dflash.py already do.
@pytest.mark.timeout(0)
@pytest.mark.skipif(bool(_MISSING_INPUTS), reason="missing input(s): " + "; ".join(_MISSING_INPUTS))
# pretrained ONLY, and as an explicit single-value param rather than a shared axis: a random-weight drafter
# cannot be compared to a golden produced by the real checkpoint, and adding a second value would double a
# CI leg against a ~93-minute step budget. The literal id keeps this test inside the existing
# `-k "mesh-8x4 and pretrained"` selection.
@pytest.mark.parametrize("use_pretrained", [True], ids=["pretrained"], indirect=True)
@pytest.mark.parametrize(
    "ctx_len, n_chunks, cache_seq",
    [
        # The simple single-run case: one 5k chunk into a cache exactly that deep. Note what it CANNOT see —
        # with cache_seq == chunk_global the table's band term (seq_chunk * blocks_per_chunk_local) is
        # identically 0 and blocks_local == blocks_per_chunk_local, so the head/layer stride and the seq
        # stride are numerically indistinguishable. Hence the two rows below.
        pytest.param(CHUNK_GLOBAL, 1, CHUNK_GLOBAL, id="ctx5k-1chunk-tightcache"),
        # Two aligned 5k chunks written, two bands addressed: exercises the band term and the writer's
        # non-degenerate block-cyclic placement.
        pytest.param(2 * CHUNK_GLOBAL, 2, 2 * CHUNK_GLOBAL, id="ctx10k-2chunk"),
        # The PRODUCTION cache shape (tt_prefill_runtime allocates max_seq_len, not the chunk size): one
        # chunk written into a 12-band cache, so blocks_local (240) is an order of magnitude larger than the
        # band term and a swapped head/layer/seq stride can no longer alias. Pin THIS row if the test goes
        # into CI.
        pytest.param(CHUNK_GLOBAL, 1, MAX_SEQ_LEN, id="ctx5k-1chunk-deepcache"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            _FABRIC_2D,
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_trace_kv_cache_table(
    mesh_device,
    device_params,
    num_links,
    topology,
    ctx_len,
    n_chunks,
    cache_seq,
    use_pretrained,
    drafter_cfg,
    drafter_state_dict,
    hf_context_kv,
):
    cfg = drafter_cfg
    sd = drafter_state_dict

    trace_dir, golden_dir = TRACE_DIR, GOLDEN_DIR

    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    num_layers, num_kv_heads, head_dim = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    assert num_kv_heads % tp == 0, f"num_kv_heads {num_kv_heads} not divisible by tp {tp}"
    heads_per_chip = num_kv_heads // tp
    assert heads_per_chip > 1, "the per-head strided bank walk is only exercised with >1 head per chip"

    assert ctx_len % n_chunks == 0, f"ctx_len {ctx_len} not divisible by n_chunks {n_chunks}"
    chunk_global = ctx_len // n_chunks
    # The table's period must equal the period the cache was WRITTEN at; a mismatch yields addresses that
    # look plausible and are wholly wrong, so pass chunk_size_global explicitly everywhere below.
    assert cache_seq % chunk_global == 0, f"cache_seq {cache_seq} is not a whole number of {chunk_global}-token bands"
    assert ctx_len <= cache_seq, f"ctx_len {ctx_len} exceeds cache_seq {cache_seq}"
    logger.info(
        f"ctx_len={ctx_len} n_chunks={n_chunks} chunk_global={chunk_global} cache_seq={cache_seq} "
        f"sp={sp} tp={tp} heads/chip={heads_per_chip} layers={num_layers}"
    )

    # ---- references (host only; fails before any drafter/cache work if the two artifacts disagree) ----
    ctx = _load_trace_ctx(trace_dir, cfg.target_layer_ids, ctx_len, cfg.hidden_size)
    golden_k, golden_v = _load_golden_kv(golden_dir, ctx_len, num_layers, num_kv_heads, head_dim)
    real = hf_context_kv(ctx)  # {layer: (k, v)} fp32, from the SAME trace taps
    ref_k = torch.stack([real[i][0] for i in range(num_layers)])
    ref_v = torch.stack([real[i][1] for i in range(num_layers)])

    logger.info("golden vs in-process HF reference (both from the same trace taps), host side:")
    _pcc_per_head("golden-V vs ref-V", ref_v, golden_v, GOLDEN_V_VS_REF_THRESHOLD, num_layers)
    # Logged, never gated: V agrees to ~1e-5 through this same path while K does not, which localizes the
    # disagreement to the only stage V skips (RoPE). Reported as measurement, not as a claim about the file.
    k_ref_vs_golden = _pcc_per_head("golden-K vs ref-K", ref_k, golden_k, None, num_layers)
    logger.info(
        f"golden-K vs ref-K min = {k_ref_vs_golden:.6f} — expected LOW (~0.05 at ctx 5120). V never touches "
        f"RoPE and agrees to ~1e-5, so K's hard gate below is the in-process reference, not the golden."
    )

    # ---- device: one drafter, streamed chunk by chunk exactly as the prefill runner does ----
    drafter = TtDFlashDrafter(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=cache_seq,
        num_links=num_links,
        topology=topology,
    )
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3  # tap hidden TP-sharded on the hidden dim
    hidden_shard[sp_axis] = 2  # ALSO SP-shard the tap on seq -> each chip taps its own [chunk_local] rows
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    k_cache, v_cache = allocate_dflash_kv_cache(mesh_device, cfg, cache_seq, sp_axis=sp_axis, tp_axis=tp_axis)
    H = cfg.hidden_size
    for c in range(n_chunks):
        lo = c * chunk_global
        drafter.reset()
        for j, tid in enumerate(cfg.target_layer_ids):
            h_j = ctx[:, lo : lo + chunk_global, j * H : (j + 1) * H].to(torch.bfloat16).reshape(1, 1, chunk_global, H)
            h_tt = ttnn.from_torch(
                h_j,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            drafter.tap(h_tt, tid)
        # Every write here is ALIGNED (lo is a multiple of chunk_global), so a plain SP shard of the chunk's
        # natural rows is exactly what the writer expects. Unaligned resume offsets are test_dflash.py's
        # test_dflash_multiturn_pcc; this test is about the readback path, not the rotation.
        drafter.forward(k_cache, v_cache, lo, slot_idx=0)
    ttnn.synchronize_device(mesh_device)

    # ---- build the address table over the caches that were just written ----
    # 2 * num_kv_heads configs: the table key is (layer, position, slot) with NO head axis, so every
    # (tensor, head) needs its own config, and each config's device groups are SINGLE-MEMBER because TP
    # carries distinct heads rather than replicas. Config id order K then V is the src<->dst migration
    # contract (same order as test_dflash_kv_cache_mock).
    chunk_size_bytes = _dram_chunk_size_bytes(k_cache)
    assert chunk_size_bytes == _dram_chunk_size_bytes(v_cache) == (head_dim // 32) * 1088, chunk_size_bytes
    assert k_cache.buffer_address() != v_cache.buffer_address(), "K and V must be distinct allocations"
    K_BASE, V_BASE = 0, num_kv_heads

    def _table_config():
        c = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        c.num_layers = num_layers
        c.max_sequence_length = cache_seq
        c.num_slots = 1
        c.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        c.chunk_size_bytes = chunk_size_bytes
        return c

    configs = [_table_config() for _ in range(2 * num_kv_heads)]
    lookup_table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs)
    assert lookup_table.num_configs() == 2 * num_kv_heads, f"got {lookup_table.num_configs()}"
    for base, cache in ((K_BASE, k_cache), (V_BASE, v_cache)):
        for head_idx in range(num_kv_heads):
            populate_kv_chunk_address_table_dflash(
                lookup_table=lookup_table,
                config=configs[base + head_idx],
                mesh_device=mesh_device,
                mesh_shape=list(mesh_shape),
                seq_len=cache_seq,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                kv_cache=cache,
                chunk_size_bytes=chunk_size_bytes,
                num_kv_heads=num_kv_heads,
                head_idx=head_idx,
                num_users=1,
                config_id=base + head_idx,
                chunk_size_global=chunk_global,
            )

    # ---- readback 1: through the table (the path under test) ----
    n_reads = 2 * num_layers * num_kv_heads * (ctx_len // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)
    logger.info(f"reading {n_reads} 32-token chunks through the address table")
    dk = _read_cache_via_table(lookup_table, K_BASE, num_layers, num_kv_heads, ctx_len, head_dim)
    dv = _read_cache_via_table(lookup_table, V_BASE, num_layers, num_kv_heads, ctx_len, head_dim)

    # ---- readback 2: the existing host-math path, as a cross-check of two independent inverses ----
    # Both decode the SAME bfp8 DRAM bytes, so they must agree EXACTLY. This is a free, sharp check of the
    # one thing neither PCC can isolate: that the table's address arithmetic and blockcyclic_positions agree
    # on where every token lives, AND that both order heads as col * heads_per_chip + h_local (the table
    # computes that explicitly; ConcatMesh2dToTensor(dims=(2, 1)) produces it by concatenating TP columns in
    # order). A mismatch here means one of the two inverses is wrong — the golden comparison says which.
    dk_host = _read_cache_natural(k_cache, mesh_device, mesh_shape, sp, chunk_global, num_layers, ctx_len)
    dv_host = _read_cache_natural(v_cache, mesh_device, mesh_shape, sp, chunk_global, num_layers, ctx_len)
    for tag, tbl, host in (("K", dk, dk_host), ("V", dv, dv_host)):
        assert tbl.shape == host.shape, f"{tag}: table readback {tuple(tbl.shape)} vs host {tuple(host.shape)}"
        max_diff = (tbl - host.float()).abs().max().item()
        logger.info(f"  {tag}: table readback vs host-math readback max|diff| = {max_diff:g}")
        assert max_diff == 0.0, (
            f"{tag}: the address-table readback and the blockcyclic_positions readback disagree "
            f"(max|diff| {max_diff:g}) over identical DRAM bytes — one of the two layout inverses is wrong"
        )

    # ---- the gates ----
    # Threshold: the same bar as test_dflash.py's device-vs-HF gate. That path already reads this bfp8 cache
    # back, so 0.999 is calibrated WITH the cache quantization included; and the golden agrees with the
    # in-process reference to ~1e-5 (V), so the same bar carries over to the golden comparison.
    logger.info(f"device (read via the KV chunk address table) vs golden trace, threshold {PCC_THRESHOLD}:")
    _pcc_per_head("dev-V vs golden-V", golden_v, dv, PCC_THRESHOLD, num_layers)
    _pcc_per_head("dev-K vs golden-K", golden_k, dk, GOLDEN_K_PCC_THRESHOLD, num_layers)

    logger.info(f"device (read via the KV chunk address table) vs in-process HF reference, {PCC_THRESHOLD}:")
    # V first: V never touches RoPE, so V-passes-while-K-fails localizes the fault to the rope table rather
    # than to the taps, the tap ORDER, the writer offset, or the table's addressing.
    _pcc_per_head("dev-V vs ref-V", ref_v, dv, PCC_THRESHOLD, num_layers)
    _pcc_per_head("dev-K vs ref-K", ref_k, dk, PCC_THRESHOLD, num_layers)
