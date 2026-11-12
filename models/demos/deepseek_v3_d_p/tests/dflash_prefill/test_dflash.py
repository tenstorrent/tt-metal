# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Tenstorrent DFlash drafter (``TtDFlashDrafter``) vs z-lab HF ``DFlashDraftModel``.

This is the validation of the *validator* — it compares the device's context-KV build to the
ground truth produced by the actual HF drafter's forward.

    DFLASH_HF_MODEL=/path/to/Kimi-K2.x-DFlash MESH_DEVICE=8x4 \
    pytest models/demos/deepseek_v3_d_p/tests/dflash_prefill/test_dflash.py -svv

``test_dflash_golden_pcc`` instead feeds a GPU trace's own taps and compares against the context K/V the GPU
drafter built from them (layouts in ``_load_golden_taps`` and ``dflash_kv_validation._load_golden_kv``).
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_kv_validation import GOLDEN_KV_ENV, _load_golden_kv
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.tt_dflash_drafter import TtDFlashDrafter
from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_to_halfsplit_perm
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_dflash_kv_cache
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.999
# PCC is scale-invariant, so a K off by a constant (e.g. a missing yarn cos/sin amplitude) still passes it.
K_SCALE_TOL = 0.01
# The GPU trace's taps that the golden K/V (GOLDEN_KV_ENV) was built from.
GOLDEN_TAPS_ENV = "PREFILL_DFLASH_GOLDEN_TAPS_DIR"

# The production chunk width: 5120 global, 640 per chip on the target 8x4 mesh (sp=8).
CHUNK_GLOBAL = PREFILL_CHUNK_TOKENS

# Per-user cache depth
MAX_SEQ_LEN = 11 * CHUNK_GLOBAL


def _unrotate_blockcyclic(rotated: torch.Tensor, sp: int, chunk_global: int) -> torch.Tensor:
    """Un-rotate a drafter cache read back as ``[.., .., cache_len, head_dim]`` in block-cyclic
    shard-row order (an SP-contiguous concat of dim 2) into natural token order.
    """
    cache_len = rotated.shape[2]
    p = blockcyclic_positions(sp, chunk_global, cache_len)
    natural = torch.zeros_like(rotated)
    natural[:, :, p, :] = rotated
    return natural


def _reshuffle_k_to_interleaved_layout(rk: torch.Tensor, cfg) -> torch.Tensor:
    """Reindex the HALF-SPLIT HF reference K (``rk``) to the drafter's persisted-K convention before the PCC
    compare. ``hf_context_kv`` ropes K half-split; the meta-rope drafter persists K interleaved (the same K
    with its head_dim ``src``-permuted, ``interleaved[j] == halfsplit[src[j]]``), so under ``"interleaved"``
    reindex the reference by ``src`` to compare like with like. V never touches rope, so it is untouched."""
    if cfg.rope_convention == "interleaved":
        src = torch.argsort(interleaved_to_halfsplit_perm(cfg.head_dim))
        return rk[..., src]
    return rk


def _read_cache_natural(cache, mesh_device, mesh_shape, sp: int, chunk_global: int, num_layers: int, out_len: int):
    """Read a drafter K/V cache back as ``[num_layers, kv_heads, out_len, head_dim]`` in natural token order.

    The cache is SP-sharded on seq and TP-sharded on kv-head, so concat SP along seq (dim 2) and TP along
    kv-head (dim 1). Slot 0 is dim0 rows ``[0, num_layers)`` under the writer's user-major linearization
    (``slot = user_id * num_layers + layer_idx``).
    """
    host = ttnn.to_torch(
        cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_shape)
    )
    natural = _unrotate_blockcyclic(host[:num_layers].float(), sp, chunk_global)
    return natural[:, :, :out_len, :]


def _load_golden_taps(taps_dir: Path, target_layer_ids, out_len: int) -> list:
    """The GPU trace's taps for positions ``[0, out_len)``: one ``[out_len, hidden]`` bf16 tensor per target layer,
    in ``target_layer_ids`` order. ``tap_layer_{L}.safetensors`` holds tensor ``tap_layer_{L}`` of shape
    ``[seq, hidden]``, the verifier hidden state the drafter taps for target layer ``L``."""
    taps = []
    for tid in target_layer_ids:
        with safe_open(taps_dir / f"tap_layer_{tid}.safetensors", framework="pt") as f:
            sl = f.get_slice(f"tap_layer_{tid}")
            seq = sl.get_shape()[0]
            assert seq >= out_len, f"tap_layer_{tid} has only {seq} positions, need {out_len}"
            taps.append(sl[:out_len].to(torch.bfloat16))
    return taps


@pytest.mark.timeout(0)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.parametrize(
    "ctx_len, n_chunks",
    [
        pytest.param(5120, 1, id="ctx5k-1chunk"),
        pytest.param(10240, 2, id="ctx10k-2chunk"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_pcc(
    mesh_device,
    device_params,
    num_links,
    ctx_len,
    n_chunks,
    use_pretrained,
    drafter_cfg,
    drafter_state_dict,
    hf_context_kv,
):
    topology = per_axis_topology(device_params["fabric_config"])[1]
    logger.info(f"weights={'pretrained' if use_pretrained else 'random'}  ctx_len={ctx_len}  n_chunks={n_chunks}")
    cfg = drafter_cfg
    sd = drafter_state_dict

    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert cfg.num_key_value_heads % tp == 0, f"num_kv_heads {cfg.num_key_value_heads} not divisible by tp {tp}"
    H = cfg.hidden_size
    assert ctx_len % n_chunks == 0, f"ctx_len {ctx_len} not divisible by n_chunks {n_chunks}"
    chunk_global = ctx_len // n_chunks

    gen = torch.Generator().manual_seed(0)
    ctx = torch.randn(1, ctx_len, cfg.target_feature_size, generator=gen, dtype=torch.float32)

    # ---- ground truth: the REAL HF drafter forward (context slice of its KV cache) ----
    real = hf_context_kv(ctx)

    # ---- device ----
    drafter = TtDFlashDrafter(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=ctx_len,
        chunk_size=chunk_global,
        num_links=num_links,
        topology=topology,
    )
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3  # tap hidden TP-sharded on the hidden dim
    hidden_shard[sp_axis] = 2  # ALSO SP-shard the tap on seq → each chip taps its own [seq/sp] slice
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    # Caller owns the K/V caches (like the MLA prefill runner) and passes them into forward().
    # Cache depth is the FULL context, so with n_chunks > 1 it spans several aligned 5k chunks.
    k_cache, v_cache = allocate_dflash_kv_cache(mesh_device, cfg, ctx_len, sp_axis=sp_axis, tp_axis=tp_axis)

    # Stream the context chunk by chunk
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
        # 3rd arg is the chunk's absolute global KV offset (0 for the single-chunk case).
        drafter.forward(k_cache, v_cache, lo)
    ttnn.synchronize_device(mesh_device)

    # [num_layers, kv_heads, ctx_len, head_dim] in natural token order
    dk = _read_cache_natural(k_cache, mesh_device, mesh_shape, sp, chunk_global, cfg.num_hidden_layers, ctx_len)
    dv = _read_cache_natural(v_cache, mesh_device, mesh_shape, sp, chunk_global, cfg.num_hidden_layers, ctx_len)

    for i in range(cfg.num_hidden_layers):
        rk, rv = real[i]
        rk = _reshuffle_k_to_interleaved_layout(rk, cfg)  # HF ref is half-split; device persists interleaved K
        ok_k, pcc_k = comp_pcc(rk, dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(rv, dv[i], PCC_THRESHOLD)
        k_scale = (dk[i].norm() / rk.norm()).item()
        logger.info(f"layer {i}: K pcc={pcc_k} (ok={ok_k}) scale={k_scale:.4f}  V pcc={pcc_v} (ok={ok_v})")
        # V (matmul-only) should be ~1.0; if V passes but K fails, suspect the RoPE (deepseek-yarn vs the
        # trained model's rope) or k_norm, not the weights.
        assert ok_v, f"V layer {i}: device vs HF PCC {pcc_v} < {PCC_THRESHOLD} (matmul/weights mismatch)"
        assert ok_k, f"K layer {i}: device vs HF PCC {pcc_k} < {PCC_THRESHOLD} (norm/rope mismatch if V passed)"
        assert abs(k_scale - 1.0) <= K_SCALE_TOL, f"K layer {i}: ||device|| / ||HF|| = {k_scale:.4f} (yarn amplitude?)"


@pytest.mark.timeout(0)
@pytest.mark.skipif(
    not (os.environ.get(GOLDEN_KV_ENV) and os.environ.get(GOLDEN_TAPS_ENV)),
    reason=f"set {GOLDEN_KV_ENV} (drafter golden K/V) and {GOLDEN_TAPS_ENV} (the taps it was built from)",
)
@pytest.mark.parametrize("use_pretrained", [True], ids=["pretrained"], indirect=True)
@pytest.mark.parametrize(
    "ctx_len, n_chunks",
    [
        pytest.param(5120, 1, id="ctx5k-1chunk"),
        pytest.param(56320, 11, id="ctx55k-11chunk"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_golden_pcc(
    mesh_device,
    device_params,
    num_links,
    ctx_len,
    n_chunks,
    use_pretrained,
    drafter_cfg,
    drafter_state_dict,
):
    """Teacher-forced against a GPU trace: the trace's own taps go in, and the device context K/V must match the
    K/V the GPU drafter built from those same taps. The verifier is not involved, so this measures the drafter
    alone and holds it to the same threshold as against HF."""
    topology = per_axis_topology(device_params["fabric_config"])[1]
    cfg = drafter_cfg
    sd = drafter_state_dict
    golden_dir, taps_dir = Path(os.environ[GOLDEN_KV_ENV]), Path(os.environ[GOLDEN_TAPS_ENV])
    logger.info(f"golden K/V={golden_dir}  taps={taps_dir}  ctx_len={ctx_len}  n_chunks={n_chunks}")

    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert cfg.num_key_value_heads % tp == 0, f"num_kv_heads {cfg.num_key_value_heads} not divisible by tp {tp}"
    H = cfg.hidden_size
    assert ctx_len % n_chunks == 0, f"ctx_len {ctx_len} not divisible by n_chunks {n_chunks}"
    chunk_global = ctx_len // n_chunks

    taps = _load_golden_taps(taps_dir, cfg.target_layer_ids, ctx_len)
    gk, gv = _load_golden_kv(
        golden_dir,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        out_len=ctx_len,
        rope_convention=cfg.rope_convention,  # reindexes the half-split golden K to the device's stored K
    )

    drafter = TtDFlashDrafter(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=ctx_len,
        chunk_size=chunk_global,
        num_links=num_links,
        topology=topology,
    )
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3  # tap hidden TP-sharded on the hidden dim
    hidden_shard[sp_axis] = 2  # ALSO SP-shard the tap on seq → each chip taps its own [seq/sp] slice
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)
    k_cache, v_cache = allocate_dflash_kv_cache(mesh_device, cfg, ctx_len, sp_axis=sp_axis, tp_axis=tp_axis)

    for c in range(n_chunks):
        lo = c * chunk_global
        drafter.reset()
        for tap, tid in zip(taps, cfg.target_layer_ids):
            h_tt = ttnn.from_torch(
                tap[lo : lo + chunk_global].reshape(1, 1, chunk_global, H),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            drafter.tap(h_tt, tid)
        drafter.forward(k_cache, v_cache, lo)
    ttnn.synchronize_device(mesh_device)

    dk = _read_cache_natural(k_cache, mesh_device, mesh_shape, sp, chunk_global, cfg.num_hidden_layers, ctx_len)
    dv = _read_cache_natural(v_cache, mesh_device, mesh_shape, sp, chunk_global, cfg.num_hidden_layers, ctx_len)

    for i in range(cfg.num_hidden_layers):
        ok_k, pcc_k = comp_pcc(gk[i], dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(gv[i], dv[i], PCC_THRESHOLD)
        k_scale = (dk[i].norm() / gk[i].norm()).item()
        logger.info(f"layer {i}: K pcc={pcc_k} (ok={ok_k}) scale={k_scale:.4f}  V pcc={pcc_v} (ok={ok_v})")
        assert ok_v, f"V layer {i}: device vs GPU PCC {pcc_v} < {PCC_THRESHOLD} (weights or tap order)"
        assert ok_k, f"K layer {i}: device vs GPU PCC {pcc_k} < {PCC_THRESHOLD} (norm/rope mismatch if V passed)"
        assert abs(k_scale - 1.0) <= K_SCALE_TOL, f"K layer {i}: ||device|| / ||GPU|| = {k_scale:.4f} (yarn amplitude?)"


_MULTITURN_ITERS = [
    pytest.param([640, 5120], id="aligned_min"),  # turn 0 = 1 chip valid (7 chips pad), then chip-1 rotated
    pytest.param([672, 5120], id="midchip_straddle"),  # frontier 1 tile into chip 1 → offset=32 straddle
    pytest.param([4480, 5120], id="lastchip"),  # turn 0 = 7 chips, rotation at the LAST chip (chip 7)
    pytest.param([1280, 1920, 5120], id="rot_partial"),  # turn 1 is rotated AND partial (3 valid, 5 pad)
    pytest.param([5120, 1280, 5120], id="multichunk"),  # rotation in aligned chunk 1, partial then full
]


@pytest.mark.timeout(0)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.parametrize("iters_isl", _MULTITURN_ITERS)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_multiturn_pcc(
    mesh_device,
    device_params,
    num_links,
    iters_isl,
    use_pretrained,
    drafter_cfg,
    drafter_state_dict,
    hf_context_kv,
):
    topology = per_axis_topology(device_params["fabric_config"])[1]
    cfg = drafter_cfg
    sd = drafter_state_dict

    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert cfg.num_key_value_heads % tp == 0, f"num_kv_heads {cfg.num_key_value_heads} not divisible by tp {tp}"
    H = cfg.hidden_size
    tile = ttnn.TILE_SIZE
    chunk_global = CHUNK_GLOBAL
    chunk_local = chunk_global // sp
    assert chunk_global % (tile * sp) == 0, f"chunk_global {chunk_global} % (TILE*sp={tile * sp}) != 0"
    for v in iters_isl:
        assert 0 < v <= chunk_global and v % tile == 0, f"iter isl {v}: must be tile-aligned and <= {chunk_global}"

    total_len = sum(iters_isl)

    cache_seq = MAX_SEQ_LEN
    assert cache_seq % chunk_global == 0, f"cache_seq {cache_seq} must be a whole number of aligned 5k chunks"
    assert (
        total_len - iters_isl[-1] + chunk_global <= cache_seq
    ), f"iters {iters_isl} write past the {cache_seq}-token cache"

    logger.info(
        f"weights={'pretrained' if use_pretrained else 'random'}  iters={iters_isl}  total_len={total_len}  "
        f"cache_seq={cache_seq}  chunk_global={chunk_global}  chunk_local={chunk_local}"
    )

    gen = torch.Generator().manual_seed(0)
    ctx = torch.randn(1, total_len, cfg.target_feature_size, generator=gen, dtype=torch.float32)

    # ---- ground truth: one HF drafter forward over the WHOLE concatenated conversation ----
    # The drafter's context K/V is a pure per-token function of the 6 verifier taps, so a multi-turn stream
    # must reproduce, token for token, what a single-shot prefill of the concatenation produces.
    real = hf_context_kv(ctx)

    # ---- device ----
    drafter = TtDFlashDrafter(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=cache_seq,
        chunk_size=chunk_global,
        num_links=num_links,
        topology=topology,
    )
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3  # tap hidden TP-sharded on the hidden dim
    hidden_shard[sp_axis] = 2  # ALSO SP-shard the tap on seq → each chip taps its own [chunk_local] rows
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    k_cache, v_cache = allocate_dflash_kv_cache(mesh_device, cfg, cache_seq, sp_axis=sp_axis, tp_axis=tp_axis)

    kv_actual = 0
    for it, isl in enumerate(iters_isl):
        valid_end = kv_actual + isl
        # The taps must arrive in the WRITER's rotated row order, not as a natural slice: when kv_actual falls
        # inside an aligned 5k chunk the low chips are pushed into the NEXT one (at kv_actual=640, chip 0
        # carries global positions 5120..5759, not 640..1279), so chip c's row r carries positions[c][r].
        # Clamp the gather so pad rows
        # index in-bounds, then zero them — as test_mla.py:747-753 does.
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = [positions[c][r] for c in range(sp) for r in range(chunk_local)]
        assert len(flat) == chunk_global
        gather_idx = torch.tensor([min(gp, total_len - 1) for gp in flat], dtype=torch.long)
        pad_rows = torch.tensor([gp >= valid_end for gp in flat])

        drafter.reset()
        for j, tid in enumerate(cfg.target_layer_ids):
            h_j = ctx[0, :, j * H : (j + 1) * H][gather_idx].clone()
            h_j[pad_rows] = 0.0  # pad taps → zero target_hidden → zero cache row, overwritten by a later turn
            h_tt = ttnn.from_torch(
                h_j.reshape(1, 1, chunk_global, H).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            drafter.tap(h_tt, tid)
        drafter.forward(k_cache, v_cache, kv_actual)
        logger.info(
            f"turn {it}: kv_actual={kv_actual} isl={isl} valid_end={valid_end} pad_rows={int(pad_rows.sum())} "
            f"{'ROTATED' if kv_actual % chunk_global else 'aligned'}"
            f"{' mid-chip' if kv_actual % chunk_local else ''}"
        )
        kv_actual = valid_end
    ttnn.synchronize_device(mesh_device)

    # Every position < total_len is covered, and the LAST write to reach it is a valid (non-pad) one: the
    # turns' valid ranges [kv_actual_i, valid_end_i) tile [0, total_len) consecutively, and a turn's pad tail
    # only ever lands on territory a later turn reclaims. So the whole [0, total_len) window must match.
    dk = _read_cache_natural(k_cache, mesh_device, mesh_shape, sp, chunk_global, cfg.num_hidden_layers, total_len)
    dv = _read_cache_natural(v_cache, mesh_device, mesh_shape, sp, chunk_global, cfg.num_hidden_layers, total_len)

    for i in range(cfg.num_hidden_layers):
        rk, rv = real[i]
        rk = _reshuffle_k_to_interleaved_layout(rk, cfg)  # HF ref is half-split; device persists interleaved K
        ok_k, pcc_k = comp_pcc(rk, dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(rv, dv[i], PCC_THRESHOLD)
        logger.info(f"layer {i}: K pcc={pcc_k} (ok={ok_k})  V pcc={pcc_v} (ok={ok_v})")
        # V never touches rope, so K-fails-while-V-passes localizes the fault to the permuted rope table
        # (wrong row order) rather than to the taps, the writer offset, or the readback un-rotate.
        assert ok_v, f"V layer {i}: device vs HF PCC {pcc_v} < {PCC_THRESHOLD} (tap order / writer offset)"
        assert ok_k, f"K layer {i}: device vs HF PCC {pcc_k} < {PCC_THRESHOLD} (rope row order if V passed)"
