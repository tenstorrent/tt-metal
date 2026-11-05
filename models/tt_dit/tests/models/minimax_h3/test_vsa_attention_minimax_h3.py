# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VSA attention on device vs the torch VSA oracle, in a 4x8 ladder rung.

`test_vsa_fine_attention`: `ttnn.transformer.sparse_ring_sdpa` computes exactly the blocks the VSA pre-pass selects.
Synthetic Q/K/V (pad rows repeat a real row of their tile, as in the model), the device VSAPrepass mask, and the op
without pooled K/V (exact step only) at `logical_n` = rung, against the torch block-sparse reference fed the same mask.
The ragged geometry runs in a rung whose shards end in a 32-row straggler, and also without `block_counts`, which
must miss the bar: that proves the counts reach the kernel. The `logical_n_ab` case runs the op at `logical_n` = rung
and at the geometry's real length and compares output and traced time.

`test_vsa_attention`: MiniMaxH3Attention on the VSA path: pre-pass, coarse branch, gate and fine attention. The torch
reference is the diffusers attention with dense SDPA swapped for the oracle in device row order. The check case
reports three numbers:
- `pcc_given_mask` (gates): module output vs the reference driven by the device's own block selection, captured from
  the module's call into the fine op. It measures the module's numerics with routing held equal.
- `selection_agreement` (gates): fraction of (head, scored query block, candidate key block) entries where the device
  keeps the same block as the oracle's fp32 top-k. Exempt and pad blocks are forced on both sides, so excluded.
  Near-ties at the k-th score flip between bf16 device scores and fp32 oracle scores; this bounds how many.
- `pcc_independent` (logged only): module output vs the reference with the oracle's own selection, i.e. both effects.
"""

from __future__ import annotations

import math
import statistics
import time

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed, _apply_rotary_emb
from loguru import logger
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....models.transformers.minimax_h3.vsa_prepass import MiniMaxH3VSAConfig, VSAPrepass, compute_topk
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_BUCKET_LADDER_4X8
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_vsa_geometry, vsa_min_padded_len
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from .common import (
    _BH_ONLY,
    GALAXY_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    _ring_8k,
    format_host_times,
    grid_position_ids,
    randomize_norm_weights,
    upload_rope,
)
from .test_transformer_minimax_h3 import logical_length_tensor
from .vsa_oracle import VSA_INDEX_SENTINEL, coarse_output, coarse_scores, fine_attention_batched, select_index_rows

MIN_PCC_GIVEN_MASK = 0.999
MIN_SELECTION_AGREEMENT = 0.998
SPARSITY = 0.9
NUM_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]

FINE_MIN_PCC = 0.999
FINE_NUM_HEADS = 4
FINE_HEAD_DIM = 128
K_CHUNK = 256
AB_REPS = 6
AB_TRACE_REGION_BYTES = 50_000_000
# A request the in-place check swaps in: 128 candidates, so a different top-k k than either case.
SWAPPED_IN = ((300, 0, 200), (8, 32, 32))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def mask_to_kept(mask: torch.Tensor, geometry, nb_pad: int) -> torch.Tensor:
    """[H, S * nb_pad, S * nb_pad] 0/1 (local Q block rows per shard, padded global columns) -> [H, n_blocks,
    n_blocks] bool in global block ids."""
    slot = torch.arange(geometry.sp_factor * nb_pad)
    real_slot = (slot % nb_pad) < geometry.blocks_per_shard
    assert bool((mask[:, :, ~real_slot] == 0).all()), "the mask selects a padding column"
    kept = mask[:, real_slot][:, :, real_slot] != 0
    pad = geometry.valid_counts == 0
    assert not bool(kept[:, :, pad].any()), "the mask selects a pad block"
    assert not bool(kept[:, pad].any()), "a pad block's query row keeps a block"
    return kept


def kept_to_indices(kept: torch.Tensor) -> torch.Tensor:
    """[H, n_blocks, n_blocks] bool -> oracle index rows [1, H, n_blocks, n_blocks], sentinel-tailed."""
    heads, n_tiles, _ = kept.shape
    indices = torch.full((1, heads, n_tiles, n_tiles), VSA_INDEX_SENTINEL, dtype=torch.int64)
    for h in range(heads):
        for r in range(n_tiles):
            cols = torch.nonzero(kept[h, r]).reshape(-1)
            indices[0, h, r, : cols.numel()] = cols
    return indices


def _torch_reference(torch_model, x, rope_cos, rope_sin, geometry, gate_weight):
    """The oracle's own selection, [H, n_blocks, n_blocks] bool, and `output(kept)`: the attention output for a
    given selection (fine attention over `kept`, coarse branch, gate, `to_out`), in packed order."""
    with torch.no_grad():
        heads = lambda t: t.unflatten(-1, (NUM_HEADS, HEAD_DIM))
        q = _apply_rotary_emb(torch_model.norm_q(heads(torch_model.to_q(x))), rope_cos, rope_sin)
        k = _apply_rotary_emb(torch_model.norm_k(heads(torch_model.to_k(x))), rope_cos, rope_sin)
        v = heads(torch_model.to_v(x))
        gate = heads(x @ gate_weight.T)
        # Packed [1, S, H, D] -> [1, H, rung, D] in device row order, zeros at pad rows.
        to_device_order = lambda t: geometry.pack_rows(t[0]).permute(1, 0, 2).unsqueeze(0)
        qt, kt, vt, gt = (to_device_order(t) for t in (q, k, v, gate))

        scores = coarse_scores(qt, kt, geometry)
        indices, _ = select_index_rows(scores, geometry, SPARSITY, with_sets=False)
        listed = indices[0].long()
        n_blocks = geometry.n_blocks
        kept = torch.zeros(NUM_HEADS, n_blocks, n_blocks + 1, dtype=torch.bool)
        kept.scatter_(-1, torch.where(listed == VSA_INDEX_SENTINEL, n_blocks, listed), True)

    def output(selection: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fine = fine_attention_batched(qt, kt, vt, kept_to_indices(selection), geometry)
            coarse = coarse_output(scores, vt, geometry, fine.dtype)
            out = geometry.unpack_rows((fine + gt * coarse)[0].permute(1, 0, 2))
            return torch_model.to_out[0](out.reshape(1, -1, NUM_HEADS * HEAD_DIM))[0]

    return kept[..., :n_blocks], output


# The A/B case times captured traces, which need a trace region; GALAXY_RING's 4x8 row has none.
FINE_RING = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            1,
            0,
            2,
            {**_ring_8k, "trace_region_size": AB_TRACE_REGION_BYTES},
            ttnn.Topology.Ring,
            False,
            id="4x8sp1tp0nl2_ring_is_fsdp0",
            marks=_BH_ONLY,
        ),
    ],
    indirect=["mesh_device", "device_params"],
)


def _traced_replay_ms(mesh_device, fn) -> list[float]:
    """One trace capture of `fn` (already warm), AB_REPS timed replays: the replay times in ms."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(AB_REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(out)
    return times


@FINE_RING
@pytest.mark.parametrize(
    ("prefix_segments", "grid", "rung"),
    [
        pytest.param((512, 0, 0), (8, 16, 16), 18944, id="quick-aligned"),
        pytest.param((70, 5, 130), (9, 10, 13), 20224, id="quick-ragged-straggler"),
        pytest.param((39, 0, 414), (37, 24, 42), 45312, id="5s-1344x768"),
    ],
)
@pytest.mark.parametrize("case", ["check", "logical_n_ab"])
def test_vsa_fine_attention(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    prefix_segments,
    grid,
    rung,
    case,
    reset_seeds,
) -> None:
    ab = case == "logical_n_ab"
    if ab and rung == 18944:
        pytest.skip("the A/B shapes are the ragged straggler and 5 s")
    if not ab and rung == 45312:
        pytest.skip("5 s runs the A/B case only")
    sp = tuple(mesh_device.shape)[sp_axis]
    heads = NUM_HEADS // tuple(mesh_device.shape)[tp_axis] if ab else FINE_NUM_HEADS
    assert rung in MINIMAX_H3_BUCKET_LADDER_4X8 and rung >= vsa_min_padded_len(prefix_segments, grid, sp)
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung)
    ragged = bool((geometry.valid_counts[geometry.valid_counts > 0] < VSA_TILE_TOKENS).any())
    logger.info(
        f"{geometry.n_real_tiles} tiles in {geometry.n_blocks} blocks at rung {rung} "
        f"({geometry.rows_per_shard} rows/device), ragged={ragged}, {heads} heads"
    )

    torch.manual_seed(0)
    packed = [torch.randn(geometry.seq_len, heads, FINE_HEAD_DIM).bfloat16().float() for _ in range(3)]
    # Packed [S, H, D] -> [1, H, rung, D] in device row order.
    q, k, v = (geometry.permute_metadata(x).permute(1, 0, 2).unsqueeze(0) for x in packed)

    rows = geometry.rows_per_shard
    to_shards = lambda x: x[0].reshape(heads, sp, rows, FINE_HEAD_DIM).permute(1, 0, 2, 3).contiguous()
    upload = lambda x: from_torch(
        to_shards(x), device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None]
    )
    tq, tk, tv = upload(q), upload(k), upload(v)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    def new_prepass() -> VSAPrepass:
        return VSAPrepass(
            rung=rung,
            sp_factor=sp,
            sparsity=SPARSITY,
            num_heads=heads,
            head_dim=FINE_HEAD_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
        )

    prepass = new_prepass()
    prepass.update(geometry, traced=False)
    routing = prepass(tq, tk, tv, coarse=False)

    grid_size = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_size.x - 1, grid_size.y),
        q_chunk_size=VSA_TILE_TOKENS,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))

    def op(logical_n: int, with_counts: bool = True) -> ttnn.Tensor:
        return ttnn.transformer.sparse_ring_sdpa(
            tq,
            tk,
            tv,
            routing["block_mask"],
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(tk.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(tv.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            logical_n=logical_n,
            program_config=program_config,
            compute_kernel_config=compute_config,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(grid_size.x - 1, 0),
            use_column_major_ccl=True,
            block_counts=routing["block_counts"] if with_counts else None,
        )

    def run(logical_n: int = rung, with_counts: bool = True) -> torch.Tensor:
        out = op(logical_n, with_counts)
        ttnn.synchronize_device(mesh_device)
        return ttnn.to_torch(out, mesh_composer=composer)[:, :heads].float()

    real_rows = geometry.gather_index >= 0
    if ab:
        _logical_n_ab(mesh_device, op, run, geometry, real_rows & (torch.arange(rung) >= prepass.prefix_rows))
        return

    # The TP replicas of the mask are identical; keep the first.
    mask = ttnn.to_torch(routing["block_mask"], mesh_composer=composer)[0, :heads]
    indices = kept_to_indices(mask_to_kept(mask, geometry, prepass.nb_pad))
    reference = fine_attention_batched(q, k, v, indices, geometry)

    out = run()
    assert torch.equal(out, run()), "the cached program (rebound buffers) must reproduce the first call"
    pcc = _pcc(out[:, :, real_rows], reference[:, :, real_rows])
    density = (indices[0] != VSA_INDEX_SENTINEL).sum().item() / (heads * geometry.n_blocks**2)
    k = compute_topk(SPARSITY, int(geometry.is_candidate.sum()))
    logger.info(f"kept density {density:.3f}, k={k} of top-k width {prepass.k_max}: PCC with block counts {pcc:.6f}")
    assert pcc >= FINE_MIN_PCC, pcc

    if ragged:
        pcc_without = _pcc(run(with_counts=False)[:, :, real_rows], reference[:, :, real_rows])
        logger.info(f"PCC without block counts {pcc_without:.6f}")
        assert pcc_without < FINE_MIN_PCC, "pad rows of ragged tiles are attended without counts, so this must miss"

    # The pipeline rewrites a rung's pre-pass in place for each request; it must route exactly as a fresh one.
    def routed_mask(p: VSAPrepass, qkv: tuple) -> torch.Tensor:
        r = p(*qkv, coarse=False)
        out = ttnn.to_torch(r["block_mask"], mesh_composer=composer)
        ttnn.deallocate(r["block_mask"])
        return out

    first = routed_mask(prepass, (tq, tk, tv))
    other = build_vsa_geometry(*SWAPPED_IN, sp_factor=sp, padded_len=rung)
    other_k = compute_topk(SPARSITY, int(other.is_candidate.sum()))
    assert other_k != k, "the swapped-in request must change k"
    # Its own Q/K/V in its own layout, as a real request has: distinct pooled keys, so no ties at the threshold.
    other_qkv = tuple(
        upload(other.permute_metadata(torch.randn(other.seq_len, heads, FINE_HEAD_DIM)).permute(1, 0, 2).unsqueeze(0))
        for _ in range(3)
    )
    fresh = new_prepass()
    fresh.update(other, traced=False)
    prepass.update(other, traced=True)
    other_mask = routed_mask(prepass, other_qkv)
    assert torch.equal(other_mask, routed_mask(fresh, other_qkv)), "an in-place update must route as a fresh pre-pass"
    assert not torch.equal(other_mask, first), "the other geometry must change the routing"
    other_kept = mask_to_kept(other_mask[0, :heads], other, prepass.nb_pad)
    scored_rows = (other.valid_counts > 0) & ~other.is_exempt
    kept_candidates = other_kept[:, scored_rows][:, :, other.is_candidate].sum(dim=-1)
    assert bool((kept_candidates == other_k).all()), "each scored row keeps exactly the request's k candidates"
    prepass.update(geometry, traced=True)
    assert torch.equal(routed_mask(prepass, (tq, tk, tv)), first), "updating back in place must restore the routing"


def _logical_n_ab(mesh_device, op, run, geometry, compared_rows: torch.Tensor) -> None:
    """Call A at `logical_n` = rung, call B at the geometry's real length; same mask, counts and inputs."""
    calls = {"A": geometry.padded_len, "B": geometry.logical_n}
    outputs = {}
    for name, logical_n in calls.items():
        run(logical_n)
        before = mesh_device.num_program_cache_entries()
        outputs[name] = run(logical_n)
        misses = mesh_device.num_program_cache_entries() - before
        assert misses == 0, f"call {name} (logical_n {logical_n}) compiled {misses} programs on its warm call"

    a, b = outputs["A"][:, :, compared_rows], outputs["B"][:, :, compared_rows]
    pcc = _pcc(a, b)
    # bf16 keeps 8 significant bits: allow two units in the last place of each row's largest magnitude.
    row_scale = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    worst = ((a - b).abs() / row_scale).max().item()
    logger.info(
        f"A/B logical_n {calls['A']} vs {calls['B']}: PCC {pcc:.6f}, max |A-B| / row max {worst:.3e} "
        f"(bit-exact {torch.equal(a, b)})"
    )

    medians = {}
    for name, logical_n in calls.items():
        times = _traced_replay_ms(mesh_device, lambda: op(logical_n))
        medians[name] = statistics.median(times)
        logger.info(f"A/B call {name}: median {medians[name]:.3f} ms (all {' '.join(f'{t:.3f}' for t in times)})")
    logger.info(f"A/B time ratio A/B {medians['A'] / medians['B']:.4f}")

    assert pcc >= 0.9999, pcc
    assert worst <= 2.0**-7, worst


@GALAXY_RING
@pytest.mark.parametrize(
    ("prefix_segments", "grid", "rung"),
    [
        pytest.param((512, 0, 0), (8, 16, 16), 18944, id="quick-aligned"),
        pytest.param((70, 5, 130), (9, 10, 13), 20224, id="quick-ragged-straggler"),
        pytest.param((39, 0, 414), (37, 24, 42), 45312, id="5s-1344x768"),
        pytest.param((39, 0, 1206), (107, 24, 42), 120320, id="15s-1344x768"),
    ],
)
@pytest.mark.parametrize("perf_only", [False, True], ids=["check", "perf_only"])
@pytest.mark.parametrize("attention", ["dense", "vsa"])
def test_vsa_attention(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    prefix_segments,
    grid,
    rung,
    attention,
    perf_only,
    reset_seeds,
    monkeypatch,
) -> None:
    """`dense` runs the same request at the same rung without VSA, for comparison; it has no oracle, so perf only."""
    if attention == "dense" and not perf_only:
        pytest.skip("the oracle is VSA; dense attention correctness is test_minimax_h3_attention")
    vsa = attention == "vsa"
    sp = tuple(mesh_device.shape)[sp_axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    assert rung in MINIMAX_H3_BUCKET_LADDER_4X8 and rung >= vsa_min_padded_len(prefix_segments, grid, sp)
    seq_len = sum(prefix_segments) + math.prod(grid)

    torch.manual_seed(0)
    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_HEADS, dim_head=HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()
    gate_weight = 0.02 * torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
    state = dict(torch_model.state_dict())
    if vsa:
        state["to_gate_compress.weight"] = gate_weight

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(grid_position_ids(seq_len))
    x = torch.randn(1, seq_len, HIDDEN_SIZE)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp),
        cfg_parallel=None,
    )
    tt_model = MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=rope_cos.shape[-1],
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        vsa_config=MiniMaxH3VSAConfig(sparsity=SPARSITY) if vsa else None,
    )
    tt_model.load_torch_state_dict(state)

    host_ms = {}
    update_ms = {}
    if vsa:
        geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung)
        prepass = VSAPrepass(
            rung=rung,
            sp_factor=sp,
            sparsity=SPARSITY,
            num_heads=NUM_HEADS // tp,
            head_dim=HEAD_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
        )
        prepass.update(geometry, traced=False)
        tt_model.set_vsa_prepass(prepass)

        if not perf_only:
            oracle_kept, reference = _torch_reference(torch_model, x, rope_cos, rope_sin, geometry, gate_weight)
            mask_dims = [None, None]
            mask_dims[sp_axis] = 2
            mask_dims[tp_axis] = 1
            mask_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=mask_dims, mesh_shape=tuple(mesh_device.shape))
            device_masks = []
            sparse_attention = tt_model._sparse_attention

            # The forward deallocates the mask after the fine op, so copy it to host on the way in.
            def capture_mask(q, k, v, routing):
                device_masks.append(ttnn.to_torch(routing["block_mask"], mesh_composer=mask_composer))
                return sparse_attention(q, k, v, routing)

            monkeypatch.setattr(tt_model, "_sparse_attention", capture_mask)

        # Timed again once warm.
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung)
        host_ms["build_vsa_geometry"] = (time.perf_counter() - t0) * 1e3
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        prepass.update(geometry, traced=False, timings=update_ms)
        host_ms["prepass.update"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        tt_model.set_vsa_prepass(prepass)
        host_ms["set_vsa_prepass"] = (time.perf_counter() - t0) * 1e3

        # Device row order: pad rows repeat a real row of their tile, for the input and the rotary tables alike.
        to_rows = geometry.permute_metadata
        logical_n = geometry.logical_n
    else:
        # Packed order, padded to the rung with copies of row 0; the op masks rows past `logical_n`.
        def to_rows(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t[:1].expand(rung - seq_len, *t.shape[1:])])

        logical_n = seq_len

    cos_t, sin_t = prepare_rope_tables(to_rows(rope_cos), to_rows(rope_sin), HEAD_DIM)
    tt_cos, tt_sin = upload_rope(cos_t, sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_x = bf16_tensor_2dshard(
        to_rows(x[0]).reshape(1, 1, rung, HIDDEN_SIZE),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
    )
    tt_logical_n = logical_length_tensor(mesh_device, logical_n)

    def run_attention(timings: dict[str, float] | None = None) -> ttnn.Tensor:
        out = tt_model(tt_x, logical_n=tt_logical_n, rope_cos=tt_cos, rope_sin=tt_sin, timings=timings)
        ttnn.synchronize_device(mesh_device)
        return out

    ttnn.deallocate(run_attention())
    ttnn.synchronize_device(mesh_device)
    # Warm run only, so the profile excludes setup and kernel compilation.
    attention_ms = {}
    signpost("start")
    t0 = time.perf_counter()
    tt_out = run_attention(attention_ms)
    host_ms["warm run"] = (time.perf_counter() - t0) * 1e3
    signpost("stop")
    logger.info(
        f"{attention} host times:\n"
        + format_host_times(host_ms, {"prepass.update": update_ms, "warm run": attention_ms})
    )
    if perf_only:
        return

    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 3
    out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )
    out = geometry.unpack_rows(out.reshape(geometry.padded_len, HIDDEN_SIZE).float())

    device_kept = mask_to_kept(device_masks[-1][0], geometry, prepass.nb_pad)
    scored_rows = (geometry.valid_counts > 0) & ~geometry.is_exempt
    # Unscored rows (exempt and pad query blocks) keep the oracle's dense rows; the module computes the exempt rows
    # densely outside the fine op.
    given_kept = oracle_kept.clone()
    given_kept[:, scored_rows] = device_kept[:, scored_rows]
    pcc_given_mask = _pcc(out, reference(given_kept))
    pcc_independent = _pcc(out, reference(oracle_kept))

    device_scored = device_kept[:, scored_rows][:, :, geometry.is_candidate]
    oracle_scored = oracle_kept[:, scored_rows][:, :, geometry.is_candidate]
    flips = device_scored != oracle_scored
    selection_agreement = 1.0 - flips.float().mean().item()
    flipped_rows = int(flips.any(dim=-1).sum())
    n_rows = flips.shape[0] * flips.shape[1]
    logger.info(
        f"{geometry.n_real_tiles} tiles at rung {rung}: pcc_given_mask {pcc_given_mask:.6f}, "
        f"selection_agreement {selection_agreement:.6f} ({int(flips.sum())} of {flips.numel()} entries differ, "
        f"{flipped_rows} of {n_rows} scored rows have a flip), pcc_independent {pcc_independent:.6f}"
    )
    assert pcc_given_mask >= MIN_PCC_GIVEN_MASK, pcc_given_mask
    assert selection_agreement >= MIN_SELECTION_AGREEMENT, selection_agreement
