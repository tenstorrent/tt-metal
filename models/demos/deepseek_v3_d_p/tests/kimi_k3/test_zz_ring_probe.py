# Temporary probe (not for commit): measure ring_mla's fused SP all-gather on its own.
# Runs ring_mla with the K3 warm geometry, then with shrunken attention work (fewer heads / query rows)
# so the op becomes ring-bound, plus the standalone ring all-gather on the same cache shard.
import os
from collections import OrderedDict

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_mla_perf import (
    _DEVICE_PARAMS,
    PERF_FABRIC,
    PERF_WORKLOAD,
    SCENARIOS,
    _local_tokens,
    _op_code,
    _shape_normalized_config_and_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

REPEATS = int(os.environ.get("RING_PROBE_REPEATS", 3))
# (heads per chip, query rows per chip)
VARIANTS = [tuple(map(int, v.split("x"))) for v in os.environ.get("RING_PROBE_VARIANTS", "24x640,24x32,1x640,1x32").split(",")]


def _device_programs(mesh_device, fn):
    _, records = profile_realtime_program(mesh_device, fn, collect_all=True, record_timeout_seconds=30.0)
    progs = OrderedDict()
    for r in records:
        rid = r["runtime_id"]
        if not rid:
            continue
        d = float(r["duration_ns"])
        p = progs.setdefault(rid, {"max": 0.0, "min": float("inf"), "src": r["kernel_sources"]})
        p["max"] = max(p["max"], d)
        p["min"] = min(p["min"], d)
    return progs


@pytest.mark.parametrize("mesh_device", [PERF_WORKLOAD.mesh_shape], ids=[PERF_WORKLOAD.id], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], ids=["fabric"], indirect=True)
@pytest.mark.parametrize("variant", ["kimi_k3"], ids=["k3"], indirect=True)
@pytest.mark.timeout(0)
def test_ring_probe(mesh_device, device_params, variant, random_weights):
    torch.manual_seed(0)
    torch.manual_seed(0)
    sp_axis, tp_axis = 0, 1
    sp, tp = tuple(mesh_device.shape)
    cache = _local_tokens(SCENARIOS[os.environ.get("RING_PROBE_SCENARIO", "warm")]["cache"], sp)
    chunk = PERF_WORKLOAD.chunk_tokens
    total = cache + chunk

    fixture_config, weights = random_weights
    config, weights = _shape_normalized_config_and_weights(fixture_config, weights, tp)
    config.max_seq_len = total
    config.max_position_embeddings = total
    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=KimiK3Config.mla_layer_ids()[0],
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        topology=per_axis_topology(PERF_FABRIC),
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        has_indexer=False,
    )
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )
    # Random latent rows so the exp/softmax path sees realistic data.
    storage = kvpe_cache.storage
    logger.info(f"cache storage shape {list(storage.shape)} (per chip), mesh {sp}x{tp}")
    local_rows = storage.shape[2]
    src_torch = torch.randn(storage.shape[0], storage.shape[1], local_rows * sp, storage.shape[3])
    filled = ttnn.from_torch(
        src_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None]),
    )
    ttnn.copy(filled, storage)

    kvpe_dim = storage.shape[3]
    row_bytes = kvpe_dim * 1088 / 1024
    prog_cfg = mla._get_sdpa_program_config(chunk // sp)  # K3 tuned q/k chunk config for 640 local rows
    logger.info(f"sdpa program config: grid {mla.ring_sdpa_compute_grid}, q{prog_cfg.q_chunk_size} k{prog_cfg.k_chunk_size}")

    # Placements: "GXxGY@OX,OY[c|r]" -> SDPA grid, CCL core offset, column- or row-major worker order.
    placements = os.environ.get("RING_PROBE_PLACEMENTS", "10x10@10,0c").split(";")
    results = []
    for placement in placements:
      g, o = placement.split("@")
      grid = tuple(map(int, g.split("x")))
      off = tuple(map(int, o[:-1].split(",")))
      col_major = o[-1] == "c"
      prog_cfg = ttnn.SDPAProgramConfig(
          compute_with_storage_grid_size=grid,
          q_chunk_size=prog_cfg.q_chunk_size,
          k_chunk_size=prog_cfg.k_chunk_size,
          exp_approx_mode=False,
      )
      logger.info(f"PLACEMENT {placement}: grid {grid} offset {off} col_major {col_major}")
      for heads, q_rows in VARIANTS:
          # Keep the gathered length identical across variants: logical_n = kv_actual_isl + q_rows * sp = total.
          kv_actual_isl = total - q_rows * sp
          q = ttnn.from_torch(
              torch.randn(1, heads * tp, q_rows * sp, kvpe_dim),
              device=mesh_device,
              layout=ttnn.TILE_LAYOUT,
              dtype=ttnn.bfloat16,
              memory_config=ttnn.DRAM_MEMORY_CONFIG,
              mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, 1]),
          )

          def run():
              out, _ = ttnn.transformer.ring_mla(
                  q,
                  storage,
                  persistent_output_buffer_kv=mla._chunked_kv_buf,
                  head_dim_v=mla.kv_lora_rank,
                  logical_n=total,
                  program_config=prog_cfg,
                  scale=mla.scale,
                  compute_kernel_config=mla.default_compute_kernel_config,
                  dim=2,
                  multi_device_global_semaphore=mla.tt_ccl.ring_attention_ccl_semaphore_handles,
                  num_links=mla.ccl_num_links,
                  cluster_axis=sp_axis,
                  mesh_device=mesh_device,
                  topology=mla.sp_ccl_topology,
                  ccl_core_grid_offset=ttnn.CoreCoord(*off),
                  use_column_major_ccl=col_major,
                  is_balanced=False,
                  kv_cache_batch_idx=0,
                  kv_actual_isl=kv_actual_isl,
              )
              return out

          out0 = run()  # compile
          ttnn.synchronize_device(mesh_device)
          o = ttnn.to_torch(ttnn.get_device_tensors(out0)[0]).float()
          logger.info(f"CHECKSUM {placement} {heads}x{q_rows}: sum {o.sum().item():.6f} abs {o.abs().sum().item():.6f}")
          times = []
          for _ in range(REPEATS):
              progs = _device_programs(mesh_device, lambda: (run(), ttnn.synchronize_device(mesh_device)))
              for rid, p in progs.items():
                  kinds = sorted({os.path.basename(k) for k in p["src"]})
                  logger.info(f"  program {rid}: max {p['max']/1e3:.1f} us min {p['min']/1e3:.1f} us kernels {kinds}")
              ring = [p for p in progs.values() if _op_code(p["src"]) == "RingJointSDPA"]
              times.append((max(p["max"] for p in ring), min(p["min"] for p in ring)))
          best = min(times)
          n_eff = total - q_rows / 2
          flops = 2 * q_rows * heads * n_eff * (kvpe_dim + mla.kv_lora_rank)
          results.append((placement, heads, q_rows, best[0], best[1], flops))
          logger.info(
              f"ring_mla heads={heads} q_rows={q_rows}: max-chip {best[0]/1e3:.1f} us, min-chip {best[1]/1e3:.1f} us, "
              f"all repeats {[round(t[0]/1e3,1) for t in times]}, attention {flops/1e9:.1f} GFLOP"
          )

    # Expanded-MLA ring attention: each chip's cache rows already expanded to per-head K (nope 128 + rope 64 = 192)
    # and V (128), gathered by ring_joint_sdpa instead of the 576-wide latent. 13x more ring bytes, ~0.3x the FLOPs.
    exp_results = []
    exp_gathers = []
    for heads in [int(h) for h in os.environ.get("RING_PROBE_EXPANDED", "").split(",") if h]:
        dk, dv = 192, 128
        q_rows = chunk // sp
        kv_actual_isl = total - chunk

        def upload(t, dims):
            return ttnn.from_torch(
                t, device=mesh_device, layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16 if t.shape[2] == q_rows * sp else ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=dims))

        q_e = upload(torch.randn(1, heads * tp, q_rows * sp, dk), [2, 1])
        k_torch = torch.randn(1, heads * tp, local_rows * sp, dk)
        v_torch = torch.randn(1, heads * tp, local_rows * sp, dv)
        k_e = upload(k_torch, [2, 1])
        v_e = upload(v_torch, [2, 1])
        pk = upload(torch.zeros(1, heads * tp, local_rows * sp, dk), [None, 1])
        pv = upload(torch.zeros(1, heads * tp, local_rows * sp, dv), [None, 1])
        off = tuple(map(int, placements[0].split("@")[1][:-1].split(",")))
        exp_col_major = placements[0][-1] == "c"

        def run_e():
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_e, k_e, v_e, None, None, None,
                persistent_output_buffer_k=pk, persistent_output_buffer_v=pv,
                joint_strategy="rear", logical_n=total, program_config=prog_cfg,
                scale=dk**-0.5, compute_kernel_config=mla.default_compute_kernel_config, dim=2,
                multi_device_global_semaphore=mla.tt_ccl.ring_attention_ccl_semaphore_handles,
                num_links=mla.ccl_num_links, cluster_axis=sp_axis, mesh_device=mesh_device,
                topology=mla.sp_ccl_topology, ccl_core_grid_offset=ttnn.CoreCoord(*off),
                use_column_major_ccl=exp_col_major, is_causal=True, is_balanced=False,
                kv_cache_batch_idx=0, kv_actual_isl=kv_actual_isl)
            return out

        out0 = run_e()
        ttnn.synchronize_device(mesh_device)
        o = ttnn.to_torch(ttnn.get_device_tensors(out0)[0]).float()
        g = ttnn.to_torch(ttnn.get_device_tensors(pk)[0]).float()
        if os.environ.get("RING_PROBE_FUSED_DIAG"):
            ref = ttnn.to_torch(ttnn.from_torch(k_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
            for di, dt in enumerate(ttnn.get_device_tensors(pk)):
                got = ttnn.to_torch(dt).float()
                c, r = di % tp, di // tp
                refc = ref[:, c * heads:(c + 1) * heads]
                bad = ~torch.isclose(got, refc, atol=1e-3) | ~torch.isfinite(got)
                bad_rows = bad.any(dim=-1)[0]  # [heads, rows]
                per_rank = [int(bad_rows[:, k * local_rows:(k + 1) * local_rows].sum()) for k in range(sp)]
                nonfin = [int((~torch.isfinite(got[..., k * local_rows:(k + 1) * local_rows, :])).any(-1).sum()) for k in range(sp)]
                zero = [int((got[..., k * local_rows:(k + 1) * local_rows, :] == 0).all(-1).sum()) for k in range(sp)]
                if di == 0 and os.environ.get("RING_PROBE_DUMP"):
                    torch.save({"got": got.to(torch.bfloat16), "ref": refc.to(torch.bfloat16)}, os.environ["RING_PROBE_DUMP"])
                logger.info(f"DIAG2 dev{di} rank{r}: nonfinite rows {nonfin}, all-zero rows {zero}")
                tile_rows = bad_rows.reshape(heads, -1, 32).any(-1)  # [heads, row tiles]
                first = tile_rows.nonzero()[:6].tolist()
                logger.info(f"DIAG dev{di} rank{r}: bad rows per source rank {per_rank} of {heads * local_rows}, "
                            f"bad heads {sorted(set(tile_rows.nonzero()[:, 0].tolist()))[:8]}, first (head, rowtile) {first}")
        logger.info(f"CHECKSUM expanded {heads}: out sum {o.sum().item():.6f} abs {o.abs().sum().item():.6f}, "
                    f"gathered K abs {g.abs().sum().item():.3f}")
        times = []
        for _ in range(REPEATS):
            progs = _device_programs(mesh_device, lambda: (run_e(), ttnn.synchronize_device(mesh_device)))
            ring = [p for p in progs.values() if _op_code(p["src"]) == "RingJointSDPA"]
            times.append((max(p["max"] for p in ring), min(p["min"] for p in ring)))
        best = min(times)
        n_eff = total - q_rows / 2
        flops = 2 * q_rows * heads * n_eff * (dk + dv)
        shard = heads * local_rows * (dk + dv) * 1088 / 1024
        exp_results.append((heads, best, [round(t[0] / 1e3, 1) for t in times], flops, shard))

        # Standalone gathers of the same expanded K and V.
        for kind in [k for k in os.environ.get("RING_PROBE_EXP_GATHERS", "").split(",") if k]:
            def op_g():
                if kind == "ring":
                    return ttnn.experimental.ring_attention_all_gather_async(
                        [k_e, v_e], [pk, pv], dim=2,
                        multi_device_global_semaphore=mla.tt_ccl.ring_attention_ccl_semaphore_handles,
                        num_links=2, cluster_axis=sp_axis, mesh_device=mesh_device, topology=mla.sp_ccl_topology)
                ttnn.experimental.fabric_all_gather(k_e, dim=2, output_tensor=pk, cluster_axis=sp_axis, num_links=2)
                return ttnn.experimental.fabric_all_gather(v_e, dim=2, output_tensor=pv, cluster_axis=sp_axis, num_links=2)
            try:
                for buf in (pk, pv):
                    ttnn.fill(buf, 0.0) if hasattr(ttnn, "fill") else None
                op_g()
                ttnn.synchronize_device(mesh_device)
                err = 0.0
                for buf, ref_t in ((pk, k_torch), (pv, v_torch)):
                    ref = ttnn.to_torch(ttnn.from_torch(ref_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
                    for di, dt in enumerate(ttnn.get_device_tensors(buf)):
                        got = ttnn.to_torch(dt).float()
                        c = di % tp
                        r = di // tp
                        refc = ref[:, c * heads:(c + 1) * heads]
                        mask = torch.ones(got.shape[2], dtype=torch.bool)
                        if kind == "ring":
                            mask[r * local_rows:(r + 1) * local_rows] = False
                        err = max(err, (got[:, :, mask] - refc[:, :, mask]).abs().max().item())
                logger.info(f"EXPGATHER {kind} heads={heads}: max|err| {err}")
                ts = []
                for _ in range(max(REPEATS, 3)):
                    progs = _device_programs(mesh_device, lambda: (op_g(), ttnn.synchronize_device(mesh_device)))
                    ts.append(sum(p["max"] for p in progs.values()))
                exp_gathers.append((heads, kind, ts, shard))
            except Exception as e:
                logger.warning(f"EXP GATHER {kind} failed: {str(e)[:300]}")

    # Standalone ring all-gather of the same shard (default worker placement, not the fused op's).
    gather_out = ttnn.from_torch(
        torch.zeros(storage.shape[0], storage.shape[1], local_rows * sp, kvpe_dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
    )

    def gather():
        return ttnn.experimental.ring_attention_all_gather_async(
            [filled],
            [gather_out],
            dim=2,
            multi_device_global_semaphore=mla.tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=mla.ccl_num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=mla.sp_ccl_topology,
        )

    ag_times = []
    try:
        if os.environ.get("RING_PROBE_SKIP_GATHER"):
            raise RuntimeError("skipped")
        gather()
        ttnn.synchronize_device(mesh_device)
        for _ in range(REPEATS):
            progs = _device_programs(mesh_device, lambda: (gather(), ttnn.synchronize_device(mesh_device)))
            ag = list(progs.values())
            ag_times.append((max(p["max"] for p in ag), [_op_code(p["src"]) for p in ag]))
    except Exception as e:  # report and keep the ring_mla numbers
        logger.warning(f"standalone gather failed: {e}")

    gather_results = []
    for kind in [k for k in os.environ.get("RING_PROBE_GATHERS", "").split(",") if k]:
        out = ttnn.from_torch(
            torch.zeros(storage.shape[0], storage.shape[1], local_rows * sp, kvpe_dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
        )
        src = storage if kind.endswith("_nd") else filled
        name = kind.removesuffix("_nd")

        def op():
            if name == "ring":
                return ttnn.experimental.ring_attention_all_gather_async(
                    [src], [out], dim=2,
                    multi_device_global_semaphore=mla.tt_ccl.ring_attention_ccl_semaphore_handles,
                    num_links=2, cluster_axis=sp_axis, mesh_device=mesh_device, topology=mla.sp_ccl_topology)
            if name == "hbw":
                return ttnn.experimental.high_bw_all_gather(src, dim=2, output_tensor=out, cluster_axis=sp_axis, num_links=2)
            if name == "fab":
                return ttnn.experimental.fabric_all_gather(src, dim=2, output_tensor=out, cluster_axis=sp_axis, num_links=2)
            raise ValueError(name)

        try:
            op()
            ttnn.synchronize_device(mesh_device)
            ref = ttnn.to_torch(ttnn.from_torch(src_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
            err = 0.0
            for di, dt in enumerate(ttnn.get_device_tensors(out)):
                got = ttnn.to_torch(dt).float()
                mask = torch.ones(got.shape[2], dtype=torch.bool)
                if name == "ring":  # the ring-attention gather leaves the chip's own shard unwritten by design
                    r = di // tp
                    mask[r * local_rows:(r + 1) * local_rows] = False
                err = max(err, (got[:, :, mask] - ref[:, :, mask]).abs().max().item())
            ts = []
            for _ in range(max(REPEATS, 3)):
                progs = _device_programs(mesh_device, lambda: (op(), ttnn.synchronize_device(mesh_device)))
                ts.append((sum(p["max"] for p in progs.values()), len(progs),
                           sorted({os.path.basename(k) for p in progs.values() for k in p["src"]})))
            gather_results.append((kind, ts, err))
            logger.info(f"GATHER {kind}: {[round(t[0]/1e3,1) for t in ts]} us, programs {ts[0][1]}, kernels {ts[0][2]}, max|err| {err}")
        except Exception as e:
            logger.warning(f"GATHER {kind} failed: {str(e)[:400]}")
            gather_results.append((kind, None, str(e)[:200]))

    shard_bytes = local_rows * row_bytes
    logger.info(f"=== RING PROBE SUMMARY: SP={sp} TP={tp} total={total} rows, shard={local_rows} rows/chip "
                f"= {shard_bytes/1e6:.2f} MB, links={mla.ccl_num_links}, topology={mla.sp_ccl_topology}")
    for pl, heads, q_rows, tmax, tmin, flops in results:
        logger.info(f"RESULT {pl} ring_mla heads={heads:>2} q_rows={q_rows:>3}: {tmax/1e3:8.1f} us (min chip {tmin/1e3:8.1f}), "
                    f"{flops/1e9:7.1f} GFLOP")
    for kind, ts, err in gather_results:
        if ts:
            best = min(t[0] for t in ts)
            busiest = (sp - 1) * shard_bytes / 2  # sp-1 shards cross the busiest link direction of a line, split over 2 links
            logger.info(f"RESULT gather {kind:>8}: best {best/1e3:7.1f} us, all {[round(t[0]/1e3,1) for t in ts]}, "
                        f"{busiest/best:.2f} GB/s per link direction, max|err| {err}")
        else:
            logger.info(f"RESULT gather {kind:>8}: FAILED {err}")
    for heads, best, alls, flops, shard in exp_results:
        logger.info(f"RESULT expanded ring_joint_sdpa heads={heads}: {best[0]/1e3:8.1f} us (min chip {best[1]/1e3:8.1f}), "
                    f"all {alls}, {flops/1e9:6.1f} GFLOP, K+V shard {shard/1e6:.1f} MB/chip, "
                    f"ring floor @48.5GB/s/link {(sp - 1) * shard / 2 / 48.5e3:.0f} us")
    for heads, kind, ts, shard in exp_gathers:
        best = min(ts)
        logger.info(f"RESULT expanded gather {kind:>4} heads={heads}: best {best/1e3:8.1f} us, all {[round(t/1e3,1) for t in ts]}, "
                    f"{(sp - 1) * shard / 2 / best / 1e3:.2f} GB/s per link direction")
    for t, codes in ag_times:
        logger.info(f"RESULT standalone ring all-gather: {t/1e3:.1f} us ({codes})")
