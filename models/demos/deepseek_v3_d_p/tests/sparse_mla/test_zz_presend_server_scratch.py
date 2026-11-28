# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH experiment server: the GLM-5.3 shared layer (REST + pre-send beside sparse SDPA), push vs dispatch arms,
one mesh bring-up. The shared layer reuses a full layer's top-k, dumped by test_sparse_mla_real_data_perf
(SCRATCH_DUMP_TOPK, files topk_L<layer>_c<cache>.pt), so no full layer and no cache fill.

PRESEND_SERVER_DIR=<dir> PRESEND_TOPK_DIR=<dumps>: requests req_<id>.json
  {"cases": ["L58_c61440", ...], "arms": [{"name": "A", "env": {...}}, ...], "sections": bool, "stop": bool}
result res_<id>.txt: per case and arm, untraced / traced ms (min and median), or the per-section medians."""
import gc
import glob
import json
import os
import statistics
import time
import traceback

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.tests.measure.timing import time_untraced
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, time_traced
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import BOX, SparsePerfCase, real_data_params
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tt.mla import mla as mla_mod
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from tt_py_test_utils_common.perf.cpu_clock import hold_full_cpu_clock

_P = real_data_params()[0]


def _reset_planner(mesh_device):
    """A fresh push planner per arm: its sub-device grids and plans depend on the arm's env."""
    ccl = get_tt_ccl(mesh_device)
    if ccl.sparse_kv_push_planner is not None:
        ccl.sparse_kv_push_planner.release()
        ccl.sparse_kv_push_planner = None
    gc.collect()
    mesh_device.disable_and_clear_program_cache()  # (KVD3 knobs are env, not in the program hash)
    mesh_device.enable_program_cache()


@pytest.mark.skipif(
    not (os.environ.get("PRESEND_SERVER_DIR") and os.environ.get("PRESEND_TOPK_DIR")) or bool(_P.marks),
    reason="experiment server: set PRESEND_SERVER_DIR and PRESEND_TOPK_DIR (Galaxy only)",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    [_P.values[:4]],
    indirect=["mesh_device", "device_params", "variant"],
)
def test_zz_presend_server(mesh_device, device_params, variant, case_name, config_only):
    q, topk_dir = os.environ.get("PRESEND_SERVER_DIR"), os.environ.get("PRESEND_TOPK_DIR")
    if not q or not topk_dir:
        pytest.skip("experiment server: set PRESEND_SERVER_DIR and PRESEND_TOPK_DIR")
    clock = hold_full_cpu_clock()
    sp_axis = BOX.sp_axis
    tp_axis = 1 - sp_axis
    chunk = BOX.chunk
    topology = per_axis_topology(device_params["fabric_config"])
    weights = dict(random_mla_weights(config_only))  # (random: the shared layer runs no indexer)
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    ctx = {}  # the current case's device tensors

    def case_ctx(name):
        if ctx.get("name") == name:
            return ctx
        for k in ("x", "indices"):
            if k in ctx:
                ttnn.deallocate(ctx[k])
        ctx.clear()
        gc.collect()
        layer = int(name.split("_")[0][1:])
        dump = torch.load(f"{topk_dir}/topk_{name}.pt")
        cache = dump["chunk_start"]
        total = cache + chunk
        import copy

        config = copy.deepcopy(config_only)
        config.max_seq_len = total
        full = [i for i, m in enumerate(config.indexer_types) if m == "full"]
        pos = full.index(layer)
        first, full_idx = full[pos - 1 : pos + 1]
        cache_layer_idx = full_idx - first
        ctx.update(
            name=name,
            cache=cache,
            total=total,
            config=config,
            first=first,
            layer_idx=full_idx + 1,
            layer_num=cache_layer_idx + 3,
            cache_layer_idx=cache_layer_idx + 1,
        )
        ctx["rope"] = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(
            total, chunk
        )
        ctx["x"] = ttnn.from_torch(
            make_hidden(chunk, config.hidden_size, seed=42)[:, :chunk].unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        idx = dump["indices"]  # [chips, 1, 1, rows, topk], row-major over the mesh
        ctx["indices"] = ttnn.from_torch(
            idx.reshape(-1, *idx.shape[2:]).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        return ctx

    def run_arm(c, arm, sections, profile=0):
        saved = {k: os.environ.get(k) for k in arm.get("env", {})}
        os.environ.update({k: str(v) for k, v in arm.get("env", {}).items()})
        if sections:
            os.environ["SCRATCH_TIME_SECTIONS"] = "1"
        mla = kv = None
        try:
            _reset_planner(mesh_device)
            mla = ttMLA(
                c["config"],
                weights,
                mesh_device,
                topology=topology,
                layer_idx=c["layer_idx"],
                seq_len=c["total"],
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                is_chunked=True,
                active_seq_len=chunk,
                layer_num=c["layer_num"],
                first_layer_idx=c["first"],
                has_indexer=True,
                sparse_kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
            )
            kv = init_mla_kv_cache(
                cache_format=MlaKvCacheFormat.SCALED_FP8,
                hf_config=c["config"],
                mesh_device=mesh_device,
                seq_len=c["total"],
                mesh_shape=list(mesh_device.shape),
                sp_axis=sp_axis,
                num_kvpe_cache_layers=c["layer_num"],
                tp_axis=tp_axis,
            )
            case = SparsePerfCase(
                mla, c["x"], c["rope"], kv, None, c["cache_layer_idx"], tuple(shard_dims), indexer_indices=c["indices"]
            )
            # warm-up: sends ALL and pre-sends the next layer's prefix with the same (cached) plan; this layer's own
            # prefix entry is that plan in this layer's buffer, as the full layer's pre-send leaves it
            ttnn.deallocate(case.forward(c["cache"]))
            planner = mla.tt_ccl.sparse_kv_push_planner
            nxt = planner.prefetched.get(c["layer_idx"] + 1) if planner is not None else None
            if nxt is not None and not os.environ.get("SCRATCH_NO_PRESEND"):
                case.prefix_entry = (nxt[0], mla._push_buffer_parity[c["layer_idx"]])
            mla_mod._SCRATCH_SECTIONS.clear()
            if profile:  # host: cProfile over `profile` untraced forwards (synced once at the end)
                import cProfile
                import io
                import pstats

                for _ in range(3):  # (record, then replay: every program variant compiled before profiling)
                    ttnn.deallocate(case.forward(c["cache"]))
                pr = cProfile.Profile()
                for _ in range(profile):  # (synced around each enqueue: host time only, no queue back-pressure)
                    ttnn.synchronize_device(mesh_device)
                    pr.enable()
                    out_t = case.forward(c["cache"])
                    pr.disable()
                    ttnn.synchronize_device(mesh_device)
                    ttnn.deallocate(out_t)
                buf = io.StringIO()
                st = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
                st.print_stats(
                    r"(_kv_send|sparse_kv_dispatch|sparse_kv_push\b|_presend|_sparse_pushed_attn|_beside_prefetch|"
                    r"_finish_sparse|_sparse_push_plan|dispatch_resources|_kvd_|\bforward\b|sub_device|_sparse_mla\b|"
                    r"_attention|_kv_stem|_q_|_prepare_sparse|sparse_kv|decorators|__call__)"
                )
                return ["profile"] + [
                    f"  {l.replace('/home/pjosipovic/tt-metal-worktrees/kv-dispatch-glm/', '')}"
                    for l in buf.getvalue().splitlines()
                    if l.strip()
                ][-60:]
            untraced, _ = time_untraced(lambda: case.forward(c["cache"]), lambda: ttnn.synchronize_device(mesh_device))
            out = [f"untraced min {min(untraced):.3f} med {statistics.median(untraced):.3f}"]
            if sections:
                for k, v in mla_mod._SCRATCH_SECTIONS.items():
                    out.append(f"  {k}: {sorted(v)[len(v) // 2]:.0f} us")
                mla_mod._SCRATCH_SECTIONS.clear()
            else:
                with TracedStep(
                    mesh_device,
                    lambda: case.forward(c["cache"]),
                    attach=mla.set_trace_controller,
                    detach=lambda: mla.set_trace_controller(None),
                ) as traced:
                    traced.capture()
                    tr = time_traced(traced)
                out.append(f"traced min {min(tr):.3f} med {statistics.median(tr):.3f}")
            return out
        finally:
            os.environ.pop("SCRATCH_TIME_SECTIONS", None)
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            if kv is not None:
                ttnn.deallocate(kv.storage)
            del mla, kv
            _reset_planner(mesh_device)

    open(f"{q}/ready", "w").write("ready\n")
    try:
        while True:
            reqs = sorted(glob.glob(f"{q}/req_*.json"), key=os.path.getmtime)
            if not reqs:
                time.sleep(0.2)
                continue
            path = reqs[0]
            rid = os.path.basename(path)[4:-5]
            try:
                req = json.load(open(path))
            except json.JSONDecodeError:
                time.sleep(0.5)
                continue
            os.rename(path, f"{q}/run_{rid}.json")
            if req.get("stop"):
                open(f"{q}/res_{rid}.txt", "w").write("stopped\n")
                break
            t0, out = time.time(), []
            for name in req.get("cases", []):
                for arm in req.get("arms", [{"name": "base"}]):
                    try:
                        lines = run_arm(case_ctx(name), arm, bool(req.get("sections")), int(req.get("profile", 0)))
                    except Exception:
                        lines = ["ERROR " + traceback.format_exc()]
                    out.append(f"{name} {arm['name']}: " + " | ".join(lines[:2]))
                    out += lines[2:]
                    with open(f"{q}/res_{rid}.partial", "w") as f:
                        f.write("\n".join(out) + "\n")
            out.append(f"({time.time() - t0:.1f} s)")
            with open(f"{q}/res_{rid}.tmp", "w") as f:
                f.write("\n".join(out) + "\n")
            os.rename(f"{q}/res_{rid}.tmp", f"{q}/res_{rid}.txt")
    finally:
        clock.restore()
        _reset_planner(mesh_device)
