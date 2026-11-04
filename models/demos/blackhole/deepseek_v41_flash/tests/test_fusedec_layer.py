# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-layer loop: ONE typical MoE layer (layer 3, CSA2 reuse; layer 2 is its kv source) run EAGER (no trace) under the device profiler, B=8 k=3 (T=8 rows per mesh row), the real routed expert ids injected,
dense position bucket, no prefill. Metric = sum of DEVICE KERNEL DURATION over the ops of the layer (per chip; reported on the slowest chip and on the mean chip), read from the profiler digest
(cpp_device_perf_report.csv = the repo's ops_perf_results); the eager host launch gaps are NOT part of it (the device-timeline gap from the kernel start / end cycles is reported separately).

Env: FD_ARMS  ';'-separated arms 'name:ENV1=v1,ENV2=v2' (default 'base'); every arm is run FD_LREPS times (default 3) eagerly after two warm-up calls; the env of an arm is applied only while it runs (call-time flags).
     FD_LAYERS (default 2,3), FD_B (8), FD_K (3), FD_ROUTES (routes.pt list, as the round harness), FD_OUT.
Output: <FD_OUT>/layer_arms.json (op id ranges per arm and repetition, PCC / bit equality of every arm's layer output against the first arm), oprows.json (recorder rows), prof/.logs/cpp_device_perf_report.csv (digest, written at process end).
Offline: tools/fusedec_layer_budget.py <run dir>.
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests import test_fusedec_round_prof as H
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder, dev_id

LAYER = int(os.environ.get("FD_LAYER", "3"))
ARMS = os.environ.get("FD_ARMS", "base")
LREPS = int(os.environ.get("FD_LREPS", "3"))


def _arms():
    out = []
    for a in ARMS.split(";"):
        a = a.strip()
        if not a:
            continue
        name, _, envs = a.partition(":")
        kv = dict(x.split("=", 1) for x in envs.split(",") if x)
        out.append((name, kv))
    return out


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if float(a.std()) == 0 or float(b.std()) == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _grab(x):
    cols = 8
    return [ttnn.to_torch(ttnn.get_device_tensors(x)[i]).float().clone() for i in (0, 9, 18, 31)]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", H.PARAMS, indirect=True)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_fusedec_layer(mesh_device):
    md = mesh_device
    out_dir = H.OUT
    os.makedirs(out_dir, exist_ok=True)
    rec = OpRecorder()
    rec.install()
    gen, runner, X, base = H._setup(md, rec)
    dec = runner.dec
    if H.ROUTES:
        held = H._inject_routes(
            md, dec, H._load_routes(H.ROUTES), runner.U, runner.n, int(os.environ.get("FD_ROUTE_SEED", "7"))
        )
    Xr = torch.cat([X, X[:, :1].repeat(1, 8)], dim=1)[:, : runner.n]
    runner._ensure(base)
    runner.set_bucket(runner.bucket_for(int(base.max()) + runner.n + 1))
    dec.set_force(-1)
    runner._feed(Xr, base)
    ttnn.synchronize_device(md)
    T = dec.ctrl.shape[0]
    tokens = ttnn.typecast(ttnn.slice(dec.ctrl, [0, 0], [T, 1]), ttnn.uint32)
    dec.pos = ttnn.reshape(ttnn.slice(dec.ctrl, [0, 1], [T, 2]), [T])
    states = {k: ss.build(dec.pos) for k, ss in dec.step_states.items()}
    x, pre = dec.embedding.forward(tokens)
    layers = {lid: (layer, st) for lid, layer, st in dec.layers}
    if os.environ.get("FD_SYNTH", "1") == "1":
        # no real layer 2: synthetic stand-ins of the right shape for what layer 3 reads from its kv source: the pooled latent rows of the step (source.last_lat, bf16 TILE [1,1,T,512]) and the
        # indexer selection of the step (st["topk_ids"], uint32 [rows,1,1,512], distinct valid entries, random); no real state
        lay, st = layers[LAYER]
        g = torch.Generator().manual_seed(5)
        lat = ttnn.from_torch(
            torch.randn(1, 1, T, 512, generator=g).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        src = getattr(lay.attention, "source", None)
        if src is not None:
            src.last_lat = lat
        ratio = getattr(lay.attention, "ratio", 0)
        nent = max(int(base.min()) // max(ratio, 1) - 1, 512)
        if ratio and (runner.cur_bucket or 0) > 512:
            ids = torch.stack([torch.randperm(nent, generator=g)[:512] for _ in range(T * md.shape[0])]).to(torch.int32)
            states[st]["topk_ids"] = ttnn.from_torch(
                ids.sort(dim=1).values.reshape(-1, 1, 1, 512),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
            )
    else:
        for lid in sorted(layers):
            if lid < LAYER:  # real kv source layers, eager, outside the measurement
                lay, st = layers[lid]
                x, pre = lay.forward(x, pre, states[st])
    lay, st = layers[LAYER]
    ttnn.synchronize_device(md)
    _ = [lay.forward(x, pre, states[st]) for _ in range(2)]  # warm-up (program compile, persistent buffers)
    ttnn.synchronize_device(md)
    ref = None
    arms_out = {}
    keep = []
    for name, kv in _arms():
        old = {k: os.environ.get(k) for k in kv}
        os.environ.update(kv)
        try:
            for _ in range(2):  # warm-up of this arm's programs
                keep.append(lay.forward(x, pre, states[st]))
            ttnn.synchronize_device(md)
            ttnn.ReadDeviceProfiler(md)
            ranges = []
            for r in range(LREPS):
                rec.enabled, rec.depth = True, 0
                i0 = dev_id()
                if os.environ.get("FD_L1CLEAN") == "1":
                    from models.demos.blackhole.deepseek_v41_flash.tests import l1_clean as L1C

                    _b = L1C.snapshot(md, f"{name} before")
                o = lay.forward(x, pre, states[st])
                if os.environ.get("FD_L1CLEAN") == "1":
                    ttnn.synchronize_device(md)
                    L1C.assert_clean(md, _b, f"{name} after")
                ttnn.synchronize_device(md)
                i1 = dev_id()
                rec.enabled = False
                ranges.append([i0, i1])
                keep.append(o)
                ttnn.ReadDeviceProfiler(md)
            outs = _grab(o[0])
            rec_ = dict(ranges=ranges, env=kv)
            if os.environ.get("FD_TRACE") == "1":
                # replay profile: the layer captured once and replayed FD_TREPS times; the digest rows carry METAL TRACE ID / REPLAY SESSION ID (offline: dsv4-logs/cores/tracewin.py)
                _l1 = L1C.snapshot(md, "pre-capture " + name) if os.environ.get("FD_L1CLEAN") == "1" else None
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                keep.append(lay.forward(x, pre, states[st]))
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ttnn.ReadDeviceProfiler(md)
                for _ in range(int(os.environ.get("FD_TREPS", "3"))):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                    ttnn.ReadDeviceProfiler(md)
                rec_["trace_id"] = int(tid)
                # otherlayers: wall time per replay of the captured layer (back-to-back non-blocking replays, mean of FD_WALLREPS): the profiler-free metric of the arm
                import time as _time

                _nw = int(os.environ.get("FD_WALLREPS", "30"))
                ttnn.synchronize_device(md)
                _t0 = _time.perf_counter()
                for _ in range(_nw):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                rec_["wall_us"] = (_time.perf_counter() - _t0) / _nw * 1e6
                print(f"LAYERWALL {name} {rec_['wall_us']:.1f} us", flush=True)
                if _l1 is not None:
                    rec_["l1_clean"] = bool(L1C.assert_clean(md, _l1, "post-replay " + name))
            if ref is None:
                ref = outs
            else:
                rec_["pcc"] = min(_pcc(a, b) for a, b in zip(ref, outs))
                rec_["maxabs"] = max(float((a - b).abs().max()) for a, b in zip(ref, outs))
                rec_["bit_equal"] = all(torch.equal(a, b) for a, b in zip(ref, outs))
            arms_out[name] = rec_
            print("LAYERARM " + json.dumps({name: rec_}), flush=True)
            json.dump(
                dict(arms=arms_out, layer=LAYER), open(os.path.join(out_dir, "layer_arms.json"), "w")
            )  # after every arm: a later arm that crashes the process keeps the earlier ones
            json.dump(
                dict(rows=rec.rows, marks=rec.marks, traces={}, meta=dict(layers=H.LAYERS, B=H.B, k=H.K, T=T)),
                open(os.path.join(out_dir, "oprows.json"), "w"),
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    json.dump(dict(arms=arms_out, layer=LAYER), open(os.path.join(out_dir, "layer_arms.json"), "w"))
    json.dump(
        dict(rows=rec.rows, marks=rec.marks, traces={}, meta=dict(layers=H.LAYERS, B=H.B, k=H.K, T=T)),
        open(os.path.join(out_dir, "oprows.json"), "w"),
    )
    if os.environ.get("FD_MOE_DUMP") == "1":
        # the MoE block alone on a deterministic synthetic ffn input (independent of the attention / KV state), real routed ids: outputs of all 32 chips saved for a cross-process comparison (bfp8 vs bfp4 experts)
        gg = torch.Generator().manual_seed(11)
        nrows = md.shape[0]
        hh = (torch.randn(nrows, 1, T, 5120, generator=gg) * 0.5).to(torch.bfloat16)
        h = ttnn.from_torch(
            hh,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
        )
        ht = ttnn.from_torch(
            hh.reshape(nrows * T, 1, 1, 5120),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
        )
        for _ in range(2):
            mo = lay.moe.forward(h, ht)
        ttnn.synchronize_device(md)
        torch.save([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(mo)], os.path.join(out_dir, "moe_out.pt"))
        print("FD_MOE_DUMP saved", flush=True)
    print("FD_LAYER_DONE", flush=True)
