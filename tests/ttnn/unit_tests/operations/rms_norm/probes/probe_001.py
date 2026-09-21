# analyze-op dynamic pass for rms_norm: build every case's ProgramDescriptor and dump JSON.
import json, os, sys, traceback
import torch, ttnn

sys.path.insert(0, os.path.join(os.getcwd(), ".claude"))
from eval.sharding import auto_shard_config, shard_config          # noqa: E402
from ttnn.operations.rms_norm import rms_norm_program_descriptor as PD  # noqa: E402
from ttnn.operations.rms_norm.rms_norm import validate, default_compute_kernel_config  # noqa: E402

OUT = os.environ.get("RMS_DUMP", "/tmp/rms_as_shipped_dump.json")

_captured = {}
_orig_plan = PD._plan_placement
def _spy_plan(*a, **kw):
    p = _orig_plan(*a, **kw)
    _captured["plan"] = p
    _captured["n_calls"] = _captured.get("n_calls", 0) + 1
    return p
PD._plan_placement = _spy_plan

TL, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
ML = ttnn.TensorMemoryLayout
BF16, FP32, BF8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
_TD = {BF16: torch.bfloat16, FP32: torch.float32, BF8: torch.bfloat16}

# id, shape, dtype, layout, gamma(dtype|None), gamma_layout, memory_layout, fp32acc, shard(pin), why
CASES = [
 ("c01",(1,1,64,128),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"baseline ROWS/RESIDENT; gamma-present half of ablation A1"),
 ("c02",(1,1,64,128),BF16,TL,None,None,ML.INTERLEAVED,True,None,"ablation A1 absent: same cell, gamma omitted"),
 ("c03",(1,1,64,256),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"Wt=8 == DEST_ACC_SQUARE_MAX_WT: fold ON (straddle below)"),
 ("c04",(1,1,64,288),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"Wt=9 > DEST_ACC_SQUARE_MAX_WT: fold OFF (straddle above)"),
 ("c05",(1,1,32,72),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"Wt=3 < REDUCE_ACC_VIA_ADD_MIN_WT: ReduceTile (straddle below); w_non_aligned"),
 ("c06",(1,1,32,104),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"Wt=4 == REDUCE_ACC_VIA_ADD_MIN_WT, partial_w keeps fold off: AccViaAdd (straddle above)"),
 ("c07",(1,1,17,64),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"h_non_aligned, rank 4"),
 ("c08",(1024,1024),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"rank 2; WIDTH_SPLIT_MIN_GAIN declines the split"),
 ("c09",(1,32,4096),BF16,TL,BF16,TL,ML.INTERLEAVED,True,None,"rank 3"),
 ("c10",(1,1,8192,1024),BF16,TL,BF16,TL,ML.INTERLEAVED,False,None,"prefill RESIDENT interleaved; fp32_dest_acc_en=False"),
 ("c11",(1,1,8192,7168),BF16,TL,BF16,TL,ML.INTERLEAVED,False,None,"prefill wide: expect ROW_RESIDENT (D14)"),
 ("c12",(1,1,4096,11008),BF16,TL,BF16,TL,ML.INTERLEAVED,False,None,"one tile-row of x+gamma over budget: expect STREAM"),
 ("c13",(1,1,32,7168),BF16,TL,BF16,TL,ML.INTERLEAVED,False,None,"decode wide: interleaved cross-core width split (D11 AUTO)"),
 ("c14",(1,1,64,128),FP32,TL,FP32,TL,ML.INTERLEAVED,True,None,"dtype float32 (+fp32_dest_acc_en=True, the only legal cell)"),
 ("c15",(1,1,64,128),BF8,TL,BF8,TL,ML.INTERLEAVED,True,None,"dtype bfloat8_b; gamma_trim demotes to half page (D23)"),
 ("c16",(1,1,64,128),BF16,RM,BF16,RM,ML.INTERLEAVED,True,None,"ROW_MAJOR x + ROW_MAJOR gamma: stick staging, depth forced to 1"),
 ("c17",(1,1,32,50),BF16,RM,BF16,RM,ML.INTERLEAVED,True,None,"ROW_MAJOR + w_non_aligned: stage_zero"),
 ("c18",(1,1,256,512),BF16,TL,BF16,TL,ML.HEIGHT_SHARDED,True,None,"SCHEME_SHARD_H: zero-copy CBs, local reduce"),
 ("c19",(1,1,32,1024),BF16,TL,BF16,TL,ML.WIDTH_SHARDED,False,([32,128],(8,1)),"combine, group 8, block_rows==1 identity path; gamma-present half of ablation A2"),
 ("c20",(1,1,8192,1024),BF16,TL,BF16,TL,ML.BLOCK_SHARDED,False,([1024,128],(8,8)),"combine, group 8, block_rows>1 COMPACT path (D27)"),
 ("c21",(1,1,32,5120),BF16,TL,BF16,TL,ML.WIDTH_SHARDED,False,([32,160],(8,4)),"group 32 -> deleted 20 >= 18: SLOT TREE (straddle above)"),
 ("c22",(1,1,32,7168),BF16,TL,BF16,TL,ML.WIDTH_SHARDED,False,([32,256],(7,4)),"group 28 -> deleted 17 < 18: FLAT root (straddle below)"),
 ("c23",(1,1,224,3072),BF16,RM,BF16,RM,ML.WIDTH_SHARDED,False,None,"ROW_MAJOR width shard: the BAND scheme (D10)"),
 ("c24",(1,1,32,4064),BF16,RM,BF16,RM,ML.INTERLEAVED,False,None,"Wt=127 prime -> wt_chunk==1 < REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT: ReduceTile"),
 ("c25",(1,1,32,1024),BF16,TL,None,None,ML.WIDTH_SHARDED,False,([32,128],(8,1)),"ablation A2 absent: combine path without gamma"),
]

def crs_cores(crs):
    try: return int(crs.num_cores())
    except Exception:
        try: return len(list(ttnn.corerange_to_cores(crs, None, True)))
        except Exception: return None

def dump_cb(cb):
    fds = []
    for fd in cb.format_descriptors:
        try: df = int(fd.data_format_as_uint8)
        except Exception: df = None
        fds.append({"buffer_index": int(fd.buffer_index), "data_format_u8": df,
                    "page_size": int(fd.page_size)})
    d = {"total_size": int(cb.total_size), "cores": crs_cores(cb.core_ranges),
         "formats": fds}
    for m in ("has_buffer", "has_global_circular_buffer"):
        try: d[m] = bool(getattr(cb, m)())
        except Exception: d[m] = None
    if fds and fds[0]["page_size"]:
        d["pages"] = int(cb.total_size) // fds[0]["page_size"]
    return d

def run_case(device, cid, shape, dt, lay, gdt, glay, mlay, acc, pin, why):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_TD[dt])
    if mlay == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif pin is not None:
        mc = shard_config(pin[0], pin[1], mlay, layout=lay, dtype=dt,
                          orientation=ttnn.ShardOrientation.ROW_MAJOR, device=device)
    else:
        mc = auto_shard_config(list(shape), mlay, layout=lay, dtype=dt, device=device)
    xt = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=mc)
    gt = None
    if gdt is not None:
        W = shape[-1]
        g = torch.randn(W, dtype=torch.float32).to(_TD[gdt]).reshape(1, 1, 1, W)
        gt = ttnn.from_torch(g, dtype=gdt, layout=glay, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ck = ttnn.ComputeConfigDescriptor()
    ck.math_fidelity = ttnn.MathFidelity.HiFi4
    ck.fp32_dest_acc_en = acc
    ck.math_approx_mode = False
    axes = validate(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=ck)
    omc = xt.memory_config()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(xt.shape)), xt.dtype, xt.layout, device, omc)
    _captured.clear()
    pd = PD.create_program_descriptor(xt, out, gamma=gt, epsilon=1e-6, compute_kernel_config=ck)
    plan = _captured["plan"]

    kernels = {}
    for k in pd.kernels:
        name = os.path.basename(str(k.kernel_source)).replace(".cpp", "")
        ra = {}
        try:
            for w in plan.assignment:
                ra[f"{w.core.x},{w.core.y}"] = [int(v) for v in k.runtime_args[w.core.x][w.core.y]]
        except Exception as e:
            ra = {"__error__": repr(e)}
        first = next(iter(ra.values())) if ra and "__error__" not in ra else None
        kernels[name] = {"ct": [int(v) for v in k.compile_time_args],
                         "n_ct": len(k.compile_time_args),
                         "cores": crs_cores(k.core_ranges),
                         "rt_len": (len(first) if first else 0),
                         "rt_first": first,
                         "rt_distinct": len({tuple(v) for v in ra.values()}) if "__error__" not in ra else None}
    sems = [{"initial_value": int(s.initial_value), "cores": crs_cores(s.core_ranges)} for s in pd.semaphores]
    cbs = {}
    for cb in pd.cbs:
        d = dump_cb(cb)
        cbs[str(d["formats"][0]["buffer_index"])] = d

    asg = [{"x": w.core.x, "y": w.core.y, "row_start": w.row_start, "row_count": w.row_count,
            "w_start": w.w_start, "w_real": w.w_real, "is_root": bool(w.is_root), "slot": w.slot,
            "stick_base": w.stick_base, "stick_count": w.stick_count,
            "w_off_elems": w.w_off_elems, "w_real_elems": w.w_real_elems} for w in plan.assignment]
    return {
        "id": cid, "shape": list(shape), "why": why,
        "axes": {k: str(v) for k, v in axes.items()},
        "fp32_dest_acc_en": acc,
        "plan": {"scheme": plan.scheme, "band": bool(plan.band), "native_in": bool(plan.native_in),
                 "native_out": bool(plan.native_out), "wt_per_core": plan.wt_per_core,
                 "combine": bool(plan.combine), "group_size": plan.group_size,
                 "mcast": type(plan.mcast).__name__ if plan.mcast is not None else None,
                 "gather_sem_id": plan.gather_sem_id, "l1_reserved": plan.l1_reserved,
                 "band_out_local": bool(plan.band_out_local),
                 "shard_row_bytes": plan.shard_row_bytes, "out_shard_row_bytes": plan.out_shard_row_bytes,
                 "n_cores": len(plan.assignment), "n_active": sum(1 for w in plan.assignment if w.row_count),
                 "replans": _captured.get("n_calls")},
        "assignment_sample": asg[:3] + ([asg[-1]] if len(asg) > 3 else []),
        "max_rows_per_core": max((w.row_count for w in plan.assignment), default=0),
        "kernels": kernels, "semaphores": sems, "cbs": cbs,
        "cb_total_bytes": sum(c["total_size"] for c in cbs.values() if not c.get("has_buffer")),
        "cb_arena_bytes_all": sum(c["total_size"] for c in cbs.values()),
        "in_mem": str(xt.memory_config().memory_layout),
        "in_shard": (list(map(int, xt.memory_config().shard_spec.shape))
                     if xt.memory_config().shard_spec is not None else None),
        "in_shard_grid_cores": (crs_cores(xt.memory_config().shard_spec.grid)
                                if xt.memory_config().shard_spec is not None else None),
    }

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    res = {"device": {"grid_x": grid.x, "grid_y": grid.y,
                      "l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
                      "arch": str(device.arch())},
           "knobs": {k: getattr(PD, k) for k in dir(PD) if k.isupper() and isinstance(getattr(PD, k), (int, float, tuple))},
           "cases": []}
    for c in CASES:
        try:
            r = run_case(device, *c)
        except Exception as e:
            r = {"id": c[0], "shape": list(c[1]), "why": c[-1],
                 "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
            print(f"[{c[0]}] FAILED: {type(e).__name__}: {e}", flush=True)
        else:
            p = r["plan"]
            print(f"[{c[0]}] {r['shape']} scheme={p['scheme']} band={p['band']} "
                  f"combine={p['combine']} gs={p['group_size']} cores={p['n_cores']} "
                  f"cbs={sorted(int(k) for k in r['cbs'])}", flush=True)
        res["cases"].append(r)
        ttnn.synchronize_device(device)
finally:
    ttnn.close_device(device)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1, default=str)
print("WROTE", OUT, os.path.getsize(OUT))
