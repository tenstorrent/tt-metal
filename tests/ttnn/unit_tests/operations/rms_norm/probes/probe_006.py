import os
os.environ["RMS_TRACE_BLOCKING"] = "1"
import json, torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as SEED
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as SUCC

NAMES = {0:"cb_input_sticks",1:"cb_input_tiles",2:"cb_x_squared",3:"cb_scaler",4:"cb_row_stat",
5:"cb_gamma_sticks",6:"cb_gamma_tiles",7:"cb_normalized",8:"cb_output_tiles",9:"cb_output_sticks",
10:"cb_sum_handoff",11:"cb_partials_gathered",12:"cb_stat_handoff",13:"cb_row_final",14:"cb_bank",
15:"cb_compact_handoff",16:"cb_mcast_in",17:"cb_gather_l1",18:"cb_node_out",19:"cb_residual_sticks",
20:"cb_residual_tiles",21:"cb_x_sum",22:"cb_bias_sticks",23:"cb_bias_tiles",24:"cb_gamma_compact",
25:"cb_bias_compact"}

REC = []
def mk(mod, tag):
    orig = mod._cb
    def patched(index, page_size, num_pages, data_format, core_ranges):
        try:
            nc = len(list(ttnn.corerange_to_cores(core_ranges, None, True)))
        except Exception:
            nc = -1
        REC.append(dict(side=tag, idx=index, name=NAMES.get(index, f"cb{index}"),
                        page=int(page_size), pages=int(num_pages),
                        bytes=int(page_size)*int(num_pages), fmt=str(data_format), cores=nc))
        return orig(index, page_size, num_pages, data_format, core_ranges)
    mod._cb = patched
mk(SEED, "seed"); mk(SUCC, "succ")

DT = {"BFLOAT16": ttnn.bfloat16, "FLOAT32": ttnn.float32, "BFLOAT8_B": ttnn.bfloat8_b}
LY = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}

CELLS = [
 # name, shape, dtype, layout, gamma(dtype,layout) or None, fp32_dest_acc
 ("A_8192_nogamma", [1,1,32,8192], "BFLOAT16", "TILE", None, False),
 ("B_8192_gamma",   [1,1,32,8192], "BFLOAT16", "TILE", ("BFLOAT16","TILE"), False),
 ("C_3000_gamma",   [1,1,992,3000],"BFLOAT16", "TILE", ("BFLOAT16","TILE"), False),
 ("D_2047_gamma",   [1,1,544,2047],"BFLOAT16", "TILE", ("BFLOAT16","TILE"), False),
]

def cta(pd):
    out = []
    for k in pd.kernels:
        src = getattr(k, "kernel_source", "?")
        src = src.split("/")[-1] if isinstance(src, str) else str(src)
        out.append((src, list(getattr(k, "compile_time_args", []) or [])))
    return out

device = ttnn.open_device(device_id=0)
results = {}
try:
    for name, shape, dt, ly, g, f32acc in CELLS:
        torch.manual_seed(0)
        ti = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16 if dt=="BFLOAT16" else torch.float32)
        x = ttnn.from_torch(ti, dtype=DT[dt], layout=LY[ly], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.from_torch(torch.zeros_like(ti), dtype=DT[dt], layout=LY[ly], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gt = None
        if g:
            gd, gl = g
            tg = torch.randn(shape[-1], dtype=torch.float32).reshape(1,1,1,shape[-1])
            gt = ttnn.from_torch(tg, dtype=DT[gd], layout=LY[gl], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ck = ttnn.ComputeConfigDescriptor()
        ck.math_fidelity = ttnn.MathFidelity.HiFi4
        ck.fp32_dest_acc_en = f32acc
        ck.math_approx_mode = False
        for side, mod in (("seed", SEED), ("succ", SUCC)):
            REC.clear()
            print(f"\n######## {name} / {side} shape={shape} dt={dt} ly={ly} gamma={g} f32acc={f32acc}")
            try:
                if side == "seed":
                    pd = mod.create_program_descriptor(x, o, gamma=gt, epsilon=1e-6, compute_kernel_config=ck)
                else:
                    pd = mod.create_program_descriptor(x, o, weight=gt, epsilon=1e-6, compute_kernel_config=ck)
                tot = sum(r["bytes"] for r in REC)
                for r in REC:
                    print(f"  {r['name']:<24} idx={r['idx']:<3} page={r['page']:<6} pages={r['pages']:<5} bytes={r['bytes']:<9} fmt={r['fmt']:<24} cores={r['cores']}")
                print(f"  TOTAL_ARENA_BYTES={tot}")
                for src, args in cta(pd):
                    print(f"  CTA {src}: {args}")
                results[(name, side)] = (list(REC), tot)
            except Exception as e:
                print(f"  EXC {type(e).__name__}: {str(e)[:300]}")
        for t in [x, o] + ([gt] if gt is not None else []):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(device)

print("\n\n===== SUMMARY TOTALS =====")
for k, v in results.items():
    print(k, v[1])
