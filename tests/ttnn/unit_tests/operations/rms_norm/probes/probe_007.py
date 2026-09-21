import os, sys, json, traceback
os.environ["RMS_TRACE_BLOCKING"] = "1"
import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as SEED
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as SUCC
from eval.golden_tests.rms_norm.helpers import create_ttnn_input_tensor

NAMES = {0:"cb_input_sticks",1:"cb_input_tiles",2:"cb_x_squared",3:"cb_scaler",4:"cb_row_stat",
5:"cb_gamma_sticks",6:"cb_gamma_tiles",7:"cb_normalized",8:"cb_output_tiles",9:"cb_output_sticks",
10:"cb_sum_handoff",11:"cb_partials_gathered",12:"cb_stat_handoff",13:"cb_row_final",14:"cb_bank",
15:"cb_compact_handoff",16:"cb_mcast_in",17:"cb_gather_l1",18:"cb_node_out",19:"cb_residual_sticks",
20:"cb_residual_tiles",21:"cb_x_sum",22:"cb_bias_sticks",23:"cb_bias_tiles",24:"cb_gamma_compact",
25:"cb_bias_compact"}
REC=[]; TRACE=[]
def mk(mod, tag):
    orig = mod._cb
    def patched(index, page_size, num_pages, data_format, core_ranges):
        REC.append(dict(idx=index, name=NAMES.get(index,f"cb{index}"), page=int(page_size),
                        pages=int(num_pages), bytes=int(page_size)*int(num_pages), fmt=str(data_format).split('.')[-1]))
        return orig(index, page_size, num_pages, data_format, core_ranges)
    mod._cb = patched
    if hasattr(mod, "ttnn"):
        pass
mk(SEED,"seed"); mk(SUCC,"succ")
# zero-copy shard CBs cost 0 arena bytes; record them as 0 so the inventory is complete
_orig_zc = ttnn.cb_descriptor_from_sharded_tensor
def zc(index, tensor):
    REC.append(dict(idx=index, name=NAMES.get(index,f"cb{index}"), page=0, pages=0, bytes=0, fmt="ZEROCOPY_SHARD"))
    return _orig_zc(index, tensor)
ttnn.cb_descriptor_from_sharded_tensor = zc
SEED.ttnn.cb_descriptor_from_sharded_tensor = zc
SUCC.ttnn.cb_descriptor_from_sharded_tensor = zc

# capture the successor's blocking trace
import builtins
_print = builtins.print
def cap(*a, **k):
    s = " ".join(str(x) for x in a)
    if s.startswith("RMS_BLOCKING"): TRACE.append(s)
    else: _print(*a, **k)
builtins.print = cap

DT = {"BFLOAT16": ttnn.bfloat16, "FLOAT32": ttnn.float32, "BFLOAT8_B": ttnn.bfloat8_b, "none": None}
LY = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "none": None}
ML = {"INTERLEAVED": ttnn.TensorMemoryLayout.INTERLEAVED,
      "HEIGHT_SHARDED": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
      "WIDTH_SHARDED": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
      "BLOCK_SHARDED": ttnn.TensorMemoryLayout.BLOCK_SHARDED}
TD = {"BFLOAT16": torch.bfloat16, "FLOAT32": torch.float32, "BFLOAT8_B": torch.float32}

cells = json.load(open('/tmp/claude-1211409778/-localdev-dnijemcevic-tt-metal/2dfdebb7-e233-419d-be2f-5033ab4c1eee/scratchpad/cells.json'))
device = ttnn.open_device(device_id=0)
out = []
try:
    for i, c in enumerate(cells):
        rec = dict(c)
        try:
            torch.manual_seed(0)
            ti = torch.randn(c["shape"], dtype=torch.float32).to(TD[c["dtype"]])
            x = create_ttnn_input_tensor(ti, device, dtype=DT[c["dtype"]], layout=LY[c["layout"]],
                                         memory_layout=ML[c["ml"]])
            omc = x.memory_config() if c["ml"] != "INTERLEAVED" else x.memory_config()
            o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, omc)
            g = None
            if c["gm"] == "gamma":
                tg = torch.randn(c["shape"][-1], dtype=torch.float32).reshape(1,1,1,c["shape"][-1]).to(TD[c["gd"]])
                g = create_ttnn_input_tensor(tg, device, dtype=DT[c["gd"]], layout=LY[c["gl"]])
            ck = ttnn.ComputeConfigDescriptor()
            ck.math_fidelity = ttnn.MathFidelity.HiFi4
            ck.fp32_dest_acc_en = bool(c["f32"])
            ck.math_approx_mode = False
            for side, mod in (("seed", SEED), ("succ", SUCC)):
                REC.clear(); TRACE.clear()
                try:
                    if side == "seed":
                        mod.create_program_descriptor(x, o, gamma=g, epsilon=1e-6, compute_kernel_config=ck)
                    else:
                        mod.create_program_descriptor(x, o, weight=g, epsilon=1e-6, compute_kernel_config=ck)
                    rec[side] = dict(total=sum(r["bytes"] for r in REC), cbs=list(REC),
                                     trace=list(TRACE), err=None)
                except Exception as e:
                    rec[side] = dict(total=None, cbs=list(REC), trace=list(TRACE),
                                     err=f"{type(e).__name__}: {str(e)[:200]}")
            for t in [x, o] + ([g] if g is not None else []):
                ttnn.deallocate(t)
        except Exception as e:
            rec["setup_err"] = f"{type(e).__name__}: {str(e)[:200]}"
        out.append(rec)
        _print(f"[{i+1}/{len(cells)}] {c['tag']} {'x'.join(map(str,c['shape']))} {c['dtype']} {c['layout']} {c['ml']} g={c['gm']}/{c['gd']}/{c['gl']} f32={c['f32']} -> seed={rec.get('seed',{}).get('total')} succ={rec.get('succ',{}).get('total')} {rec.get('setup_err','')}", flush=True)
finally:
    builtins.print = _print
    ttnn.close_device(device)
json.dump(out, open('/tmp/claude-1211409778/-localdev-dnijemcevic-tt-metal/2dfdebb7-e233-419d-be2f-5033ab4c1eee/scratchpad/arena.json', "w"), indent=1)
print("wrote", '/tmp/claude-1211409778/-localdev-dnijemcevic-tt-metal/2dfdebb7-e233-419d-be2f-5033ab4c1eee/scratchpad/arena.json')
