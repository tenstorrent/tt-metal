"""Plan numbers for Mistral-Small-24B-Base-2501 LoRA on one Blackhole p300 chip."""
H, NH, NKV, D, I, V, L = 5120, 32, 8, 128, 32768, 131072, 40
T, B, R = 1024, 1, 16
PEAK = 148.5e12   # 110 cores x 1.35 TFLOP/s bf16 (ttml.common.performance)
BW = 512e9        # nominal GDDR6 per chip
DRAM = 8 * 4272341376   # 31.83 GiB, from ttnn.device.get_memory_view on chip 0
MM_EFF = 0.62     # measured on this box: large HiFi4 matmuls, Qwen3.8 Tracy 2026-10-21
SDPA_EFF = 0.50
BW_EFF = 0.70
bf = 2
QD, KVD = NH*D, NKV*D
lin = {"qkv": (H, QD+2*KVD), "out": (QD, H), "gate_up": (H, 2*I), "down": (I, H)}
P_layer = sum(i*o for i, o in lin.values())
P_emb = V*H; P_head = V*H
P_total = L*P_layer + P_emb + P_head + (2*L+1)*H
lora_layer = sum(R*(i+o) for i, o in lin.values())
print(f"params/layer {P_layer/1e6:.1f}M  total {P_total/1e9:.3f}B  emb {P_emb/1e6:.0f}M  LoRA/layer {lora_layer/1e6:.3f}M  LoRA total {L*lora_layer/1e6:.1f}M")
tok = B*T
# ---- FLOPs per step (useful): fwd + dX for frozen linears, 3x fwd for LoRA, attention fwd+bwd, lm head fwd+dX
f_lin_fwd = 2*tok*P_layer
f_lora = 3*2*tok*lora_layer
f_attn_fwd = 4*NH*D*T*T*B          # QK^T and PV, full square (standard 12·L·H·Q·T convention: 4 fwd + 8 bwd)
f_attn_bwd = 2*f_attn_fwd
f_layer = 2*f_lin_fwd + f_lora + f_attn_fwd + f_attn_bwd
f_head = 2*2*tok*P_head
f_total = L*f_layer + f_head
print(f"FLOPs/step: layer {f_layer/1e12:.4f} T (lin fwd {f_lin_fwd/1e12:.4f}, lora {f_lora/1e9:.1f} G, attn {(f_attn_fwd+f_attn_bwd)/1e9:.1f} G), head {f_head/1e12:.3f} T, total {f_total/1e12:.2f} T, per token {f_total/tok/1e9:.2f} G")
# ---- roofline per op (ms)
def mm(m, n, k, rd_bytes, wr_bytes):
    fl = 2*m*n*k
    return fl/PEAK*1e3, (rd_bytes+wr_bytes)/BW*1e3, fl
rows = []
def add(name, cls, t_c, t_m, mult=1, flops=0):
    rows.append((name, cls, t_c, t_m, max(t_c, t_m)*mult, mult, flops*mult))
# linears fwd/dX
for name, (i, o) in lin.items():
    w = i*o*bf; x = tok*i*bf; y = tok*o*bf
    tc, tm, fl = mm(tok, o, i, w+x, y); add(f"{name} fwd", "math", tc, tm, 1, fl)
    tc, tm, fl = mm(tok, i, o, w+y, x); add(f"{name} dX", "math", tc, tm, 1, fl)
    # LoRA fused ideal: x@A^T (read x once more), h@B^T + add (write/read update) -> count A/B matmul flops + traffic of x and y once each
    fl = 3*2*tok*R*(i+o)
    add(f"{name} LoRA (fused ideal)", "lora", fl/PEAK*1e3, (x+y + x+y)/BW*1e3, 1, fl)
# attention
add("sdpa fwd", "math", f_attn_fwd/PEAK*1e3, (3*tok*D*bf*NH + tok*QD*bf)/BW*1e3, 1, f_attn_fwd)
add("sdpa bwd", "math", f_attn_bwd/PEAK*1e3, (6*tok*QD*bf)/BW*1e3, 1, f_attn_bwd)
# swiglu
gu = tok*2*I*bf; h = tok*I*bf
add("swiglu fwd", "math", 0, (gu+h)/BW*1e3)
add("swiglu bwd", "math", 0, (gu+h+gu)/BW*1e3)
xb = tok*H*bf
add("rmsnorm fwd x2", "math", 0, 2*(2*xb)/BW*1e3)
add("rmsnorm bwd x2 (no dgamma)", "math", 0, 2*(3*xb)/BW*1e3)
rq = tok*(QD+KVD)*bf
add("rope q,k fwd", "math", 0, 2*rq/BW*1e3)
add("rope q,k bwd", "math", 0, 2*rq/BW*1e3)
add("residual add x2 fwd + grad accum bwd", "glue", 0, (2*3*xb + 2*3*xb)/BW*1e3)
add("heads creation/fusion fwd+bwd", "layout", 0, (2*(tok*(QD+2*KVD)*bf) + 2*tok*QD*bf)/BW*1e3)
layer_ms = sum(r[4] for r in rows)
layer_math = sum(r[4] for r in rows if r[1] in ("math", "lora"))
print("\nPer-layer roofline (ms): name | class | compute | memory | roofline")
for r in rows: print(f"  {r[0]:36s} {r[1]:7s} {r[2]:7.3f} {r[3]:7.3f} {r[4]:7.3f}")
print(f"per layer roofline total {layer_ms:.3f} ms (math {layer_math:.3f}, overhead {layer_ms-layer_math:.3f})")
# fixed
fixed = []
w = P_head*bf; xo = tok*H*bf; lg = tok*V*bf
tc, tm, fl = mm(tok, V, H, w+xo, lg); fixed.append(("lm_head fwd", max(tc,tm)))
tc, tm, fl = mm(tok, H, V, w+lg, xo); fixed.append(("lm_head dX", max(tc,tm)))
fixed.append(("cross-entropy fwd", (lg)/BW*1e3)); fixed.append(("cross-entropy bwd", (2*lg)/BW*1e3))
fixed.append(("embedding lookup", (2*xo)/BW*1e3)); fixed.append(("final rmsnorm fwd+bwd", (5*xo)/BW*1e3))
nlora = L*lora_layer
fixed.append(("AdamW on LoRA params", (nlora*bf*7)/BW*1e3))
fixed_ms = sum(f[1] for f in fixed)
print("\nFixed roofline (ms):")
for f in fixed: print(f"  {f[0]:28s} {f[1]:7.3f}")
print(f"fixed total {fixed_ms:.2f} ms")
roof = L*layer_ms + fixed_ms
print(f"\nROOFLINE step = {L}x{layer_ms:.3f} + {fixed_ms:.2f} = {roof:.1f} ms -> MFU {f_total/PEAK/(roof/1e3)*100:.1f}%")
# achievable
# LoRA adapters as the existing ttml path runs them: rank 16 padded to 32, plain ttnn matmuls at the rates
# mm_eff.py measured on this chip for M=1024, rank 32 at K/N=5120, extrapolated to K=32768 / N=65536 (TFLOP/s): [T,K]x[K,32] 4.3, [T,32]x[32,N] 3.6,
# dB [N,T]x[T,32] 6.9, dA [32,T]x[T,K] 7.7; plus the scale/add chain of LoraLinear streamed at BW_EFF.
RP = 32
A_CLS, B_CLS, DB_CLS, DA_CLS = 4.3e12, 3.6e12, 6.9e12, 7.7e12
def lora_measured(i, o):
    fwd = 2*tok*i*RP/A_CLS + 2*tok*RP*o/B_CLS
    bwd = 2*tok*o*RP/A_CLS + 2*tok*o*RP/DB_CLS + 2*tok*RP*i/B_CLS + 2*tok*RP*i/DA_CLS
    glue = (2*(3*tok*o*bf) + 3*tok*o*bf + 3*tok*i*bf)/(BW*BW_EFF)
    return (fwd + bwd + glue)*1e3
lora_meas_layer = sum(lora_measured(i, o) for i, o in lin.values())
def ach(r):
    name, cls, tc, tm, _, mult, _ = r
    if cls == "lora":
        i, o = lin[name.split()[0]]
        return lora_measured(i, o)
    if cls != "math": return max(tc, tm)
    if "sdpa" in name: return max(tc/SDPA_EFF, tm/BW_EFF)
    if tc > 0: return max(tc/MM_EFF, tm/BW_EFF)
    return tm/BW_EFF
layer_ach = sum(ach(r) for r in rows)
fixed_ach = 0
for name, t in fixed:
    fixed_ach += t/MM_EFF if "lm_head" in name else t/BW_EFF
OPS_PER_LAYER, GAP_US = 90, 16   # exposed gap at full scale (Qwen3.8 diary); host floor is ~25 us/op
dispatch = (L*OPS_PER_LAYER + 40) * GAP_US / 1e3
target = L*layer_ach + fixed_ach + dispatch
print(f"LoRA adapters at measured skinny-matmul rates: {lora_meas_layer:.2f} ms/layer (fused ideal {sum(r[4] for r in rows if r[1]=='lora'):.2f})")
print(f"ACHIEVABLE (existing kernels): per layer {layer_ach:.2f} ms, fixed {fixed_ach:.1f} ms, dispatch {dispatch:.0f} ms -> {target:.0f} ms/step, MFU {f_total/PEAK/(target/1e3)*100:.1f}%")
stretch = target - L*(lora_meas_layer - sum(r[4] for r in rows if r[1]=='lora')/BW_EFF)
print(f"STRETCH (if a memory-bound LoRA path lands): {stretch:.0f} ms/step, MFU {f_total/PEAK/(stretch/1e3)*100:.1f}%")
# ---- memory
w_layer = P_layer*bf; w_fixed = (P_emb+P_head)*bf
act_layer = (xb + xb + tok*(QD+2*KVD)*bf + tok*QD*bf + NH*T*32*4 + tok*QD*bf + xb + xb + gu + h)
# What ttml autograd actually keeps: every op output stays alive until its backward node runs (each grad
# closure captures `out`; autograd/tensor.cpp releases closures only as backward proceeds). Per layer:
qkv_w, lse = tok*(QD+2*KVD)*bf, NH*T*32*4
def retained(lora_ops):
    # lora_ops: outputs kept per LoRA linear besides the base output: LoraLinear 5-op chain = h, update, scaled, sum;
    # LoraApply = h, sum.  Sizes: h [T,32], update/scaled/sum [T,out].
    keep = xb            # norm1 out
    keep += qkv_w        # qkv base out
    keep += qkv_w        # heads (3 tensors)
    keep += tok*(QD+KVD)*bf  # rope outputs (q, k)
    keep += tok*QD*bf + lse   # sdpa out + lse
    keep += tok*QD*bf    # concat heads
    keep += xb           # out-proj base out
    keep += xb           # residual sum (block output after attention)
    keep += xb           # norm2 out
    keep += gu           # gate_up base out
    keep += h            # swiglu out
    keep += xb           # down base out
    keep += xb           # residual sum (block output)
    for i, o in lin.values():
        keep += tok*RP*bf + lora_ops * tok*o*bf
    return keep
ret_chain, ret_apply = retained(3), retained(1)
print(f"\nRETAINED per layer (ttml keeps every op output until its backward): LoraLinear chain {ret_chain/2**20:.0f} MiB, LoraApply {ret_apply/2**20:.0f} MiB (minimal set {act_layer/2**20:.0f} MiB)")
print(f"MEMORY: weights/layer {w_layer/2**30:.3f} GiB, emb+head {w_fixed/2**30:.2f} GiB, full model {(L*w_layer+w_fixed)/2**30:.1f} GiB vs DRAM {DRAM/2**30:.0f} GiB")
print(f"saved activations/layer {act_layer/2**20:.0f} MiB; logits bf16 {lg/2**20:.0f} MiB (+ fp32 CE intermediates {tok*V*4/2**20:.0f} MiB)")
lora_state = nlora*bf*4
print(f"LoRA params+grads+2 moments {lora_state/2**20:.0f} MiB")
print("layers | LoraLinear chain | LoraApply | minimal set   (weights + retained activations + logits/CE + LoRA state + 1.5 GiB workspace; limit 95% of DRAM)")
for n in (1,2,4,8,12,16,40):
    line = f"  {n:2d} layers:"
    for act in (ret_chain, ret_apply, act_layer):
        tot = n*w_layer + w_fixed + n*act + 3*lg + tok*V*4 + lora_state + 1.5*2**30
        line += f"  {tot/2**30:5.1f} GiB {'fits' if tot < DRAM*0.95 else 'NO  '}"
    print(line)

