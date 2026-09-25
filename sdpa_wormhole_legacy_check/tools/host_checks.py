# Host-side argument handling checks on Wormhole (HEAD). Each check in its own subprocess
# (a native crash, e.g. SIGFPE, must not take the others down).
import subprocess, sys, textwrap

PRE = textwrap.dedent('''
import torch, ttnn
dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
def t(*s):
    return ttnn.from_torch(torch.randn(*s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
q, k, v = t(1, 2, 256, 64), t(1, 2, 256, 64), t(1, 2, 256, 64)
jq, jk, jv = t(1, 2, 64, 64), t(1, 2, 64, 64), t(1, 2, 64, 64)
pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g, q_chunk_size=64, k_chunk_size=64)
def expect_raise(name, fn):
    try:
        fn(); print("RESULT", name, "NO-RAISE")
    except Exception as e:
        msg = str(e).splitlines()
        m = [l for l in msg if "TT_FATAL" in l or "TT_THROW" in l or "info:" in l or "error" in l.lower()]
        print("RESULT", name, "RAISED", type(e).__name__, "|", (m[:2] if m else msg[:2]))
''')

CHECKS = {
 "enum": "print('RESULT enum', list(ttnn.SDPAPrecision.__members__))",
 "sdpa_precision_FAST": "expect_raise('sdpa_precision_FAST', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,precision=ttnn.SDPAPrecision.FAST))",
 "sdpa_precision_ACCURATE_pc": "expect_raise('sdpa_precision_ACCURATE_pc', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,program_config=pc,precision=ttnn.SDPAPrecision.ACCURATE))",
 "sdpa_precision_LOW_prepared": "expect_raise('sdpa_precision_LOW_prepared', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,precision=ttnn.SDPAPrecision.LOW_PRECISION, inputs_prepared=True))",
 "joint_precision": "expect_raise('joint_precision', lambda: ttnn.transformer.joint_scaled_dot_product_attention(q,k,v,jq,jk,jv,joint_strategy='rear',program_config=pc,precision=ttnn.SDPAPrecision.BALANCED))",
 "sdpa_inputs_prepared_no_precision": "expect_raise('sdpa_inputs_prepared_no_precision', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,program_config=pc,inputs_prepared=True))",
 "joint_inputs_prepared_no_precision": "expect_raise('joint_inputs_prepared_no_precision', lambda: ttnn.transformer.joint_scaled_dot_product_attention(q,k,v,jq,jk,jv,joint_strategy='rear',program_config=pc,inputs_prepared=True))",
 "sdpa_chunk0": "expect_raise('sdpa_chunk0', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g)))",
 "sdpa_qchunk0_only": "expect_raise('sdpa_qchunk0_only', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=True,program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g, k_chunk_size=64)))",
 "joint_chunk0": "expect_raise('joint_chunk0', lambda: ttnn.transformer.joint_scaled_dot_product_attention(q,k,v,jq,jk,jv,joint_strategy='rear',program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g)))",
 "sdpa_explicit_ok": "o = ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=True,program_config=pc); print('RESULT sdpa_explicit_ok OK', o.shape)",
 "sdpa_no_pc_ok": "o = ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False); print('RESULT sdpa_no_pc_ok OK', o.shape)",
 "sdpa_bad_chunk_48": "expect_raise('sdpa_bad_chunk_48', lambda: ttnn.transformer.scaled_dot_product_attention(q,k,v,is_causal=False,program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g,q_chunk_size=48,k_chunk_size=64)))",
 "pc_repr_defaults": "p = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g); print('RESULT pc_repr_defaults', p.q_chunk_size, p.k_chunk_size, p.exp_approx_mode, p.max_cores_per_head_batch)",
 "chunked_sdpa_chunk0": textwrap.dedent('''
    import math
    pt = ttnn.from_torch(torch.arange(8, dtype=torch.int32).reshape(1, 8), dtype=ttnn.int32, device=dev)
    kc = ttnn.from_torch(torch.randn(8, 2, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    qq = t(1, 2, 64, 64)
    expect_raise('chunked_sdpa_chunk0', lambda: ttnn.transformer.chunked_scaled_dot_product_attention(qq, kc, kc, pt, 64, program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g)))
 '''),
 "decode_k0": textwrap.dedent('''
    dq = t(1, 1, 32, 64)
    dk = t(1, 1, 256, 64)
    cur = [255]
    o = ttnn.transformer.scaled_dot_product_attention_decode(dq, dk, dk, cur_pos=cur, program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g, q_chunk_size=0, k_chunk_size=0))
    print('RESULT decode_k0 OK', o.shape)
 '''),
}

sel = sys.argv[1:] or list(CHECKS)
for name in sel:
    code = PRE + CHECKS[name] + "\nttnn.close_device(dev)\n"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    print(f"[{name}] rc={p.returncode}", *lines or ["(no RESULT)"] + p.stderr.strip().splitlines()[-3:], sep="\n    ", flush=True)
