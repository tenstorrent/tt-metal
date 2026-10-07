# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Env-gated tuning knobs of the PREFILL attention / dense linears / shared expert (all default = the baseline behaviour).

SDPA (dense window / compressed-mask path of tt/prefill_attention.py):
  DSV41_PFA_QC, DSV41_PFA_KC        q / k chunk sizes (default 128 / 128)
  DSV41_PFA_EXP_APPROX=1            SDPAProgramConfig.exp_approx_mode (default 0)
  DSV41_PFA_QKV8=1                  cast Q and K(=V) to bfp8_b before the SDPA (output stays bf16)
  DSV41_PFA_SDPA_FID                HiFi4|HiFi2|LoFi for the prefill SDPA (default: DSV41_ATTN_SDPA_FID or HiFi4)
  DSV41_PFA_SDPA_FP32=0|1           fp32 dest accumulation of the SDPA (default 1)
sparse path (tt/prefill_sparse.py): DSV41_PFA_SP_KC (sparse_sdpa k_chunk, default 128), same SDPA_FID / SDPA_FP32 / QKV8 apply to its compute config / inputs.
dense linears (wqkv, wq_b, wo_a, wo_b; compressor weights only with DSV41_PFA_LIN_COMP=1, the rope-rotation matmul only with DSV41_PFA_LIN_ROPE=1):
  DSV41_PFA_LIN_FID                 HiFi4 (default, baseline config untouched) | HiFi2 | LoFi
  DSV41_PFA_LIN_FP32=0|1            fp32 dest accumulation (default 1)
  DSV41_PFA_LIN_L1ACC=0|1           packer_l1_acc (default 0 for fid HiFi4, else 1)
matmul program: DSV41_PFA_MM=minimal uses ttnn.experimental.minimal_matmul (as tt_transformers / llama3_70b_galaxy do for prefill) for the four projections,
  DSV41_PFA_MM_BLK="M,K,N" (tiles, default 8,8,8), DSV41_PFA_MM_SUB="h,w" (optional subblock), DSV41_PFA_MM_GRID="x,y" (default 8,10).
sparse kv table: DSV41_PFA_SP_FP8=1 stores the sparse_sdpa kv table (latents + window rows) as fp8_e4m3 (the kernel gathers 640 rows of 1 KB per query: DRAM-bound)
indexer fp4 simulation: DSV41_PFA_FP4=fast runs the 32-wide block chain in bf16 (every step is an exact power-of-two scaling or a small-integer grid value, floor(t + 0.5) stays fp32)
  on the four 32-column slices of q / k instead of a fp32 [N, 32] relayout: the same values as the fp32 chain, about half the DRAM traffic and no [..,128] <-> [N,32] reshapes.
  DSV41_PFA_FP4=fused: ONE fused generic_op kernel (tt/pf_fp4.py, tt/pf_kernels/fp4_*), bit-identical to fast (tests/test_pf_fp4_fused.py).
RoPE: DSV41_PFA_ROPE_PE=1 rotates only the last 64 (rotary) dims of the 512-wide q / kv / latent / output rows (a 64x64 pair-swap matmul on the slice, the 448 other dims pass
  through unchanged; bit-identical to the full-width 512x512 matmul formulation, whose cos = 1 / sin = 0 columns are an identity).
indexer linears of the prefill (wproj, wq_b, wk): DSV41_PFA_IDX_FID, DSV41_PFA_IDX_FP32 (default 1)
shared expert of the prefill (prefill_layer.shared_big): DSV41_PFA_SH_FID (default: unchanged), DSV41_PFA_SH_FP32.
"""

import os

import ttnn

_CACHE = {}

# umbrella flag DSV41_PREFILL_OPT (default 1; =0 restores the unmodified baseline kernels and numerics): the validated best prefill attention / linear settings below become the defaults of the knobs that are not set explicitly
# (an explicitly set DSV41_PFA_* variable always wins). Read at call time like every knob.
_OPT = {
    "DSV41_PFA_ROPE_PE": "1",
    "DSV41_PFA_LIN_FID": "HiFi2",
    "DSV41_PFA_SH_FID": "HiFi2",
    "DSV41_PFA_MM": "minimal",
    "DSV41_PFA_FP4": "fast",
    "DSV41_PFA_ENGRAM_BATCH": "1",  # one T=256 Engram forward per 8-chunk group (bit-identical)
    "DSV41_PF_MHC": "packed",  # packed mHC carrier + own-chunk routing (tt/mhc_packed.py; DSV41_PF_ROUTE_OWN defaults to 1 with it)
}


def opt_enabled():
    return os.environ.get("DSV41_PREFILL_OPT", "1") == "1"


def _env(name, default=None):  # explicit env var > umbrella default > default
    v = os.environ.get(name)
    if v is None and opt_enabled():
        v = _OPT.get(name)
    return default if v is None else v


# every knob is read from the environment at the time it is used (not at import), so a driver can flip them between two captures of the same process (tools/pfa_e2e.sh with
# DSV41_PFA_AB, demo/text_demo.py): module attributes QC, KC, EXP_APPROX, QKV8, SP_KC, SP_FP8, LIN_COMP, LIN_ROPE, ROPE_PE, MM, FP4 resolve through ``__getattr__``.
_KNOBS = {
    "QC": ("DSV41_PFA_QC", "128", int),
    "KC": ("DSV41_PFA_KC", "128", int),
    "EXP_APPROX": ("DSV41_PFA_EXP_APPROX", "0", lambda v: v == "1"),
    "QKV8": ("DSV41_PFA_QKV8", "0", lambda v: v == "1"),
    "SP_KC": ("DSV41_PFA_SP_KC", "128", int),
    "SP_FP8": ("DSV41_PFA_SP_FP8", "0", lambda v: v == "1"),
    "LIN_COMP": ("DSV41_PFA_LIN_COMP", "0", lambda v: v == "1"),
    "LIN_ROPE": ("DSV41_PFA_LIN_ROPE", "0", lambda v: v == "1"),
    "ROPE_PE": ("DSV41_PFA_ROPE_PE", "0", lambda v: v == "1"),
    "MM": ("DSV41_PFA_MM", "", str),
    "FP4": ("DSV41_PFA_FP4", "", str),
}


def __getattr__(name):
    if name in _KNOBS:
        env, default, conv = _KNOBS[name]
        return conv(_env(env, default))
    raise AttributeError(name)


def _flag(name, default):
    return _env(name, default) == "1"


ROPE_DIM = 64
_P64 = {}


def p64(md):
    """[1,1,64,64] bf16 pair-swap matrix on every device (the rotary block of ``attention.full_pair_swap``)."""
    if id(md) not in _P64:
        from models.demos.blackhole.deepseek_v41_flash.tt.attention import pair_swap_matrix

        _P64[id(md)] = ttnn.from_torch(
            pair_swap_matrix().reshape(1, 1, ROPE_DIM, ROPE_DIM),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
    return _P64[id(md)]


def rope_pe(x, c, s, P, ckc):
    """x [..., 512] -> x with its last 64 dims rotated: x_pe * c_pe + (x_pe @ P) * s_pe (c, s [1,1,n,512] tables, only their last 64 columns are read)."""
    sh = list(x.shape)
    D = sh[3]
    lo = D - ROPE_DIM
    xp = ttnn.slice(x, [0, 0, 0, lo], sh)
    cs = [ttnn.slice(t, [0, 0, 0, lo], list(t.shape)) for t in (c, s)]
    y = ttnn.add(ttnn.multiply(xp, cs[0]), ttnn.multiply(ttnn.matmul(xp, P, compute_kernel_config=ckc), cs[1]))
    out = ttnn.concat([ttnn.slice(x, [0, 0, 0, 0], sh[:3] + [lo]), y], dim=3)
    for t in (xp, y, *cs):
        ttnn.deallocate(t)
    return out


def _ckc(arch, fid, fp32, l1acc):
    key = (fid, fp32, l1acc)
    if key not in _CACHE:
        _CACHE[key] = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=l1acc,
        )
    return _CACHE[key]


def sdpa_ckc(attn):
    """compute config of the prefill SDPA (the decode attention's ``ckc_sdpa`` unless an env knob changes it)."""
    fid = _env("DSV41_PFA_SDPA_FID")
    fp32 = _env("DSV41_PFA_SDPA_FP32")
    if fid is None and fp32 is None:
        return attn.ckc_sdpa
    fid = fid or _env("DSV41_ATTN_SDPA_FID", "HiFi4")
    return _ckc(attn.mesh_device.arch(), fid, fp32 != "0", False)


def lin_ckc(attn, kind="lin"):
    """compute config of the prefill dense linears. kind: 'lin' (projections), 'comp' (compressor), 'rope' (rotation matmul)."""
    fid = _env("DSV41_PFA_LIN_FID")
    if (
        fid is None
        or (kind == "comp" and not __getattr__("LIN_COMP"))
        or (kind == "rope" and not __getattr__("LIN_ROPE"))
    ):
        return attn.ckc
    fp32 = _flag("DSV41_PFA_LIN_FP32", "1")
    l1acc = _flag("DSV41_PFA_LIN_L1ACC", "0" if fid == "HiFi4" else "1")
    return _ckc(attn.mesh_device.arch(), fid, fp32, l1acc)


def shared_ckc(sh, md):
    fid = _env("DSV41_PFA_SH_FID")
    if fid is None:
        return sh.ckc
    return _ckc(md.arch(), fid, _flag("DSV41_PFA_SH_FP32", "1"), fid != "HiFi4")


def to8(x):
    """bfp8_b copy of a bf16 tile tensor (no-op unless DSV41_PFA_QKV8=1)."""
    return ttnn.typecast(x, ttnn.bfloat8_b) if __getattr__("QKV8") and x.dtype != ttnn.bfloat8_b else x


def sdpa_cfg(md):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=md.compute_with_storage_grid_size(),
        q_chunk_size=__getattr__("QC"),
        k_chunk_size=__getattr__("KC"),
        exp_approx_mode=__getattr__("EXP_APPROX"),
    )


def idx_ckc(dec):
    """compute config of the prefill indexer linears (the decode indexer's ``ckc`` unless DSV41_PFA_IDX_FID is set)."""
    fid = _env("DSV41_PFA_IDX_FID")
    if fid is None:
        return dec.ckc
    return _ckc(
        dec.md.arch() if hasattr(dec, "md") else dec.mesh_device.arch(),
        fid,
        _flag("DSV41_PFA_IDX_FP32", "1"),
        fid != "HiFi4",
    )


def _mm_cfg(md):
    key = (
        "mm",
        id(md),
        *(_env(k, "") for k in ("DSV41_PFA_MM_BLK", "DSV41_PFA_MM_GRID", "DSV41_PFA_MM_SUB")),
    )
    if key not in _CACHE:
        m, k, n = (int(v) for v in _env("DSV41_PFA_MM_BLK", "8,8,8").split(","))
        gx, gy = (int(v) for v in _env("DSV41_PFA_MM_GRID", "8,10").split(","))
        kw = {}
        if _env("DSV41_PFA_MM_SUB"):
            kw["subblock_h"], kw["subblock_w"] = (int(v) for v in _env("DSV41_PFA_MM_SUB").split(","))
        _CACHE[key] = ttnn.MinimalMatmulConfig(
            M_block_size=m, K_block_size=k, N_block_size=n, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), **kw
        )
    return _CACHE[key]


def linear(x, w, ckc, md):
    """ttnn.linear of the prefill projections (auto program config) or, with DSV41_PFA_MM=minimal, the swept-block minimal_matmul."""
    if __getattr__("MM") == "minimal":
        cfg = _mm_cfg(md)
        M, K, N = x.shape[-2] // 32, x.shape[-1] // 32, w.shape[-1] // 32
        if (
            M % cfg.M_block_size == 0 and K % cfg.K_block_size == 0 and N % cfg.N_block_size == 0
        ):  # else the auto program config
            return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg)
    return ttnn.linear(x, w, compute_kernel_config=ckc)


def fp4_fast(x):
    """fp4 (e2m1, per-32 e8m0 scale) quantise-dequantise of a bf16 tile tensor [..., 128]; same values as ``DSV41DecodeIndexer._fp4_blocks`` on the fp32 [N, 32] view."""
    sh = list(x.shape)
    outs = []
    for k in range(sh[3] // 32):
        xk = ttnn.slice(x, [0, 0, 0, 32 * k], sh[:3] + [32 * k + 32])
        amax = ttnn.typecast(ttnn.max(ttnn.abs(xk), dim=-1, keepdim=True), ttnn.float32)
        scale32 = ttnn.exp2(ttnn.ceil(ttnn.log2(ttnn.multiply(ttnn.clamp(amax, min=6 * 2.0**-126), 1.0 / 6.0))))
        scale = ttnn.typecast(scale32, ttnn.bfloat16)  # a power of two: exact in bf16
        y = ttnn.clamp(ttnn.divide(xk, scale), min=-6.0, max=6.0)
        mag = ttnn.abs(y)
        step = ttnn.add(ttnn.multiply(ttnn.ge(mag, 2.0), 0.5), ttnn.add(ttnn.multiply(ttnn.ge(mag, 4.0), 1.0), 0.5))
        t = ttnn.typecast(ttnn.divide(mag, step), ttnn.float32)
        r = ttnn.typecast(ttnn.floor(ttnn.add(t, 0.5)), ttnn.bfloat16)
        outs.append(ttnn.multiply(ttnn.multiply(ttnn.multiply(r, step), ttnn.sign(y)), scale))
    return ttnn.concat(outs, dim=3)
