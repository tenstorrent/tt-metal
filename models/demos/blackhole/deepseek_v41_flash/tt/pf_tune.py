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
    "DSV41_PFA_FP4": "fused",
    "DSV41_PFA_ENGRAM_BATCH": "1",  # one T=256 Engram forward per 8-chunk group (bit-identical)
    "DSV41_PF_MHC": "packed",  # packed mHC carrier + own-chunk routing (tt/mhc_packed.py; DSV41_PF_ROUTE_OWN defaults to 1 with it)
    "DSV41_PF_MHC_UMOE": "1",  # packed mHC carrier also with the unified MoE (moe_cols on the packed hh); =0 restores the 32-token chunk-loop mHC
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


def sp_kc(ratio, rows, fp8=True):
    """k_chunk of the sparse_sdpa of a layer of compress ratio ``ratio`` on ``rows`` query rows per op. DSV41_SDPA_DEDUP=1 (default OFF; T16): one chunk = the whole index row (window 128 + top-512 = 640 keys),
    which the T16 K-row-cache kernels (TT_METAL_KERNEL_PATH = tt/t16_kernels/sdpa, tests/test_t16_sdpa.py) turn into a gather of only the rows the previous query token of the core did not hold. Used where it
    pays: layers whose consecutive tokens select overlapping rows (compress ratio >= 2: 76 % overlap measured; ratio 1: 17-24 %) and every layer once the op has >= DSV41_SDPA_DEDUP_ROWS (default 1024) rows (a core then
    owns >= 8 tokens: the cold first token is amortised even with the low overlap). fp8 pool only (a bf16 pool needs twice the L1: stays on 128). Without the kernel overlay k_chunk 640 is correct but ~8 % slower than 128."""
    if _env("DSV41_SDPA_DEDUP", "0") != "1":
        return SP_KC_default()  # default OFF: nothing below runs
    if fp8 and (int(ratio) >= 2 or int(rows) >= int(_env("DSV41_SDPA_DEDUP_ROWS", "1024"))):
        _balance(False, int(rows))
        _note(("dedup", int(ratio), int(rows)), f"SDPA_DEDUP: sparse_sdpa k_chunk 640 (K-row cache) for compress ratio {ratio}, {rows} rows per op")
        return 640
    _balance(True)
    _note(("kc", int(ratio), int(rows), bool(fp8)), f"SDPA_DEDUP on but not used: sparse_sdpa k_chunk {SP_KC_default()} for compress ratio {ratio}, {rows} rows per op, fp8 pool {bool(fp8)}")
    return SP_KC_default()


_SEEN = set()


def _note(key, msg):
    """one log line per distinct (layer kind, rows) so a server log shows which sparse_sdpa configuration was chosen (DSV41_SDPA_DEDUP=1 only)"""
    if key not in _SEEN:
        _SEEN.add(key)
        print(msg, flush=True)


_BAL = {}


_MAP512 = "85.65,83.92,84.63,84.05,84.69,83.99,84.41,84.77,84.78,84.09,85.67,85.21,86.46,86.20,86.21,86.33,86.19,76.03,85.95,85.77,86.58,86.77,85.77,86.27,87.58,87.65,87.30,86.56,87.27,87.53,87.34,87.31,87.61,87.62,87.44,87.31,86.28,85.47,86.21,86.08,85.75,86.02,85.64,86.25,86.75,86.15,86.43,86.20,84.44,85.68,85.06,83.97,83.71,84.09,84.77,85.47,84.76,85.56,85.97,86.07,75.17,73.97,75.09,75.89,83.48,84.15,84.76,82.60,84.02,85.28,85.98,85.97,70.87,71.82,74.01,75.52,82.74,83.34,84.27,81.92,83.52,84.86,85.69,85.47,70.27,71.24,73.59,74.35,82.56,82.48,83.82,75.27,81.80,83.73,85.10,84.94,69.80,70.14,71.77,72.59,75.59,75.58,82.86,75.08,75.47,82.03,83.97,84.18,68.94,69.70,71.04,72.22,74.21,75.15,75.76,73.99,74.61,74.76,83.07,82.87"  # per-core us-per-token of the K-row-cache kernel at 512 rows (measured, tests/test_t16_sdpa.py zone profile: cold first token 133-191 us by core position + 53 us per warm token), for DSV41_SDPA_BALANCE_MAP


def _balance(orig, rows=0):
    """The op reads DSV41_SDPA_BALANCE when its program is created (the per-core token counts of the position-dependent gather cost table of the baseline kernel). The K-row-cache kernel has a uniform
    per-token cost: its ops are created with the balance table OFF, every other sparse_sdpa with the process value (only when DSV41_SDPA_DEDUP=1)."""
    if "v" not in _BAL:
        _BAL["v"] = os.environ.get("DSV41_SDPA_BALANCE")
        if "DSV41_SDPA_BALANCE_MAP" in os.environ:
            _BAL["m"] = os.environ["DSV41_SDPA_BALANCE_MAP"]
    if orig:
        os.environ.pop("DSV41_SDPA_BALANCE_MAP", None) if "m" not in _BAL else os.environ.__setitem__("DSV41_SDPA_BALANCE_MAP", _BAL["m"])
        if _BAL["v"] is None:
            os.environ.pop("DSV41_SDPA_BALANCE", None)
        else:
            os.environ["DSV41_SDPA_BALANCE"] = _BAL["v"]
    elif rows == 512 and _env("DSV41_SDPA_DEDUP_MAP", "0") == "1":  # DSV41_SDPA_DEDUP_MAP=1 (default OFF): the measured per-core table (-5 % op time at 512 rows)
        os.environ["DSV41_SDPA_BALANCE"] = "1"
        os.environ["DSV41_SDPA_BALANCE_MAP"] = _MAP512
    else:
        os.environ["DSV41_SDPA_BALANCE"] = "0"


def SP_KC_default():
    return int(_env("DSV41_PFA_SP_KC", "128"))


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


def rope_pe(x, c, s, P, ckc, inplace=False):
    """x [..., 512] -> x with its last 64 dims rotated: x_pe * c_pe + (x_pe @ P) * s_pe (c, s [1,1,n,512] tables, only their last 64 columns are read).
    inplace=True (and DSV41_PFA_ROPE_WRITE=1, default OFF): the rotated 64 columns are written back into ``x`` (slice_write) instead of slicing the other 448 columns and
    concatenating (2 full-tensor copies on 120 cores, ~120 us for 8 heads x 1024 rows); returns ``x`` itself, the caller must not need the unrotated values.
    """
    sh = list(x.shape)
    D = sh[3]
    lo = D - ROPE_DIM
    xp = ttnn.slice(x, [0, 0, 0, lo], sh)
    cs = [ttnn.slice(t, [0, 0, 0, lo], list(t.shape)) for t in (c, s)]
    cg = _env(
        "DSV41_PFA_ROPE_CG"
    )  # "y,x": core grid of the pair-swap matmul (default OFF: the auto config used 16 cores for the 8-head tensors)
    mkw = {"core_grid": ttnn.CoreGrid(y=int(cg.split(",")[0]), x=int(cg.split(",")[1]))} if cg else {}
    y = ttnn.add(ttnn.multiply(xp, cs[0]), ttnn.multiply(ttnn.matmul(xp, P, compute_kernel_config=ckc, **mkw), cs[1]))
    if inplace and _env("DSV41_PFA_ROPE_WRITE", "0") == "1":
        ttnn.experimental.slice_write(y, x, [0, 0, 0, lo], sh, [1, 1, 1, 1])
        for t in (xp, y, *cs):
            ttnn.deallocate(t)
        return x
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


# DSV41_PFA_MM_TABLE=1 (default OFF): per-shape minimal_matmul blocks on the 12x10 worker grid for the M = 1024 prefill projections (cores-for-prefill study, dsv4-logs/cores/CORES_PF.md).
# (K, N) per chip -> (M_block, K_block, N_block, subblock_h, subblock_w); swept single chip, tests/test_cores_pf_mm.py (shipped: 8x10 grid, 8x8x8 blocks = 80 cores). Pattern: tt_dit/utils/matmul.py
# (per-grid blocking tables, grid 11x10 / 12x10 on Blackhole Galaxy, M_block = ceil(M_tiles / grid_x)). wo_a (4608 -> 1024) stays on the shipped config (8x10 was the best of the sweep).
_MM_TABLE = {
    (5120, 1792): (8, 8, 8, 4, 1),  # wqkv   154 -> 125 us
    (1280, 4096): (3, 8, 16, 1, 4),  # wq_b   124 ->  93 us
    (1024, 5120): (2, 8, 16, 2, 2),  # wo_b   135 ->  99 us
}


def shared_1d_cfgs(h):
    """DSV41_PFA_SH_1D=1 (default OFF): 1D-multicast program configs of the two shared-expert linears of ``prefill_layer.shared_big`` for M = 128 tokens per chip (4 tile rows):
    w01 (5120 -> 4608) on 72 cores (10x8 grid, per_core_N 2), w2 (2304 -> 5120) on 80 cores (12x7 grid, per_core_N 2); the auto config of core_grid 8x8 used 48 / 56 cores.
    Swept single chip (tests/test_cores_pf_micro.py): w01 132 -> 101 us, w2 76 -> 62 us, outputs bit-identical. Pattern: models/demos/blackhole/deepseek_v41_flash/tt/attention.py:_cfg_1d.
    """
    if _env("DSV41_PFA_SH_1D", "0") != "1" or int(h.shape[-2]) != 128:
        return None, None
    if "sh1d" not in _CACHE:

        def mk(gx, gy, sbw):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=sbw,
                per_core_M=4,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        _CACHE["sh1d"] = (mk(10, 8, 2), mk(12, 7, 2))
    return _CACHE["sh1d"]


def _mm_table_cfg(x, w):
    if _env("DSV41_PFA_MM_TABLE", "0") != "1" or int(x.shape[-2]) != 1024:
        return None
    e = _MM_TABLE.get((int(w.shape[-2]), int(w.shape[-1])))
    if e is None:
        return None
    key = ("mmt",) + e
    if key not in _CACHE:
        _CACHE[key] = ttnn.MinimalMatmulConfig(
            M_block_size=e[0],
            K_block_size=e[1],
            N_block_size=e[2],
            subblock_h=e[3],
            subblock_w=e[4],
            compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
        )
    return _CACHE[key]


def linear(x, w, ckc, md):
    """ttnn.linear of the prefill projections (auto program config) or, with DSV41_PFA_MM=minimal, the swept-block minimal_matmul."""
    if __getattr__("MM") == "minimal":
        tcfg = _mm_table_cfg(x, w)
        if tcfg is not None:
            return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=tcfg)
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
