# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One llama decoder layer, prefill, on device, against a torch reference.

Eleven steps, every one of them a unified kernel:

    xn   = rmsnorm(x, w_attn)                       rmsnorm.cpp
    q    = xn @ Wq   k = xn @ Wk   v = xn @ Wv      matmul.cpp     x3
    q    = rope(q)   k = rope(k)                    rope.cpp       x2
    a    = flash_attention(q, k, v, mask)           flash_attention.cpp
    ao   = a @ Wo                                   matmul_blocked.cpp
    h    = x + ao                                   binary.cpp
    hn   = rmsnorm(h, w_ffn)                        rmsnorm.cpp
    g    = hn @ Wg   u = hn @ Wu                    matmul.cpp     x2
    f    = silu(g) * u                              binary.cpp (BN_SILU_MUL)
    y    = h + f @ Wd                               matmul.cpp + binary.cpp

Fourteen launches, one core. What this checks is the COMPOSITION: every kernel here is
covered by its own test, so what is new is whether their layouts agree end to end and
whether the numbers survive eleven steps of bf16.

ONE step is not on device, and it is named rather than hidden. The projections produce
[S, d_model], and the attention kernel still reads Q head-major and K grid-transposed --
the mirror of the output-side problem already fixed by its strided store. `to_flash_layout`
below does that rearrangement on the host. It is a layout permutation with no arithmetic, so
it does not affect what the numbers prove, but the layer is not yet a pure device pipeline
and should not be described as one. The fix is the same custom-load pattern the store uses;
see unified_llama_prefill.md.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_layer.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
import test_unified_attention_proj as proj
import test_unified_flash as flash
import test_unified_rope as rope
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

TILE = 32
EPS = 1e-5

RMSNORM_KERNEL = "unified_kernels/rmsnorm.cpp"
MATMUL_KERNEL = "unified_kernels/matmul.cpp"
ROPE_KERNEL = "unified_kernels/rope.cpp"
BINARY_KERNEL = "unified_kernels/binary.cpp"


def _dram():
    return ttnn.DRAM_MEMORY_CONFIG


# The layer's two formats. ACT is what flows between stages and WGT is what the weight
# matrices are stored as; both default to bfloat16 and run() swaps them together. They are
# separate because they are separately defensible: a weight is read once per launch and
# never accumulated, an activation is the output of one stage and the input of the next.
# Everything a kernel keeps to ITSELF -- partial sums, row statistics, the rotation matrix,
# scalar constants -- stays bfloat16 regardless. That is not caution, it is required in at
# least one place: the packer's L1 accumulate reads its destination back and adds in place,
# which a shared-exponent format cannot do.
ACT = ttnn.bfloat16
WGT = ttnn.bfloat16


def to_dev(device, t, dtype=None):
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape).to(torch.bfloat16),
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_dram(),
    )


def nan_out(device, rows, cols, dtype=None):
    """An output buffer pre-filled with NaN, so a block nothing writes is unmistakable.

    bfloat8_b cannot represent NaN -- the fill lands as some large finite value instead --
    so at that format this catches a missed block by failing the comparison rather than by
    propagating, which is weaker but not silent.
    """
    return to_dev(device, torch.full([rows, cols], float("nan")), dtype)


def bf16_pair(v):
    bits = int(torch.tensor([v], dtype=torch.bfloat16).view(torch.uint16)[0])
    return (bits << 16) | bits


# --- one launch per kernel ------------------------------------------------------------


def launch(device, kernel, cbs, tensors, rt_args=None, defines=None, named_ct_args=None):
    """`cbs` entries are (name, pages) or (name, pages, dtype); the default is bfloat16.

    A buffer's format must match the DRAM tensor it is read from or written to: the entry
    size follows the format (1088 bytes for bfloat8_b, 2048 for bfloat16) and a disagreement
    reads the wrong bytes rather than failing.

    `tensors` is {parameter name: ttnn.Tensor}, and `rt_args` {name: value} -- both named, so
    neither can drift out of position the way the old positional lists could.
    """
    core_ranges, cores = single_core()
    spec = unified_program_spec(
        kernel_source=kernel,
        nodes=core_ranges,
        dfbs=[dfb(c[0], c[1], dtype=(c[2] if len(c) > 2 else ttnn.bfloat16)) for c in cbs],
        tensors=tensors,
        named_compile_time_args=named_ct_args,
        runtime_arg_names=sorted(rt_args or {}),
        defines=defines,
        name=kernel.split("/")[-1].removesuffix(".cpp"),
    )
    run_unified_spec(device, spec, tensors, runtime_args=rt_args, nodes=cores)
    return tensors["out"]


def rmsnorm(device, x, w, ht, wt):
    out = nan_out(device, ht * TILE, wt * TILE, ACT)
    cbs = [
        ("x", ht * wt, ACT),
        ("w", wt, WGT),
        ("eps", 1),
        ("inv_n", 1),
        ("sq", ht * wt),
        ("mean", ht),
        ("rsqrt", ht),
        ("normed", ht * wt),
        ("out", ht * wt, ACT),
    ]
    # The last two are the row-chunk range. rmsnorm walks the tensor in chunks of its
    # compile-time ht, and this layer passes the whole height as one chunk -- so one chunk,
    # starting at zero. Omitting them does not fail to compile; it feeds the loop bound
    # whatever is in that argument slot, which is how this hung the device once.
    # rmsnorm walks the tensor in chunks of its compile-time ht; this layer passes the whole
    # height as one chunk. Named now, so leaving one out is an error rather than a garbage
    # loop bound -- which is how this hung the device once.
    rt = {"eps_bits": bf16_pair(EPS), "chunk_begin": 0, "chunk_count": 1}
    return launch(device, RMSNORM_KERNEL, cbs, {"x": x, "w": w, "out": out}, rt, None, [("ht", ht), ("wt", wt)])


def matmul(device, a, b, rt_dim, ct_dim, kt_dim):
    """Single-shot [rt, kt] @ [kt, ct]; the whole of b lives in L1."""
    out = nan_out(device, rt_dim * TILE, ct_dim * TILE, ACT)
    cbs = [
        ("in0", rt_dim * kt_dim, ACT),
        ("in1", kt_dim * ct_dim, WGT),
        ("out", rt_dim * ct_dim, ACT),
        ("acc", rt_dim * ct_dim),
        # Declared even unfused: matmul.cpp declares its bias Storage unconditionally.
        ("bias", ct_dim, WGT),
    ]
    defines = [
        ("MM_RT_DIM", str(rt_dim)),
        ("MM_CT_DIM", str(ct_dim)),
        ("MM_KT_DIM", str(kt_dim)),
        ("MM_K_BLOCKS", "1"),
        ("MM_SINGLE_SHOT", "1"),
    ]
    # bias is bound even unfused: the kernel names tensor::bias on every projection.
    return launch(device, MATMUL_KERNEL, cbs, {"in0": a, "in1": b, "out": out, "bias": b}, None, defines)


def apply_rope(device, x, cos, sin, m, seq_t, dim_t, chunk):
    out = nan_out(device, seq_t * TILE, dim_t * TILE, ACT)
    total = seq_t * dim_t
    assert total % chunk == 0
    cbs = [
        ("x", chunk, ACT),
        ("cos", chunk),
        ("sin", chunk),
        ("m", 1),
        ("rot", chunk),
        ("out", chunk, ACT),
    ]
    # The last two are the chunk range rope.cpp partitions across cores. One core here, so
    # it owns all of them. Omitting them compiles and feeds the loop a garbage bound.
    rt = {"chunk_begin": 0, "chunk_count": total // chunk}
    return launch(
        device,
        ROPE_KERNEL,
        cbs,
        {"x": x, "cos": cos, "sin": sin, "m": m, "out": out},
        rt,
        None,
        [("chunk", chunk), ("num_chunks", total // chunk)],
    )


def binary(device, a, b, rows, cols, mode=None):
    """Elementwise over rows x cols tiles; mode None is add, "silu_mul" is silu(a) * b."""
    out = nan_out(device, rows * TILE, cols * TILE, ACT)
    tiles = rows * cols
    cbs = [("in0", 2 * tiles, ACT), ("in1", 2 * tiles, ACT), ("out", 2 * tiles, ACT)]
    # The last two are the block range binary.cpp partitions across cores. This layer is
    # one core and one block, so it owns block 0 and there is exactly one of them. Omitting
    # them compiles fine and feeds the loop whatever is in that arg slot -- a hang.
    rt = {"block_begin": 0, "block_count": 1}
    defines = [("BN_SILU_MUL", "1")] if mode == "silu_mul" else None
    named = [("num_blocks", 1), ("tiles_per_block", tiles)]
    return launch(device, BINARY_KERNEL, cbs, {"in0": a, "in1": b, "out": out}, rt, defines, named)


# --- the host-side layout gap --------------------------------------------------------


def to_flash_layout(x_torch, n_heads, dt, sq, num_q, transpose):
    """[S, n_heads*dt] -> the per-(head, chunk) blocks the attention kernel reads.

    HOST GLUE, standing in for a strided load the attention kernel does not have yet. Pure
    permutation, no arithmetic. `transpose` produces K's grid-transposed [dt, sk] blocks;
    otherwise the [sq, dt] blocks Q and V use.
    """
    blocks = []
    for h in range(n_heads):
        for i in range(num_q):
            blk = x_torch[i * sq * TILE : (i + 1) * sq * TILE, h * dt * TILE : (h + 1) * dt * TILE]
            if transpose:
                # The same tile-grid transpose the flash harness does on K: permute the
                # TILE grid, leaving each tile's interior to matmul's transpose flag.
                blk = blk.reshape(sq, TILE, dt, TILE).permute(2, 1, 0, 3).reshape(dt * TILE, sq * TILE)
            blocks.append(blk)
    return torch.cat(blocks, dim=0)


# --- the layer -----------------------------------------------------------------------


def run(device, st, dt, n_heads, n_kv_heads, ffn_mult=2, seed=0, fmt=None):
    """st = sequence in tiles, dt = head dim in tiles. One query chunk, one k chunk.

    `fmt` sets the activation and weight formats together, bfloat16 by default. Flash
    attention and the output projection are NOT covered by it: both cross the host on the
    way in (to_flash_layout, and project's torch weight), so they build their own tensors
    and stay bfloat16. That is a gap in the coverage, not a claim about them.
    """
    global ACT, WGT
    ACT = WGT = fmt or ttnn.bfloat16
    torch.manual_seed(seed)
    dm = n_heads * dt  # d_model in tiles
    dkv = n_kv_heads * dt
    dff = ffn_mult * dm
    S, D, Dkv, Dff = st * TILE, dm * TILE, dkv * TILE, dff * TILE
    kv_group = n_heads // n_kv_heads

    # Weights scaled so eleven bf16 steps do not drift out of range.
    def w(rows, cols):
        return ((torch.rand([rows, cols]) - 0.5) / rows**0.5).to(torch.bfloat16)

    x = (0.5 + torch.rand([S, D])).to(torch.bfloat16)  # away from zero, so RMS is well posed
    w_attn, w_ffn = torch.rand([D]) + 0.5, torch.rand([D]) + 0.5
    wq, wk, wv = w(D, D), w(D, Dkv), w(D, Dkv)
    wo = w(D, D)
    wg, wu, wd = w(D, Dff), w(D, Dff), w(Dff, D)

    cos, sin = rope.cos_sin(S, dt * TILE)
    rot_m = rope.trans_mat()

    dev = {}
    for name, t, fm in (
        ("x", x, ACT),
        ("w_attn", w_attn.reshape(1, D), WGT),
        ("w_ffn", w_ffn.reshape(1, D), WGT),
        ("wq", wq, WGT),
        ("wk", wk, WGT),
        ("wv", wv, WGT),
        ("wo", wo, WGT),
        ("wg", wg, WGT),
        ("wu", wu, WGT),
        ("wd", wd, WGT),
        ("m", rot_m, ttnn.bfloat16),  # the rotation matrix is a constant, not a weight
    ):
        dev[name] = to_dev(device, t, fm)

    # --- attention block -------------------------------------------------------------
    xn = rmsnorm(device, dev["x"], dev["w_attn"], st, dm)
    q = matmul(device, xn, dev["wq"], st, dm, dm)
    k = matmul(device, xn, dev["wk"], st, dkv, dm)
    v = matmul(device, xn, dev["wv"], st, dkv, dm)

    # RoPE is a flat per-tile stream, so one launch covers all heads at once: the cos/sin
    # tables repeat every dt tiles, which is exactly how the heads are laid out.
    cos_all = torch.cat([cos] * n_heads, dim=1)
    sin_all = torch.cat([sin] * n_heads, dim=1)
    cos_kv = torch.cat([cos] * n_kv_heads, dim=1)
    sin_kv = torch.cat([sin] * n_kv_heads, dim=1)
    q = apply_rope(device, q, to_dev(device, cos_all), to_dev(device, sin_all), dev["m"], st, dm, dm)
    k = apply_rope(device, k, to_dev(device, cos_kv), to_dev(device, sin_kv), dev["m"], st, dkv, dkv)

    # The host gap: rearrange into the per-(head, chunk) blocks flash reads.
    q_t = ttnn.to_torch(q).to(torch.float32)[0, 0]
    k_t = ttnn.to_torch(k).to(torch.float32)[0, 0]
    v_t = ttnn.to_torch(v).to(torch.float32)[0, 0]
    q_fl = to_flash_layout(q_t, n_heads, dt, st, 1, transpose=False)
    k_fl = to_flash_layout(k_t, n_kv_heads, dt, st, 1, transpose=True)
    v_fl = to_flash_layout(v_t, n_kv_heads, dt, st, 1, transpose=False)

    attn = flash.run_preloaded(device, q_fl, k_fl, v_fl, sq=st, sk=st, dt=dt, n_heads=n_heads, n_kv_heads=n_kv_heads)
    ao = proj.project(device, attn, wo.to(torch.float32), sq=st, dt=dt, num_q=1, n_heads=n_heads)

    # --- residual, then the FFN ------------------------------------------------------
    h = binary(device, dev["x"], to_dev(device, ao, ACT), st, dm)
    hn = rmsnorm(device, h, dev["w_ffn"], st, dm)
    g = matmul(device, hn, dev["wg"], st, dff, dm)
    u = matmul(device, hn, dev["wu"], st, dff, dm)
    f = binary(device, g, u, st, dff, mode="silu_mul")
    dproj = matmul(device, f, dev["wd"], st, dm, dff)
    y = binary(device, h, dproj, st, dm)

    def dl(t):
        return ttnn.to_torch(t).to(torch.float32)[0, 0]

    # --- reference ------------------------------------------------------------------
    def rms(t, weight):
        tf = t.to(torch.float32)
        return tf / torch.sqrt(tf.pow(2).mean(dim=-1, keepdim=True) + EPS) * weight.to(torch.float32)

    xf = x.to(torch.float32)
    xnr = rms(xf, w_attn)
    qr = xnr @ wq.to(torch.float32)
    kr = xnr @ wk.to(torch.float32)
    vr = xnr @ wv.to(torch.float32)

    def rope_ref(t, heads):
        out = t.clone()
        for hh in range(heads):
            sl = slice(hh * dt * TILE, (hh + 1) * dt * TILE)
            out[:, sl] = t[:, sl] * cos + rope.rotate_pairs(t[:, sl]) * sin
        return out

    qr, kr = rope_ref(qr, n_heads), rope_ref(kr, n_kv_heads)

    causal = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    heads_out = []
    for hh in range(n_heads):
        qh = qr[:, hh * dt * TILE : (hh + 1) * dt * TILE]
        kvh = hh // kv_group
        kh = kr[:, kvh * dt * TILE : (kvh + 1) * dt * TILE]
        vh = vr[:, kvh * dt * TILE : (kvh + 1) * dt * TILE]
        sc = qh @ kh.T / (dt * TILE) ** 0.5
        sc = sc.masked_fill(~causal, float("-inf"))
        heads_out.append(torch.softmax(sc, dim=-1) @ vh)
    ar = torch.cat(heads_out, dim=1)
    hr = xf + ar @ wo.to(torch.float32)
    hnr = rms(hr, w_ffn)
    fr = torch.nn.functional.silu(hnr @ wg.to(torch.float32)) * (hnr @ wu.to(torch.float32))
    want = hr + fr @ wd.to(torch.float32)

    # EVERY stage is checked, not just the output, and that is not thoroughness for its own
    # sake -- the final residual dominates it. y = h + f @ Wd with h = x + ao, and x here is
    # positive and of unit scale while the branches are smaller, so an error inside a branch
    # is diluted before it reaches y. Measured: swapping silu's operands (silu(u) * g instead
    # of silu(g) * u) moved the output pcc from 0.999948 to 0.999815, and dropping RoPE on K
    # entirely moved it to 0.999954 -- both inside any tolerance worth setting. Checking the
    # stages catches each of those where it happens instead.
    return [
        ("rmsnorm", dl(xn), xnr),
        # Q, K and V where they are produced, which is not optional. Checking them only
        # through the attention output does not work: with random weights the scores are
        # near-uniform, so softmax returns roughly the mean of V whatever the scores are,
        # and the output barely moves. Measured -- dropping RoPE on K entirely changed the
        # attention stage from 0.018 to 0.015, i.e. not at all. That is the same vacuity the
        # flash harness fixes with a ramp on the keys; here the stages do it instead.
        ("rope_q", q_t, qr),
        ("rope_k", k_t, kr),
        ("v_proj", v_t, vr),
        ("attention", attn, ar),
        ("out_proj", ao, ar @ wo.to(torch.float32)),
        ("residual", dl(h), hr),
        ("rmsnorm_ffn", dl(hn), hnr),
        ("silu_mul", dl(f), fr),
        ("layer", dl(y), want),
    ]


def main(argv=None):
    p = argparse.ArgumentParser()
    # Relative L2 per stage. Tight, because each stage is compared where it happens rather
    # than after the residual has diluted it.
    p.add_argument("--rel", type=float, default=0.06)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        # Both formats, same threshold. bfloat8_b roughly triples the per-stage error and
        # still clears it by a wide margin -- the point of running it here rather than only
        # on a matmul is that eleven stages compound, and a format that is fine once is not
        # automatically fine in series.
        for label, fmt in (("bf16", None), ("bf8 ", ttnn.bfloat8_b)):
            for st, dt, nh, nkv in ((2, 2, 2, 1), (2, 2, 4, 2), (4, 2, 2, 2), (2, 4, 2, 1)):
                stages = run(device, st, dt, nh, nkv, fmt=fmt)
                tag = f"S={st * TILE} d_model={nh * dt * TILE} heads={nh} kv={nkv}"
                parts = []
                for name, got, want in stages:
                    rel = ((got - want).norm() / want.norm()).item()
                    ok = rel <= args.rel
                    parts.append(f"{name}={rel:.5f}{'' if ok else '!'}")
                    if not ok:
                        failed.append(f"{label.strip()}-{name}-{st}-{dt}-{nh}-{nkv}")
                logger.info(f"layer {label} {tag}: " + " ".join(parts))
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
