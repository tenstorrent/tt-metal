# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GPTQ for ttnn bfloat4_b weights (pure torch, no ttnn): quantizer, ttnn BFP emulator, device groupings, marker.

ttnn's BFP4_b/BFP8_b block = one 16-element FACE ROW of a tile, i.e. 16 contiguous elements along the LAST dim of the
tensor handed to ttnn. Our weights reach ttnn as [K=in, N=out] (tp_common.shard_w transposes HF [out, in]), so a block
is 16 OUTPUT rows of one input column k ("N-grouped"). The shared exponent is the max fp32 exponent field of the 16
bf16 values, recomputed by ttnn on every conversion (tt_metal/impl/data_format/blockfloat_common.cpp get_max_exp /
convert_u32_to_bfp). A value is m * 2^(E-129), m in 0..7, sign separate.

tt-lab (requant.cpp:691-960) instead groups along K: its 16-value section is 16 consecutive INPUT columns of one
output row, it writes the packed BFP bits itself, and it picks each section's exponent once, at the section's first
column, from the already error-updated weights (choose_bfp_exponent: max exp or max exp - 1, H-diagonal-weighted MSE),
then runs GPTQ over the 16 columns with that exponent fixed; its act-order permutes whole 16-column sections.

Adaptation here (N-grouping):
  * Every block lives inside ONE GPTQ column, so the exponent is chosen when that column is quantized, from its
    current (error-updated) values. No exponent state is carried across columns, so act-order can permute single
    columns freely (no section granularity, no static groups).
  * We cannot write packed bits: the pipeline emits bf16 values that ttnn re-packs. Every emitted value is an exact
    grid point m * 2^(E-129) (<= 3 mantissa bits, exactly representable in bf16) and every non-zero block keeps one
    element with m >= 4, so ttnn's recomputed exponent is E again and its round-to-nearest-even is a no-op: the
    conversion is a fixed point (checked by `ttnn_bfp_roundtrip`, and on device by test_gptq_bfp4_ttnn_roundtrip).
  * tt-lab's "max exp - 1" option (clip the block max to m=7, finer step for the rest) stays a fixed point (the
    clipped element has m=7 at exponent E-1), so it is available as exp_search=True. All 16 elements of a block share
    the column's Hessian diagonal, so tt-lab's H-weighted section error reduces to plain squared error here.
  * Block membership is the DEVICE layout's, not HF's: TP sharding and the GDN in-proj fusion pack per-device
    [a_d | b_d] (nv/tp rows each) into shared blocks (see `device_groups`). That layout depends on TP, so a
    checkpoint is calibrated for one TP degree (recorded in the marker; the model refuses a mismatch).
"""

import hashlib
import json
import os

import torch

BLOCK = 16
BFP4_MANT_BITS = 3
BFP8_MANT_BITS = 7

# Marker written next to config.json of a GPTQ checkpoint. Its presence makes Qwen36ModelArgs.weight_cache_path use a
# separate tensor cache dir, so GPTQ tensors never collide with (or load as) the RTN caches under the same TT_CACHE_PATH.
MARKER_NAME = "gptq_bfp4.json"
MARKER_FORMAT = "ttnn-bfp4_b-ngroup16-v1"

# QWEN36_GPTQ_BFP4=1: preset for a GPTQ checkpoint. Changes the DEFAULTS of QWEN36_MLP_GATEUP_BFP8 (->0),
# QWEN36_BFP4_MLP_DOWN, QWEN36_BFP4_GDN_IN, QWEN36_BFP4_GDN_OUT (->1); an explicit value of any of those still wins.
# Attention and LM head stay BFP8. Requires a checkpoint with MARKER_NAME calibrated for the running TP degree.
GPTQ_PRESET_ENV = "QWEN36_GPTQ_BFP4"


def gptq_preset_enabled():
    return os.environ.get(GPTQ_PRESET_ENV, "0") == "1"


def read_marker(ckpt_dir):
    """Parsed marker dict (+ "_id": sha256 of the marker bytes) or None when ckpt_dir is not a GPTQ checkpoint."""
    if ckpt_dir is None:
        return None
    p = os.path.join(str(ckpt_dir), MARKER_NAME)
    if not os.path.isfile(p):
        return None
    raw = open(p, "rb").read()
    meta = json.loads(raw)
    if meta.get("format") != MARKER_FORMAT:
        raise ValueError(f"{p}: format {meta.get('format')!r} != {MARKER_FORMAT!r}")
    meta["_id"] = hashlib.sha256(raw).hexdigest()
    return meta


def cache_suffix(meta):
    """Tensor-cache dir suffix for a GPTQ checkpoint ("" for a normal checkpoint)."""
    return "" if meta is None else f"_gptq{meta['_id'][:12]}"


# ---------------------------------------------------------------------------------------------------------------------
# Device groupings (HF orientation: W [N=out, K=in]; a group = 16 row indices of N, -1 = tile padding)
# ---------------------------------------------------------------------------------------------------------------------


def consecutive_groups(n):
    """Rows 16j..16j+15. Exact for every target whose per-device N slice is a multiple of 16 and whose fusions only
    move whole >=16-row blocks: MLP gate/up (also the tile-pair-interleaved prefill gate_up), MLP down, GDN out
    (row- and column-parallel copies), GDN in_proj_qkv (prepare_gdn_qkv moves 128-row head blocks) and in_proj_z."""
    assert n % BLOCK == 0, f"N={n} not a multiple of {BLOCK}"
    return torch.arange(n, dtype=torch.long).view(-1, BLOCK)


def gdn_ab_groups(nv, tp, base=0):
    """in_proj_a / in_proj_b stacked as rows [a (nv); b (nv)] starting at `base`. gdn/tp.py puts device d's
    a[d*nvp:(d+1)*nvp] and b[same] side by side (fused qkvzab after qkv_per+z_per columns, or the separate "ab"
    weight at column 0), then the tile pads the device slice with zeros. So a block mixes a and b rows of ONE device:
    TP8/nv48 -> [a6 b6 pad4]; TP4 -> [a12 b4], [b8 pad8]."""
    assert nv % tp == 0, f"nv={nv} not divisible by tp={tp}"
    nvp = nv // tp
    groups = []
    for d in range(tp):
        rows = [base + d * nvp + i for i in range(nvp)] + [base + nv + d * nvp + i for i in range(nvp)]
        rows += [-1] * ((-len(rows)) % BLOCK)
        groups += [rows[i : i + BLOCK] for i in range(0, len(rows), BLOCK)]
    return torch.tensor(groups, dtype=torch.long)


def gdn_in_groups(qkv_rows, z_rows, nv, tp):
    """Groups of the stacked GDN in-proj [in_proj_qkv; in_proj_z; in_proj_a; in_proj_b] (all share one input).
    Requires the fused qkvzab per-device offset (qkv_rows+z_rows)/tp to be a multiple of 16, else a/b would share a
    block with z rows and this grouping would be wrong."""
    assert qkv_rows % tp == 0 and z_rows % tp == 0
    assert ((qkv_rows + z_rows) // tp) % BLOCK == 0, "a/b offset in the fused qkvzab device slice not 16-aligned"
    assert (qkv_rows // tp) % BLOCK == 0 and (z_rows // tp) % BLOCK == 0
    return torch.cat(
        [consecutive_groups(qkv_rows + z_rows), gdn_ab_groups(nv, tp, base=qkv_rows + z_rows)],
        dim=0,
    )


# ---------------------------------------------------------------------------------------------------------------------
# ttnn BFP emulation (bit-exact port of blockfloat_common.cpp convert_u32_to_bfp, non-truncating)
# ---------------------------------------------------------------------------------------------------------------------


def bfp_blocks(x, mant_bits):
    """x float32 [..., 16]: each last-dim row is one block. Returns what ttnn's BFP{4,8}_b pack + unpack produces.
    Same algorithm as qwen38_handoff/emulation/bfp.py, which was validated against ttnn (max diff 0)."""
    assert x.shape[-1] == BLOCK
    shape = x.shape
    x = x.float().contiguous().reshape(-1, BLOCK)
    b = x.view(torch.int32)
    exp = (b >> 23) & 0xFF
    sign = (b >> 31) & 1
    man = (b & 0x7FFFFF) | (1 << 23)
    man = torch.where(exp == 0, torch.zeros_like(man), man)  # zero / fp32 subnormal -> 0
    smax = exp.max(dim=1, keepdim=True).values
    d = smax - exp
    man = man >> d.clamp(max=31)  # truncating align
    man = torch.where(d > 31, torch.zeros_like(man), man)
    S = 24 - mant_bits
    rv = man & ((1 << S) - 1)
    m = man >> S
    tie = 1 << (S - 1)
    up = (rv > tie) | ((rv == tie) & ((m & 1) == 1))  # RNE
    m = torch.clamp(m + up.to(m.dtype), max=(1 << mant_bits) - 1)  # saturate, no carry into the exponent
    scale = torch.ldexp(torch.ones_like(x), (smax - 127 - (mant_bits - 1)).to(torch.int32))
    out = m.float() * scale
    out = torch.where(sign.bool() & (m != 0), -out, out)
    out = torch.where(smax == 0, torch.zeros_like(out), out)
    return out.reshape(shape)


def ttnn_bfp_roundtrip(w_hf, groups, mant_bits=BFP4_MANT_BITS, k_chunk=2048):
    """Emulate shard_w(w_hf, dtype=bfloat{4,8}_b) -> to_torch for HF-orientation w_hf [N, K] with the given device
    groups [G, 16] (-1 = zero pad). The weight is cast to bf16 first, as shard_w does. Returns float32 [N, K]."""
    N, K = w_hf.shape
    idx = groups.clamp(min=0).to(w_hf.device)
    valid = (groups >= 0).to(w_hf.device)
    out = torch.zeros(N, K, dtype=torch.float32, device=w_hf.device)
    flat_idx = idx[valid]
    for k0 in range(0, K, k_chunk):
        xb = w_hf[:, k0 : k0 + k_chunk].to(torch.bfloat16).float()
        blk = xb[idx] * valid.unsqueeze(-1)  # [G, 16, k]
        q = bfp_blocks(blk.transpose(1, 2), mant_bits).transpose(1, 2)  # blocks along the 16 axis
        out[flat_idx, k0 : k0 + k_chunk] = q[valid]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Grid quantizer + GPTQ
# ---------------------------------------------------------------------------------------------------------------------


def _round_at(v, E):
    """v [G,16] float32, E [G] int exponent fields -> nearest grid values m*2^(E-129), m<=7 (torch.round = RNE)."""
    scale = torch.ldexp(torch.ones_like(v), (E - 129).to(torch.int32).unsqueeze(-1))
    m = torch.clamp(torch.round(v.abs() / scale), max=7.0)
    return torch.copysign(m * scale, v)


def quantize_column(w, groups, exp_search=False):
    """One GPTQ column w [N] (float32) -> its BFP4 grid values [N]; blocks = device groups [G,16]. Pad (-1) entries
    are zero, as ttnn's tile padding. The result is a ttnn fixed point: every nonzero block keeps an element with
    m >= 4 at the chosen exponent."""
    idx = groups.clamp(min=0)
    valid = groups >= 0
    v = torch.where(valid, w[idx], torch.zeros((), dtype=w.dtype, device=w.device)).float()
    E = ((v.view(torch.int32) >> 23) & 0xFF).amax(dim=1)  # sign masked by the & (exp field only)
    q = _round_at(v, E)
    if exp_search:
        E1 = (E - 1).clamp(min=0)
        q1 = _round_at(v, E1)
        better = ((q1 - v) ** 2).sum(1) < ((q - v) ** 2).sum(1)
        q = torch.where((better & (E >= 4)).unsqueeze(-1), q1, q)
    # Blocks whose grid would be bf16-subnormal (E < 3, |w| < 2^-124) flush to zero exactly as ttnn reads them.
    q = torch.where((E >= 3).unsqueeze(-1), q, torch.zeros_like(q))
    out = torch.zeros_like(w, dtype=torch.float32)
    out[idx[valid]] = q[valid]
    return out


def rtn_quantize(W, groups, exp_search=False):
    """Round-to-nearest baseline over all columns (what ttnn does to a bf16 weight when exp_search=False)."""
    Q = torch.empty_like(W, dtype=torch.float32)
    for k in range(W.shape[1]):
        Q[:, k] = quantize_column(W[:, k].float(), groups, exp_search)
    return Q


def proxy_loss(W, Q, H):
    """tr((W-Q) H (W-Q)^T): GPTQ's layer-output squared error proxy (H = sum x x^T over calibration tokens)."""
    D = (W.float() - Q.float()).to(H.dtype)
    return float(((D @ H) * D).sum())


def gptq_quantize(W, H, groups, blocksize=128, damp=0.01, act_order=True, exp_search=False):
    """GPTQ (Frantar et al. 2022; reference implementation structure) with the N-grouped BFP4 column quantizer.

    W [N, K] HF-orientation weight (any float dtype), H [K, K] float32 Hessian (sum of x x^T), groups [G,16].
    Returns float32 Q [N, K] of BFP4 grid values in the ORIGINAL column order. Columns whose input never fired
    (H_kk == 0) are rounded to nearest and pass no error on (their weights are kept, not zeroed)."""
    W = W.float().clone()
    N, K = W.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    H += torch.eye(K, device=H.device, dtype=H.dtype) * (damp * torch.mean(torch.diag(H)))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = quantize_column(w, groups, exp_search)
            Q1[:, i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q
