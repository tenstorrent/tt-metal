# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 differential contract suite for ttnn.topk.

Pins the *actual implemented* contract of the three engines behind the one
public symbol (single-core insertion sort, multi-core bitonic, and the
Blackhole large-k route through ttnn.experimental.topk_large_indices),
covering the input classes no existing test exercises:

  nan          +NaN / -NaN placement + payload/sign preservation (I2/I3)
  zeros        mixed +0/-0 mass straddling the k boundary (I4)
  subnormal    datapath survival of bf16/fp32 subnormals (I6 - the one unknown)
  ties         constructed duplicate mass at the k-th boundary (I7)
  infleak      fewer-than-k finite rows: stock padding-index leak vs routed
               0xFFFFFFFF sentinel (I8)
  determinism  same input x3 launches, bit-identical outputs (I13)
  gates        routing-gate boundary flips (k=64/65, k=2048/2049, pow2,
               width 65504/65534/65536, W=63/64, k=W, sorted no-op, stable)

BF16 DATAPATH CANONICALIZATION (measured on silicon 2026-08-16, p150a):
  the bf16 compute datapath (unpack -> Dst -> SFPU -> pack) canonicalizes
  special values BEFORE the sort ever sees them.  Proven with a pure SFPU
  identity op (no topk logic): NaN of any payload -> Inf of the same sign
  (0x7FFF/0x7FC0/0x7FC1 -> 0x7F80; 0xFFC0/0xFFC1/0xFFFF -> 0xFF80),
  -0 -> +0, +-subnormal -> +0 (sign dropped); +-Inf, +-min-normal and all
  normals are bit-preserved.  fp32 passes through bit-exact.  Consistent
  with Dst not supporting NaN (tt-isa-documentation BlackholeA0 Dst.md:69)
  and datapath denormal flushing (SFPSTORE.md:48).  The sort therefore
  ranks the CANONICALIZED values (ties between originally-distinct specials
  resolve in implementation-defined order); the values output is
  canonicalized while indices keep the original positions.  All reference
  models below apply canonicalize_bf16_datapath() first; a T3
  documented-divergence ledger row is emitted whenever it mutates any lane.

Three assertion tiers (research report section 4.3):
  T1  hard invariants any correct topk must satisfy: the returned value
      multiset is the exact top-k of the CANONICALIZED input under the
      SFPSWAP sign-magnitude total order (-NaN < -Inf < ... < -0 < +0 < ...
      < +Inf < +NaN, ISA: SFPSWAP.md:94-98); indices unique and in [0, W)
      except the documented per-engine +-inf-lane exceptions;
      canonicalized_input[index] == value bit-exact.  Failure => test fails.
  T2  incumbent-contract pins (exact sorted bit-sequence under the
      sign-magnitude model, index-dtype boundary, routed sentinel,
      sorted-flag no-op, determinism).  Failure => ledger row + test fails.
  T3  informational torch-parity diffs and padding-leak observations:
      recorded in the divergence ledger, never asserted.

Divergence ledger: JSON-lines file, one row per T2/T3 mismatch (and one
"engine_predicted" info row per cell).  Path: $TOPK_CONTRACT_LEDGER, default
<TT_METAL_HOME or cwd>/generated/topk_contract_ledger.jsonl.

Engine detection is a pure-Python reimplementation of the routing gates with
file:line references (see predict_engine below); the selected factory is not
observable from Python, so the ledger records the *predicted* engine.

Default run is the decisive ~50-cell subset; TOPK_CONTRACT_FULL=1 unlocks the
full matrix (fp32/largest=False mirrors, routed width ceiling 2^19, dim=1
transpose cell, more k-alignment cells).
"""

import json
import math
import os
import time

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn
from models.common.utility_functions import is_blackhole

FULL = os.environ.get("TOPK_CONTRACT_FULL", "0") == "1"

# ---------------------------------------------------------------------------
# Bit-pattern constants
# ---------------------------------------------------------------------------
UINT16_MAX = 65535  # topk_device_operation.cpp:70,294

BF16_POS_QNAN = 0x7FC0
BF16_POS_NAN_MAXPAY = 0x7FFF
BF16_NEG_QNAN = 0xFFC0
BF16_POS_INF = 0x7F80
BF16_NEG_INF = 0xFF80
BF16_POS_ZERO = 0x0000
BF16_NEG_ZERO = 0x8000
BF16_MIN_NORMAL = 0x0080

FP32_POS_QNAN = 0x7FC00000
FP32_POS_NAN_PAYLOAD = 0x7FC00001
FP32_POS_NAN_MAXPAY = 0x7FFFFFFF
FP32_NEG_QNAN = 0xFFC00000
FP32_POS_INF = 0x7F800000
FP32_NEG_INF = 0xFF800000
FP32_POS_ZERO = 0x00000000
FP32_NEG_ZERO = 0x80000000
FP32_MIN_NORMAL = 0x00800000

# Routed -inf-lane sentinel: emitted as 0xFFFFFFFF by topk_large_indices
# (topk_large_indices_device_operation.cpp:63-64), 0xFFFF after the UINT16
# typecast (topk.cpp:211-217, :315-318).
SENTINEL_U32 = 0xFFFFFFFF
SENTINEL_U16 = 0xFFFF


# ---------------------------------------------------------------------------
# Divergence ledger (JSON lines)
# ---------------------------------------------------------------------------
def _ledger_path():
    p = os.environ.get("TOPK_CONTRACT_LEDGER")
    if p:
        return p
    base = os.environ.get("TT_METAL_HOME", os.getcwd())
    return os.path.join(base, "generated", "topk_contract_ledger.jsonl")


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        v = v.flatten().tolist()[:16]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in list(v)[:16]]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)


def ledger(tier, check, cell, engine, expected=None, actual=None, note=""):
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
        "tier": tier,
        "check": check,
        "engine": engine,
        "cell": _jsonable(cell),
        "expected": _jsonable(expected),
        "actual": _jsonable(actual),
        "note": note,
    }
    path = _ledger_path()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------
def bits_of(t):
    """IEEE bits of a bf16/fp32 tensor as int64 (unsigned interpretation)."""
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).to(torch.int64) & 0xFFFF
    assert t.dtype == torch.float32
    return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def signmag_keys(t):
    """Monotone map from IEEE bits to the SFPSWAP sign-magnitude total order.

    SFPSWAP min/max treats operands as 32-bit sign-magnitude integers
    (tt-isa-documentation/BlackholeA0/.../SFPSWAP.md:3,:94-98), giving
    -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN.  Both bitonic device
    paths and topk_large_indices compare exclusively with SFPSWAP
    (ckernel_sfpu_topk.h:648-1015).
    """
    b = bits_of(t)
    if t.dtype == torch.bfloat16:
        sign = 1 << 15
        full = (1 << 16) - 1
    else:
        sign = 1 << 31
        full = (1 << 32) - 1
    return torch.where(b < sign, b + sign, full - b)


def bf16_from_bits(bits):
    b = torch.as_tensor(bits, dtype=torch.int64)
    b = torch.where(b >= 0x8000, b - 0x10000, b).to(torch.int16)
    return b.view(torch.bfloat16)


def fp32_from_bits(bits):
    b = torch.as_tensor(bits, dtype=torch.int64)
    b = torch.where(b >= 0x80000000, b - 0x100000000, b).to(torch.int32)
    return b.view(torch.float32)


def from_bits(bits, dtype):
    return bf16_from_bits(bits) if dtype == torch.bfloat16 else fp32_from_bits(bits)


def canonicalize_bf16_datapath(t):
    """Silicon-measured model of the Blackhole bf16 compute-datapath value
    canonicalization (see module docstring: identity-op probe, 2026-08-16):

        NaN (any payload)  -> Inf of the same sign
        -0                 -> +0
        subnormal (+-)     -> +0 (sign dropped)
        everything else    -> bit-preserved

    fp32 tensors pass through unchanged (fp32 cells verify bit-exactness).
    The device sorts these canonicalized values, so every reference model
    must be built on top of this map."""
    if t.dtype != torch.bfloat16:
        return t
    b = bits_of(t)
    exp = b & 0x7F80
    man = b & 0x007F
    sign = b & 0x8000
    is_nan = (exp == 0x7F80) & (man != 0)
    is_subn = (exp == 0) & (man != 0)
    is_negzero = b == 0x8000
    out = torch.where(is_nan, sign | 0x7F80, b)
    out = torch.where(is_subn | is_negzero, torch.zeros_like(b), out)
    return from_bits(out, t.dtype).reshape(t.shape)


def flush_subnormals(t, mode):
    """Model of a datapath that flushes subnormals: 'keep_sign' -> signed
    zero, 'pos0' -> +0.  Used only to classify the observed subnormal
    behavior (I6); the classification is ledgered, not assumed."""
    b = bits_of(t)
    if t.dtype == torch.bfloat16:
        exp_mask, mant_mask, sign_mask = 0x7F80, 0x007F, 0x8000
    else:
        exp_mask, mant_mask, sign_mask = 0x7F800000, 0x007FFFFF, 0x80000000
    subn = ((b & exp_mask) == 0) & ((b & mant_mask) != 0)
    flushed = torch.where(subn, (b & sign_mask) if mode == "keep_sign" else torch.zeros_like(b), b)
    return from_bits(flushed, t.dtype).reshape(t.shape)


# ---------------------------------------------------------------------------
# Engine prediction: pure-Python mirror of the routing gates
# ---------------------------------------------------------------------------
def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _roundup(x, m):
    return m * ((x + m - 1) // m)


def _largest_pow2_le(x):
    # topk_utils.cpp:23
    return 0 if x == 0 else 1 << (x.bit_length() - 1)


# tile sizes per tt::tile_size for a 32x32 tile
_TILE_BYTES = {"bf16": 2048, "fp32": 4096, "u16": 2048, "u32": 4096}


def multicore_cost_ok(padded_w, adjusted_k, dtype, grid_x, grid_y, l1_size):
    """Mirror of find_topk_core_config / verify_multi_core_cost
    (topk_utils.cpp:55-197), called from select_program_factory with the
    full-grid core range (topk.cpp:462-466 default sub_core_grids ->
    num_cores_to_corerangeset over the whole compute grid, whose first range
    is CoreRange((0,0),(grid_x-1,grid_y-1)))."""
    tile_w = 32
    # core_range end coords are inclusive: end_x = grid_x-1, end_y = grid_y-1.
    end_x, end_y = grid_x - 1, grid_y - 1
    # topk_utils.cpp:66-67 (asymmetric -1 on y is faithful to the source)
    max_cores = (end_y - 0 - 1) * (end_x - 0)
    if max_cores <= 0:
        return False
    value_tile = _TILE_BYTES[dtype]
    # fp32 forces UInt32-sized index CBs (topk_device_operation.cpp:85-87)
    index_tile = _TILE_BYTES["u32"] if dtype == "fp32" else _TILE_BYTES["u16"]
    # transposed intermediates are at least bf16-sized (topk_utils.cpp:78-83)
    transposed_tile = max(value_tile, _TILE_BYTES["bf16"])
    # topk_utils.cpp:75-76
    start_split = (padded_w // tile_w // _largest_pow2_le(max_cores)) * tile_w
    if start_split == 0:
        return False  # C++ TT_FATALs here; treat as infeasible
    max_dim = padded_w // 2  # topk_device_operation.cpp:101
    min_dim = 64  # min_dim_per_core, topk_constants.hpp:12
    split = start_split
    while split <= max_dim:
        rem = padded_w % split
        num_cores = padded_w // split + (1 if rem > 0 else 0)
        # topk_utils.cpp:99-108 per-core L1 cost model
        wt_final = (num_cores * max(adjusted_k, tile_w)) // tile_w
        wt_local = split // tile_w
        shared = 4 * (value_tile + index_tile) + wt_final * (transposed_tile + index_tile) + 2 * index_tile
        final_cost = shared + 2 * value_tile + wt_final * (transposed_tile + index_tile)
        local_cost = shared + wt_local * (transposed_tile + index_tile) + 2 * transposed_tile
        per_core_cost = max(final_cost, local_cost)
        # topk_utils.cpp:111-117
        max_x, max_y = end_x - 0, end_y - 0 - 1
        if num_cores <= max_x * max_y:
            # topk_utils.cpp:127-136 contiguous rectangle search
            contiguous = any(x * y == num_cores for y in range(max_y, 0, -1) for x in range(max_x, 0, -1))
            # topk_utils.cpp:138-143
            if (
                num_cores <= max_cores
                and per_core_cost < l1_size
                and num_cores > 1
                and split >= min_dim
                and contiguous
                and rem == 0
            ):
                return True
        split *= 2
    return False


def predict_engine(
    w,
    k,
    *,
    dtype="bf16",
    largest=True,
    stable=False,
    dim_last=True,
    user_indices=False,
    prealloc=False,
    sub_core_grids=False,
    sharded=False,
    tile_layout=True,
    blackhole=True,
    grid_x=13,
    grid_y=10,
    l1_size=1572864,
):
    """Which engine answers ttnn.topk(w-wide rows, k)?  Pure function of the
    call parameters, mirroring the two routing levels:

    Level 1 (composite): should_route_to_topk_large_indices, topk.cpp:247-295.
    Level 2 (device op): select_program_factory, topk_device_operation.cpp:59-115.
    Returns 'routed' | 'multi_core' | 'single_core'.
    """
    # ---- Level 1: large-k Blackhole route (topk.cpp:247-295) ----
    k_rounded16 = _roundup(k, 16)  # large_k_route_k_multiple, topk.cpp:240,:290
    if (
        largest  # topk.cpp:257-259
        and not stable  # topk.cpp:262-264
        and not (user_indices or prealloc or sub_core_grids)  # topk.cpp:267-269
        and dim_last  # topk.cpp:271-273
        and 64 < k <= 2048  # topk.cpp:236,:238,:274-276
        and dtype == "bf16"  # topk.cpp:277-279
        and tile_layout  # topk.cpp:280-282
        and not sharded  # topk.cpp:283-285
        and blackhole  # topk.cpp:286-288
        and k_rounded16 <= w <= (1 << 19)  # topk.cpp:245,:289-294
    ):
        return "routed"

    # ---- Level 2: device op sees the host-padded tensor ----
    # width < 64 is host-padded to 64 (topk.cpp:503-519, min_dim_per_core),
    # then tile padding rounds to a multiple of 32; the gates use padded_shape
    # (topk_device_operation.cpp:63,:66,:70,:72).
    padded_w = _roundup(max(w, 64), 32)
    # k is rounded up to a multiple of 32 before the device op
    # (get_nearest_supported_k_value, topk.cpp:42-44, applied :470-472).
    adjusted_k = _roundup(k, 32)

    multicore = (
        padded_w >= 8192  # multi_core_min_width, topk_constants.hpp:11 / device_operation.cpp:66
        and padded_w < UINT16_MAX  # strictly < 65535, device_operation.cpp:70
        and _is_pow2(padded_w)  # bitonic requirement, device_operation.cpp:72
        and adjusted_k <= 64  # device_operation.cpp:75
        and multicore_cost_ok(padded_w, adjusted_k, dtype, grid_x, grid_y, l1_size)  # :98-107
    )
    return "multi_core" if multicore else "single_core"  # device_operation.cpp:111-114


def predict_engine_for_device(device, w, k, **kw):
    grid = device.compute_with_storage_grid_size()
    # device->l1_size_per_core(): 1,572,864 B on Blackhole, 1,499,136 B on
    # Wormhole (not bound to Python; margins at our cells are ~3x so this
    # cannot flip a prediction).
    l1 = 1572864 if is_blackhole() else 1499136
    kw.setdefault("blackhole", is_blackhole())
    return predict_engine(w, k, grid_x=grid.x, grid_y=grid.y, l1_size=l1, **kw)


def expected_index_ttnn_dtype(w, dtype):
    """Index dtype contract (compute_output_specs, topk_device_operation.cpp:
    290-304): UINT16 iff tile-padded width <= 65535 AND input != FLOAT32.
    The routed path replicates the same boundary (topk.cpp:313-318)."""
    padded_w = _roundup(max(w, 64), 32)
    if padded_w <= UINT16_MAX and dtype != "fp32":
        return ttnn.uint16
    return ttnn.uint32


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def build_input(W, specials, *, dtype=torch.bfloat16, rows=32, largest=True, filler="worse", seed=1234):
    """(1,1,rows,W) tensor.  `specials` (a 1-D tensor already in `dtype`) is
    planted at distinct random positions per row.  Filler is strictly worse
    than any finite special for the given `largest` direction ('worse'), or
    the pad extremum itself ('inf_tail')."""
    g = torch.Generator().manual_seed(seed)
    if filler == "worse":
        base = (
            -1000.0 + 500.0 * torch.rand((rows, W), generator=g)
            if largest
            else 1000.0 - 500.0 * torch.rand((rows, W), generator=g)
        )
        base = base.to(dtype)
    elif filler == "inf_tail":
        base = torch.full((rows, W), -math.inf if largest else math.inf, dtype=dtype)
    else:
        raise ValueError(filler)
    if specials is not None and len(specials) > 0:
        s = specials.to(dtype)
        for r in range(rows):
            pos = torch.randperm(W, generator=g)[: len(s)]
            base[r, pos] = s
    return base.reshape(1, 1, rows, W)


def gaussian_input(W, *, dtype=torch.bfloat16, rows=32, seed=2005):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((1, 1, rows, W), generator=g) * 0.9).to(dtype)


# ---------------------------------------------------------------------------
# Core verifier
# ---------------------------------------------------------------------------
def verify_topk_cell(
    device,
    torch_input,
    k,
    *,
    largest=True,
    sorted_flag=True,
    stable=False,
    dim=-1,
    cell_id="",
    subnormal_mode=False,
    index_checks_hard=True,
    torch_parity=True,
):
    """Runs ttnn.topk on `torch_input` and applies the 3-tier contract checks.

    Returns dict with predicted engine, output values/indices (reduction dim
    moved last), and the sign-magnitude keys of both sides."""
    rank = torch_input.dim()
    dim_n = dim if dim >= 0 else dim + rank
    dim_last = dim_n == rank - 1
    W = torch_input.shape[dim_n]
    is_fp32 = torch_input.dtype == torch.float32
    dtype_str = "fp32" if is_fp32 else "bf16"
    tt_dtype = ttnn.float32 if is_fp32 else ttnn.bfloat16

    engine = predict_engine_for_device(device, W, k, dtype=dtype_str, largest=largest, stable=stable, dim_last=dim_last)
    cell = dict(
        cell_id=cell_id,
        shape=list(torch_input.shape),
        dim=dim,
        W=int(W),
        k=int(k),
        dtype=dtype_str,
        largest=largest,
        sorted=sorted_flag,
        stable=stable,
    )
    ledger("info", "engine_predicted", cell, engine, actual=engine)

    padded_w = _roundup(max(W, 64), 32)

    # -- reference under the sign-magnitude order model, applied to the
    #    bf16-datapath-canonicalized input (documented divergence: the device
    #    mutates NaN->same-sign Inf, -0->+0, subnormal->+0 before sorting) --
    ref_input = torch_input.movedim(dim_n, -1).contiguous()
    canon_input = canonicalize_bf16_datapath(ref_input)
    n_mutated = int((bits_of(canon_input) != bits_of(ref_input)).sum())
    if n_mutated:
        ledger(
            "T3",
            "documented_divergence:bf16_datapath_canonicalization",
            cell,
            engine,
            expected="input bits preserved through the op",
            actual={"lanes_mutated": n_mutated},
            note="bf16 datapath canonicalizes NaN->same-sign Inf, -0->+0, subnormal->+0 "
            "pre-sort (silicon identity-op probe 2026-08-16); values output is "
            "canonicalized, indices keep original positions",
        )
    exact_keys = signmag_keys(canon_input)
    ref_keys, _ = torch.topk(exact_keys, k, dim=-1, largest=largest, sorted=True)

    # -- run --
    tt_in = ttnn.from_torch(torch_input, tt_dtype, layout=ttnn.Layout.TILE, device=device)
    tt_vals, tt_idx = ttnn.topk(tt_in, k, dim=dim, largest=largest, sorted=sorted_flag, stable=stable)

    # T2: index dtype boundary (padded width vs 65535, fp32 forces u32)
    exp_idx_dtype = expected_index_ttnn_dtype(W, dtype_str)
    if tt_idx.dtype != exp_idx_dtype:
        ledger("T2", "index_dtype_boundary", cell, engine, expected=str(exp_idx_dtype), actual=str(tt_idx.dtype))
    assert tt_idx.dtype == exp_idx_dtype, f"{cell_id}: index dtype {tt_idx.dtype} != expected {exp_idx_dtype}"
    u16 = exp_idx_dtype == ttnn.uint16

    # shape contract
    exp_shape = list(torch_input.shape)
    exp_shape[dim_n] = k
    assert list(tt_vals.shape) == exp_shape, f"{cell_id}: values shape {list(tt_vals.shape)} != {exp_shape}"
    assert list(tt_idx.shape) == exp_shape, f"{cell_id}: indices shape {list(tt_idx.shape)} != {exp_shape}"

    out_vals = ttnn.to_torch(tt_vals).movedim(dim_n, -1).contiguous()
    out_idx = ttnn.to_torch(tt_idx, dtype=torch.uint16 if u16 else torch.uint32).movedim(dim_n, -1).to(torch.int64)
    act_keys = signmag_keys(out_vals)

    # -- subnormal survival classification (I6) --
    # 'exact' here means "the canonicalized-datapath model is bit-exact"; for
    # fp32 canon_input == ref_input so the original semantics are unchanged.
    # For bf16 the canonical model already flushes subnormals to +0, so the
    # two flush_* variants only ever fire on fp32 (or a future arch change).
    ref_used = canon_input
    variant = "exact"
    if subnormal_mode:
        candidates = [
            ("exact", canon_input),
            ("flush_keep_sign", flush_subnormals(canon_input, "keep_sign")),
            ("flush_to_pos0", flush_subnormals(canon_input, "pos0")),
        ]
        variant = None
        for name, cand in candidates:
            ck, _ = torch.topk(signmag_keys(cand), k, dim=-1, largest=largest, sorted=True)
            if torch.equal(ck, act_keys):
                variant, ref_used = name, cand
                ref_keys = ck
                break
        ledger(
            "T3",
            "subnormal_survival",
            cell,
            engine,
            expected="exact",
            actual=variant or "NONE_OF_3_MODELS",
            note="which datapath model reproduces the observed output bit-sequence",
        )
        assert variant is not None, (
            f"{cell_id}: output matches neither exact nor flushed subnormal models; "
            f"row0 actual keys {act_keys.reshape(-1, k)[0][:12].tolist()} vs exact "
            f"{ref_keys.reshape(-1, k)[0][:12].tolist()}"
        )

    in_bits = bits_of(ref_used).reshape(-1, W)
    val_bits = bits_of(out_vals).reshape(-1, k)
    idx = out_idx.reshape(-1, k)
    a_keys = act_keys.reshape(-1, k)
    r_keys = ref_keys.reshape(-1, k)

    # -- T1: value multiset is the exact top-k under sign-magnitude order --
    a_sorted, _ = torch.sort(a_keys, dim=-1)
    r_sorted, _ = torch.sort(r_keys, dim=-1)
    if not torch.equal(a_sorted, r_sorted):
        bad_rows = (a_sorted != r_sorted).any(dim=-1).nonzero().flatten().tolist()
        ledger("T1", "value_multiset", cell, engine, expected=r_sorted[bad_rows[0]], actual=a_sorted[bad_rows[0]])
        assert False, f"{cell_id}: value multiset differs from sign-magnitude top-k in rows {bad_rows[:8]}"

    # -- T2: exact sorted bit-sequence (descending for largest, ascending
    #    otherwise; sorted-always is the de-facto contract, topk.cpp/report 2.3).
    #    A total order on bits makes this sequence unique, so it also pins NaN
    #    payload/sign preservation and the -0 < +0 boundary preference. --
    if not torch.equal(a_keys, r_keys):
        bad_rows = (a_keys != r_keys).any(dim=-1).nonzero().flatten().tolist()
        ledger("T2", "value_order_signmag", cell, engine, expected=r_keys[bad_rows[0]], actual=a_keys[bad_rows[0]])
        assert False, f"{cell_id}: sorted value sequence differs from sign-magnitude model in rows {bad_rows[:8]}"

    # -- T1/T2: index validity --
    extremum = (
        (BF16_NEG_INF if largest else BF16_POS_INF) if not is_fp32 else (FP32_NEG_INF if largest else FP32_POS_INF)
    )
    sentinel = (SENTINEL_U16 if u16 else SENTINEL_U32) if engine == "routed" else None

    in_range = (idx >= 0) & (idx < W)
    is_sent = (idx == sentinel) if sentinel is not None else torch.zeros_like(idx, dtype=torch.bool)
    leak = (~in_range) & (~is_sent) & (idx >= W) & (idx < padded_w)
    bad = ~(in_range | is_sent | leak)

    def _idx_fail(mask, check, note):
        rows_bad = mask.any(dim=-1).nonzero().flatten().tolist()
        ledger(
            "T1" if index_checks_hard else "T3",
            check,
            cell,
            engine,
            actual={"rows": rows_bad[:8], "idx": idx[rows_bad[0]][mask[rows_bad[0]]][:8]},
            note=note,
        )
        if index_checks_hard:
            assert False, f"{cell_id}: {check} in rows {rows_bad[:8]} ({note})"

    if bad.any():
        _idx_fail(bad, "index_out_of_domain", f"index not in [0,{W}) nor leak [{W},{padded_w}) nor sentinel")

    # gather: input[index] == value, bit-exact (T1)
    gathered = torch.gather(in_bits, 1, idx.clamp(0, W - 1))
    g_bad = in_range & (gathered != val_bits)
    if g_bad.any():
        _idx_fail(g_bad, "gather_bit_mismatch", "input bits at returned index != returned value bits")

    # leak lanes may only carry the pad extremum value (stock engines'
    # documented -inf padding-index leak, topk.cpp:215-217)
    l_bad = leak & (val_bits != extremum)
    if l_bad.any():
        _idx_fail(l_bad, "leak_lane_not_extremum", "index >= W allowed only for +-inf pad-valued lanes")
    if leak.any():
        ledger(
            "T3",
            "padding_index_leak_observed",
            cell,
            engine,
            actual={"lanes": int(leak.sum()), "rows": int(leak.any(dim=-1).sum())},
            note="stock-path indices pointing into [W, padded_W) on pad-extremum lanes (documented divergence I8)",
        )

    # routed sentinel contract (T2): every -inf-valued lane carries the
    # sentinel, and every sentinel lane is -inf-valued (topk.cpp:211-217).
    if engine == "routed":
        neg_inf_bits = BF16_NEG_INF
        s_bad = is_sent & (val_bits != neg_inf_bits)
        v_bad = (val_bits == neg_inf_bits) & ~is_sent
        if s_bad.any() or v_bad.any():
            ledger(
                "T2",
                "routed_sentinel",
                cell,
                engine,
                actual={"sentinel_nonneginf": int(s_bad.sum()), "neginf_nonsentinel": int(v_bad.sum())},
            )
            assert not s_bad.any() and not v_bad.any(), f"{cell_id}: routed -inf sentinel contract violated"

    # uniqueness of in-range indices per row (T1)
    for r in range(idx.shape[0]):
        v = idx[r][in_range[r]]
        if v.unique().numel() != v.numel():
            ledger("T1" if index_checks_hard else "T3", "index_duplicates", cell, engine, actual=v[:16])
            if index_checks_hard:
                assert False, f"{cell_id}: duplicate in-range indices in row {r}"

    # -- T3: torch parity (informational, never asserted) --
    if torch_parity:
        try:
            t_vals, _ = torch.topk(ref_input, k, dim=-1, largest=largest, sorted=True)
            t_keys = signmag_keys(t_vals).reshape(-1, k)
            ndiff = int((t_keys != a_keys).sum())
            if ndiff:
                pos = (t_keys != a_keys).nonzero()[:8].tolist()
                ledger(
                    "T3",
                    "torch_value_order_diff",
                    cell,
                    engine,
                    expected="torch.topk order",
                    actual={"lanes_diff": ndiff, "first_positions": pos},
                    note="expected on NaN/+-0 cells: hw sign-magnitude order vs torch (report 2.1)",
                )
        except Exception as e:  # torch-side reference failure is informational
            ledger("T3", "torch_parity_error", cell, engine, actual=str(e))

    return dict(engine=engine, values=out_vals, indices=out_idx, keys=act_keys, ref_keys=ref_keys, variant=variant)


def _skip_routed_off_bh(engine):
    if engine == "routed" and not is_blackhole():
        pytest.skip("large-k route is Blackhole-only (topk.cpp:286-288)")


def _assert_engine(res, engine, cell_id):
    # Engine labels in the matrix were derived for Blackhole; only pin there.
    if is_blackhole():
        assert res["engine"] == engine, f"{cell_id}: predicted engine {res['engine']} != matrix cell {engine}"


# ---------------------------------------------------------------------------
# Engine cells (report section 4.4 decisive set)
#   single-core:  W=10000 (non-pow2 -> single, topk_device_operation.cpp:72)
#   multi-core:   W=8192 pow2, k<=64 (topk_device_operation.cpp:66-107)
#   routed:       W=8192, k=96 (should_route_to_topk_large_indices)
# ---------------------------------------------------------------------------
STOCK_CELLS_TF = [
    ("single_core", 10000, 32, True),
    ("single_core", 10000, 32, False),
    ("multi_core", 8192, 32, True),
    ("multi_core", 8192, 32, False),
]
ROUTED_CELL = [("routed", 8192, 96, True)]
ENGINE_CELLS = STOCK_CELLS_TF + ROUTED_CELL


def _ids(cells):
    return [f"{e}-W{w}-k{k}-{'largest' if l else 'smallest'}" for (e, w, k, l) in cells]


# ===========================================================================
# nan — I2 (+NaN payloads) / I3 (-NaN, documented divergence from torch)
# ===========================================================================
BF16_NAN_SPECIALS = [
    BF16_POS_NAN_MAXPAY,
    BF16_POS_QNAN,
    BF16_NEG_QNAN,
    BF16_POS_INF,
    BF16_NEG_INF,
]
BF16_NAN_NORMALS = [3.0, -3.0, 1.5, -1.5, 0.5, -0.5, 100.0, -100.0]


@pytest.mark.parametrize("engine,W,k,largest", ENGINE_CELLS, ids=_ids(ENGINE_CELLS))
def test_contract_nan_bf16(engine, W, k, largest, device):
    _skip_routed_off_bh(engine)
    specials = torch.cat([bf16_from_bits(BF16_NAN_SPECIALS), torch.tensor(BF16_NAN_NORMALS, dtype=torch.bfloat16)])
    x = build_input(W, specials, largest=largest, seed=101)
    cell_id = f"nan-{engine}-W{W}-k{k}-{largest}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)

    # bf16 datapath canonicalization: both planted +NaNs surface as +Inf and
    # tie with the planted +Inf; both -NaN and -Inf surface as -Inf.  NaN
    # payload/sign-magnitude NaN ordering is NOT observable in the values
    # output on bf16 (documented divergence; fp32 cells still pin payloads).
    vb = bits_of(res["values"]).reshape(-1, k)
    if largest:
        # 0x7FFF, 0x7FC0 -> 0x7F80; +Inf-class lanes = exactly 3, all on top.
        assert (vb[:, :3] == BF16_POS_INF).all(), "top-3 must be +Inf (2 canonicalized +NaNs + planted +Inf)"
        assert ((vb == BF16_POS_INF).sum(dim=-1) == 3).all(), "exactly 3 +Inf-class lanes expected"
        assert not (vb[:, 3] == BF16_POS_INF).any(), "lane 3 must be finite"
    else:
        # 0xFFC0 -> 0xFF80; -Inf-class lanes = exactly 2, both at the bottom.
        # torch would never return NaN-derived mass as the smallest ->
        # documented divergence (report 2.1, I3) still holds, post-canon.
        assert (vb[:, :2] == BF16_NEG_INF).all(), "bottom-2 must be -Inf (canonicalized -NaN + planted -Inf)"
        assert ((vb == BF16_NEG_INF).sum(dim=-1) == 2).all(), "exactly 2 -Inf-class lanes expected"
        ledger(
            "T3",
            "documented_divergence:neg_nan",
            {"cell_id": cell_id, "W": W, "k": k},
            res["engine"],
            expected="torch: all NaN above +Inf regardless of sign",
            actual="ttnn bf16: -NaN canonicalized to -Inf and returned as bottom-1 "
            "(SFPSWAP sign-magnitude order on the canonicalized value)",
        )
    ledger(
        "T3",
        "documented_divergence:nan_payload_canonicalized",
        {"cell_id": cell_id, "W": W, "k": k},
        res["engine"],
        expected="NaN payload bits preserved in values output",
        actual="bf16 datapath returns same-sign Inf for every NaN payload",
    )


FP32_NAN_CELLS = [("single_core", 4096, 32, True)] + ([("single_core", 4096, 32, False)] if FULL else [])


@pytest.mark.parametrize("engine,W,k,largest", FP32_NAN_CELLS, ids=_ids(FP32_NAN_CELLS))
def test_contract_nan_fp32(engine, W, k, largest, device):
    specials = torch.cat(
        [
            fp32_from_bits(
                [FP32_POS_NAN_MAXPAY, FP32_POS_NAN_PAYLOAD, FP32_POS_QNAN, FP32_NEG_QNAN, FP32_POS_INF, FP32_NEG_INF]
            ),
            torch.tensor(BF16_NAN_NORMALS, dtype=torch.float32),
        ]
    )
    x = build_input(W, specials, dtype=torch.float32, largest=largest, seed=102)
    cell_id = f"nan-fp32-{engine}-W{W}-k{k}-{largest}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)
    vb = bits_of(res["values"]).reshape(-1, k)
    if largest:
        assert (vb[:, 0] == FP32_POS_NAN_MAXPAY).all()
        assert (vb[:, 1] == FP32_POS_NAN_PAYLOAD).all(), "fp32 NaN payload LSB must be preserved and ordered"
        assert (vb[:, 2] == FP32_POS_QNAN).all()
    else:
        assert (vb[:, 0] == FP32_NEG_QNAN).all()


# ===========================================================================
# zeros — I4: mixed +0/-0 mass straddling the k boundary
# ===========================================================================
@pytest.mark.parametrize("engine,W,k,largest", ENGINE_CELLS, ids=_ids(ENGINE_CELLS))
def test_contract_zeros_bf16(engine, W, k, largest, device):
    _skip_routed_off_bh(engine)
    if k <= 64:
        n_win, n_pz, n_nz = 8, 20, 40  # boundary lands inside the zero mass
    else:  # routed k=96
        n_win, n_pz, n_nz = 40, 40, 60
    winners = torch.tensor([(1.0 + 0.25 * i) * (1 if largest else -1) for i in range(n_win)], dtype=torch.bfloat16)
    zeros = bf16_from_bits([BF16_POS_ZERO] * n_pz + [BF16_NEG_ZERO] * n_nz)
    x = build_input(W, torch.cat([winners, zeros]), largest=largest, seed=103)
    cell_id = f"zeros-{engine}-W{W}-k{k}-{largest}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)

    # bf16 datapath canonicalization: every -0 in the input surfaces as +0, so
    # the +-0 mass is one indistinguishable tie class; the boundary split
    # among zero POSITIONS is implementation-defined, but every zero-valued
    # lane must read +0 and the zero-lane COUNT is pinned by the multiset.
    vb = bits_of(res["values"]).reshape(-1, k)
    n_from_zeros = k - n_win
    assert (
        (vb == BF16_POS_ZERO).sum(dim=-1) == n_from_zeros
    ).all(), f"expected {n_from_zeros} zero lanes per row, all canonicalized to +0"
    assert ((vb == BF16_NEG_ZERO).sum(dim=-1) == 0).all(), "-0 must never appear in bf16 values output"
    ledger(
        "T3",
        "documented_divergence:signed_zero_order",
        {"cell_id": cell_id},
        res["engine"],
        expected="sign-magnitude order: -0 < +0 distinct in values output",
        actual=f"ttnn bf16: -0 canonicalized to +0 pre-sort; {n_from_zeros} zero lanes "
        "all read +0, boundary split among +-0 positions implementation-defined",
    )


ZEROS_FP32_CELLS = [("single_core", 4096, 32, True)]


@pytest.mark.parametrize("engine,W,k,largest", ZEROS_FP32_CELLS, ids=_ids(ZEROS_FP32_CELLS))
def test_contract_zeros_fp32(engine, W, k, largest, device):
    winners = torch.tensor([1.0 + 0.25 * i for i in range(8)], dtype=torch.float32)
    zeros = fp32_from_bits([FP32_POS_ZERO] * 20 + [FP32_NEG_ZERO] * 40)
    x = build_input(W, torch.cat([winners, zeros]), dtype=torch.float32, largest=largest, seed=104)
    cell_id = f"zeros-fp32-{engine}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)
    vb = bits_of(res["values"]).reshape(-1, k)
    assert ((vb == FP32_POS_ZERO).sum(dim=-1) == 20).all()
    assert ((vb == FP32_NEG_ZERO).sum(dim=-1) == 4).all()


# ===========================================================================
# subnormal — I6: the one true unknown (datapath survival), outcome ledgered
# ===========================================================================
BF16_SUBN_SPECIALS = [
    BF16_MIN_NORMAL,  # 0x0080, smallest normal
    0x0001,
    0x0040,
    0x007F,  # + subnormals
    0x8001,
    0x8040,
    0x807F,  # - subnormals
    0x8080,  # -min normal
    BF16_POS_ZERO,
    BF16_NEG_ZERO,
]
SUBN_CELLS = STOCK_CELLS_TF + ROUTED_CELL


@pytest.mark.parametrize("engine,W,k,largest", SUBN_CELLS, ids=_ids(SUBN_CELLS))
def test_contract_subnormal_bf16(engine, W, k, largest, device):
    _skip_routed_off_bh(engine)
    x = build_input(W, bf16_from_bits(BF16_SUBN_SPECIALS), largest=largest, seed=105)
    cell_id = f"subnormal-{engine}-W{W}-k{k}-{largest}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id, subnormal_mode=True)
    _assert_engine(res, engine, cell_id)
    # The classification (exact / flush_keep_sign / flush_to_pos0) is in the
    # ledger; any of the three passes.  A fourth behavior hard-fails inside
    # verify_topk_cell with both key rows printed.


def test_contract_subnormal_fp32_single(device):
    specials = fp32_from_bits(
        [
            FP32_MIN_NORMAL,
            0x00000001,
            0x00400000,
            0x007FFFFF,
            0x80000001,
            0x80400000,
            0x807FFFFF,
            0x80800000,
            FP32_POS_ZERO,
            FP32_NEG_ZERO,
        ]
    )
    x = build_input(4096, specials, dtype=torch.float32, largest=True, seed=106)
    res = verify_topk_cell(device, x, 32, largest=True, cell_id="subnormal-fp32-single", subnormal_mode=True)
    _assert_engine(res, "single_core", "subnormal-fp32-single")


# ===========================================================================
# ties — I7: constructed duplicate mass straddling k; all-equal row
# ===========================================================================
@pytest.mark.parametrize("engine,W,k,largest", ENGINE_CELLS, ids=_ids(ENGINE_CELLS))
def test_contract_ties_boundary(engine, W, k, largest, device):
    _skip_routed_off_bh(engine)
    if k <= 64:
        n_win, n_tie = 24, 16  # 24 strict winners + 16 copies of v; k=32 -> 8 of 16
    else:
        n_win, n_tie = 80, 32  # routed k=96 -> 16 of 32
    sgn = 1.0 if largest else -1.0
    winners = torch.tensor([sgn * (2.0 + 0.25 * i) for i in range(n_win)], dtype=torch.bfloat16)
    tie_val = torch.tensor([sgn * 1.0], dtype=torch.bfloat16)
    ties = tie_val.repeat(n_tie)
    x = build_input(W, torch.cat([winners, ties]), largest=largest, seed=107)
    cell_id = f"ties-{engine}-W{W}-k{k}-{largest}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)

    # Winner-set property: all Cgt strict winners present + exactly k - Cgt
    # boundary-equals (implied by the T1 multiset check; asserted explicitly).
    vb = bits_of(res["values"]).reshape(-1, k)
    tie_bits = int(bits_of(tie_val)[0])
    assert ((vb == tie_bits).sum(dim=-1) == k - n_win).all(), f"expected exactly {k - n_win} boundary-equal lanes"
    win_bits = bits_of(winners)
    for wbit in win_bits.tolist():
        assert ((vb == wbit).sum(dim=-1) == 1).all(), f"strict winner bits {wbit:#x} missing from some row"


ALL_EQUAL_CELLS = [("single_core", 10000, 32, True), ("multi_core", 8192, 32, True)] + (
    [("routed", 8192, 96, True)] if FULL else []
)


@pytest.mark.parametrize("engine,W,k,largest", ALL_EQUAL_CELLS, ids=_ids(ALL_EQUAL_CELLS))
def test_contract_ties_all_equal_row(engine, W, k, largest, device):
    _skip_routed_off_bh(engine)
    x = torch.full((1, 1, 32, W), 3.140625, dtype=torch.bfloat16)  # bf16-exact constant
    cell_id = f"ties-allequal-{engine}-W{W}-k{k}"
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)


# ===========================================================================
# infleak — I8: fewer-than-k finite rows.  Stock: indices may point into the
# tile padding [W, padded_W) (topk.cpp:215-217 acknowledges the leak).
# Routed: 0xFFFFFFFF / 0xFFFF sentinel on -inf lanes (topk.cpp:211-217).
# ===========================================================================
def _finite_specials(n, largest, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = (torch.rand(n, generator=g) * 8 + 1) * (1.0 if largest else -1.0)
    return vals.to(dtype)


INFLEAK_STOCK_CELLS = [
    # (cell_id, W, k, largest, finite_count)
    ("single-W10000-largest", 10000, 32, True, 10),  # 16 pad lanes at [10000,10016)
    ("single-W10000-smallest", 10000, 32, False, 10),  # +inf mirror
    ("single-W63-hostpad", 63, 32, True, 5),  # host-padded to 64 (topk.cpp:503-519)
]


@pytest.mark.parametrize("cell_id,W,k,largest,finite", INFLEAK_STOCK_CELLS, ids=[c[0] for c in INFLEAK_STOCK_CELLS])
def test_contract_infleak_stock(cell_id, W, k, largest, finite, device):
    x = build_input(W, _finite_specials(finite, largest, seed=108), largest=largest, filler="inf_tail", seed=108)
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=f"infleak-{cell_id}")
    if is_blackhole():
        assert res["engine"] == "single_core"
    # Leak occurrences (indices >= W on extremum lanes) are recorded by
    # verify_topk_cell as T3 'padding_index_leak_observed' — never asserted:
    # the engine may legally return in-row -inf positions instead.


INFLEAK_ROUTED_CELLS = [
    # W=8192: padded 8192 <= 65535 -> uint16 indices -> sentinel 0xFFFF
    ("routed-u16", 8192, 96, 50),
    # W=65536: padded > 65535 -> uint32 indices -> sentinel 0xFFFFFFFF
    ("routed-u32", 65536, 96, 50),
]


@pytest.mark.skipif(not is_blackhole(), reason="large-k route is Blackhole-only (topk.cpp:286-288)")
@pytest.mark.parametrize("cell_id,W,k,finite", INFLEAK_ROUTED_CELLS, ids=[c[0] for c in INFLEAK_ROUTED_CELLS])
def test_contract_infleak_routed_sentinel(cell_id, W, k, finite, device):
    x = build_input(W, _finite_specials(finite, True, seed=109), largest=True, filler="inf_tail", seed=109)
    res = verify_topk_cell(device, x, k, largest=True, cell_id=f"infleak-{cell_id}")
    assert res["engine"] == "routed"
    # sentinel <-> -inf equivalence asserted inside verify_topk_cell (T2);
    # additionally require that some sentinel lanes actually occurred:
    idx = res["indices"].reshape(-1, k)
    sentinel = SENTINEL_U16 if expected_index_ttnn_dtype(W, "bf16") == ttnn.uint16 else SENTINEL_U32
    assert (idx == sentinel).any(), "cell was constructed to produce -inf lanes; none observed"


# ===========================================================================
# determinism — I13: same input, 3 launches (program-cache-hit path),
# bit-identical values AND indices (tie choice included)
# ===========================================================================
DET_CELLS = [("single_core", 10000, 32), ("multi_core", 8192, 32), ("routed", 8192, 96)]
DET_RUNS = 5 if FULL else 3


@pytest.mark.parametrize("engine,W,k", DET_CELLS, ids=[f"{e}-W{w}-k{k}" for e, w, k in DET_CELLS])
def test_contract_determinism(engine, W, k, device):
    _skip_routed_off_bh(engine)
    # tie-heavy input: 16 quantized levels -> massive tie population
    g = torch.Generator().manual_seed(110)
    x = (torch.randint(0, 16, (1, 1, 32, W), generator=g).to(torch.float32) * 0.25 + 1.0).to(torch.bfloat16)
    cell_id = f"determinism-{engine}-W{W}-k{k}"
    res = verify_topk_cell(device, x, k, largest=True, cell_id=cell_id)
    _assert_engine(res, engine, cell_id)

    tt_in = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    u16 = expected_index_ttnn_dtype(W, "bf16") == ttnn.uint16
    runs = []
    for _ in range(DET_RUNS):
        v, i = ttnn.topk(tt_in, k, dim=-1, largest=True, sorted=True)
        runs.append(
            (bits_of(ttnn.to_torch(v)), ttnn.to_torch(i, dtype=torch.uint16 if u16 else torch.uint32).to(torch.int64))
        )
    for n, (vb, ib) in enumerate(runs[1:], start=2):
        if not (torch.equal(vb, runs[0][0]) and torch.equal(ib, runs[0][1])):
            ledger(
                "T2",
                "determinism",
                {"cell_id": cell_id, "run": n},
                res["engine"],
                actual={"values_equal": torch.equal(vb, runs[0][0]), "indices_equal": torch.equal(ib, runs[0][1])},
            )
            assert (
                False
            ), f"{cell_id}: run {n} differs from run 1 ('deterministic but unspecified' must at least be deterministic)"


# ===========================================================================
# gates — routing-gate boundary flips (one-sided tests of every predicate
# clause; each cell also runs the full T1/T2 correctness battery)
# ===========================================================================
# (cell_id, W, k, largest, expected_engine_on_blackhole)
GATE_CELLS = [
    # device-op multicore k gate: <= 64 (topk_device_operation.cpp:75); k=65
    # flips to the route when largest=True (topk.cpp:274) and to single-core
    # when largest=False (topk.cpp:257).
    ("k64-multi", 8192, 64, False, "multi_core"),
    ("k65-single-smallest", 8192, 65, False, "single_core"),
    ("k65-routed", 8192, 65, True, "routed"),
    # route k ceiling: 2048 (topk.cpp:238,:274-276)
    ("k2048-routed", 8192, 2048, True, "routed"),
    ("k2049-single", 8192, 2049, True, "single_core"),
    # pow2 gate (topk_device_operation.cpp:72)
    ("pow2-multi", 8192, 32, True, "multi_core"),
    ("nonpow2-single", 8224, 32, True, "single_core"),
    # width/index-dtype boundaries: gate is padded width vs 65535
    # (topk_device_operation.cpp:70,:294; topk.cpp:315-318)
    ("w65504-u16", 65504, 32, True, "single_core"),
    ("w65534-u32-padded", 65534, 32, True, "single_core"),  # pads to 65536 -> u32
    ("w65536-pow2-but-u16-gate", 65536, 32, True, "single_core"),
    # host-pad path (W < 64 padded to 64, topk.cpp:503-519) + k=W degenerate
    ("w63-hostpad", 63, 32, True, "single_core"),
    ("w64-kW", 64, 64, True, "single_core"),
    # k degenerate / alignment (adjusted_k = roundup32(k), topk.cpp:42-44)
    ("k1-multi", 8192, 1, True, "multi_core"),
    ("k31-multi", 8192, 31, True, "multi_core"),
    ("k97-routed-round112", 8192, 97, True, "routed"),
] + (
    [
        # FULL: routed width ceiling 2^19 inclusive (topk.cpp:245,:289-294)
        ("w524288-routed-ceiling", 524288, 96, True, "routed"),
        ("w524320-single-over-ceiling", 524320, 96, True, "single_core"),
        ("w131072-routed", 131072, 512, True, "routed"),
        ("w100000-routed-nonpow2", 100000, 512, True, "routed"),
        ("w32768-multi", 32768, 64, True, "multi_core"),
        ("k33-multi-round64", 8192, 33, True, "multi_core"),
        ("k95-routed-round96", 8192, 95, True, "routed"),
    ]
    if FULL
    else []
)


@pytest.mark.parametrize("cell_id,W,k,largest,exp_engine", GATE_CELLS, ids=[c[0] for c in GATE_CELLS])
def test_contract_gates(cell_id, W, k, largest, exp_engine, device):
    _skip_routed_off_bh(exp_engine)
    x = gaussian_input(W, seed=111)
    res = verify_topk_cell(device, x, k, largest=largest, cell_id=f"gates-{cell_id}")
    if is_blackhole():
        assert res["engine"] == exp_engine, (
            f"gates-{cell_id}: predicted {res['engine']}, matrix expects {exp_engine} "
            f"(routing gates: topk.cpp:247-295, topk_device_operation.cpp:59-115)"
        )


SORTED_NOOP_CELLS = [("single_core", 10000, 32), ("multi_core", 8192, 32), ("routed", 8192, 96)]


@pytest.mark.parametrize("engine,W,k", SORTED_NOOP_CELLS, ids=[f"{e}-W{w}-k{k}" for e, w, k in SORTED_NOOP_CELLS])
def test_contract_gates_sorted_flag_noop(engine, W, k, device):
    """sorted is accepted but ignored — output always fully sorted (report
    2.3: single-core never passes it to the kernel, multi-core reads it into
    an unused constexpr, routed always emits descending).  Pin byte-identical
    outputs for sorted=True vs sorted=False."""
    _skip_routed_off_bh(engine)
    x = gaussian_input(W, seed=112)
    tt_in = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    u16 = expected_index_ttnn_dtype(W, "bf16") == ttnn.uint16
    idt = torch.uint16 if u16 else torch.uint32
    outs = {}
    for s in (True, False):
        v, i = ttnn.topk(tt_in, k, dim=-1, largest=True, sorted=s)
        outs[s] = (bits_of(ttnn.to_torch(v)), ttnn.to_torch(i, dtype=idt).to(torch.int64))
    same = torch.equal(outs[True][0], outs[False][0]) and torch.equal(outs[True][1], outs[False][1])
    if not same:
        ledger(
            "T2",
            "sorted_flag_noop",
            {"engine": engine, "W": W, "k": k},
            engine,
            expected="byte-identical outputs for sorted=True/False",
            actual="outputs differ",
        )
    assert same, f"sorted flag is documented as a no-op (always sorted) but outputs differ on {engine}"


def test_contract_gates_stable_values_only(device):
    """stable=True is EXPERIMENTAL best-effort (topk_nanobind.cpp:49, issue
    #33492: 'can still return incorrect indices for tied values').  Pin the
    VALUES contract; index mismatches are ledgered (T3), not asserted."""
    x = gaussian_input(8192, seed=113)
    res = verify_topk_cell(
        device, x, 32, largest=True, stable=True, cell_id="gates-stable-values-only", index_checks_hard=False
    )
    if is_blackhole():
        assert res["engine"] == "multi_core"  # stable does not change factory selection


@pytest.mark.skipif(not FULL, reason="full-matrix cell; set TOPK_CONTRACT_FULL=1")
def test_contract_gates_transposed_dim(device):
    """Non-last dim rides the host transpose path (topk.cpp:474-475) and can
    never route (is_dim_last_idx gate, topk.cpp:271-273)."""
    g = torch.Generator().manual_seed(114)
    x = (torch.randn((1, 8192, 32, 64), generator=g) * 0.9).to(torch.bfloat16)
    res = verify_topk_cell(device, x, 32, dim=1, largest=True, cell_id="gates-dim1")
    if is_blackhole():
        assert res["engine"] == "multi_core"  # transposed width 8192, k=32
