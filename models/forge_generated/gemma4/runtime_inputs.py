"""Runtime input synthesizers for gemma4 prefill / decode.

Phase 4 source-of-truth for the per-call runtime inputs (KV caches,
position-id arrays, scratch buffers). Produces the same {slot: ttnn.Tensor}
map that gemma4_{prefill,decode}/main.py:_load_runtime_inputs returns
from the codegen tensorbins, but built from constants — no fixture files.

Inventory (Phase 4 Task 1):
    PREFILL: 183 slots — 121 zeros (TILE bf16, KV caches), 60 zeros
             (ROW_MAJOR int32, position/idx scalars), 1 ones (slot 26,
             "position helper" int32[1,256]), 1 prompt-token-ids
             (slot 7, int32[1,19]), 1 bf16 scalar 1.0 (slot 9, shape []).

    DECODE:  183 slots — 121 zeros (TILE bf16, KV caches), 61 zeros
             (ROW_MAJOR int32), 1 ones (slot 26), 1 bf16 scalar 1.0
             (slot 9, shape []). No prompt token IDs (decode step 0
             continues from initial KV cache).

All replicated across the (1,4) mesh — verified per-shard via
`torch.equal` on the loaded tensorbins.

The reference logits at gemma4/reference_logits/{prefill,decode}.pt
match this initial state exactly (PCC=1.0).
"""
import torch

import ttnn

# -----------------------------------------------------------------------------
# Special non-zero/non-one inputs (handled out-of-band by the synthesizers).
# -----------------------------------------------------------------------------

# Prefill prompt token IDs (slot 7), captured from the codegen tensorbin.
# The reference logits at gemma4/reference_logits/prefill.pt are what the
# model produces for THIS prompt — change these and PCC will not be 1.0.
_PREFILL_TOKEN_IDS = [
    2,
    105,
    2364,
    107,
    3689,
    563,
    822,
    8126,
    3207,
    236881,
    106,
    107,
    105,
    4368,
    107,
    100,
    45518,
    107,
    101,
]

# bf16 scalar 1.0 for slot 9 in both sides (0-d tensor, captured from
# arg9.tensorbin). Likely an attention-scale factor at decode step 0.
_LIFTED_SCALAR_VALUE = 1.0


# -----------------------------------------------------------------------------
# Per-side runtime slot tables.
# Format: (slot, shape, dtype_name, layout_name, fill).
# Slots with custom semantics (token IDs, 0-d scalar) are handled in the
# synthesizer function and excluded from this list.
# -----------------------------------------------------------------------------

# Slots that need the same recipe across both sides.
_COMMON_ZEROS_TILE_BF16_4x256x256 = [
    1,
    10,
    13,
    30,
    33,
    47,
    50,
    64,
    67,
    81,
    101,
    115,
    118,
    132,
    135,
    149,
    152,
    166,
    169,
    183,
    202,
    216,
    219,
    233,
    236,
    250,
    253,
    267,
    270,
    284,
    303,
    317,
    320,
    334,
    337,
    351,
    354,
    368,
    371,
    385,
    404,
    418,
    421,
    435,
    438,
    452,
    455,
    469,
    472,
    486,
    505,
    519,
    522,
    536,
    539,
    553,
    556,
    570,
    573,
    587,
    606,
    620,
    623,
    637,
    640,
    654,
    657,
    671,
    674,
    688,
    707,
    721,
    724,
    738,
    741,
    755,
    758,
    772,
    775,
    789,
    808,
    822,
    825,
    839,
    842,
    856,
    859,
    873,
    876,
    890,
    909,
    923,
    926,
    940,
    943,
    957,
    960,
    974,
    977,
    991,
]
_COMMON_ZEROS_TILE_BF16_1x256x512 = [
    98,
    99,
    199,
    200,
    300,
    301,
    401,
    402,
    502,
    503,
    603,
    604,
    704,
    705,
    805,
    806,
    906,
    907,
    1007,
    1008,
]
_COMMON_ZEROS_RM_INT32_1 = [
    0,
    12,
    32,
    49,
    66,
    83,
    100,
    117,
    134,
    151,
    168,
    185,
    201,
    218,
    235,
    252,
    269,
    286,
    302,
    319,
    336,
    353,
    370,
    387,
    403,
    420,
    437,
    454,
    471,
    488,
    504,
    521,
    538,
    555,
    572,
    589,
    605,
    622,
    639,
    656,
    673,
    690,
    706,
    723,
    740,
    757,
    774,
    791,
    807,
    824,
    841,
    858,
    875,
    892,
    908,
    925,
    942,
    959,
    976,
    993,
]


def _expand_common_recipes():
    rows = []
    for s in _COMMON_ZEROS_TILE_BF16_4x256x256:
        rows.append((s, (1, 4, 256, 256), "BFLOAT16", "TILE", 0.0))
    for s in _COMMON_ZEROS_TILE_BF16_1x256x512:
        rows.append((s, (1, 1, 256, 512), "BFLOAT16", "TILE", 0.0))
    for s in _COMMON_ZEROS_RM_INT32_1:
        rows.append((s, (1,), "INT32", "ROW_MAJOR", 0))
    rows.append((26, (1, 256), "INT32", "ROW_MAJOR", 1))
    return rows


_RUNTIME_SLOTS_PREFILL = _expand_common_recipes() + [
    # slot 7 = prompt token IDs, shape [1, 19] — special-cased in synthesizer.
    # slot 9 = bf16 scalar 1.0, shape [] — special-cased in synthesizer.
]
_RUNTIME_SLOTS_DECODE = _expand_common_recipes() + [
    (7, (1, 1), "INT32", "ROW_MAJOR", 0),
    # slot 9 = bf16 scalar 1.0, shape [] — special-cased in synthesizer.
]


_DT_MAP = {
    "INT32": (ttnn.DataType.INT32, torch.int32),
    "UINT32": (ttnn.DataType.UINT32, torch.int32),
    "BFLOAT16": (ttnn.DataType.BFLOAT16, torch.bfloat16),
    "FLOAT32": (ttnn.DataType.FLOAT32, torch.float32),
}
_LAYOUT_MAP = {
    "TILE": ttnn.Layout.TILE,
    "ROW_MAJOR": ttnn.Layout.ROW_MAJOR,
}


def _build_slot(shape, dtype_name, layout_name, fill, mesh_device):
    """Build one ttnn.Tensor for a runtime slot. All slots are replicated
    across the (1,4) mesh. The fill argument is one of:
      0/1/float — torch.full(shape, fill, dtype)
      ("token_ids", list_of_ints)        — torch.tensor(list).reshape(shape)
    """
    ttnn_dt, torch_dt = _DT_MAP[dtype_name]
    if isinstance(fill, tuple) and fill[0] == "token_ids":
        t = torch.tensor(fill[1], dtype=torch_dt).reshape(*shape)
    else:
        t = torch.full(list(shape), float(fill), dtype=torch_dt)
    return ttnn.as_tensor(
        t,
        dtype=ttnn_dt,
        layout=_LAYOUT_MAP[layout_name],
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_lifted_scalar(value, mesh_device):
    """Build the slot 9 0-d bf16 scalar (shape Shape([])) replicated."""
    t = torch.tensor(value, dtype=torch.bfloat16)
    return ttnn.as_tensor(
        t,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def synthesize_prefill_inputs(mesh_device, *, seq_len=19):
    """Return the runtime-inputs dict {slot: ttnn.Tensor} for prefill.

    `seq_len` defaults to 19 (the codegen-baked length); at this value
    the token-ID fill is the canonical sequence and PCC matches the
    reference logits. For other seq_lens the caller is expected to
    overwrite `out[7]` with their own token IDs (the synthesizer just
    sizes the slot correctly).
    """
    out = {}
    for slot, shape, dt, layout, fill in _RUNTIME_SLOTS_PREFILL:
        out[slot] = _build_slot(shape, dt, layout, fill, mesh_device)
    if seq_len == 19:
        token_ids_fill = ("token_ids", _PREFILL_TOKEN_IDS)
    else:
        # Caller will likely overwrite out[7]; zero-fill keeps shape correct.
        token_ids_fill = 0
    out[7] = _build_slot(
        (1, seq_len),
        "INT32",
        "ROW_MAJOR",
        token_ids_fill,
        mesh_device,
    )
    out[9] = _build_lifted_scalar(_LIFTED_SCALAR_VALUE, mesh_device)
    return out


def synthesize_decode_inputs(mesh_device):
    """Return the runtime-inputs dict {slot: ttnn.Tensor} for decode."""
    out = {}
    for slot, shape, dt, layout, fill in _RUNTIME_SLOTS_DECODE:
        out[slot] = _build_slot(shape, dt, layout, fill, mesh_device)
    out[9] = _build_lifted_scalar(_LIFTED_SCALAR_VALUE, mesh_device)
    return out
