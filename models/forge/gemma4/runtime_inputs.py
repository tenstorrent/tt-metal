"""Runtime input synthesizers for gemma4 prefill / decode.

After Phases 2-3 (KV caches lifted onto Gemma4Caches; position scalars
built from `current_pos` inside `Gemma4ForCausalLM.__call__`), the
synthesized input dict contains only the slots that are *not*
managed by the model itself: the (1,256) ones helper at slot 26, the
prompt token IDs at slot 7, and the bf16 scalar 1.0 at slot 9. All
other slots are populated by the model's __call__ from runtime kwargs.

Slots are replicated across the (1,4) mesh.

The reference logits at `gemma4/reference_logits/{prefill,decode}.pt`
match this initial state exactly (PCC=1.0 for decode, ~0.999 prefill).
"""
import torch

import ttnn

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

# bf16 scalar 1.0 for slot 9 in both sides. Likely an attention-scale
# factor at decode step 0.
_LIFTED_SCALAR_VALUE = 1.0


def _expand_common_recipes():
    """Per-side common runtime slots — currently just slot 26."""
    return [(26, (1, 256), "INT32", "ROW_MAJOR", 1)]


_RUNTIME_SLOTS_PREFILL = _expand_common_recipes()
_RUNTIME_SLOTS_DECODE = _expand_common_recipes() + [
    (7, (1, 1), "INT32", "ROW_MAJOR", 0),
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
      ("token_ids", list_of_ints) — torch.tensor(list).reshape(shape)
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


def synthesize_prefill_inputs(mesh_device, *, seq_len=128):
    """Return the runtime-inputs dict {slot: ttnn.Tensor} for prefill.

    `seq_len` defaults to 128 — matches `Gemma4ForCausalLM.from_state_dict`'s
    default. Pass `seq_len=19` to get the codegen-baked canonical token
    sequence at slot 7 (the only seq_len at which the bundled token-IDs
    fill produces the reference PCC). For other seq_lens the caller
    overwrites `out[7]` with their own token IDs; the synthesizer just
    sizes the slot correctly.
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
