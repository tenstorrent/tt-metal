# Move LTX Per-Head Gate from Host to Device (HiFi4 fp32)

How to promote the LTX-2 per-head gate head from the current host `F.linear` path
to an on-device `Linear` with the highest-precision HiFi4 compute config, without
losing the quality that the host path currently buys.

Touches a single file: `attention_ltx.py`. No checkpoint changes; existing
`TT_DIT_CACHE_DIR` caches must be **regenerated** because the gate parameter
shape, dtype, layout, and location all change.

## Why this works

`models/tt_dit/layers/linear.py` maps `dtype=ttnn.float32` to the highest-precision
compute config available:

```37:51:models/tt_dit/layers/linear.py
MATH_FIDELITY = {
    ttnn.bfloat16: ttnn.MathFidelity.HiFi2,
    ttnn.float32: ttnn.MathFidelity.HiFi4,
}

...

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY[dtype],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
```

Constructing the gate as `Linear(..., dtype=ttnn.float32)` therefore gives you:

- `math_fidelity=HiFi4` (full bf16-input precision via 4-pass multiply)
- `math_approx_mode=False` (no LUT-approximated nonlinearities downstream)
- `fp32_dest_acc_en=True` (fp32 destination register accumulation)
- `packer_l1_acc=True` (fp32 partial sums in L1)
- fp32 weight and bias storage on device (no bf16 quantization of near-zero gate
  weights, which is what compounded into PCC 0.92 over 48 layers × N steps)

The host fp32 `F.linear` baseline is PCC 0.997+; HiFi4 + fp32 dest acc + fp32
weights matches it without the per-step gather→host→push round-trip.

## Step 1 — Imports

In `attention_ltx.py`, add `Linear` to the linear imports and drop `Parameter`
(no longer needed once the gate is a child `Linear`):

```python
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module
```

If you'd rather not churn the `Parameter` import (e.g. local changes still use
it), leave it in place — the fix only requires adding `Linear`.

## Step 2 — Replace the host `Parameter` block in `__init__`

Find the current host-gate block:

```127:160:models/tt_dit/models/transformers/ltx/attention_ltx.py
        # Per-head gate weights kept on host for exact fp32 gate computation.
        # Gate logits are small (32 outputs), so host F.linear is fast and avoids
        # bf16 matmul precision issues on the K=4096 reduction that compound over
        # 48 layers × 40 denoising steps.
        ...
        self._gate_weight_host = None  # cached torch (num_heads, query_input_dim)
        self._gate_bias_host = None  # cached torch (num_heads,)
        if apply_gated_attention:
            # Replicated across the full mesh; host-side; fp32 for precision.
            self.gate_weight = Parameter(
                total_shape=[self.num_heads, self.query_input_dim],
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.float32,
                on_host=True,
            )
            self.gate_bias = Parameter(
                total_shape=[self.num_heads],
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.float32,
                on_host=True,
            )
```

Replace with a single `Linear` child module:

```python
# Per-head gate head on device with HiFi4 fp32 compute. The K=in_features
# reduction with near-zero gate weights is sensitive to bf16 rounding and
# compounds over 48 layers × N denoise steps (latent PCC fell to 0.92 with a
# bf16 device matmul, vs 0.997 with host fp32 F.linear). Storing the weight
# as ttnn.float32 routes the matmul through HiFi4 + fp32 dest accumulation
# + packer_l1_acc (see MATH_FIDELITY in layers/linear.py).
self.apply_gated_attention = apply_gated_attention
if apply_gated_attention:
    self.to_gate_logits = Linear(
        in_features=self.query_input_dim,
        out_features=self.num_heads,
        bias=True,
        dtype=ttnn.float32,
        mesh_device=mesh_device,
    )
```

Drop the `self._gate_weight_host` / `self._gate_bias_host` caches — they're
unused once the gate is on device.

## Step 3 — Simplify `_prepare_torch_state`

Remove the manual `pop_substate("to_gate_logits") → state["gate_weight"]/["gate_bias"]`
mapping. The child `Linear` consumes `to_gate_logits.*` directly via the standard
`Module._load_torch_state_dict_inner` descent, and `Linear._prepare_torch_state`
handles the `[num_heads, in_features] → [in_features, num_heads]` transpose plus
the bias `[num_heads] → [1, num_heads]` reshape.

Defensively zero-fill `to_gate_logits.bias` if the checkpoint omits it (the
on-device `Linear` always has a bias term):

```python
# Replace the existing `gate_state = pop_substate(state, "to_gate_logits")` block with:
if (
    self.apply_gated_attention
    and "to_gate_logits.weight" in state
    and "to_gate_logits.bias" not in state
):
    state["to_gate_logits.bias"] = torch.zeros(self.num_heads)
```

## Step 4 — Rewrite `_compute_gate`

Replace the entire body of `_compute_gate` (`attention_ltx.py:300-366` in the
current file) with the on-device version:

```python
def _compute_gate(self, spatial_1BND: ttnn.Tensor) -> ttnn.Tensor | None:
    """Compute per-head gate on device with HiFi4 fp32 compute.

    spatial_1BND is TP-gathered on dim 3 by `forward()`, and may still be
    SP-sharded on N (dim 2). Each device runs the small matmul over its
    local N shard; we then mesh_partition heads across TP so the result
    matches the BHNE layout of the SDPA output.

    Returns a bf16 tensor with shape (B, H_local, N_local, 1) for BHNE
    broadcast multiply, or None if gating is disabled.
    """
    if not self.apply_gated_attention or self.to_gate_logits.weight._data is None:
        return None

    gate_logits = self.to_gate_logits(spatial_1BND, dtype=ttnn.float32)
    gate = ttnn.multiply(ttnn.sigmoid(gate_logits), 2.0)

    # (1, B, N, H) → (B, H, N, 1) via squeeze + transpose + unsqueeze.
    # `transpose(-2, -1)` is the well-supported last-two-dim swap on 3-D tiles;
    # avoids `permute(1, 3, 2, 0)` which would move the size-1 dim into
    # trailing position and force a ROW_MAJOR round-trip inside TTNN.
    gate = ttnn.squeeze(gate, 0)
    gate = ttnn.transpose(gate, -2, -1)
    gate = ttnn.unsqueeze(gate, -1)
    gate = ttnn.typecast(gate, ttnn.bfloat16)

    if self.parallel_config.tensor_parallel.factor > 1:
        gate = ttnn.mesh_partition(
            gate, dim=1, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
        )
    return gate
```

Key invariants:

- **TP gather happens upstream.** `forward()` already TP-gathers `spatial_1BND`
  on dim 3 before calling `_compute_gate`, so the matmul sees the full K. With
  the host gate in place, your local `forward()` may also be doing this gather
  conditionally (`needs_explicit_ag = ... or apply_gated_attention`). Once the
  gate is on device, the matmul still needs the full K, so keep that branch as
  long as the TP-gather isn't already fused into the QKV matmul; the matmul
  just no longer round-trips to host.
- **SP shard is preserved.** Each SP device computes the gate for its own N
  shard; no SP all-gather is required (the previous host code did the gather
  only to feed `torch.nn.functional.linear`).
- **Mesh-partition on heads (dim 1) after BHNE reshape.** This drops each TP
  device down to its `H_local` slice so the BHNE multiply at the call site is a
  pure local elementwise op.

## Step 5 — Forward call site (`forward()`)

No change required. The call site is:

```python
gate_bhne = self._compute_gate(spatial_1BND)
...
if gate_bhne is not None:
    spatial_BHNE = ttnn.multiply(spatial_BHNE, gate_bhne)
```

Both branches still work with the new device-side `gate_bhne` because the shape
and dtype contract are identical (`(B, H_local, N_local, 1)` bf16).

If you want a slight perf win, you can drop the `apply_gated_attention`-only
explicit TP all-gather guard added for the host gate path — the on-device gate
matmul does not require it beyond what QKV already requires. Look for:

```python
needs_explicit_ag = use_nonfused_agmm or (
    self.apply_gated_attention and self.parallel_config.tensor_parallel.factor > 1
)
```

and remove the `apply_gated_attention` term.

## Step 6 — Caches

The gate parameter shape, dtype, layout, and location all change. Existing
`TT_DIT_CACHE_DIR` entries for any `*/attn*/gate_weight.tensorbin` and
`*/attn*/gate_bias.tensorbin` will fail to match the new `*/attn*/to_gate_logits/weight.tensorbin`
and `*/attn*/to_gate_logits/bias.tensorbin` paths. Easiest: delete the cached
transformer directory and let the next run rebuild it.

```bash
rm -rf "$TT_DIT_CACHE_DIR"/ltx*/transformer
```

## Step 7 — Validation

1. **Per-layer PCC**: run `models/tt_dit/tests/models/ltx/test_av_per_layer_pcc.py`
   with `apply_gated_attention=True`. The test already asserts gated TT vs gated
   reference; the on-device gate should keep the PCC ≥ 0.998 baseline reported
   in `BRINGUP.md`. A drop to ~0.99 indicates the HiFi4 compute config wasn't
   actually applied (most often: `dtype=ttnn.float32` got changed to bf16).
2. **End-to-end latent PCC**: 40-step run of `ltx_av_fast_*` should match the
   host-gate baseline within ~1e-3 (host: 0.997 video, 0.998 audio per
   `BRINGUP.md`). If you see audible doubling / harmony, the SP shard handling
   regressed — confirm the `mesh_partition` is on dim 1 (heads), not dim 2
   (sequence).
3. **Multi-stage pipelines (LTX-Fast)**: confirm the gate survives the
   destroy/rebuild path. Loading from disk should populate the on-device
   `Linear`; no lazy host materialization is needed anymore.

## What to update in surrounding docs

- `HOST_OPS.md`: remove the "Per-Head Gate" section, or mark it as resolved with
  a pointer to this file.
- `BRINGUP.md`: update the workaround row "Host-side per-head gate (fp32
  logits) | Exact gate logits | On-device gate Linear in bf16" — the on-device
  Linear with fp32/HiFi4 is no longer a rejected alternative.

## Rollback

If quality regresses unexpectedly, the host path is purely additive — restore
the four blocks above (imports, `__init__`, `_prepare_torch_state`,
`_compute_gate`) from `git`. No checkpoint or pipeline plumbing needs to change.
