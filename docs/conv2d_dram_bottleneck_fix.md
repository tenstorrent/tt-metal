# Conv2d DRAM Bottleneck — Spatial Packing Fix

**Issue:** https://github.com/tenstorrent/tt-metal/issues/46831
**Hardware:** Wormhole N150 · 64 Tensix cores · 288 GB/s peak DRAM BW
**Test file:** `tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py`

---

## Problem: PermuteDeviceOperation + MatmulDeviceOperation Bottleneck

The TTNN IR chain for a 1×1 pointwise conv2d with small input channels (C=3) produces
a severe DRAM bandwidth bottleneck. Both the `PermuteDeviceOperation` and the
`MatmulDeviceOperation` (inside `ttnn.conv2d`) are dominated by tile-padding waste.

### Root Cause

When `C=3` is placed in the X-dimension of a NHWC TILE tensor, every 32-element
tile row holds only 3 valid values and 29 zeros — a **10.7× DRAM read inflation**:

```
TILE row (32 cols × 2 B = 64 B):  [v  v  v  0  0  0 … 0]
Real data per pixel:                3 × 2 B = 6 B
Inflation factor:                   64 / 6  = 10.7×
```

| Op | Kernel | FPU % | Bytes read | Inflation |
|----|--------|-------|-----------|-----------|
| `PermuteDeviceOperation` (NCHW→NHWC) | 6.9–8.5 ms | 8–10% | 188.7 MB | 10.7× |
| `MatmulDeviceOperation` (conv2d) | 6.4–7.9 ms | **0.008%** | 188.7 MB | 10.7× |
| **Total pipeline** | **14.7–17.0 ms** | | | |

Both ops account for **97% of pipeline time**. The matmul runs at 277 GB/s —
96% of peak DRAM bandwidth — with virtually no compute.

---

## Test Case 1 — Baseline (Bottleneck)

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import math
import pytest
import torch
import torch.nn.functional as F
import ttnn

_CONFIGS = [
    pytest.param(1, 3, 3, 1536, 1536, id="conv2d_1_1x3x1536x1536"),
    pytest.param(1, 3, 3, 1280, 2304, id="conv2d_2_1x3x1280x2304"),
]


def _make_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )


def _make_conv_config():
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_dram_bottleneck(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    compute_config = _make_compute_config(device)
    conv_config = _make_conv_config()
    spatial = input_height * input_width
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

    tt_weight = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=dram_interleaved,
        input_layout=ttnn.TILE_LAYOUT,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_bias = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=dram_interleaved,
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=dram_interleaved
    )
    tt_nchw_tile = ttnn.to_layout(tt_input, layout=ttnn.TILE_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_input)
    tt_nhwc = ttnn.permute(tt_nchw_tile, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw_tile)
    tt_flat = ttnn.reshape(tt_nhwc, shape=(batch, 1, spatial, in_channels))
    ttnn.deallocate(tt_nhwc)

    [tt_out, [out_h, out_w], [d_w, d_b]] = ttnn.conv2d(
        input_tensor=tt_flat,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.deallocate(tt_flat)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    tt_out = ttnn.reshape(tt_out, shape=(batch, out_h, out_w, out_channels))
    tt_nchw_out = ttnn.permute(tt_out, dims=(0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_out)
    tt_output = ttnn.to_memory_config(tt_nchw_out, memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw_out)

    result = ttnn.to_torch(tt_output)
    assert result.shape == torch.Size((batch, out_channels, input_height, input_width)), (
        f"Shape mismatch: got {tuple(result.shape)}, "
        f"expected ({batch}, {out_channels}, {input_height}, {input_width})"
    )
    ttnn.deallocate(tt_output)
```

### Profiler Output (Baseline)

| Config | Op | Kernel | FPU % | Bytes |
|--------|-----|--------|-------|-------|
| 1×3×1536×1536 | `TilizeDeviceOperation` | 0.320 ms | 0.0% | — |
| | `PermuteDeviceOperation` (×2) | **7.835 ms** | 8.4% | 188.7 MB |
| | `MatmulDeviceOperation` | **6.429 ms** | **0.008%** | **277 GB/s** |
| | `CopyDeviceOperation` | 0.113 ms | 0.001% | — |
| | **Total** | **14.697 ms** | | |
| 1×3×1280×2304 | **Total** | **16.764 ms** | | |

---

## Solution: Spatial Row-Group Packing

Group **K=32 adjacent rows** into the channel dimension so `C×K=96` channels fill
tile rows completely, eliminating all padding waste.

### Key Idea

```
Before: [H×W,  3] TILE — each row: [v v v  0 0 … 0]   9.4% useful
After:  [H×W/32, 96] TILE — each row: 3 full tile cols  100% useful

Activation DRAM reads: 188.7 MB → 17.7 MB  (10.7× less)
```

### Packing Factor

```python
K = TILE_WIDTH // math.gcd(in_channels, TILE_WIDTH)
# For C=3: K = 32 // gcd(3,32) = 32,  C*K = 96 = 3×TILE_WIDTH (zero waste)
```

### Block-Diagonal Weight

The 1×1 conv is spatially independent — each pixel's output depends only on its
own channels. K row-groups are processed simultaneously via a block-diagonal weight:

```
W_packed[c*K+k, oc*K+k] = W[oc, c]   for all k=0..K-1
W_packed[c*K+j, oc*K+k] = 0           for j ≠ k
```

This preserves correctness: output `[oc, k*(H/K)+h', w] = Σ_c W[oc,c] × input[c, k*(H/K)+h', w]`

### Critical Deallocation Rule

`ttnn.reshape` on ROW_MAJOR returns a **view** sharing the source buffer. Violating
this ordering corrupts data (PCC ≈ 0.5):

```python
# WRONG — frees source while view is still alive
view   = ttnn.reshape(source, new_shape)
ttnn.deallocate(source)         # ← source freed, view points to garbage!
result = ttnn.permute(view, ...) # reads corrupted data → PCC ≈ 0.5

# CORRECT — consume view before freeing source
view   = ttnn.reshape(source, new_shape)
result = ttnn.permute(view, ...) # ← consume view first
ttnn.deallocate(source)          # ← now safe
ttnn.deallocate(view)
```

---

## Test Case 2 — Fixed (Spatial Packing)

```python
TILE_WIDTH = 32


def _spatial_pack_factor(in_channels: int) -> int:
    return TILE_WIDTH // math.gcd(in_channels, TILE_WIDTH)


def _make_packed_weight(torch_weight: torch.Tensor,
                        in_channels: int, out_channels: int, K: int) -> torch.Tensor:
    import numpy as np
    W_orig = torch_weight.reshape(out_channels, in_channels).float().numpy()
    W_block = np.zeros((in_channels * K, out_channels * K), dtype=np.float32)
    for k in range(K):
        W_block[k::K, k::K] = W_orig.T
    return torch.from_numpy(W_block).to(torch.bfloat16)


def _make_packed_bias(torch_bias: torch.Tensor, out_channels: int, K: int) -> torch.Tensor:
    return torch_bias.reshape(out_channels).repeat_interleave(K)


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_method2_approach2_dram_bottleneck(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG
    K = _spatial_pack_factor(in_channels)
    assert input_height % K == 0, f"input_height={input_height} must be divisible by K={K}"

    packed_ic = in_channels * K
    packed_oc = out_channels * K
    packed_h = input_height // K
    packed_spatial = packed_h * input_width

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

    torch_w_packed = _make_packed_weight(torch_weight, in_channels, out_channels, K)
    torch_b_packed = _make_packed_bias(torch_bias, out_channels, K)

    tt_weight = ttnn.from_torch(
        torch_w_packed.reshape(1, 1, packed_ic, packed_oc),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram_interleaved,
    )
    tt_bias = ttnn.from_torch(
        torch_b_packed.reshape(1, 1, 1, packed_oc),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram_interleaved,
    )

    # --- Input packing (all on device) ---
    # ttnn.reshape on ROW_MAJOR returns a VIEW — never free source before view
    # is consumed by the next op (permute / to_layout).

    tt_nchw = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram_interleaved)

    tt_packed_nchw = ttnn.reshape(tt_nchw, (batch, packed_ic, packed_h, input_width))
    tt_nhwc        = ttnn.permute(tt_packed_nchw, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw)           # safe: permute consumed tt_packed_nchw (view)
    ttnn.deallocate(tt_packed_nchw)

    tt_flat = ttnn.reshape(tt_nhwc, (batch, 1, packed_spatial, packed_ic))
    tt_tile = ttnn.to_layout(tt_flat, layout=ttnn.TILE_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_nhwc)           # safe: to_layout consumed tt_flat (view)
    ttnn.deallocate(tt_flat)

    # --- Single matmul — reads only 17.7 MB vs 188.7 MB baseline ---
    tt_out_packed = ttnn.linear(tt_tile, tt_weight, bias=tt_bias, memory_config=dram_interleaved)
    ttnn.deallocate(tt_tile)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    # --- Output unpacking (all on device) ---
    tt_out_rm = ttnn.to_layout(tt_out_packed, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_packed)

    tt_out_nhwc     = ttnn.reshape(tt_out_rm, (batch, packed_h, input_width, packed_oc))
    tt_out_nchw_pk  = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_rm)         # safe: permute consumed tt_out_nhwc (view)
    ttnn.deallocate(tt_out_nhwc)

    tt_output = ttnn.reshape(tt_out_nchw_pk, (batch, out_channels, input_height, input_width))
    result    = ttnn.to_torch(tt_output)
    ttnn.deallocate(tt_out_nchw_pk)    # safe: to_torch consumed tt_output (view)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size((batch, out_channels, input_height, input_width)), (
        f"Shape mismatch: got {tuple(result.shape)}, "
        f"expected ({batch}, {out_channels}, {input_height}, {input_width})"
    )

    # Golden comparison — verify numerical correctness
    golden = F.conv2d(
        torch_input.float(),
        torch_weight.float().reshape(out_channels, in_channels, 1, 1),
        bias=torch_bias.float().reshape(out_channels),
        stride=1, padding=0,
    )
    result_flat = result.float().flatten()
    golden_flat = golden.float().flatten()
    pcc = torch.corrcoef(torch.stack([result_flat, golden_flat]))[0, 1].item()
    assert pcc >= 0.99, (
        f"PCC {pcc:.6f} is below threshold 0.99 for config "
        f"({batch}, {in_channels}, {out_channels}, {input_height}, {input_width})"
    )
```

### Profiler Output (Fixed)

| Config | Op | Kernel | FPU % | vs Baseline |
|--------|-----|--------|-------|-------------|
| 1×3×1536×1536 | `PermuteDeviceOperation` (×2) | 0.992 ms | 12.1% | **7.9× faster** |
| | `MatmulDeviceOperation` | 0.435 ms | **2.38%** | **14.8× faster** |
| | `TilizeDeviceOperation` | 0.190 ms | 0.001% | — |
| | `UntilizeDeviceOperation` | 0.176 ms | 38.0% | — |
| | **Total** | **1.793 ms** | | **8.20×** |
| 1×3×1280×2304 | **Total** | **1.950 ms** | | **8.60×** |

---

## Performance Summary

| Config | Baseline | Fixed | Speedup |
|--------|----------|-------|---------|
| 1×3×1536×1536 (Block A) | 14.697 ms | **1.793 ms** | **8.20×** |
| 1×3×1280×2304 (Block C) | 16.764 ms | **1.950 ms** | **8.60×** |

---

## Applicability Conditions

Apply this fix when **all** of the following hold:

| Condition | Value |
|-----------|-------|
| Kernel size | 1×1 |
| Stride | 1×1 |
| Padding | 0,0,0,0 |
| Dilation | 1×1 |
| Groups | 1 |
| `in_channels` | < TILE_WIDTH (32) |
| `input_height` | divisible by K = `TILE_WIDTH // gcd(C, TILE_WIDTH)` |

---

## Pattern-Matching Rewrite for tt-forge-onnx / tt-mlir

### Before (compiler-generated IR)

```
%tile    = ttnn.to_layout(%rm_input, TILE_LAYOUT)
%nhwc    = ttnn.permute(%tile, {0,2,3,1})            # [N,C,H,W]→[N,H,W,C] TILE  188.7MB
%flat    = ttnn.reshape(%nhwc, {N,1,H*W,C})
%out     = ttnn.conv2d(%flat, %weight, %bias,
                       kernel=1×1, stride=1, pad=0)  # MatmulDeviceOp  188.7MB read
```

### After (spatial-packing IR)

```python
K       = TILE_WIDTH // gcd(C, TILE_WIDTH)           # compile-time constant = 32
# weight: [OC,IC,1,1] → block-diagonal [1,1,C*K,OC*K]  (compile-time transform)
# bias:   [OC] → repeat_interleave(K) → [OC*K]         (compile-time transform)

%packed = ttnn.reshape(%rm_input, {N, C*K, H/K, W})  # free ROW_MAJOR view
%nhwc   = ttnn.permute(%packed, {0,2,3,1})            # 17.7MB (10.7× less)
# deallocate %rm_input and %packed AFTER permute (view safety)
%flat   = ttnn.reshape(%nhwc, {N, 1, H/K*W, C*K})    # free view
%tile   = ttnn.to_layout(%flat, TILE_LAYOUT)           # 17.7MB, C*K=96=3×TILE_WIDTH
# deallocate %nhwc and %flat AFTER to_layout (view safety)
%out_p  = ttnn.linear(%tile, %w_packed, %b_packed)    # 17.7MB read (14.8× faster)
%out_rm = ttnn.to_layout(%out_p, ROW_MAJOR_LAYOUT)    # 17.7MB
%out_hw = ttnn.reshape(%out_rm, {N, H/K, W, OC*K})   # free view
%out_nc = ttnn.permute(%out_hw, {0,3,1,2})            # 14.2MB
# deallocate %out_rm and %out_hw AFTER permute (view safety)
%out    = ttnn.reshape(%out_nc, {N, OC, H, W})        # free view
```

**Compile-time constants** (zero runtime overhead):

| Symbol | Formula | C=3 example |
|--------|---------|------------|
| K | `TILE_WIDTH // gcd(C, TILE_WIDTH)` | 32 |
| packed_ic | `C × K` | 96 |
| packed_oc | `OC × K` | 96 |
| packed_h | `H // K` | 48 |
| Weight shape | `[1, 1, C*K, OC*K]` | `[1,1,96,96]` |
| Bias shape | `[1, 1, 1, OC*K]` | `[1,1,1,96]` |
