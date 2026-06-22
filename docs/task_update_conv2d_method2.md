Hi @Nikola Vukobrat
Task update from my side,

BEV YUV Conv2d DRAM Bottleneck — Method 2 (Spatial Packing)

The issue comment (https://github.com/tenstorrent/tt-metal/issues/46831#issuecomment-4691303095) pointed to kernel stride folding as a reference technique. Reviewed the existing implementation and then designed an analogous scheme that targets the tile-padding waste for small in_channels.

Existing Kernel Stride Folding (already in tt-metal)

Kernel stride folding is built into the conv2d path for strided convolutions (stride > 1). The idea is to fold the stride step into the channel dimension so the conv can run at stride=1, which is cheaper to schedule and avoids redundant halo/padding ops.

  [N, H, W, C]  NHWC  stride=S
       |
       v  ttnn.fold(stride_h=S, stride_w=S)
          [N, H/S, W/S, C×S×S]     -- S² spatial neighbours folded into channels
       |
       v  conv2d  kernel=ceil(K/S)  stride=1  padding=0
          [N, H/S, W/S, OC]

Result: a stride-S conv on a large spatial map becomes a stride-1 conv on a S²× smaller map. Spatial dimensions shrink by S, in_channels inflate by S². No tile-padding waste is introduced because the original C is typically already tile-aligned or large enough.

Why it does not help for C=3 stride=1: this path is gated on stride > 1. A 1×1 stride=1 conv with C=3 bypasses it entirely and falls through to the matmul path, where C=3 is padded to 32 in every tile row — 90.6% of each tile is zero. That is the root cause of the 277 GB/s DRAM saturation reported in the issue.

Our Technique — Spatial Row-Group Packing

The same folding principle is applied, but the goal is different: instead of absorbing stride, we pack K adjacent spatial rows into the channel dimension so that C×K hits the next multiple of TILE_WIDTH, eliminating tile-padding waste entirely.

Packing factor: K = TILE_WIDTH // gcd(C, TILE_WIDTH)
For C=3: K = 32 // gcd(3,32) = 32  →  C×K = 96 = 3 × TILE_WIDTH  (0% waste)

Activation packing and unpacking flow (full on-device pipeline):

  [N, C, H, W]  ROW_MAJOR  DRAM
       |
       v  reshape [N, C×K, H/K, W]       -- free view, no data copy
       |
       v  permute [N, H/K, W, C×K]       -- 17.7 MB  (vs 188.7 MB in baseline)
       |                                     C×K=96 fills tile rows 100%
       v  reshape [1, 1, N×H/K×W, C×K]  -- free view
       |
       v  to_layout TILE                  -- 17.7 MB
       |
       v  linear  [C×K → OC×K]           -- reads 17.7 MB (vs 188.7 MB)
       |
       v  to_layout ROW_MAJOR             -- 17.7 MB
       |
       v  reshape [N, H/K, W, OC×K]      -- free view
       |
       v  permute [N, OC×K, H/K, W]      -- 14.2 MB
       |
       v  reshape [N, OC, H, W]          -- free view

Weight packing: block-diagonal [C×K, OC×K] where W_packed[c×K+k, oc×K+k] = W[oc,c] for all k, zero elsewhere. Built using only TTIR-compatible ops — broadcast + arange + eq + typecast + multiply + permute + reshape — with no numpy loops or explicit zero tensors. A single permute [1,2,0,3] absorbs the weight transpose, keeping the op count at 9.

Bias packing: repeat_interleave(K) on the original [OC] bias → [OC×K], where b_packed[oc×K+k] = b[oc].

Comparison with kernel stride folding:

| Aspect              | Kernel Stride Folding        | Spatial Row-Group Packing        |
| ------------------- | ---------------------------- | -------------------------------- |
| Problem solved      | Strided conv overhead        | Tile-padding waste (small C)     |
| Trigger condition   | stride > 1                   | C < TILE_WIDTH                   |
| Fold dimension      | S² spatial neighbours → C    | K spatial rows → C               |
| Channel inflation   | C → C×S²                     | C → C×K (K = 32/gcd(C,32))      |
| Spatial reduction   | H/S × W/S                    | H/K × W (rows only)              |
| Weight change       | kernel ceil(K/S), stride=1   | block-diagonal [C×K, OC×K]      |

Spatial packing eliminates the tile-padding waste entirely for any C where C < TILE_WIDTH. The packed activation is 10.7× smaller than the baseline NHWC TILE tensor (17.7 MB vs 188.7 MB), which reduces both DRAM traffic and the matmul's working set. The weight and bias transformations are mathematically exact — the packed linear is equivalent to the original conv2d for 1×1 stride=1 pad=0, verified numerically with PCC > 0.999.

Tracy profiling results:

| Config             | Input Shape          | Method   | Total Kernel Time | vs Baseline |
| ------------------ | -------------------- | -------- | ----------------- | ----------- |
| Conv2d 1 — Block A | `(1, 3, 1536, 1536)` | Baseline | 14.697 ms         | 1.00×       |
| Conv2d 1 — Block A | `(1, 3, 1536, 1536)` | Method 2 | 1.789 ms          | 8.22×       |
| Conv2d 2 — Block C | `(1, 3, 1280, 2304)` | Baseline | 16.764 ms         | 1.00×       |
| Conv2d 2 — Block C | `(1, 3, 1280, 2304)` | Method 2 | 1.959 ms          | 8.56×       |
