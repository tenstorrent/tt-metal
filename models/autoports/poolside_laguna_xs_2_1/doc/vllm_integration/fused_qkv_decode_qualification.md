<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fused decode-QKV qualification (2026-08-22)

**Verdict: keep `TT_LAGUNA_FUSE_QKV_DECODE=0`.** The experimental split is
correct, but it regresses traced single-layer decode latency by 0.83%. It also
does not run in the qualified p150x2 server: `MultichipDecoder.decode_forward`
overrides the single-device implementation and calls `_split_qkv` directly.

## Scope

- Model tree: `2de59ac6ee92232be42186a2f9227cd4b707036d`
- Installed TT runtime: `559921b40a8b7b21c807d5592323c2fa5e8c7ecb`
- Hardware: one P150 ASIC (`TT_VISIBLE_DEVICES=0`)
- Path under test: `OptimizedDecoder.decode_forward`, DRAM-interleaved packed
  QKV input, `nlp_create_qkv_heads_decode`, and the unchanged norm/RoPE/cache tail

## Correctness

The focused regression test added to `test_optimized_decoder.py` checks exact
BF16 equality for every Q, K, and V value after splitting `[Q|K|V]`. It covers
both Laguna attention geometries (48Q/8KV full attention and 64Q/8KV sliding
attention) at batch 1 and batch 32: **4/4 passed**.

With `TT_LAGUNA_FUSE_QKV_DECODE=1`, the existing real-weight tests also passed:

| Test | Result |
|---|---:|
| full-attention decode after 32-token prefill | pass |
| full-attention batch-32 prefill/decode | pass |
| full-attention traced decode replay | pass |
| sliding-MoE decode after 32-token prefill | pass |

The single-device sliding test at position 513 reports PCC 0.01380 with both
the flag off and on. This is the pre-existing D1 explicit-SDPA-program issue,
not a fused-split delta; the focused split test remains exact for sliding QKV.

## Performance

Command shape, run from `/tmp` against the installed self-consistent runtime:

```bash
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=/home/ttuser/.local/lib/model-bringup/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
TT_LAGUNA_FUSE_QKV_DECODE=<0|1> \
PYTHONPATH=/home/ttuser/dev/laguna/tt-metal \
python -m models.autoports.poolside_laguna_xs_2_1.tests.perf_trace_opt 0 512 5000
```

| p150 D1, layer 0, prefill 512, 5,000 traced replays | ms/token/layer | Delta |
|---|---:|---:|
| legacy slice/reshape split | 0.5047 | baseline |
| fused QKV split | 0.5089 | **+0.83% latency** |

Feeding the QKV matmul's L1 width-sharded output directly was also rejected: the
fused op failed because its 16,384-byte circular buffer exceeded the source
shard's 8,192-byte L1 bank. Reusing fused V directly as the cache input was
correct but did not recover the latency regression.

## Decision

No runtime change is enabled. Keep the flag experimental and default-off. Do
not count it as a p150x2 optimization unless the multichip decoder is explicitly
wired and a downstream sharded norm/RoPE/cache design removes enough conversions
to beat the qualified traced baseline.
