# AutoFix report: batch-32 active experts

## Starting evidence

`AUTODEBUG.md` found that the selected batch-32 path is a correct,
device-resident dense all-expert implementation, but that it does not satisfy
the optimize skill's active-expert preference. The selected path is 3.330 ms;
functional is 11.122 ms. Existing dynamic `ttnn.sparse_matmul` attempts were
20.535–21.896 ms.

Three independent model-side hypotheses were verified or refuted in isolation.
No speculative implementation change was retained.

## H1: binary mask and exact static nnz

Hypothesis: separate a BF16 binary sparsity mask from sigmoid routing weights
and pass exact `nnz=token_count*top_k=256`, avoiding the 4096-entry dynamic
receiver/compute loop.

Experiment:

- Under `TT_METAL_WATCHER=10`, device top-k/scatter produced a BF16 row-major
  `[1,32,1,128]` mask with exactly 256 nonzeros, eight per row, values `{0,1}`.
- The first static-nnz projection requested a 201,326,592-byte L1 sparse
  output; 1,830,912 bytes per bank exceeded the 1,461,504-byte limit.
- Projection outputs were adapted to DRAM. SiLU then hit the same L1 limit, so
  every expert intermediate was adapted to DRAM.
- The adapted one-replay watcher run completed without deadlock or watcher
  signature.
- Warmed traced full layer-1 batch-32 decode used 3 warmups and 20 iterations.

Result:

| Check | Result |
|---|---:|
| PCC | 0.99784756, pass |
| mean / min | 17.8311 / 17.6786 ms |
| versus dynamic sparse | 18.6% faster |
| versus functional | 1.60x slower |
| versus selected | 5.35x slower |

Verdict: refuted. Static `nnz` is safe and helps, but full padded
token-by-expert output zero-fill and synchronization still regress the serving
batch.

## H2: packed sparse gate/up

Hypothesis: pack `[E,H,I]` gate and up weights into `[E,H,2I]`, use one
static-nnz sparse projection, split, activate/multiply, and use one sparse down
projection.

Temporary probe form:

```bash
timeout 180 python tests/_tmp_packed_sparse_probe.py \
  --gate-grid <grid> --down-grid 8x8 \
  --gate-block <block> --down-block <block> \
  --warmups 2 --iterations <8|10>
```

The temporary file was removed after the experiment.

| Packed gate config | Mean / min | PCC |
|---|---:|---:|
| 8x3, block 16; down block 12 | 20.8982 / 20.7541 ms | 0.99784756 |
| 8x4, block 16; down block 12 | 20.9420 / 20.6917 ms | 0.99784756 |
| 8x6, block 16; down block 12 | **19.5838 / 19.3762 ms** | 0.99784756 |
| 8x6, block 8; down block 12 | 19.6320 / 19.5580 ms | 0.99788428 |
| 8x6, block 32; down block 24 | 19.7319 / 19.4391 ms | 0.99777560 |
| 8x6, block 16; `ttnn.swiglu` | 19.6017 / 19.5040 ms | 0.99529010 |

Verdict: refuted. Packing removes a launch but doubles the N width of the
remaining full-surface projection. The best adapted legal geometry is 76.1%
slower than functional and 5.89x slower than selected.

## H3: fused single-card moe_compute

Hypothesis: `ttnn.experimental.moe_compute(compute_only=True)` can use its
active-token dispatch, packed gate/up, activation, and down kernels without
materializing the full 32x128 expert surface.

Exact supported-shape probe:

```python
from ttnn.operations.ccl import MoEActivationFunction
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card import (
    _run_moe_compute_single_card_test,
)

_run_moe_compute_single_card_test(
    mesh,
    (1, 1),
    128,  # experts
    32,   # tokens
    8,    # top-k
    768,  # intermediate
    2048, # hidden
    4,
    4,
    ttnn.bfloat16,
    MoEActivationFunction.SILU,
    False,
)
```

The exact North dimensions are supported on 1x1 Blackhole with BF16 inputs and
BFP4 packed weights. Counts, activation metadata, E-to-T mapping, and final
double-buffer validation passed. Component timing:

| Mode | Result |
|---|---:|
| warmed untraced, 20 iterations | 1.7089 ms mean |
| traced, 50 replays | 1.6419 ms mean |

The hard output contract blocks model integration:

- `compute_only=True` returns a matmul tensor `[110,2,32,2048]`.
- The `2` axis is a rolling kernel double buffer. With 128 experts, only expert
  outputs 126 and 127 remain after completion; earlier outputs were consumed
  by the in-kernel combine and overwritten.
- All routed contributions therefore cannot be reconstructed or score-weighted
  without a host fallback or shared kernel/API change.
- Full mode is the only built-in combine. On a 1x1 mesh it first failed with
  `Trying to get un-initialized fabric context`. The adapted
  `FabricConfig.FABRIC_1D` attempt timed out waiting for a remote handshake
  partner (`expected LOCAL_HANDSHAKE_COMPLETE`). All boards were healthy after
  the bounded attempt.

Verdict: blocked by the current single-device output API. The fast compute
kernel exists, but a single-card combine/output mode is not exposed.

## Final status

AutoFix exhausted the legal model-local active-expert families:

- dynamic sparse matmul;
- exact static-nnz sparse matmul with every necessary DRAM adaptation;
- packed static-nnz sparse gate/up across legal grids and block widths;
- fused single-card active-token `moe_compute`, including its full-mode fabric
  adaptation.

No active-expert candidate both exposes a correct complete decoder output and
meets the batch-32 no-regression requirement. A shared TTNN change is needed:
either compact routed outputs from sparse matmul, a persistent all-expert
output from `moe_compute(compute_only=True)`, or a fabric-free single-card
combine. Shared-kernel work is outside this stage's model-local file scope.

The selected 3.330-ms device-resident path remains because it is correct,
traceable, substantially faster than functional, and does not regress batch
32. Batch 1 continues to use the active-expert sparse path. This limitation is
forwarded to independent stage review rather than misrepresented as a
completed active-expert implementation.
