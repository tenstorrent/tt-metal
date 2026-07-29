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

## Review 5 OPT-015 shard-advisor attempt

### Starting evidence

`STAGE_REVIEW_5.md` found that the mandatory compiler seed was absent:
`doc/optimized_decoder/shard_advise/report.json` and `final_ir.mlir` had never
been produced, no emitted per-op layouts/programs had been interpreted, and
no compiler-seeded candidate had been compared with the selected
DRAM-sharded decode path.

### Hypothesis experiment

Hypothesis: the missing evidence could be closed by wiring the final dense
layer-0 attention+MLP decode block into the repo-local advisor and running its
bootstrap in a fresh process.

Experiment:

1. Added `tests/advise_optimized_decoder.py`. It uses the established local
   config and synthetic-state helpers, a 1x1 advisor mesh, dense layer 0,
   batch-shaped decode inputs, one physical KV page, the exact paged-cache
   and RoPE arguments, and the final `OptimizationConfig`. The default capture
   batch is 32; `NORTH_MINI_SHARD_ADVISE_BATCH=1` exposes the primary
   small-batch geometry in a separate fresh run.
2. Compiled and imported the capture contract without opening a device.
3. Searched the visible `/opt` and `/home/mvasiljevic` toolchains for
   `ttnn-advise` and `ttnn_jit`.
4. Pointed `TTMLIR_ADVISOR_HOME` at the only visible tt-mlir source tree and
   sourced the required bootstrap in a fresh process:

```text
python -m py_compile \
  models/autoports/coherelabs_north_mini_code_1_0/tests/advise_optimized_decoder.py

TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-xla/third_party/tt-mlir/src/tt-mlir \
bash -lc '
  export TTMLIR_ADVISOR_HOME
  cd "$TTMLIR_ADVISOR_HOME"
  source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
'
```

Result: the capture target passes static validation, but bootstrap exits 1
before system-descriptor generation or capture:

```text
shard-advise: 'ttnn-advise' not on PATH after activating the advisor env.
  The tt-mlir build may lack -DTTMLIR_ENABLE_TTNN_JIT=ON, or was not
  reinstalled after a code change (run: cmake --build build).
```

Neither the CLI nor the `ttnn_jit` Python package exists in the visible
toolchain. The complete bootstrap result is preserved in
`shard_advise/bootstrap.txt`. Per the shard-advisor setup contract, building
tt-mlir inside this model experiment is operator setup and was not attempted.
No TT device, profiler, watcher, or model execution was opened.

Verdict: the model capture hypothesis is verified; the required advisor run
is blocked by unavailable external operator setup. OPT-015 remains open.

### Candidate comparison and safe application boundary

The current selected decode baseline is explicit and remains unchanged:

| Role | Selected layout/strategy | Cores | `in0_block_w` |
|---|---|---:|---:|
| residual + RMS norm | L1 width-sharded | 16 | n/a |
| packed QKV | DRAM-sharded weight/program, L1 width-sharded output | 16 | 4 |
| output projection | DRAM-sharded weight/program, L1 width-sharded output | 16 | 8 |
| dense gate and up | DRAM-sharded weight/program, L1 width-sharded output | 16 | 4 |
| dense down | DRAM-sharded weight/program, L1 width-sharded output | 16 | 6 |

The compiler candidate cannot be stated exactly until the North-Mini
`final_ir.mlir` exists. Its later A/B must take, per emitted matmul, the
advisor's exact required input layout, output layout, core grid,
`in0_block_w`, `per_core_M`, `per_core_N`, and output subblock fields. The
expected comparison family is 1D multicast with L1-interleaved inputs,
DRAM-interleaved candidate weight copies, L1 width-sharded outputs, and every
advisor-emitted boundary revert. Output subblocks must be clamped to the
active compute-kernel register budget. This is a comparison contract, not a
claimed recommendation.

No candidate was added to `optimized_decoder.py` or its performance harness:
without the authoritative IR, doing so would invent the very per-op advice
OPT-015 requires. After operator setup installs the pinned advisor, use a
fresh process and preserve its native output:

```text
TTMLIR_ADVISOR_HOME=/path/to/pinned/tt-mlir bash -lc '
  set -e
  export TTMLIR_ADVISOR_HOME
  cd "$TTMLIR_ADVISOR_HOME"
  source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
  NORTH_MINI_SHARD_ADVISE_BATCH=32 \
    ttnn-advise capture \
    /home/mvasiljevic/tt-metal/models/autoports/coherelabs_north_mini_code_1_0/tests/advise_optimized_decoder.py:decode \
    --out /tmp/north-mini-shard-advice-b32 2>/dev/null
'

cp /tmp/north-mini-shard-advice-b32/report.json \
  models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/shard_advise/report.json
cp /tmp/north-mini-shard-advice-b32/final_ir.mlir \
  models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/shard_advise/final_ir.mlir
cp /tmp/north-mini-shard-advice-b32/report.txt \
  models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/shard_advise/report.txt
cp /tmp/north-mini-shard-advice-b32/pipeline.log \
  models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/shard_advise/pipeline.log
```

Run the batch-1 capture in a second fresh process with
`NORTH_MINI_SHARD_ADVISE_BATCH=1` and a distinct output directory. Only after
reading both `report.json` and authoritative `final_ir.mlir` should the exact
candidate be wired for later PCC and warmed traced-decode A/B at batches 1
and 32. Those hardware measurements were explicitly outside this attempt.

## Review 5 batch-32 dense-expert prefill geometry

The retained 117.903-ms JSON is not a reproducible correct baseline. It came
from an uncommitted predecessor in which `_attention_prefill` called
`_qkv_prefill(..., batch_size=1)` for aligned batch-32 input, has only three
samples, has no PCC, and omits code revision, activation/router/prefill
policy, weight/activation source, and warmup/iteration provenance. Commit
`f77d4e00940` added the stale JSON and already-corrected implementation
together, so the producing source cannot be recovered from Git.

The corrected one-BFP4-family result is 139.965 ms, the final mixed-residency
result is 139.959 ms, and a new final-code one-BFP4-family A/B is 140.177 ms.
The last control refutes duplicate BFP8 expert residency as the latency cause.
It does confirm 612 MiB of avoidable capacity: the mixed expert projections
hold 612 MiB BFP8 plus 324 MiB BFP4, while the aliasing control holds only the
324-MiB BFP4 family.

The isolated BFP4/LoFi sweep measures split 64/80 cores at 100.909 ms, split
88/88 cores at 107.555 ms, and packed 80/80 cores at 96.844 ms for layer 1.
The packed layer-4 control is 96.644 ms. Authentic sequence-33 batch-32
prefill passes at PCC 0.99923857/0.99993403 for layers 1/4
(`artifacts/review5_packed_prefill_authentic.xml`), verifying the candidate.

Promoted packed 80/80 as a prefill-only default:
`prefill_packed_dense_experts=True`; gate/up grid 10x8, inner block 8, core
M/N 4/5, block 4x5, subblock 1x5; down grid 10x8, inner block 6, core M/N
4/7, block 4x7, subblock 1x7. The old global
`packed_dense_experts=False` and zero-valued legacy decode programs are
unchanged, so decode remains split/automatic. Packed weights materialize when
either phase needs them. Eight static geometry/phase contracts, the
optimized-path source audit, compilation, and CLI parsing pass.

Whole-tensor L1 placement is not a legal M=1024 alternative: the smallest
relevant BF16 expert intermediate is 192 MiB, versus only about 165 MiB
theoretical aggregate worker L1 before runtime and CB reservations. The
minimal serialized hardware matrix and exact authentic sequence-33 follow-up
command are in `PREFILL_GEOMETRY_AUTOFIX.md`. No TT device, profiler, watcher,
or default-policy measurement was run while implementing the promotion. The
candidate hardware results were supplied by the serialized parent run.

The prefill-only topology adds a 216-MiB packed BFP4 gate/up family alongside
the split decode weights, increasing selected sparse-layer expert residency
from 936 MiB to 1,152 MiB. The existing advertised-context capacity gate must
be repeated for the new final resident set before closure.
