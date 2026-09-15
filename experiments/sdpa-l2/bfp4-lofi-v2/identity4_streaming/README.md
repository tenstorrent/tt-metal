# Identity-only FP32 recurrence batching

Isolated native-exp LoFi streaming experiment. No existing streaming,
qualification, frozen, fullchip, or resident source file is edited.

## Change and numerical contract

The private compute_streaming.hpp is a byte-for-byte copy of the current
v2/streaming/compute_streaming.hpp except for one SDPA_IDENTITY4 block inside
the FP32 branch of salad_correct_fused. The copied base header SHA256 was
85dbc4ecfeeb682bfc59c7c24a28446f80bafd00f1604178c201efb232dd3aa8.

When the existing maximum-comparison logic identifies identity rescaling and
the output row has four tiles, load all four old FP32 output tiles into the
four-tile DST half in one acquisition. Pack all four using the same scalar
pack operations, order, FP32 L1 accumulation, and output addresses as before.
Then execute the original separate first-column denominator update.

No SFPU arithmetic is added, deleted, or reordered per output element:
identity rescaling already skips the multiply. The only intended changes are
DST acquire/commit/wait/release grouping and redundant unary init/uninit work.
The nonidentity path is unchanged. This relies on the standard non-ring
harness's distinct previous/current recurrent buffers; no new alias is used.

For eight query tile rows per full K chunk:
- Before: two output-pair batches + one sum batch per row = 24.
- Identity4: one four-output batch + one sum batch per row = 16.
- Still 32 scalar output packs and eight scalar sum packs, with the same
  elementwise accumulation order. No blocked state packing is introduced.

Q256/K512/D128; FP32 score/P/DST/recurrent state; native approximate exp and
the existing matched LoFi denominator. Inputs are Q RNE7/BF16 and K/V
RNE5/BFP8. Input slots remain Q=2, K=1, V=1. Fullchip uses the unchanged
per-head chain reader, source-linear K and barrier2. Resident uses the
unchanged repeated-input reader, with host input preparation outside timing.

## Controls and qualification

Both drivers accept --identity4-mode off|on|both; **both is the default**.
They execute the same private header, with or without SDPA_IDENTITY4.
Before any timing, check all output values finite, L2 < --max-l2 (default5%),
PCC >= --min-pcc (default0.99 when defined), and all BF16 output bits exactly
equal between off/on. Undefined PCC for a constant reference is reported and
does not reject that case. A failure saves both complete output matrices.
These are experimental smoke gates, not a revised attention acceptance SKU.

Resident passes do not qualify general recurrence: repeated KV should make
almost all updates identity. Distinct-KV all-Q comparison is mandatory.
Neither driver records device branch counts, so no branch-coverage guarantee
is inferred from bitwise equality alone.

Run with fresh labels from the worktree root:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/identity4_resident.py --label identity4-resident-smoke --q-repeats 2 --k-chunks 8
python experiments/sdpa-l2/bfp4-lofi-v2/identity4_streaming.py --label identity4-distinct-smoke --length 4096 --heads 2 --cores 4 --sample-rows 4096 --check-preprocess
python experiments/sdpa-l2/bfp4-lofi-v2/identity4_resident.py --label identity4-resident-perf --q-repeats 16 --k-chunks 512 --iters 20
```

Results go under identity4_streaming/<label>.json. Device trace timing occurs
only after both candidates pass all gates. Replayed outputs are checked bitwise
again. Source hashes are captured before execution and verified afterward.
Fullchip reports attention/preprocessing/combined useful TFLOPs; resident
reports attention TFLOPs for one core and excludes input preparation.

CB payload is unchanged at 1,212,416 bytes/core. With the previously observed
111,616-byte reserved L1 allowance, estimated free L1 is 248,832 bytes.
The actual device allocator remains authoritative.

Local validation: Python syntax, CLI parser/entrypoint checks, off/on mocked
host descriptors for both drivers, relative include paths, and byte-identical
base-header comparison after removing only the inserted opt-in block.
No JIT or device jobs were run by this implementation agent.
