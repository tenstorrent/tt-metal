# Prefill cache reorder

The SP gather returns rank-major stripes. Viewing them as
[4, capacity/1024, 256, 128], swapping the first two axes, and viewing the result
as [1, 1, capacity, 128] restores chronological token order. Both tile dimensions
stay aligned. This replaces capacity/256 slice operations and their concat with
a fixed number of operations. The full-capacity collective and the final
ceil(actual_end/32)*32 prefix slice are unchanged.

At capacity 1024, rank order is already chronological. An owning clone keeps
caller deallocation independent of the persistent gather buffer. For larger
capacities, the transpose owns its output; reshapes share that allocation.
Borrowed views must not be force-deallocated.

## Evidence and limits

The portable component regression passed eight cases: capacities 1024, 2048,
8192 and 131072, each with BF16 and BFP8_B caches, on all 32 Blackhole chips.
The two-plane fixture checked both slots, chronological values reconstructed
independently from stored cache data, padded prefixes, repeats, retained outputs
after gather reuse, and persistent-buffer survival after caller deallocation.
All 6,208 exact decoded-value checks passed, all source pins stayed unchanged,
and the mesh closed cleanly.

Component acceptance:

- /data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/cache-reorder-validation-runs-001/component-c06-001/root-result-verification.json
- SHA256 3d454cb85d70ea1955b53e63fa4d85ed997077c8bed9416e3dbb9401ae904311
- Device-tested attention SHA256 c7494bfd17549378660e166d0d0fd5b03d94084d610502683816836fa1edff75
- Published attention SHA256 45350c26f5299dfd9144ddf9354bea17223899883ee300cd07a3e87095df1e59

The normal Black hook joined one `ttnn.reshape` call from three lines to one.
The two source hashes above have identical parsed Python ASTs. The host and
device regression files retain their reviewed SHA256 values
2acb13a3ac499da2b917595673f639af1a3d2e5977d441805d2cc125f0aa38d9 and
e74a1555eb344a346eaf2fbb9837b58e579c96a5791e6095174fc224c2eaacdd.

An integrated 2K regression also passed against saved accepted device output.
It covered four full 32-layer forwards with BFP8 caches across two slots, 128
hidden-shard checks, 538 selected full-vocabulary rows, 128 input-preservation
checks, exact candidate source identity and clean mesh close.

Integrated acceptance:

- /data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/cache-reorder-validation-runs-001/integrated-c06-001/root-result-verification-v2.json
- SHA256 6120e54f09c28a9953209c2f0052140028458a5322a1a73025b86dc99c65dac4

These checks establish component decoded-value and ownership parity plus the
saved-output 2K integration case. They do not establish packed-byte identity,
new Hugging Face accuracy acceptance or a speedup. The portable regression uses
the production method directly and an independent cache oracle; it omits a
redundant comparison against an archived implementation.

An earlier full 128K benchmark reached its 10,800-second pytest limit:
actual/dispatch exits 124, verification 90, no persisted completed-request
report, and no clean close. All 14,729 source pins were unchanged. The preserved
terminal receipt is evidence/device-activity/20260917T1945-c06-post-timeout/terminal-preservation.json,
SHA256 52b751791d24b0a20ecaf5e05f068a16753c74cc03d90a629a26411cf61a691c,
under the same shared root above. Recovery and a fresh indexed-RoPE health check
subsequently passed. A sampled stack and static operation counts motivated this
rewrite; they do not measure its share of runtime.

A new 128K performance run is in progress and has no accepted result yet.
Existing 2K-64K performance results remain measurements of their recorded
source versions, not measurements of this rewrite.

## Regression commands

Host-only ordering, ownership, padded-tail and bounded-operation checks do not
import Torch or TTNN:

```sh
python3 -I -S -B models/demos/llama_3p1_8b_d_p/tests/unit/test_cache_reorder_host.py
```

On an assigned, reviewed Galaxy with the normal repository environment:

```sh
python3 -m pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_cache_reorder_vs_ref.py
```

This is one two-plane cache, with no model weights or full-layer cache. At 128K,
only a full prefix and the padded 1025-token prefix run. Smaller capacities
cover additional awkward tails. Outputs stay live across slot reuse; the
report records actual readback bytes and separates component pass from model
acceptance.

## Remaining integrated validation

The accepted 2K saved-output regression covers both slots with BFP8 caches
through four full-model chunk forwards, hidden-shard checks, selected
full-vocabulary rows and input preservation. No 2K performance remeasurement
is needed.

The 128K full-32-layer execution/performance run remains pending acceptance. Its
contract uses two sequential slots, one warmup plus three measured requests per
slot, 1,024 chunk calls and 32,768 finite/repeat output checks, followed by clean
mesh close. Timing definitions and CPU-thread context must remain explicit. No
128K golden KV, decode or generation claim follows from that run.
