# laneMQ — the two-operand 2^32 sem-vs-hand silicon streamer

The laneMK galaxy streamer (`fp32_stream_lib.py` / `fp32_stream_sweep.py` /
`LANEMK_STREAM` in `test_sfpu_unary.py`) proves single-input fp32/int32 ops
bit-exact-equal on silicon by streaming the entire 2^32 raw-uint32 space through
both certified corpus legs and comparing per-leg SHA-256. This is that tool
**widened by exactly one dimension** so it can close a genuinely two-operand op —
`binarypow` (`SfpuElwpow`), the last `z3-timeout` cert cell — whose base **and**
exponent are each an independent per-element `bf16` draw.

Because both operands are `bf16` (16-bit), the joint input space is
`2^16 x 2^16 = 2^32`, **not** `2^64` (correction to the earlier assumption — see
laneMP). It is therefore exhaustible on silicon exactly like a single-input
`2^32` op, at ~1 op's cost.

## The one-dimension-wider idea

A joint index `J in [0, 2^32)` splits into two raw `bf16` bit patterns:

    base16 = J >> 16      exp16 = J & 0xFFFF

Sweeping `J` over `[0, 2^32)` visits every `(base, exponent)` bf16 pair exactly
once. `base16`/`exp16` are raw bit patterns — no host `float()` ever touches them,
so subnormals / inf / NaN payloads are delivered exactly (the laneMK discipline).

### Device ABI — why no separate raw-B write is needed here

`sources/sfpu_binary_test.cpp` consumes **both** operands from `buffer_A`: it loops
`for (tile = 0; tile < N; tile += 2)` reading operand 0 from the even tile and
operand 1 from the odd tile of each pair (see `test_sfpu_binary.py`
`_pair_operand_specs`: "operand 0 from the even tile of each pair and operand 1
from the odd one"). So the two-operand injection is a single **interleaved
`buffer_A` payload** — even tile = base, odd tile = exponent — driven through the
**existing, validated** laneJN raw-A L1 path. `binary_stream_lib.interleaved_payload`
builds it: per tile pair `p`, the even tile is a constant `base16` and the odd tile
is the consecutive `exp16` run `[lo, lo+1024)`. Dispatch/tile alignment guarantees a
run never crosses a `2^16` base boundary, so the cover is exact and gap-free. The
intra-tile face permutation is irrelevant to the verdict: both legs receive the
identical payload, so it only permutes which output slot holds which joint point,
and coverage is a set property.

### The reusable raw-B path (for a different ABI)

For a future binary op that instead keeps operand B in a **separate** `buffer_B`,
`helpers/stimuli_config.py` gains a `lanejn_raw_b` path that mirrors the laneJN
`lanejn_raw_a` path (writes raw per-tile bytes to `buf_b_addr`, inert unless set).
It is additive and reusable; `binarypow` does not use it (its ABI is interleaved-A),
but laneMO's separate-buffer binary ops can adopt it.

## Soundness

* **Object identity** — both legs are the certified pin-59 kernels of the booked
  node `test_fresh_cpp_binary_pow` (sem = `fresh_cpp_impl:1`, the fresh semantic
  `2^(b*log2 a)`; hand = `fresh_cpp_impl:3`, the byte-untouched
  `calculate_sfpu_binary_pow`). The streamer re-fires the *same* `configuration` /
  ELF each dispatch, and the `.text` gate refuses on `sem == hand` or a hash
  mismatch. Labels are authoritative by compiling each node **alone** (the laneMK
  compile-together mislabel lesson).
* **Whole-region hash** — only whole dispatches run (band sizes and
  `joint_per_dispatch` are powers of two), so the whole *cleared* result region is
  hashed each dispatch. A result tile the kernel does not write stays at the `0xA5`
  clear sentinel in **both** legs and cannot manufacture a spurious divergence; the
  tiles it does write carry the only place the two kernels can differ.
* **Full-space attestation** — bands tile `[0, 2^32)` with `covered == 2^32`
  asserted; the verdict is `BIT-EXACT-ALL-INPUTS` only if every band's
  `sem_sha == hand_sha`, else `DIVERGENT` with the DIFF bands as witness bands.

## Files

| file | role |
|------|------|
| `binary_stream_lib.py` | device-independent core: joint enumeration, interleaved-payload packing, coverage checksums, per-leg digest, band verdict (reuses laneMK `first_divergence` + `texthash_gate`) |
| `binary_stream_sweep.py` | orchestrator: resume-safe band sweep, one band-leg per pytest invocation on a flocked chip, `.text` identity gate, coverage assert, VERDICT |
| `selftest_binary_stream.py` | mandated device-independent selftest (enumeration, payload+full-cover, known-equal, divergent+witness, text-gate) |
| `LANEMK_STREAM_BINARY` hook in `test_sfpu_binary.py::_lanemk_run_binary_stream` | the device leg (persistent session; per dispatch inject interleaved A, clear Res, run, read, fold SHA) |
| `lanejn_raw_b` path in `helpers/stimuli_config.py` | reusable raw-B (separate-buffer) inject path, additive + inert |

## Run

```
# 1. build both certified legs into a RUNNER_TEMP
CHIP_ARCH=blackhole SHORT_ARCH=bh RUNNER_TEMP=$RT \
  python -m pytest -q --compile-producer \
  'test_sfpu_binary.py::test_fresh_cpp_binary_pow[...fresh_cpp_impl:1]' \
  'test_sfpu_binary.py::test_fresh_cpp_binary_pow[...fresh_cpp_impl:3]'

# 2. sweep the full 2^32 joint space (per-chip; shard bands across chips/galaxies)
python binary_stream_sweep.py --op binarypow \
  --sem-node  '...fresh_cpp_impl:1]' --hand-node '...fresh_cpp_impl:3]' \
  --farm <tests/python_tests> --venv <python> --llk-home <tt-llk> \
  --runner-temp $RT --band-bits 26 --chip 0 --out <dir> [--idmap <tsv>]
```

`lanemq_run_op.sh` (ONE op, run-to-completion-and-quit, resume-safe) + a Slurm
`--array` submitter mirror the laneMK galaxy model when running on the exabox
galaxies. The DIFF bands of a `DIVERGENT` verdict are narrowed by re-running that
band at a smaller `--band-bits`; a first-witness input is then sim-confirmed on the
pinned instrument (`formal_equiv.py` / `LANEJO_SRC_OVERRIDE`).
