# KDA prefill offsets: one scan at any 32-aligned offset

Supersedes the implementation half of
[`kda_prefill_offset_verdict.md`](kda_prefill_offset_verdict.md). That document
compared two prototypes as they stood; this one reports the line of work that
removed prototype B's second code path, so **B now handles every 32-aligned
offset on one path** rather than 3 of 19 Galaxy splits.

Branch `mvasilijevic/kda-wrap-runtime`, LoudBox SP2xTP4, Kimi-K3 dimensions.
Baseline is the same layer at `actual_start=0`.

## Verdict

The only constraint remains that the offset is a multiple of 32.

| offset | C=640 (Galaxy per-chip geometry) | C=2560 (LoudBox geometry) |
| --- | ---: | ---: |
| rotation only (`o=0`) | +0.08% | +1.65% |
| smallest split (`o=32`) | +8.05% | +19.21% |
| worst case (`o=C/2`) | +8.06% | +21.23% |
| largest split (`o=C-32`) | +8.19% | +22.71% |

Overhead is **flat to 0.14 pp across every split**, which is the point of the
change: the cost no longer depends on where the wrap lands, and no offset needs
a group size chosen to divide it. Rotation without a split stays free.

The two columns differ for a structural reason, not a plumbing one. A split pins
the chip to a single group, because groups after the wrap would each need an
entry state from a second intra-chip chain and the cross-core prefix takes one
group count for the whole mesh. At C=640 the baseline is one group anyway, so the
columns measure the same thing; at C=2560 the baseline runs four groups and the
split gives that up. **Supporting G>1 under a split is the remaining prize at
large per-chip row counts** and is unrelated to the wrap machinery.

## Correction: the earlier -2.6% / +3.2% figures were a bug

They were measured while the wrap leaked to every chip. `wrap_chunk` is a
mesh-wide attribute and `AddRuntimeArgsForNode` varies runtime args per core,
not per device, so all four chips truncated their published summary and reloaded
a carry at the wrap. Every chip did *less* work than correctness requires, which
is why one reading was negative -- an impossible result that was reported as a
bonus rather than treated as the tell it was.

Per-device behaviour is not expressible as control flow here, so it became
tensor content:

- `summarize_chunk_recurrence` takes a half-open chunk range. The host gets the
  head and tail piece transforms from two disjoint passes that together touch
  every chunk exactly once, so the summary work is unchanged.
- A per-device predicate -- one per candidate boundary chip, built once at
  construction -- selects what differs: the wrapped chip publishes its head
  transform while every other chip publishes its whole partition, and the wrapped
  chip reloads the prefix's final carry while every other chip reloads its own
  carry at that chunk, composed in FP32 so its scan is unchanged.

## Accuracy

PCC alone did not catch the leak: eleven wrong rows in 1280 held PCC at 0.9996
and moved relative RMSE from 1.3e-2 to 1.6e-2, inside the BF16 output's own
noise. Both metrics average a localised fault away. Peak error relative to the
tensor's own RMS does not, so `test_offset.py` now gates all three.

| tensor | worst clean value over every offset | gate | leaking build |
| --- | ---: | ---: | ---: |
| output (peak absolute) | 1.56e-2 | -- | 1.25e-1 to 2.73e-1 |
| output (relative peak) | 8.5e-2 | 0.25 | -- |
| recurrent carry (relative peak) | 3.2e-1 | 0.6 | -- |
| convolution carry (relative peak) | 1.4e-2 | 0.1 | -- |
| all tensors | PCC >= 0.99990, rel RMSE <= 2.1e-2 | 0.999 / 0.05 | PCC 0.9996 |

The bounds differ per tensor because the three concentrate their signal
differently; correct runs elsewhere in the suite reach 1.6e+0, so the peak gate
is opt-in rather than a shared default.

Determinism: trace capture, replay bit-identity, input immutability and output
disjointness are covered per offset by `assert_runtime_contract` at the op level
and by the two determinism cases in `test_offset.py`.

## Where the cost is

Per-program device time, baseline against `o=C-32` at C=640, on the commit
before the L1 change:

| program | ops | delta | share |
| --- | ---: | ---: | ---: |
| matmul | 5 -> 8 | +235.2 us | 38% |
| eltwise | 8 -> 19 | +197.4 us | 32% |
| data_movement | 20 -> 32 | +65.9 us | 11% |
| `ccl.all_gather` | 2 -> 3 | +64.8 us | 10% |
| copy+data_movement | 0 -> 4 | +27.8 us | 4% |
| **`recurrent_chunk_scan`** | **2 -> 3** | **+19.6 us** | **3%** |
| **`qkv_causal_conv1d_silu`** | **1 -> 1** | **+3.9 us** | **1%** |
| total | 58 -> 89 | +623.8 us | +19.0% |

**The kernels are nearly free.** The extra summary pass costs +19.6 us and the
convolution +3.9 us, so 84% of the overhead was the small-tensor affine algebra
on the host graph -- and all of it was in DRAM. Two changes with no effect on
what is computed took C=640 from +18.0% to +8.05%: the algebra now runs in L1
(three [B*H,K,K] matmuls of 100 MFLOP each had been costing 78 us apiece on DRAM
round trips), and `ttnn.where` broadcasts the per-device predicate so each select
is one op instead of three, taking eleven eltwise ops down to four.

## Named follow-ups

- **G>1 under a split.** The dominant cost at C=2560; nothing to do with the wrap.
- **Trace staleness.** The boundary chip selects a pre-built tensor and
  `wrap_chunk` is hashed into the program, so a captured trace is valid only for
  the offset it was captured at. Two offsets can share `wrap_chunk` and differ in
  boundary chip, so shape alone is not a safe trace key. Making the predicate a
  caller-updated input would remove the constraint.
- **The final-state gather** (+64.8 us) fans the whole SP axis in to move one
  state, because `all_reduce` deadlocks the fabric router under trace capture
  here. On Galaxy SP8 it costs 12.6 MB.
- **Galaxy SP8xTP4 remains unmeasured**, as it did for the original verdict.
