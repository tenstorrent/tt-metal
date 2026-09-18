# KDA padding on PR #56632 at f20c87e9de6

## Revision and semantic integration

The parent is `f20c87e9de64b5e791ecc7c0f048491ed67beda9`, following
`bbb915fa5c30023c7b9667d64a1b86d7c48ae91d`. The published padding tip before
this rebase was `3c7ae41f9be8cb2c991bfd9b941fbf79164ef830`; it is retained as
`backup/kda_pad_before_pr56632_rebase_f20c`. The rebased measured source is
`3539a17b474f1d5995870c61f88ed8f0210fa3e6`.

The upstream commit removes redundant recurrent-scan writer handoffs and uses
the canonical `_scan_chunks` orchestration. Both changes are preserved.
The writer chronology channel now exists only for summaries or when `actual_end`
is supplied: a padded writer needs the valid chunk count and destination of the
last active group's final state. With no end bound, an ordinary recurrent scan
uses its physical chunk count without this channel. Reader publication and writer
consumption are compiled under the same `KDA_WRITER_CHRONOLOGY` condition.
The upstream compute-state reference and required-seed validation also remain.

## Accuracy

All 40 selected cases passed: three native padding cases and 37 layer/selection
cases. The latter include three changing-padding traces (SP1/TP8, SP2/TP4,
transposed SP4/TP2), ten parent offset/production trace cases, and 24 layer and
chronology-selection contract cases. Independent references use pure Torch.

| Scope | Tensor | Minimum observed PCC | Maximum relative RMSE |
| --- | --- | ---: | ---: |
| Changing padding | Output | 0.999938 | 0.01342 |
| Changing padding | Recurrent carry | 0.999874 | 0.02531 |
| Changing padding | Convolution carry | 0.999996 | 0.002732 |
| Production local offset traces | Output | 0.999951 | 0.01045 |
| Production local offset traces | Recurrent carry | 0.999788 | 0.02982 |
| Production local offset traces | Convolution carry | 0.999997 | 0.002585 |

These are minima/maxima of the logged comparisons, not aggregate correlations.
Padding gates use PCC >=0.999; production local traces use >=0.9995. The shared
relative-RMSE gate is 0.05. Peak-error gates are tensor/configuration-specific;
the recurrent peak gate is intentionally absent for the existing SP1/full-length
padding and production-local policies. Maximum observed recurrent relative L-inf
was 0.822 for padding and 2.29 for production local traces; these are reported,
not interpreted as passing a uniform peak-error bound.

The padding cases exercise changing device bounds in a single capture, nonzero
starts/carries, partial groups, empty ranks, separated tails, and returning to
full length. Exact-equality checks cover repeated replay, padded-data invariance,
input-carry immutability, applicable physically trimmed carries, and full-length
unbounded carries. Native preparation/scan and analytical prefix checks pass exact
comparisons, including fresh-bound rebinding. Production local accuracy covers
640/2560 rows and both TP orientations; changing-padding CPU-reference checks use
256 rows per SP rank. The timing sweep at 5120 physical rows does not itself
perform an accuracy check for each of its nine padding values. No real checkpoint
or Galaxy validation was performed in this rebase.

## Controls and timing domain

Baseline: unmodified parent production code, physical/valid length 5120, no end
bound. Treatment: rebased padding code, physical length 5120, valid length
`5120-padding`, including zero-padding control. Padding values are
0, 32, 224, 512, 1024, 2048, 2560, 4096, 5088. These are fixed-capacity
comparisons, not equal-valid-length comparisons with increasing allocations.

Blackhole, eight devices, SP1 x TP8, deterministic synthetic Kimi-K3 weights and
inputs, 96 global heads, K=V=128. `actual_start=0`, zero input carries, same seeded
physical input for each case; padding contents are not zeroed. Each invocation
reads the original carry; results are not fed to the next replay. Recurrence
excludes projection/convolution/gate setup; layer measures full layer forward.

Each case: two eager warmups, one trace capture, 400 warm trace replays, then
16 samples of 16 asynchronous replays followed by synchronization. Units are
synchronized host wall milliseconds per replay. No device profiling or theoretical
utilization claim is made. Run order: parent before; ascending treatment;
reverse treatment; parent after; nine repeated unpadded parent cases per stage.
The last control was added after observing a 1.74% zero-padding layer shift
between treatment orders, to match the number of cases in a sweep. The primary
reference pools these nine repeated PR cases; the before/after PR results remain
independent drift controls. Raw samples and per-run medians are retained.
Small effects within the observed baseline/order variation are inconclusive.

The same isolated checkout and local Python environment were rebuilt at each
revision transition. Parent runs add only the identical standalone benchmark
harness to the checkout; production source remains exactly the parent. The
branch and its native build are restored afterwards. No device reset is issued.

## Reproduction and evidence

Checkout: `/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_runtime`.
Runtime artifacts: `generated/kda_rebase_f20c/` in that checkout. Native builds
use `source python_env/bin/activate` then
`./build_metal.sh --enable-ccache --build-ttnn-tests`. Post-build imports verify
that `ttnn` and `_ttnn.so` resolve within this checkout.

Hardware tests use the serialized `ai_workspace.tt.testing.test` API from the
workspace environment. Exact item selections and individual outcomes are in
retained runner JSON and the committed evidence manifest. The timing harness is
`perf/test_fixed_capacity_padding.py` with `KDA_FIXED_VARIANT=pr` or `early` and
`KDA_FIXED_PADDING=0,32,224,512,1024,2048,2560,4096,5088` (parent selects only zero).
Read-only sysfs AICLK telemetry accompanies the sweeps. Full logs remain in the
runtime artifact directory.

## Performance results

The matched-count PR reference medians are **1.687821 ms recurrence** and
**12.059313 ms layer**, pooling 144 samples per stage. Treatment medians pool
32 samples per configuration across the two orders. Negative changes mean
lower latency. These are ratios of pooled medians, not paired-effect estimates.

| Padding | Valid | Recurrence ms | vs PR | Layer ms | vs PR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5120 | 1.6912 | +0.20% | 11.9934 | -0.55% |
| 32 | 5088 | 1.6874 | -0.02% | 12.0302 | -0.24% |
| 224 | 4896 | 1.6688 | -1.13% | 12.0614 | +0.02% |
| 512 | 4608 | 1.6246 | -3.75% | 11.9670 | -0.77% |
| 1024 | 4096 | 1.5625 | -7.42% | 11.9025 | -1.30% |
| 2048 | 3072 | 1.4673 | -13.07% | 11.7587 | -2.49% |
| 2560 | 2560 | 1.4077 | -16.59% | 11.6835 | -3.12% |
| 4096 | 1024 | 1.3200 | -21.79% | 11.4962 | -4.67% |
| 5088 | 32 | 0.2233 | -86.77% | 10.5759 | -12.30% |

### Repeatability and interpretation

PR recurrence medians before/after were 1.686529 / 1.685834 ms; the nine repeated
PR medians ranged 1.686717–1.689081 ms. Treatment forward/reverse recurrence
medians agree within 0.18%. Zero-padding recurrence is slightly slower on both
orders: approximately 0.20% (3.3 microseconds) versus the matched-count baseline,
or 0.31% versus the bracketed baseline. A strict zero-overhead claim is not
supported. The 32-padding-token case is effectively unchanged; savings from
224 padded tokens onward are repeatable against all these controls.

The layer comparison is noisier. PR before/after medians were 11.789527 /
11.854810 ms, and its nine repeated medians rose from 11.859574 to 12.132670 ms.
The zero-padding treatment changed from 11.919109 to 12.126175 ms between orders
(1.74%). Comparing the same treatment samples against the short bracketed PR
reference gives +1.68% at zero padding and +2.26% at 224 padding, whereas the
matched-count reference gives -0.55% and +0.02%. All controls are retained;
the choice of baseline materially affects these small layer percentages.

**Decision:** accuracy supports the rebase. The recurrence padding benefit is
supported, with a small measured zero-padding overhead. A blanket layer
no-regression claim, or an exact gain near 0–3%, is inconclusive under the
observed variation. Large-padding layer savings remain visible against every
PR control, especially at 4096/5088 padding, but their exact percentages remain
sensitive to timing conditions. The source was not tuned in response to these
measurements. Resolving sub-percent layer effects would require a separate
controlled measurement investigation, not another interpretation of these samples.

| Padding | Recurrence forward ms | Recurrence reverse ms | Layer forward ms | Layer reverse ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.690480 | 1.691934 | 11.919109 | 12.126175 |
| 32 | 1.687279 | 1.687514 | 11.991542 | 12.128088 |
| 224 | 1.668955 | 1.668753 | 11.968804 | 12.108223 |
| 512 | 1.623455 | 1.625152 | 11.971854 | 11.934299 |
| 1024 | 1.561594 | 1.564371 | 11.843114 | 11.910472 |
| 2048 | 1.466227 | 1.467640 | 11.782163 | 11.730259 |
| 2560 | 1.407143 | 1.407852 | 11.717474 | 11.654917 |
| 4096 | 1.318915 | 1.320554 | 11.544452 | 11.432894 |
| 5088 | 0.223407 | 0.223124 | 10.620826 | 10.490219 |

Read-only AICLK telemetry contains 1095 one-second samples, spanning
800–1350 MHz across active and idle periods. No clock correction
or thermal cause is inferred.

## Validation record and limitations

[Raw samples, exact test selections, build identities and accuracy metrics](KDA_PADDING_REBASE_F20C.json)
are committed beside this report. All 40 accuracy/contract cases and 58 benchmark
cases passed; wrapper records report no observed hangs or device resets. Native
builds and post-build imports succeeded at both revisions. The first treatment
build took 33.927 seconds. Normal JIT/cache behavior was used; this is not an
isolated cold-compilation proof.

Observed warnings: dependency CMake minimum-version deprecations (protobuf,
gtest/googlemock/googletest, md4c), CPM's nanobind version notice (requested
2.10.2 versus included version label 0), md4c CMP0069/IPO development warnings,
and Python SWIG/Pydantic deprecations. Full warnings remain in build/test logs.
No test assertion or native build failure occurred during this rebase.

No new CI test selection or workload was added. Earlier KDA padding reports
remain historical evidence for their stated revisions; this report supersedes
them for the measured f20c-based branch.
