# Fixed-5120 KDA padding experiment

> Historical measurements from 2026-09-17, before the rebase onto PR head
> `bbb915fa5c3`. They do not establish performance of the rebased implementation.
> For reproduction, use the original revisions recorded below and the harness
> at pre-rebase tip `924c1b79ecb`, preserved on
> `backup/kda_pad_before_pr56632_rebase_20260918`.
> Current integration evidence is in [the rebase report](KDA_PADDING_REBASE.md).

**Recurrence is faster than the unpadded 5120-token PR baseline at every tested
padding value.** At 20%, 50%, and 80% padding, measured recurrence savings are
8.44%, 17.62%, and 22.72%. Whole-layer differences are smaller and noisier.

## Comparison

- **Baseline:** published PR #56632, 5120 physical tokens, all 5120 valid.
- **Treatment:** early-exit prototype, always 5120 physical tokens, with varying
  aligned tail padding. No input cropping or shape change in either arm.
- The zero-padding treatment separates prototype overhead from padding savings.

The input tensors have identical seeded synthetic contents across cases.
Padding is designated by the runtime `actual_end` bound; the tail contents are
not zeroed. `actual_start=0`, initial recurrent/convolution states are the same,
and outputs do not feed subsequent timed invocations.

## Results

All tensors remain physically 5120 tokens. Primary PR baseline medians are
**1.707343 ms recurrence** and **12.150426 ms whole layer**. Negative changes
mean lower latency relative to these fixed references.

| Padding tokens | Valid tokens | Padding | Recurrence ms | Change | Layer ms | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5120 | 0% | 1.6893 | −1.06% | 12.0200 | −1.07% |
| 32 | 5088 | 0.625% | 1.6840 | −1.37% | 12.0496 | −0.83% |
| 224 | 4896 | 4.375% | 1.6662 | −2.41% | 12.0914 | −0.49% |
| 512 | 4608 | 10% | 1.6229 | −4.94% | 11.9691 | −1.49% |
| 1024 | 4096 | 20% | 1.5633 | −8.44% | 11.9133 | −1.95% |
| 2048 | 3072 | 40% | 1.4667 | −14.10% | 11.7405 | −3.37% |
| 2560 | 2560 | 50% | 1.4065 | −17.62% | 11.6560 | −4.07% |
| 4096 | 1024 | 80% | 1.3195 | −22.72% | 11.4581 | −5.70% |
| 5088 | 32 | 99.375% | 0.2226 | −86.96% | 10.6107 | −12.67% |

The zero-padding treatment already saves about 1% in recurrence versus the PR.
The listed changes therefore include this implementation-path difference, as
well as savings from skipping padding. Savings are not proportional to padded
token count: 80% padding reduces recurrence latency by about 23%, not 80%.

### Repeatability and limitations

Treatment run medians, in milliseconds:

| Padding | Recurrence forward | Recurrence reverse | Layer forward | Layer reverse |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.688065 | 1.689537 | 11.855611 | 12.156729 |
| 32 | 1.682799 | 1.685077 | 11.970958 | 12.122758 |
| 224 | 1.665846 | 1.666984 | 12.014355 | 12.101512 |
| 512 | 1.622649 | 1.623217 | 11.974476 | 11.966281 |
| 1024 | 1.563077 | 1.563317 | 11.881939 | 11.974459 |
| 2048 | 1.466688 | 1.466641 | 11.789740 | 11.728514 |
| 2560 | 1.406386 | 1.406641 | 11.673595 | 11.653231 |
| 4096 | 1.319458 | 1.319496 | 11.529237 | 11.376165 |
| 5088 | 0.223388 | 0.221642 | 10.672971 | 10.542549 |

- Recurrence forward/reverse medians agree within 0.14%, except the 32-valid-token
  case, which differs by 1.75 µs (0.79%). The recurrence conclusion holds against
  both short PR controls as well as the repeated baseline.
- Short PR A/B recurrence medians are 1.707062 / 1.706430 ms; layer medians are
  11.808284 / 12.074808 ms. The nine repeated PR layer medians range from
  12.017865 to 12.221376 ms. Across all PR controls, layer medians span about
  3.5%, despite unchanged inputs and code.
- The zero-padding treatment's layer median shifts by 2.54% between sweeps.
  Reversing treatment order and matching baseline case count do not eliminate
  this drift. **Layer changes near 0–2% are inconclusive**; the displayed ratios
  are descriptive measurements, not precise isolated padding effects. Larger
  padding values show larger savings, but their exact layer percentages retain
  this uncertainty.
- Read-only AICLK telemetry contains 752 one-second samples spanning 800–1350 MHz
  across active and idle periods. Clocks vary during the experiment; no thermal
  cause or exact correction is inferred. No new device-kernel profiling was
  performed in this sweep, and wall times are not kernel-time decompositions.

## Method

Blackhole, eight devices, SP1×TP8, synthetic Kimi-K3, 96 global heads / 12 local
heads, K=V=128. Each case performs two eager warmups, captures a trace, warms
400 replays, then records 16 samples of 16 replays plus synchronization.
Times are synchronized host wall milliseconds per replay.

Run order: PR A; treatment with increasing padding; treatment with decreasing
padding; PR B; nine repeated PR cases per stage. Each treatment sweep covers
both recurrence and the whole layer.
Both use padding 0, 32, 224, 512, 1024, 2048, 2560, 4096, and 5088 tokens.
Reported treatment medians pool the 32 samples from the two runs. The primary
baseline pools 144 samples from the nine repeated PR cases per stage, matching
the treatment sweep's case count. This additional control was added after
observing order effects in layer timing; short PR A/B runs are preserved as
diagnostics. Percent changes always use the same stage's fixed-5120 PR median,
never a cropped reference.

Recurrence timing excludes projections, convolution and gate preparation.
Whole-layer timing includes these operations and output communication. Each
padding value is captured separately; this experiment measures latency at each
bound and does not add a changing-bound replay correctness test.

## Provenance and reproduction

- Baseline checkout:
  `/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_pr_baseline`,
  SHA `84ae832f1787fc1b8495e9a1b40f8090601a658d`.
- Prototype checkout:
  `/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_runtime`,
  parent SHA `da4bed3135c23887c69f118e080ff34c80fa243b`;
  operation implementation from `57e03d621b9970b5f2c870e67e9df9d9c4675728`.
- The existing isolated native builds are reused; this experiment changes only
  Python benchmark and documentation. Native imports were checked to resolve
  within their respective worktrees.
- Identical [benchmark source](perf/test_fixed_capacity_padding.py) copied into
  both checkouts; SHA-256 and all raw samples are in
  [the result JSON](KDA_FIXED_CAPACITY_PADDING.json).

From the appropriate checkout, export `TT_METAL_HOME="$PWD"` and
`PYTHONPATH="$PWD"`. The serialized safe wrapper activates that checkout's
Python environment:

```bash
# PR checkout: run before and after the two treatment sweeps.
KDA_FIXED_VARIANT=pr KDA_FIXED_PADDING=0 \
  scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_fixed_capacity_padding.py -sv

# Prototype checkout, increasing padding.
KDA_FIXED_VARIANT=early KDA_FIXED_PADDING=0,32,224,512,1024,2048,2560,4096,5088 \
  scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_fixed_capacity_padding.py -sv

# Prototype checkout, decreasing padding.
KDA_FIXED_VARIANT=early KDA_FIXED_PADDING=5088,4096,2560,2048,1024,512,224,32,0 \
  scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_fixed_capacity_padding.py -sv

# PR checkout, baseline with the same sweep case count.
KDA_FIXED_VARIANT=pr KDA_FIXED_PADDING=0,0,0,0,0,0,0,0,0 \
  scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_fixed_capacity_padding.py -sv
```

Full logs, read-only 1 Hz AICLK samples, and aggregation script are preserved in
`/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_runtime/generated/kda_fixed/`.
Clock sampling started after PR A and includes idle periods. No clock overrides,
kernel instrumentation, or operation changes were applied.

These are timing experiments. Their passing pytest outcomes establish completed
execution of the measured cases, not new output/state accuracy coverage.
Existing prototype correctness evidence remains in
[the implementation results](KDA_EARLY_EXIT_RESULTS.md).

## Validation

All **58 timing cases passed**, with explicit safe-wrapper PASS, no failed or
skipped items, and no hangs or resets:

| Run | Cases passed | Pytest elapsed |
| --- | ---: | ---: |
| PR A | 2 | 27.25 s |
| Treatment forward | 18 | 227.58 s |
| Treatment reverse | 18 | 228.41 s |
| PR B | 2 | 27.39 s |
| Repeated PR | 18 | 231.98 s |

JIT telemetry reports 100% cache hits (295 PR / 296 prototype kernels), so no new
kernel compilation was observed. Logs retain existing `to_layout` memory-config
warnings and SWIG/Pydantic deprecations. An initial Black invocation warned
about Python 3.10 versus its default target version; the explicit
`python -m black --check --target-version py310` check passed. No host rebuild
was needed because operation source and bindings were unchanged.
