# Review of the merged PRs for issue #21412 (FFT and inverse FFT)

Issue: [#21412 Add FFT and inverse FFT](https://github.com/tenstorrent/tt-metal/issues/21412).
The request is: "Implement an efficient, multi-core, FFT and inverse FFT function, for bf16 and fp32 dtypes."
The parent issue [#11835](https://github.com/tenstorrent/tt-metal/issues/11835) was closed as a duplicate of #21412.

## Summary

The FFT works for the main cases, but #54930 has two bugs that crash the process or give silent wrong results, plus three other bugs. No FFT test runs in CI on any architecture. The "efficient" part of the request is not met: in my timing the op was slower than one CPU thread at every size I tried. The fixes I would want before closing the issue are bugs 1 to 5 of #54930, an FFT CI job, and either real efficiency work or an agreed decision that "efficient" is out of scope.

## Scope and method

Merged PRs reviewed, in merge order:

| PR | Title | Merged |
|---|---|---|
| [#54930](https://github.com/tenstorrent/tt-metal/pull/54930) | Add ttnn.experimental.fft / ifft (Wormhole, fp32 + bf16, Metal 2.0) | 2026-09-07 |
| [#55660](https://github.com/tenstorrent/tt-metal/pull/55660) | Enable ttnn.experimental.fft / ifft on Blackhole | 2026-09-16 |
| [#56790](https://github.com/tenstorrent/tt-metal/pull/56790) | Round FFT bf16 writer stores to nearest-even instead of truncating (fixes [#56532](https://github.com/tenstorrent/tt-metal/issues/56532)) | 2026-09-20 |

No other FFT PR was merged:
- #44030, #48330, #48530 and #56199 were closed without merging.
- Three later repo-wide cleanup commits (#55007, #56537, #57273) touched FFT files without changing behavior.

How the findings were checked:
- **Code review.** The #54930 code was read at its merge commit `836341b1f40`, and the CI configuration at main `f8a72ce56fd`.
- **Device runs.** Every finding marked "confirmed on device" was run on a Wormhole n150.
- **What the device build contained.** The local build had the FFT code of #54930 unchanged. It did not include #55660 or #56790.
- **Isolation.** Each suspected bug ran in its own Python process, so state left over from earlier tests cannot explain the results.
- **Not tested.** Nothing was tested on Blackhole.

Code links below point to the #54930 merge commit unless they say otherwise.

## 1. Review of each PR

### #54930 (Wormhole)

#### Bugs

1. **Heap corruption crash (confirmed on device, still present on main).**
   - **Inputs that crash:** an fp32 input of shape (3, 4096), and a TILE-layout (1, 4096) input. Both abort the Python process with `malloc(): corrupted top size` or `double free or corruption`.
   - **Trigger:** any pow-2 N ≥ 4096 that fails the two-pass entry check (for example, because of the batch size or the layout) falls through to `prim::fft`.
   - **Cause:** `prim::fft` builds the Stockham twiddle table before it validates N ([fft_device_operation.cpp#L157-L165](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/device/fft_device_operation.cpp#L157-L165)).
   - **Out-of-bounds write:** the table builder writes N/2 entries into each 1024-entry row ([stockham_host.hpp#L107-L121](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/device/stockham_host.hpp#L107-L121)).
   - **Why nothing catches it:** the only size guard is a C `assert` ([stockham_host.hpp#L130-L133](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/device/stockham_host.hpp#L130-L133)), and Release builds remove it.
   - **Fix direction:** validate before building the table, and replace `assert` with `TT_FATAL`.

2. **Silent NaN results from the public helper `ttnn.experimental.complex_mul` (confirmed on device).**
   - **Steps:** call `complex_mul` with all four inputs in DRAM. Then call it again with the same tensor shapes, but with `b_real` and `b_imag` in SRAM.
   - **Result:** the second call reuses the cached program and returns NaN. With `b` in SRAM on a fresh program cache, the same call is correct.
   - **Cause:** the program cache key includes only the memory config of `a_real` ([complex_mul_device_operation.cpp#L56-L64](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/device/complex_mul_device_operation.cpp#L56-L64)). Validation does not require all four inputs to have the same memory config.
   - **Related helpers (found by reading, not run):** `ttnn.experimental.fft_radix_pass` and `ttnn.experimental.apply_twiddles` do not put the memory config of `input_imag` in their cache keys.
   - **What is not affected:** the main `fft` / `ifft` entry points. They require the real and imaginary inputs to have the same memory config ([fft.cpp#L983-L986](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/fft.cpp#L983-L986)).

3. **Wrong output shape for rank-3 or higher input (confirmed on device).**
   - **Example:** an fp32 input of shape (2, 2, 32768) returns tensors of shape (4, 32768). The values are correct.
   - **Cause:** when a row is larger than 64 KB, the two-pass path returns the 2-D result of `rebank_rm_merge` and does not restore the input shape ([fft.cpp#L300-L308](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/fft.cpp#L300-L308)). The docstring says the output has the same shape as the input.

#### Inputs that are rejected although the PR or docstring imply support

4. **A batch size that is not a power of 2 fails for every N (confirmed on device).**
   - **Observed:** B = 3 with N = 64 or N = 100 gives a TT_FATAL. With N = 4096 it gives the crash in bug 1.
   - **Documentation:** the PR lists this limit only for the pow-2 tiers. The docstring says only that "leading dims are batched".
   - **Misleading error text:** the message tells the user to route through the composite `ttnn.experimental.fft` entry point ([fft_device_operation.cpp#L49-L57](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/device/fft_device_operation.cpp#L49-L57)). That is the entry point the user already called.

5. **Non-pow-2 N is not supported for "any N" (confirmed on device).**
   - **Limit in the code:** non-pow-2 N that is not a multiple of 1024 is rejected when N is above 16384 (fp32) or 32768 (bf16) ([bluestein.cpp#L264-L272](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/bluestein.cpp#L264-L272)).
   - **Observed failures:** N = 16385, 20000 and 44100 in fp32, and N = 40000 in bf16. N = 44100 is a common audio length.
   - **Documentation:** the PR router table says "non-pow-2, any N". Neither the PR's Known Limits section nor the docstring mentions this limit.

#### Test coverage: no FFT test runs in CI

6. **The nightly FFT folder is excluded.** All three nightly "experimental" jobs (Wormhole, Blackhole, Blackhole viommu) pass `--ignore=tests/ttnn/nightly/unit_tests/operations/experimental/fft` ([ops_unit_tests.yaml#L80-L82, main](https://github.com/tenstorrent/tt-metal/blob/f8a72ce56fd5cc876b43dc6a6ce95220310f95dd/tests/pipeline_reorg/ops_unit_tests.yaml#L80-L82), [#L95](https://github.com/tenstorrent/tt-metal/blob/f8a72ce56fd5cc876b43dc6a6ce95220310f95dd/tests/pipeline_reorg/ops_unit_tests.yaml#L95), [#L129](https://github.com/tenstorrent/tt-metal/blob/f8a72ce56fd5cc876b43dc6a6ce95220310f95dd/tests/pipeline_reorg/ops_unit_tests.yaml#L129)). The PR said a dedicated job would be registered in a follow-up. It has not been added.
7. **The smoke test folder is not run either.** `tests/ttnn/unit_tests/operations/experimental/fft/` calls itself "PR-gate smoke" ([test_fft.py#L5-L7](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/tests/ttnn/unit_tests/operations/experimental/fft/test_fft.py#L5-L7)), but no workflow or pipeline file collects it. Its 30 tests pass locally.
8. **Tolerances are loose.**
   - Relative error limits are 5e-2 for bf16 and 1.5e-1 for bf16 Bluestein ([test_fft_all_n.py#L82-L89](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/tests/ttnn/nightly/unit_tests/operations/experimental/fft/test_fft_all_n.py#L82-L89)). Round-trip tests multiply these limits by 4.
   - As a result, the bf16 truncation bias that #56790 later fixed passed every test.
9. **Untested input classes.** No test covers any of these:
   - rank 3 or higher
   - a batch size that is not a power of 2
   - SRAM or sharded inputs
   - TILE layout
   - error paths

   Each of bugs 1, 3 and 4 falls in one of these gaps.
10. **Some tests fail when run together.** The nightly conftest says some multi-pass tests fail when the whole folder runs in one pytest session, and it recommends `--forked` ([conftest.py#L37-L47](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/tests/ttnn/nightly/unit_tests/operations/experimental/fft/conftest.py#L37-L47)). My guess is that this hides a real state-leak bug between calls. I did not investigate it.

#### Efficiency

11. **No PR gives performance data, and my timing shows the op is slower than one CPU thread.**

    Method:
    - Wall-clock time per call, averaged over 10 calls after 2 warm-up calls, without trace capture.
    - Input dtype fp32, compared with `torch.fft.fft` running on one CPU thread.
    - Because this is wall-clock time, it includes host dispatch time. That is the time a user sees.

    | Input (B, N) | Tier | Device | torch, 1 CPU thread | Device / CPU |
    |---|---|---|---|---|
    | (1, 1024) | Stockham | 424 µs | 71 µs | 6.0× slower |
    | (64, 1024) | Stockham | 449 µs | 372 µs | 1.2× slower |
    | (1, 4096) | two-pass | 870 µs | 74 µs | 11.8× slower |
    | (1, 65536) | two-pass | 2319 µs | 1548 µs | 1.5× slower |
    | (8, 65536) | two-pass | 17118 µs | 2876 µs | 6.0× slower |
    | (1, 1048576) | two-pass | 29150 µs | 25195 µs | 1.2× slower |
    | (1, 1000) | Bluestein | 2477 µs | 46 µs | 53.6× slower |
    | (1, 12288) | Bluestein | 3198 µs | 318 µs | 10.1× slower |

    Observations from the code review:
    - **Parallel work:** batch rows are split across up to 64 cores. A single transform with N ≤ 1024 runs on one core.
    - **Per-core work:** each core processes its rows one after another.
    - **Twiddle reads:** each core reads the twiddle row from DRAM again at every stage of every row.
    - **No overlap:** the state buffer holds one row, so DRAM reads, compute and writes do not overlap.

#### Minor issues

- **bf16 truncation bias.** bf16 outputs were truncated instead of rounded. I measured a mean magnitude bias of −0.28% for (64, 1024). #56790 fixes this.
- **`precision="fast"` has no effect.** It is accepted and documented, but no factory reads it.
- **Wrong documented gap.** The PR says pow-2 N between the twiddle-table limit and 2^20 fails with TT_FATAL. On device, N = 2^18 (fp32) and N = 2^19 (bf16) both run and give correct results. The documentation and code comment ([fft.cpp#L337-L358](https://github.com/tenstorrent/tt-metal/blob/836341b1f402b607b5dd62e8bfb100120e91229c/ttnn/cpp/ttnn/operations/experimental/fft/fft.cpp#L337-L358)) are wrong in the harmless direction.
- **Three-pass twiddle accuracy.** The three-pass tier (N > 2^20) computes twiddles with an fp32 recurrence. A host simulation of the kernel arithmetic gives about 1e-5 twiddle error, compared with about 4e-8 for a table. This was not measured on the device.
- **Pow-2 batch restrictions in the helpers.** The public helpers `ttnn.experimental.transpose_rm` and `apply_twiddles` fail with a clear TT_FATAL when the work count is not a power of 2. For example, `transpose_rm` of shape (1, 416, 32) fails. The FFT itself never produces such counts.

### #55660 (Blackhole)

- **The change is small and correct.** It changes only the architecture check in `preflight_fft_input` ([fft.cpp#L960-L967, main](https://github.com/tenstorrent/tt-metal/blob/f8a72ce56fd5cc876b43dc6a6ce95220310f95dd/ttnn/cpp/ttnn/operations/experimental/fft/fft.cpp#L960-L967)) and replaces skips with `@run_for_blackhole()` markers in the tests.
- **No CI coverage.** The Blackhole nightly jobs also ignore the FFT folder (see item 6 of #54930), so the 16 new Blackhole tests never run in CI. `test_fft_blackhole_diagnose.py` prints results but has no asserts, so it cannot fail.
- **The old Blackhole failure was never explained.** #54930 skipped the Blackhole tests because of a "NoC ordering / L1 coherence issue in batch_fft_reader.cpp". #55660 changes no kernel and says only that the tests now pass on one 11×10 machine. In the current code each core transforms whole rows and does not exchange data with other cores. My guess is that the old note described an earlier design, and that no race is hiding.
- **Validation was manual only,** on one Blackhole machine.

### #56790 (bf16 rounding; automated PR)

- **The rounding helper is correct.**
  - Round-to-nearest-even gives the expected result for normal values, overflow to Inf, NaN and negative values.
  - It replaces the truncating conversion in all three bf16 writer kernels, and no truncating conversion is left in the FFT kernels.
  - A reviewer ran a before/after comparison on Wormhole. In that comparison the bias went away and the Stockham output matched host rounding bit for bit.
- **No test was added.** Reverting #56790 would still pass every existing test, because the truncation error (4e-3 to 2e-2) is below the bf16 limits of 5e-2 and 1.5e-1.
- **Part of #56532 is still open but untracked.** #56532 is closed as completed, but its second proposal was not done: keep intermediate results in fp32 when the caller's dtype is bf16. The two-pass and Bluestein tiers still store bf16 between steps. In the reviewer's measurement, bf16 results still differed from correctly rounded bf16 output by 2.5e-3 (two-pass, N = 4096) and 4.4e-3 (Bluestein, N = 97).

## 2. Gaps against the issue

| Asked for in #21412 | Status |
|---|---|
| FFT and inverse FFT | Done: `ttnn.experimental.fft` and `ttnn.experimental.ifft` |
| fp32 and bf16 | Done. bf16 accuracy in the two-pass and Bluestein tiers is limited by bf16 intermediate results |
| Multi-core | Partly. Batch rows and the multi-pass tiers use up to 64 cores. One transform with N ≤ 1024 runs on one core |
| Efficient | Not shown by any PR. In the timing above the op is slower than one CPU thread at every tested size |

Other gaps:
- **Items from the parent issue #11835.** The parent was closed as a duplicate of #21412, so its list arguably carries over:
  - Complex input: done.
  - 2D and 3D FFT: not done. Only the last dimension is transformed, and there is no `dim` argument.
- **Input coverage.** Bugs 4 and 5 of #54930 (non-pow-2 batch, large non-pow-2 N), TILE layout and sharded inputs are not supported. Bug 1 should at least become a clear error.
- **API compared with `torch.fft`.** There is no `n` (pad or trim) argument, no `norm` argument, and no `rfft` or `irfft`. The issue did not ask for these explicitly.
- **Namespace.** The op is still in `ttnn.experimental`. The PR leaves the move to `ttnn.fft` for a follow-up.
- **CI.** No FFT test runs in CI on Wormhole or Blackhole.

## Reproducing the confirmed findings

Each snippet runs on a Wormhole device, with `device = ttnn.open_device(device_id=0)` and `import torch, ttnn`.

```python
def to_dev(x, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=device, memory_config=mem)

# Bug 1: heap corruption, aborts the process (each line on its own)
ttnn.experimental.fft(to_dev(torch.randn(3, 4096)))
ttnn.experimental.fft(to_dev(torch.randn(1, 4096), layout=ttnn.TILE_LAYOUT))

# Bug 2: NaN on a program cache hit
a_r, a_i, b_r, b_i = (torch.randn(64, 256) for _ in range(4))
ttnn.experimental.complex_mul(to_dev(a_r), to_dev(a_i), to_dev(b_r), to_dev(b_i))
o_r, o_i = ttnn.experimental.complex_mul(
    to_dev(a_r), to_dev(a_i), to_dev(b_r, ttnn.L1_MEMORY_CONFIG), to_dev(b_i, ttnn.L1_MEMORY_CONFIG))
print(ttnn.to_torch(o_r))  # NaN

# Bug 3: output shape (4, 32768) instead of (2, 2, 32768)
re, im = ttnn.experimental.fft(to_dev(torch.randn(2, 2, 32768)))
print(re.shape)

# Bug 4: TT_FATAL for batch 3
ttnn.experimental.fft(to_dev(torch.randn(3, 64)))

# Bug 5: TT_FATAL for non-pow-2 N above 16384 that is not a multiple of 1024
ttnn.experimental.fft(to_dev(torch.randn(1, 44100)))
```
