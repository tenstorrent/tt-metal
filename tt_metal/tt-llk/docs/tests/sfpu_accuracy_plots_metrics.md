# SFPU Accuracy Plots & Metrics

This page documents how we visualize and quantify SFPU accuracy, and summarizes results for a set of SFPU instructions:

- **reciprocal**, **reciprocal (stress)**
- **atanh**, **atanh (stress)**
- **elu**, **gelu**, **celu**, **silu**, **hardsigmoid**
- **log**, **log1p**, **exp**

The same structure can be reused for other SFPU ops.

## Plot Layout

Each SFPU test produces one figure with five vertically stacked plots on the left and a summary panel on the right:

1. **First plot – function vs x**

    - Blue line and dots: **golden** reference computed in software (PyTorch).
    - Orange dashed line and hollow markers: result produced by the SFPU hardware under test.
    - Gray vertical bands: x-ranges that are not sampled in this test (excluded by the input spec).
    - Red vertical bands: x-ranges where the math function is **undefined**.
    - Triangle markers at the top:
        - Red: both golden and HW are non-finite (`inf` or `nan`).
        - Orange: HW-only non-finite.
        - Blue: golden-only non-finite.
    - For some ops (e.g. `atanh` in stress mode), thin red dashed vertical lines show known **asymptotes** (`x = ±1`).

2. **Second plot – signed error in ULPs**

    - A stem plot of signed error measured in ULPs:

          signed_ulp_error(x) = (hw(x) - golden(x)) / local_ulp(golden(x))

    - Positive stems mean HW is above golden.
    - Negative stems mean HW is below golden.
    - We compute this using **true local ULP** from the golden output value, based on the spacing to the next representable float.
    - Dashed horizontal reference lines are shown at adaptive thresholds such as `±1, ±3, ±10, ±100 ULP` when those ranges are reached.

3. **Third plot – relative error**

    - Scatter plot of

          rel_error(x) = |hw(x) - golden(x)| / |golden(x)|

      for points where `golden(x) ≠ 0`.
    - Y-axis is **log scale**.
    - Adaptive horizontal reference lines (only drawn when the data actually reaches the previous threshold):
        - Green dashed: `1 ULP`.
        - Orange dashed: `3 ULP`.
        - Red dashed: `10 ULP`.
        - Purple dashed: `100 ULP`.
    - Points below the `1-ULP` line are effectively within rounding noise of the format.

4. **Fourth plot – CDF of absolute ULP error**

    - A CDF of **|ULP error|** on a **log x-axis**.
    - This directly answers: **what fraction of points are within** `N ULP`**?**
    - Vertical reference lines are shown at adaptive thresholds such as `1, 3, 10, 100 ULP` when reached.
    - The plot is shown with a **zoomed y-axis** so the interesting part of the curve is easier to read.
    - We also explicitly show the fraction of **exact matches** (`0 ULP`), since `0` cannot be displayed directly on a log x-axis.

5. **Fifth plot – per-bin |ULP| percentile**

    - The input x-axis is divided into **32 uniform bins**. For each bin, we compute and plot:
        - **p50** (green circles) – median `|ULP|` in the bin
        - **p95** (orange squares) – 95th percentile
        - **p99** (red upward triangles) – 99th percentile
        - **max** (purple downward triangles) – worst observed point in the bin
    - The y-axis uses a **symlog scale**, so both very small ULP values and large multi-ULP outliers remain visible in the same plot.
    - A **faint blue** background curve, read using the right-hand y-axis (`0–1`), shows the fraction of points in each bin where hardware matched golden exactly (`0 ULP`).
    - To avoid misleading spikes from under-sampled regions, percentile lines are shown **only for bins with at least 8 samples**. Bins with fewer than 8 points are hidden, so a single outlier cannot dominate `p99` or `max`. A small note in the corner reports how many bins were hidden.
    - **Interpretation caveat**
        - ULP magnitude is normalized by the **local spacing** of the output format. For functions whose output approaches zero (for example, `log(x)` near `x = 1`), the local ULP becomes extremely small. In that case, even a small absolute error can appear as a very large ULP count.
        - As a result, large spikes in this plot near such regions are often a **normalization artifact**, not necessarily a sign of catastrophic absolute error.

6. **Right-hand panel – analysis summary**

    - One main summary box plus a separate box for non-finite details, showing:
        - input x-range and number of points,
        - non-finite outputs (`inf`/`nan`),
        - absolute and relative errors,
        - ULP statistics (for `float16`, `bfloat16`, and `float32`),
        - top worst-case ULP points,
        - a short table of non-finite inputs and outputs,
        - monotonicity check for ops that should be monotonic on their domain.

---

## Terminology and Metrics

### Floating-point and ULPs

- **ULP (Unit in the Last Place)**
  ULP is the distance between two adjacent representable floating-point numbers at a given magnitude.

    Examples near 1.0:

    - **bfloat16 (Float16_b)**
      `1 ULP ≈ 2^-7 ≈ 7.81e-3`
    - **float16**
      `1 ULP ≈ 2^-10 ≈ 9.77e-4`
    - **float32**
      `1 ULP ≈ 2^-23 ≈ 1.19e-7`

    If HW differs from golden by 1 ULP, it's exactly one representable step away.

- **Relative error**
  For each x where `golden(x) ≠ 0` we define:

      relative_error = |hw - golden| / |golden|

- **ULP error (error in ULP units)**
  For the signed-ULP **plot** and the CDF, we measure error in units of the **true local ULP** of the golden output.

    The local ULP is the distance from a value to the next representable floating-point value in the target format.
    For each output value **y**, we define:

      local_ulp(y) = nextafter(|y|, +∞) - |y|

    Then the signed ULP error is:

      signed_ulp_error(x) = (hw(x) - golden(x)) / local_ulp(golden(x))

    This tells us how many local floating-point steps the hardware result is away from the golden result.

    - A value near `0` means HW matches golden closely.
    - A **positive** value means HW is above golden.
    - A **negative** value means HW is below golden.
    - The **magnitude** tells us how many ULPs apart they are.

### Non-finite outputs

A point is **non-finite** if either golden or HW produces `inf` or `nan`.

- We **exclude** non-finite points from error and ULP statistics (to avoid `inf - inf` and `nan` corrupting the metrics).
- We still **execute** them on hardware, and:
    - Mark them on the top plot (triangle markers).
    - Summarize them in the analysis panel.

In the summary you will see:

- `Non-finite outputs (inf/nan): K`
  Total number of points where at least one side is non-finite.
- Breakdown:
    - `both non-finite: […]`
    - `HW-only: […]`
    - `golden-only: […]`
- Per-side classification:
    - `HW inf: […]`, `HW nan: […]`
    - `golden inf: […]`, `golden nan: […]`
- Optional detail table (up to 10 points):

    ```
    x              golden            hw          type
    0.000000e+00   inf               inf         inf
    ...
    ```

    `type` indicates whether the non-finite was `inf`, `nan`, or another non-finite category.

### Error statistics

For all **finite** points:

- **Max absolute error**

      max_abs_error = max_x |hw(x) - golden(x)|

  This is the largest absolute difference between HW and golden over all sampled x.
- **Mean absolute error**

      mean_abs_error = mean_x |hw(x) - golden(x)|

  This is the average absolute difference between HW and golden.
- **Max / median relative error**
    - **Max relative error**:

          max_rel_error = max_x rel_error(x)

    - **Median relative error**:

          median_rel_error = median(rel_error(x))

      (50% of the points have relative error below this value.)

### Bits of precision

We approximate the effective number of bits of precision as:

    bits(x) = -log2(relative_error(x))

and then summarize:

- **Bits of precision (worst)** – smallest `bits(x)` over all finite points
- **Bits of precision (median)** – median of `bits(x)` over all finite points

### ULP statistics

When the data format is `Float16`, `Float16_b`, or `Float32`, we compute:

- `ULP unit (eps)` – the relative epsilon used for `ulp_err`.
- `Mean ULP` – average of `ulp_err`.
- `p99 ULP` – 99th percentile of `ulp_err`.
- `Max ULP` – maximum observed `ulp_err`.
- `Points > 3 ULP`, `Points > 10 ULP`, `Points > 100 ULP` – number of outliers beyond 3, 10 or 100 ULP.

### Top ULP offenders

We include a small table of the worst x-points in ULP units:

```
x              golden            hw            ulp
...            ...               ...           ...
```

This helps us see where the approximation is weakest.

### Monotonicity

Monotonicity is different from ULP accuracy.
An operation can be very close to golden in ULP terms, but still locally break the expected ordering of outputs.

For operations that should be monotonic on their valid domain, we sort inputs in ascending order and compare each pair of neighboring hardware outputs:

- for **increasing** ops, we expect `y_hw[i+1] >= y_hw[i]`
- for **decreasing** ops, we expect `y_hw[i+1] <= y_hw[i]`

The comparison is **non-strict**, so equal neighboring outputs are allowed. This is expected in low-precision formats such as `bfloat16`, where multiple nearby inputs can quantize to the same output value.

For operations with discontinuities (for example reciprocal around `x = 0`), we check each valid input interval separately and never compare points across the gap.

The summary reports:

- **Pairs checked** – number of neighboring output pairs tested
- **Violations** – number of ordering violations
- **Rate** – percentage of violating pairs
- **Worst |Δy|** – largest ordering violation by output difference
- **Top 5 violations** – worst offending pairs, shown as `x1 / x2 / y1_hw / y2_hw / |Δy|`

This section is shown only for operations that are expected to be monotonic.

---

## Shaded Regions and Reference Lines

- **Gray bands – excluded intervals**
  These are x-ranges that we did not test for this run (for example, regions around singularities when we only want a "normal" test).
- **Red bands – undefined domain (optional)**
  These show where the math function itself is undefined (e.g. `x ≤ 0` for `log`). This is for context and does not necessarily mean we sampled x in that region.
- **Asymptote lines (Atanh)**
  For `atanh` stress plots we draw red dashed vertical lines at `x = −1` and `x = +1` to show where the function has vertical asymptotes.

---

## Per-Operation Summaries

Below we list short summaries for each sfpu operation.

### Reciprocal

- **Normal region**
- **Stress region near 0**

---

### Atanh

- **ULP sweep** (testing all representable values for `Float16_b` format on specified domain)
- **Stress region including asymptotes**

---

### ELU

---

### GELU

---

### CELU

---

### SILU

---

### Hardsigmoid

---

### Log

---

### Log1p

### EXP

---

## Summary

With this setup:

- Plots show **shape** (golden vs HW),
- Error panels show **where and how** HW differs,
- ULP statistics quantify **how close** we are to format-limited rounding,
- Non-finite handling makes **edge cases** (e.g. `1/0`, `atanh(±1)`) explicit.
