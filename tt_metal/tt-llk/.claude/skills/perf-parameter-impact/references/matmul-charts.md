# Default absolute-cycle charts for matmul-style reports

Read with `perf-parameter-impact`, "Canvas report".

For matmul-style reports with format, fidelity, and destination-accumulation
axes, use this design unless the user requests another:

1. Add clickable fidelity options for values the **current sweep actually
   emits**. Quasar matmul keeps LoFi–HiFi4 for Float16 and LoFi-only for
   Float16_b, Int8, and MX; do not show empty HiFi panels as if they were
   measured.
2. Under the selected fidelity, show absolute cycles by composite
   input/register mode, output format, and geometry/destination accumulation.
   Geometry labels should name dest-full **tall vs wide** and `kt=1` vs
   `kt=4`, not the old 12/9 dest-fill factorization set.
3. Before placing `dest_acc=No` and `Yes` bars side by side, audit whether both
   arms contain identical geometries and other relevant configurations.
4. If common geometry exists, restrict the split bars to that common support,
   state the reduced coverage, and use the same color for each metric pair with
   `dest_acc` distinguished by opacity or another restrained treatment.
5. If no common geometry exists, as in the current Quasar matmul sweep (8-tile
   vs 4-tile dest-full), do not split input/register or output-format
   categories into adjacent `dest_acc` bars. Either omit that split in favor of
   the geometry chart or use separate panels labeled **observed bundled
   population: dest_acc + geometry**. Explicitly state that gaps between
   panels are descriptive and are not a `dest_acc` effect.
6. In the geometry chart, encode both geometry and destination accumulation in
   the x-axis label. Keep four metric bars per composite category and do not
   split them again.
7. Add a red min–max whisker to every absolute-cycle bar. Calculate the minimum
   and maximum from exactly the rows summarized by that bar.
8. Call this an **observed min–max range**, not variance. It is sensitive to
   extreme configurations and is not a repeated-run noise estimate.
9. Hover must show category, metric, `dest_acc` when applicable, geometric
   mean, minimum, maximum, and max-minus-min range.
10. State sample counts separately for every displayed subgroup and composite
    geometry/destination category.

Use a custom themed SVG when the built-in chart cannot draw per-bar min–max
whiskers. Use host-theme colors; never hardcode red or other colors.
