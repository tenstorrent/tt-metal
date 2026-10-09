# Further precision controls: prepared, not queued

At 10:46:40 UTC the unchanged all-BFP8 GPQA service was live with 166 correct
out of 188 completed and zero truncations. With only ten questions left, the
maximum possible final score was 176/198, below the unchanged 177/198 gate.
The user then explicitly requested finishing and capturing the full run;
no running experiment or frozen source was changed and no new hardware
diagnostic was launched.

Two optional configurations and diagnostic policy validation are prepared:

1. BFP8 decoder projections with HiFi4, retaining the BFP8/HiFi2 head. This
   separates projection compute fidelity from weight quantization relative
   to the completed BFP8/HiFi2 layer reference.
2. BF16 decoder projections with HiFi4, retaining the same head. Comparing
   against the first control separates decoder weight precision at matched
   compute fidelity. The isolated head reference still measures the remaining
   BFP8 head error independently.

Both retain BF16 activations/residuals, FP32 recurrent state, BFP8 KV, native
recurrence and the accurate full-tile attention path. The validator rejects
mixed projection modes, changed head/state/cache settings, and unsupported
weight/fidelity combinations. No default precision or production kernel code
changed. These policies are not added to an image or serving qualification.

The existing eager B1, short-public-prompt, eight-position reference probe can
execute them after separate admission/resource checks. Neither its numerical
result nor memory admission is proven. They cannot inherit a G0 or GPQA pass,
and no throughput or serving-capacity claim is made for BF16.

Twenty-five local tests passed across policy validation and reference metrics,
including new rejection cases. The host-only run extracted the repository's
`expect_error` fixture to avoid importing unavailable device dependencies;
the retained fixture and exact command document that scope. Pytest emitted
a harmless cache-write warning because `-c /dev/null` selected `/dev` as its
cache root; assertions ran and the JUnit receipt was saved in the task directory.
Native imports/hardware execution remain unrun.
