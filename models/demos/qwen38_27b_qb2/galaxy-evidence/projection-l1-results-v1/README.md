# Completed projection sweep and prefill baseline failure

Projection v7 completed at 21:47:45 UTC on October 10 with clean hardware teardown:
62 cases and 50 comparisons. All 25 B16 comparisons qualified numerically and
by matched controls, but every candidate was slower than its baseline. Of 25 B32
comparisons, 13 qualified and were slower; 12 output-projection comparisons
failed the numerical gate. No faster configuration is promoted, and the earlier
estimated 1.5-3.5-ms projection gain was not realized by this knob sweep.
The compact qualified model remains unchanged at 20.035 native TSU at B16/32K.
Raw data is retained as `receipts/projection-sweep-v7/projections.json.gz`.

The independent prefill-attention experiment then failed on the **before** arm
for batch 3, start position 32, chunk length 65. Its selected-row dense reference
check reported per-user relative RMS of 2.25-2.40%, with PCC 0.999736-0.999795.
The new batched arm had not executed for that case. This is not evidence that
batching introduced the error; reference construction, existing prefill numerics
and the ragged/prefix case need investigation. Hardware teardown completed.

The earlier batch-2/128-token case had bit-identical before/candidate/after outputs
and correct cache hashes, but its before/after wall timing drifted by 159%, so
it provides no qualified speedup either. Do not advertise its raw timing ratio.

The three recurrence followers exited without starting hardware after the prefill
failure. They were subsequently detached from that independent experiment and
requeued behind the completed projection sweep, preserving their exact source
and numerical gates. See [persistent queue receipts](../gdn-independent-queue-v1/README.md).
