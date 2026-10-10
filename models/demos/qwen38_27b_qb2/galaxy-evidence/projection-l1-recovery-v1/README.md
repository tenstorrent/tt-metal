# Projection sweep L1 failure and bounded recovery

At 21:40:37 UTC on October 10 the compact-L1 B16 MLP-down candidate with two
readers, eight activation cores and K-block 34 failed the program's L1 allocation
check. Static dataflow buffers ended at byte 1504000 while a live L1 allocation
began at 1441024. This was a rejected experimental geometry, not an accuracy
failure or an observed device hang. The hardware receipt records clean teardown;
the controller records failure. All original logs and receipts are preserved.

The preceding B16 compact output-projection bracket completed nine candidate
comparisons. Every comparison passed its numerical and changed-input checks,
but all were slower than the 60.83-us baseline, including layout and collective
cost. None is promoted. The incomplete down bracket has no accepted speedup.

The next frozen sweep defers both block-34 down variants: two-reader was observed
to fail; three-reader is untested and deferred conservatively. This leaves
62 cases and 50 candidate comparisons with the same accuracy gates, input
boundaries and before/after controls. The model and precision source is unchanged.
CPU preflight of this frozen experiment source passed **572 tests and 73 subtests,
one skipped**. This is the older sweep snapshot's suite, not the larger current
branch suite.

Five replacement user services launched after verifying exact original unit
identities, empty control groups, manifests and dependency-only failures for
unstarted followers. The new queue is projection sweep v7, prefill attention v4,
compact long-horizon v4, resident GDN v2, then compact gates v2. It preserves the
existing hardware lock and resource/time limits and survives client disconnects,
not reboot. At the captured 21:47:31 UTC snapshot all five units were active;
this launch receipt is not proof that their hardware tests passed.

The expected resident-state and compact-gate gains remain approximately
0.5-1.5 ms each at full-model scale, subject to hardware measurement and overlapping
savings. No additional model TSU is claimed. The output-projection null result
reduces confidence in the original 1.5-3.5-ms combined projection estimate.
