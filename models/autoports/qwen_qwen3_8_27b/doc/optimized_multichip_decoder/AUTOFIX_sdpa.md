# SDPA mapped-capacity guard

AutoDebug source diagnosis: `AUTODEBUG_sdpa.md`. Requested chunk512 reads 144 page IDs for decode positions4096–4223 while the original table owns136 IDs. Blackhole's transfer padding is not mapped page storage. The artifact records NaN PCC, not proven NaN tensor elements.

Countertest: `sdpa_pad_aligned512_l3.json` (see matching candidate/results files for exact name) explicitly allocates4608 physical tokens and144 owned mappings. The true512 kernel passes all128 outputs, minimum PCC0.99993949, median warmed traced host latency368.405us; the256 control is364.54us. This rejects512 on measured performance rather than interpreting invalid page reads as precision loss. The1024 FP16/padded retry still exceeds physical L1 (1,690,496 versus1,572,864 bytes).

Correction: resolve requested chunk downward to a divisor of explicitly mapped capacity, from table shape metadata before dispatch. This preserves all caller mappings, avoids host reads, and works with captured page-table updates. It never invents ownership or aliases another user's page. This is a safe capacity guard, not evidence for512 when it resolves to256. The runner records effective chunks explicitly. Tables must contain valid owned IDs for their mapped extent, as required by the existing external table contract.

Pending verification: original4352 capacity should resolve512→256 and pass; explicit4608 capacity should remain512 and pass. Ragged/remapped trace and native-context lower-fidelity checks follow before selecting defaults.

Guard verification completed: `sdpa_guard_guard512_l3.json` retains4352-token physical capacity, explicitly records effective256, passes all128 decodes at minimumPCC0.99994349 and364.068us median traced host latency. `sdpa_guard_owned512_l3.json` retains4608 physical tokens/effective512, passes minimumPCC0.99993949 and369.688us. These same-harness controls confirm the guard and preserve the true512 performance rejection. Page remap/ragged validation is queued separately.

Ragged/remap verification passed: sdpa_guard_remap_guard512_l3.json, B8/S129/8 decode steps, minimum PCC 0.999974731, page_remap_exact=True, effective chunks=[256]. This exercises device page-table updates under a live trace.
