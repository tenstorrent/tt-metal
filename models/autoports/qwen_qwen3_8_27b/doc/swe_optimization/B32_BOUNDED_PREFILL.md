# B32 bounded physical prefill

Run40687173740 attempted native262144/B32/shared2048 with the qualified combined6faa07fb image. All64layers and grouped64/masked48/flat64 loaded; no task agent started. At1024-token physical-prefill warmup, reduce_scatter allocation100270080bytes failed: each bank needed12533760bytes, largest freebankblock9819776bytes, allocated4227184960/free28379200bytes per bank. Pool sizing arithmetic alone did not guarantee transient/fragmented fit.

Bound residentB32 physical prefill chunks to512, still split at scheduler checkpoint boundaries. Logical262144 context,2048 checkpoint spacing, precision, cache backing budget and generation policy are unchanged. Existing256/512 serving still takes min(chunk,checkpoint), so its actual chunk size is unchanged. The2048 pool must be qualified again; no scored or throughput claim follows from this source preparation.

Test combined full64 B32 state/logit continuation against baseline with512 chunks, then actual shared2048 startup/known-task smoke. Keep failed run and its cleanup receipt separate. Runtime image must bind this exact source; no silent edit of an already qualified image.
