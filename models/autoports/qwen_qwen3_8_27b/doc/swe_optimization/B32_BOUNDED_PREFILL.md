# B32 bounded physical prefill

Run40687173740 attempted native262144/B32/shared2048 with the qualified combined6faa07fb image. All64layers and grouped64/masked48/flat64 loaded; no task agent started. At1024-token physical-prefill warmup, reduce_scatter allocation100270080bytes failed: each bank needed12533760bytes, largest freebankblock9819776bytes, allocated4227184960/free28379200bytes per bank. Pool sizing arithmetic alone did not guarantee transient/fragmented fit.

Bound residentB32 physical prefill chunks to512, still split at scheduler checkpoint boundaries. Logical262144 context,2048 checkpoint spacing, precision, cache backing budget and generation policy are unchanged. Existing256/512 serving still takes min(chunk,checkpoint), so its actual chunk size is unchanged. The2048 pool must be qualified again; no scored or throughput claim follows from this source preparation.

Test combined full64 B32 state/logit continuation against baseline with512 chunks, then actual shared2048 startup/known-task smoke. Keep failed run and its cleanup receipt separate. Runtime image must bind this exact source; no silent edit of an already qualified image.

Validation28October: local paired all64B32/512 fixture passes exact outputs/all512cache-statehashes, ragged reset and three continuation state/logit checks. One warmed fixed-work repeat9.084→13.220tokens/s/user and4340.6→3836.3msTTFT agrees with prior individual evidence; no repeatability or scored speedclaim. Actual-pool smoke40691014802 passes knownAstropy14309,10/10complete2485outputtokens, reward1, no exception, actual isolation/settled cleanup and exact stable exports. The771-ID2048 pool prepares with512 physicalchunks and2,598,716,928freebytes/device afterward; sampledwarmup minimum1,581,904,384bytes is not a continuouspeak. See compact receipts.

Paired scored32-task B32/C32 arms40694868501/40694874230 are now running with identical task/protocol/image settings except phase and optimization flags. No score or full-suite forecast is inferred from dispatch. The predeclaredfirst32SWE100 contains8Astropy+24Django, limiting population/runtime extrapolation. Native262144 remains unchanged.
