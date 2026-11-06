# Phase archive runtime benchmark

Passed: eight exact4096input/128output B1/C1 headline requests and eight unchanged stage11 TTI nominal128/128/C1 matrix requests, plus original one-request headline warmup. All client commands exited0. Headline TTFT **753.760876ms**, TPOT **28.311765ms**, decode **35.321005tokens/s/user**; client elapsed34.797407s. TTI matrix elapsed27.097935s, TTFT113.655533ms. Actual native matrix prompt lengths141/142 reflect stock chat serialization; nominal128matrix remains unchanged.

All13runtime source hashes match publication `ab2730c27e557b09b2eafe3494e47790adc210aa`; all15 launch snapshot hashes remain unchanged. Old rootHEAD is not the candidate binding. Phase identity confirms full60, B1, devices0–3/context262144. Actual server summary preserves native parsers, pinned842da revision, APC/async andtop_k32.

Complete archive71frames contains4370events:2185unique matching submit/completionpairs, contiguous IDs1–2185, zero duplicates/gaps.2159decode events retain physical rows;26prefillsteps split17device/9hostcompatibility. Final pending0, no errors/loss/truncation; observed captures3 throughout. Last snapshot buffered66 means66events included before export deletes them from memory; they are present on disk, not missing. Raw phase/retained values remain private.

Actual queue **0.076054659deviceh (273.796773s)** versus0.096678908hforecast, within0.18hbox. Idle0/0 and owners empty. Earlier sameprotocol TTFT753.316303ms/decode35.318112t/s provides descriptive context only; no causal repair-speed/overhead claim. Device time/roofline fields remain unknown because no profiling occurred. No score or release-hours saving claimed.
