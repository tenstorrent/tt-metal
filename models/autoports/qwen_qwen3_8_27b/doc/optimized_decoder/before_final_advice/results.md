## Final measurements

All headline device values use4096+128,B1,one request. Every layer kind is reported separately. `performance.json` contains formulas, useful FLOPs/estimated bytes, theoretical peaks and exact source references. Lower precision explains the small PCC change; all results exceed the functional .995 bar.

| Layer kind | Fused prefill us | Final prefill us | Speedup | Fused decode us | Final decode us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|linear_attention|82260.06|16340.53|5.03x|2326.36|816.68|2.85x|
|full_attention|92861.61|18038.95|5.15x|2193.56|705.88|3.11x|

| Layer kind | Fused / final prefill PCC | Fused / final initial decode PCC | Fused / final128th decode PCC |
| --- | ---: | ---: | ---: |
|linear_attention|0.99999675 / 0.99980716|0.99999921 / 0.99981594|0.99999724 / 0.99982184|
|full_attention|0.99999493 / 0.99986445|0.99999938 / 0.99959877|0.99999933 / 0.99963450|

| Layer kind | Ideal compute prefill us | Actual device prefill us | Host prefill us | Prefill chip FLOPs % | Ideal DRAM decode us | Actual device decode us | Host last decode us | Decode DRAM % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|linear_attention|4766.83|16340.53|17150.44|29.17|435.93|816.68|839.18|53.38|
|full_attention|4909.23|18038.95|18744.77|27.21|426.96|705.88|721.95|60.49|

The ideal times are lower bounds, not mixed-kernel attainable ceilings. DRAM estimates count physical BFP4 weight tiles/headers/padding and state/KV traffic, omitting spills and rereads. Device gaps remain inside the denominator. Host values here are from the same profiler-enabled run; profiler-off screens below provide selection comparisons. Trace replay removes per-op host dispatch; remaining firmware gaps and single replay synchronization are included, not subtracted.

| Layer | Length | Fused prefill / decode host us | Final prefill / decode host us |
| --- | ---: | ---: | ---: |
|linear_attention|513|12892.45 / 2319.32|3004.33 / 797.36|
|linear_attention|4096|See device baseline above|16984.06 / 796.95|
|full_attention|513|12250.38 / 2148.65|2776.38 / 670.91|
|full_attention|4096|See device baseline above|18805.19 / 689.27|

These profiler-off decode screens use100 static-position replays after one initial restore for candidate comparison. The headline uses the valid advancing128-step workload. Larger-batch values below use its final valid step, not the static-position diagnostic loop.

| Layer kind | Batch | Prefill PCC | Final128th decode PCC | Final valid-step host us |
| --- | ---: | ---: | ---: | ---: |
|linear_attention|8|0.99981065|0.99984509|2301.86|
|linear_attention|16|0.99980923|0.99983479|3827.44|
|linear_attention|32|0.99980924|0.99982258|6965.34|
|full_attention|8|0.99987515|0.99984434|1316.90|
|full_attention|16|0.99987140|0.99976380|1988.58|
|full_attention|32|0.99987388|0.99982765|3288.06|

## Dominant matmul search

Role shares below are kernel time divided by the final complete layer device window; layout and internal gaps are outside the numerator but inside the denominator. Search latencies are whole-layer traced host screens at4096. Exact per-role configs/PCC and other precision/topology candidates are in `screen_results.csv` and matching JSON/policy files. Rows use the same final BFP4/LoFi decode topology; only the named role changes.

| Kind | Role | Final layer time % | Config candidate | Traced decode us | Minimum output PCC | Decision |
| --- | --- | ---: | --- | ---: | ---: | --- |
|linear_attention|mix|12.97|control|797.33|0.99982408|Keep|
|linear_attention|mix|12.97|mix40k4|816.27|0.99982040|Reject: slower|
|linear_attention|out|6.87|control|797.33|0.99982408|Keep|
|linear_attention|out|6.87|out32k6|802.33|0.99979927|Reject: slower|
|linear_attention|gateup|26.48|control|797.33|0.99982408|Keep|
|linear_attention|gateup|26.48|gateup40k4|840.73|0.99982182|Reject: slower|
|linear_attention|gateup|26.48|gateup16k10|—|—|Reject: L1 capacity; legal smaller alternatives measured|
|linear_attention|down|12.82|control|797.33|0.99982408|Keep|
|linear_attention|down|12.82|down34k16|850.05|0.99981934|Reject: slower|
|linear_attention|down|12.82|down17k32|852.93|0.99981047|Reject: slower|
|linear_attention|down|12.82|down8k68|—|—|Reject: L1 capacity; legal smaller alternatives measured|
|full_attention|mix|13.12|control|690.18|0.99954898|Keep|
|full_attention|mix|13.12|mix40k4|707.62|0.99956438|Reject: slower|
|full_attention|out|7.85|control|690.18|0.99954898|Keep|
|full_attention|out|7.85|out32k6|695.45|0.99955198|Reject: slower|
|full_attention|gateup|30.70|control|690.18|0.99954898|Keep|
|full_attention|gateup|30.70|gateup40k4|733.08|0.99954952|Reject: slower|
|full_attention|gateup|30.70|gateup16k10|—|—|Reject: L1 capacity; legal smaller alternatives measured|
|full_attention|down|14.94|control|690.18|0.99954898|Keep|
|full_attention|down|14.94|down34k16|744.15|0.99955263|Reject: slower|
|full_attention|down|14.94|down17k32|748.12|0.99955699|Reject: slower|
|full_attention|down|14.94|down8k68|—|—|Reject: L1 capacity; legal smaller alternatives measured|

Final mix/gateup remain K2 on80 input-storage cores: two tiles/shard is the largest divisor. The crossed16/32/40/80-core1/2/3-reader search and isolated40/K4,16/K10 trials earn this choice. Output64/K3 beats32/K6; down68/K8 beats34/K16 and17/K32;8/K68 fails by12928 L1 bytes. The DRAM factory chooses output subblocks internally; its public program config exposes no override. Prefill has explicit subblock height2 where legal, wider K10/K32, and11x10 grids.

Final dtype is also verified directly above from measured CSV columns. `readers_final_l0.device.json` and `readers_final_l3.device.json` recheck one/two/three readers per role on the selected policy, in alternating order, with complete role envelopes and physical byte estimates.
