# r04-b03-a02 result: 1.6119 (ok). +1.1% vs the parent r04-b03-a01 (1.5945), but most of that is the probe being compiled out. Against the probe-free root r03-b02-a03 (1.6103) it is +0.1%, which is neutral. The down phase got much shorter where its chain was latency-bound, and the saving was paid back in gate/up on the DRAM-bound shapes

## What happened vs expected
- All 9 shapes are valid on the first and only run. PCC, max_abs and mean_abs are bit-identical to the parent, root and baseline.
- There was no hang and no JIT error. No program-size overflow either: the parent's probe was compiled out (`RE_PROBE`), and the reader grew by about 0.2 KB of source-equivalent.
- The polling bookkeeping is correct: binary valid semaphores, one outstanding ack per stream, matching write pointers.

Chip µs. "root" is r03-b02-a03, the same code without the probe and with the lock-step down loop.

| shape | root | parent | this | vs root |
|---|---|---|---|---|
| dsv3 | 514.0 | 519.7 | 520.0 | +1.2% |
| kimi-k2.7 | 729.5 | 732.6 | 744.8 | **+2.1%** |
| kimi-k3 | 1365.8 | 1359.7 | 1314.6 | **-3.7%** |
| glm | 451.2 | 455.6 | 452.1 | +0.2% |
| mistral | 185.9 | 194.9 | 183.0 | -1.6% |
| dsv4-flash | 316.9 | 316.9 | 309.3 | **-2.4%** |
| dsv4-pro | 1070.9 | 1069.7 | 1087.8 | **+1.6%** |
| minimax-m3 | 365.5 | 370.8 | 368.8 | +0.9% |
| gpt-oss | 261.8 | 263.9 | 264.4 | +1.0% |

I expected DN to drop by 5-15% where the activated wait was large, plus a geomean gain of +0.5..+2%. DN did drop on most shapes, but the geomean is flat.

## Why (profiler evidence)
Data: `/tmp/r04b03a01/probe.py` and `ends.py` on reports/r04-b03-a02 vs reports/r04-b03-a01. These are UNPACK-thread windows per core per call, as means over the 10 measured calls.

| shape | DN window µs (parent → this) | DN act wait % | DN down-weight wait % | GU window µs | mean core end µs |
|---|---|---|---|---|---|
| dsv3 | 148.6 → 142.3 | 31.9 → 23.8 | 6.3 → 11.9 | 337 → 339 | 495.9 → 489.7 |
| kimi-k2.7 | 223.2 → 217.5 | 46.8 → 33.3 | 11.3 → 23.4 | **485 → 503** | **720.8 → 731.0** |
| kimi-k3 | **460.7 → 367.8 (-20%)** | 57.5 → 23.7 | 9.3 → 34.8 | **767 → 834** | 1252.4 → 1225.6 |
| glm | 130.4 → 125.5 | 33.3 → 24.3 | 6.4 → 12.9 | 291 → 296 | 431.1 → 430.5 |
| mistral | **49.8 → 43.2 (-13%)** | 34.3 → 22.9 | 3.6 → 4.9 | 111 → 113 | 167.7 → 160.9 |
| dsv4-flash | **97.2 → 82.5 (-15%)** | 49.8 → 29.0 | 9.0 → 21.9 | 191 → 193 | 299.7 → 284.7 |
| dsv4-pro | 339.9 → 347.8 | 49.7 → 39.4 | 11.3 → 22.5 | 706 → 711 | **1055.6 → 1068.3** |
| minimax-m3 | 103.6 → 102.2 | 30.4 → 22.8 | 4.6 → 11.7 | 233 → 235 | 341.6 → 341.6 |
| gpt-oss | 60.2 → 63.0 | 39.0 → 39.5 | 2.9 → 5.0 | 168 → 170 | 233.6 → 237.7 |

1. **The coupling was real.** The activated-wait share fell on 8 of 9 shapes, by 8-34 points. So part of the parent's "activated" wait was really the weights of block kb-1 arriving late: the lock-step loop delayed the act ack for kb+1 behind them. Now the compute waits on the down weights directly: their share roughly doubled everywhere.
2. **On the latency-bound down chains the window really shrank.** kimi-k3 fell from 461 to 368 µs per core (28 small experts, many short blocks), dsv4-flash fell 15% and mistral 13%. DN cycles per tile-mult went 61 → 50 on kimi-k3, 47 → 41 on dsv4-flash, and 29 → 25 on mistral.
3. **On the DRAM-bound shapes the time was conserved, or worse.** kimi-k2.7's GU grew +18 µs per core and kimi-k3's +67 (it still nets -27 µs mean core end). dsv4-pro's DN grew +8. This is the same conserved-GU+DN signature as r02-b02-a03, r03-b04-a03 and r04-b04-a01.
   - With the in1 ack decoupled from the activated stream, receivers ack the next down-weight block as soon as its slot frees. So the column senders issue down reads earlier and more of them run concurrently. Those reads take DRAM share from groups that are still in gate/up, or still in phase A.
   - On kimi-k2.7 and dsv4-pro (about 90% of DRAM bandwidth) this is a net loss of +1.6..+2.1% chip. This also matches r02-b04-a02, where a free-running down stream cost 10-20% on the DRAM-bound shapes.
4. gpt-oss is unchanged. Its DN act wait stayed at 39%, so there the activated stream is not paced by the weights. It is paced by the activation, which finishes at different times on different row cores, and each act sender waits for its own compute.

## Classification
Neutral (within noise vs the probe-free root: +0.1%; the +1.1% vs the parent is mostly the probe removal). The mechanism itself is redistributive:
- It is a real win on the latency-bound down chains: kimi-k3 -3.7%, dsv4-flash -2.4%, mistral -1.6% chip vs root.
- It is a loss on the DRAM-bound shapes: kimi-k2.7 +2.1%, dsv4-pro +1.6%, dsv3 +1.2%.

## What a child of this node should try next
1. **Decouple only the activated stream.** Keep the polling loop's act half, which is where the k3 / dsv4-flash / mistral win comes from. Put the down-weight receivers' ack back behind the activated arrival: ack w(kb+1) only after act(kb) is pushed, as the lock-step loop did. That stops the down reads from running ahead into the gate/up groups' DRAM share. This is a one-line condition in the weight branch (`w_ack == w_rx && act_rx > w_rx`). Expect it to keep most of the k3 / dsv4-flash gain and to remove the k2.7 / dsv4-pro loss.
2. **Then port it onto r04-b02-a01 (best, 1.6275).** Its down look-ahead (an early read into the sender's own free slot, with a re-check during the act spin) acts on the weight-sender side. This node's act-stream decoupling acts on the receiver / act-sender side. In this polling loop the look-ahead becomes simple: issue block w_rx+1's read into the next slot whenever it is reservable.
3. Don't chase DN on kimi-k2.7 / dsv4-pro / dsv3 by delivering down weights faster. Total DRAM bytes / time is already about 90% of peak there, and every earlier delivery moved time into GU (r02-b02-a03, r02-b04-a02, this node).
4. Tools: `/tmp/r04b03a01/probe.py <profile_log_device.csv>` gives the RE-UW wait shares (the probe line is absent now). `ends.py` gives the mean / max core end. The probe can be re-enabled with `-DRE_PROBE` in the compute defines if needed.
