# Fused MLP: silicon results v8-v10 and the multi-layer weight-streaming design (for review)

## Silicon (d27u08, P150 device 0, trace replay x200, same shapes/data as DSMC; PCC 0.99998 for every variant)

| Variant | us / call | GB/s (18.8 MB) | Change |
|---|---|---|---|
| DSMC gate_up + down (2 ops, no SwiGLU/glue) | 57.9-58.1 | 324 | baseline |
| v7 (x gathered to 1 root, h via 1 root) | 77.9 | 241 | previous best |
| v8: x delivered by producer (XPRE, the all-reduce-fed case) + h via 4 roots on both NOCs | 70.6 | 266 | x wait now ~0 |
| v9: v8 with the root role as a runtime arg (2 kernel groups instead of 6) | 66.3 | 284 | -4.3 us launch |
| v10: v9 with one kernel group (gate_up / down role runtime) | **63.2** | **297** | -3.1 us launch |
| v10 + FM_BAL (equal bytes per DRAM bank) | 66.2 | 284 | worse: cores read farther banks |

Device profile, v8 (x delivered): kernel span 56 us vs DSMC 55.4 us (gate_up 37 + down 18.4). Gate|up compute per core
median 39 us, slowest 49 us (bank tail: banks holding 9 gate|up slices stream ~2.44 MB vs 2.25 MB, ~50 GB/s per bank);
h is everywhere ~5 us after the slowest gate|up core; down tail is short. The remaining wall gap is launch overhead:
~7 us over the kernel span (v10) vs ~2.5 us for DSMC's two ops.

Conclusion: delivering x from the all-reduce and broadcasting h through 4 roots removes the data-movement cost
(that was 27-75 us in v7), but a standalone fused MLP still does not beat DSMC: the drain is bank-bound (~390 GB/s
peak, tail-limited) and the single op pays more launch overhead. A fused MLP wins only when it is persistent across
layers, so launch overhead is paid once per pass and the bank tail of layer L overlaps layer L+1's stream.

## Multi-layer weight-streaming design (proposal, not built)

Goal: per decode pass, one persistent program per layer type (or per layer group), where each core's weight stream
never idles. Today the verify pass is ~1,200 ops; the MLP alone is 64 x (gate_up + silu*mul + down + norm + glue).

1. Persistent per-core loop over layers. Each core owns a fixed bank-local slice of every layer's gate|up / down
   weights (re-laid out at load, as in bench v8-v10). Its two data-movement RISCs issue reads for layer L+1's slice
   as soon as layer L's reads are issued, double-buffered in L1 (slice ~235 KB per core per MLP; two layers plus x / h
   landing buffers fit in ~1 MB of L1 on the 80-core rectangle).
2. x comes from the decode all-reduce: its receive stage writes the reduced (and, later, RMS-normed) x straight into
   each core's x landing buffer and raises one flag (the v8 XPRE contract). Norm folds into gate_up's input path.
3. h via 4 roots on both NOCs (v8 scheme, measured ~5 us after the slowest gate|up core). Down output goes straight
   to the next all-reduce's send buffer.
4. Only layer type boundaries need program switches (MLP / attention / GDN); within the MLP program, all 64 layers
   are a loop driven by a per-pass descriptor in L1 (layer index, weight base pages), so launch is paid once.
5. Integration: a plain ttnn generic_op captured into the qwen36 decode trace (as MiMo's moe_stream / proj_stream);
   it replaces Qwen36MLP._forward_tp_gu_fused_decode; M = 32 rows covers the spec-verify T = 4 x 8 users unchanged.

Expected MLP per layer at the measured ~390 GB/s drain with launch and tail hidden: ~48 us vs ~60 us today
(52 + norm 6.6 + glue), i.e. -0.8 to -1.0 ms per verify pass at u8. Main risks: the all-reduce receive-stage change
(ccl op host C++ and kernels), L1 budget with attention / GDN blocks co-resident, and per-layer sync cost inside the
loop (must stay ~1-2 us, measured x/h flags ~1-2 us in v8).

Open questions for review: (a) one persistent program per layer type vs. a single program interleaving MLP with
attention / GDN layers; (b) whether to fold the all-reduce entirely into the persistent program (ring on the fabric
from inside the kernel) or keep the existing all_reduce_async with a modified receive stage.
