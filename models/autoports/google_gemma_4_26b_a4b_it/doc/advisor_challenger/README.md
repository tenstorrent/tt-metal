# Gemma-4 26B A4B advisor contribution

## Full-model estimate: 36,224.1 ± 36.5 us before; 36,076.2 ± 36.5 us after

At decode batch 1, shard advice contributes an estimated **147.9 us per full
30-layer model decode**. The saving is larger than the conservative 36.5 us
uncertainty band. This is a derived full-model estimate: each measured
single-layer delta is multiplied by its model-config layer count (25 sliding,
5 full) and the two kinds are summed.

The shipped change adds DRAM-sharded O projection to sliding-attention layers.
Its five repeats were 1.253684, 1.253986, 1.254487, 1.253223, and 1.253954 ms;
all beat the frozen incumbent's 1.258866--1.260157 ms range. A fresh-process
confirmation produced 1.253043--1.253644 ms. Real Gemma weights pass the
unchanged 0.995 oracle bar at PCC 0.998358 prefill and 0.999499 decode.
The final constructor default was exercised again without candidate placement
environment overrides and reproduced those same real-weight PCC values.

Full-attention QKV DRAM sharding was measured despite being doubtful. It lost
at 1.290230--1.291618 ms against the 1.261299--1.262151 ms control range and
remains default-off. Other losing roles and residual-grid knobs remain
environment-gated and default-off.

## Reconciliation and reach

Both reconciliations close at 100% and neither is `DEGRADED`. Sliding's
1,206.737 us window assigns 52.40% to the declared uncapturable routed-expert
suffix; full attention's 1,211.136 us window assigns 52.08% there. The advisor
drops 3.832 us/layer of sliding boundaries and 6.373 us/layer of full
boundaries. It agrees with shipped boundaries costing 9.433 and 10.715 us,
respectively; these are reported real costs and were never screened or booked
as advisor contribution. The layer-handoff report finds no extra boundary
round trip in either kind, and was likewise not screened.

The advised graph placed material norms on 88 cores while the incumbent profile
shows one core. Those observations are retained in the reconciliation output;
they were not credited to this shipped attention-chain result. The advised
core count was not treated as a recommendation, and no sweep was bounded at or
below it.

The pinned advisor was `618cd4e75d`. The incumbent was frozen before both
captures, all timing used the fixed harness protocol in fresh processes, and
all capture/timing batches were exactly 1.
