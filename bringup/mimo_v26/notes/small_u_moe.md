# What the decode MoE needs for small U (users per mesh row 1..32) - decode agent, for the MoE agent

Today (tt/moe_decode.py MiMoMoEDecode): x [1,1,U,4096] is padded to 64 rows per mesh row (masked_bincount needs tokens % 64 == 0), the U live rows keep their router output, ALL other rows (dead
slots, pad rows) get the round-robin spread routing (spread_pad_routing) over the 256 experts. Consequence: every step ~all 256 experts have tokens, the expert op reads (almost) all expert weights
whatever U is, and the traced MoE costs ~1.5 ms/layer for U = 1..32 (measured by the MoE agent; the decode step time at B=64 and B=128 is the same 123 ms: tests/test_decode_timing_u.py breakdown).

To make the step time scale with U the MoE would have to
1. NOT dispatch dead/pad rows (no spread): give them an "invalid" routing that dispatch/masked_bincount/combine ignore (e.g. expert id >= 256 or a padding_config that marks rows invalid; the
   dispatch op has a `padding_config` argument that is None today - unknown semantics, please check), so the per-expert counts only contain the 4*U*8 real assignments (U=1: <= 32 distinct experts
   instead of 256);
2. run the unified expert op only on experts with count > 0 (it already takes counts: check whether empty experts are skipped or still stream their weights; if they stream, that is the DRAM-bound cost);
3. accept N = 32 tokens per row (or any multiple of 32) in masked_bincount / offset_cumsum / combine (today N % 64 == 0: decode pads inside the MoE module, the padding rows then must be invalid, see 1);
4. keep the per-layer router (exact) cost: 0.29 ms/layer is also U independent (router_select needs <= 32 rows: fine for U <= 32).
Expected effect (model): expert weight traffic ~ distinct active experts: U=1 -> ~1/8, U=4 -> ~1/2, U>=8 -> ~all experts (B*8/256 = 1 assignment per expert already at B=32).
Until then decode keeps padding to 64 rows inside MiMoMoEDecode and the U sweep only shows the attention / KV / lm_head part scaling.
