# Layer34 incoming-state discrepancy: smallest next discriminator

Source/CPU-only recommendation after the combined precision candidate still fails268 positions. No production/harness edits or device execution. Read the selected/combined source, current immediate-copy capture diagnostic, existing prefix-injection diagnostic and canonical HF chunk/recurrent implementations. `v5_layer34_cache_boundary_cpu.json` independently compares the actual selected-v5 captures at137/165/181; the combined candidate's new captures are still needed before attributing those exact counts to it.

## What the present captures establish

Compare **consumed BF16 state**, not raw persistent FP32 storage. Incoming state differs from HF in105/310/404 of524288 BF16 values at137/165/181; the outgoing BF16 images differ in128/345/403. At these same three tokens, normalized K, V, beta and FP32 decay are exact; Q differs in684/468/636 of4096 repeated-head values. These are actual capture comparisons. The other investigator reports a separate same-input CPU recurrence replay exactly matching TT memory/delta/new state/core; this memo does not mislabel that as its own device test.

Q only enters the final output contraction. The state update is:

```
D = decay * BF16(incoming_state).float()
memory = sum(D * k)
delta = (v - memory) * beta
next = D + k * delta
output = sum(next * scaled_q)
```

Therefore current or prior Q/Z discrepancies cannot cause the recurrent cache discrepancy. The stored state does not depend on gated norm, output projection, residual, postnorm, routes or experts either. The harness supplies fixed external tokens rather than feeding TT outputs back as new inputs. Precise Q/Z and downstream repairs can improve the visible output but cannot repair this existing state difference.

Exact K/V/beta/decay at137/165/181 does not establish their equality at every earlier token. A single earlier BF16 midpoint difference can enter state and persist. The current three captures do not contain position128's incoming state or the exact post-prefix cache; **the original prefix effective BF16 error is presently unmeasured**, so no claim that it is equal or different is justified.

## Source-supported competing causes

1. Prefix arithmetic association. HF `torch_chunk_gated_delta_rule` (modeling_qwen3_5_moe.py:244–322) uses chunk_size64, cumulative g, triangular transformed values and matrix products to update final state. The model's prefill state correction explicitly scans tokenwise FP32 outer products after its native core call. It avoids known native TF32 state loss and is mathematically equivalent, but its FP32 association differs from HF's chunk transform. BF16 casting at the first decode boundary can expose a rare midpoint difference even with equal prefix K/V/g/beta. This is a plausible prefix source, not a proven actual mismatch.
2. Earlier frontend input differences. The source explicitly rounds BF16 square, sum, epsilon, rsqrt and final multiplication in Q/K L2; beta is FP32 sigmoid followed by BF16 round/widen; decay is an exact table lookup from BF16 A. These contracts match intended boundaries but do not prove all layer34 inputs exact across128–180. Earlier K/L2/V/beta differences remain falsifiable alternatives.
3. A new state-update bug at the current captures is disfavored by exact same-input CPU replay. That evidence covers the recorded operands only; it is not a proof for every earlier token or prefix path.

## Recommended bounded test

First add position128 to the already planned snapshot-aware combined capture (retain137/165/181). The immediate `delta_state_input` clone before the first decode is the effective post-prefix state. Compare its BF16 image to HF and also compare mapped convolution history: HF `[1,8192,4]` last3 columns transposed to TT `[1,3,8192]`. This is the smallest read-only discriminator and requires only54 decode tokens to reach181.

If128 is already different, make one **diagnostic-only** rerun with the canonical HF prefix cache injected after the normal TT prefix and before warming/capturing decode. Use the same frozen combined source, same actual layer34 prefix/tokens and oracle identity; leave TT prefix outputs untouched. Inject recurrent HF BF16 state widened toFP32 into the existing persistent FP32 TILE buffer. Inject the last3 HF convolution columns as BF16 ROW_MAJOR `[1,3,8192]` into the existing RM buffer. Keep buffer addresses stable by copying into them, rather than replacing the state's tensor objects. Save pre-injection differences and verify post-copy BF16 equality. This host-prepared injection is only a diagnostic, not a proposed runtime host fallback or acceptance evidence.

Important existing-helper incompatibility: `tests/prefix_state_probe.py:48–57` currently uploads **both** seed tensors in TILE layout. That legacy helper predates v5's RM convolution cache. Do not run it unchanged and interpret an API/layout failure as evidence against prefix seeding. A snapshot-aware diagnostic must upload convolution seed in ROW_MAJOR and select the intended snapshot class consistently. Its recurrent mapping atlines39–42 is otherwise appropriate.

After canonical seeding, compare K/V/beta/decay and consumed/new BF16 state at128 and then at each token through the first divergence (at most181 for this small experiment). If inputs and incoming state are exact but outgoing BF16 state first differs, replay that single step onCPU and inspect its intermediate reduction/boundary. If inputs first differ, inspect that producer at the same token. If seeding keeps state exact but181 still fails, the captured downstream comparison cleanly separates a non-state route issue. If seeding removes181 while same frontend inputs persist, prefix state contributes causally; it still does not prove all8192 failures are fixed.

A full8192 seeded run should follow only if the short state discriminator is informative. There is no need to spend another full stream merely to discover whether the first incoming cache already differs. Do not round state every internal prefill token: HF's cache BF16 boundary is once per public call, and changing that would introduce a different algorithm.

## Evidence binding

- `v5_layer34_cache_boundary_cpu.json` SHA256 `a8e4e9339810834f69cdfde1ad61037577e999576480eb2ca52bce741847c202`
- `v5_layer34_stream_probe.json.pos137.pt` SHA256 `e627062654d4343b420dc2bd489dc55df7dd35424b33ea7fa9faae361d616990`
- `v5_layer34_stream_probe.json.pos165.pt` SHA256 `2d38d8bdd35ef570a8a33449a7c818ea8bd12cee76dd47ea474c706d0ccc373d`
- `v5_layer34_stream_probe.json.pos181.pt` SHA256 `1c4c7d88f46344482949b0d6162666cd2bad1853c8097bfeb7da047b743f12ce`
- `v5_ordered_qz_postnorm_bf16_gate.py.snapshot` SHA256 `16fdf3f632f685af1b8e279f933fb28137308d8dc1f3965dafbb50c797401870`
- `v5_ordered_precision_layer34_comparison.json` SHA256 `0e4ca104f2f1a1c0fa079f5f8ed61df47745d1f3a0bff74bb2d00adb241d496b`
