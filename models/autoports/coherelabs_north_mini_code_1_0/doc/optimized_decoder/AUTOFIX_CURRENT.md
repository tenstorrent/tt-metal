# Current-arm AutoFix result

## Attention precision: fixed

QKV and O weight dtype/fidelity are now independent; non-matmul attention
fidelity stays fixed. A real-checkpoint sequence-33 prefill feeds a warmed
cache-consuming traced decode at batches 1 and 32 on the final topology.

All 16 BFP4 candidates failed the 0.995 decode bar: QKV-only 0.99458730,
O-only 0.99474771, and cumulative 0.99258065 at both batches across LoFi and
HiFi2. Selected BFP8/LoFi passed with decode PCC 0.99664833 and prefill PCC
0.99896813/0.99897853. Twenty-replay latency was 0.178979 ms b1 and
0.252752 ms b32. Evidence:
`candidates/review_attention_precision/{results,selected}.xml`.

## Sparse output subblocks: fixed and promoted

The legal cumulative 1x2 candidate uses 12 cores for gate/up and 32 for down.
Authentic layer-1/layer-4 decode passed at PCC 0.99931071/0.99974180.

| workload | 1x1 control | cumulative 1x2 | delta |
|---|---:|---:|---:|
| traced decode b1 | 0.791673 ms | 0.725390 ms | -8.37% |
| prefill b1 seq128 | 13.990679 ms | 13.544480 ms | -3.19% |

The cumulative geometry is now default. Evidence:
`candidates/sparse_subblock_{baseline,cumulative}_{b1,prefill_b1}.json` and
`candidates/sparse_subblock_cumulative.xml`.

## Batch-32 routed MoE: AutoFix exhausted

No model-local composition retains every contribution from the fast rolling
`moe_compute` buffer. Full-output sparse candidates remain 5–7x slower than
the selected dense path, while the complete fused consumer requires fabric.
Host combine violates the traced device-resident contract. The missing
capability is a shared-TTNN local-only combine or compact persistent routed
output, outside the user's authorized stage files.

AutoFix closed both model-local findings and failed on this third finding
after exhausting the legal model-local families. The stage cannot claim the
current optimize checklist or a clean review while that scope conflict
remains.
