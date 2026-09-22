# Collective family contracts

All comparisons use axis1 of the physical1x4 mesh, real TP4 weights and both
unlike layers in one traced stack. The local hidden width is1280 when the
residual is sharded and5120 when replicated. The comparison-only gather is
outside timing for sharded results. Policies and numerical results are indexed
in `topology_family_results.json`; exact per-run overrides and commands are
retained beside each report.

| Family | Residual before / after | Material communication and placement | Buffer plan | Result |
| --- | --- | --- | --- | --- |
| Separate row matmul + native AR | Replicated L1 / replicated L1 | Attention output and MLP down each sum partial outputs directly; no boundary gather | Shared1.25MiB L1 AR workspace, CQ0 semaphore pool | Selected; ~.706ms stack |
| Separate row matmul + RS/AG | Replicated / replicated | RS then AG after each row projection | Preallocated DRAM or L1 RS/AG outputs | .744–.747ms; loses |
| Separate row matmul + RS | Sharded1280 / sharded1280 | Hidden gather for consuming column projections; distributed norms and sharded residual adds | DRAM/L1, persistent on/off | Basic norm family .95ms+; fused distributed norm improves to .775ms, still loses |
| Row AGMM | Sharded1280 / sharded1280 | Gather local1536/4352 attention/MLP channels while multiplying output-partitioned row weights; next residual remains sharded | Persistent gathered inputs on/off; fused K8/16, down17; legal4/5 workers | ~1.24ms before fused-norm refinement, ~1.09ms with fused norm; loses |
| Row MMRS | Sharded1280 / sharded1280 | Output and down matmuls fuse their output reduction/scatter | Persistent outputs, L1 rolling window2, shared progress/credit buffers | ~1.01ms with fused norm; loses |
| Column AGMM + row RS | Sharded1280 / sharded1280 | Fused distributed norm produces local normalized hidden1280; AGMM feeds packed attention and MLP directly | Persistent gathered5120 input, K8/20/40, N8/16 | Best~1.248ms; loses |

The alternatives cross BF16/BFP8 CCL payloads and compatible activation dtypes,
with BFP4/LoFi real weights retained. BF8 direct AR gives .705608ms versus
BF16 .705983ms, below a material repeatable gain and with weaker PCC.
Separate and fused layouts include their actual input resharding, dtype casts,
residual adds and next-layer consumers in timing. No family is rejected only
because it restores the old replicated residual immediately after a reduction.

For one padded decode tile, hidden5120 occupies327,680B in BF16 or174,080B
in BFP8; hidden1280 occupies81,920B or43,520B. A four-rank ring's algorithmic
per-rank payload is3/4 of the full tensor for RS or AG and twice that for AR:
245,760/130,560B per RS/AG, or491,520/261,120B per AR (BF16/BFP8).
These are semantic traffic estimates, excluding protocol headers, bidirectional
scheduling, alignment and retransmissions. Two row reductions are required per
layer. Full four-rank gather of local1536/4352 row inputs and normalized1280
column inputs follows the same3x-local-payload receive rule. Distributed norm
also communicates statistics; it avoids full residual replication.

The final path removes no mathematically required reduction by pretending that
partial TP outputs are complete. It keeps both internal reductions and adds
no collective between layers. The prefill boundary remains DRAM interleaved;
its row projection sum uses asynchronous RS followed by AG. Larger-batch
public TILE rows stay in DRAM while each layer packs internal rows into L1.

Material API failures were adapted: AGMM worker/grid divisibility, MMRS output
shape and persistent counters, fused norm buffers, packed/separate projection
weights, and split attention padding. Corresponding adapted runs passed PCC
and were measured before rejection. Native multi-reader and watcher failures
were repaired through AutoFix instead of rejecting those capabilities.
