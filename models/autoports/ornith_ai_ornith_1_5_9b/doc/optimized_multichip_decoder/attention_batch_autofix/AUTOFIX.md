# Batched attention core ownership repair

The failed contract is independent per-user Q/K/V and cache ownership under
traced batch32 decode (PCC >=0.995). The first row passed; row1 was0.972004.
AutoDebug source inspection identifies the first inconsistent transition:
singleton CoreRange sets sort by x before y, whereas SDPA expects rectangular
row-major ownership. The repair changes only the range description to an ordered
rectangle prefix. No table stride or math changes are made in this intervention.

Discriminating check: repeat the same batch32 stack with the same frozen stage4
candidate base, reference tensor, weights, positions and shuffled page tables.
Command and output: attention_batch_grid_32_command.json / .log / .json.
Device outcome pending. Follow with batches2,3,7,8,12,13,16,31,32, state checks,
changing positions and long contexts, trace allocation checks and watcher.

Independent AutoFix source check by /root/verify_attention_grid verified actual
host core metadata for n=1..32 at both offsets. It identified concat's distinct
requirement of >=4 output cores. The repair omits the optional concat subgrid for
n=2 or3, permitting its ordinary factory, while keeping exactly n input users.
n=1 delegates to the preserved original attention implementation.

## Focused result

`attention_batch_grid_32.json` passes all 64 prefill/continuation and128 changing-input
decode comparisons, minimum PCC0.999824047. Traced warmed stack3.30521ms versus
5.53172ms for the fused-recurrent/serial-full-attention candidate and18.62204ms
fresh stage4. Rectangle-only mapping repair restores the observed failing row;
no page-table padding or precision changes were needed. All-batch state checks,
long-context and watcher validation remain pending before production selection.
