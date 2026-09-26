# V7 source delta and validation attribution

Current source is `daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`.
Its exact v6 ancestor is [the saved source](minimal_pairs_v6/runtime_snapshot.py.txt),
SHA256 `b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846`.
This audit executes CPU source/configuration checks and reads existing artifacts;
it runs no hardware and changes no model runtime.

Only two of 40 methods differ. [The JSON proof](source_delta_v7.json) retains the
text diff, hashes of all 38 unchanged method ASTs, and exact inverse transforms:
removing the new factory keyword/auto resolution/call argument and the QKV
constructor compute-clone branch reproduces the v6 ASTs exactly.

| Changed setup site | Exact change | Preserved behavior |
| --- | --- | --- |
| `OptimizedDecoder.from_state_dict` | Adds `prefill_qkv_fidelity="auto"`: HiFi4 sliding, HiFi2 full; forwards only to minimal QKV setup. | All other defaults, override resolution and construction are unchanged. Disabled minimal QKV ignores this setting. |
| `MinimalPrefillQKV.__init__` | Clones architecture-specific compute flags when fidelity is supplied, changing only fidelity. Explicit `None` reuses the source compute object. | Source compute, weight alias, wrapped decode callable, K/M/N/subblocks, memory and dtype remain unchanged. |

CPU execution of the exact scalar auto branch passes ten default/explicit
cases. Execution of the exact constructor with CPU stand-ins passes eight
Broadcast/Tied source cases, checking all six compute fields, object aliasing,
unchanged source compute and preserved decode source. This is source/config
proof, not accelerator arithmetic emulation. Both new audit generators pass
pinned Black23.10.1, isort5.13.2 and applicable pre-commit hooks.

The generic minimal projection call, one-row decode delegation, tied-K/V
concatenation, every public/forward method, cache validation/fill/update,
normalization, routing, experts, output projection, RoPE and v6 tail/capacity
repairs are AST-identical. Sliding's newly constructed config has the same
complete compute values as its original HiFi4 object. Full prefill changes
numerically, including the cache used by later decode; it is not described as
numerically equivalent merely because decode source is unchanged.

## Why only full attention changes

[Eight paired samples per candidate](minimal_pairs_v6_summary.md) show a HiFi2
whole-prefill improvement of median paired −473.456µs sliding and −545.573µs
full. Both candidate stress runs pass. Sliding nevertheless fails exact
maximum-context sampled row32 at PCC0.9949930133358956 and therefore retains
HiFi4. Full passes all 291 sampled maximum-context rows, minimum0.996029643684647,
and selects HiFi2. The threshold remains .995. Grid11×10 gives no demonstrated
gain on either kind, so11×8 is retained. These candidate reports keep their
v6-plus-probe attribution; they are not production v7 executions.

## Current gates and inherited scope

[The v7 manifest](validated_v7_validation_summary.json) has status
`current_targeted_and_inherited_gates_passed`. It validates all12 primary
commands and five additional affected-prefill commands, with no pending or
failed correctness items:

- All eight full-attention public contracts run on v7: B32, prefix, allocation-
  tracked nine-request reuse, BF16-cache compatibility,262144,262143, headline
  and Watcher. Both long cases pass every one of291 sampled rows, final tails
  and boundary decode; minimum sampled-row PCC is0.996029643684647.
- Sliding headline/Watcher, all four small-input pytest cases, and both-kind
  1025/512 HF plus fused-preservation stress run on v7. HF stress minima are
  0.995555061903 sliding and0.996214257441 full; direct preservation minima are
  0.995158301497 and0.996295795028.
- A current tight1025/cache1152 Watcher test records Q64/K128/read-end1152 from
  the actual native call. Four full short/tail timing controls cover logical
  65/1023/1024/1025, with physical[96]/[1024]/[1024]/[1024,32], actual inputs,
  clean guards and repeated traced decode.

Unchanged sliding B32/prefix/BF16-cache/max-context coverage remains attributed
to the completed v5 suite, and allocation-tracked lifecycle coverage to v6.
The manifest rechecks those report hashes and the source-delta chain; it does
not claim those six broad sliding contracts reran on v7. Current full maximum-
context results replace full v5 references in [the context contract](../context_contract.json).

Checkpoint, expert, shared, QKV/output, cache and RoPE resident tensor payloads
have **zero byte delta**. No setup tensor is added; a host compute-config object
is constructed. The inherited v6 skip of unused final sliding-tail clones
remains at most8MiB of private payload, separately scoped from resident storage
and allocator peaks. Current native profiles and independent final review are
separate work; ancestor CSVs and timings are never relabeled with the v7 hash.
