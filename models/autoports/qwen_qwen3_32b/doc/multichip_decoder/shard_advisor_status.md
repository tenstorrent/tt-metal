# Multichip shard-advisor status

The required current-pass attempt was made after the final per-role layout was
implemented.

## Command and blocker

```bash
cd /home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
ttnn-advise capture --help
```

The bootstrap resolves `/opt/ttmlir-toolchain/venv/bin/ttnn-advise` and the
system descriptor, but importing the pinned tt-mlir `_ttnn.so` fails before a
capture can begin:

```text
ImportError: .../_ttnn.so: undefined symbol:
_ZN4ttnn12experimental11moe_computeE...
```

Running the bootstrap from the tt-metal checkout also resolves the wrong
relative build paths and fails with `ModuleNotFoundError: No module named
'ttnn'`. Running it from the intended tt-mlir checkout reaches the more precise
binary-version-skew failure above. Per the skill instructions, rebuilding
tt-mlir is one-time operator setup and is outside this decoder stage.

## Disposition

The completed optimized-decoder advisor report remains at
`doc/optimized_decoder/shard_advise/report.json`. It supplied the original
per-rank L1 layout family, but its advised L1 matmuls were already measured at
1.737 ms versus 1.217 ms for the single-chip DRAM-sharded baseline. The advisor
skill explicitly does not select the DRAM-sharded-weight strategy.

For this multichip stage, the advisor additionally cannot optimize the Ring CCL
graph as one TP program. The final path therefore preserves the proven
DRAM-sharded per-rank skeleton and uses direct, correctness-gated measurements
for the new axes:

- boundary sharding versus replicated compiler provenance;
- QKV, O, gate/up, and down grids and input block widths;
- BF16 versus BFP8 collective payloads;
- persistent versus non-persistent collectives;
- fused versus separate row-parallel collectives.

The exact final layouts and program configs are embedded in
`results/final_selected_o8.json`; all candidates have source and hardware
provenance. A fresh advisor report should be regenerated after the pinned
tt-mlir/tt-metal binary mismatch is repaired, but this does not invalidate the
measured selected topology.
