# Functional decoder work log

## Scope

- Model: `tiiuae/Falcon3-10B-Base`
- Autoport: `models/autoports/tiiuae_falcon3_10b_base`
- Stage: functional decoder translated from TTNN IR only
- Representative layer: 20 of 40
- Source mesh/TP: 1×4, collapsed to one dense device
- Repository: `/home/mvasiljevic/tt-metal`
- Branch/start commit: `mvasiljevic/model/tiiuae-falcon3-10b-base` at `66533e5bc32b3f355a9df517c2728d201de89a91`

No optimized-decoder, multichip, full-model, generator, or vLLM work was started.

## Commands and findings

1. Graph classification:

   ```text
   .agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh <IR_DIR>
   g0 prefill: fill_cache=2560, paged_update_cache=0, decode_sdpa=0, compiler/non-runtime
   g1 decode: fill_cache=0, paged_update_cache=80, decode_sdpa=40, compiler/non-runtime
   g2/g3: corresponding logits-returning variants
   runtime g0-g3: same roles wrapped in runtime plumbing; not selected
   ```

2. Readable emits:

   ```text
   .agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh ...g0...mlir \
     /tmp/tiiuae_falcon3_10b_base_ir_emit/prefill
   wrote prefill.py (38619 lines)

   .agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh ...g1...mlir \
     /tmp/tiiuae_falcon3_10b_base_ir_emit/decode
   wrote decode.py (11159 lines)
   ```

3. HF configuration (`AutoConfig.from_pretrained`): hidden 3072, 40 layers, Q heads 12, KV heads 4, head dim 256, MLP 23040, RMS epsilon `1e-6`, default RoPE theta 1000042, vocab 131072, advertised context 32768.

4. Flat-graph segmentation: representative layer 20 is delimited by prefill RMS norms 40/41 and decode RMS norms 40/41. The layer-21 input norms are index 42. The layer-20 const-eval input list is V,K,Q, then reversed, transposed, and concatenated as Q,K,V. Local TP widths are 768/256/256 and 5760; dense widths are 3072/1024/1024 and 23040. O/down projections feed `all_reduce(cluster_axis=1)` and therefore become full dense matmuls with the reductions removed.

5. Static checks:

   ```text
   python -m py_compile models/autoports/tiiuae_falcon3_10b_base/tt/functional_decoder.py \
     models/autoports/tiiuae_falcon3_10b_base/tests/test_functional_decoder.py
   python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_functional_decoder.py --collect-only
   # 3 tests collected
   python -m pytest -q \
     models/autoports/tiiuae_falcon3_10b_base/tests/test_functional_decoder.py::test_runtime_forwards_have_no_host_fallback
   # 1 passed
   ```

## Device usage ledger

- `timeout 60 tt-smi -ls --local`: four Blackhole p300c devices visible.
- All hardware-facing commands were serialized.
- A bounded 1×1 open/close smoke on the complete second p300c board passed:

  ```text
  TT_VISIBLE_DEVICES=2,3 timeout 60 python <1x1 open/close smoke>
  # MESH_SMOKE_OK
  ```

- The smoke and tests emitted benign environment warnings: firmware bundle 19.8.0 is newer than the latest fully tested 19.5.0; the unknown `B850M-C` motherboard falls back to PCI bus IDs as tray IDs; nanobind reports process-exit binding leaks. Controls: the 1×1 mesh opened and closed, both model tests passed, JIT cache completion was reported, and UMD closed the selected devices cleanly after every run. None affected outputs, cache probes, or device health.
- `ShmResourceTracker` removed an orphaned device shared-memory entry at startup. No open or runtime failure followed; devices closed cleanly and no reset, lock removal, process kill, or recovery was needed.

### Context-capacity boundary

The first stage review required measured capacity evidence beyond the IR's emitted cache length. Serialized full-layer probes used a 1x1 mesh, batch 32, synthetic bf16 weights, bf16 TILE/DRAM activations and caches, and converted the final output back to the host before declaring success.

| Sequence | Result |
|---:|---|
| 256, 512, 1024, 2048, 4096, 6144, 6400 | pass |
| 6528 | pass; output `[1,32,6528,3072]`, cache `[32,4,6528,256]` |
| 6560 | first tile-aligned failure; TTNN DRAM OOM at MLP gate/up multiply output allocation (9,673,113,600 bytes) |
| 6656, 7168, 8192 | DRAM OOM; superseded by the adjacent 6528/6560 boundary |

The functional decoder now defaults to `max_cache_len=6528`, allocates exactly that cache shape, and rejects mismatched caller caches. The numerical PCC suite intentionally constructs the model with the emitted length 128; its scope and the capacity-only 6528 evidence are kept distinct.

## Validation table

| Weights | Path | Shape | PCC | Result |
|---|---|---|---:|---|
| Synthetic bf16 | Prefill | `[1,32,17,3072]` | 0.99893845 | pass |
| Synthetic bf16 | Prefill | `[1,32,128,3072]` | 0.99897713 | pass |
| Synthetic bf16 | Decode Q before SDPA | `[32,12,256]`, position 17 | 0.99991428 | pass |
| Synthetic bf16 | Decode | `[1,32,1,3072]`, position 17 | 0.99911663 | pass |
| Synthetic bf16 | Prefill K/V cache | `[32,4,17,256]` | 0.99991646 / 0.99991996 | pass |
| Synthetic bf16 | Post-decode K/V cache | `[32,4,18,256]` | 0.99991628 / 0.99991987 | pass |
| Real layer 20 | Prefill | `[1,32,17,3072]` | 0.99879820 | pass |
| Real layer 20 | Decode | `[1,32,1,3072]`, position 17 | 0.99851787 | pass |

Hardware commands proving the translated paths:

```text
TT_VISIBLE_DEVICES=2,3 timeout 1900 python -m pytest -q \
  .../test_functional_decoder.py::test_synthetic_prefill_small_and_larger_and_decode -s
# 1 passed in 6.89s

TT_VISIBLE_DEVICES=2,3 timeout 1900 python -m pytest -q \
  .../test_functional_decoder.py::test_real_weight_prefill_and_one_decode_step -s
# 1 passed in 3.00s
```

The exact real-weight shard `model-00003-of-00005.safetensors` is cached locally. The runtime methods contain no torch conversion or host fallback.

## Review and checkpoint

- Complete file gate after review remediation: `3 passed in 8.16s` (static fallback audit, synthetic prefill/decode, and real-weight prefill/decode). Structured pytest outcomes and all ten reported PCC values are retained as JUnit properties in `doc/functional_decoder/pytest_results.xml`.
- Independent `$stage-review`: initial verdict `more-work-needed`; required measured context-capacity evidence and an exact configured-cache invariant. Both were remediated. A fresh independent rereview returned `clean-pass` with no required work or hard-check gaps.
- Stage implementation checkpoint: `4ed7c269820843621d1b5d326859badd9891a622` (`Add Falcon3 10B IR functional decoder`).

## Multichip provenance

`multichip_provenance.json` retro-captures the selected prefill/decode IR sharding for representative layer 20: a 1×4 mesh with TP degree 4 on mesh/cluster axis 1. Q/K/V and gate/up are column-parallel, O/down are row-parallel, and the complete per-layer collective set is two ring-sum `ttnn.all_reduce` operations per path (after O and down); the selected graphs contain no other collective type.
