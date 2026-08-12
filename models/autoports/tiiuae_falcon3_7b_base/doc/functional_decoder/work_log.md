# Functional decoder work log

## Scope

- Model: `tiiuae/Falcon3-7B-Base`
- Autoport: `models/autoports/tiiuae_falcon3_7b_base`
- Stage: functional decoder translated from TTNN IR only
- Representative layer: 14 of 28
- Source mesh/TP: 1×4, collapsed to one dense device
- Repository: `/home/mvasiljevic/tt-metal`
- Branch/start commit: `model/tiiuae-falcon3-7b-base` at `66533e5bc32`

No optimized-decoder, multichip, full-model, generator, or vLLM work was started.

## Commands and findings

1. Graph classification:

   ```text
   .agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh <IR_DIR>
   g0 prefill: fill_cache=1792, paged_update_cache=0, decode_sdpa=0, compiler/non-runtime
   g1 decode: fill_cache=0, paged_update_cache=56, decode_sdpa=28, compiler/non-runtime
   g2/g3: corresponding logits-returning variants
   ```

2. Readable emits:

   ```text
   scripts/ir_to_emit.sh ...g0...mlir /tmp/tiiuae_falcon3_7b_base/prefill
   wrote prefill.py (27135 lines)

   scripts/ir_to_emit.sh ...g1...mlir /tmp/tiiuae_falcon3_7b_base/decode
   wrote decode.py (7907 lines)
   ```

3. HF configuration (`AutoConfig.from_pretrained`): hidden 3072, 28 layers, Q heads 12, KV heads 4, head dim 256, MLP 23040, RMS epsilon `1e-6`, default RoPE theta 1000042, vocab 131072, advertised context 32768.

4. Flat-graph segmentation: layer 14 begins at input-norm formal `arg_85` in prefill / `arg_58` in decode. The const-eval input list is V,K,Q, then reversed, transposed, and concatenated as Q,K,V. Local TP widths are 768/256/256 and 5760; dense widths are 3072/1024/1024 and 23040. O/down projections feed `all_reduce(cluster_axis=1)` and therefore become full dense matmuls with the reductions removed.

5. Static checks:

   ```text
   python -m py_compile models/autoports/tiiuae_falcon3_7b_base/tt/functional_decoder.py
   python -m py_compile models/autoports/tiiuae_falcon3_7b_base/tests/test_functional_decoder.py
   pytest -q .../test_functional_decoder.py --collect-only
   # 3 tests collected
   pytest -q .../test_functional_decoder.py::test_runtime_forwards_have_no_host_fallback
   # 1 passed

   .agents/scripts/check_context_contract.py \
     --model-dir models/autoports/tiiuae_falcon3_7b_base \
     --hf-model tiiuae/Falcon3-7B-Base --require-contract
   # Context contract OK: target=32768, supported=128 (DRAM-limited)
   ```

6. Host-reference checks loaded the exact nine real layer-14 tensors (237,508,608 parameters) from
   `model-00002-of-00004.safetensors`, assigned them strictly into
   `transformers.models.llama.modeling_llama.LlamaDecoderLayer`, and verified that the HF
   `DynamicCache` for layer 14 grows from two prefill tokens to three after one decode step. A
   real-weight PyTorch control using the translated fused-QKV order, dense full weights, GQA/RoPE,
   residuals, and SwiGLU produced PCC 0.999987 against the HF layer at batch 2 / sequence 4. This
   verifies the TP-collapse algebra independently of TT device availability; it is not substituted
   for the required TTNN PCC gates below.

## Device recovery ledger

- Initial `timeout 60 tt-smi -ls --local`: four Blackhole p300c devices visible.
- First 1×1 open/close smoke failed before model code: sysmem was mapped at `0x1000000040000000` instead of `0x1000000000000000`, indicating another/stale owner of the UMD sysmem NOC range.
- Process inspection found no live pytest, model, profiler, or serving process from this run. No process was killed and no lock was cleared.
- Recovery executed serially: `timeout 180 tt-smi -r`, then `timeout 60 tt-smi -ls --local`; reset returned and all four devices remained visible.
- The repeated full-cluster 1×1 smoke failed with the same sysmem address conflict.
- `build/tools/umd/lock_virus` reported all 21 present UMD mutexes `FREE`; `/proc` FD/map inspection
  and `fuser` found no visible holder of `/dev/tenstorrent/*` or the stale sysmem metadata. This is
  consistent with an owner outside the container's visible PID namespace.
- A bounded 1×1 smoke explicitly requesting physical device 1 still initializes the whole local
  cluster and failed at the same `0x1000000040000000` mapping before selecting the mesh device.
- `tt-smi -ls --local` showed two p300c boards: devices 0/1 share board number
  `0000046131931058`, while devices 2/3 share `000004613193100c`. The AutoFix source check found
  that `TT_VISIBLE_DEVICES` filters UMD enumeration before cluster startup. A single-device filter
  was invalid for a paired p300c board, but `TT_VISIBLE_DEVICES=2,3` preserved the complete second
  board and a bounded 1×1 open/close smoke passed.
- All subsequent test commands were serialized and restricted to that healthy board with
  `TT_VISIBLE_DEVICES=2,3`. Devices closed cleanly after every run; no reset or recovery was needed
  during model debugging.

## AutoFix investigation

- Infrastructure hypothesis: the initial fresh AutoDebug report identified the full-cluster sysmem
  address conflict. A forked AutoFix source check found the safe pre-enumeration
  `TT_VISIBLE_DEVICES` control. The paired-board experiment verified `TT_VISIBLE_DEVICES=2,3` as the
  in-scope bypass and refuted the earlier assumption that every device selection must initialize
  the unhealthy board.
- First translated prefill run failed because RoPE padded sequence 17 to 32 while V remained 17.
  Raw `g0` contains explicit post-RoPE slices; translating those slices fixed the shape failure and
  produced prefill PCC 0.99893845 at sequence 17 and 0.99897713 at 128.
- RoPE table slices can alias their persistent device tables. Changing only those releases from
  forced buffer deallocation to metadata-only deallocation fixed repeat-call lifetime failure.
- Linear-cache `paged_update_cache` must omit the paged-only `num_kv_heads` override. The corrected
  calls now exactly match the `g1` emit (`share_cache=False`, `page_table=None`).
- Decode then executed but returned PCC about 0.8464. Fresh source-only AutoDebug ranked the
  explicit mask contract first. The exact device-side mask A/B left PCC unchanged at 0.84642097,
  refuting mask selection as the root cause; the emitted mask form was retained for graph fidelity.
- Exact-shape probes localized the defect: prefill caches matched HF (K 0.99991646, V 0.99991996),
  the rotated decode Q matched HF (0.99991428), and the caches still matched after append through
  position 17 (K 0.99991628, V 0.99991987). This refuted QKV fusion, head order, RoPE values,
  cache fill, and cache update.
- The proven bug was the decode-Q memory boundary. `g1` converts the height-sharded RoPE output to
  interleaved L1, slices exact Q heads, then moves Q to DRAM before SDPA. The initial translation
  re-sharded Q like K. Separating Q (DRAM SDPA input) from K (height-sharded cache-update input)
  raised synthetic decode PCC from about 0.8464 to 0.99911663. The original failing gate passed.

## Validation table

| Weights | Path | Shape | PCC | Result |
|---|---|---|---:|---|
| Synthetic bf16 | Prefill | `[1,32,17,3072]` | 0.99893845 | pass |
| Synthetic bf16 | Prefill | `[1,32,128,3072]` | 0.99897713 | pass |
| Synthetic bf16 | Decode Q before SDPA | `[32,12,256]`, position 17 | 0.99991428 | pass |
| Synthetic bf16 | Decode | `[1,32,1,3072]`, position 17 | 0.99911663 | pass |
| Synthetic bf16 | Prefill K/V cache | `[32,4,17,256]` | 0.99991646 / 0.99991996 | pass |
| Synthetic bf16 | Post-decode K/V cache | `[32,4,18,256]` | 0.99991628 / 0.99991987 | pass |
| Real layer 14 | Prefill | `[1,32,17,3072]` | 0.99895504 | pass |
| Real layer 14 | Decode | `[1,32,1,3072]`, position 17 | 0.99882657 | pass |
| Real layer 14, PyTorch dense-collapse control | Prefill | `[2,4,3072]` | 0.999987 | pass (host only) |

Hardware commands proving the translated paths:

```text
TT_VISIBLE_DEVICES=2,3 timeout 1900 python -m pytest -q \
  .../test_functional_decoder.py::test_synthetic_prefill_small_and_larger_and_decode -s
# 1 passed

TT_VISIBLE_DEVICES=2,3 timeout 1900 python -m pytest -q \
  .../test_functional_decoder.py::test_real_weight_prefill_and_one_decode_step -s
# 1 passed
```

The exact real-weight shard `model-00002-of-00004.safetensors` is cached locally.

## Review and checkpoint

- Complete file gate: `3 passed` (static fallback audit, synthetic prefill/decode, real-weight prefill/decode).
- Independent `$stage-review`: `clean-pass`; no P1/P2 findings. Its only P3 clarity concern was fixed
  by marking `AUTODEBUG.md` as historical, and the narrow rereview returned `clean-pass`.
- Local stage checkpoint: `e0f096c91ee` (`Add IR-translated Falcon3 functional decoder`).
- No push was performed.

## Multichip provenance

`multichip_provenance.json` records the complete layer-14 sharding prior from the selected prefill and
decode IR. The source mesh is `data=1 × tensor=4` (TP degree 4); the collective set is ring
`ttnn.all_reduce(sum, cluster_axis=1)` after `self_attn.o_proj` and `mlp.down_proj` in both paths, with
no other collective op in the segmented layer.
