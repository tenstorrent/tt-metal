**AutoDebug investigation: batch versus solo MoE reduction**

2026-11-14. Inspection only; no device operations, resets, reproductions, or repository source edits. Prepared HEAD: `2589dcde879a81213c03e5f5eaef5c85f68dc6eb`, with existing working-tree changes retained. This report supplements the older evidence and does not supersede its failed gates.

**Leading finding**

The isolated reduction discrepancy is explained by the two selected arithmetic paths. There is no demonstrated indexing, ownership, circular-buffer, or phase-restoration defect for the tested `K=10, H=2560, B=2/32` native path. A new CPU-only replay of the documented Blackhole arithmetic matches **all 61,440 saved `native_batch` BF16 output bit patterns exactly**, including the one-owned-expert case. The earlier CPU RNE phase approximations were incomplete goldens.

The full-model failure remains real and unresolved. B1 selects a HiFi4 matmul with FP32 DEST; B2 selects native HiFi4 scalar multiplication with BF16 DEST and serial BF16 L1 addition. This is an established batch-dependent numerical boundary, not proof that it alone causes the token43 accuracy failure. The supplied model probe does not capture the internal first divergent producer. No precision widening, slower single-row fallback, or gate relaxation is justified by this investigation.

**Direct observations, separated from interpretation**

The requested [shared-pool v5 evidence][pool] records:

- Pooled/private B2 prefill and decode are exact. Pool migration agrees exactly with the uninterrupted B2 reference. The independent private migration control also agrees exactly at forced tokens 42/43/44 with request-row remapping.
- Solo repetition is exact. The first forced decode passes its frozen batch accuracy gate despite nonexact logits: whole PCC `0.9989974775530347`, minimum row PCC `0.9982631220259596`, top1/top5/top100 agreement all `1.0`.
- The next forced decode has whole PCC `0.9796496555936922`, minimum row PCC `0.9702018039015299`, top1/top5 agreement `0.5`, maximum absolute delta `1.044921875`. The B1 survivor after B2 history has PCC `0.9942773245679304` and also fails its gate.
- All 120 nonexact entries among 1,734 comparisons are in solo-versus-B2-history comparison families. The earliest recorded state discrepancy is the next stack layer's conv/recurrent state; layer 0 conv/recurrent remain exact for both requests at the first decode. These are state/logit endpoints, not internal MoE taps.

The requested [reduction v1 evidence][v1] uses three synthetic same-input fixtures. It bypasses router, expert execution, combine, cache, migration, and CCL. Each native repeat, token permutation, duplicate-row, B2-versus-all32-position, and B32 relocation check is exact.

| Synthetic fixture | Local experts per rank, per row | B1 versus native differing elements /20,480 | Maximum absolute delta |
|---|---|---:|---:|
| `zero_local_rank0` | 0 / 10 / 0 / 0 | 3,386 | 0.01171875 |
| `one_local_rank0` | 1 / 3 / 3 / 3 | 8,633 | 0.0078125 |
| `multiple_each_rank` | 3 / 3 / 2 / 2 | 9,309 | 0.0078125 |

Within the second fixture, rank0 has exactly one owned expert in each row, at slots 0 and 3. Its B1/native comparison differs in **753/5,120** elements, maximum `0.0009765625`. Compute skips the other slots and packs the sole active product with L1 accumulation disabled. Therefore accumulation across multiple experts cannot be the complete explanation.

Interpretation: deterministic arithmetic differences explain the isolated fixtures. Drift propagating from an earlier MLP into later recurrent/KV state is consistent with the recorded model sequence, but the first real-model divergence and its quantitative contribution to the failed gate remain unmeasured. Exact migration controls do not establish cross-geometry arithmetic equivalence.

**Selected operation chain and precision boundary**

[Batched decode][batch] processes attention with independent request state, concatenates the prepared MLP rows, calls one `self.mlp(mixed)`, and slices the resulting rows back into their request residual paths. Within [MeshMoe][moe], the router produces paired top10 IDs and weights; weights become BF16, IDs UINT16 row-major. Logical counts and aligned expert-region offsets use the actual `n`, before reduction padding. Fused expert execution uses the accepted expert dtype, LoFi, BF16 DEST, and disabled stochastic rounding. B1 and B2 both have `input_m_tiles=1`; the untilized expert-output input extents passed into combine are 320 and 640 rows respectively (combined outputs contain 10 and 20 expert rows before reduction padding). The common shared-expert path and rank merge follow the routed reduction.

| Reduction selection | Inputs after preparation | Effective arithmetic |
|---|---|---|
| `n == 1`, [model line 790][single] | BF16 tiled values `[1,1,10,2560]`, weights `[1,1,1,10]`; nonowned values selected to zero before arithmetic | HiFi4 matmul, FP32 DEST, packer L1 accumulation disabled, BF16 output. Compute config inherited from [WeightLoader][loader]. M/K tile storage pads to32. |
| `2 <= n < 32`, [model line 752][reduce] | Row-major combined `[1,1,32,10,2560]`, BF16 weights `[1,1,32,10]`, UINT16 IDs of the same shape; appended rows/weights/IDs are zero | Native fixed HiFi4, FP32 DEST disabled, first local product overwrites BF16 scratch, later local products use BF16 L1 accumulation; tilize and crop to `n`. |
| `n == 32` | Same native shape without appended rows | Identical native factory and kernel configuration. |

The C++ [wrapper][wrapper] forwards operands in that order without swapping or choosing another numerical implementation. [Validation/output construction][validation] requires row-major BF16 values and weights, UINT16 row-major IDs, INT32 table, and BF16 tiled output. Both optional IDs/table are supplied, so this is the dispatch-table skip variant, not the weight-zero/GPT-OSS variant. No row-index tensor is supplied. `expert_dim=3` and explicit DRAM output survive lowering. The [factory][factory] fixes the native fidelity and destination settings; there is no caller compute-config override hidden in this branch.

**Arithmetic adjudication and new CPU evidence**

Blackhole HiFi4 executes four scalar-multiply phases into the configured DEST. The exact ISA model includes common-exponent alignment at 10 fractional bits for 16-bit DEST and ties-away rounding; a PyTorch BF16 RNE operation after each phase is not equivalent. [Official exact FPU model](https://raw.githubusercontent.com/tenstorrent/tt-isa-documentation/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnitArithmetic.md), explicitly shared by [Blackhole](https://raw.githubusercontent.com/tenstorrent/tt-isa-documentation/main/BlackholeA0/TensixTile/TensixCoprocessor/MatrixUnitArithmetic.md).

After each native product, packer accumulation adds in the final BF16 L1 format with ties-even rounding. The first active product overwrites rather than adds. [Official packer accumulation model](https://raw.githubusercontent.com/tenstorrent/tt-isa-documentation/main/WormholeB0/TensixTile/TensixCoprocessor/Packers/Accumulation.md). B1's final deterministic FP32-to-BF16 early pack rounds midpoint cases away from zero. [Official format-conversion model](https://raw.githubusercontent.com/tenstorrent/tt-isa-documentation/main/WormholeB0/TensixTile/TensixCoprocessor/Packers/FormatConversion.md).

The later [v2 artifact][v2] records matching fixture metrics and model/native source hashes, and supplies a saved tensor path. I read that existing tensor file with `torch.load(weights_only=True, map_location="cpu")`; no TTNN import or device execution was used. Replay procedure:

1. Read the saved BF16 value/weight bit patterns and UINT16-compatible IDs. Preserve mesh-rank, request-row, expert-slot, and feature correspondence.
2. For each owned slot, evaluate four integer scalar phases using the published `ElementwiseDatum`/`LanePairResult` rules, BF16 DEST, deterministic rounding, and `DstLow=0`.
3. Initialize from the first product, then add later owned products with exact integer significand alignment and BF16 ties-even rounding. No-local rows produce zero, as in these finite fixtures.
4. Compare predicted 16-bit words directly to saved `native_batch` output, including all ranks and both rows.

| CPU replay versus saved native output | Different bit patterns |
|---|---:|
| `zero_local_rank0` | **0 /20,480** |
| `one_local_rank0` | **0 /20,480** |
| `multiple_each_rank` | **0 /20,480** |

Additional independent contrasts on the one-local rank0 subset:

- Exact BF16-input product followed by ties-away BF16 packing matches all **5,120 B1 outputs**. CPU RNE differs at 190 outputs; there are 416 exact product midpoints. This explains the B1-versus-CPU-RNE discrepancy in this subset.
- Native exact replay with `DstLow=0` matches all 5,120 outputs. Setting the alternative low-state bit uniformly to4096 changes 501 predictions. The zero-low-state model is sufficient; this does not measure physical residue bits or establish arbitrary execution-history independence.
- The older v2 phase-RNE hypotheses miss 396 native outputs on this same rank/subset; plain product-RNE misses 943. These mismatches are accounted for by the more faithful replay, rather than being residual evidence of bad indexing.

Saved tensor: `/home/ttuser/qwen38-flash/reference/moe_reduction_rows_v2.pt`, SHA256 `fc993d7611099f85665d8b6772d8efcdd8b12be470afbb2c0af1d986bf938614`. This CPU result explains the saved finite fixtures; it is not a new hardware reproduction, all-domain simulator validation, real-model capture, or acceptance result.

**Native coordinate and buffer audit**

For both padded B2 and direct B32, instantiate `tokens=32`, `K=10`, `H=2560`: `emb_dim_cb_tiles=ceil(2560/1024)=3`, output width `2560/32=80` tiles, one chunk, one core. Reader runtime arguments are `[combine,0,1]`, compute `[0,1]`, writer `[weights,output,table,indices,0,1]`. Streaming is selected because `2*10*3*2048 > 64KiB` and `2*10*2048 > 32KiB`. [Factory sizing and work assignment][factory].

| Buffer | Capacity on this Blackhole DRAM path | Per-chunk ledger |
|---|---:|---|
| c0 combined values | 3 BF16 tiles,6144B | Reader pushes 320 groups of 3 pages; compute waits/pops 3 for every slot, including skipped slots. |
| c1 scalar weights | 1 BF16 tile,2048B | Writer pushes 320 pages; compute waits/pops 1 per slot. |
| c2 ownership table | 1 page,2048B | Writer loads once, compute waits; remains resident for all chunks. |
| c3 IDs | 32 aligned 64 B pages,2048B | Writer pushes 32, compute waits 32 and pops 32 after the chunk. |
| c4 weight scratch | 2048B | Writer-only32-token preload; each row stride 64 B. |
| c17 row-major result | 96 BF16 tiles,196608B | Reserve 96, indexed writes, push 96; tilize consumes 96. |
| c16 tiled result | 96 BF16 tiles,196608B | Tilize produces 96; writer waits/pops 96 and writes 80 live output tiles. |

The raw IDs/weights row is20 B, but physical DRAM row stride is 64 B; readers consistently use the aligned stride. Each token's c17 scratch row has 3072 elements, of which 2560 are live. Packing at `i*3+j` followed by tilizing a 32×3072 rectangle places padding on the right. Writing the first 80 horizontal output tiles crops the padded columns; it does not concatenate padded token rows into a2560-wide output. [Reader][reader], [compute][compute], [writer][writer].

Ownership is checked with table entry `id`, where INT32 `-1` means nonlocal. The caller fills precisely `[rank*128,(rank+1)*128)` with local entries. IDs stay UINT16 and are read at `(token_page, expert_slot)`; no INT32 expansion participates. The two active rank0 slots in the one-local fixture genuinely reach the overwrite branch, with all other slots consumed and skipped. No queue-count mismatch distinguishes the passing and failing comparisons.

The [broadcast API][bcast] lowers to scalar ELWMUL with combined data in SrcA and weights in SrcB. The Blackhole [binary LLK][binary] advances phases 0→1→2→3, clears the fidelity counter after the final phase, retains scalar SrcB across the face runs, and releases it afterward. Multiply initialization resets the relevant counters. Startup, BF16 pack formats, fast-tilize init/uninit, and pack/unpack restoration are coherent on this selected branch. No concrete state-lifecycle violation was found. [Tilize selection and reconfiguration][tilize].

Host op calls enqueue work; their return alone is not completion evidence. The diagnostic reads every per-rank result through `to_torch` before computing equality, including repeat outputs. The shared-pool probe also records readbacks/synchronizations. These completed numeric comparisons provide the relevant observation boundary. The report does not infer correctness merely from a host function returning or from queued assertions.

**Minimal remaining discriminators, preserving the accepted policy**

These are suggested controls, not performed hardware work or instructions to stop independent gates.

1. **Capture the first real divergent boundary.** On identical first-step request inputs/state, compare layer 0 normalized MLP input, router logits, paired IDs/weights, counts/offsets, owned combined values, routed reduction, shared output, per-rank added partial, CCL result, and residual write. Capture IDs with their paired weights; do not infer expert-slot correspondence from unlabelled samples. The existing layer 1 state delta only bounds the divergence upstream.
2. **Replay real pre-reduction operands on CPU.** Apply the arithmetic replay validated on the saved finite fixtures to a captured B2 operand set. Compare B1 and B2 producers separately before comparing their reductions. Exact replay agreement would clear the native reduction for that captured input while identifying whether routing/expert outputs already differ. Keep BF16 storage/DEST, HiFi4, expert precision, and CCL settings unchanged.
3. **Use a one-active-slot addressing control only if needed.** Keep K=10 and the same native geometry; move one identical owned value/weight pair through slots 0..9, with finite nonowned poison data. With weight 1 or exact powers of two, place feature markers across 15/16,31/32,255/256,1023/1024,2047/2048,2559. Equal outputs isolate slot/face/page addressing without a change in summation order or precision. Ownership endpoints 127/128,255/256,383/384,511 give an independent table-boundary check. Reducing K to 1 is a weaker contrast because it also changes both CB batching decisions.
4. **Separate layout only if the real fixture disagrees with arithmetic.** Capture c17 before tilize and compare it with decoded final output. This retains the optimized batched kernel and its arithmetic while distinguishing product/accumulation from the layout phase.

The smallest justified intervention now is to correct the diagnostic golden to model the selected arithmetic, then localize the real-model discrepancy. A model/kernel fix is not established by these artifacts. In particular, silently switching native DEST to FP32 would alter the accepted numerical policy; looping over `_reduce_single` would alter the optimized execution path. Neither follows from the findings.

**Other Potential Issues and findings rejected on review**

- **No-local NaN poisoning is a test gap, not a proven bug.** Compute forces the final slot when none are local; writer supplies zero weight. An IEEE `NaN*0` argument initially looks suspicious, but the selected FPU uses source-zero gating rather than IEEE NaN propagation. The documented arithmetic predicts zero in this case. The synthetic fixtures also zero nonowned rows. A nonfinite poison control can test robustness, but this cannot headline the observed finite drift.
- **Unsorted route slots are not automatically wrong.** IDs and weights stay paired; changing their common order preserves the mathematical sum. In finite precision, reordering multiple local contributions can alter rounding. Report the actual paired sequence if a route delta appears; do not label order alone an ownership defect.
- **Persistent low DEST residue is not needed to explain these fixtures.** The published ISA discusses history-sensitive low state when mixing destination widths. The zero-state replay already matches every native output. A history-order control becomes relevant only if a future controlled test actually shows history dependence; current repeats do not show it.
- **Stale comment:** device validation lines 112–113 still describe writer expansion of UINT16 indices to INT32. The current writer/compute retain UINT16. This documentation discrepancy is not executable behavior.
- **Scope/provenance:** v1/v2 hash the model file, not the probe script. Current model SHA256 is `7effbc77ea8edc11cef524e1bf68b0bb6e2eff4bf134d451d83b784ba467d5bc`, matching both artifacts; all four recorded native source hashes also match. The probe itself is unpinned, shared-pool v5 has no source hashes, and the full-model evidence covers real layers 0/1/3, not all 48 layers. No all-model or arbitrary-shape kernel correctness claim is made.

**Report review**

The initial candidate “serial BF16 addition explains the discrepancy” was rejected as incomplete because the one-owned-expert path performs no such addition. The exact product-phase replay closes that missing link. The apparent scratch-width mismatch was instantiated through tilize and writer cropping and is coherent. The apparent no-local NaN bug was demoted after auditing target arithmetic rather than importing IEEE assumptions. Rechecked claims separate exact saved-fixture reproduction from the still-unlocalized full-model gate failure.

**Executable CPU replay appendix**

The following diagnostic is embedded in this report to avoid modifying repository source. It handles the saved ordinary finite fixtures; it is not a general FPU simulator. Run it using `python_env/bin/python` after extracting this Python code block. It imports only CPU tensor/file utilities and does not import TTNN. Script SHA256: `bf9a13c78a5513a3860e3dc44e6f2efb0f46b7abae37a56adb14b0b811f411df` (UTF-8 contents between the fences, including the final newline). The input hash is asserted before deserialization. The observed JSON follows the script.

```python
import hashlib
import json
from pathlib import Path

import torch

artifact = Path('/home/ttuser/qwen38-flash/reference/moe_reduction_rows_v2.pt')
artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
assert artifact_sha == 'fc993d7611099f85665d8b6772d8efcdd8b12be470afbb2c0af1d986bf938614'
cases = torch.load(artifact, weights_only=True, map_location='cpu')


def shift_round(value, shift):
    if shift == 0:
        return value
    return 0 if shift >= 63 else (value + (1 << (shift - 1))) >> shift


def phase(a, b, dest, index, low=0):
    ae, be = (a >> 7) & 255, (b >> 7) & 255
    am, bm = (a & 127) << 3, (b & 127) << 3
    av = ((am >> 1) & 31) if index & 1 else 16 | (am >> 6)
    bv = ((bm & 15) << 3) if index & 2 else 64 | (bm >> 4)
    product = av * bv if ae and be else 0
    exponent = ((ae + be if ae and be else 0) - 127
                - (5 if index & 1 else 0) - (7 if index & 2 else 0)) & 1023
    if exponent == 0 or exponent >= 512:
        exponent = product = 0
    if (a ^ b) & 32768:
        product = -product
    de, ds = (dest >> 7) & 255, dest >> 15
    dm = (((128 | (dest & 127)) << 16) if de else 0) | low
    common = max(exponent, de)
    acc = shift_round(product, min(common - exponent, 31)) << 13
    term = shift_round(dm, min(common - de, 31) + 13) << 13
    acc += -term if ds and de else term
    if common == 0 or acc == 0:
        return 0
    sign, magnitude = int(acc < 0), abs(acc)
    top = magnitude.bit_length() - 1
    result_exp = common + top - 23
    normalized = magnitude << (28 - top)
    rounded = (normalized >> 21) + ((normalized >> 20) & 1)
    if rounded >> 8:
        rounded >>= 1
        result_exp += 1
    if result_exp <= 0:
        return 0
    if result_exp >= 255:
        return (sign << 15) | (255 << 7)
    return (sign << 15) | (result_exp << 7) | (rounded & 127)


def add_bf16(a, b):
    ae, be = (a >> 7) & 255, (b >> 7) & 255
    if not ae:
        return b
    if not be:
        return a
    exponent = min(ae, be)
    value = (-1 if a >> 15 else 1) * ((128 | (a & 127)) << (ae - exponent))
    value += (-1 if b >> 15 else 1) * ((128 | (b & 127)) << (be - exponent))
    if not value:
        return 0
    sign, magnitude = int(value < 0), abs(value)
    top = magnitude.bit_length() - 1
    drop = top - 7
    if drop > 0:
        quotient = magnitude >> drop
        remainder = magnitude - (quotient << drop)
        half = 1 << (drop - 1)
        quotient += remainder > half or (remainder == half and quotient & 1)
    else:
        quotient = magnitude << -drop
    result_exp = exponent + top - 7
    if quotient == 256:
        quotient >>= 1
        result_exp += 1
    return (sign << 15) | (result_exp << 7) | (quotient & 127)


def words(tensor):
    return tensor.contiguous().view(torch.int16).to(torch.int32) & 65535


report = {'cpu_only': True, 'artifact_sha256': artifact_sha, 'cases': []}
for case in cases:
    fixture = case['fixture']
    values = words(fixture['combined']).tolist()
    weights = words(fixture['weights']).tolist()
    ids = fixture['ids'].tolist()
    actual = words(case['native_batch']).tolist()
    mismatches = total = 0
    for rank in range(4):
        for row in range(2):
            local = [k for k, expert in enumerate(ids[row])
                     if rank * 128 <= expert < (rank + 1) * 128]
            for col in range(2560):
                accumulator = 0
                for k in local:
                    product = 0
                    for index in range(4):
                        product = phase(values[rank][row][k][col],
                                        weights[row][k], product, index)
                    accumulator = add_bf16(accumulator, product)
                mismatches += accumulator != actual[rank][row][col]
                total += 1
    report['cases'].append({'pattern': fixture['provenance']['pattern'],
                            'different_bits': mismatches, 'elements': total})

case = cases[1]
fixture = case['fixture']
away_mismatches = rne_mismatches = midpoints = low_mismatches = 0
for row in range(2):
    k = ((fixture['ids'][row] >= 0) & (fixture['ids'][row] < 128)).nonzero().item()
    a = fixture['combined'][0, row, k]
    b = fixture['weights'][row, k]
    exact_product = a.float() * b.float()
    bits = exact_product.view(torch.int32).to(torch.int64) & 0xffffffff
    away = (((bits + 0x8000) >> 16) & 65535).to(torch.int32)
    rne = words(exact_product.to(torch.bfloat16))
    actual = words(case['single'][0, row])
    away_mismatches += int((away != actual).sum())
    rne_mismatches += int((rne != actual).sum())
    midpoints += int(((bits & 65535) == 32768).sum())
    native = words(case['native_batch'][0, row]).tolist()
    for col, av in enumerate(words(a).tolist()):
        product = 0
        for index in range(4):
            product = phase(av, int(words(b)), product, index, low=4096)
        low_mismatches += product != native[col]
report['one_local_rank0'] = {
    'elements': 5120, 'b1_ties_away_mismatches': away_mismatches,
    'b1_rne_mismatches': rne_mismatches, 'exact_product_midpoints': midpoints,
    'native_uniform_low4096_mismatches': low_mismatches,
}
print(json.dumps(report, indent=2))
assert all(case['different_bits'] == 0 for case in report['cases'])
assert away_mismatches == 0
```

```json
{
  "cpu_only": true,
  "artifact_sha256": "fc993d7611099f85665d8b6772d8efcdd8b12be470afbb2c0af1d986bf938614",
  "cases": [
    {"pattern": "zero_local_rank0", "different_bits": 0, "elements": 20480},
    {"pattern": "one_local_rank0", "different_bits": 0, "elements": 20480},
    {"pattern": "multiple_each_rank", "different_bits": 0, "elements": 20480}
  ],
  "one_local_rank0": {
    "elements": 5120,
    "b1_ties_away_mismatches": 0,
    "b1_rne_mismatches": 190,
    "exact_product_midpoints": 416,
    "native_uniform_low4096_mismatches": 501
  }
}
```

[pool]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_generator_v5.json
[v1]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/full_model/moe_reduction_rows_v1.json
[v2]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/full_model/moe_reduction_rows_v2.json
[batch]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py:981
[moe]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py:672
[single]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py:790
[reduce]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py:749
[loader]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py:33
[wrapper]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/post_combine_reduce.cpp:16
[validation]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_device_operation.cpp:90
[factory]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:45
[reader]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_reader.cpp:75
[compute]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_compute.cpp:80
[writer]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_writer.cpp:150
[bcast]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/inc/api/compute/bcast.h:403
[binary]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:164
[tilize]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:79
