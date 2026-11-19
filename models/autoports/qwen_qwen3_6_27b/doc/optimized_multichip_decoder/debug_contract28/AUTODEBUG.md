# AutoDebug: inherited GDN LoFi policy fails the HF contribution contract

**Finding:** recorded TP4 LoFi attention arithmetic does not meet the unchanged 0.995 per-token HF contribution gate. Frozen stage4 reproduces the entire failing S31 metric entry. Changing only `attention_fidelity="HiFi4"` passes the complete saved layer-0 contract. This establishes a fidelity-policy repair boundary; it does not identify a defective kernel or the first inaccurate intermediate.

**Smallest demonstrated correction:** retain the existing decoder and use the **GDN attention HiFi4 default**, preserving the BF16 HF oracle, all original gates, weights, state ownership and explicit policy overrides. That one-line default is **already present in the inspected checkout** at `tt/multichip_decoder.py:78`. This investigation made no source edits. A prefill-only version is source-feasible, but needs additional phase plumbing and has no saved isolated pass.

Paths below are relative to `models/autoports/qwen_qwen3_6_27b/`; **D** means `doc/optimized_multichip_decoder/`.

## Direct observations and provenance

| Saved run | Effective attention policy | Recorded result |
|---|---|---|
| `D/final_validation28_contract_l0_tp4.json` | LoFi | S1 passes. B1/S31 output PCC **0.9993891826645709**, contribution row **12** PCC **0.9860137708810336**, finite. Stops at this gate. |
| `D/final28_contract_stage4_l0.json` | Frozen stage4 LoFi | Same entire S31 entry, including **all 31 contribution PCCs**. S1 also passes, with different metrics. Stops at S31. |
| `D/final28_contract_hifi4_l0.json` | Current-at-run source + HiFi4 override | **Pass**: 12 lengths, 7 independent-HF resume checks and the two-tail mixed-list case. S31 row12 improves to **0.9997051475414599**; output PCC **0.9999459690610657**. |

HiFi4 checks lengths `1,31,32,33,127,128,129,511,512,513,1025,17`; worst contribution is **0.9979419678784937** (S1025). Resume starts `1,32,33,128,511,512,544` share a 545-token fixture; the last checks one token. All positive HF aggregate/output-row/contribution-row gates and TP4 self-consistency gates pass; all dropped-history negative controls meet their required `<0.995` condition. Mixed tails have lengths 33/17, starts 32/33 and minimum contribution **0.999283025520026**.

Default versus HiFi4 selected policies differ **only** in attention fidelity. Their command-recorded model/native/build/environment manifests match. `D/final28_contract_hifi4.py:5–10` only supplies this construction override; the harness and assertions are unchanged. The stage4 loader verifies frozen decoder SHA256 `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`, also matching the decoder at Git HEAD `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. This is original TP4 code under the shared current runtime, not proof of historical native-runtime behavior or bit-identical raw tensors.

**Revision distinction:** receipts pin then-current production decoder SHA256 `8ceccf19b6b77228210e5fd4e8b03777054efe0a97d92eec4c7d152543f2aa46`; stage4's substituted decoder is identified separately by `diagnostic_decoder_source`. Current production source is `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b`. Removing only the added GDN default line `"attention_fidelity": "HiFi4",` from an in-memory string reproduces the old hash exactly. Current default therefore carries the passing override's effective policy; its new hash still needs a normal production-default receipt. The harness, optimized decoder, flat-prefill and HF-config result pins match the checkout. Among the HiFi4 command's model/test source pins, only the production decoder differs.

## Source causal chain

1. **The inherited policy overrides the apparent HiFi4 config.** Base policy supplies LoFi (`tt/multichip_decoder.py:134`; frozen stage4 `:61`). Attention config initially chooses HiFi4 for GDN (`:189–196`), but the role override rebuilds it from policy (`:205–213`). The added stage5 GDN default at `:78` now selects HiFi4 through expansion at `:159`, before caller overrides at `:160`. This line leaves full-attention defaults and MLP fidelity unchanged.
2. **B1/S31 uses ordinary multi-token prefill.** Flattening requires B>1 (`tt/flat_prefill.py:15–33`), so scheduling calls `_legacy_chunk → _gdn` (`:151–155,194–204`). `_gdn` projects `[1,31,5120]`, pads sequence31 to32, and selects native carry convolution. Tiled convolution requires B>1 unless its separate B1 flag is set; that flag is absent. Packed decode convolution requires S1 (`tt/multichip_decoder.py:1121–1176`). Chunk recurrence uses local Hk4/Hv12, chunk32 and unchanged HiFi4 `self.ckc` (`:1257–1272`).
3. **The override changes several arithmetic producers.** Packed projection uses the owner's attention config (`tt/packed_correction_candidate.py:261–295`); native carry convolution receives it (`tt/multichip_decoder.py:1171`); separate output projection uses it through inherited `_linear_raw` (`tt/optimized_decoder.py:499–523`). S31 is below `mmrs_min_rows=256`, so fused output MMRS is not selected (`tt/multichip_decoder.py:534–539`). FP32 destination accumulation, precise math, GDN packer-L1-acc=false and stored tensor dtypes remain unchanged.
4. **Native convolution fidelity is live.** Under `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/`, wrapper `qkv_causal_conv1d_silu.cpp:118–125,39–45` propagates the supplied config; `operations/core/compute_kernel/compute_kernel_config.cpp:30–31` preserves it. Factory `device/qkv_causal_conv1d_silu_program_factory.cpp:474` forwards it to hardware, and compute `device/kernels/compute/qkv_causal_conv1d_silu.cpp:62–70` performs diagonal matmuls.

**Interpretation:** inherited LoFi attention arithmetic is the leading diagnosis. Stage5 packed correction is unnecessary to produce this symptom: frozen stage4 lacks that path and reproduces the S31 metric vector. The HiFi4 intervention changes projection and convolution together. It does not establish which producer first diverges or exclude a shared defect masked by higher fidelity. No matched intermediate tensors support a deeper kernel accusation.

## Broad default versus prefill-only correction

**Recommend retaining the already-present GDN HiFi4 default as the smallest demonstrated gate repair.** It uses an existing policy and introduces no fallback. It also changes actual decode projections, so old LoFi decode timing/quality receipts cannot become final HiFi4 evidence by relabeling.

A semantic **prefill-only GDN attention HiFi4 policy** could retain this contract's passing arithmetic: every harness forward is `prefill_forward` (`tests/run_multichip_contract.py:122–129`). It requires the following complete propagation, rather than a row-size condition:

| Boundary | Necessary propagation |
|---|---|
| API phase | Carry explicit prefill/decode identity, including S1 prefill and resume tails. B1/S31 prefill and compact B31 decode both become `[1,31,H]`. `decode_batch is None` distinguishes them only with selected `flat_decode=True`; real noncompact decode also passes None (`tt/multichip_decoder.py:1299–1307`). |
| Packed input projection | Ensure semantic prefill, including <=32 rows, uses prefill config. Today `_linear` selects `project(...phase="decode")` for B1/S31 (`:526–530`). Both phase contracts currently use the same attention config and FP32 output/accumulation, so merely calling `project_prefill` changes no arithmetic (`tt/packed_correction_candidate.py:156–164,261–315`). Also cover `_prefill_packed_gdn` in flat prefill and separate primary/correction paths when explicitly selected. |
| Convolution | Use prefill attention config in native carry convolution (`tt/multichip_decoder.py:1171`). B>1 tiled prefill convolution already hardcodes HiFi4 (`tt/tiled_prefill_conv_candidate.py:287–292`); preserve it. |
| Output projection | Cover S1/multi-token `_gdn` returns (`tt/multichip_decoder.py:1254,1286`) **and** the selected flat B>1 path (`tt/flat_gdn_prefill_candidate.py:183`). Propagate through inherited raw matmul/chunk recursion (`tt/optimized_decoder.py:499–523,590,603`) and the independently derived fused-MMRS fidelity descriptor (`tt/multichip_decoder.py:545–574`). |

Prepare immutable configs during setup and pass phase/config locally. Do not temporarily mutate shared `policy`/`matmul_configs` in forward. One new dictionary key is insufficient. A projection-only or convolution-only HiFi4 fix, row12/S31 exception, or `rows>32` rule is **not supported by the saved contrast**. No new optimization sweep is warranted under the user's cap.

## Other discrepancies and limits

- The CPU diagnosis's “current effective LoFi” claim is stale after the added default. Its claim that `load_layer()` seeds17 is false: `tests/run_decoder.py:28–39` only loads weights; seed17 belongs to its separate `main` (`:58`). Contract seed183 is at `tests/run_multichip_contract.py:43`; HF construction, page permutation and prior S1 generation still matter for exact fixture recreation.
- Its description of B>1 tiled convolution as SFPU-based misses the reused native diagonal-matmul compute (`tt/tiled_prefill_conv_candidate.py:170–189`), already configured HiFi4. Its narrow-repair map also omits the selected flat GDN output call at `tt/flat_gdn_prefill_candidate.py:183`.
- LoFi runs did not execute later lengths, resume or list tests. HiFi4 covers layer0 eager prefill, including resumed state use; it does not cover `decode_forward`, trace replay or long stress beyond1025. The mesh has `trace_region_size=0` (`tests/run_multichip_contract.py:93`). Boundary device-to-host reads (`:107–109,144`) materialize output before PCC: this failure is not inferred merely from enqueued work.
- List comparisons lack an explicit returned-count assertion before `zip` (`tests/run_multichip_contract.py:231–253`). Both expected tails are present in the receipt, so this test gap does not explain S31.
- No evidence supports changing HF precision, removing residual-subtracted contribution checks, lowering 0.995, changing fixtures/support, or substituting a baseline runtime fallback. The HiFi4 pass disproves intrinsic incompatibility with this BF16 HF oracle for the recorded contract.

## Required closure outside this inspection

Retain failed and diagnostic receipts. Obtain production-default validation of the selected source with the unchanged contract and remaining required stage gates, including actual decode/state quality, selected long stress and changed-input trace replay where required. Refresh final source pins and timing affected by fidelity; keep frozen baseline controls immutable. These are gate/provenance repairs, not authorization to restart optimization. Any prefill-only implementation needs its own unchanged contract pass and decode checks; it cannot inherit the broad control's pass by assertion.

Post-draft review: independently rechecked the causal chain and gate/provenance claims against source and receipts. Standard-library assertions verified the sole failing row, fidelity-only policy difference, every saved gate, exact source delta, immutable stage4 source and preserved norm report. No material corrections remained.

Inspection only, 2026-11-13: source reads, JSON parsing and standard-library hash/static comparisons. No devices, target execution, builds, source edits or AutoDebug relaunch. Only `./AUTODEBUG.md` and `D/debug_contract28/AUTODEBUG.md` are investigation outputs. The unrelated `D/debug_norm_capacity27/AUTODEBUG.md` is preserved (SHA256 `7a7202f30ae3349171d896058faae1eb68fc0cf0d62fb1ebd5dca3ebd663d376`).
