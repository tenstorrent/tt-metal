# 06 — Gate ledger (append-only)

Verdicts: `PASS` · `FAIL` · `PASS-WITH-DEVIATION` (needs a `DEC`) · `BLOCKED` (needs a `07_RISKS.md`
entry naming the blocker) · `NOT-RUN` (needs the reason). A gate with no raw log did not happen.

| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| G-CARD | P0 | every architectural fact has provenance | doc review + 0 bad citations | 53/53 citations verified, 115/115 doc refs resolved, 0 `UNVERIFIED` rows | PASS | 2026-09-04 | `raw/G-CARD_20260904T034338Z.log` (+ `raw/G-CARD_20260904T034303Z.log`) |
| G-REF | P1 | the reference is deterministic and self-consistent | bit-identical x2; hand-written vs HF PCC >= 0.9999 (expect bit-exact) | 10/10 tests pass; both oracles SHA-256 `0a163300…`; PCC **1.0**, `max|Δ| = 0.0` | PASS | 2026-09-04 | `raw/G-REF_20260904T035140Z.log` |
| G-SURVEY | P2 | reuse decided, with citations | doc review + 0 bad citations | 30 component rows, 30 with a decision, 27 with a full `path:line`; 123/123 citations verified, 195/195 doc refs resolved | PASS | 2026-09-04 | `raw/G-SURVEY_20260904T035957Z.log` |
| G-OUTLINE | P3 | file tree + shapes pinned; every gate owns a file | doc review + 0 bad citations | 41 files contracted (49/49 non-`__init__` tree files); **32/32** Appendix A gate rows owned; 18/18 shape rows filled; 225/225 citations, 338/338 doc refs | PASS | 2026-09-04 | `raw/G-OUTLINE_20260904T083915Z.log` |
| G-CCL-PLAN | P4 | every collective placed and justified | doc review + 0 bad citations | 8/8 module placement rows justified; **9** collective call sites with `dim`/`axis`/topology/`num_links`; 14 semaphores (6+4+2+2), barrier depth 2; 253/253 citations, 404/404 doc refs | PASS | 2026-09-04 | `raw/G-CCL-PLAN_20260904T084559Z.log` |
| G-MESH | P5.1 | `MeshConfig` arithmetic + refusals; `CCLManager` builds and allocates its semaphores once | exact asserts | 16/16 tests pass; `shard_size(4096)=512`, `shard_size(14336)=1792`; grid **(12,10)**, CCL offset **(11,0)**; semaphores **6/4/2/2 = 14**, unchanged after 128 barrier cycles; 4/4 sub-axis-TP shapes refused | PASS | 2026-09-04 | `raw/G-MESH_20260904T085727Z.log` |
| G-RMS | P5.2 | plain RMSNorm vs an fp32 torch reference, `(1,1)` | PCC >= 0.9999; gap to the floor recorded | random weights **0.9999957 / 0.9999958 / 0.9999957** (floor 0.9999973/0.9999973/0.9999972 -> **1.56 / 1.57 / 1.54x**); real layer-0 weights **0.9999971** x3 (floor 0.9999986 -> **2.11 / 2.13 / 2.12x**) | PASS | 2026-09-04 | `raw/G-RMS_20260904T090144Z.log` |
| G-ROPE | P5.3 | llama3-scaled RoPE + the Meta convention, `(1,1)` | PCC >= 0.999; control must collapse | **0.9999969 / 0.9999964 / 0.9999959** (floor 0.9999983/0.9999982/0.9999980 -> **1.76 / 1.95 / 2.08x**); control **0.01367**; tables bit-identical to the test's own Meta tables | PASS | 2026-09-04 | `raw/G-ROPE_20260904T091040Z.log` |
| G-MLP | P5.4 | dense SwiGLU (`down(silu(gate)*up)`), `(1,1)` | PCC >= 0.999 @bf8_b, >= 0.9995 @bf16, **<= 3x floor** at each | bf8_b **0.9999133 / 0.9999144 / 0.9999144** (floor 0.9999213/0.9999221/0.9999222 -> **1.10 / 1.10 / 1.10x**); bf16 **0.9999851 / 0.9999852 / 0.9999852** (floor 0.9999929 -> **2.11 / 2.10 / 2.09x**); control **0.64715 / 0.64722** | PASS | 2026-09-04 | `raw/G-MLP_20260904T093653Z.log` |
| G-ATTN | P5.5 | GQA + full RoPE + causal SDPA + `o_proj`, `(1,1)` | PCC >= 0.999; own stages <= 3x floor; block <= 8x | bf8_b block **0.9997364 / 0.9997080 / 0.9996723** (raw **2.21 / 2.17 / 2.22x**); bf16 block **0.9998463 / 0.9998275 / 0.9998029** (raw **12.32 / 11.82 / 12.17x**, SDPA-attributed residual **1.10 / 1.01 / 0.70x**); stages **1.00-2.50x**; fused SDPA **26.7-28.6x** in-pipeline, **52.8-55.0x** standalone; control **0.51174** | PASS-WITH-DEVIATION (`DEC-042`) | 2026-09-04 | `raw/G-ATTN_20260904T095359Z.log` |
| G-KV | P5.6 | KV cache **primitive**: write correctness, position map, no collateral writes | PCC >= 0.99 @bf8_b, <= 3x floor; positional read-back **bit-exact** | bf8_b worst-of-8-heads K **0.9999743** / V **0.9999752** (**1.00x** the floor at every head and both seq lens); bf16 **0.9999986** (**1.00x**); 128 positions x 4 `kv_actual` offsets **bit-identical**; pad tail + 3 other (user, layer) slots **exactly zero**; bf8_b costs **17.8x** on K / **17.7x** on V vs bf16 | PASS | 2026-09-04 | `raw/G-KV_20260904T100312Z.log` |
| G-LAYER | P6.1 | one decoder layer, norm->attn->residual->norm->MLP->residual (integration check) | PCC >= 0.999, <= 8x floor | bf8_b **0.9997665 / 0.9998273 / 0.9998736** (floors 0.9998709/0.9998953/0.9999138 -> **1.81 / 1.65 / 1.47x**); bf16 **0.9998774 / 0.9999172 / 0.9999480** (floors 0.9999826/0.9999859/0.9999885 -> **7.05 / 5.86 / 4.51x**); SDPA-attributed residual **1.13-1.15x** / **2.01-2.14x**; real weights + real input **0.9998649** (floor 0.9999647 -> **3.82x**); controls **0.99864** / **0.99993** / **0.66830** | PASS | 2026-09-04 | `raw/G-LAYER_20260904T113153Z.log` |
| G-WEIGHTS | P6.2 | no missing/unused keys; cache-only rebuild identical; loader bit-exact | exact | **291/291** keys, 0 missing, 0 unused (all 32 layers); **12/12** device tensors `max\|delta\| = 0.000e+00` through transpose + Q/K Meta swizzle + dtype ladder; **12/12** SHA-256 identical on a cache-only rebuild; 3/3 controls discriminate | PASS | 2026-09-04 | `raw/G-WEIGHTS_20260904T113320Z.log` |
| G-MODEL | P6.3 | full stack hidden states; top-1 agreement | at 2 and 4 layers PCC >= 0.999 and <= 8x floor; at full depth per-layer step <= 4x from L3; 100% top-1 at every depth (`DEC-053`) | L2/s128 **0.9997314** (floor 0.9998795 -> **2.23x**); L4/s128 **2.36x**; **L32/s512 0.9984849**, worst step **1.27x**, top-1 374 == 374; corrected floor -> **1.53x** (`R-021`); control **0.16180** | PASS | 2026-09-04 | `raw/G-MODEL_20260904T115256Z.log`, `raw/G-MODEL-H5_20260904T114724Z.log`, `raw/G-MODEL_per_layer_pcc.json` |
| G-CHUNK | P7 | chunked == one-shot for **deltas 1-2** (indexed RoPE, advancing chunk write), and both vs the fp32 golden | >= 0.999 mutual **per layer** (expect exact); >= 0.99 K / >= 0.98 V vs golden; L0 ratio <= 3x; step <= 4x from L3; delta 1 alone **bit-exact** | delta 1 in isolation **`torch.equal`, max\|delta\| = 0.0** on all 4 chunks; mutual **1.0000000** on K and V at **32/32** layers; vs golden K min **0.9990570** (L13) / V min **0.9957049** (L26); L0 K **1.10x** its complete floor (1.35x storage), V **1.02x** (1.70x storage); worst gated step K **1.90x** (L13), V **1.44x** (L11); controls **0.72466** (delta 1, K-only) and **0.22048 / 0.04473** (delta 2) | PASS | 2026-09-04 | `raw/G-CHUNK_20260904T132854Z.log`, `raw/G-CHUNK_per_layer_pcc.json` (earlier identical runs: `raw/G-CHUNK_20260904T124825Z.log`, `raw/G-CHUNK_20260904T131500Z.log`) |
| G-GOLDEN | P7 | the fp32 golden trace's structure and content over all 32 layers | clean per-layer table; generator + verifier exit 0; streamed driver == `LlamaModel`'s own loop bit-exactly; a zeroed and a deleted layer must both exit non-zero | 32 layers / 512 tokens / **fp32**, 128.0 MB; streamed vs `LlamaModel` `max\|delta\|` **0.0** on K, V and the post-norm hidden over **all 32** layers (`torch.equal`); verifier **exit 0**; zeroed-layer control **exit 1** with 7 named problems; deleted-layer control **exit 1** | PASS | 2026-09-04 | `raw/G-GOLDEN-GEN_20260904T123642Z.log`, `raw/G-GOLDEN_20260904T125230Z.log` |
| G-RUNTIME | P7 | the runtime satisfies the engine's **real** call site, statically; every refusal loud and matched | every unguarded name and parameter present; every refusal matched on its message; the audit itself must reject a doc-faithful runtime | **37/37** tests, no device, 8.2 s. Engine call site: **11** runtime attributes, **2** `config` fields (`is_last_rank`, `use_trace`), **9** called methods, **6** `getattr`-guarded hooks. Audit of `TtPrefillRuntime`: **0** missing attributes, **0** missing config fields, **0** signature problems. Control `_BrokenRuntime`: **3** missing methods, **1** missing config field, **4** signature problems including `metadata_msg`. **25** refusals asserted against **25** `raise` statements | PASS | 2026-09-04 | `raw/G-RUNTIME_20260904T131649Z.log` |
| G-CHUNK-ATTN | P7 -> **P8** | chunk *k* attending the prefix read back out of the cache (**delta 3**) | >= 0.999 at layer 1; deep layers by step <= 4x; both vs golden | **not measured** — needs the ring path and TP=8, neither of which P7 owns. The configuration is **refused** at two levels rather than approximated | **BLOCKED** (`R-023`, owner **P8**) | 2026-09-04 | n/a — see `raw/G-RUNTIME_20260904T125242Z.log` for the refusal that stands in its place |
| G-FABRIC-MATRIX | P8 | which (mesh, topology, links, axis) combinations can run a collective on ONE galaxy | every case matches its **stated** expectation, subprocess-isolated with a timeout | **7/12** matched. The 5 that did not are the finding: all four `FABRIC_1D_RING` cases `error` (the only single-galaxy RING/RING descriptor does not map — `topology_mapper.cpp:544`, "32 target node(s) are not mapped to any global node"), and `Ring`-on-`FABRIC_1D` was `ok` and **bit-exact** where the recipe says it hangs. Addendum **5/5**. `overlap...no_quiesce` **hang** (246.3 s, box reset 43.1 s); `...quiesce` **ok**. Harness control: a wrong expectation was reported, not absorbed | PASS-WITH-DEVIATION (`DEC-071`, `DEC-079`, `DEC-081`) | 2026-09-04 | `raw/G-FABRIC-MATRIX_20260904T142819Z.log`, `raw/G-FABRIC-MATRIX-ADDENDUM_20260904T144233Z.log`, `raw/G-FABRIC-MATRIX-CONTROL_20260904T144434Z.log` |
| G-KV-TP8 | P8 | the **model -> cache** path at TP=8 on a `(1,8)` submesh | head->column **bit-exact** (`rtol=atol=0`); K >= 0.99 / V >= 0.98 vs the fp32 golden; L0 ratio <= 3x; rotated-column control | head->column **8/8 columns bit-identical** on both probes (V with RoPE, K without); write offsets {0,128,256} x 8 columns **24/24 bit-identical**; pad tail **exactly 0**; control (read column `(c+1)%8` as head `c`) **PCC 0.99887, `torch.equal` False, max\|delta\| 7.0** — recipe §2.5 measured 0.99890; arm B over 32 layers **min K 0.9986432** (L22) / **min V 0.9942853** (L26); L0 K **1.10x** its complete floor (1.35x storage), V **1.02x** (1.70x storage); pad tail 0.0 over 64 (layer, cache) pairs | PASS | 2026-09-04 | `raw/G-KV-TP8_20260904T144803Z.log`, `raw/G-KV-TP8_per_layer_pcc.json` |
| G-SP-RING | P8 | the ring-joint SP attention core **alone**, and the `fp32_dest_acc_en` A/B | PCC >= 0.99 vs an fp32 torch reference; ratio to its own floor **recorded** | **0.9996672** against a **0.9999450** floor -> **6.05x** (the recipe's figure for this op is 7.98x; the single-card SDPA is 52.8-55.0x standalone). `fp32_dest_acc_en=True` **REFUSED**: `ring_joint_sdpa_program_factory.cpp:1308`, "kv_actual_isl requires the ring-joint streaming compute path"; `False` accepted, output `(1,4,128,128)`. Controls: correct `kv_cache_batch_idx` **0.9996770** vs `batch_idx=slot` **-0.00337**; `kv_actual_isl=0` **refused** (`ring_joint_sdpa_device_operation.cpp:278`) | PASS | 2026-09-04 | `raw/G-SP-RING_20260904T145354Z.log` (the `Topology.Ring` abort that settled `DEC-081`: `raw/G-SP-RING_20260904T145034Z.log`) |
| G-CHUNK-ATTN | P8 | **delta 3** — chunk *k*'s queries attending the prefix read back out of the cache | **>= 0.999 at layer 1**; deep layers by per-layer step <= 4x from L3; both arms vs the fp32 golden at `G-CHUNK`'s thresholds; accumulated min recorded, not gated | L0 (no attention has run) mutual K **1.0000000**; **L1 (one attention layer) K 0.9999505, V 0.9997938**; L8 K 0.9995954; **min over 32 layers K 0.9967975 (L22) — recorded, NOT gated**; worst gated step K **2.14x** (L13) / V **1.58x** (L3) against 4x; vs golden one-shot min K 0.9987994 / V 0.9942686, ring min K 0.9967119 / V 0.9868228 (ring carries **2.74x** the one-shot's K error); control (chunk 1 on an unwritten prefix) L0 **1.0000000** — unmoved — L1 **0.99695**, worst **0.87279** | PASS (supersedes the P7 `BLOCKED` row above; closes `R-023`) | 2026-09-04 | `raw/G-CHUNK-ATTN_20260904T150921Z.log.gz`, `raw/G-CHUNK-ATTN_per_layer_pcc.json` |
| G-TP-PARITY | P8 | the TP collectives are exact: each module's multi-device output vs its own `(1,1)` output | PCC >= 0.999 on 5 shapes incl. `(2,8)`; control <= 0.95 | `(1,2)`/`(1,4)`/`(1,8)`/`(2,8)`/`(4,8)`, worst device of each: **rms_norm 1.0000000 at all five** (sequence-sharded, sliced); **mlp 0.9999915**; **attention 0.9999917**; **layer 0.9999733**. Control (reference rolled by one 512-feature TP shard) **0.00307**. 6 submesh shapes in one process, `quiesce_devices()` on both sides of every phase, no hang | PASS | 2026-09-04 | `raw/G-TP-PARITY_20260904T145701Z.log` |
| G-RACE | P8 | no semaphore races | 3 runs **bit-identical**, one process, one `CCLManager` | 3 runs of the full 32-layer chunked harness -> **one** hash `b7abb6481ee1efb569e072924e08256310fcf8ab96c77205c76d16cb9d639d28`; min K 0.9967119 / V 0.9868228 identical on all three; 365.0 / 366.8 / 366.8 ms. The hash also equals the separate single-run process's, so determinism holds across processes too | PASS | 2026-09-04 | `raw/G-RACE_20260904T150647Z.log.gz` |
| G-SEMAPHORE | P8 | `CCLManager` allocates its CCL state **once** | exact list lengths at construction, after dozens of getter cycles, and after a real multi-layer run | **6 RS + 4 AG + 2 barrier + 2 ring-attention = 14** at construction, after 32x4 getter cycles, after a `(4,8)` build, and after a real **32-layer** forward — unchanged at every point; ping-pong indices rs=0 ag=0 barrier=0, all inside a depth-2 ring; the runtime reused the manager passed to it rather than building a second. Control: 3 managers -> **42** semaphores | PASS | 2026-09-04 | `raw/G-SEMAPHORE_20260904T151056Z.log.gz` |
| G-MESH-KV | P8 | full-model KV vs the fp32 golden on the target `(4,8)` mesh, one-shot and chunked, at more than one chunk size | per-layer min recorded; K >= 0.99 / V >= 0.98 | **one-shot** (chunk 1024 == cache, `sp_bootstrap`): min K **0.9987994** (L22) / V **0.9942686** (L28), 220.5 ms, 4643 tok/s. **chunked c512** (2 chunks, `sp_ring`): min K **0.9967119** (L22) / V **0.9868228** (L28), 365.0 ms. **chunked c256** (4 chunks, chunk_local 64): min K **0.9967844** (L22) / V **0.9866232** (L28). Two different block-cyclic periods (128 and 64) both score, which is what makes the read-back's layout claim falsifiable | PASS | 2026-09-04 | `raw/G-MESH-KV-oneshot_20260904T150307Z.log`, `raw/G-MESH-KV-chunked512_20260904T150451Z.log.gz`, `raw/G-MESH-KV-chunked256_20260904T150549Z.log.gz` + the three `G-MESH-KV_*_per_layer_pcc.json` |
| G-WEIGHTS (P8 ext) | P8 | the cache-only rebuild **at TP=8**, where `ttnn.as_tensor` persists an already-sharded tensor | every device tensor SHA-256-identical | **354 device shards over 12 tensors, all SHA-256-identical** after a cache-only rebuild. **8** tensors genuinely sharded with **8 distinct** shard hashes each (the 8 TP columns, replicated across the 4 SP rows — the expected geometry), 4 replicated; `q_proj` and `lm_head` asserted sharded and the vocab table asserted replicated, so a silently-unsharded build fails rather than passes faster | PASS | 2026-09-04 | `raw/G-WEIGHTS-TP8_20260904T151945Z.log` |
| P8-REGRESSION | P8 | the whole package suite still passes after P8's additions | 0 failed | **196 passed, 0 failed** in 21:23 (P7 stood at 177; P8 adds 19 tests). Run on the final tree with `-p no:randomly`, `PREFILL_TRACE_DIR` at the 1024-token trace so `G-CHUNK-ATTN` runs rather than skipping | PASS | 2026-09-04 | `raw/P8-REGRESSION_20260904T155300Z.log.gz` (an earlier identical-count run, `raw/P8-REGRESSION_20260904T152156Z.log.gz`, was mutated mid-flight by `black` and is superseded — `DEC-090`) |
| G-CITE (P8) | P8 | every `path:line` and every cited **raw artefact** resolves | 0 mismatched, 0 unresolved, 0 missing artefacts | **539/539** content-checked citations (CITES 488 -> 539), **900/900** doc refs, and — from the **pass 3** this phase added — **98/100** cited raw artefacts present. The 2 missing are dangling P5.4-P5.6 references that pre-date this phase and were invisible to passes 1 and 2 (`R-042`) | PASS-WITH-DEVIATION (`R-042`) | 2026-09-04 | `raw/G-CITE_20260904T162457Z.log` |
| G-ADAPTER | P10 | the engine's adapter contract: every abstract method, every default path, and an import that pulls no device stack | the `ADDING_A_PREFILL_MODEL.md` checklist item by item; 0 abstract methods left; every `model_config` constant == `config.json`; import **measured** with no heavy module | **28/28** tests, no device. 0 abstract methods; 4/4 implemented on this class. **9/9** `model_config` constants equal `config.json` (+ `HEAD_DIM` = `derive_head_dim` = 128, and `config.json` asserted to have no `head_dim` key). Import: **40.3 ms**, heavy modules `[]` (budget 1.0 s, `DEC-101`); control (adapter + `tt/model_config.py`) reports `torch` **and** `ttnn`, so the probe discriminates. `PREFILL_MODEL=llama31_8b_d_p` resolves and memoizes; `TEST_VARIANTS["llama31_8b_d_p"]` is our instance. `weight_cache_path((4,8))` **equals** `ModelArgs`' answer; `(1,8)` differs. 3/3 refusals fire (`use_trace`, `dflash`, unset `HF_MODEL`); a disagreeing `config.json` is refused naming `num_hidden_layers` | PASS | 2026-09-04 | `raw/G-ADAPTER_20260904T173636Z.log` |
| G-REQUEST | P10 | request-mode serving through the engine, end to end, on the target mesh | every chunk accepted and served; the shutdown sentinel received; clean exit on both sides | **arm 1 (gate geometry, chunk 256 / cache 2816):** 11/11 chunks served `[0,256)`…`[2560,2816)`, sentinel received after 11 chunks, `shutdown complete`, **producer rc 0 / runner rc 0**; 2816 tokens, 11 pushes, p50 0.1 ms / p99 225.8 ms, 16.0 s of device time. **arm 2 (the real deployment geometry, chunk 8192 / cache 131072):** 2/2 chunks served `[0,8192)`, `[8192,16384)`, sentinel received, both rc 0, 13.6 s — the pair `R-039` recorded as never run | PASS | 2026-09-04 | `raw/G-REQUEST-runner_20260904T173723Z.log`, `raw/G-REQUEST-producer_20260904T173723Z.log`, `raw/G-REQUEST-DEPLOYMENT-runner_20260904T174126Z.log`, `raw/G-REQUEST-DEPLOYMENT-producer_20260904T174126Z.log` |
| G-MOCK-MIG | P10 | prefill writes correct KV **and** `build_kv_chunk_table` is right, read back device-lessly in a **second process** (the doc's Gate 1) | producer PCC >= 0.93 (`PREFILL_STANDALONE_CHUNKED_PCC`, the engine's number); measured per-layer minimum recorded | **Two arms, identical numbers.** `KV cache PCC PASSED (min 0.986623 >= 0.93 across 1 slots; per cache: k=0.996784, v=0.986623)` over `[0,1024)` across **32/32** local layers; per-layer min **K 0.996784 (L22)**, **V 0.986623 (L28)** — 5.6x the threshold's error budget on V, and it agrees with `G-MESH-KV`'s on-device chunk-256 row **to 6 decimals and on both argmin layers** (0.9967844 / L22, 0.9866232 / L28): two readers, two processes, two read paths. LayerAck drain **128/128** = 32 layers x 4 chunks, in 0.42 s. **Arm 2** adds `PREFILL_ENABLE_MIGRATION=1`, which takes the engine's real stage-gather path (`prefill_runner.py:626`, `:644`) and calls `kv_migration_base_address` — the branch `DEC-111` would have crashed on, and the one arm 1 cannot reach; same PCC to every digit | PASS | 2026-09-04 | `raw/G-MOCK-MIG-producer_20260904T173939Z.log`, `raw/G-MOCK-MIG-runner_20260904T173939Z.log.gz`, `raw/G-MOCK-MIG-STAGED-producer_20260904T180914Z.log`, `raw/G-MOCK-MIG-STAGED-runner_20260904T180914Z.log.gz` |
| G-KV-TABLE | P10 | the address table **alone**, isolated from every numerical question | **bit-exact** over UMD read-back (`torch.equal`, `rtol=atol=0`); a control that reads one head through another's config must fail | **11/11** tests on the full `(4,8)` mesh, 38.0 s. **2048 chunks bit-identical** (1024 per block-cyclic period, at periods 512 and 128) against both the live device tensor *and* the labelled host probe. 16 configs (k_h0-7, v_h0-7), 1024 entries each period, 4352 B/chunk, `chunk_n_tokens` 32. head->config->chip: every config's device group is a **single** chip and it is exactly `MeshCoordinate(sp_row, head)`, compared on fabric-node ids. UMD path == the table's own control-plane `read_device_chunk` on 8 sampled entries. Protobuf round trip preserves every address, size, device group **and** the zero-padded config names `00..15` | PASS | 2026-09-04 | `raw/G-KV-TABLE_20260904T172909Z.log` |
| G-LOOPBACK | P10 | the real DRAM -> transport -> DRAM migration copy (the doc's Gate 2) | `dst-bytes` identical | **not run.** It needs the tt-llm-engine binaries (`migration_endpoint`, `migration_worker`, `_migration_client*.so`), none of which is in this tree, and it verifies the **engine's** model-agnostic byte copy rather than this model — `HUMAN GATE H4`'s question ("whose bug would a red gate be?") answers *the engine's* | **OUT OF SCOPE** by `DEC-103`; residual gap enumerated as `R-043` | 2026-09-04 | n/a — `G-KV-TABLE` proves the table the copy reads, bit-exactly, over the same UMD path the worker uses |
| G-RUNTIME (P10 ext) | P10 | the migration hooks the engine calls, and what they still refuse | every unguarded name present with a binding signature; every refusal matched on its message; **no refusal on a value the engine really sends** | **48/48** tests (P7 stood at 37; P10 adds 6 and removes 1). The removed one is the finding: `metadata_msg` was **refused** and the engine passes it on every chunk, so the runner died on its first served chunk after the mesh open and the weight load (`DEC-108`). Two tests now stand in its place — one drives a non-`None` `metadata_msg` past that point to the next refusal, one reads the engine's keyword set out of the AST and forbids a `<param> is not None` refusal for any of them except `d2h_service`. New: `set_layer_ack_channel` refuses before `compile()` and injects exactly `num_layers` acks per chunk; `assert_single_rank_stage` refuses a non-zero `first_layer_idx`, a partial `num_my_layers`, a multi-rank `stage_layout` and a `stage_layouts` list, and accepts the three single-rank shapes; `kv_migration_stages` asserted **absent**; `kv_migration_base_address` returns K's. **`DEC-111` then found the same class of defect twice more**, after every P10 gate had passed: `stage_layout` is a **list** of per-rank dicts and the guard demanded a dict (which would have blocked every `PREFILL_ENABLE_MIGRATION=1` run), and two of its guards compared a value with itself (so `R-032`'s advertised protection did not exist). Both are now refused on the gathered list's **length**, the only argument carrying other ranks' data, and a new test AST-checks the *producing* function so the type belief cannot drift again. Refusal census: **22** `raise` in the runtime + **7** in `tt/runners/kv_chunk_table.py` | PASS | 2026-09-04 | `raw/G-RUNTIME_20260904T180813Z.log` |
| P10-REGRESSION | P10 | the whole package suite still passes after P10's additions | 0 failed | **246 passed, 0 failed** in 23:40 (P8 stood at 196; P10 adds 50 — `G-ADAPTER` 28, `G-KV-TABLE` 11, `G-RUNTIME` +11). Second run: the first was killed at ~67 tests when the review that found `DEC-111` landed mid-flight, so its result belonged to pre-fix code (`DEC-090`'s reasoning, applied earlier) | PASS | 2026-09-04 | `raw/P10-REGRESSION_20260904T181140Z.log.gz` |
| G-CITE (P10) | P10 | every `path:line` and every cited raw artefact resolves | 0 mismatched, 0 unresolved, 0 missing artefacts | **604/604** content-checked citations (`CITES` 539 -> 604), **1169/1169** doc refs, **128/128** raw artefacts. First fully clean run of all three passes. It caught **13** of this phase's own citations wrong, and **one P7 citation that this phase's own edit to `prefill_producer.py` invalidated** (`_read_slot_kv_and_check_pcc_mla` moved 511 -> 694) — the latter only because that row is content-checked; as prose it would have resolved in range and stayed wrong | PASS | 2026-09-04 | `raw/G-CITE_20260904T184140Z.log` |

```
STATUS after P0: gates PASS=1 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P1 (reference)
STATUS after P1: gates PASS=2 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P2 (repo survey)
STATUS after P2: gates PASS=3 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P3 (package outline)
Open DECs needing review: DEC-002 (docs/ + scripts/__init__.py vs the P3 tree — P3/P9 must settle),
DEC-004 (chunk size deferred to P7), DEC-008 (TestFactory.setup_test deferred to P5.1),
DEC-012 (checkpoint loader parked in tests/test_factory.py until P6.2),
DEC-013 (import gpt_oss_d_p/utils rather than copying it, against the P3 tree's comment)
```

**STOPPED HERE, ON A GATE BOUNDARY.** P0, P1 and P2 are complete and gated; **P3 (package outline,
`03_OUTLINE.md`, gate `G-OUTLINE`) is next**, followed by P4. No device code exists yet: `tt/` holds
only its `__init__.py`. The two things P3 must settle before it writes the tree are `DEC-002`
(does the package keep `docs/` and `scripts/__init__.py`, and does it vendor a copy of the recipe?)
and `DEC-013` (is `utils/` created at all, given the helpers are imported from
`models/demos/gpt_oss_d_p/utils/`).

---

### G-CARD — every architectural fact has a source
- **Command:** document review of `00_MODEL_CARD.md` +
  `python models/demos/llama31_8b_d_p/scripts/verify_citations.py`
- **Mesh / device:** none for the review; `ttnn.get_num_devices()` / `get_arch_name()` opened and
  closed the cluster once to record the machine facts.
- **Threshold:** every card row has a non-empty `Source`; zero rows say "from memory"; the
  "does NOT have" section exists; the `(mesh, TP, SP)` arithmetic is shown including the
  `TP == num_key_value_heads` derivation; every `UNVERIFIED` row also appears in `07_RISKS.md`;
  `verify_citations.py` exits 0. Source: recipe P0 `G-CARD`.
- **Measured:**
  - card rows with a `Source`: **100%** (2 identity + 25 architecture + 5 derived-geometry + 4
    deployment rows); rows sourced "from memory": **0**; rows marked `UNVERIFIED`: **0** (so the
    `07_RISKS.md` cross-check is vacuously satisfied, and §5 says so explicitly).
  - `verify_citations.py`: `citations checked 53 / verified 53 / mismatched 0 / missing 0`;
    `doc refs scanned 115 / resolved 115 / unresolved 0`; exit 0.
  - identity: `md5 3cd5831d379b509d53afade0e24c36e9` for all three of
    `$HF_MODEL/config.json`, `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`,
    and the bundled `configs/Llama-3.1-8B-Instruct/config.json`.
  - machine: `get_num_devices() = 32`, `get_arch_name() = 'blackhole'`,
    `hasattr(ttnn, "BlackholeComputeKernelConfig") = False` — all three as the recipe states.
  - arithmetic, computed in the raw log rather than asserted in prose: `head_dim 4096/32 = 128`;
    `4096/8 = 512 (%32=0)`; `14336/8 = 1792 (%32=0)`; `128256/8 = 16032 (%32=0)`;
    `nqh 4 >= nkv 1 && 4%1 == 0`; `SP = 4` → `CHUNK_SIZE % 128 == 0`.
- **Verdict:** **PASS**
- **Negative control:** this gate produces no PCC, so §1.4 exempts it — but it has one anyway, and it
  fired. The first two runs of `verify_citations.py` **failed** (exit 1): six of this agent's own
  first-draft `CITES` line numbers were wrong (`sdpa_device_operation.cpp` 99→98,
  `PREFILL_MIGRATION_TESTING.md` 31→62, the vendored `config.json` 7→13,
  `models/demos/minimax_m3/conftest.py` 17→16, `models/demos/gpt_oss_d_p/utils/substate.py` 6→15), and a
  citation-shadowing false positive resolved a bare `tt/config.py` onto a gpt-oss file. Both raw logs
  are kept; the failing run is the control that shows the check discriminates.
- **Deviations:** none to the gate. Two recipe-internal conflicts were resolved by `DEC-002`
  (`docs/` + `scripts/__init__.py`, and not vendoring the recipe into the package) and one by
  `DEC-009` (which test file owns the bundled-config byte-identity assertion).
- **What this does NOT prove:** that the card's *interpretation* of the config is right — only that
  every claim is traceable. `hidden_act: silu` is a fact; "SwiGLU is `down(silu(gate)*up)`" is a
  reading of it, and `G-REF` is what checks the reading. It also does not prove the checkpoint's
  **weights** match the config (`G-WEIGHTS`, P6), nor that the live HF repo agrees (`07_RISKS.md`
  R-002).
- **Notes:** `raw/G-CARD_20260904T034303Z.log` is the full transcript (identity, config dump, device
  facts, arithmetic, first citation pass); `raw/G-CARD_20260904T034338Z.log` is the re-run of the
  identity + citation passes after `DEC-003`'s fix, and is the log the verdict cites.

---

---

### G-REF — the reference is deterministic and self-consistent
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q`
- **Mesh / device:** **none** — host only, no `ttnn` import, no `mesh_device` fixture (`DEC-010`).
  Runs on a box with no card and while the mesh is busy.
- **Input distribution:** standard normal (`torch.randn`), seed 0, for hidden states and for every
  random weight (`scale = 0.02` for projections, `1.0 + 0.02*randn` for norm gains). randn is the
  harder of the two distributions for a norm and the recipe requires the choice to be stated rather
  than shopped (§2.1(b)). Real layer-0 weights from the staged checkpoint are used by the two
  `requires_hf_reference` tests. seq_len 128 (random weights) / 64 (real weights).
- **Reference dtype policy:** **fp32 everywhere** (`DEC-006`). The HF layer is built bare from the
  bundled config and `.float()`ed with `_attn_implementation = "eager"` — never `from_pretrained`,
  which would load at the checkpoint's `torch_dtype: bfloat16` and give a reference that shares the
  device's own rounding. Checkpoint tensors are `.float()`ed at load. Nothing is rounded to bf16 in
  this gate.
- **Threshold:** (a) two runs bit-identical; (b) hand-written vs HF PCC ≥ 0.9999 on one decoder
  layer, expecting bit-exact; (c) `01_REFERENCE.md` documents the invocation and the dtype policy.
  Source: recipe P1 `G-REF` / Appendix A.
- **Noise floor (computed):** **not applicable, and that is the honest answer.** Both sides here are
  fp32 torch on identical inputs, so the floor for this comparison is exact equality — PCC 1.0, and
  an error ratio of 0/0. The gate is therefore stated as bit-exactness; the 0.9999 threshold is a
  ceiling on how far a *faithful transcription* may drift, not a target to sit near. Measured
  `max|Δ| = 0.0`, i.e. at the floor. The floor helpers themselves (`quantize_like_device`,
  `err_ratio`) ship in `tests/test_factory.py` in this phase but are first *used* by P5.
- **Measured:** **10 passed, 0 failed, 13.6 s.**

  | Check | Result |
  |---|---|
  | determinism, hand-written (SHA-256 ×2) | `0a16330068bce3807911a80462fad1b38c84cfd3f08ba9098df38156f67cb96b` / identical |
  | determinism, HF `LlamaDecoderLayer` (SHA-256 ×2) | `0a16330068bce3807911a80462fad1b38c84cfd3f08ba9098df38156f67cb96b` / identical |
  | hand-written vs HF, random weights, seq 128 | **PCC 1.0**, `max|Δ| = 0.000e+00` |
  | hand-written vs HF, **real layer-0** weights, seq 64 | **PCC 1.0**, `max|Δ| = 0.000e+00` |
  | llama3 `inv_freq` vs `models/tt_transformers/tt/common.py:489` | `max|Δ| = 0.000e+00` |
  | bundled `config.json` vs checkpoint's | byte-identical (`filecmp`, `shallow=False`) |
  | `rope_theta` on transformers 5.12.1 | `cfg.rope_theta` → `AttributeError`; `getattr(cfg,"rope_theta",10000.0)` → **10000.0**; `get_rope_theta(config.json)` → **500000.0** |
  | `position_embeddings` in `LlamaAttention.forward` / `LlamaDecoderLayer.forward` | present in both (trap 5 branch resolves to the modern path) |

  The two oracles produce the **same hash**, which is the bit-exactness result restated.
- **Negative control:** five, all of which must fail and all of which do.
  1. **Causality** (recipe P1 trap 3): perturbing the last token moves rows `[:-1]` by
     `max|Δ| = 0.000e+00` with an explicit mask and by **`3.515e+00`** with `attention_mask=None` —
     HF's own default. Measured identically for both oracles, so the transcription reproduces the
     trap rather than hiding it.
  2. **Wrong theta** (10000.0, the value the `getattr` trap substitutes): **PCC 0.87437**.
  3. **GQA `repeat` instead of `repeat_interleave`**: **PCC 0.55396**.
  4. **No RoPE at all** (`cos=1`, `sin=0`): **PCC 0.86480**.
  5. **llama3 scaling inactive**: scaled vs unscaled `inv_freq` differ by `max|Δ| = 8.894e-04`, and
     `cos` at position 8192 by `1.882e+00` — so the piecewise scaling is demonstrably applied, not
     assumed.

  Note what controls 2 and 4 say about thresholds: a RoPE that is **wrong at every position** still
  scores 0.86–0.87 at the layer level. Anything in the 0.9 band is not "nearly right".
- **Verdict:** **PASS**
- **Deviations:** none to the gate. Three recipe-ordering issues were resolved by `DEC-008`
  (`TestFactory.setup_test` deferred to P5.1 — it builds `MeshConfig`/`CCLManager`, which are P5
  deliverables), `DEC-012` (the checkpoint loader parked in `tests/test_factory.py` until `ModelArgs`
  exists in P6.2) and `DEC-009` (the bundled-config byte-identity assertion lives in this file).
- **What this does NOT prove:** that either oracle is **right about Llama's architecture**. Two
  transcriptions agreeing bit-exactly proves the transcription is faithful and nothing more; a shared
  misreading survives untouched (`07_RISKS.md` R-006). It proves nothing about device code — none
  exists yet — nothing about layers other than 0 with real weights, nothing about the weight
  *mapping* (`G-WEIGHTS`), and nothing at sequence lengths past 128, where the llama3 scaling's
  interpolated band starts to matter (`G-ROPE` and `G-MODEL` own that).
- **Raw log:** `raw/G-REF_20260904T035140Z.log`. The earlier `raw/G-REF_20260904T035011Z.log` is the
  same suite before the trap-5 signature test was added; both are kept.

---

### G-SURVEY — reuse decided, with citations
- **Command:** document review of `02_SURVEY.md` (structure checks scripted into the raw log) +
  `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` + the per-phase regression run
  `pytest models/demos/llama31_8b_d_p -q`
- **Mesh / device:** none.
- **Threshold:** every component row has a decision + citation; the "not bringing over" list exists;
  the `models/common/` verdict is present with its two citations; no row's decision is "write" where
  an importable equivalent exists (else a `DEC`); the verifier re-verifies every `path:line` in the
  survey. Source: recipe P2 `G-SURVEY`.
- **Measured:**
  - **30** component rows (the recipe's minimum list is 20; the extra 10 are `substate`,
    `num_links`/cache naming, the noise-floor helpers, the KV chunk table, the fabric descriptors,
    the PCC helpers, the compute-kernel-config convention, indexed RoPE, the ring SDPA, and the
    normalised `hf_config` constructor).
  - **30/30** rows carry a decision. Split: **adapt 16**, **import 9** (6 plain + `import + convention`
    + `import (config, not code)` + `import the op`), **import-the-op-adapt-the-caller 3**,
    **adapt + import the math 1**, **copy-with-a-`DEC` 1**, **write 0**.
  - **27/30** rows carry a full `path:line`. The three that do not (23, 27, 30) cite a whole file
    plus abbreviated `:NN` refs in the same row, or — row 30 — two `.textproto` config files where a
    line number would be meaningless. Both forms are resolved by the verifier's pass 2.
  - **15** "not bringing over" entries (MoE, router, EP dispatch/combine, shared expert,
    `swigluoai`, sinks, sliding window, QK-norm, partial RoPE, YaRN/mscale, MLA, sparse/MSA, MXFP4,
    biases, decode/paged/trace).
  - `models/common/` verdict present with both required citations
    (`models/common/modules/mlp/mlp_2d.py:461`, `models/common/models/llama3_8b/model.py:890`), plus
    two the recipe does not require and that this run verified independently:
    `models/common/modules/mlp/mlp_2d.py:256`/`:259` (the `cluster_axis = 1` reduce-scatter) and
    `models/common/modules/attention/attention_1d.py:319` (the only `Attention*` class under
    `models/common/modules/`).
  - `verify_citations.py`: `citations checked 123 / verified 123 / mismatched 0 / missing 0`;
    `doc refs scanned 195 / resolved 195 / unresolved 0`; exit 0.
  - per-phase regression: `10 passed` (the whole package suite, 13.6 s) — unchanged from P1, i.e.
    P2 broke nothing.
- **Verdict:** **PASS**
- **Negative control:** the citation pass, again, and it fired twice.
  1. Of the 70 `CITES` entries added for this survey, **one** was wrong on the first run —
     `models/demos/gpt_oss_d_p/README.md:24` (the P7-unification roadmap line is at `:26`) — cited
     inside `DEC-013` as the reason a cross-package import is acceptable. It would have read as
     authoritative.
  2. The gate's own **structure check** was wrong before the survey was: the first run reported
     "rows with a decision AND a path:line: 18/30", which was a broken regex (the `**bold**` markers
     around `**import**`), not a broken survey. Both raw logs are kept; the first
     (`raw/G-SURVEY_20260904T035900Z.log`) is the control showing the check discriminates, the
     second is the verdict's evidence. *A failing check is not evidence of a failing artefact until
     the check's own logic is verified* — the same lesson §2.5 states for probes.
- **Deviations:** none to the gate. One decision deviates from the recipe's *tree comment* rather
  than from the gate: `DEC-013` imports `models/demos/gpt_oss_d_p/utils/` instead of copying it, as
  the P3 tree's `# (copy from gpt_oss_d_p/utils)` suggests, because agent-contract rule 4 says reuse
  means import.
- **What this does NOT prove:** that the reuse decisions are *correct*. A row saying "import X" is
  not evidence that X works at Llama's shapes — nothing in P2 executes any of the cited code.
  `07_RISKS.md` R-007 (the chunked-KV / indexed-RoPE ops have no Llama-shaped upstream exercise) and
  R-009 (no dense, bias-free, full-RoPE attention template exists) name the two widest gaps, and
  P5's gates are what close them. It also does not prove the *absence* of a better template
  somewhere in the tree — the search was directed by the recipe's list of locations, not exhaustive.

---

### G-OUTLINE — file tree + shapes pinned; every gate owns a file
- **Command:** document review of `03_OUTLINE.md`, with the structure checks scripted into the raw
  log, + `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` + the per-phase regression
  run `pytest models/demos/llama31_8b_d_p -q`
- **Mesh / device:** none. P3 writes no code and opens no device.
- **Threshold:** `03_OUTLINE.md` lists every file with (i) a one-sentence responsibility, (ii) a
  public interface signature, (iii) input/output tensor shapes with dtype and layout, (iv) the
  template it mirrors (`path:line`); **every Appendix A gate maps to a named owner in the tree**; and
  the per-layer tensor-shape table is filled in with real numbers. Source: recipe P3 `G-OUTLINE`
  (`BRINGUP_RECIPE.md:1092-1098`).
- **Measured:**
  - **32/32** Appendix A gate rows have a named owner (§4), plus the per-phase regression gate. The
    Appendix A table has 32 gate rows, not 33: `G-WEIGHTS` appears twice (P6.2 and its P8 extension)
    and `G-KV-TP8` is easy to miscount because it is the one gate id containing a digit.
  - **41** files in the committed tree at the end of P10 (9 exist today); **49/49** non-`__init__.py`
    files are named in a per-file contract, including all 22 test/harness files (§2.15's per-test-file
    table) and `README.md` (§2.16).
  - **18/18** rows of the per-layer shape table carry four filled cells with no unresolved `/TP` or
    `/SP` — real numbers at `(4,8)`, TP=8, SP=4: `512`, `128`, `1792`, `16032`, `[1,4,S_loc,128]`,
    `[1,1,S_loc,128]`.
  - **6** deviations from the recipe's tree, each labelled `[DEV-n]` and each justified in §1.1.
  - `verify_citations.py`: `citations checked 225 / verified 225 / mismatched 0 / missing 0`;
    `doc refs scanned 338 / resolved 338 / unresolved 0`; exit 0. `CITES` grew by **102** entries
    this phase (123 → 225).
  - per-phase regression: `10 passed` in 13.76 s — unchanged from P1/P2, i.e. P3 broke nothing.
- **Verdict:** **PASS**
- **Negative control:** doc gates produce no PCC, so §1.4's four numeric fields are waived
  (`BRINGUP_RECIPE.md:354-355`) — but the two mechanical checks both fired before they passed, which
  is what shows they discriminate.
  1. **The citation verifier caught one wrong line** in this phase's own first draft of `CITES`:
     `models/demos/gpt_oss_d_p/tt/attention/prefill.py:104` was claimed to hold
     `activation_dtype = ttnn.bfloat16` (it holds `hidden_size = hidden_states.shape[-1]`; the
     activation-dtype ladder is at `:106-109`). That reference is cited inside `DEC-022` as the
     evidence for the dtype ladder, so it would have read as authoritative.
  2. **The structure check failed its first run**, correctly: 27/49 tree files had a contract entry,
     because the 22 test files were covered only collectively in §2.15 and in the gate-owner map. The
     fix was the per-test-file table (gate / mesh / reference / negative control per file), which is
     the more useful artefact — P5 now inherits its control for each gate rather than inventing one.
     A third failing run was the check's **own** bug, not the document's: its gate regex was
     `G-[A-Z-]+`, which silently dropped `G-KV-TP8` and reported 31 gates instead of 32. *A failing
     check is not evidence of a failing artefact until the check's own logic is verified* — the same
     lesson §2.5 states for probes, and the second time this run has hit it (`G-SURVEY`'s control 2).
- **Deviations:** none to the gate. Six deviations from the recipe's **tree** (`03_OUTLINE.md` §1.1),
  of which two are new judgement calls (`DEC-017`, `DEC-018`), one is a recipe omission that is not a
  judgement call at all (`[DEV-4]`: `tt/runners/adapters/__init__.py`, without which the registry
  cannot import the adapter), and three follow from earlier decisions. One recipe sentence had to be
  *resolved* rather than obeyed (`03_OUTLINE.md` §5.1: `BRINGUP_RECIPE.md:1262-1264` requires both
  that sub-axis TP raises and that TP need only *divide* the axis — `4` divides `8`, so the two
  halves contradict each other; the refusal is taken as binding).
- **What this does NOT prove:** that any of these interfaces or shapes are **right**. Nothing in P3
  runs. Every signature here is a prediction that P5–P10 may falsify, and the shape table is
  arithmetic on the P0 card rather than a measurement — a wrong `config.json` reading would propagate
  through it untouched (`G-WEIGHTS` and `G-MODEL` are what catch that). It also does not prove the
  gate→owner map is *sufficient*: a gate with a file is not a gate with a test, and four gates
  (`G-MESH`, `G-SEMAPHORE`, `G-WEIGHTS`, `G-TP-PARITY`) exist only because the recipe warns they are
  the ones that go unowned. Finally, the three deferred numbers (`CHUNK_SIZE`, `MAX_SEQ_LEN`,
  matmul block sizes) mean §3's KV-cache row is parameterised, not pinned.
- **Raw log:** `raw/G-OUTLINE_20260904T083915Z.log`

---

### G-CCL-PLAN — every collective placed and justified
- **Command:** document review of `04_CCL_PLAN.md`, with the structure checks scripted into the raw
  log, + `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` + the per-phase regression
  run `pytest models/demos/llama31_8b_d_p -q`
- **Mesh / device:** none. P4 writes no code and opens no device.
- **Threshold:** `04_CCL_PLAN.md` contains the `(mesh, TP, SP)` arithmetic; the collective-placement
  table with **every row justified**; the residual-scheme `DEC` **with the cost-equivalence
  argument**; the semaphore-lifetime statement ("allocated once in `CCLManager.__init__`, cycled per
  call, never per layer") **and its depth**; and a list of **every** collective call site with its
  `cluster_axis`, `dim` and `topology`. Source: recipe P4 `G-CCL-PLAN`
  (`BRINGUP_RECIPE.md:1210-1216`).
- **Measured:**
  - `(4, 8)`, TP=8 on `tp_axis = 1`, SP = `32 / 8 = 4` on `sp_axis = 0`, `num_links` 2 at `(4,8)` /
    1 on any `(1,N)` submesh. `TP == num_key_value_heads == 8` stated as an equality with both bounds.
  - **8/8** placement rows (Embedding, RMSNorm, Attention, Attention-SP, MLP, LM head, DecoderLayer,
    Model) carry a non-empty justification, and each justification says *why the tensor is incomplete
    at that point* rather than which collective is conventional there.
  - **9** collective call sites, each with all four of `dim`, `axis`, topology and `num_links` filled
    — 4 of them the scheme-B/P8 seams that **refuse** until then, 2 of them SP-axis and P8-only.
  - **3/3** scatter-width rows arithmetically correct: `4096/8 = 512 = 16*32`,
    `128256/8 = 16032 = 501*32`, `256/8 = 32`. No `pad_size` is needed anywhere — computed, not
    asserted.
  - Semaphores: `3*2 = 6` RS + `2*2 = 4` AG + `2*1 = 2` barrier + 2 ring-attention = **14**, and the
    table's arithmetic re-checks in the raw log. Barrier depth **2**, with the one-op reuse gap
    written out and 4 barrier-consuming collectives per layer counted (128 per forward).
  - Collective op usage counts **re-measured** on this tree rather than quoted from the recipe:
    `all_gather_async` **29**, `reduce_scatter_minimal_async` **18**, `all_reduce_async` **2** —
    identical to `BRINGUP_RECIPE.md:1160-1162`.
  - All three BH-galaxy torus mesh-graph descriptors named in §7 exist (listed in the raw log).
  - `verify_citations.py`: `citations checked 253 / verified 253 / mismatched 0 / missing 0`;
    `doc refs scanned 404 / resolved 404 / unresolved 0`; exit 0. `CITES` grew by **28** this phase
    (225 → 253).
  - per-phase regression: `10 passed` in 13.77 s — unchanged.
- **Verdict:** **PASS**
- **Negative control:** doc gate, so §1.4's four numeric fields are waived
  (`BRINGUP_RECIPE.md:354-355`). The structure check is the control and it **failed its first run**,
  on two real defects in this phase's own document:
  1. §1.1 asserted "TP = 8 is an equality" without ever writing the words
     `num_key_value_heads` — the token the recipe's own derivation turns on, and the one a reviewer
     greps for. Fixed by stating `TP == num_key_value_heads == 8` explicitly.
  2. The semaphore-lifetime statement was line-wrapped mid-phrase, so the required sentence
     ("allocated once in `CCLManager.__init__`") did not exist as a contiguous string anywhere in the
     document. Cosmetic to a human reader and invisible to `grep` — which is the failure mode the
     check exists for.
  The check's other three sections re-derive arithmetic rather than matching text (scatter widths,
  semaphore counts, the 14 total), so a wrong number in the document fails them regardless of how it
  is worded.
- **Deviations:** none to the gate, and none to the recipe's plan: the two-object pattern
  (`CCLManager` + `MeshConfig`), the TP-axis-only rule, all-reduce as RS+AG rather than
  `all_reduce_async`, and scheme A are all taken as the recipe prescribes. Five decisions were logged
  (`DEC-024`–`DEC-028`), of which `DEC-026` (ship barrier depth 2, do not reset across chunks) and
  `DEC-028` (the one allowed raw `ttnn.all_gather`) are the two the recipe explicitly demands "either
  way".
- **What this does NOT prove:** **nothing about any collective actually running.** No device was
  opened; every row of §5 is a plan, and four of them describe code that will *refuse* to run in this
  iteration. Specifically unproven: that barrier depth 2 is sufficient (`G-RACE`), that the chosen
  topology/fabric pairing works on this box (`G-FABRIC-MATRIX`), that the `(4,8)` shape can be opened
  and submeshed at all (P8 step 1), that scheme A's cost-equivalence claim — an **op-count**
  argument, not a measurement — holds in device time, and that the SP ring path's semaphores and grid
  offset compose with the pinned 8x8 SDPA grid (`G-SP-RING`). The plan also cannot prove
  *completeness* of the call-site list: it enumerates the collectives this design will issue, and a
  module written in P5 that reaches for a collective not in this table is a deviation the table will
  not detect on its own — `G-CLEAN`'s "no raw `ttnn.experimental.*` in a module" grep is what closes
  that.
- **Raw log:** `raw/G-CCL-PLAN_20260904T084559Z.log`

---

### G-MESH — `MeshConfig` arithmetic and refusals; `CCLManager` allocates once
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_mesh_config.py
  models/demos/llama31_8b_d_p/tests/unit/test_ccl_semaphores.py -x -q`
- **Mesh / device:** (a) none — device-free arithmetic; (b) `(1,1)`, Blackhole. Only (b) takes the
  `mesh_device` fixture (`BRINGUP_RECIPE.md:1267`).
- **Input distribution:** n/a — this gate has no numeric input. Its inputs are mesh shapes:
  `(1,1)`, `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)`, `(8,4)`, and the four refused
  `(mesh, tp)` pairs below.
- **Reference dtype policy:** n/a — no reference tensor. The "reference" is arithmetic stated in
  `00_MODEL_CARD.md` §4 and re-derived from the bundled `config.json` in the test rather than
  restated as a literal.
- **Threshold:** exact asserts (`BRINGUP_RECIPE.md:1749`). No PCC, so §1.4's floor field does not
  apply.
- **Noise floor (computed):** n/a.
- **Measured:**
  - **16/16 tests pass** (9 device-free + 7 on the card).
  - shard arithmetic at `(1,8)`/TP=8: `sp=1`, `shard_size(4096)=512`, `shard_size(14336)=1792`,
    both `% 32 == 0`.
  - deployment `(4,8)`/TP=8: `sp=4`; `tp == num_key_value_heads == 8`; `local_q=4`, `local_kv=1`,
    so SDPA's `nqh >= nkv && nqh % nkv == 0` holds as `4 >= 1 && 4 % 1 == 0`.
  - real compute grid **(12, 10)** — not 8x8 — and `ring_attention_ccl_core_grid_offset = (11, 0)`,
    i.e. `grid.x - 1`. `num_links = 1` at `(1,1)`
    (`models/demos/gpt_oss_d_p/utils/general_utils.py:33`: a single-row mesh gets 1 link).
  - the build-time form of the ring op's assert holds: a pinned 8x8 SDPA grid needs
    `8 <= grid.x - 1 = 11`.
  - semaphore inventory **6 RS / 4 AG / 2 barrier / 2 ring-attention = 14**, identical after
    32 layers x 4 collectives = 128 barrier cycles; all three ping-pongs cycle with period 2.
- **Verdict:** **PASS**
- **Negative control:** four, and all four fired.
  1. **Sub-axis TP refuses.** `MeshConfig((1,8), tp=4)`, `((1,8), tp=2)`, `((4,8), tp=4)` and
     `((1,8), tp=16)` all raise `ValueError: ... sub-axis TP is unsupported`. §1.4 admits a
     configuration that must *refuse* as a control (`BRINGUP_RECIPE.md:351-352`).
  2. **Its complement.** `(1,2)`, `(1,4)`, `(2,8)`, `(8,4)` with matching TP must **warn and
     build** — without this, "raise on anything unusual" would satisfy control 1 while making
     every `(1,1)` P5 gate unrunnable.
  3. **A per-layer `CCLManager`.** Three managers stand in for three layers and produce
     `3 x 14 = 42` semaphores, the shape of the bug (`n_layers x` the constant).
  4. **`reset_global_semaphores` must not rewind the barrier index**, asserted, because `DEC-026`
     ships the template's deliberate skip and a later change must show up as a failure here.
- **Deviations:** none to the gate. Two `DEC`s came out of writing it: `DEC-029` (four dead
  members of the `CCLManager` template dropped, which shortens `03_OUTLINE.md` §2.2's attribute
  list) and `DEC-034` (the `prefer-expect-error` hook fires on the fixture's name in **prose**, so
  a docstring had to be reworded).
- **Raw log:** `raw/G-MESH_20260904T085727Z.log`
- **What this does NOT prove:**
  - **that any collective works.** At `(1,1)` no collective is issued: `MeshConfig`'s three
    wrappers are never called by this gate, only constructed around. `G-TP-PARITY` and
    `G-FABRIC-MATRIX` (P8) are what exercise them.
  - **that the semaphores are correct under concurrency.** This is a counting and cycling test on
    a single card. `G-RACE` (three runs, one process, one `CCLManager`, bit-identical) is the one
    that can see a race, and `G-SEMAPHORE`'s target-mesh half is P8's.
  - **that `num_links` is right for the deployment.** `(1,1)` yields 1 link by the helper's
    single-row branch, so the `num_links = 2` the `(4,8)` deployment uses is untested here (P8
    step 3).
  - **that the deployment mesh can be opened at all.** The `(4,8)` assertions in this gate are
    arithmetic on a shape tuple, not a device open (P8 step 1).
- **Notes:** `G-SEMAPHORE` is a **P8** gate and is *not* being recorded as PASS here; its
  one-card half runs in this file because `G-MESH` already requires the assertion
  (`BRINGUP_RECIPE.md:1265-1266`) and writing it twice would let the two copies disagree
  (`03_OUTLINE.md` §1.1 `[DEV-6]`). The 16 tests above include those 5.

---

### G-RMS — plain RMSNorm vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_rms_norm_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole
- **Input distribution:** **standard normal** activations, `[1, 1, S, 4096]`, `S ∈ {32, 512, 4096}`,
  `reset_seeds` (repo-root `conftest.py:34`, seed 213919). Two weight sources, both run
  (`DEC-035`): standard-normal random weights, and the **real** `model.layers.0.input_layernorm.weight`.
  Stated because it must never be chosen to pass — recipe §2.1(b) measures the bf16 floor as
  identical under `rand[0,1)` and `randn`, and randn is the harder of the two for a norm.
- **Reference dtype policy:** reference weight and activations **fp32**, all arithmetic fp32
  (`DEC-006`). Only what the device *stores* is quantised, and only to compute the floor: bf16
  activations and a bf16 norm weight in its stored `(1, 1, 128, 32)` shape (`DEC-022`). A
  bf16-weight reference would share the device's own rounding and flatter the number — recipe
  §2.1(a) measures 0.9999867 against 0.99995 for that mistake.
- **Threshold:** PCC >= 0.9999 (`BRINGUP_RECIPE.md:1751`). The error ratio is **recorded, not
  asserted**: a correct module sits right on §2.2's 3x stage bound, so asserting it would gate on
  the wrong side of the noise (`BRINGUP_RECIPE.md:1290-1293`).
- **Noise floor (computed):** **0.9999973 / 0.9999973 / 0.9999972** with random weights;
  **0.9999986** at all three lengths with the real layer-0 weight. The floor **moves with the
  weight distribution** — a trained norm gain is a narrow positive distribution, not standard
  normal — which is the whole reason `DEC-035` runs both rather than picking one.
- **Measured:**

  | weights | S=32 | S=512 | S=4096 |
  |---|---|---|---|
  | random, PCC | 0.9999957 | 0.9999958 | 0.9999957 |
  | random, floor | 0.9999973 | 0.9999973 | 0.9999972 |
  | random, ratio | **1.56x** | **1.57x** | **1.54x** |
  | real layer-0, PCC | 0.9999971 | 0.9999971 | 0.9999971 |
  | real layer-0, floor | 0.9999986 | 0.9999986 | 0.9999986 |
  | real layer-0, ratio | **2.11x** | **2.13x** | **2.12x** |

  The real-weight figure **reproduces the recipe's expected value against the same floor**: §2.4
  predicts 0.9999955 with `fp32_dest_acc_en=True` against a 0.9999986 floor, and this module
  measures **0.9999971** — slightly better, at 2.1x the floor rather than ~3x.
- **The §2.4 A/B, measured in-suite** (`DEC-014`'s falsifier, `DEC-030`):

  | S | `fp32_dest_acc_en=True` | `=False` | error reduction |
  |---|---|---|---|
  | 32 | 0.9999957 (1.56x floor) | 0.9999707 (10.79x) | **6.90x** |
  | 512 | 0.9999958 (1.57x floor) | 0.9999633 (13.58x) | **8.66x** |

  So the flag is worth ~7-9x of module error on this box, against the recipe's stated ~7x — and
  note that **both** settings clear the 0.9999 threshold at S=32, which is precisely §2.2's point:
  the absolute PCC does not distinguish them and the ratio to the floor does.
- **Negative control:** the **zero-gain probe** — `weight = 0` must give `max|out| = 0.0`.
  Measured **0.0**. A Gemma `(1 + weight)` fold would instead return the normalised input, whose
  per-channel magnitude is ~1, so this is the discriminator between plain and folded RMSNorm and
  therefore between Llama and the two nearest templates' feature set (`00_MODEL_CARD.md` §3).
  A second control: building with an empty `state_dict` and no `tensor_cache_path` must raise
  rather than run on a `None` gain — Appendix B's "cache-only build silently wrong" row.
- **Deviations:** none to the gate. `DEC-035` records the two-weight-source choice, forced by the
  recipe specifying the input three different ways.
- **Raw log:** `raw/G-RMS_20260904T090144Z.log`
- **What this does NOT prove:**
  - **the distributed (3-op) branch.** It is dormant (`is_distributed=False`, `DEC-025`) and no
    test executes it. `DEC-031` and `07_RISKS.md` R-011 record that the template's version of that
    branch would in fact raise `TypeError` — which is exactly what a never-executed branch is worth.
  - **cache-only loading.** The gate builds from a `state_dict` every time; the
    `tensor_cache_path` branch is only proven to *refuse* when absent. `G-WEIGHTS` (P6.2) owns the
    positive case.
  - **that the norm is wired into anything.** `G-LAYER` and `G-MODEL` own placement — this gate
    would pass equally if the two norms in a layer were swapped, which is `G-LAYER`'s own control.
  - **`eps` correctness beyond agreement.** Both sides read `1e-05` from the same bundled
    `config.json`, so a wrong value in that file would cancel. `G-CARD` is the provenance check.

---

### G-ROPE — llama3-scaled RoPE, Meta convention, vs the HF `rotate_half` reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_rope_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole (two of the eleven tests are host-only)
- **Input distribution:** **standard normal**, `[1, 4, S, 128]` — 4 heads, the local Q-head count
  per chip at the deployment TP=8 — with `S ∈ {32, 512, 4096}`, `reset_seeds`.
- **Reference dtype policy:** fp32 input, fp32 cos/sin, fp32 arithmetic. The **HF** convention
  (`x * cos + rotate_half(x) * sin`) is the reference; the device runs the **Meta** convention, and
  both tables come from **one** frequency set, as
  `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83` `_build_cos_sin` does, so the
  test cannot silently compare two different RoPEs. For the floor, the three tensors the device
  stores — input, cos, sin — are quantised to bf16 and the rest stays fp32.
- **Threshold:** PCC >= 0.999 (`BRINGUP_RECIPE.md:1779`), expecting ~0.99999.
- **Noise floor (computed):** **0.9999983 / 0.9999982 / 0.9999980**.
- **Measured:** **0.9999969 / 0.9999964 / 0.9999959** at S = 32 / 512 / 4096 — **1.76x / 1.95x /
  2.08x** the floor. 11/11 tests pass.
  - `rope_params` reads `(theta, factor, original_max_position_embeddings) =
    (500000.0, 8.0, 8192)` through `get_rope_theta` / `get_rope_scaling` on the raw dict, each
    asserted non-`None` — which is `07_RISKS.md` R-005 enforced in device code rather than by
    convention.
  - `build_prefill_rope`'s device tables are **bit-identical** (`torch.equal`, `max|Δ| = 0.0`) to
    the test's independently built Meta tables at all three lengths. A table layout is a mapping
    claim, so it is gated on bit-equality, not PCC (recipe §2.5).
  - `get_rot_transformation_mat()` and `get_rot_transformation_mat(dhead=128)` are
    `torch.equal` and both 32x32 — recipe P1 trap 4 confirmed on this version, not assumed.
- **Negative control:** two, both fired.
  1. **HF-layout tensor into the Meta op** — the classic convention mismatch — scores
     **0.01367** (the recipe measured 0.01296 for the same mistake). Without it, 0.99999 could
     equally mean both sides are wrong the same way.
  2. **The llama3 scaling must be provably active.** Asserted on the piecewise **band structure**
     of the frequencies `apply_scaling` consumes: of 64 frequencies, **29 low** (wavelength >
     8192) divided by exactly 8.0, **29 high** (wavelength < 2048) `torch.equal` to the base
     frequencies, and **6 mid** strictly interpolated between the two. This is a **deviation from
     the recipe's stated control** and `DEC-036` records why: the recipe asks that the scaled and
     unscaled tables differ "for positions beyond `original_max_position_embeddings`"
     (`BRINGUP_RECIPE.md:1340-1349`), and measured, `max|cos_scaled - cos_unscaled|` is
     **1.99933 inside** the window and **1.99398 beyond** it — both saturated at the theoretical
     maximum of 2, because llama3 scaling divides long-wavelength frequencies at *every* position
     and `cos` oscillates. That assertion therefore cannot fail for the reason it exists. Both
     numbers are still recorded, as the recipe asks; the band test is what gates.
  3. A third refusal, counted with the controls: the contiguous builder must reject
     `start_pos > seq_len` (the `gather_cos_sin` out-of-bounds landmine), and the indexed builder
     must reject `chunk_size % (32*sp) != 0` and `max_seq_len % chunk_size != 0`.
- **Deviations:** `DEC-036` (the scaling control, above) and `DEC-033` (three signature deviations
  from `03_OUTLINE.md` §2.5, all forced by the helpers being wrapped). Verdict is `PASS` rather
  than `PASS-WITH-DEVIATION` because the gate's own threshold and its required controls are met —
  `DEC-036` makes the control *stronger* than specified, not weaker.
- **Raw logs:** `raw/G-ROPE_20260904T091040Z.log` (the verdict) and
  `raw/G-ROPE_20260904T090652Z.log` (a **failed** earlier run, kept deliberately: the first
  attempt at the recipe's scaling assertion recovered each frequency from a cos table by `arccos`,
  and for the lowest frequency `cos(1 * f)` rounds to 1.0 in fp32, giving `0/0 = nan`. That is
  `LANDMINES.md`'s "a failing probe is not evidence of a failing module until the probe's own
  numerics are checked", hit live — the module was correct throughout.) Three intermediate passing
  runs made while adding tests were discarded as superseded; they were authoring runs, not gate runs.
- **What this does NOT prove:**
  - **that Q/K projection weights are `reverse_permute`d on the real load path.** The HF -> Meta
    permutation is applied *by the test*, not by `tt/attention/weights.py`, which does not exist
    until P5.5. `G-ATTN`'s "loaded without the Meta permute" control (recipe: 0.9475) is what
    closes it — and note how high that broken variant scores.
  - **the indexed RoPE numerically.** `build_indexed_rope` is exercised only structurally, at
    SP=1, where the block-cyclic reorder is the identity; the table is asserted `torch.equal` to
    the plain Meta table and the two shape constraints are asserted as refusals. Nothing about the
    SP>1 layout is testable on `(1,1)`. `G-CHUNK` (P7) and `G-CHUNK-ATTN` (P8) own it.
  - **long-context correctness.** The longest sequence gated is 4096, well inside
    `original_max_position_embeddings` = 8192, so the scaled band of the tables is never exercised
    *by the device*: the band structure is proved on the host. Appendix B's "PCC good at short seq,
    bad past ~8192" symptom would still be invisible here; `G-MODEL` at long context is where it
    would show.
  - **that the RoPE is applied to the right tensors.** Only Q and K may be rotated; this gate
    rotates a bare tensor. `G-ATTN` asserts the invariant.

---

```
STATUS after P3: gates PASS=4 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P4 (parallelism + CCL plan)
STATUS after P4: gates PASS=5 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P5 (naive module implementations, bottom-up)
STATUS after P5.1-P5.3: gates PASS=8 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P5.4 (dense SwiGLU MLP, G-MLP)
Per-phase regression gate: whole package suite **47 passed, 0 failed**
(`raw/P5-REGRESSION_20260904T092212Z.log`)
Citations after P5.3: **279/279 verified, 0 mismatched; 522/522 doc refs resolved**, exit 0
(`raw/G-CITE_20260904T092334Z.log`)
Open DECs needing review: DEC-004 (chunk size deferred to P7 — now a parameter of build_indexed_rope, DEC-033),
DEC-012 (checkpoint loader moves to ModelArgs in P6.2), DEC-013/DEC-018 (import gpt_oss_d_p/utils; no utils/ package),
DEC-019 (three Q/K/V projections + nlp_create_qkv_heads(q, cat(k,v))),
DEC-021 (bf8_b KV dtype; the bf16 delta is owed at G-KV), DEC-025 (residual scheme A; scatter_output seam must refuse),
DEC-026 (barrier depth 2 — G-RACE's first move if it fails), DEC-027 (descriptor not yet pinned; G-FABRIC-MATRIX picks it),
DEC-029 (four dead CCLManager members dropped — P8 re-checks if the ring path wants a sub-device),
DEC-030 (compute-kernel config home; P6.2 may want to own it), DEC-032 (derive_head_dim moves into ModelArgs at P6.2),
DEC-031/R-011 (upstream fix owed against gpt_oss_d_p's dormant distributed RMSNorm),
DEC-037/R-012 (the root .gitignore excluded every raw log; re-included in-package, kit fix owed)
Closed this phase: DEC-008 (TestFactory.setup_test written in P5.1), DEC-017 (docs/ and scripts/__init__.py deleted),
R-005 (rope_theta substitution now enforced in device code), R-010 (in-package assert added)
```

**STOPPED HERE, ON A GATE BOUNDARY.** Supersedes the end-of-P2 and end-of-P4 stop notes above.
**P0-P4 and P5.1, P5.2, P5.3 are complete and gated; P5.4 (the dense SwiGLU MLP, `G-MLP`) is
next**, then P5.5 (attention, `G-ATTN`) and P5.6 (the KV cache, `G-KV`). Phase P5 is split across
sessions deliberately; this session's scope was P5.1-P5.3 only.

Device code now exists: `tt/{config,ccl,rms_norm,rope}.py`, with `tests/unit/{test_mesh_config,
test_ccl_semaphores,test_rms_norm_vs_ref,test_rope_vs_ref}.py`. `tests/test_factory.py` gained
`TestFactory.setup_test`. `docs/` and `scripts/__init__.py` are gone (`DEC-017`, executed).

What P5.4 can rely on, and what it must not assume:

1. **`MeshConfig` and `CCLManager` are constructed and counted, never exercised.** No collective
   has run on this box in this package. `MLP.__call__`'s TP tail must therefore be written behind
   `if self.mesh_config.tp > 1`, and `G-MLP` at `(1,1)` will not execute it — the first real
   collective is P8's.
2. **`default_compute_kernel_config(mesh_device)` in `tt/config.py` is the only compute-kernel
   config** (`DEC-030`). Pass it to both `ttnn.linear` calls and to `down_proj`'s. Do **not**
   copy `models/demos/gpt_oss_d_p/tt/attention/config.py:71`'s explicit `fp32_dest_acc_en=False`;
   measured here, `False` costs 6.9-8.7x on the norm alone, and recipe §2.4 puts it at 96-1168x on
   a matmul.
3. **`derive_head_dim(hf)` in `tt/config.py`** is the package's one head-dim derivation
   (`DEC-032`); `tt/mlp.py` does not need it, but `tt/attention/` (P5.5) does, and it must call it
   rather than reach for `hf_config.head_dim`, which does not exist.
4. **`G-MLP` gates both dtypes** — `>= 0.999 @bf8_b` and `>= 0.9995 @bf16`, **and `<= 3x` the
   computed floor at each** (`BRINGUP_RECIPE.md:1753`). Unlike `G-RMS`, that ratio bound is
   *asserted*, so compute a separate floor per dtype: quantise the weights at the dtype under test
   and the activations at bf16 (`DEC-022`).
5. **The negative control is SiLU on `up` instead of `gate`** (recipe: 0.6462). It is what proves
   the fused unary is on the argument you think it is.
6. **`scatter_output` must be wired and must refuse** what it cannot honour (`DEC-025`), not
   half-implemented.
7. **Run `pre-commit run --files ...` before recording any `path:line`** — and note `DEC-034`: the
   `prefer-expect-error` hook is a `pygrep`, so it fires on the fixture's name in comments and
   docstrings too, not only on a call.

---

### G-MLP — dense SwiGLU vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_mlp_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole. TP=1, so the module's TP all-reduce tail
  (`bringup_log/04_CCL_PLAN.md` §5 row 2) is **not executed** — see "what this does NOT prove".
- **Input distribution:** `x` **standard normal**, `[1, 1, S, 4096]` with `S ∈ {32, 512, 4096}`;
  the three projection weights `randn * 0.02`, the scale both templates use
  (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:169`,
  `models/demos/minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py:98`). At that scale `gate` and
  `up` land at std ~1.3, so SiLU is exercised across the curved part of its range rather than in a
  locally-linear tail — stated because the distribution must never be chosen to pass.
- **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic,
  `torch.nn.functional.silu`. Identical random weights drive both sides.
- **Noise floor (computed):** per dtype, and per recipe §2.2's literal definition — quantise the
  **inputs and weights** to the device dtype, all remaining math in fp32. bf8_b weights are
  quantised in the `[1, 1, in, out]` orientation the device stores them in, because `bfloat8_b`
  shares an exponent per tile and quantising HF's `[out, in]` would block along the wrong axis.
  The bf16 *intermediates* the device also stores are deliberately **not** quantised: that is the
  conservative reading, since quantising them would lower the floor and flatter every ratio below.
  - bf8_b: **0.9999213 / 0.9999221 / 0.9999222** at S = 32 / 512 / 4096
  - bf16: **0.9999929** at all three
- **Threshold:** PCC >= **0.999** @bf8_b and >= **0.9995** @bf16, **and <= 3x the floor at each
  dtype** (`BRINGUP_RECIPE.md:2069`). Unlike `G-RMS`, Appendix A states a ratio bound for this gate,
  so the ratio is **asserted**, not merely recorded.
- **Measured:** 14/14 tests pass.
  - bf8_b: **0.9999133 / 0.9999144 / 0.9999144** -> **1.10x / 1.10x / 1.10x** the floor
  - bf16: **0.9999851 / 0.9999852 / 0.9999852** -> **2.11x / 2.10x / 2.09x** the floor
  - Both dtypes run and both are recorded, as `BRINGUP_RECIPE.md:2069` requires. bf8_b clears its
    threshold comfortably, so `DEC-021`'s "keep bf16 if bf8_b misses" contingency is not needed.
  - PCC is flat in sequence length to 7 decimal places, which is what a token-pointwise block
    should do; the S=32 bf8_b value differs only because a 32-row activation is one tile tall.
- **Negative control:** SiLU applied to `up` instead of `gate` — the mistake that proves the fused
  unary is on the argument the code claims — scores **0.64715** (bf8_b) / **0.64722** (bf16). The
  recipe measured **0.6462** for the same mistake. Driven by swapping the `gate_proj` / `up_proj`
  entries of the state dict, so the control runs the **real module** and not a hand-copied device
  path that could drift from it.
- **Two A/Bs, recorded as measurements (direction asserted, not a fitted threshold):**
  1. **`fp32_dest_acc_en`, recipe §2.4 reproduced on this box, at module level:**
     bf8_b `True` **0.9999144** (1.10x) vs `False` **0.9925127** (**96.13x** the floor);
     bf16 `True` **0.9999852** (2.10x) vs `False` **0.9917324** (**1167.80x**). The recipe's table
     predicts 96x and 1168x for the bare `ttnn.linear` — matched to three significant figures
     through a three-matmul module. This is the single most valuable number in the gate: an
     inherited `fp32_dest_acc_en=False` from
     `models/demos/gpt_oss_d_p/tt/attention/config.py:71` would still have cleared a 0.99 gate at
     bf8_b (0.99251) and even the 0.999 bf8_b threshold is what rejects it.
  2. **The SwiGLU spelling** (`DEC-039`): fused `input_tensor_a_activations=[SILU]` vs a separate
     `ttnn.silu` are **numerically identical** at both dtypes (0.9999144 / 0.9999852 either way).
     The fused keyword is bound (`ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:1469`)
     but **absent from `ttnn.mul.__doc__`**, so availability had to be established by calling it.
- **Refusals (counted with the controls):** an empty `state_dict` with no `tensor_cache_path` raises
  `ValueError` rather than building three `None` projections (Appendix B's "cache-only build
  silently wrong"), and `scatter_output=True` raises `NotImplementedError` rather than half-wiring
  residual scheme B (`DEC-038`). Both via the repo-root `expect_error` fixture (`conftest.py:948`),
  because the `prefer-expect-error` hook forbids the alternative in `tests/`.
- **Verdict:** **PASS**
- **Deviations:** none to the gate. Two new decisions: `DEC-038` (the `scatter_output` refusal) and
  `DEC-039` (the fused-SiLU spelling, with its measurement).
- **What this does NOT prove:**
  - **The TP collective.** At `(1,1)` `tp == 1` and `MLP.__call__`'s all-reduce tail is skipped
    entirely, so `bringup_log/04_CCL_PLAN.md` §5 row 2 has still never executed in this package.
    `G-TP-PARITY` (P8) owns it. This is the recipe's own "a gate that passes on a mesh the
    deployment never uses" caveat (`BRINGUP_RECIPE.md:791-793`) applied to a collective rather than
    to a head count.
  - **Column/row-parallel sharding.** At TP=1 `column_parallel` and `row_parallel` produce the same
    (unsharded) tensor, so this gate cannot tell the two mappers apart. `G-TP-PARITY` and
    `G-WEIGHTS` (P8 ext) own that.
  - **The cache-only build.** `tensor_cache_path` is exercised only as the *absence* that makes the
    weightless build refuse; no tilized weight is written or reloaded here. `G-WEIGHTS` (P6.2) owns
    the round trip.
  - **Real weights.** All numbers above are on random weights. A trained `gate_proj` is not
    standard-normal, and `G-RMS` measured that the floor itself moves between random and real
    weights; for the MLP the real-weight comparison arrives with `G-LAYER` / `G-MODEL`.

---

### G-ATTN — the GQA attention block vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_attention_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole (compute grid **(12,10)**). TP=1, so the module's TP
  all-reduce tail is not executed — `bringup_log/04_CCL_PLAN.md` §5 row 1 is P8's.
- **Input distribution:** `x` **standard normal** `[1, 1, S, 4096]` with `S ∈ {128, 512, 2048}`; the
  four projection weights `randn * 0.02`, the scale both templates use
  (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:169`). The standalone SDPA probe
  uses iid standard-normal Q/K/V. `reset_seeds`.
- **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic. The causal mask is
  built **explicitly** (`torch.triu(full((S,S), -inf), diagonal=1)`) and the KV heads are
  `repeat_interleave`d by the GQA group — the device does neither, because
  `ttnn.transformer.scaled_dot_product_attention` is causal and group-aware internally
  (`nqh >= nkv && nqh % nkv == 0`,
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`). cos/sin come
  from **one** frequency set (`tt/rope.py::llama3_freqs`): the reference takes the HF pair, the
  device takes the Meta pair via `build_prefill_rope`, so the test cannot silently compare two
  different RoPEs.
- **Noise floor (computed):** per dtype, and — for the stage budgets — **locally per stage**. A
  chained floor (quantise once, propagate) carries upstream rounding and inflates `1 - floor`; it
  put `concat_heads`, a pure layout op, at **0.01x**, which is not a kernel beating arithmetic but a
  broken floor. Each stage's floor is now computed from that stage's own quantised inputs.
  - block floor: bf8_b **0.9998805 / 0.9998657 / 0.9998521**; bf16 **0.9999875 / 0.9999854 / 0.9999838**
- **Threshold:** PCC >= **0.999**; stages this package implements **<= 3x**; block **<= 8x**
  (`BRINGUP_RECIPE.md:2070`). See **Deviations** for how the block budget is applied.
- **Measured:** 17/17 tests pass.

  | dtype | S | block PCC | raw ratio | SDPA-attributed residual |
  |---|---|---|---|---|
  | bf8_b | 128 | 0.9997364 | 2.21x | 1.03x |
  | bf8_b | 512 | 0.9997080 | 2.17x | 1.00x |
  | bf8_b | 2048 | 0.9996723 | 2.22x | 0.96x |
  | bf16 | 128 | 0.9998463 | 12.32x | 1.10x |
  | bf16 | 512 | 0.9998275 | 11.82x | 1.01x |
  | bf16 | 2048 | 0.9998029 | 12.17x | 0.70x |

  Per-stage, **stage-isolated** (each stage fed the reference's own quantised input, so the number
  is the stage and not the accumulation), at S=512:

  | stage | bf8_b PCC / ratio | bf16 PCC / ratio |
  |---|---|---|
  | `qkv_proj_q` | 0.9999728 / **1.06x** | 0.9999958 / **1.52x** |
  | `qkv_proj_k` | 0.9999727 / **1.06x** | 0.9999958 / **1.51x** |
  | `qkv_proj_v` | 0.9999726 / **1.06x** | 0.9999958 / **1.52x** |
  | `rope_q` | 0.9999954 / **2.50x** | 0.9999954 / **2.50x** |
  | `rope_k` | 0.9999954 / **2.48x** | 0.9999954 / **2.48x** |
  | `concat_heads` | 0.9999986 / **1.00x** | 0.9999986 / **1.00x** |
  | `o_proj` | 0.9999727 / **1.06x** | 0.9999958 / **1.52x** |
  | `sdpa_fused` (**not** in the 3x budget) | 0.9998361 / **26.72x** | 0.9998361 / **26.72x** |

  Every hand-written stage is **1.00x-2.50x** of its floor; the recipe's own run measured
  1.00-1.47x for the same set. The RoPE stages are dtype-independent, as they must be — they touch
  no weight.
- **The fused kernel, isolated and tracked (recipe §2.3):**
  - **standalone probe**, iid bf16 Q/K/V, GQA 32/8, head_dim 128: PCC **0.9998309 / 0.9998127 /
    0.9998087** against modelled floors 0.9999969 / 0.9999966 / 0.9999964 — **54.86x / 54.95x /
    52.83x**. The recipe measured 0.9999204 at 71x; same order, same conclusion. Kept permanently
    so the slack stays a named term.
  - **in-pipeline**, on the block's own post-RoPE tensors: **26.7x-28.6x** (lower than the iid probe
    because real post-RoPE Q/K are correlated across the head dim).
  - **the attribution is quantitative, not rhetorical.** `1 - PCC` is variance-like, so independent
    error sources add to first order. floor error + the kernel's own excess predicts the block PCC
    to 5-6 decimals: bf8_b S=512 predicted **0.9997079** vs measured **0.9997080**; bf16 S=512
    predicted **0.9998277** vs measured **0.9998275**. Subtracting it leaves a residual of
    **0.70x-1.10x** — this package's code, sitting *at* its floor.
- **Negative control:** Q/K weights reaching the device **without** the Meta `reverse_permute`
  scores **0.51174** (bf8_b) / **0.51178** (bf16). Constructed by pre-applying
  `models/tt_transformers/tt/load_checkpoints.py:895` `permute`, the exact inverse of the loader's
  `reverse_permute` (`:891`), so the control runs the **real loader** rather than bypassing it.
  **This is a discrepancy with the recipe, and in the safe direction:** `BRINGUP_RECIPE.md:1428-1430`
  expects ~**0.9475** for the same mistake, i.e. a control that barely fires; measured here it
  collapses to 0.51. At head_dim 128 with full rotary the unswizzled weight scrambles 128 channels
  per head, and both Q *and* K are unswizzled — which may be the difference from whatever variant
  produced 0.9475. Either way the control discriminates, and the recipe's warning about *how high*
  a broken variant can score is unaffected: 0.9475 would still have cleared a 0.99 gate.
- **Invariants and refusals (counted with the controls), all fired:**
  1. **Only Q and K are rotated.** Scored against a reference that also rotates V: **0.71785**,
     versus **0.9998275** for the correct reference. A V-rotating reference must and does fit worse.
  2. **A derived SDPA program grid is refused at construction.** On this (12,10) box the CCL offset
     is `grid.x - 1 = 11` and the pinned SDPA grid is 8, so `11 >= 8` holds; `ProgramConfig(sdpa_grid_x=12)`
     raises, naming
     `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`.
     This is the landmine that otherwise passes **every** single-card gate and fails only at SP > 1.
  3. **`cached_len > 0` on the dense path is refused** (`NotImplementedError` naming the
     chunk-position-aware SDPA), rather than running an `is_causal` mask that is off by `cached_len`
     and silently wrong.
  4. **A weightless build is refused** — no `state_dict` and no `tensor_cache_path` raises rather
     than building four `None` projections.
- **Verdict:** **PASS-WITH-DEVIATION** — `DEC-042`.
- **Deviations:**
  - **`DEC-042`** — the 8x block budget is asserted raw at **bf8_b** (where it holds at 2.17-2.22x)
    and, at both dtypes, on the **SDPA-attributed residual** (0.70-1.10x against the same 8x). The
    raw bf16 ratio, 11.8-12.3x, is recorded and not asserted. The arithmetic in `DEC-042` shows an
    8x raw budget at bf16 is unreachable for **any** correct implementation given this kernel: it
    would require the kernel at <= ~17x its own floor, and the recipe's own §2.3 measurement of it
    is 71x. Note the direction of the paradox — bf16 has the *higher* absolute PCC and the *worse*
    ratio, because a smaller floor error divides the same fixed slack.
  - **`DEC-040`** — `ProgramConfig` delegates the compute-kernel config to `tt/config.py` instead of
    holding the outline's four local fields.
  - **`DEC-041`** — attention's `apply_reduce_scatter` refuses (scheme B seam), and is deliberately
    untested because nothing in this iteration calls it.
  - **`DEC-043`** — the head-split op's Python keyword is `num_heads`, not `num_q_heads` as
    `DEC-019` and `03_OUTLINE.md` §2.7 both spell it.
- **What this does NOT prove:**
  - **The TP collective, or any sharding.** At `(1,1)` `tp == 1`: `apply_allreduce` returns its
    input untouched, and `column_parallel` / `row_parallel` produce the same unsharded tensor, so
    this gate cannot tell the two mappers apart. `G-TP-PARITY` and `G-WEIGHTS` (P8 ext) own that.
  - **The GQA head->column map.** At TP=1 all 32 Q and all 8 KV heads live on one chip, so the
    per-chip `nq=4 / nkv=1` configuration the deployment actually runs is never built here. The
    group *arithmetic* is exercised (SDPA sees 32/8, group 4); the *distribution* is not.
  - **The KV-cache write.** `kv_cache=None` throughout — `G-KV` owns the primitive and `G-KV-TP8`
    the model -> cache path.
  - **The indexed RoPE.** `apply_rope`'s `kv_actual_global` branch is wired and never taken here;
    `G-CHUNK` (P7) owns it.
  - **Real weights.** All numbers are on random weights; `G-LAYER` / `G-MODEL` bring the real ones.
  - **Long context.** The longest gated sequence is 2048, well inside
    `original_max_position_embeddings` = 8192, so the *scaled* band of the llama3 RoPE tables is
    never exercised on device (the band structure is proved on the host by `G-ROPE`).

---

### G-KV — the KV-cache primitive: write, read back, and write nothing else
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_cache_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole. `sp = 1`, so the **block-cyclic layout degenerates to the
  identity** (local row == global position) and no inverse reorder is needed for read-back — which
  is also precisely why this gate cannot test the reorder.
- **Input distribution:** two payloads, deliberately different.
  - *PCC half:* realistic **post-RoPE K** and **raw V** — standard-normal `x [S, 4096]`,
    `randn * 0.02` k/v projections, the package's own llama3 RoPE tables — at `S ∈ {128, 512}`,
    all 8 KV heads, one per layer slot. A cache round trip should be measured on the values it will
    actually hold.
  - *Bit-exact half:* an integer payload where **every row names its own global position**, plus a
    head-id lane block and a chunk-id lane block. 4 chunks x 32 tokens = 128 positions at
    `kv_actual ∈ {0, 32, 64, 96}`.
- **Reference dtype policy:** fp32 reference; only the tensor the device *stores* is quantised, for
  the floor.
- **Noise floor (computed):** `quantize_like_device(ref, cache_dtype)` against the fp32 reference,
  per head. The write is a copy, not arithmetic, so the device should land *on* it — it does.
  - bf8_b: **0.9999743**-**0.9999754** (K), **0.9999752**-**0.9999753** (V)
  - bf16: **0.9999986**
- **Threshold:** PCC >= **0.99** at the cache dtype and **<= 3x its floor**
  (`BRINGUP_RECIPE.md:2071`); the layout claims on **bit-equality** (`torch.equal`, `rtol=atol=0`),
  never PCC (§2.5).
- **Measured:** 15/15 tests pass.
  - **Round trip, worst of 8 heads:** bf8_b `S=128` K **0.9999743** / V **0.9999753**; `S=512`
    K **0.9999754** / V **0.9999752**. bf16 **0.9999986** throughout. **Every head at both seq
    lengths measures a ratio of 1.00x** — exactly at its floor, as a pure copy must.
  - **Positional read-back, bit-exact:** 128 rows across 4 chunks, `torch.equal` at both dtypes,
    and re-checked after **every** chunk write, so "an earlier chunk is unchanged after a later
    chunk's write" is asserted 10 times (chunks 0..k for each k) rather than once.
  - **No collateral writes, bit-exact:** writing `(user 0, layer 1)` of a 2-user x 2-layer cache
    leaves the **352-position pad tail** exactly zero and all **3** other `(user, layer)` slots
    exactly zero, at both dtypes. Target `(0, 1)` rather than `(0, 0)` on purpose, so a
    `slot = user * num_layers + layer` bug would show.
  - **Geometry, asserted exactly:** per-chip `(64, 1, 384, 128)` bf8_b TILE for 2 users x 32 layers,
    DRAM `NdShardSpec` `[1, 1, 32, 128]`, and `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 32` — the
    producer's block geometry, kept so P10 can reuse its packed-GQA read-back
    (`BRINGUP_RECIPE.md:1456-1458`).
  - **The dtype delta `DEC-021` owed:** bf8_b K **0.9999756** / V **0.9999757** versus bf16
    **0.9999986** / **0.9999986** — **17.8x** the error on K and **17.7x** on V, for half the bytes
    (128 vs 256 per token per head). Measured, not assumed; `DEC-021` stands.
- **Negative controls / refusals — five, all fired:**
  1. **The bit-exactness assertion itself is the control for the position map**, and it *caught a
     real failure*: the first version of the probe, built to `BRINGUP_RECIPE.md:1475-1477`'s stated
     "<= 256" ceiling, failed at bf8_b on chunk 2's **odd** rows with `max|delta| = 1.0`. The cache
     was correct; the probe was not. Measured, the first inexact integer is **129** at bf8_b and
     **257** at bf16 — §2.5's ceiling is the **bf16** ceiling, and the cache dtype is bf8_b.
     `DEC-044`. This is §2.5's own warning ("a failing probe is not evidence of a failing module
     until the probe's own numerics are checked") firing against §2.5's own rule.
  2. `write_kv_chunk` with `batch = 2` raises — the op ignores the leading dim and would silently
     write only `slot_idx`.
  3. `slot_idx` out of range raises (a silent OOB write into another user's cache).
  4. `layer_idx` out of range raises.
  5. A non-tile-aligned `kv_actual` raises (breaks the block-cyclic per-device write); and
     `max_seq_len = 48` at `sp = 1` raises because `seq_local` would not be tile-aligned.
- **Verdict:** **PASS**
- **Deviations:** `DEC-044` (probe ceiling 128, not 256 — the probe changed, no threshold did) and
  `DEC-045` (`expect_error`'s `message` is matched as a **regex**, not the substring its docstring
  describes, so `"multiple of TILE_SIZE*sp"` silently never matched; the matcher now uses a
  metachar-free substring).
- **What this does NOT prove** — stated here rather than left for the `PASS` to imply:
  - **The model -> cache path.** At `(1,1)` the model emits all 8 KV heads on one chip while the
    per-chip cache holds exactly **one**, and the write op refuses the mismatch outright
    (`TT_FATAL: cache and input num-heads dim must match`,
    `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
    This gate therefore drives the op **one head at a time**, into its own layer slot. `G-KV-TP8`
    (P8) owns the real path; `07_RISKS.md` R-001 is the standing gap. `bringup_log/00_MODEL_CARD.md`
    §4.3 states the general rule: a gate that passes on a mesh the deployment never uses can be
    testing a configuration the model cannot produce.
  - **The block-cyclic reorder.** At `sp = 1` it is the identity. `G-KV-TP8` and `G-MESH-KV` own it.
  - **The head -> mesh-column map.** There is one column here. That map is a layout claim and must
    be gated on bit-equality at TP=8 — recipe §2.5 measured a *rotated* map still scoring PCC
    0.99890, which is why `G-KV-TP8` has a rotated-column control.
  - **Cache reads.** Nothing here reads the cache back *for attention*; `attention/prefill.py`
    refuses `cached_len > 0` and `G-CHUNK-ATTN` (P8) owns the read path.
  - **Positions past 128.** The bit-exact probe stops at 127 (`DEC-044`). A deployment chunk is far
    longer, so `G-KV-TP8`'s probe must split the position id across lane blocks rather than raise
    the ceiling.

---

```
STATUS after P5.4-P5.6: gates PASS=10 FAIL=0 DEVIATION=1 BLOCKED=0 | next: P6 (layer and model assembly)
Per-phase regression gate: whole package suite **93 passed, 0 failed**
(`raw/P5-REGRESSION_20260904T102256Z.log`)
Citations after P5.6: **330/330 verified, 0 mismatched; 629/629 doc refs resolved**, exit 0
(`raw/G-CITE_20260904T102606Z.log`). CITES grew 279 -> 330: every load-bearing P5.4-P5.6 ref was
promoted into the content-checked list, because the doc-ref pass only range-checks (R-016).
Open DECs needing review: DEC-004 (chunk size deferred to P7), DEC-012 (checkpoint loader moves to
ModelArgs in P6.2), DEC-013/DEC-018 (import gpt_oss_d_p/utils; no utils/ package),
DEC-021 (bf8_b KV dtype — the owed delta is now MEASURED at G-KV: 17.8x on K, 17.7x on V, for half
the bytes; the decision stands), DEC-025 (residual scheme A), DEC-026 (barrier depth 2 — G-RACE's
first move if it fails), DEC-027 (descriptor not yet pinned; G-FABRIC-MATRIX picks it),
DEC-029 (four dead CCLManager members dropped), DEC-030/DEC-040 (compute-kernel config home; P6.2
may want to own it), DEC-032 (derive_head_dim moves into ModelArgs at P6.2),
DEC-038/DEC-041 (the scatter_output seams refuse; P8 owns scheme B),
DEC-042/R-015 (**the one that needs a decision, not just review**: G-ATTN's 8x block budget is
unreachable at bf16 for any correct implementation given the fused SDPA kernel; P6's G-LAYER (8x)
and G-MODEL (8x + 4x step) contain the same kernel and will meet the same wall),
DEC-044/R-013 (bf8_b's exact-integer ceiling is 128, not the recipe's 256 — P8/P10 probes must
split the id across lanes), DEC-045/R-014 (expect_error's message is a regex)
Closed this phase: R-009 (the dense/bias-free/full-RoPE adaptation is now measured, G-ATTN),
R-007 (write half — the KV write path is bit-exact at head_dim 128), R-012 (kit half — the recipe
now carries the raw-log .gitignore guidance)
```

**STOPPED HERE, ON A GATE BOUNDARY.** Supersedes the end-of-P5.3 stop note above.
**P0-P4 and all of P5 (P5.1-P5.6) are complete and gated; P6 (layer and model assembly) is next.**
This session's scope was P5.4, P5.5 and P5.6 only — `tt/layer.py` and `tt/model.py` are deliberately
not written.

All six P5 gates are recorded: `G-MESH`, `G-RMS`, `G-ROPE`, `G-MLP`, `G-KV` are `PASS` and `G-ATTN`
is `PASS-WITH-DEVIATION` (`DEC-042`), so `BRINGUP_RECIPE.md:1482-1484`'s "all of G-MESH, G-RMS,
G-ROPE, G-MLP, G-ATTN, G-KV must be PASS before P6" is satisfied under §1.4's definition of the
verdicts.

Device code now exists: `tt/{config,ccl,rms_norm,rope,mlp}.py` and
`tt/attention/{__init__,config,weights,operations,prefill,kv_cache,dense_sp}.py`, with
`tests/unit/{test_mesh_config,test_ccl_semaphores,test_rms_norm_vs_ref,test_rope_vs_ref,
test_mlp_vs_ref,test_attention_vs_ref,test_kv_cache_vs_ref}.py`.

What P6 can rely on, and what it must not assume:

1. **No collective has ever executed in this package.** Every P5 gate ran at `(1,1)`, where
   `tp == 1` and both module tails (`MLP.__call__`, `attention/operations.apply_allreduce`) return
   their input untouched. `bringup_log/04_CCL_PLAN.md` §5 rows 1-2 are still unexercised, and so are
   `column_parallel` / `row_parallel` as *distinct* mappers. `tt/layer.py` must not read a P5 `PASS`
   as evidence that the TP path works.
2. **`Attention` takes an `AttentionConfig`, not the raw `hf` dict** — the one deliberate exception
   to the module signature convention (`bringup_log/03_OUTLINE.md` §5). `tt/layer.py` builds that
   config **once** and shares it across all 32 layers: there is no per-layer `dataclasses.replace`,
   because Llama has no sliding-window alternation.
3. **`ProgramConfig.get_compute_kernel_config` now takes `mesh_device`** (`DEC-040`), and
   `ProgramConfig.validate_grid(mesh_device)` runs at `Attention.__init__`. Build the program config
   once per model, not per layer.
4. **`attention_forward` refuses `cached_len > 0`** and `MLP`/`operations.apply_reduce_scatter`
   refuse `scatter_output=True`. All three refusals are load-bearing, not stubs: P6 must not route
   around them.
5. **The KV cache is proved as a primitive only.** `write_kv_chunk` is bit-exact at `head_dim = 128`
   and writes nothing it should not, but the **model -> cache** path has never run — at TP=1 the op
   refuses the model's 8 local KV heads outright. If `tt/model.py` wires a cache write, its first
   real test is P8's `G-KV-TP8`.
6. **`G-LAYER` and `G-MODEL` will hit `R-015`.** Both blocks contain
   `ttnn.transformer.scaled_dot_product_attention`, whose slack accounts for the whole of `G-ATTN`'s
   block gap (26.7-28.6x in-pipeline, 52.8-55.0x standalone). `G-MODEL`'s **4x per-layer step** is
   the tighter constraint and needs the same attribution `DEC-042` sets out — decide it before
   measuring, not after.
7. **Stage floors must be computed locally.** A floor propagated through a quantised chain carries
   the upstream stages' rounding and produces ratios below 1.0 (measured: `concat_heads` at 0.01x),
   which is a broken floor, not a kernel beating arithmetic. `G-ATTN`'s stage helper is the pattern.
8. **Never read a `path:line` out of a multi-file `cat -n`** (R-016), and pick refusal-message
   substrings with no regex metacharacters (`DEC-045`).

---

### Re-run after the citation corrections (end of P5.6)

Every `path:line` in this session's modules and log entries was re-resolved and **21 were wrong**
(R-016): four refs into `models/demos/gpt_oss_d_p/tt/attention/operations.py` carried a **+209 line
offset** — they had been read out of a `cat -n weights.py operations.py` whose numbering ran across
both files — and seventeen refs into `BRINGUP_RECIPE.md` had been **interpolated from the section
headings rather than read**, e.g. Appendix A's `G-ATTN` row cited as `:1731` when it is at `:1770`.
The verifier's doc-ref pass reported all of the in-range ones as `resolved`, because that pass
checks the line *number*, not the line's *content*.

All are corrected, and `CITES` now content-checks **every** recipe reference this package makes
(`RCP` entries, 359 total citations, up from 279 at the end of P5.3) so a future recipe edit that
shifts a section produces a `MISMATCH` rather than a silent lie.

Because the modules changed after the first gate runs — docstrings and comments only, no executable
line — all three gates and the regression were **re-run** so the recorded evidence matches the tree:

| Gate | Result | Raw log |
|---|---|---|
| `G-MLP` | 14 passed, 0 failed | `raw/G-MLP_20260904T101937Z.log` |
| `G-ATTN` | 17 passed, 0 failed | `raw/G-ATTN_20260904T102107Z.log` |
| `G-KV` | 15 passed, 0 failed | `raw/G-KV_20260904T102206Z.log` |
| per-phase regression | **93 passed, 0 failed** | `raw/P5-REGRESSION_20260904T102256Z.log` |
| citations | 359/359 verified, 631/631 doc refs resolved, exit 0 | `raw/G-CITE_20260904T102606Z.log` |

The measured numbers are identical to the first runs in every case; the earlier raw logs
(`G-MLP_20260904T093653Z`, `G-ATTN_20260904T095359Z`, `G-KV_20260904T100312Z`) are kept because the
detail blocks above quote them, and because §0.2 rule 4 is right that a log records what actually
ran. **These re-run logs are the ones the verdicts rest on.**

### G-LAYER — one decoder layer vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_decoder_layer_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole. TP=1, so **no collective executes** — the layer never calls
  one by construction (`04_CCL_PLAN.md` §4) and both sublayer tails are no-ops at `tp == 1`.
- **Inputs (distribution AND scale — `R-018`):** three arms.
  1. **Gate arm:** `x` **standard normal** `[1, 1, S, 4096]`, `S ∈ {128, 512, 2048}`; seven
     projections `randn * 0.02`; two norm gains `1 + randn * 0.02`. Both weight dtypes.
  2. **Real-weight arm:** real layer-0 weights (all nine tensors) with a `randn` input — recorded
     because it is what showed the control was weak, not because it gates anything.
  3. **Real-weight, real-input arm:** real layer-0 weights **and** real `embed_tokens` rows
     (RMS **0.0106**, i.e. ~100x smaller than `randn`), which is what layer 0 actually receives.
- **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic; explicit causal mask
  `triu(full((S,S), -inf), 1)`; KV heads `repeat_interleave`d by the GQA group. The staged reference
  in the test is **bit-exact** against both the `G-REF` oracle and HF `LlamaDecoderLayer`
  (PCC 1.000000000, `max|Δ| = 0.0`), so the gate is scored against math that was already gated.
- **Noise floor (computed):** the same fp32 layer with its **inputs and weights** rounded to the
  device dtypes and everything else in fp32. Internal intermediates are **not** quantised.
- **Threshold:** PCC ≥ 0.999 and ≤ 8x the floor (`BRINGUP_RECIPE.md:1511`, Appendix A `:1854`),
  applied per `DEC-051`: the raw ratio asserted at bf8_b, and the **SDPA-attributed residual**
  asserted at ≤ 8x at both dtypes.
- **Measured:**

  | dtype | S | PCC | floor | raw ratio | kernel excess (x floor err) | attributed residual |
  |---|---|---|---|---|---|---|
  | bf8_b | 128 | 0.9997665 | 0.9998709 | 1.81x | 0.68x | **1.13x** |
  | bf8_b | 512 | 0.9998273 | 0.9998953 | 1.65x | 0.51x | **1.14x** |
  | bf8_b | 2048 | 0.9998736 | 0.9999138 | 1.47x | 0.31x | **1.15x** |
  | bf16 | 128 | 0.9998774 | 0.9999826 | 7.05x | 5.04x | **2.01x** |
  | bf16 | 512 | 0.9999172 | 0.9999859 | 5.86x | 3.83x | **2.03x** |
  | bf16 | 2048 | 0.9999480 | 0.9999885 | 4.51x | 2.37x | **2.14x** |

  Real weights + real input, bf8_b, S=512: **0.9998649** against a floor of **0.9999647** — **3.82x**.
  This is the number comparable with recipe §2.1's table (also measured on real weights).
- **Verdict:** **PASS** — and note that unlike `G-ATTN` the **raw 8x holds at both dtypes**
  (worst 7.05x). `R-015`'s wall does not reach layer level: the layer adds two norms and three more
  quantised projections to the floor's error budget while the fused kernel's absolute slack is
  unchanged, so the kernel's share falls. `DEC-051` carries the arithmetic.
- **Negative controls (three, and only the third discriminates — `DEC-058`, `R-018`):**
  - random weights, norm gains swapped: **0.99864** (bf8_b) / **0.99873** (bf16) — rejected by the
    gate's assertion, but by 1.4e-3;
  - real layer-0 weights, `randn` input, gains swapped: **0.99993** — *further* from failing,
    because the residual attenuates the sublayers by ~65x at that input scale;
  - real layer-0 weights, **real** embedding input, gains swapped: **0.66830** (recipe quotes
    0.9471), attenuation 3.40x. This is the control the verdict rests on.
- **Additional assertions:** causality on the **device** layer — perturbing the last token leaves
  rows `[:-1]` at `max|Δ| = 0.000e+00` while the last row moves by 8.969; the `LLAMA_DELTA_PROBE`
  helper runs on a real tensor and warns rather than raising on garbage (the package's one
  `except Exception`).
- **Deviations:** none from the threshold. `DEC-051` (attribution method), `DEC-054` (signature),
  `DEC-058` (control input scale) are the phase's judgement calls.
- **What this gate does not prove.** It is an **integration** check
  (`BRINGUP_RECIPE.md:1514`): it cannot localise a sublayer fault, and it may not substitute for
  `G-RMS`/`G-ROPE`/`G-MLP`/`G-ATTN`, all of which are met on their own. It also runs at TP=1, so
  neither module's TP collective has ever executed (P8), and it writes no KV cache — at TP=1 the
  packed cache refuses the model's 8 local KV heads outright (`00_MODEL_CARD.md` §4.1, `R-001`).

### G-WEIGHTS — real-checkpoint weight loading, bit-exact
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py -x -q`
- **Mesh / device:** one card, `(1,1)`. Cache-only at TP > 1 is the P8 extension
  (`BRINGUP_RECIPE.md:1550`).
- **Inputs (distribution):** not a synthetic distribution — the inputs **are** the real
  Llama-3.1-8B-Instruct checkpoint tensors at their stored dtype, `torch.bfloat16` (measured and
  logged by `torch_dtype_of`, matching `00_MODEL_CARD.md` §2). Stated rather than omitted because
  §1.4 requires it.
- **Reference dtype policy:** the **same torch tensor object** drives both sides — the device path
  and `quantize_like_device` — with **no fp32 detour**, so a bf16→bf8_b vs fp32→bf8_b double-rounding
  difference cannot be mistaken for a loader fault. Every comparison is `torch.equal`
  (`rtol = atol = 0`), never PCC: recipe §2.5 measured a completely wrong mapping still scoring
  PCC 0.99890, and a transpose or swizzle applied twice is that class of bug
  (`BRINGUP_RECIPE.md:1547-1548`).
- **Noise floor:** not applicable, and that is the point — a bit-exactness gate has no floor because
  the tolerance is zero. The nearest equivalent, recorded instead: `max|Δ| = 0.000e+00` on all
  twelve tensors.
- **Threshold:** exact, in three parts (`BRINGUP_RECIPE.md:1543-1550`).
- **Measured:**
  - **(a) no missing, no unused.** Checkpoint keys **291**, expected **291**, missing **0**,
    unused **0** — over all 32 layers, read from `model.safetensors.index.json` with no tensor data
    loaded. Both difference sets printed (both empty).
  - **(c) every device weight bit-exact.** All **12/12** tensors of a one-layer model at
    `max|Δ| = 0.000e+00` *through* the loader's transpose, the Q/K Meta swizzle and the dtype ladder:
    `embed_tokens` `(128256, 4096)` bf16; both layer norms and `model.norm` `(1,1,128,32)` bf16;
    `q_proj` `(1,1,4096,4096)` and `k_proj`/`v_proj` `(1,1,4096,1024)` and `o_proj`
    `(1,1,4096,4096)` bf8_b; `gate_proj`/`up_proj` `(1,1,4096,14336)` and `down_proj`
    `(1,1,14336,4096)` bf8_b; `lm_head` `(1,1,4096,128256)` bf8_b. Per-tensor SHA-256 logged.
    `DEC-055` records why this is all twelve of one layer rather than a sample of 32.
  - **(b) cache-only rebuild bit-identical.** 12/12 SHA-256 identical between a checkpoint build and
    a rebuild from an **empty** `state_dict` against the same `tensor_cache_bfp8_1x1` directory;
    12 cache files written.
- **Verdict:** **PASS**
- **Negative controls (three, all discriminate):**
  1. **Meta-renamed keys** (`map_hf_to_meta_keys`): 291 keys in, **0** still consumable by this
     package (`layers.0.attention.wq.weight`, …), and `Model` **raises** rather than building on
     `None`s. This replaces the recipe's "bypass `map_hf_to_meta_keys`" wording, which is
     inapplicable because this package does not apply that map — `DEC-046`.
  2. **Double Meta swizzle:** with Q/K pre-`reverse_permute`d, `q_proj` and `k_proj` stop being
     bit-equal to the clean load while `v_proj` and `o_proj` stay equal — i.e. the check sees a
     transform applied twice, which is exactly what PCC would not (`DEC-047`).
  3. **Cross-dtype cache:** a bf16 build wrote **12** cache files into `tensor_cache_bf16_1x1` and
     **0** into `tensor_cache_bfp8_1x1`; ttnn additionally suffixes each file
     `_dtype_<DT>_layout_<L>.tensorbin`, so both defences hold (`DEC-048`).
- **Refusals asserted:** `weight_cache_path` with `TT_CACHE_PATH` unset (closes `R-003`);
  `load_state_dict(convert_to_meta_format=True)` (`DEC-047`); a `transformers` config **object**
  passed as `hf_config` (recipe P1 trap 1 — the test also re-asserts that this version's config has
  no `rope_theta` attribute).
- **Deviations:** `DEC-046` (no Meta key mapping, control inverted), `DEC-047`, `DEC-048`,
  `DEC-055` (one layer for the per-tensor and cache halves).
- **What this gate does not prove.** The **32-layer** cache-only rebuild, and any rebuild at TP > 1
  where the cached tensor is sharded — `G-WEIGHTS (P8 ext)` owns both. It also says nothing about
  whether the loaded weights are *numerically useful*; `G-MODEL`'s top-1 against HF is what closes
  that.

### G-MODEL — full stack hidden states + top-1 agreement (P6.3)
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_model_vs_ref.py -x -q`
- **Mesh / device:** (1,1), Blackhole. **Input:** real tokenized-prompt token ids, real checkpoint.
  **Reference dtype policy:** fp32 HF `LlamaForCausalLM`, eager; floor from inputs and weights only
  (no intermediates) — see the correction below.
- **Threshold (per the phase text, `DEC-053`):** at 2 and 4 layers PCC ≥ 0.999 and ≤ 8x floor; at full
  depth per-layer step ≤ 4x from L3; 100% top-1 at every depth.
- **Measured:**
  - 2 layers / s128: **0.9997314** (floor 0.9998795, **2.23x**), top-1 agrees
  - 4 layers / s128: **2.36x**, top-1 agrees
  - **32 layers / s512: 0.9984849**, top-1 **374 == 374**, worst per-layer step **1.27x** (budget 4x),
    curve smooth and monotone with **no step anywhere**
  - **Floor correction (`R-021`):** the gate's floor used HF's own class, which computes RoPE cos/sin
    internally in fp32, so it omitted the bf16 rounding the device pays on those tables in all 32
    layers. `1 - floor` = 5.4300e-04 (fp32 tables) vs **9.9160e-04** (bf16 tables) → the same
    measurement is **2.79x** against the incomplete floor and **1.53x** against the correct one. The
    omitted term is 45% of the correct floor error. A staged chain reproduced the gate's floor exactly
    (0.9994570) with fp32 tables, isolating the cause to that single term.
- **Negative control:** layer weights rotated by one → **0.16180**.
- **Verdict:** **PASS** (phase text). Recorded honestly: the full-depth absolute PCC 0.9984849 is
  **below** the 0.999 that Appendix A's compressed row appeared to require at all depths — that
  conflict is `R-020` and is fixed in the kit rather than worked around here.
- **What this does NOT prove:** §2.3.1's kernel attribution does not hold at this depth (`R-022`): the
  substituted chain scored 0.9981153, *worse* than the device, so the subtraction over-removes and the
  attributed residual came out at 0.63x. What it does establish is the fused kernel's **share**:
  58.9% of total model-scale error.

---

### G-GOLDEN — the fp32 golden KV trace: structure, content, and equality with HF's own loop (P7)
- **Commands (two, both host-only):**
  ```
  HF_MODEL=... PREFILL_TRACE_DIR=/home/mstojkovic/prefill_traces/llama31_8b_d_p/s512 \
    python models/demos/llama31_8b_d_p/scripts/generate_golden_kv_cache.py --tokens 512
  python models/demos/llama31_8b_d_p/scripts/verify_golden_kv.py $PREFILL_TRACE_DIR
  ```
- **Mesh / device:** none. Both scripts **import no ttnn** — asserted by the gate design, and the
  reason `verify_golden_kv.py` scores nothing against the device (`DEC-060`, `R-025`).
- **Inputs:** the package's bring-up prompt, tokenized with the checkpoint's own tokenizer and
  **tiled to exactly 512 tokens** — the same prompt and the same tiling `G-MODEL` uses
  (`DEC-056`), so a golden trace and a top-1 run see the same tokens. The 512 ids are recorded
  verbatim in `metadata.json`, which is what makes `G-CHUNK` reproducible against a regenerated
  trace.
- **Reference dtype policy:** **fp32 throughout** (`DEC-059`) — weights cast to `float32` as they
  come off the safetensors shards, all math in fp32, K and V saved fp32. Never through
  `from_pretrained`, which would load at the checkpoint's `torch_dtype` (bf16) and hand back a
  reference sharing the device's own rounding. There is no `--dtype` flag, so a bf16 golden cannot
  be written by accident. Attention is `eager` **with an explicit
  `transformers.masking_utils.create_causal_mask`**, asserted non-`None` — `eager_attention_forward`
  applies only the mask it is handed, so a `None` mask is a silently non-causal reference.
- **Computed noise floor:** not applicable — this gate has no device measurement. The floor that the
  golden *supports* is `G-CHUNK`'s, and `G-CHUNK` records both a storage and a complete version of
  it (`DEC-064`).
- **Threshold:** the verifier exits 0 over all 32 layers and prints a per-layer table; the streamed
  driver equals `LlamaModel`'s own loop at `rtol=atol=0`; a zeroed layer and a deleted layer must
  both make the verifier exit non-zero (`BRINGUP_RECIPE.md:1682-1684`).
- **Measured:**
  - **32 layers, 512 tokens, fp32, 128.0 MB** (expected 128.0 MB); every layer `[1, 8, 512, 128]`
    for both K and V; every element finite over the **whole** tensor, not a leading sample.
  - **The streamed driver is bit-identical to `LlamaModel`'s own loop.** Over all 32 layers,
    `max|delta|` = **0.0** on K, **0.0** on V and **0.0** on the post-final-norm hidden state, all
    via `torch.equal`. Streamed pass 37.6 s; cross-check pass 61.8 s.
  - Per-layer table, over all 32 layers: K RMS **1.44134 → 2.19652**, K `absmax`
    **9.96930 → 32.69120**; V RMS **0.03345 → 0.58581**, V `absmax` **0.47060 → 5.70590**. No
    all-zero token row anywhere — the smallest row norm in the whole trace is **0.01076** (V, layer
    0) and the smallest K row norm is **2.21719**.
  - No two of the 64 tensors are bit-identical (SHA-256 over each), so the streamed driver did not
    write the same layer twice.
- **Verdict:** **PASS**
- **Negative controls (both required by the gate, both discriminate):**
  1. **Layer 7's K and V zeroed** — structurally perfect, contents wrong. Verifier **exit 1** with
     **seven** named problems (RMS 0, std 0, an all-zero row, for each of K and V, plus "layer 7 V
     is bit-identical to layer 7 K"). Worth noting: the template verifier
     (`models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py:111`) **passes** this control — it
     checks shape, dtype and the finiteness of the first 1000 elements and nothing else — so the
     content checks in this package's version are what make the gate's own control meaningful.
  2. **Layer 13's file deleted** — verifier **exit 1**, "layer 13: layer_13.safetensors is missing".
- **Deviations:** `DEC-059` (fp32, against both templates' bf16), `DEC-060` (the file follows the
  gate text, not P7 step 2 — `R-025`), `DEC-066` (the trace lives outside the repo — `R-026`),
  `DEC-069` (this stays a script gate, so the P7 regression does not cover it).
- **What this gate does NOT prove.** Nothing about the device. It proves the reference is a
  reference: fp32, causal, complete, and not an artifact of this package's own streaming driver.
  Whether the *device* reproduces it is `G-CHUNK`'s question, and at TP=8 `G-KV-TP8`/`G-MESH-KV`'s.

---

### G-CHUNK — chunked KV production == one-shot (deltas 1-2), and both vs the fp32 golden (P7)
- **Command:**
  ```
  HF_MODEL=... PREFILL_TRACE_DIR=/home/mstojkovic/prefill_traces/llama31_8b_d_p/s512 \
    pytest models/demos/llama31_8b_d_p/tests/unit/test_attention_chunked_vs_ref.py -x -q
  ```
- **Mesh / device:** **(1,1)**, Blackhole. 32 layers, seq 512, `CHUNK_SIZE = 128` → **4 chunks** at
  write offsets `{0, 128, 256, 384}` (`DEC-061`). Cache dtype `bfloat8_b`, weights `bfloat8_b`,
  activations `bfloat16`.
- **Raw log:** `raw/G-CHUNK_20260904T132854Z.log` (4 tests, 69 s) + `raw/G-CHUNK_per_layer_pcc.json`.
  Two earlier runs (`..._124825Z`, `..._131500Z`) produced **identical** numbers to seven decimal
  places; the test file changed between them twice (a DRAM-lifetime fix, then the delta-1 bit-equality
  test), so all three are kept and this one is what the verdict rests on.
- **Delta 1, in isolation, on bit-equality rather than PCC.** The indexed RoPE applied per chunk with
  `kv_actual_global ∈ {0, 128, 256, 384}` is **bit-identical** to the contiguous builder over the
  whole sequence: `max|delta| = 0.0` on every chunk, `torch.equal → True`. This is an *addressing*
  claim — the op derives each chunk's start row on device from one runtime argument — and §2.5 says
  an address claim gated on PCC is not gated, so it is gated on `torch.equal`. The sub-test needs
  neither the checkpoint nor the trace, so it runs on a weightless box and is the cheapest first
  check for anyone debugging a chunked number. It also closes `R-007`'s second half: the indexed
  RoPE had no Llama-shaped numerical test anywhere.
- **What makes the decomposition exact.** Both KV producers are fed *the same hidden states*,
  captured from **one** one-shot forward of the real 32-layer model through the `on_layer_output`
  seam (`DEC-050`). Given identical inputs, deltas 1 and 2 are the **entire** difference between
  them: the one-shot arm uses the contiguous RoPE over the whole sequence and one write at
  `kv_actual = 0`; the chunked arm uses the **indexed** RoPE with `kv_actual_global = c * 128` and a
  write at the same offset. Nothing re-runs the attention core, so delta 3 is absent by construction.
- **Why it runs on one card.** A model-level KV write cannot at TP=1 — the model emits all 8 local
  KV heads and the per-chip slot holds one. The cache is therefore driven through `write_kv_chunk`
  **one head at a time, head `h` → layer slot `h`**: the same op, the same DRAM `NdShard` geometry
  and the same `head_dim = 128` a chip performs at TP=8 (`BRINGUP_RECIPE.md:1659`).
- **Inputs:** the golden trace's own 512 `token_ids` — real prompt tokens, real checkpoint weights.
  The hidden states the producers see are the ones the assembled model actually produces, i.e. the
  real-scale arm §2.2.2 and `R-018` say predicts model behaviour, not a synthetic one.
- **Reference dtype policy:** the fp32 golden (`G-GOLDEN`, `DEC-059`). The device's K is
  Meta-swizzled over `head_dim` (the loader `reverse_permute`s Q/K), so **the golden and the floor
  are permuted HF → Meta before comparison, and the permutation is applied *before* quantising** —
  `bfloat8_b` shares one exponent per 16-element block of the last dim, so the block boundaries move
  with the permutation. V is not swizzled and is compared as-is.
- **Thresholds (`BRINGUP_RECIPE.md:1663-1674`):** mutual PCC ≥ **0.999 per layer** (expected exact);
  vs golden ≥ **0.99** K / ≥ **0.98** V; layer-0 error ratio ≤ **3x**; per-layer error step ≤ **4x**
  from layer 3.
- **Computed noise floor — two definitions, both recorded (`DEC-064`, applying `R-021` to P7):**

  | layer 0 | K | V |
  |---|---|---|
  | measured (both producers, identical) | **0.9999617** | **0.9999369** |
  | **storage** floor — the golden quantised to bf8_b, which is what `:1583` names | 0.9999716 | 0.9999628 |
  | ratio to the storage floor | 1.35x | 1.70x |
  | **complete** floor — bf16 input, bf16 norm gain, bf8_b projection weight, bf16 RoPE tables, bf8_b store | 0.9999651 | 0.9999382 |
  | **ratio to the complete floor (the asserted one)** | **1.10x** | **1.02x** |

  Both clear 3x, so no verdict turns on the choice — but they differ by 23% on K and 67% on V, and
  the storage floor is the one that inflates, because it omits the projection weight's rounding that
  the device pays on every K and V it produces. No internal intermediate is quantised (§2.2's
  conservative reading), and the complete floor's ratios sit at 1.02-1.10x rather than below 1.0, so
  the floor is complete without being over-complete.
- **Measured:**
  - **chunked == one-shot, exactly, at every layer.** Mutual PCC = **1.0000000** on K and on V for
    **32/32** layers; the worst of 64 comparisons is 1.0000000. The indexed RoPE is not merely close
    to the contiguous one, it is **bit-identical**: a standalone probe rotating the same tensor both
    ways gave `max|delta| = 0.0` and `torch.equal → True` across all four chunk offsets.
  - **vs the fp32 golden**, per layer, both producers identical to 7 decimal places:
    K min **0.9990570** (L13), max 0.9999662 (L1); V min **0.9957049** (L26), max 0.9999369 (L0).
    The curve decays with depth as the hidden state accumulates the model's own error (`G-MODEL`
    measures 0.9984849 on that hidden state at 32 layers), which is what the step gate exists for.
  - **per-layer error step:** worst gated K step **1.90x** at L13; worst gated V step **1.44x** at
    L11. Budget 4x. No step anywhere in either curve.
  - Full curves in `raw/G-CHUNK_per_layer_pcc.json`.
- **Verdict:** **PASS**
- **Negative controls — two, one per delta (`DEC-065`, `R-027`):**

  | control | breaks | mutual K | mutual V |
  |---|---|---|---|
  | rope every chunk at `kv_actual_global = 0` | delta 1 | **0.72466** | 1.00000 — *invariant* |
  | write every chunk at `kv_actual = 0` | delta 2 | **0.22048** | **0.04473** |

  The recipe specifies only the first and quotes **two** numbers for it (0.706 / 0.655). Its K
  figure and this run's 0.72466 agree to within 3%, so it is the same experiment — but **V cannot
  move under it**, because V is never rotated and both producers see identical hidden states. The
  delta-1 control therefore *asserts* V is unchanged (turning the invariant into a check), and
  delta 2 gets its own control. A single control for a two-delta gate would have left a dropped
  `kv_actual` — the most likely chunked-prefill bug there is — completely ungated.
- **Deviations:** `DEC-061` (the chunk geometry), `DEC-064` (the floor definition), `DEC-065` (two
  controls), `DEC-066` (the trace lives outside the repo).
- **What this gate does NOT prove.** Four things, and they are not small:
  1. **Delta 3** — a chunk-*k* query attending the cached prefix. `BLOCKED`, `R-023`, owner P8.
  2. **The model → cache path.** The write is driven head-by-head from the *stage* level, not from
     `Model.prefill_forward` with a cache attached, because at TP=1 that op refuses. `R-001`,
     `G-KV-TP8`.
  3. **The block-cyclic reorder.** At `sp = 1` it is the identity, so local row == global position
     and the inverse gather is never exercised. `G-MESH-KV`.
  4. **`tt/tt_prefill_runtime.py`.** This gate calls the modules directly; the runtime is not
     instantiated anywhere in P7 (`R-029`).

---

### G-RUNTIME — the runtime against the engine's real call site, statically (P7)
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_prefill_runtime_chunked.py -q`
- **Raw log:** `raw/G-RUNTIME_20260904T131649Z.log` (first run: `raw/G-RUNTIME_20260904T125242Z.log`, identical)
- **Mesh / device:** **none** (`BRINGUP_RECIPE.md:2024` gives this gate device "none"). 37 tests,
  8.2 s, no mesh opened. The two `__init__` refusals are reached with a `_MeshStub` exposing only
  `.shape` — both run before any `ttnn` object is constructed — and the per-chunk refusals with an
  `object.__new__` instance carrying only the three attributes those checks read (`DEC-068`). The
  code under test is the real method on the real class with the real messages; only the state it
  reads is supplied directly.
- **Inputs:** the engine's own source, parsed with `ast`. **Not the contract doc** — that is the
  whole point of the gate (`BRINGUP_RECIPE.md:1687-1690`).
- **Reference dtype policy:** not applicable; no numbers are measured.
- **Threshold:** every unguarded name the engine touches exists on the runtime with a signature that
  binds the engine's actual call; every `runtime.config` field the engine reads exists; every
  refusal raises and is matched on its message; and **the audit itself must reject** a runtime
  written to the doc.
- **Measured — what the engine actually does, read out of `prefill_runner.py`:**
  - **11** attributes accessed on the runtime handle; **9** methods called; **2** fields read off
    `runtime.config` (`is_last_rank` at `:301`, **`use_trace`** at `:303`, `:745`, `:773`); **6**
    hooks reached only behind `getattr`/`hasattr` (`capture_trace`, `kv_migration_base_address`,
    `kv_migration_stages`, `release_trace`, `trace_metadata_msg`, `warmup_ack_count`).
  - **`prefill_chunk` is called with 2 positional arguments and six keywords** —
    `slot_id, actual_start, actual_end, request_id, d2h_service, metadata_msg`
    (`prefill_runner.py:286`). The doc's §2 signature (`ADDING_A_PREFILL_MODEL.md:129`) lists
    neither of the last two.
  - **`build_kv_chunk_table` is called five times, with `path` positional on three of them
    (`:644`, `:655`, `:674`) and keyword on two (`:570`, `:699`)**, plus a `**stage_layout` splat —
    so the parameter must be positional-or-keyword. The recipe's P10 warning names the two
    `prefill_chunk` parameters but not this.
  - Audit of `TtPrefillRuntime`: **0** missing attributes, **0** missing config fields, **0**
    signature problems.
- **Verdict:** **PASS**
- **Negative controls (three, all discriminate):**
  1. **The audit's own control.** `_BrokenRuntime` is a runtime written faithfully to the doc:
     `metadata_msg` absent, `set_layer_completion_sink` / `set_d2h_ack_service` /
     `build_kv_chunk_table` absent, `config.use_trace` absent. The same audit function reports
     **3** missing methods, **1** missing config field and **4** signature problems, including
     *"prefill_chunk cannot accept the engine's call at prefill_runner.py:286 ... got an unexpected
     keyword argument 'metadata_msg'"* — the exact `TypeError` that would otherwise arrive on the
     first served chunk, after the mesh is open and 15 GB of weights are loaded.
  2. **The audit's precondition.** A separate test asserts the walk actually found a
     `runtime.prefill_chunk` call, a `runtime.compile` call, at least one `config` access and at
     least one guarded hook — so an audit over a moved or renamed engine file cannot report a clean
     pass by finding nothing (the failure mode `R-016` describes for citations).
  3. **A refusal census.** The module's `raise` statements are counted with `ast` (**25**) and
     compared against the refusal list this file asserts, so a deleted `raise` shows up as a failing
     count rather than as silently-lost coverage.
- **The 25 refusals, all matched on a metachar-free message substring** (`expect_error`'s `message`
  is a **regex**, not the substring its docstring describes — `R-014`, `DEC-045`): six config-geometry
  refusals (a chunk size that does not divide `max_seq_len`, one below `TILE_SIZE x sp`, a capacity
  that is not a multiple of it, `sp_axis == tp_axis`, zero users, zero layers); two construction
  refusals (a mesh of the wrong shape; **`tp != num_key_value_heads`**, tested at `(1,1)`, `(4,4)`
  and `(4,2)`); eight `prefill_chunk` argument refusals (`d2h_service`, `metadata_msg`, an
  unsupported chunk size, a slot out of range, an empty range, a range wider than one chunk, a chunk
  past capacity, an unaligned `actual_start`); **the delta-3 refusal** (`actual_start > 0` on the
  dense path, naming P8 and `R-023`); three cache-resolution refusals; `make_chunk_input` on a short
  chunk; `compile` on a non-first rank; and five unimplemented engine hooks (`R-024`).
- **Deployment geometry, asserted with its arithmetic (`DEC-061`, closes `R-004`):**
  `chunk_size = 8192`, `max_seq_len = 131072`, mesh `(4,8)` → `sp = 4`, `tp = 8`;
  `8192 % (32 x 4) = 8192 % 128 = 0`; `131072 / 8192 = 16` chunks exactly.
- **Deviations:** `DEC-062` (no cache ownership; `compile(kv_caches)` required, against the outline's
  `=None`), `DEC-063` (the unimplemented hooks are present and raise), `DEC-067` (a 4D per-chip
  `make_chunk_input`, following the outline over the template), `DEC-068` (device-free access to the
  instance methods).
- **What this gate does NOT prove — and this is the largest gap P7 leaves.** **The runtime has never
  been instantiated** (`R-029`). `G-RUNTIME` is static by design and the `tp == num_key_value_heads`
  equality forbids `(1,1)`, so `__init__` past its refusals, `_build_indexed_rope`,
  `make_chunk_input`, `compile` past its rank check and the whole happy path of `prefill_chunk` have
  not executed. What *is* proved is that the class cannot fail with a `TypeError` on the engine's
  call, that all 25 refusals fire with their stated messages, and that the deployment arithmetic is
  right. P8's first act should be to build one and call `compile()` — which warms a second chunk at
  `actual_start = chunk` and so exercises delta 3 immediately.

---

### G-CHUNK-ATTN — chunk *k* attending the cached prefix (delta 3): **BLOCKED**, owner P8
- **Command:** none run. The gate belongs to `tests/unit/test_sp_attention_chunked.py`, a P8 file.
- **Mesh / device:** `(4,8)` with the ring fabric — neither of which P7 owns.
- **Threshold (unchanged, for P8):** ≥ 0.999 at layer 1; deep layers gated by step ≤ 4x; both vs the
  golden (`BRINGUP_RECIPE.md:1981`).
- **Verdict:** **BLOCKED** — `07_RISKS.md` `R-023`, which names **P8** as the owner, as
  `BRINGUP_RECIPE.md:1598-1623` requires.
- **Why it is not weakened into `G-CHUNK`.** The recipe forbids both available shortcuts explicitly,
  and the third — running a cache-backed chunk on the dense path anyway — is the dangerous one: plain
  `is_causal` SDPA assumes Q row 0 aligns with K row 0, so the mask is off by `actual_start` and the
  answer is *plausible* rather than obviously wrong.
- **What stands in its place.** Two refusals, both asserted:
  `tt/attention/prefill.py::attention_forward` refuses `cached_len > 0` (P5.5), and
  `tt/tt_prefill_runtime.py::prefill_chunk` refuses `actual_start > 0` on the dense path with a
  message naming P8, this gate and `R-023` (`G-RUNTIME`'s
  `test_prefill_chunk_refuses_a_cache_backed_chunk_on_the_dense_path`).

---

```
STATUS after P6: gates PASS=13 FAIL=0 DEVIATION=1 BLOCKED=0 | next: P7 (chunked prefill + golden KV)
```
_(Filled in by the P7 session: the P6 session recorded `G-LAYER`, `G-WEIGHTS` and `G-MODEL` as detail
blocks and put `G-LAYER`/`G-WEIGHTS` in the summary table, but left the `G-MODEL` summary row and
this two-line checkpoint unwritten — §1.5 requires one after every phase. The `G-MODEL` row above is
transcribed from that session's own detail block and raw logs; no number is new.)_

```
STATUS after P7: gates PASS=16 FAIL=0 DEVIATION=1 BLOCKED=1 | next: P8 (multi-device: TP, SP and the CCL gates)
Per-phase regression gate: whole package suite **177 passed, 0 failed** in 18m46s (`raw/P7-REGRESSION_20260904T133027Z.log`)
Citations after P7: **488/488 verified, 0 mismatched; 811/811 doc refs resolved**, exit 0
(`raw/G-CITE_20260904T135004Z.log`). CITES grew 427 -> 488 (**+61**): every P7 reference is
content-checked, including all 17 into the recipe's own P7 section and Appendix A rows, and all 20
into the engine's own source (`prefill_runner.py` x19, `prefill_producer.py` x1) — the two files
whose line numbers this phase's whole design rests on (R-016, R-017).
Open DECs needing review: DEC-021 (bf8_b KV dtype — measured at G-KV, stands), DEC-025 (residual
scheme A; P8 owns the switch), DEC-026 (barrier depth 2 — G-RACE's first move if it fails),
DEC-027 (mesh descriptor not yet pinned; G-FABRIC-MATRIX picks it), DEC-038/DEC-041 (the
scatter_output seams still refuse), DEC-042/R-015 (G-ATTN's 8x block budget — settled at layer and
model level, still unreachable at bf16 for the block alone), **DEC-059** (the fp32 golden is 128 MB
at 512 tokens and would be 32 GB at the deployment 128k — revisit before a long-context trace),
**DEC-061** (chunk 8192 is functionally derived but its *performance* is unmeasured; P8/P10 may
revise), **DEC-063** (the six engine hooks that raise — P10 replaces the bodies), **DEC-064** (P8's
G-KV-TP8 and G-MESH-KV should use the same floor term list), **DEC-066** (the golden trace is not in
the repo — a CI question), **DEC-067** (the 4D make_chunk_input shape is first exercised on device in
P8), **DEC-069** (G-GOLDEN stays a script gate, so the regression does not cover it)
Closed this phase: **R-004** (CHUNK_SIZE and MAX_SEQ_LEN chosen with their arithmetic — DEC-061),
**R-007** (the indexed RoPE now has a Llama-shaped test: **bit-identical** to the contiguous
builder at four chunk offsets, `torch.equal`, and mutual PCC exactly 1.0000000 on all 32 layers at
the KV level; the `sp > 1` block-cyclic path stays open under R-001), **R-021's P7 obligation** (the
corrected floor definition is applied at G-CHUNK, with both floors recorded).
Opened this phase: R-023 (delta 3 BLOCKED, owner P8), R-024 (six engine hooks raise, owner P10),
R-025 (the recipe describes verify_golden_kv.py two incompatible ways), R-026 (the 128 MB golden
trace is not in the repo), R-027 (the recipe's delta-1 control quotes a V number no correct
implementation can produce), R-028 (P7 step 4's named template is the delta-3 test P7 is forbidden to
run), **R-029 (high: TtPrefillRuntime has never been instantiated)**.
```

**STOPPED HERE, ON A GATE BOUNDARY.** Supersedes the end-of-P5.6 stop note above.
**P0-P6 and all of P7 are complete and gated; P8 (multi-device) is next**, then P10, then P9.

P7's three gates are recorded: `G-CHUNK` **PASS**, `G-GOLDEN` **PASS**, `G-RUNTIME` **PASS**, and
`G-CHUNK-ATTN` is **BLOCKED** with `R-023` naming P8 as its owner — which
`BRINGUP_RECIPE.md:1598-1623` requires by construction, not as a concession.

New in the tree: `tt/tt_prefill_runtime.py`, `scripts/generate_golden_kv_cache.py`,
`scripts/verify_golden_kv.py`, `tests/unit/test_attention_chunked_vs_ref.py`,
`tests/unit/test_prefill_runtime_chunked.py`. `scripts/verify_citations.py` gained 61 `CITES` entries.
**No P0-P6 module was touched**: every file under `tt/` that existed before this phase is
byte-identical to the state P6 gated (`git status` shows four modified files — the three
append-only logs and `scripts/verify_citations.py`, which §1.6 requires extending every phase).
So the 13 gates P0-P6 recorded still hold against this tree without re-running, and the P7
regression re-ran them anyway.

What P8 can rely on, and what it must not assume:

1. **Deltas 1 and 2 are exact, not merely close.** The indexed RoPE is **bit-identical** to the
   contiguous builder — `max|delta| = 0.0` on every chunk, `torch.equal → True`, gated as an
   *address* claim rather than a PCC one (§2.5) — and the chunked KV producer scores mutual PCC
   **1.0000000** against the one-shot producer on K and V at **32/32** layers. So if a P8
   chunked-attention number is wrong, the RoPE offset and the write offset are **not** where to look
   — the ring SDPA's cache read is. Both facts hold at `sp = 1` only; the block-cyclic reorder is
   still the identity here.
2. **The runtime has never run.** `R-029`, and it is the biggest thing P7 leaves open.
   `G-RUNTIME` proves the class satisfies the engine's real call site and that all 25 refusals fire;
   it proves nothing executes. **P8's first act on the `(4,8)` mesh should be to build a
   `TtPrefillRuntime` and call `compile()`** — that warms a chunk at `actual_start = chunk_size`, so
   it is also the cheapest possible smoke test of the ring path.
3. **The golden trace is the shared reference for every later KV gate.** 32 layers, 512 tokens,
   fp32, proved bit-identical to `LlamaModel`'s own loop. `G-KV-TP8`, `G-CHUNK-ATTN` and `G-MESH-KV`
   should all score against **this** trace so their numbers are comparable to `G-CHUNK`'s, and it is
   regenerated by one command (`R-026` explains why it is not committed).
4. **Use the complete floor, not the storage floor** (`DEC-064`, and `R-021` requires it). At layer 0
   the two differ by 23% on K and 67% on V, and the storage floor is the one that inflates the ratio.
   The term list is: bf16 producer input, bf16 norm gain, bf8_b projection weight quantised **in the
   transposed Meta-swizzled orientation**, bf16 RoPE tables, bf8_b store quantised **after** the
   HF→Meta permutation. No internal intermediate.
5. **Permute before quantising.** `bfloat8_b` shares one exponent per 16-element block of the last
   dim, so quantising in HF order and then permuting to Meta is **not** the same tensor as permuting
   and then quantising. The device does the latter. Every K comparison in P8 has this hazard.
6. **`G-CHUNK` still says nothing about the model → cache path or the block-cyclic reorder.** At
   `sp = 1` the reorder is the identity and the write is driven head-by-head from the stage level.
   `R-001` is unchanged; `G-KV-TP8` and `G-MESH-KV` own both.
7. **Two refusals stand where delta 3 will go**, and P8 must remove them deliberately rather than
   route around them: `attention_forward` refuses `cached_len > 0`, and `prefill_chunk` refuses
   `actual_start > 0` on the dense path. Both name `G-CHUNK-ATTN` and `R-023` in their messages.
8. **The engine's call site is wider than its contract doc, and the AST walk is the source of
   truth.** It is re-derived at test time, so a P10 session should read `G-RUNTIME`'s gate block and
   `R-024`'s table rather than `ADDING_A_PREFILL_MODEL.md` §2: two methods and one `config` field
   the doc never mentions are called unguarded, and `build_kv_chunk_table`'s `path` is positional at
   three of its five call sites.
9. **A negative control must be able to fail for the *specific* thing it names.** `G-CHUNK` needed
   two controls where the recipe specified one, because the specified control cannot move V at all
   (`R-027`). Count the things a gate claims and count the controls.

---

## P8 — Multi-device: TP, SP and the CCL gates

`G-FABRIC-MATRIX` ran **first**, before every numerical multi-device gate, as
`BRINGUP_RECIPE.md:1755-1757` requires — and it is the reason the rest of the phase is configured the
way it is.

### G-FABRIC-MATRIX — which (mesh, topology, links, axis) combinations can run a collective
- **Command:** `python models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py`
  (+ `--cases <subset>` for the addendum, `--control` for the harness's own control)
- **Mesh / device:** one Blackhole Galaxy, 32 devices. Every shape below `(4,8)` is a **submesh** of
  one open `(4,8)`; the two `toplevel_*` cases deliberately open a partial mesh directly.
- **Inputs:** each case all-gathers a rank-labelled `[1,1,32,32]` bf16 tile along the case's axis
  through the package's own `MeshConfig.allgather` / `CCLManager` — not a raw
  `ttnn.experimental.*` call — and checks the result **bit-exactly** (`torch.equal`). Payload
  integers are <= 31, far inside `bfloat16`'s exact-integer ceiling of 256, so a mismatch cannot be
  the probe's own numerics.
- **Reference dtype policy:** none — the expected tensor is a concatenation of exact small integers,
  so the comparison is bit-equality, not a floor.
- **Threshold:** every case matches the expectation **written before the sweep ran**, including the
  cases expected to fail or hang. Expectations came from the recipe's own measured claims
  (`BRINGUP_RECIPE.md:82-84`, `:1690-1729`, `:1701-1709`).
- **Noise floor:** not applicable (bit-exact claim; floor is 1.0 by construction).
- **Measured — 12 cases, 7 matched:**

  | case | expect | got | secs | detail |
  |---|---|---|---|---|
  | `toplevel_1x8_fabric1d` | error | **error** | 29.2 | `TT_THROW @ fabric_firmware_initializer.cpp:271` |
  | `toplevel_2x8_fabric1d` | error | **error** | 29.5 | same |
  | `submesh_1x2_linear_l1_ax1` | ok | **ok** | 22.1 | bit-exact on 2 devices |
  | `submesh_1x4_linear_l1_ax1` | ok | **ok** | 22.3 | bit-exact on 4 devices |
  | `submesh_1x8_linear_l1_ax1` | ok | **ok** | 21.0 | bit-exact on 8 devices |
  | `submesh_1x8_ring_l1_ax1` | ok | error | 18.2 | `topology_mapper.cpp:544` — the torus descriptor does not map |
  | `submesh_2x8_ring_l2_ax1` | ok | error | 18.3 | same |
  | `full_4x8_ring_l2_ax1` | ok | error | 18.3 | same |
  | `full_4x8_ring_l2_ax0` | ok | error | 18.4 | same |
  | `submesh_1x8_ring_on_fabric1d` | **hang** | ok | 21.0 | bit-exact on 8 devices — the recipe's "hangs rather than errors" does not hold here |
  | `overlap_1x2_then_1x8_no_quiesce` | hang | **hang** | 246.3 | no result in 240 s; `tt-smi -r` exit 0 in 43.1 s |
  | `overlap_1x2_then_1x8_quiesce` | ok | **ok** | 22.4 | both phases bit-exact |

- **Addendum, 5/5 matched** (expectations written from the `(1,8)` result before running):
  `submesh_2x8_ring_on_fabric1d_l2_ax1`, `full_4x8_ring_on_fabric1d_l2_ax1`,
  `full_4x8_ring_on_fabric1d_l2_ax0`, `full_4x8_linear_l2_ax1`, `full_4x8_linear_l2_ax0` — all
  **ok** and bit-exact, 20.7-22.0 s each.
- **Verdict:** **PASS-WITH-DEVIATION** (`DEC-071`, `DEC-079`, `DEC-081`). Five expectations, all of
  them taken from the recipe, do not hold on this galaxy. Each has a measured cause and a decision;
  none was restated after the fact to manufacture a `PASS`.
- **Negative control:** `--control` states a known-`ok` case's expectation as `hang` and requires the
  sweep to report a mismatch. It did (`CONTROL PASSED`), so the comparison is running and "every case
  matched" is not vacuous.
- **Deviations, and what they mean:**
  1. **`FABRIC_1D_RING` is unavailable on this galaxy.** The only single-galaxy RING/RING descriptor
     fails to map — "32 target node(s) are not mapped to any global node" — and it is not the channel
     policy (a `RELAXED` copy fails identically), so the torus wrap links are not present.
     `FABRIC_1D_RING` on a LINE/LINE descriptor is refused earlier, at `mesh_graph.cpp:447-454`:
     "FabricConfig can only restrict topology (e.g., torus->mesh), not create new connections."
     `DEC-079`.
  2. **`ttnn.Topology.Ring` collectives on `FABRIC_1D` do not hang** — they are bit-exact at every
     P8 shape and both axes. The recipe's claim is false here. `DEC-079`.
  3. The two descriptors the recipe names are **multi-galaxy** and out of scope. `DEC-071`.
- **Notes:** the recipe cites `fabric_firmware_initializer.cpp:200` for the top-level-partial-mesh
  failure; the throw on this build is at **`:271`**. The `overlap` hang confirms the worst landmine in
  the set exactly as documented, including that it poisons the box: the case after it only passed
  because the harness had reset the machine. An earlier, **aborted** sweep is at
  `raw/G-FABRIC-MATRIX_20260904T142547Z.log`; a `_repo_root()` walk one level short made
  `TT_MESH_GRAPH_DESC_PATH` nonexistent and every case reported `std::filesystem::exists(...)`
  instead of what it was measuring — including two that "matched" for the wrong reason. Kept as
  evidence; `_repo_root()` now asserts the descriptor directory exists.

### G-KV-TP8 — the model -> cache path at TP=8
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_cache_tp8.py -x -q`
- **Mesh / device:** `(1, 8)` **submesh** of the full `(4,8)` galaxy. TP=8, **SP=1** — deliberately,
  because at `sp = 1` the block-cyclic sequence layout is the identity and the only thing under test
  is the head/feature distribution, so a failure can only be the mapper
  (`BRINGUP_RECIPE.md:1758-1760`). `Topology.Linear`, `FABRIC_1D`, `num_links=1`.
- **Inputs / input distribution:**
  - **arm A (head->column, and the write offset)** — a synthetic **labelled** probe, not a random
    one: lane block `[0,64)` of each head carries the position `s % 128`, lane block `[64,128)`
    carries the head id `c+1`. Both are exactly representable in `bfloat8_b`, whose measured
    exact-integer ceiling on this box is **128**, not the recipe's blanket 257 (`R-013`, `DEC-044`);
    every payload is `< 128`. A random input cannot make an address claim exact, and §2.5 requires
    this claim to be exact.
  - **arm B (vs the golden)** — the golden trace's own 512 `token_ids` (real
    Llama-3.1-8B-Instruct tokens) and **real checkpoint weights**: §2.2.2's real-embedding-scale arm,
    the one that predicts model behaviour.
- **Reference dtype policy:** arm A needs none — the expected tensor is built on the host from
  integers that both `bfloat16` (activations) and `bfloat8_b` (cache) hold exactly, so the comparison
  is `torch.equal`. Arm B scores against the **fp32** golden (`DEC-059`, bit-identical to
  `LlamaModel`'s own loop at `G-GOLDEN`), with K permuted **HF -> Meta before** the quantiser
  (recipe §2.2.3a).
- **Threshold:** head->column and the write offset **bit-exact** (`rtol=atol=0`); K >= 0.99 /
  V >= 0.98 vs the golden and L0 error ratio <= 3x, both **carried from `G-CHUNK`** rather than
  chosen here — a threshold chosen against this measurement could not fail, and carrying P7's makes
  the TP split's cost readable directly.
- **Computed noise floor (arm B, layer 0):** `G-CHUNK`'s **complete** floor — every value the device
  holds rounded to the dtype it holds it in (bf16 embedding output, bf16 norm gain, `bfloat8_b`
  projection weight in its transposed Meta-swizzled orientation, bf16 RoPE tables, `bfloat8_b`
  store), all remaining math fp32, no internal intermediate quantised. The **storage-only** floor is
  recorded beside it and is the optimistic one (`R-021`, `DEC-064`).
- **Measured:**
  - head->column: **8/8 columns bit-identical** for `v_with_rope` and **8/8** for `k_without_rope`;
    pad tail rows `[128, 512)` on column 0 `max|x| = 0.0`.
  - write offset `{0, 128, 256}` x 8 columns: **24/24 blocks bit-identical**; pad tail
    `max|x| = 0.0`.
  - arm B, 32 layers: **min K 0.9986432** (L22), **min V 0.9942853** (L26); L0 K **0.9999617**
    (complete floor -> **1.10x**, storage floor -> 1.35x), L0 V **0.9999369** (complete -> **1.02x**,
    storage -> 1.70x); pad tail `max|x| = 0.0` over all 64 (layer, cache) pairs.
- **Verdict:** PASS
- **Negative control:** read mesh column `(c+1) % 8` as KV head `c` — a completely wrong head->column
  map. **PCC 0.99887** (recipe §2.5 measured 0.99890 for the same control), `torch.equal` **False**,
  `max|delta| = 7.0`. The gate asserts *both* directions: the control must fail bit-equality **and**
  must score above 0.9, because a control that PCC also rejected would not be demonstrating the
  weakness it exists to demonstrate. This is the measurement behind "gate every layout, mapping or
  address claim on bit-equality, never PCC".
- **What this closes:** `R-001` — `G-KV` at `(1,1)` proved the cache *primitive* with one synthetic
  head, at a head count the deployment mesh never emits. This proves the model's own K and V, sharded
  by the weight loader's `column_parallel` mapper, land on the chips the cache expects.
- **What it does not cover:** the block-cyclic reorder (identity at `sp = 1`) — `G-MESH-KV`'s; and a
  bit-exact check of K's *post-RoPE* head->column placement, which is impossible by construction
  (`DEC-078`) and is covered numerically by arm B.
- **Deviations:** `DEC-078` (two probes, one per RoPE state), `DEC-080` (the write offset moves to a
  test without an attention core, because the dense core correctly refuses `cached_len > 0`).

### G-SP-RING — the ring-joint SP attention core, alone
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_dense_sp_vs_ref.py -q`
- **Mesh / device:** the deployment `(4, 8)`, SP=4 x TP=8, `num_links=2`, `Topology.Linear`,
  `FABRIC_1D`. Compute grid **(12, 10)**; CCL offset **(11, 0)**; ring SDPA grid **(11, 10)** — the
  compute grid minus the CCL column, so `ring_joint_sdpa_device_operation.cpp:421`'s
  `ccl_core_grid_offset.x >= sdpa_grid.x` holds as `11 >= 11` exactly. The dense path's grid stays a
  pinned 8x8; the two are never unified.
- **Geometry:** `chunk_global=512`, `chunk_local=128`, `kv_actual=512`, `logical_n=1024`,
  `cache_global=2048`. The cache is populated **through the real write op** from a host tensor in
  global-position order, mesh-mapped `dims=(2,1)` exactly as the model maps a chunk, so the reference
  needs no layout arithmetic at all: "causal attention, Q at global positions `[512, 1024)`, K/V at
  `[0, 1024)`".
- **Inputs / input distribution:** **standard-normal, iid** Q/K/V. Stated because it matters more
  here than anywhere else: recipe §2.3 measured that a *fused* kernel's error **ratio** is
  distribution-dependent even though the floor is not — the same single-card SDPA sits at 71x on iid
  standard-normal Q/K and 27-29x on real correlated post-RoPE activations. So **6.05x below is the
  iid number and is not comparable with an in-model measurement of the same op**; `G-CHUNK-ATTN` is
  the in-model arm.
- **Reference dtype policy:** Q rounded to **bf16** (what the projection hands the op), K/V rounded
  to **`bfloat8_b`** (what the cache holds), **all remaining math fp32** — the mask, both matmuls and
  the softmax. No internal intermediate is quantised (§2.2's conservative reading).
- **Threshold:** PCC >= 0.99 (the recipe's). The ratio to its own floor is **recorded, not asserted**:
  §2.3 measures that a fused kernel does not sit at its floor and §2.3.1 that a ratio budget is not
  portable, so a threshold on this op's ratio would be a number invented here.
- **Computed noise floor:** **0.9999450** — the same fp32 reference re-run on the *un-rounded* fp32
  inputs, scored against the reference on the device's stored values. The gap is the storage rounding
  alone, with no kernel in it.
- **Measured:** **PCC 0.9996672 -> 6.05x** the floor. (Recipe's figure for this op: 7.98x. The
  single-card SDPA measured 52.8-55.0x standalone at `G-ATTN`.)
- **Verdict:** PASS
- **Negative controls — three, and only one of them is numeric:**
  1. **`fp32_dest_acc_en=True` is REFUSED**, structurally (§1.4 counts a configuration that must
     refuse as a control). Verbatim:
     `TT_FATAL @ ring_joint_sdpa_program_factory.cpp:1308: !kv_pad_rotation_enabled || use_streaming_compute`
     — "kv_actual_isl requires the ring-joint streaming compute path; the compute_common.hpp path
     selected by fp32_dest_acc_en=true is not supported." The same call with `False` is accepted,
     output `(1,4,128,128)`, so the refusal is about the flag and not the shapes. This is the **one
     op in this model where `False` is mandatory** rather than the package default of `True`.
  2. **a wrong `kv_cache_batch_idx`** — the numeric one (`DEC-082`). Two layers populated with
     different K/V; reading layer 1 with `batch_idx = slot*num_layers + layer` scores **0.9996770**,
     and with `batch_idx = slot` alone — the template's warned-about bug, which makes every layer read
     layer 0's cache — scores **-0.00337**.
  3. **a wrong `kv_actual_isl`** is also **refused**:
     `ring_joint_sdpa_device_operation.cpp:278: new_actual_isl <= chunk_capacity` — "Got
     new_actual_isl=1024, chunk capacity=512". Worth stating as a positive: an
     off-by-`actual_start` chunk **cannot** silently produce a wrong answer through this op.
- **Notes:** the first run of this gate used `Topology.Ring` and aborted with
  `fabric.cpp:174: forwarding_direction.has_value()`, "Could not find any forwarding direction from
  src (M0, D0) to dst (M0, D3)" — the SP ring closing on itself. That abort is what settled the
  topology for the whole phase (`DEC-081`) and is kept at
  `raw/G-SP-RING_20260904T145034Z.log`.

### G-CHUNK-ATTN — delta 3, chunk *k* attending the prefix read out of the cache
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_chunked_attention_ring.py -q`
- **Mesh / device:** the deployment `(4, 8)`, SP=4 x TP=8, `Topology.Linear`, `FABRIC_1D`. **One**
  `TtPrefillRuntime`, one `CCLManager`, one set of real weights; the two arms differ only in the
  chunk size each `prefill_chunk` call passes (`DEC-086`).
- **The two arms, and which core each ran (asserted, not inferred):**
  | arm | chunk_global | chunks | core |
  |---|---|---|---|
  | one-shot | 1024 == `max_seq_len` | 1 | **`sp_bootstrap`** |
  | chunked | 512 | 2 (`actual_start` 0 and 512) | **`sp_ring`** |
- **Inputs / input distribution:** the golden trace's own **1024** `token_ids` — real
  Llama-3.1-8B-Instruct tokens — and real checkpoint weights. There is no synthetic arm: the quantity
  of interest is an accumulated model-level statistic, and a `randn` input would dilute the attention
  error through the residual stream (measured 1.47-1.81x vs 3.83x at `G-LAYER`).
- **Reference dtype policy:** the golden is **fp32** throughout, regenerated at 1024 tokens for this
  gate and re-proved bit-identical to `LlamaModel`'s own loop — `max|delta| = 0.0` on K, V **and** the
  post-norm hidden over all 32 layers. K is permuted HF -> Meta **before** any quantiser (§2.2.3a).
- **Threshold, and it names a depth** (`BRINGUP_RECIPE.md:1795-1799`): **>= 0.999 at layer 1** — one
  attention layer, i.e. the per-op claim. Deep layers by the per-layer error **step** <= 4x from L3.
  Both arms vs the golden at `G-CHUNK`'s carried K >= 0.99 / V >= 0.98. The accumulated min over 32
  layers is **recorded and not gated**.
- **Computed noise floor:** for the *mutual* comparison the floor is **1.0 by construction** — same
  weights, same dtypes, same tokens, so at layer 1 the two arms differ only by the attention core and
  the expected value is exactness. Against the golden, the layer-0 floors are `G-KV-TP8`'s (1.10x on
  K, 1.02x on V, measured on the identical producer) and are deliberately not recomputed here.
- **Measured — the depth structure the threshold has to name:**
  | layer | ring vs one-shot, K | recipe's own table |
  |---|---|---|
  | 0 (no attention has run) | **1.0000000** | 1.00000 |
  | 1 (**one** attention layer) | **0.9999505** | 0.99996 |
  | 8 | 0.9995954 | 0.99952 |
  | 22 (the min) | **0.9967975** | 0.99628 |
  Mutual V at L1: **0.9997938**. Worst gated per-layer step: K **2.14x** at L13, V **1.58x** at L3,
  against a 4x budget. Vs the golden: one-shot min K 0.9987994 / V 0.9942686; ring min K 0.9967119 /
  V 0.9868228 — the ring path carries **2.74x** the one-shot path's K error (the recipe measured
  1.45x on its own shapes).
- **Verdict:** **PASS** — and it closes `R-023`, the gate P7 recorded `BLOCKED` by construction.
- **Negative control:** serve chunk 1 against a cache whose chunk-0 prefix was **never written**,
  breaking the one thing delta 3 is. Signature: **L0 = 1.0000000 (unmoved)**, L1 = **0.99695**,
  worst over 6 layers = **0.87279**. L0 cannot move because its K and V are produced before any
  attention runs — **a layer-0-only check would have passed on a completely broken ring read**, which
  is exactly why this gate's per-op threshold sits at layer 1.
- **Deviations:** none. The 512-token golden trace could not host this gate (two chunks of >= 512
  global need >= 1024 tokens), so a 1024-token trace was generated with the same generator and the
  same `--verify-loop` check; `raw/G-GOLDEN-GEN-S1024_20260904T145507Z.log`.

### G-TP-PARITY — the TP collectives are exact
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_tp_parity.py -q`
- **Mesh / device:** `(1,1)`, `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)` — **all submeshes of one
  open `(4,8)`**, with `parent.quiesce_devices()` on both sides of every phase (`DEC-077`). `(2,8)` is
  not optional: `get_default_num_links` returns 1 for any single-row mesh, so every `(1,N)` shape runs
  `num_links=1` and never touches the deployment link count.
- **Inputs / input distribution:** **standard-normal** `[1, 1, 128, 4096]`, weights standard-normal
  scaled 0.02 so the residual blocks stay in a plausible activation range. §2.2.1's warning about
  synthetic input scale does **not** bite here: this is a device-vs-device comparison, both sides see
  the identical input, and there is no reference-scale dilution — that caveat applies to floor
  comparisons, not to an exactness claim about a collective.
- **Reference dtype policy:** **none, and that is the point.** The reference is the *same module on
  one chip* at the same dtypes (`bfloat8_b` weights, `bfloat16` activations), so no torch precision
  enters the comparison. This is a strictly sharper instrument than either arm's PCC against torch.
- **Threshold:** PCC >= 0.999 on all five multi-device shapes; control <= 0.95.
- **Computed noise floor:** not applicable, deliberately — a collective is exact up to reduction
  order, so the *expected* value is 1.0 and there is no dtype floor to sit at. What is recorded
  instead: column-parallel sharding **cannot** move `bfloat8_b`'s exponent blocks (the shard boundary
  `4096/8 = 512` is a multiple of the 16-element block), so the only source of disagreement is the
  reduction order of the row-parallel matmul plus the collective.
- **Measured — worst device of each shape:**
  | module | (1,2) | (1,4) | (1,8) | (2,8) | (4,8) |
  |---|---|---|---|---|---|
  | `rms_norm` | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | **1.0000000** |
  | `mlp` | 0.9999963 | 0.9999943 | 0.9999916 | 0.9999916 | **0.9999915** |
  | `attention` | 0.9999963 | 0.9999943 | 0.9999917 | 0.9999917 | **0.9999917** |
  | `layer` | 0.9999854 | 0.9999798 | 0.9999733 | 0.9999733 | **0.9999733** |
  `rms_norm` is **bit-identical** at every shape, which is the expected result: it has no collective
  and a replicated gain.
- **Verdict:** PASS
- **Negative control:** the `(1,1)` reference **rolled by one whole TP shard** (512 features) scores
  **0.00307** against the `(1,8)` output, while the un-rolled comparison scores 0.9999916. So the gate
  would notice a module whose output features landed one shard out of place.
- **Deviations:** `DEC-083` — the sequence is sharded (and slices compared) only for the *token-wise*
  modules `rms_norm` and `mlp`; `attention` and `layer` replicate it, because the dense causal SDPA
  mixes tokens and its sequence-sharded output is not a slice of the single-device output at all.
  The recipe's sentence at `:1768-1770` is true only for the token-wise pair.

### G-RACE — no semaphore races
- **Command:** `PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 PREFILL_RACE_ITERS=3
  PREFILL_KV_PCC_MIN_K=0.99 PREFILL_KV_PCC_MIN_V=0.98 python
  models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py`
- **Mesh / device:** the deployment `(4, 8)`, **one process, one `CCLManager`**, three full 32-layer
  two-chunk prefills with a fresh cache each time.
- **Inputs:** the golden trace's 1024 `token_ids`, real weights — i.e. the deployment workload, not a
  probe.
- **Reference dtype policy:** not applicable — this is a bit-identity claim between three device
  runs.
- **Threshold:** the three per-layer PCC tables must be **bit-identical**. Comparand: SHA-256 of the
  table at full float repr.
- **Noise floor:** not applicable (identity claim; the expected value is one hash).
- **Measured:** **one** distinct hash across three runs:
  `b7abb6481ee1efb569e072924e08256310fcf8ab96c77205c76d16cb9d639d28`
  (run 0 / run 1 / run 2, min K 0.9967119 and min V 0.9868228 on all three; 365.0 / 366.8 / 366.8 ms).
  The same hash was produced by the **separate single-run process** earlier
  (`raw/G-MESH-KV-chunked512_20260904T150451Z.log.gz`), so determinism holds across processes as well —
  recorded, not gated.
- **Verdict:** PASS
- **Negative control:** the identity comparison **is** its own control (§1.4: "a bit-identity
  comparison across repeated runs counts as its own"). Separately, `G-SEMAPHORE`'s per-layer-manager
  control shows what the failure would look like structurally.
- **Scope of this pass, as the recipe insists on stating:** 3 runs x 2 chunks x 32 layers x 2
  collectives per layer is a few hundred collectives on one user slot, with the barrier ping-pong
  only **2 deep** (`DEC-026`). Hundreds of collectives is not hundreds of thousands, and this says
  nothing about multi-user slots. The documented first move if it ever fails — deepening the barrier
  ring from 2 to 4 — was **not** taken pre-emptively, precisely so the gate could measure.

### G-SEMAPHORE — CCL state allocated once
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_ccl_semaphores.py -q`
- **Mesh / device:** the one-card arms on `(1,1)`; the P8 arm on the deployment `(4, 8)`.
- **Inputs:** the P8 arm builds the real 32-layer model **cache-only** (`state_dict={}` plus the
  `G-WEIGHTS` weight cache) and runs one 128-token chunk through every layer. What is under test is
  the semaphore inventory, not the numbers.
- **Reference dtype policy / noise floor:** not applicable — exact list lengths.
- **Threshold:** 6 RS + 4 AG + 2 barrier + 2 ring-attention = **14**, unchanged at construction, after
  dozens of getter cycles, and **after a real multi-layer run**.
- **Measured:** `{rs: 6, ag: 4, barrier: 2, ring_attention: 2}` at construction, after 32x4 getter
  cycles, after the `(4,8)` build, and after a real 32-layer forward — identical at every point.
  Ping-pong indices `rs=0 ag=0 barrier=0`, all inside the depth-2 ring. `TtPrefillRuntime` reused the
  manager passed to it rather than building a second.
- **Verdict:** PASS
- **Negative control:** construct one manager per simulated layer — 3 managers -> **42** semaphores,
  the count the correct code must never produce. This is the only one of the three arms that could
  catch a `CCLManager` rebuilt inside `prefill_chunk` or inside a layer, because the getter-cycle arms
  never construct a second manager.
- **Deviations:** none.

### G-MESH-KV — full-model KV vs the fp32 golden on the target mesh
- **Command:** `[PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=<n>] python
  models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py`
- **Mesh / device:** the deployment `(4, 8)`, SP=4 x TP=8, `Topology.Linear`, `FABRIC_1D`,
  `num_links=2`. Driven through **`TtPrefillRuntime`** including its `compile()` — the first time that
  object has ever been instantiated (`DEC-084`, closing `R-029`).
- **Inputs / input distribution:** the golden trace's 1024 `token_ids` and real checkpoint weights.
- **Reference dtype policy:** the **fp32** golden, bit-identical to `LlamaModel`'s own loop; K
  permuted HF -> Meta before any quantiser.
- **Threshold:** per-layer min recorded; K >= 0.99 / V >= 0.98, **carried** from `G-CHUNK`.
- **Computed noise floor:** the layer-0 complete floor is `G-KV-TP8`'s (1.10x K / 1.02x V) — the same
  producer, and not recomputed here.
- **Measured:**
  | configuration | core | chunk_local | min K | min V | wall |
  |---|---|---|---|---|---|
  | one-shot, chunk 1024 | `sp_bootstrap` | 256 | **0.9987994** (L22) | **0.9942686** (L28) | 220.5 ms, 4643 tok/s |
  | chunked, chunk 512 (2 chunks) | `sp_ring` | 128 | **0.9967119** (L22) | **0.9868228** (L28) | 365.0 ms |
  | chunked, chunk 256 (4 chunks) | `sp_ring` | 64 | **0.9967844** (L22) | **0.9866232** (L28) | 765.3 ms |
- **Verdict:** PASS
- **Negative control:** the read-back's block-cyclic layout claim is what could be silently wrong
  here, and it is controlled **structurally** rather than by a perturbation: the period is
  `chunk_local = chunk_global // sp`, so the three configurations read the cache at **three different
  periods (256, 128, 64)** and all three score. A read-back with the wrong period cannot do that. The
  head->column half of the same claim is gated bit-exactly by `G-KV-TP8`, and the layout map itself
  is asserted to cover every global position exactly once before any scoring happens.
- **HUMAN GATE H5 considered and resolved, not waved through.** The ring path carrying **2.74x** the
  one-shot path's K error is higher than the recipe's own measurement of the same quantity (1.45x,
  `BRINGUP_RECIPE.md:1748-1751`), which is the shape of thing §0.3's H5 says to investigate rather
  than record as clean. It resolves arithmetically and the resolution is §2.3.1's, not a new one:
  the two runs **agree on the chunked number** (chunked min K 0.9967119 here vs the recipe's
  0.99695) and differ on the *baseline* (one-shot min K 0.9987994 here vs the recipe's 0.99789 —
  ours is **better**). A roughly fixed absolute kernel error divided by a smaller baseline error
  gives a larger ratio: exactly "a fixed-error stage breaks the ratio metric, and the failure runs
  the *wrong way* — it penalises the **more accurate** configuration". Absolute error attributable
  to the ring path: `(1 - 0.9967119) - (1 - 0.9987994) = 2.088e-3` here against the recipe's
  `3.05e-3 - 2.11e-3 = 0.94e-3`; both are the same order and the difference is one trace and one
  chunk geometry apart. **Nothing is unattributed**, so this is recorded rather than escalated —
  but it is recorded, because a ratio that moved 1.9x between two runs of the same code is exactly
  what H5 exists to make someone look at.
- **Notes:** the two chunked arms are within 1e-4 of each other on both K and V, which says the ring
  op's error is not sensitive to the chunk length in this range — and it is what retires the concern
  behind `DEC-073` (a pinned `q_chunk_size=128` does not constrain the deployable chunk size; the
  `chunk_local=64` arm ran with `q_chunk_size=128` and scored the same).

### G-WEIGHTS (P8 extension) — the cache-only rebuild at TP=8
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py -q -k tp8`
- **Mesh / device:** the deployment `(4, 8)`. Weight cache root under `tmp_path`, so the arm cannot
  pass on a cache another gate wrote.
- **Inputs:** real checkpoint weights for a one-layer model plus the embedding, final norm and LM
  head; built once with the checkpoint + a cache path, then again with an **empty** state dict and the
  same path.
- **Reference dtype policy / noise floor:** not applicable — SHA-256 identity.
- **Threshold:** every device tensor SHA-256-identical across the two builds.
- **Measured:** **354 device shards over 12 tensors, all identical.** 8 tensors genuinely sharded
  with **8 distinct** shard hashes each — the 8 TP columns, replicated across the 4 SP rows, which is
  exactly the expected geometry — and 4 replicated (`model.embed_tokens.weight`, the two layer norms,
  `model.norm.weight`). 12 cache files. The cache path is `tensor_cache_bfp8_4x8`, asserted to carry
  the mesh shape so a `(1,1)` cache cannot be picked up here.
- **Verdict:** PASS
- **Negative control:** two, both structural and both about this arm not being the `(1,1)` arm with
  more devices: the replicated vocab table **must** have exactly one distinct shard hash, and
  `q_proj` and `lm_head` **must** have more than one. An arm that silently stopped sharding fails
  rather than passing faster.
- **Deviations:** `DEC-087` — the replicated `[128256, 4096]` embedding table is hashed on the first
  and last device rather than all 32 (33.6 GB of D2H per pass, twice, to re-prove a tensor the mesh
  does not shard). Its per-device claim is the `(1,1)` arm's; the first-and-last pair keeps the
  *replication* falsifiable. The gap is stated: devices 1-30's copies of that one table are not
  hashed at TP=8.

```
STATUS after P8: gates PASS=24 FAIL=0 DEVIATION=2 BLOCKED=0 | next: P10 (disaggregated-prefill integration)
Open DECs needing review: DEC-079/DEC-081 (this galaxy has no ring fabric; the whole phase ran on
FABRIC_1D + Topology.Linear, so every P8 number is a Linear measurement — R-030, R-031),
DEC-083 (G-TP-PARITY shards the sequence only for the token-wise modules, a stated deviation from
BRINGUP_RECIPE.md:1791-1793), DEC-085 (meta_head_index duplicated between the script and the tests),
DEC-087 (G-WEIGHTS's P8 arm does not hash the replicated vocab table on devices 1-30)
```

`DEVIATION=2` is `G-ATTN` (`DEC-042`, from P5.5) and `G-FABRIC-MATRIX` (`DEC-071`/`DEC-079`/`DEC-081`).
The counts follow the earlier phases' convention of tallying **Appendix A** gates only, so the two
cross-cutting rows sit outside them: `P8-REGRESSION` is `PASS` (196/196) and `G-CITE (P8)` is
`PASS-WITH-DEVIATION` — clean on all 539 content-checked citations and all 900 doc refs, red on the
two pre-existing P5 raw-log references its new artefact pass exposed (`R-042`).
`BLOCKED=0`: P7's single `BLOCKED` row, `G-CHUNK-ATTN`, is **closed** by the P8 row above, and no P8
gate needed a second galaxy — multi-galaxy is scoped out rather than deferred (`R-032`), so nothing
was recorded `BLOCKED` for it.

**STOPPED HERE, ON A GATE BOUNDARY.** Every gate Appendix A assigns to P8 has run on device with a
recorded number, a raw log, an input distribution, a reference dtype policy, a computed floor (or a
stated reason there is none) and a negative control. **P10 (disaggregated-prefill integration) is
next**, then P9 (cleanliness). Two things P10 should read first:

1. `R-030` / `R-031` — every number in this phase was measured on `FABRIC_1D` + `Topology.Linear`,
   because this galaxy has no ring fabric. If P10 runs the engine under a different fabric
   configuration, the P8 numbers do not transfer.
2. `R-039` — the deployment chunk/cache pair (8192 / 131072) has never been run, and the
   `sp_bootstrap` core has a gate but no deployment use. Both are cheap to settle and neither is a P8
   gate.

### P8-REGRESSION — the whole package suite after P8's additions
- **Command:** `pytest models/demos/llama31_8b_d_p/tests -q -p no:randomly`, with
  `PREFILL_TRACE_DIR` at the **1024**-token golden trace, `TT_CACHE_PATH` set and `HF_MODEL` staged.
- **Mesh / device:** every shape the suite uses — `(1,1)` for the P5-P7 gates, and the full `(4,8)`
  galaxy plus its submeshes for P8's.
- **Threshold:** 0 failed. Appendix A: "Add a **per-phase regression gate** as well: the whole
  package suite, 0 failed, after each phase's additions. It is cheap and it is the only thing that
  catches a new phase breaking an old gate's test file rather than its numbers."
- **Measured:** **196 passed, 0 failed**, 1283.76 s. P7 stood at 177, so P8 adds 19 tests:
  `G-KV-TP8` 5, `G-SP-RING` 4, `G-TP-PARITY` 5, `G-CHUNK-ATTN` 3 (including the script/test
  `meta_head_index` drift check), `G-SEMAPHORE` 1, `G-WEIGHTS` 1. `G-FABRIC-MATRIX` and `G-MESH-KV`
  are **scripts** and are not in this count — the same gap `DEC-069` records for `G-GOLDEN`, and it
  now covers three gates rather than one.
- **Verdict:** PASS
- **Deviations / notes, all three worth a reader's attention:**
  1. The suite was run twice. The first run (`raw/P8-REGRESSION_20260904T152156Z.log.gz`, also 196
     passed) had `black` applied to seven of its own files **while it was executing**, so its result
     belongs to the pre-format code. `DEC-090`.
  2. Running the suite with `PREFILL_TRACE_DIR` at the 1024-token trace **overwrites P7's**
     `raw/G-CHUNK_per_layer_pcc.json` with 1024-token content, while P7's ledger row cites a
     512-token measurement. Restored from the P7 commit; `DEC-091`, `R-041`.
  3. `raw/G-SP-RING-RECHECK_20260904T161926Z.log.gz` post-dates this run by 26 minutes and covers the
     one-line change in `DEC-093`. No full regression was run after it.

### G-CITE (P8) — the citation verifier, extended with a raw-artefact pass
- **Command:** `python models/demos/llama31_8b_d_p/scripts/verify_citations.py`
- **Threshold:** 0 mismatched, 0 unresolved (recipe §1.6, Appendix C item 7).
- **Measured:** **539/539** content-checked citations verified (CITES grew 488 -> 539: every
  load-bearing P8 ref — the fabric asserts, the ring op's two `TT_FATAL`s, the mesh descriptors, the
  templates, and nine recipe passages — was promoted into the content-checked list, because pass 2
  only range-checks, `R-016`). **897/897** doc refs resolved. **Pass 3, new in this phase: 98/100
  cited raw artefacts present.**
- **Verdict:** **PASS-WITH-DEVIATION** — the two missing artefacts both pre-date this phase, and
  neither was written by it.
- **What pass 3 is and why it exists.** Passes 1 and 2 both key on `path:line`, so a bare
  `` `G-FOO_<timestamp>.log` `` under `raw/` — which is exactly how this ledger cites its evidence — was scanned
  by **neither**. The recipe's own definition of a gate is that artefact ("A gate with no raw log did
  not happen", `BRINGUP_RECIPE.md:221`; Appendix C item 2), so a ledger row citing a file that is not
  there is a `PASS` with no evidence, and nothing was checking for it. Pass 3 checks it. It accepts
  `.log.gz` for `.log`, because oversized logs are gzipped losslessly (`R-040`).
- **What it found, on its first run:** two dangling references in the **P5.4-P5.6** status block
  above (`06_GATES.md:894`, `:896`):
  - `P5-REGRESSION_20260904T100738Z.log` — cited for "93 passed, 0 failed";
  - `G-CITE_20260904T101355Z.log` — cited for "330/330 verified ... 629/629 doc refs resolved".
  Neither file exists, and neither is in git history. The `raw/` directory holds a P5 regression pair
  at `T092212Z` / `T102256Z` and a `G-CITE` pair at `T092334Z` / `T102606Z`, i.e. runs ~15 minutes
  either side of the two cited timestamps — so the most likely history is a P5.4-P5.6 pass that was
  run, cited, then re-run, with only some of the citations updated.
- **Deliberately not "fixed".** The two claims are **not** re-attributed to the surviving logs,
  because this session cannot know that those logs carry the same numbers, and guessing which file
  an earlier session meant would produce exactly what §1.6 warns about: a citation that is wrong but
  reads as authoritative. The finding is recorded here and as `07_RISKS.md` **R-042**; the verifier
  stays red on those two lines until whoever owns that phase's record resolves it.
- **Negative control:** the pass's own discriminating power is demonstrated by the two failures — it
  found real dangling references on its first execution, on a ledger that four previous doc gates had
  passed clean. Separately, it caught **this session's own** first attempt at writing the finding up:
  quoting the two dangling names in the backticked `raw/`-relative form made them look like fresh
  citations and the pass flagged them, which is the correct behaviour — the citation form *means*
  "here is the evidence", so a name that is not evidence must not use it. The two names are written
  as bare basenames in this block and in `R-042` for that reason, and the P5 status block's own two
  citations were left exactly as they are, red.
- **Raw log:** `raw/G-CITE_20260904T162457Z.log` (exit 1, on the two `R-042` artefacts only).

### Note — two P5.4-P5.6 raw-log citations repointed (R-042)
P8's new artefact-existence pass found two cited logs absent:
the P5-REGRESSION log stamped 20260904T100738Z and the G-CITE log stamped 20260904T101355Z (named here without the citation form, so documenting their absence does not itself create two dangling citations). Four earlier document
gates passed over them because both existing passes key on `path:line`, and a bare backticked raw-log filename (written without backticks here, for the same reason)
matches neither.

Resolved by the orchestrator rather than left dangling. That session re-ran both gates after editing
files, and cited the **superseded first attempts**, whose logs were not retained; the authoritative
runs about 15 minutes later are `raw/P5-REGRESSION_20260904T102256Z.log` and `raw/G-CITE_20260904T102606Z.log`, and the rows now cite those. The P8 session deliberately declined to
guess, which was right — this note records the inference explicitly so a reader can reject it. What is
lost is only the transcript of two superseded attempts; the recorded numbers come from the surviving,
authoritative runs.

**Kit consequence:** re-running a gate must replace its ledger citation, not leave the first
attempt's. Timestamped filenames make the mismatch findable only if something checks that the file
exists — hence P8's third verifier pass.

## P10 — Disaggregated-prefill integration

### G-ADAPTER — the engine's adapter contract, and an import that stays cheap
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_prefill_adapter.py -q -p no:randomly`
- **Raw log:** `raw/G-ADAPTER_20260904T173636Z.log` (the same run as `G-RUNTIME`'s; the two files are
  collected together because P10 changed both and a split run would gate two different trees)
- **Mesh / device:** **none.** Appendix A gives this gate device "—", and it holds: the checklist is
  about the adapter's *contract*, and every device claim it could make belongs to `G-KV-TABLE`,
  `G-REQUEST` or `G-MOCK-MIG`. The one `PrefillRunParams`-driven call it makes (`build_runtime` with
  `mesh_device=None`) reaches only the three refusals, which fire before anything is touched.
- **Inputs / input distribution:** not applicable — no tensors. The inputs are `config.json`, the
  registry, the manifest JSON, and the engine's own source read with `ast` and `inspect`.
- **Reference dtype policy:** not applicable; no numbers are measured except a wall-clock import.
- **Threshold:** the checklist at `ADDING_A_PREFILL_MODEL.md:246-255`, item by item; **0** abstract
  methods left; `PREFILL_MODEL=llama31_8b_d_p` resolves through the registry; the registry-fed
  `variant` fixture picks it up; every `model_config` constant equals `config.json`; and the import
  is **measured** with no heavy module in `sys.modules` (`BRINGUP_RECIPE.md:1982-1985`).
- **Computed noise floor:** none applies. The one number with a threshold is the import time, whose
  budget is a *separator* rather than a measurement of anything: `DEC-101` states why 1.0 s, and the
  assertion that carries the claim is the `sys.modules` one.
- **Measured — 28/28 tests, 12.7 s, no mesh opened:**

  | checklist item (`ADDING_A_PREFILL_MODEL.md`) | evidence |
  |---|---|
  | every abstract method implemented (incl. `allocate_kv_cache`) — `:248` | `__abstractmethods__` empty; the engine's abstract set is exactly the expected four, and all four are defined **on this class** (not inherited) |
  | `name`, `model_config` and the default paths set — `:249` | `llama31_8b_d_p`; `Llama31_8BConfig`; `hf_model_default` is a real directory holding a real `config.json`; `ttnn_cache_default` and `prefill_trace_default` deliberately `""` |
  | `build_runtime` returns a §2 runtime, cache passed in — `:250-251` | it constructs `TtPrefillRuntime`; the class has all five doc-required names and **no** `owns_kv_cache` (`DEC-062`) |
  | no heavy imports at module load — `:252` | **40.3 ms**, heavy modules `[]` |
  | registered in `ADAPTER_PATHS` — `:253` | `adapter.py:291`, one line; `get_adapter` resolves and memoizes; `TEST_VARIANTS["llama31_8b_d_p"]` is an instance |
  | weight cache populated, golden trace staged — `:254` | the resolved cache dir holds `.tensorbin` files; the trace's `metadata.json` reports 8 KV heads, head_dim 128, fp32, 32 layers, and all 32 `layer_*.safetensors` exist |
  | request-mode producer PCC passes — `:255` | `G-MOCK-MIG` |

  Plus the recipe's four additions: **9/9** `model_config` constants equal `config.json`
  (`hidden_size`, `intermediate_size`, `num_hidden_layers`, `vocab_size`, `num_attention_heads`,
  `num_key_value_heads`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`), with `HEAD_DIM`
  checked against `derive_head_dim` **and** `config.json` asserted to carry no `head_dim` key;
  `FABRIC_PAYLOAD_SIZE == hidden_size`, which is the one value the engine itself reads
  (`runner_utils.py:41`); `ROTARY_DIM` absent and therefore defaulting to `HEAD_DIM`, which is what
  the producer's reader assumes (`prefill_producer.py:552`).

  Refusals asserted (4): `params.use_trace`, `params.dflash_enabled`, an unset `HF_MODEL`, and a
  `PREFILL_HF_MODEL` whose `config.json` disagrees with the bundled copy — the last one naming the
  differing key (`num_hidden_layers`) rather than failing generically. Also asserted: the adapter
  reads **only** `PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE`, `TT_CACHE_PATH` and `HF_MODEL` from the
  environment (an AST walk over every `.get`/`.getenv` literal), which is the mechanical form of
  "knobs come from `params`".
- **Verdict:** **PASS**
- **Negative controls (four, all discriminate):**
  1. **The import probe's own control.** The same cold subprocess importing the adapter *and*
     `tt/model_config.py` reports `torch` and `ttnn`. Without it, "no heavy module found" and "the
     probe looks in the wrong place" are the same observation (`R-016`'s shape).
  2. **A disagreeing checkpoint config** must be refused, and its byte-identical twin accepted —
     both halves run, so the check is not simply "always raise".
  3. **The weight-cache path must be mesh-specific:** `(1,8)` and `(4,8)` must differ, so a `(1,1)`
     cache cannot be picked up at TP=8 (the "one layer runs on garbage" failure, Appendix B).
  4. **The `getattr` census** must be empty except `max_seq_len`: a three-argument `getattr` on any
     *dimension* is the `R-005` trap, and the assertion names it.
- **Deviations:** `DEC-094` (bundled config + equality refusal instead of `AutoConfig`), `DEC-095`
  (this package's weight-cache layout, not the engine's convention), `DEC-096`
  (`kv_only_last_layer` ignored with a warning), `DEC-097` (`Topology.Linear` pinned in code),
  `DEC-098` (no cache-only build), `DEC-101` (the import budget), `DEC-102` (docstring-stripped
  source searches).
- **Notes.** Three of this gate's assertions failed on their **own prose** on the first run: the
  package's docstrings name `AutoConfig` and `get_num_devices` in order to explain why they are
  absent, and a substring search over `inspect.getsource` cannot tell a warning from an offence.
  `DEC-102` records the fix (`ast.unparse` of a docstring-stripped tree) and why the docstrings were
  not weakened instead. It is the same defect shape as the repo's own `prefer-expect-error` hook.

### G-REQUEST — request-mode serving through the engine, on the target mesh
- **Command (two processes, twice — the second at the real deployment geometry):**
  ```
  # terminal A — runner
  PREFILL_MANIFEST=models/demos/llama31_8b_d_p/tt/runners/manifests/llama31_8b_d_p.json \
  PREFILL_MODEL=llama31_8b_d_p PREFILL_SP=4 PREFILL_TP=8 PREFILL_NUM_LAYERS=32 \
  PREFILL_CHUNK_SIZE=256 PREFILL_MAX_SEQ_LEN=2816 PREFILL_NUM_USERS=1 \
  PREFILL_H2D_SERVICE_ID=llama_prefill \
    python -m models.demos.common.prefill.runners.prefill_runner
  # terminal B — producer
  <same shared env> PREFILL_PRODUCER_CHUNKS=11 PREFILL_SEND_SHUTDOWN=1 \
    python -m models.demos.common.prefill.runners.prefill_producer
  ```
  Arm 2 replaces `PREFILL_CHUNK_SIZE=8192 PREFILL_MAX_SEQ_LEN=131072 PREFILL_PRODUCER_CHUNKS=2`.
  The full matrix is `bringup_log/08_PREFILL_INTEGRATION.md` §2. Driven from one shell script so the
  hand-off is deterministic (`DEC-109`), which is the two terminals with a `grep` between them.
- **Raw logs:** `raw/G-REQUEST-runner_20260904T173723Z.log`,
  `raw/G-REQUEST-producer_20260904T173723Z.log`,
  `raw/G-REQUEST-DEPLOYMENT-runner_20260904T174126Z.log`,
  `raw/G-REQUEST-DEPLOYMENT-producer_20260904T174126Z.log`.
  Also retained: `raw/G-REQUEST-runner_20260904T173012Z.log`, the **failed** first attempt — it is
  `DEC-108`'s evidence and deleting it would erase the finding.
- **Mesh / device:** the deployment `(4,8)`, SP=4 x TP=8, `FABRIC_1D` (chosen by the engine for
  `sp <= 8`, `runner_utils.py:37`, and pinned to `1d` by the manifest), `Topology.Linear` in the
  runtime (`DEC-097`), `num_links=2` (`prefill_runner.py:489`).
- **Inputs / input distribution:** the 1024-token golden trace's own `token_ids`, tiled by the
  producer's pool to the pushed length (`prefill_producer.py:903-906` pads with token id 1 past the
  trace). Real bf16 checkpoint weights, loaded from the safetensors shards on every start
  (`DEC-098`).
- **Reference dtype policy:** not applicable — this gate scores no tensor. It is a serving-liveness
  gate; the numbers are `G-MOCK-MIG`'s.
- **Threshold:** every chunk accepted and served, the shutdown sentinel received, clean exit
  (`BRINGUP_RECIPE.md:1977-1978`).
- **Computed noise floor:** none — no numeric quantity is compared.
- **Measured:**

  | arm | chunk / cache | chunks | served | sentinel | producer rc | runner rc | device time |
  |---|---|---|---|---|---|---|---|
  | gate geometry | 256 / 2816 | 11 | `[0,256)` … `[2560,2816)`, all 11 | received after 11 chunks | **0** | **0** | 16.0 s (2816 tok) |
  | **deployment** | **8192 / 131072** | 2 | `[0,8192)`, `[8192,16384)` | received after 2 chunks | **0** | **0** | 13.6 s (16384 tok) |

  Producer side, arm 1: `pushes=11 requests=1 tokens=2816`, `push_ms p50=0.1 p90=206.4 p99=225.8`.
  Both runners logged `shutdown complete`.

  One number worth extracting from the runner log because it settles `DEC-095`: **290
  `Loading cache` lines and 0 `Generating cache` lines.** The adapter's `weight_cache_path` mirrors
  `ModelArgs`' layout rather than the engine's `{name}_{arch}_{N}dev/{sp}x{tp}` convention, and this
  is the measurement that says why — under the engine's convention all 290 tensors would have been
  re-tilized on the first start.
- **Verdict:** **PASS**
- **Negative control:** this gate's control is the run it already failed. The **first** attempt died
  on chunk 0 with `NotImplementedError: metadata_msg is the engine's trace-safe metadata tensor`
  (`DEC-108`) — so "every chunk served" is a discriminating claim rather than a formality, and the
  log that proves it is retained. Two structural controls sit alongside it: the driver asserts the
  runner reached `setup complete, entering request loop` before starting the producer (so a runner
  that died during weight load cannot look like a serving failure), and it fails the gate on either
  process's non-zero exit rather than on the producer's alone.
- **Deviations:** `DEC-106` (the geometry), `DEC-109` (one driver script instead of two terminals).
- **Notes — what arm 2 settles.** `R-039` recorded that "the deployment pair itself has never been
  run". It has now: chunk 8192 into a 131072-token cache, through the engine, on the real mesh. The
  other half of `R-039` stands — the `sp_bootstrap` core is still reachable only at
  `max_seq_len == chunk_size`, which this configuration never satisfies, so that core has a gate and
  no deployment use. Note also that `max_seq_len > chunk_size` **strictly** in both arms, which is
  the recipe's own warning (`BRINGUP_RECIPE.md:1995-1998`): at equality the SP bootstrap runs and
  "anything you measure is measuring the wrong path". The runner logs which it is only indirectly;
  `G-MESH-KV`'s harness asserts the core by name and this one inherits that geometry.

### G-MOCK-MIG — KV correctness and the chunk table, read device-lessly in another process
- **Command — two arms, and the second one exists because of `DEC-111`:**
  - **arm 1 (the doc's Gate 1 as written):** `PREFILL_MOCK_MIGRATION=1` and
    `PREFILL_ENABLE_LAYER_ACK=1` on the runner, `PREFILL_PRODUCER_CHECK_PCC=1` and
    `PREFILL_PRODUCER_CHUNKS=4` on the producer. This takes `prefill_runner.py:570` (and, thanks to
    `R-049`, `:699` as well), which passes **no** `stage_layout`.
  - **arm 2 (the stage-gather path):** the same plus `PREFILL_ENABLE_MIGRATION=1`. This takes
    `:626` (the real `allgather_kv_stage_layouts`), calls `kv_migration_base_address` at `:617`, and
    builds the table at `:644` with `stage_layout=stage_layouts[0]`. It needs **no** tt-llm-engine
    binaries — the worker handshake is only in the `else` branch at `:673` — and nothing in the doc
    says this arm exists. It is the branch `DEC-111`'s wrong `stage_layout` type would have crashed
    on, and arm 1 cannot reach it.
- **Raw logs:** `raw/G-MOCK-MIG-producer_20260904T173939Z.log`,
  `raw/G-MOCK-MIG-runner_20260904T173939Z.log.gz` (arm 1);
  `raw/G-MOCK-MIG-STAGED-producer_20260904T180914Z.log`,
  `raw/G-MOCK-MIG-STAGED-runner_20260904T180914Z.log.gz` (arm 2)
- **Mesh / device:** the deployment `(4,8)`, single rank. `PREFILL_MOCK_MIGRATION=1` is single-rank
  only by the engine's own check (`prefill_runner.py:691-697`).
- **Inputs / input distribution:** the golden trace's 1024 `token_ids`, pushed as **4 chunks of
  256**, real bf16 checkpoint weights. The read-back covers `[0, 1024)`, i.e. exactly the golden's
  extent — nothing is scored against padding.
- **Reference dtype policy:** the **fp32** golden trace, bit-identical to `LlamaModel`'s own loop
  (`G-GOLDEN`), with K permuted HF -> Meta *before* any quantiser (§2.2.3a). The device cache is
  `bfloat8_b` (`DEC-021`). The permutation is the producer's own
  (`prefill_producer.py:552-557`), and it is byte-identical to this package's `meta_head_index` for
  Llama because `ROTARY_DIM == HEAD_DIM`.
- **Threshold:** producer PCC >= **0.93** — `PREFILL_STANDALONE_CHUNKED_PCC`, **the engine's own
  default**, not ours (`BRINGUP_RECIPE.md`, Appendix A.2: "`0.93` is the disaggregated producer's own
  default ... and is the engine's number, not ours"). The measured per-layer minimum is recorded
  rather than just the pass.
- **Computed noise floor:** not recomputed here — it is `G-KV-TP8`'s layer-0 **complete** floor, the
  same producer and the same dtypes, which that gate measured against L0 K 0.9999617 (**1.10x**) and
  L0 V 0.9999369 (**1.02x**), i.e. a floor of ~0.9999652 on K and ~0.9999381 on V. Recomputing it in
  this process would mean rebuilding the whole fp32 reference chain outside pytest for no new
  information, and P8 set the same precedent for `G-MESH-KV`.
  **The ratio, at the precision available:** this gate's L0 is K **0.99996** / V **0.99994**, so the
  error ratios are **~1.15x** and **~0.97x** — layer 0 sits *at* its floor, from a completely
  different reader. The producer prints 5 decimals, which bounds those ratios at 1.01-1.29x (K) and
  0.89-1.05x (V); the honest statement is "1x within the printed precision", and it independently
  corroborates `G-KV-TP8`'s floor rather than assuming it. Depth, not the reader, is what moves the
  number: L22's K and L28's V match `G-MESH-KV` exactly (below).
- **Measured:**
  ```
  [producer] slot 0 llama31_8b_d_p packed-GQA KV PCC over [0,1024) across 32/32 local layers
             -> K=0.99678 V=0.98662 (min 0.986623)
  [producer] KV cache PCC PASSED (min 0.986623 >= 0.93 across 1 slots; per cache: k=0.996784, v=0.986623)
  [producer] kv_cache_pcc_complete slots_checked=1 min_pcc=0.986623 k_pcc=0.996784 v_pcc=0.986623
  ```
  Per-layer minima: **K 0.996784 at layer 22**, **V 0.986623 at layer 28**. Layer 0 K 0.99996 /
  V 0.99994, degrading monotonically-ish with depth as `G-CHUNK` and `G-MESH-KV` both found.
  LayerAck drain: **128/128** in 0.42 s = 32 layers x 4 chunks exactly.

  **Arm 2 reproduces every digit** — `min_pcc=0.986623 k_pcc=0.996784 v_pcc=0.986623`, drain
  `128/128` in 0.41 s — through the engine's stage-gather branch, whose runner log carries
  `[mock-migration] merged KV chunk table -> /tmp/llama_kv_chunk_table.pb (no migration worker)`
  from `_serve_request:651`, i.e. immediately after the `:644` build. That line is the evidence that
  `kv_migration_base_address` answered, `allgather_kv_stage_layouts` ran, and the gathered list
  reached `assert_single_rank_stage` and was accepted.
- **Verdict:** **PASS** — 0.986623 against 0.93 is **5.6x** the threshold's error budget on V.
- **The explicit comparison the recipe asks for** (`BRINGUP_RECIPE.md:1983-1985`: this gate "is the
  strongest evidence in the whole bring-up, because it is a second, device-less reader in a
  different process agreeing with the on-device `G-MESH-KV` number at the same shape. **Compare the
  two explicitly.**"):

  | | reader | process | read path | min K | argmin | min V | argmin |
  |---|---|---|---|---|---|---|---|
  | `G-MESH-KV` (P8) | `tests/galaxy_prefill_kv_pcc.py` | the runtime's own | `ttnn.to_torch` on `get_device_tensors`, un-block-cyclic'd host-side | **0.9967844** | L22 | **0.9866232** | L28 |
  | `G-MOCK-MIG` (P10) | `prefill_producer` | a **separate** one, no mesh open | `table.lookup` -> `read_dram_umd` -> bf8_b tile decode | **0.996784** | L22 | **0.986623** | L28 |

  The two agree **to every digit the producer prints (6 decimals) and on both argmin layers**. They
  share the golden trace and the model code and nothing else: different processes, different
  position -> address derivations (one re-derives the block-cyclic map in Python, the other walks the
  protobuf table the runtime published), different byte decoders. That is the strongest form the
  comparison can take, and it also means the **table** and the **cache** are independently right —
  a wrong table would have to be wrong in exactly the way that reproduces a correct read.
  One difference worth noting: `G-MESH-KV` ran with `max_seq_len == 1024` and this ran with
  **2816**, so the agreement additionally holds across a different cache capacity (and therefore a
  different rope table extent and a different DRAM bank sweep).
- **Negative controls (three):**
  1. **The read-back branch itself.** Without `llama31_8b_d_p` in `_PACKED_GQA_MODELS` the dispatcher
     falls through to the **MLA** reader (`prefill_producer.py:694`), which decodes a merged
     latent+rope row — plausible bytes, wrong ones. Measured discrimination: the reader's own log
     line now names `llama31_8b_d_p packed-GQA`, and had the branch been missing the gate would have
     scored a differently-shaped tensor. `DEC-104`.
  2. **The ack drain.** `128/128` is `num_layers x chunks`; the producer refuses to read at all
     without the channel (`prefill_producer.py:1065-1071`) precisely because an H2D push returning is
     not the layers being written. A runtime acking once per *chunk* instead of once per *layer*
     would hang the drain at 4/128 — so the count is a control on `set_layer_ack_channel`, and
     `G-RUNTIME` asserts the same property in isolation.
  3. **The address table** has its own five controls in `G-KV-TABLE`, which is the whole reason that
     gate exists: one PCC over one slot cannot separate a wrong table from a numerical problem.
  4. **Arm 2 against arm 1.** Two different engine branches, two different sets of arguments into
     `build_kv_chunk_table` (one with a gathered stage layout, one without), and the same PCC to
     every printed digit. That is a control on the *arguments*: a table built differently under the
     stage-gather path would not reproduce arm 1's number. It is also the control that was missing
     when `DEC-111` shipped — arm 1 alone cannot distinguish "the guard is right" from "the guard is
     never reached".
- **Deviations:** `DEC-106` (geometry, chosen so this comparison is an identity rather than an
  analogy); `DEC-111` (arm 2 added after the fact, with the defect it found).
- **Notes.** Two undocumented requirements cost this gate a run each and are recorded as `R-046`:
  the doc's hook table gives Gate 1 only `build_kv_chunk_table`, but the producer's PCC path also
  needs `set_layer_ack_channel`; and the Gate-1 binding the doc prints omits
  `PREFILL_ENABLE_LAYER_ACK=1`, which defaults to `0` on the mock path — so the documented
  configuration exits 1 with "LayerAck channel missing".

### G-KV-TABLE — the address table alone, bit-exactly
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_chunk_table.py -q -p no:randomly`
- **Raw log:** `raw/G-KV-TABLE_20260904T172909Z.log`
- **Mesh / device:** the full `(4,8)` galaxy, `FABRIC_1D`. Appendix A gives this gate "target mesh"
  and it must be: the table's entire content is the SP x TP geometry. Opened as the **full** mesh,
  never a top-level partial one (`BRINGUP_RECIPE.md:1708-1727`).
- **Inputs / input distribution:** a **labelled probe**, not random data. Each head's 128 lanes carry
  four 32-lane constant fields — `position % 128`, `position // 128 + 1`, `head + 1` (+16 for V),
  and `slot * num_layers + layer + 1` — so the bytes at any address fully determine
  `(position, head, slot, layer, K-or-V)`. 32 lanes is a multiple of `bfloat8_b`'s 16-element
  exponent block, so every block is homogeneous and the value survives the dtype exactly. Every
  label is **< 128**, `bfloat8_b`'s measured exact-integer ceiling — **not** the recipe's blanket
  256, which is the bf16 ceiling (`DEC-044`, `R-013`). Written through the real `write_kv_chunk`, in
  `SEQ_LEN // period` chunks with `kv_actual` advancing, so the cache holds the layout a chunked
  prefill of that period actually produces.
- **Reference dtype policy:** not applicable, and that is the point — **both comparands are device
  bytes**. The primary assertion compares the UMD read against `ttnn.to_torch` of the same live
  tensor, so no reference precision enters; the secondary one compares against the exact host labels,
  which are integers representable in both dtypes.
- **Threshold:** `torch.equal`, `rtol = atol = 0`, on every entry the table addresses; plus a control
  that reads one head through another's config, which must **fail** (`BRINGUP_RECIPE.md:2009-2014`;
  §2.5 — a rotated head-to-column mapping still scores PCC 0.99890, so PCC cannot be the
  discriminator).
- **Computed noise floor:** none, by construction. A bit-equality claim has no floor: the
  discriminator is exact, which is precisely why §2.5 requires it for a mapping claim.
- **Measured — 11/11 tests, 38.0 s:**

  | claim | measured |
  |---|---|
  | geometry | **16 configs** (`k_h0..7`, `v_h0..7`), **1024 entries** per period, **4352 B/chunk** = `(128/32) x 1088`, `chunk_n_tokens` 32, `num_layers` 2, `num_slots` 2, `max_sequence_length` 512 |
  | position -> address | **2048 chunks bit-identical** — 1024 at period 512 and 1024 at period 128 — against **both** the live device tensor and the host probe, `torch.equal` |
  | head -> config -> chip | every config's device group holds **exactly one** chip, and it is `MeshCoordinate(sp_row, head)` compared on `(mesh_id, chip_id)` — never on values |
  | K/V separation | configs `0..7` decode to K's labels, `8..15` to V's (the probe's head field differs by 16) |
  | two resolution paths agree | the device-map + `read_dram_umd` path (the producer's) and the table's own control-plane `read_device_chunk` return **identical bytes** on 8 sampled entries |
  | protobuf round trip | every address, size and device-group index preserved, **and** the config names come back as `00..15` — zero-padded, so `std::map` order equals numeric `config_id` order |
- **Verdict:** **PASS**
- **Negative controls (five, every one discriminates):** each reads a *confusable* neighbour of a
  correct lookup and must not return the same bytes.

  | control | what it confuses | `max|delta|` |
  |---|---|---|
  | `rotated_head` | head `c` through head `c+1`'s config — **the recipe's named control** | 1.0 |
  | `k_through_v` | K's chunk through V's config, same head | 16.0 |
  | `next_layer` | layer 0 through layer 1's row (the user-major packing's neighbour) | 1.0 |
  | `next_position` | one 32-token block along | 32.0 |
  | `next_slot` | slot 0 through slot 1 (the users share one tensor) | 2.0 |

  The deltas are small **on purpose** — they are label distances, not error magnitudes — and that is
  exactly why the gate is `torch.equal` and not PCC: `rotated_head` differs by 1.0 in one 32-lane
  field out of four, which is the perturbation §2.5 measured at PCC 0.99890.
  A sixth, structural control: the file's inverse block-cyclic map is checked against
  `tests/galaxy_prefill_kv_pcc.py`'s forward map at every one of the 512 positions and asserted
  injective, so a wrong inverse cannot make both sides of the live-cache comparison read the same
  wrong rows.
- **Deviations:** `DEC-099` (the address walk is imported from `gpt_oss_d_p`, not copied),
  `DEC-100` (serialized through `serialize_prebuilt_kv_chunk_table`, not the helper the recipe
  names — which cannot express 16 configs), `DEC-105` (function-scoped probe fixture).
- **Notes — a partial failure that would have read as a pass.** The first draft cached the probe in a
  **module**-scoped fixture. The repo's `mesh_device` fixture is function-scoped, so the mesh closes
  between tests and the cached tensors belonged to a closed mesh:
  `TT_FATAL @ tt_metal/distributed/mesh_device.cpp:845 ... cq_id 0 is out of range`. The tests that
  only read **addresses** passed anyway — a stale tensor's `buffer_address()` still returns a
  plausible number and the builder is pure host arithmetic — so a suite without the bit-exact arms
  would have gone green on a closed mesh. `DEC-105`.
  **What this gate does NOT prove:** that a real migration *worker* can use these addresses. It
  proves they are right and readable over the same UMD path the worker uses; the worker itself is
  `G-LOOPBACK`'s, which is out of scope (`DEC-103`, `R-043`).

### G-RUNTIME (P10 extension) — the migration hooks, and the refusal that was wrong
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_prefill_runtime_chunked.py -q -p no:randomly`
- **Raw log:** `raw/G-RUNTIME_20260904T173636Z.log`
- **Mesh / device:** **none**, as in P7. The three P10 hooks are reached with the same
  `object.__new__` instance carrying only the attributes they read (`DEC-068`), and
  `assert_single_rank_stage` is pure host arithmetic.
- **Inputs:** the engine's own source, parsed with `ast` — **not** the contract doc.
- **Reference dtype policy / noise floor:** not applicable; nothing numeric is measured.
- **Threshold (extended):** everything P7's row required, **plus** — new, and the reason this
  extension exists — **no refusal on a parameter the engine unconditionally passes**.
- **Measured:** **48/48** tests, 22.0 s with `G-ADAPTER` in the same run. P7 stood at 37; P10 adds 12
  and removes 2 (both of the removed ones asserted a **wrong** contract — see below).
  - **Removed:** the `metadata_msg` refusal test. That refusal was **wrong**, and P7's own audit
    passed it clean (`DEC-108`).
  - **`set_layer_ack_channel`:** refuses before `compile()` (acking the warm-up chunks would put
    `num_layers` phantom acks per warmed size into the channel and finish the producer's drain a
    chunk early); and the registered callback injects **exactly `num_layers` per chunk** — 32, which
    is what makes `G-MOCK-MIG`'s `128/128 = 32 x 4` drain the arithmetic it is.
  - **`build_kv_chunk_table`'s multi-rank refusals — rewritten by `DEC-111`, having been wrong in
    both directions.** 6 refused shapes (`first_layer_idx=8`; `num_my_layers=16` against a
    32-layer runtime; a gathered list carrying **2 ranks**; a single stage covering 16 of 32 layers;
    a bare `dict`, which is what the first draft believed the engine passed; and an empty list) and
    3 accepted ones — the exact shapes the engine passes at `prefill_runner.py:570` (`path` only)
    and `:644` (`first_layer_idx=0`, `num_my_layers=32`, and `stage_layout` as the gathered **list**
    with one entry). The template `del`s all three arguments
    (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:388`).
  - **The type belief itself is now tested**, not assumed: `test_the_gathered_stage_layout_really_is_a_list_of_dicts`
    AST-walks `allgather_kv_stage_layout` and requires it to build a list it appends dicts to. That
    is the check that would have caught `DEC-111` without a device.
  - **`DEC-112`:** `build_kv_chunk_table` refuses more than one supported chunk size, because a
    table describes exactly one block-cyclic period and a cache written at two has no single
    address map — the only silent-wrong-answer path in the module.
  - **`kv_migration_stages` asserted absent** and `kv_migration_base_address` asserted present, so
    the engine's branch selection (`prefill_runner.py:613`) cannot flip by accident (`DEC-107`).
  - **`kv_migration_base_address`** returns K's `buffer_address()`, read off the cache the engine
    passed in — checked with both the bare cache and a one-element sequence.
  - Refusal census: **22** `raise` statements in the runtime **+ 7** in
    `tt/runners/kv_chunk_table.py`, counted with `ast` so a deleted refusal fails the count rather
    than quietly losing coverage. (P10 turned three of the runtime's `raise` bodies into
    implementations and added `DEC-112`'s; the table module's seven are `DEC-099`'s layout guard and
    `DEC-111`'s six multi-rank refusals.)
- **Verdict:** **PASS**
- **Negative controls (four; P7's three plus one new):**
  1. `_BrokenRuntime`, the doc-faithful runtime, still reported on three counts.
  2. The audit's precondition: the walk must find a `prefill_chunk` call, a `compile` call, a
     `config` access and a guarded hook.
  3. The refusal census.
  4. **New:** `test_every_parameter_the_engine_always_passes_is_accepted_or_used` reads the engine's
     `prefill_chunk` keyword set out of the AST walk (asserting it is exactly the expected six, so a
     changed engine fails loudly) and then requires the runtime's **body** — docstring excluded —
     to contain no `<param> is not None` refusal for any of them **except** `d2h_service`, whose
     exemption is justified in the assertion message. This is the control that would have caught
     `DEC-108` statically.
  5. **New, and it is `DEC-111`'s:** every refusal now has a **positive** counterpart asserting the
     engine's real value is *accepted*. `test_build_kv_chunk_table_accepts_the_shapes_the_engine_really_passes`
     is the one that was missing — the first version's "accepted" test passed a bare dict, i.e. the
     same wrong belief the code held, so the pair could not falsify each other.
- **Deviations:** `DEC-108` (a refusal that had to be removed), `DEC-111` (a refusal that was wrong
  in both directions and a test that enshrined it), `DEC-112` (the period guard), plus P7's
  `DEC-062`, `DEC-063`, `DEC-067`, `DEC-068` — with `DEC-063` now only half-standing: three of its
  six hooks are implemented.
- **What this extension does NOT prove, and it is the lesson worth carrying.** A static audit shows
  that the runtime's signature **binds** the engine's call. It cannot show that the *values* the
  engine binds are acceptable, because it never runs the body. `metadata_msg` bound fine and then
  raised, and the cost of finding out was a mesh open, a 15 GB weight load, a `compile()` and a
  served chunk — the exact expense `G-RUNTIME` exists to avoid, reached from the other side. Control
  4 closes this particular hole; the general point is that a gate whose device column is "none"
  cannot be the last word on a runtime, and `G-REQUEST` is what actually proves it serves.

  **And `DEC-111` sharpens it further: a gate is only as good as the branches it takes.** Every P10
  gate passed with a `stage_layout` guard that would have rejected every real migration run and
  provided none of the multi-rank protection it advertised, because the only arm anyone had run was
  the one arm that passes no `stage_layout` at all. Two things followed: `G-MOCK-MIG` grew arm 2
  (the stage-gather path, which needs no extra binaries and which nothing in the engine's docs
  mentions), and this file grew the rule that **every refusal needs a positive test on the engine's
  real value**, not just a negative one on a plausible wrong value.

### G-LOOPBACK — the real migration copy: **OUT OF SCOPE**
- **Command:** none run.
- **Mesh / device:** would be the target mesh **plus** the tt-llm-engine `migration_endpoint` and two
  `migration_worker` processes.
- **Threshold (unchanged, for whoever runs it):** `--verify-migration dst-bytes` — every destination
  chunk byte-identical to its source.
- **Verdict:** **OUT OF SCOPE** by `DEC-103`, with the residual gap enumerated as `07_RISKS.md`
  **R-043**. Not `BLOCKED`: `HUMAN GATE H4` asks whose bug a red gate would be, and the answer here
  is the engine's — the doc itself says the gate "verifies the *engine's* model-agnostic byte copy,
  not this model", and its hook table gives `dst-bytes` **no** model-specific surface
  (`PREFILL_MIGRATION_TESTING.md:544`).
- **What stands in its place:** `G-KV-TABLE` proves the addresses the copy would read, **bit-exactly**,
  over the same `read_dram_umd` path the worker uses, and `G-MOCK-MIG` proves the bytes at those
  addresses are the right KV. **`G-MOCK-MIG` arm 2 then took most of the runner's real migration
  path as well** — `allgather_kv_stage_layouts` (`prefill_runner.py:626`),
  `kv_migration_base_address` (`:617`) and the merged-table build (`:644`) — because
  `PREFILL_ENABLE_MIGRATION=1` with `PREFILL_MOCK_MIGRATION=1` needs no worker binaries, the
  handshake being only in the `else` branch at `:673`. Nothing in the engine's documents mentions
  that arm, and adding it is what found `DEC-111`. What is left unproven is **the transport itself**:
  `publish_serialized_table_and_wait_ready`, the two worker processes, and the destination
  read-back — itemised in `R-043`.
- **Not faked.** `BRINGUP_RECIPE.md:1973` says "Do not fake it", and nothing here simulates a copy.

### P10-REGRESSION — the whole package suite after P10's additions
- **Command:** `pytest models/demos/llama31_8b_d_p/tests -q -p no:randomly`, with
  `PREFILL_TRACE_DIR` at the **1024**-token golden trace, `TT_CACHE_PATH` and `HF_MODEL` set, and
  `PREFILL_MODEL=llama31_8b_d_p` (which `tests/unit/test_kv_chunk_table.py` would otherwise
  `setdefault` itself — set explicitly so the producer import in that file cannot resolve another
  model's adapter).
- **Mesh / device:** every shape the suite uses — `(1,1)` for P5-P7, submeshes and the full `(4,8)`
  galaxy for P8 and P10.
- **Threshold:** 0 failed (Appendix A's per-phase regression gate).
- **Measured:** **246 passed, 0 failed**, 1420.59 s (23:40). P8 stood at 196, so P10 adds **50**
  tests: `G-ADAPTER` **28**, `G-KV-TABLE` **11**, and `G-RUNTIME` **+11** (37 -> 48). `G-REQUEST` and
  `G-MOCK-MIG` are two-process transcripts, not pytest, so they are not in this count — the same gap
  `DEC-069` records for `G-GOLDEN`, `G-FABRIC-MATRIX` and `G-MESH-KV`, and it now covers five gates.
- **Verdict:** **PASS**
- **Notes, three of them worth a reader's attention:**
  1. **This is the second run.** The first (started 17:49) was **killed at ~67 tests**, because the
     independent review that found `DEC-111` landed while it was executing and its result would have
     belonged to pre-fix code — the same reasoning `DEC-090` applied in P8, applied earlier this
     time. Its partial log was deleted rather than kept: an incomplete regression is not evidence,
     and a truncated log in `raw/` invites exactly the mis-citation `R-042` is about.
  2. **`R-041` bit again, in the same place.** Running with `PREFILL_TRACE_DIR` at the 1024-token
     trace rewrote `raw/G-CHUNK_per_layer_pcc.json` with 1024-token content while P7's ledger row
     cites a 512-token measurement (`seq_len` 512 -> 1024, 210 lines). Backed up before the run and
     restored after, and the restored file is byte-identical to P7's (`md5 2e89817a…`). The other
     three per-layer JSONs the run touched (`G-CHUNK-ATTN`, `G-KV-TP8`, `G-MODEL`) differ from the
     committed copies by **one line each** — a trailing newline the `end-of-file-fixer` hook added,
     with no content change, verified key-by-key.
  3. Run on the final tree, `-p no:randomly`, after every formatting hook, with
     `PREFILL_MODEL=llama31_8b_d_p` exported so `tests/unit/test_kv_chunk_table.py`'s
     `setdefault` before the producer import cannot resolve another model's adapter.

### G-CITE (P10) — every `path:line` and every cited raw artefact resolves
- **Command:** `python models/demos/llama31_8b_d_p/scripts/verify_citations.py`
- **Threshold:** 0 mismatched, 0 unresolved, 0 missing artefacts (recipe §1.6, Appendix C item 7).
- **Measured:** **604/604** content-checked citations (`CITES` grew 539 -> 604: 65 new rows, every one
  of them a reference into the engine, its two contract documents, the imported table builder or the
  recipe's own P10 section — the boundary this phase is entirely about, and pass 2 only
  range-checks it). **1169/1169** doc refs resolved. **128/128** cited raw artefacts present.
- **Verdict:** **PASS** — and it is the first fully clean run of all three passes: P8 finished
  `PASS-WITH-DEVIATION` on two dangling P5 artefacts (`R-042`), which the orchestrator has since
  repointed to the surviving authoritative logs.
- **What it caught this phase, and both were real:**
  1. **13 of this phase's own citations were wrong** on the first run — every one a line number
     estimated before an edit shifted it, exactly the failure mode `§1.6` describes. Each was
     corrected from the verifier's own "needle actually on lines" report, and the same corrections
     were applied to the prose refs that quote the same numbers (which pass 2 would have reported
     `resolved`, because they were wrong **but in range** — `R-016`).
  2. **A P7 citation that this phase's own edit invalidated.** `DEC-104` renamed the producer's
     packed-GQA reader and inserted `_PACKED_GQA_MODELS` above the dispatcher, which moved
     `_read_slot_kv_and_check_pcc_mla` from line 511 to 694 — and a P7 `CITES` row pointed at 511.
     Because that row is *content*-checked it failed loudly; had it been prose it would have
     resolved in range and stayed silently wrong. The row now cites 694 with a comment naming the
     cause.
- **Negative control:** the two findings above are the control — the verifier failed on this
  session's own work before it passed, on a tree that a formatting pass and a full regression had
  already been run against. Plus `runner_utils.py` was cited by **basename** in two places and came
  back `unresolved` until a full-path `CITES` row existed for it: the resolver builds its
  basename index from `CITES` plus doc refs, so an abbreviated reference to a file nothing cites in
  full cannot resolve. That is the right behaviour and it is why the two rows were added
  content-checked rather than the refs spelled out.

```
STATUS after P10: gates PASS=28 FAIL=0 DEVIATION=2 OUT-OF-SCOPE=1 BLOCKED=0 | next: P9 (cleanliness)
Open DECs needing review: DEC-095 (the weight-cache path carries no model name, so two models under
one PREFILL_TTNN_CACHE would collide), DEC-096 (PREFILL_KV_ONLY_LAST_LAYER ignored with a warning
rather than refused), DEC-097 (Topology.Linear pinned in code because this galaxy has no ring
fabric — a torus machine needs an edit, R-031), DEC-098 (no cache-only build path — measured at 46 ms, not the tens of
seconds the decision assumed, because safetensors mmaps and the populated cache never touches it), DEC-099 (the address walk is imported from
models/demos/gpt_oss_d_p, a cross-package dependency — R-045), DEC-103 (G-LOOPBACK scoped out —
R-043), DEC-104 (shared engine code changed: the producer's read-back branch)
```

`PASS=28` is P8's 24 plus `G-ADAPTER`, `G-REQUEST`, `G-MOCK-MIG` and `G-KV-TABLE`.
`OUT-OF-SCOPE=1` is `G-LOOPBACK` (`DEC-103`), which Appendix C item 3 permits explicitly and which
`R-043` enumerates. `DEVIATION=2` is unchanged from P8 (`G-ATTN`, `G-FABRIC-MATRIX`). `BLOCKED=0`.
The two cross-cutting rows sit outside the Appendix A tally, as in every earlier phase:
`P10-REGRESSION` and `G-CITE (P10)`.

**Every gate Appendix A assigns to P10 has a recorded number and a raw log**, and the three that
produce numbers carry an input distribution, a reference dtype policy, a floor (or a stated reason
there is none) and a negative control. **P9 (cleanliness) is next, and it is the last phase.**

Six things P9 should read before starting:

1. **`DEC-108` and `DEC-111` are the phase's most important findings, and both are about method.**
   Three times, a **refusal** was written from a parameter's *name* rather than from what the engine
   actually puts in it — `metadata_msg` (always non-`None`; found by a served chunk) and
   `stage_layout` (a list, not a dict; found by an independent review *after* every P10 gate had
   passed). `G-RUNTIME`'s static audit passed all three, and in `DEC-111`'s case its own tests
   asserted the wrong contract, so code and test could not falsify each other. Two rules came out of
   it, and P9 should apply them to anything it touches: **every refusal needs a positive test on the
   engine's real value**, and **a gate is only as good as the branches it takes** — the guard that
   would have blocked every real migration run survived because no arm had taken that branch.
2. **Three env-var reads are new**, all of them the engine's own variables rather than invented
   ones: `PREFILL_HF_MODEL` and `PREFILL_TTNN_CACHE` (`tt/runners/adapters/llama.py`) and
   `PREFILL_MODEL` (`tests/unit/test_kv_chunk_table.py`, a `setdefault` before the producer import).
   P9 item 6's grep will also surface `HF_MODEL`, `TT_CACHE_PATH` and `PREFILL_TRACE_DIR`, which
   pre-date this phase. The package still invents **no** `PREFILL_*` variable of its own: the
   topology that would have needed one is pinned in code (`DEC-097`).
3. **The README's status table** should carry `G-MOCK-MIG`'s numbers next to `G-MESH-KV`'s, because
   the two agreeing to 6 decimals from different processes is the strongest single line in the
   package's evidence.
4. **P9 item 8 wants the import cost "measured against the template as a ratio".** `G-ADAPTER`
   measures it absolutely (40.3 ms, 0 heavy modules, with a control). The ratio against
   `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py` is still to do — note that the template
   imports `models.common.utility_functions` and a reference config at module scope, so the ratio
   will favour this one.
5. **P9 item 9 (every `tt/` module owns a test).** P10 adds three modules:
   `tt/runners/adapters/llama.py` -> `tests/unit/test_prefill_adapter.py`,
   `tt/runners/kv_chunk_table.py` -> `tests/unit/test_kv_chunk_table.py` **and**
   `tests/unit/test_prefill_runtime_chunked.py` (the multi-rank refusals), and the two
   `__init__.py`s, which own nothing by convention. No gap opened.
6. **`G-MOCK-MIG` has two arms now, and arm 2 needs no extra binaries.**
   `PREFILL_ENABLE_MIGRATION=1` + `PREFILL_MOCK_MIGRATION=1` drives the engine's real stage-gather
   branch (`prefill_runner.py:626`, `:644`) and `kv_migration_base_address`. No document mentions
   it. If P9 re-runs one numerical gate (item 11), this is the one worth the device time.
7. **The delivered tree matches `03_OUTLINE.md`'s tree exactly — 57 files — but the outline's own
   prose says "41 tracked files at the end of P10" (`03_OUTLINE.md:102`).** Counting the file lines
   in its tree gives 57, and `find` over the package (excluding `bringup_log/`, `generated/` and
   `__pycache__`) gives 57. So P10 delivered the contracted tree file-for-file and the figure in the
   prose is an arithmetic error, not a deviation. P9 should correct the sentence rather than the
   tree. P10's own additions are the five the outline lists under `tt/runners/` plus
   `tests/unit/test_prefill_adapter.py` and `tests/unit/test_kv_chunk_table.py` — seven, all of them
   contracted.
8. **`R-041` bit again, in the same place.** Running the suite with `PREFILL_TRACE_DIR` at the
   1024-token trace overwrites `raw/G-CHUNK_per_layer_pcc.json`, whose P7 ledger row cites a
   512-token measurement. It was backed up before this run and restored after, exactly as `DEC-091`
   did in P8 — which means the workaround has now been needed twice and the kit defect is real.

### Note — the kit's recipe was edited **out of band while this session was live** (`R-017` again)
Between this phase's last clean citation pass and its final one, `BRINGUP_RECIPE.md` grew **23
lines** (a new passage after `:1599` about the very defects `DEC-108` and `DEC-111` record) and
`LANDMINES.md` grew a row. Nothing in this package changed. The effect was immediate and mechanical:
**38 of this phase's content-checked citations into the recipe went red at once**, every one of them
shifted by exactly +23.

They were re-pointed mechanically — every `BRINGUP_RECIPE.md:N` with `N >= 1600`, in `CITES` (104
rows rewritten) and in prose (26 files), plus the range forms — and the final pass is clean:
**604/604 · 1169/1169 · 128/128**. Raw logs were **excluded** from the rewrite: they record what ran
(`BRINGUP_RECIPE.md` §0.2 rule 4).

**The rewrite reached beyond this phase's own files**, and that is worth flagging for a reviewer of
the diff: `README.md`, `03_OUTLINE.md`, `04_CCL_PLAN.md` and fourteen P5-P8 test files all carry
recipe references past `:1600` and all were shifted. Nothing about those files' *content* changed —
only the line number each reference points at — and no `tt/` module was touched by it (the only
`tt/` file this phase modified is `tt_prefill_runtime.py`, for the migration hooks it owns). The
alternative was to leave earlier phases' citations wrong-but-in-range, which is precisely what
`R-016` says is worse than no citation at all.

Three things worth stating rather than absorbing:

1. **This is `R-017` recurring, and it is now the second time.** P6 recorded it when the recipe grew
   1986 -> 2017 lines after P5 was gated. The register's mitigation — "promote the load-bearing refs
   into `CITES`" — is what made this instance *findable in seconds* instead of invisible: pass 1
   content-checks, so all 38 failed loudly and each reported the line its needle had moved to. The
   prose refs that pass 2 only range-checks would have stayed silently wrong, which is exactly
   `R-016`. The mitigation worked; the underlying hazard did not go away.
2. **§0.2's rule is about the worktree and should be about the *recipe* too.** "Never rename, move,
   or restructure while a session is live" is written for the package. An edit to the specification
   a live phase is citing has the same effect and is not covered: it invalidated 38 references
   without touching a single file the phase owns.
3. **The `LANDMINES.md` addition is malformed.** It was appended after a blank line following the
   "Repo hooks that will block your commit" table, with **two** cells where that table has three
   (`Hook | What it rejects | What to write instead`). It therefore renders as its own separate
   two-column table rather than a row of the one above it — and its content is a *method* trap, so
   the "Method traps" table immediately below is where it belongs. Recorded, not fixed: the kit is
   not this session's to edit.

**STOPPED HERE, ON A GATE BOUNDARY.** Every gate Appendix A assigns to P10 has run, with a recorded
number and a raw log: `G-ADAPTER`, `G-REQUEST` (two arms), `G-MOCK-MIG` (two arms), `G-KV-TABLE`, and
`G-LOOPBACK` scoped out by `DEC-103` with `R-043` enumerating what that costs. The regression is
**246 passed, 0 failed** and the citation verifier is **604/604 · 1169/1169 · 128/128**.
**P9 (cleanliness) is next, and it is the last phase.**

---

## Phase P9 — cleanliness

| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| G-CLEAN | P9 | the whole-package cleanliness sweep, eleven items | all eleven pass, each with its command and output recorded | **11/11 pass**, and the sweep found six real defects (five in the logs, one in the code) — the largest being **247 of 317 prose recipe citations pointing at unrelated text while the verifier reported them `resolved`**. Items: (1) `pre-commit` clean on all 176 branch files; (2) SPDX pair on **54/54** `.py`, JSON + log markdown exempt with reasons (`DEC-123`); (3) **0** TODO/FIXME/XXX/HACK in package source; (4) **9** `except` handlers, none a silent `pass`, every one off the correctness path and logging; (5) **0** `print` in `tt/`, one stale docstring found and fixed; (6) **16** env vars by AST scan, 16/16 in the README (`DEC-122`); (7) `README.md` complete, all eight required sections (`DEC-124`); (8) adapter import **0.0155×** the template's — 37.8 ms vs 2441.6 ms, **64.6× cheaper**, asserted in-suite (`DEC-121`); (9) **21/21** `tt/` modules own a test, 0 gaps; (10) citations **629/629 · 1225/1225 · 146/146** + a new recipe-fingerprint pass (`DEC-120`); (11) regression **247 passed, 0 failed**, and `G-MOCK-MIG` arm 2 reproduced **every digit** | PASS | 2026-09-04 | `raw/G-CLEAN-item1_20260904T185713Z.log`, `raw/G-CLEAN-item2_20260904T192142Z.log`, `raw/G-CLEAN-item3_20260904T192142Z.log`, `raw/G-CLEAN-item4_20260904T192142Z.log`, `raw/G-CLEAN-item5_20260904T192142Z.log`, `raw/G-CLEAN-item6_20260904T192142Z.log`, `raw/G-CLEAN-item6_20260904T193831Z.log`, `raw/G-CLEAN-item8_20260904T191444Z.log`, `raw/G-CLEAN-item9_20260904T192142Z.log`, `raw/G-CLEAN-item10_20260904T195811Z.log`, `raw/P9-REGRESSION_20260904T193410Z.log.gz`, `raw/G-MOCK-MIG-P9RERUN-producer_20260904T193238Z.log`, `raw/G-MOCK-MIG-P9RERUN-runner_20260904T193238Z.log.gz`, `raw/G-RMS-P9RERUN_20260904T195714Z.log` |

### G-CLEAN — the eleven-item cleanliness sweep
- **Command:** eleven, one per item; each teed into `bringup_log/raw/G-CLEAN-item<N>_<stamp>.log`.
- **Mesh / device:** none for items 1-10. Item 11 uses every shape the suite uses, plus the full
  `(4,8)` galaxy for the `G-MOCK-MIG` arm-2 re-run.
- **Threshold:** all eleven items pass (`BRINGUP_RECIPE.md:2043`).
- **Verdict:** **PASS**
- **Human gates:** `H6` (residual risk) and `H7` (upstream fixes) both **ran on their defaults** —
  no human was available — and are recorded as `R-056` and `R-057` with the questions verbatim, as
  `BRINGUP_RECIPE.md:198-200` requires. Nothing in this phase is a sign-off, and 20 external defects
  are logged with 0 filed.

#### Item 1 — `pre-commit` on every file this branch touches
`pre-commit run --files $(git diff --name-only main...HEAD)` over **176** files: every hook Passed or
Skipped, **0** Failed, on the first run. Re-run after each of P9's own edits; the only failure at any
point was `trailing-whitespace` on a raw log P9 itself had just written (the column-aligned item-9
inventory), fixed by the hook and re-run clean — which is exactly the sequence `LANDMINES.md`'s
"Repo hooks" table predicts. Hooks were run **before** the two device runs, per the same table.

#### Item 2 — the SPDX header pair
**54/54** `.py` files carry both lines inside their first five (the shebang in the two `scripts/*.py`
pushes them to lines 2 and 4). `README.md` carries them. **Exempt, with reasons** (`DEC-123`):
`configs/Llama-3.1-8B-Instruct/config.json` must stay byte-identical to the checkpoint's
(`md5 3cd5831d379b509d53afade0e24c36e9`, asserted by a test — a header would break the gate that
proves the model identity); `tt/runners/manifests/llama31_8b_d_p.json` because JSON has no comment
syntax, as `models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_d_p.json` also shows; and
`bringup_log/*.md` because the kit's own five templates and `models/demos/minimax_m3/README.md`
carry none either. **Item 2 as written — "every new file" — is unsatisfiable for a bundled JSON**,
which is a recipe defect rather than a package one.

#### Item 3 — TODO / FIXME / XXX / HACK
**0** in this package's own source. The seven grep hits are all *references* to an upstream marker in
another package — `models/demos/gpt_oss_d_p/tt/ccl.py:134`'s `reset_global_semaphores` TODO — in
prose or in a content-checked `CITES` row (`DEC-028` owns the decision to live with it). Nothing here
needs a filed issue.

#### Item 4 — `except` handlers
**9** handlers in the package, listed with a per-site justification in the raw log. **None** is a
bare `except: pass` (`grep -rn 'except.*:\s*pass'` → no matches) and every one logs, prints or
records the failure it caught. The only one on a device path is `tt/layer.py:76`, the
`LLAMA_DELTA_PROBE` bring-up probe — the single case recipe §0 rule 5 allows, it logs a warning
rather than passing silently, its handler is exercised deliberately by
`tests/unit/test_decoder_layer_vs_ref.py:536`, and it computes nothing the model consumes. Four of
the remaining eight are *the measurement*: `tests/fabric_topology_matrix.py:432`/`:491` turn a child
error or a hang into a recorded `RESULT` line, and `tests/unit/test_dense_sp_vs_ref.py:349`/`:556`
capture the ring op's `TT_FATAL` text and then **assert the refusal happened**.

#### Item 5 — dead code, commented-out experiments, leftover `print`
Unused imports and variables are enforced by the `autoflake` hook (item 1, Passed). **No `print` in
`tt/`**; the 83 elsewhere are all in the three standalone scripts and the two `tests/` harnesses whose
stdout *is* the gate transcript, two of which (`verify_citations.py`, `verify_golden_kv.py`) must not
depend on `loguru` at all — `verify_golden_kv.py` imports no `ttnn` by gate contract. Eight of the 21
`tt/` modules use `loguru`; none prints. No commented-out code: the largest comment block in the
package (`tests/test_factory.py:205-227`) is the measured argument for `FABRIC_1D` + `Topology.Ring`
on this galaxy, not a disabled experiment.

**One real defect, found and fixed.** `tests/test_factory.py`'s `prefill_topology()` docstring said
"`Ring` (deployment default)" while the `os.getenv` two lines above it defaults to `"linear"`.
`DEC-079`/`DEC-097` pinned `Linear` in P8 because this galaxy has no ring fabric; the docstring
pre-dated that and was never updated, so the file contradicted itself and the README. Reworded to
name `Linear` as the default and say why. It is the only stale statement the sweep found in the code
— which is worth stating, because it found five in the logs.

#### Item 6 — the env-var table
**16** distinct variables over **31** read sites, **0 unresolved**, all 16 in the README's table.
Generated by an AST walk, not a grep, and the walk had to be written twice (`DEC-122`): the first
version keyed on a literal first argument, reported **15**, and missed `LLAMA_DELTA_PROBE` — the
package's own variable — because `tt/layer.py:54` reads it through a module constant. The second
version resolves module-level string constants and **reports unresolved name expressions loudly**
rather than returning a shorter answer. The recipe's warning that a hand list "misses the ones that
matter" is right and incomplete: an automated list misses the same variable for a different reason.
Three variables are new since P8 (`PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE`, `PREFILL_MODEL`), all
three the engine's own; the package still invents **no** `PREFILL_*` variable of its own.

The scan is logged **twice**, and the reason is a small instance of `R-041`'s lesson: the first
run predates item 5's docstring fix, which added five lines to `tests/test_factory.py` and moved
two of the reads (`PREFILL_FABRIC` 228 -> 233, `TT_MESH_GRAPH_DESC_PATH` 253 -> 258). The first log
records what ran then and is kept unaltered; the re-run is the evidence for the README's table,
which has to describe the delivered tree.

#### Item 7 — `README.md`
All eight required sections present (`BRINGUP_RECIPE.md:2030-2032`): architecture table, deployment
path with the `TP == num_key_value_heads` arithmetic, status table with measured PCC, run commands,
env-var table, layout, the "why not `models/common/`" answer with both its citations, and "what is
not implemented". P8 left it a deliberate stub; this is the full file (`DEC-124`).

Two things it now carries that the ledger alone could not:
1. **`G-MOCK-MIG`'s numbers sit in the status table beside `G-MESH-KV`'s**, in the same table, with
   the agreement stated — 0.996784 / 0.986623 device-lessly in a second process against
   0.9967844 / 0.9866232 on device, and **the same argmin layers** (K L22, V L28). Two readers, two
   processes, two position→address derivations. It is the package's strongest single line and it was
   previously visible only to someone who read `06_GATES.md`.
2. **A "known-imperfect in the record itself" section**, so a reader learns about `R-053`'s citation
   defect from the README rather than from the risk register.

#### Item 8 — adapter import cost, as a ratio
Five cold subprocesses per module. `models.demos.llama31_8b_d_p.tt.runners.adapters.llama`:
**37.7-38.0 ms**, heavy modules `[]`, 210 entries in `sys.modules`. The template it mirrors,
`models.demos.gpt_oss_d_p.tt.runners.adapters.gpt_oss`: **2445-2466 ms**, heavy modules
`['torch', 'ttnn']`, 3273 entries. **Ratio 0.0155 — the template is 64.6× more expensive.**

The cost is attributable to **one line**: `gpt_oss.py:25` imports `models.common.utility_functions`,
which alone measures **2487.9 ms** and pulls `torch` and `ttnn`. The template's other module-scope
import, `models/demos/deepseek_v3_d_p/reference/gpt_oss_120b_config.py`, costs **0.6 ms** and nothing
heavy — worth recording because the P10 hand-over named both as the cause and only one is. So the
reference adapter breaks the import-lightness `ADDING_A_PREFILL_MODEL.md:252` requires of it, which
is `R-055` and belongs to `HUMAN GATE H7`.

Asserted in-suite (`DEC-121`) rather than measured once, with a 4× margin on a 64.6× difference, and
the second assertion is the more important one: the template **must** report heavy modules. Without
it, "no heavy module found" cannot be distinguished from "the probe is blind" — the failure mode
§2.2.1 describes for controls and `DEC-111` hit for guards.

#### Item 9 — test inventory
**21 `tt/` modules with code, 21 owned.** Full mapping in
`raw/G-CLEAN-item9_20260904T192142Z.log`; the summary:

| `tt/` module | owning test(s) |
|---|---|
| `attention/__init__.py` (class `Attention`, 135 lines — **not** a shim) | `test_attention_vs_ref.py`, `test_kv_cache_tp8.py`, `test_tp_parity.py` |
| `attention/config.py` | `test_decoder_layer_vs_ref.py`, `test_model_vs_ref.py` |
| `attention/dense_sp.py` | `test_dense_sp_vs_ref.py` |
| `attention/kv_cache.py` | `test_kv_cache_vs_ref.py` + 6 more |
| `attention/operations.py` | `test_attention_vs_ref.py`, `test_attention_chunked_vs_ref.py`, `test_kv_cache_tp8.py` |
| `attention/prefill.py` | `test_attention_vs_ref.py` + 4 more |
| `attention/weights.py` | `test_attention_vs_ref.py` |
| `ccl.py` | `test_ccl_semaphores.py` + 7 more |
| `config.py` | `test_mesh_config.py` + 18 more |
| `embedding.py` | `test_embedding_vs_ref.py` |
| `layer.py` | `test_decoder_layer_vs_ref.py`, `test_model_vs_ref.py`, `test_tp_parity.py`, `test_kv_cache_tp8.py` |
| `lm_head.py` | `test_lm_head_vs_ref.py` |
| `mlp.py` | `test_mlp_vs_ref.py`, `test_tp_parity.py` |
| `model.py` | `test_model_vs_ref.py`, `test_weight_loading.py`, `test_kv_cache_tp8.py`, `test_attention_chunked_vs_ref.py` |
| `model_config.py` | `test_weight_loading.py` + 5 more |
| `rms_norm.py` | `test_rms_norm_vs_ref.py`, `test_tp_parity.py` |
| `rope.py` | `test_rope_vs_ref.py` + 6 more |
| `runners/adapters/llama.py` | `test_prefill_adapter.py` |
| `runners/kv_chunk_table.py` | `test_kv_chunk_table.py`, `test_prefill_runtime_chunked.py` |
| `tt_prefill_runtime.py` | `test_prefill_runtime_chunked.py`, `test_prefill_adapter.py`, `galaxy_prefill_kv_pcc.py` |

The three files owning nothing are `tt/__init__.py`, `tt/runners/__init__.py` and
`tt/runners/adapters/__init__.py` — **three lines each, the SPDX pair and a blank**, no code to test.
Item 9 says "close gaps rather than flagging them"; there were none to close.

#### Item 10 — citations, and the phase's largest finding
**629/629** content-checked citations (`CITES` 604 → 629), **1225/1225** doc refs, **146/146** cited
raw artefacts, and a **new fourth pass** that fingerprints the recipe. Clean.

**And the clean report is exactly what had to be checked rather than believed.** Item 10's own gate
text says to expect "un-verified citations in prose that no earlier phase scanned", so P9 content-
triaged every `BRINGUP_RECIPE.md:N` reference in the package: **247 of 317 were wrong**, every one
reported `resolved` by pass 2, which only range-checks. The cause is mechanical: the kit's recipe
grew **1896 → 2235 lines during the run**, so a citation written in P0-P5 points about 235 lines
short of its target today. `R-053` and `DEC-120` carry the method and the numbers; the short version:

- **205** were re-pointed by a provable rule — `git blame` on the citing line names the recipe
  version its author was reading, `difflib` maps the cited line to HEAD, and the move is applied only
  when the old and new text are **byte-identical**. Four of the hand-checked targets landed on lines
  `CITES` independently content-checks, which is the corroboration.
- **A second class survived that**: refs that were **wrong when written**. Sampled after the
  mechanical pass, **5 of 15** were still wrong. So the **125 code-surface refs** (`tt/`, `tests/`,
  `scripts/`, `README.md`) were read one at a time and **57 re-pointed**; a fresh sample of 12 is
  **12/12 correct**. The **202 refs in `bringup_log/`** got the mechanical pass plus the fixes a
  14-ref sample turned up, and their residual error rate is **~20%, measured** — about 40 wrong line
  numbers still in the logs, recorded rather than fixed.
- **Two automated approaches were built and discarded** before the byte-identity rule, and that is
  the reusable part: a version-search variant *oscillated* (it re-moved refs the first pass had just
  fixed correctly), and a token-overlap variant scored short recipe lines at 1.00 because its overlap
  denominator was `min(|a|,|b|)`. Both looked convincing in their summary output.
- **431 abbreviated `` `:NNN` `` continuation refs are checked by nothing** (`R-054`): the verifier's
  regex needs a filename, so they are neither resolved, nor range-checked, nor counted. Two were
  found wrong by hand during the code-surface pass and fixed.
- **A placeholder citation survived all eleven phases.** `05_DECISIONS.md`'s `DEC-055` cited its
  evidence as `` `raw/G-MODEL_<ts>.log` `` — the exact hazard `BRINGUP_RECIPE.md:249-251` warns about
  ("a generic placeholder written in the citation form is indistinguishable from a citation") — and
  both passes were blind to it. Named.
- **The new pass 4** records the recipe's SHA-256 and line count and **fails the gate** if either
  changes, printing the instruction to re-validate every prose ref. That is the mitigation `R-017`
  has wanted since P6 and never had: the recipe was edited out of band **three times** during this
  run, and the third time is why the P10 ledger's `1169` doc refs reproduces as `1168` on the
  committed tree (`R-058`).

#### Item 11 — re-run the suite, and reproduce two recorded numbers
- **Regression:** `pytest models/demos/llama31_8b_d_p/tests -q -p no:randomly` on the final tree,
  after every hook, with `HF_MODEL`, `TT_CACHE_PATH`, `PREFILL_MODEL=llama31_8b_d_p` and
  `PREFILL_TRACE_DIR` at the 1024-token golden. **247 passed, 0 failed** in **22:39** (1359.28 s).
  P10 stood at 246; P9 adds **one** — `test_the_adapter_import_is_cheaper_than_the_template_it_mirrors`
  (`DEC-121`). No other test changed: P9's edits to `tests/` were citation line numbers inside
  docstrings and comments.
- **An earlier run was killed and its log deleted.** It had started before P9's last citation edits,
  so its result would have belonged to a different tree — the same reasoning `DEC-090` applied in P8
  and the P10 session applied to its first attempt. An incomplete regression is not evidence.
- **`R-041` bit for the third time**, in the same place: the run rewrites
  `raw/G-CHUNK_per_layer_pcc.json` with 1024-token content while P7's ledger row cites a 512-token
  measurement. Backed up before and restored after, and the restored files are byte-identical to the
  committed copies (`cmp` clean on all four). The workaround has now been needed in P8, P10 and P9;
  the kit defect is real and is `R-041`.
- **Numerical gate re-run — `G-MOCK-MIG` arm 2**, chosen because it is the branch that found
  `DEC-111` and the one no document mentions (`PREFILL_ENABLE_MIGRATION=1` +
  `PREFILL_MOCK_MIGRATION=1` takes the engine's real stage-gather path and calls
  `kv_migration_base_address` with no worker binaries). **Every recorded digit reproduced:**
  `KV cache PCC PASSED (min 0.986623 >= 0.93 across 1 slots; per cache: k=0.996784, v=0.986623)`
  over `[0,1024)` across **32/32** local layers, per-layer min **K 0.996784 at L22** and
  **V 0.986623 at L28** — the same values *and* the same argmin layers as P10 recorded and as
  `G-MESH-KV` measured on device. The table built 16 configs / 45,056 entries; the LayerAck channel
  registered 32 acks per chunk; producer rc 0, runner rc 0.
  Raw: `raw/G-MOCK-MIG-P9RERUN-producer_20260904T193238Z.log`,
  `raw/G-MOCK-MIG-P9RERUN-runner_20260904T193238Z.log.gz`, `raw/G-RMS-P9RERUN_20260904T195714Z.log` (gzipped, 848 KB > the hook's 500 KB limit;
  compression is lossless so the evidence stays byte-exact).
- **Second numerical gate — `G-RMS` in isolation.** Item 11 asks for **two**, so the cheapest gate
  with recorded digits was re-run on the final tree: 10/10 tests, 13.0 s. Every digit reproduced,
  including the floors and the ratios: random weights **0.9999957 / 0.9999958 / 0.9999957** against
  floors 0.9999973 / 0.9999973 / 0.9999972 -> **1.56 / 1.57 / 1.54x**; real layer-0 gain
  **0.9999971 x3** against 0.9999986 -> **2.11 / 2.13 / 2.12x**; zero-gain control
  `max|out| = 0.0`; and the in-suite `fp32_dest_acc_en` A/B still shows the flag worth
  **6.90 / 8.66x** at seq 32 / 512. Raw: `raw/G-RMS-P9RERUN_20260904T195714Z.log`.
- **The citation verifier is the third re-run**, after every edit: it moved from 604/604 to
  **629/629** as P9 promoted 25 references into `CITES`, and never reported a mismatch it did not
  earn — it caught P9's own off-by-one on `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py`
  (P9 wrote `:26`; the import is on `:25`) and the two deliberate placeholders in this very block,
  which is the artefact pass P8 added doing precisely its job.

#### What the sweep found in the **logs**, which is where the recipe says to expect it
Six items, five of them in the logs and one in the code (item 5's docstring). In severity order:

1. **`R-053` — 247 of 317 prose recipe citations wrong**, every one reported `resolved`. Item 10.
2. **`R-054` — 431 abbreviated `` `:NNN` `` refs checked by nothing.** Item 10.
3. **A placeholder in the citation form survived eleven phases.** `DEC-055`'s evidence line read
   `` `raw/G-MODEL_<ts>.log` ``. `BRINGUP_RECIPE.md:249-251` warns about exactly this ("a generic
   placeholder written in the citation form is indistinguishable from a citation") and both passes
   were blind to it. Now names `raw/G-MODEL_20260904T115256Z.log`.
4. **`07_RISKS.md`'s own header was stale.** It said "Re-checked at every phase boundary (last: end
   of P8)" through the whole of P10, while P10 added `R-041` … `R-052` to it. Corrected to P9, with
   the miss noted in place.
5. **A P10 finding about the kit is itself false, and P9 could only tell by looking.** P10's closing
   note records that `LANDMINES.md`'s new row "was appended after a blank line following the 'Repo
   hooks' table, with **two** cells where that table has three … it therefore renders as its own
   separate two-column table" and that its content belongs in "Method traps". On the committed tree
   the row is at `models/demos/common/bringup/LANDMINES.md:59` — **inside** the two-column "Method
   traps" table, third row, correctly formed and in the right place. Either the kit was fixed
   between P10's reading and the commit, or the note was written from the diff rather than from the
   file. **It is retracted here.** The lesson is the one `R-016` keeps making: a claim about another
   file has to be read out of that file, and a phase that reports on a dependency it does not own
   should re-check the claim against the committed state.
6. **`06_GATES.md`'s `G-OUTLINE` row and `DEC-109` both say the P3 tree contracts 41 files**; it
   contracts 57, and 57 were delivered. Both are P3-era measurements in append-only files and are
   left as written; the correction is in `03_OUTLINE.md` §1's `P9 check` note. The P10 hand-over
   asked P9 to fix "41" in `03_OUTLINE.md:102` — that sentence already said **57**, so the
   hand-over item was itself misdirected at the wrong file.

### P9-REGRESSION and G-CITE (P9) — the two cross-cutting rows

| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| P9-REGRESSION | P9 | the whole package suite still passes after P9's own edits | 0 failed | **247 passed, 0 failed** in 22:39 (1359.28 s). P10 stood at 246; P9 adds **one** test, the import-cost ratio (`DEC-121`). An earlier run was **killed and its log deleted** because it had started before P9's last citation edits — `DEC-090`'s reasoning | PASS | 2026-09-04 | `raw/P9-REGRESSION_20260904T193410Z.log.gz` |
| G-CITE (P9) | P9 | every `path:line`, every doc ref and every cited raw artefact resolves — **and, new in P9, the recipe those refs point into is pinned** | 0 mismatched, 0 unresolved, 0 missing artefacts | **629/629** content-checked (`CITES` 604 -> 629: 25 new rows, 20 of them the recipe lines P9 re-pointed onto, 4 the README's load-bearing refs, 1 the template import that item 8 rests on), **1225/1225** doc refs, **146/146** raw artefacts, and pass 4 reports the recipe fingerprint **MATCHES**. What it caught this phase is in item 10: **247 of 317** prose recipe refs were wrong and all of them had been reported `resolved` by pass 2 | PASS | 2026-09-04 | `raw/G-CLEAN-item10_20260904T195811Z.log` |

```
STATUS after P9: gates PASS=29 FAIL=0 DEVIATION=2 OUT-OF-SCOPE=1 BLOCKED=0 | next: none — P9 is the last phase
Open DECs needing review: DEC-120 (247 prose recipe citations re-pointed mechanically; the log files'
residual error rate is ~20%, measured, and unfixed), DEC-123 (SPDX exempts two JSON files and the log
markdown), DEC-125 (a P10 finding about LANDMINES.md retracted)
```

`PASS=29` is P10's 28 plus `G-CLEAN`. `DEVIATION=2` (`G-ATTN`, `G-FABRIC-MATRIX`) and
`OUT-OF-SCOPE=1` (`G-LOOPBACK`, `DEC-103`) are unchanged. The three cross-cutting rows —
`P9-REGRESSION`, `G-CITE (P9)` and the per-phase regression before them — sit outside the
Appendix A tally, as in every earlier phase.

**Appendix C, item by item, at sign-off:**

| # | Definition of done | State |
|---|---|---|
| 1 | the P3 tree, no dead files | **57/57**, file-for-file; 0 TODO markers, 0 dead branches, 0 commented-out code |
| 2 | every Appendix A gate `PASS` / `PASS-WITH-DEVIATION` / scoped out, with raw logs | **31/31** gate names have a ledger row and a raw log; 2 deviations with a `DEC`, 1 scoped out with a `DEC` and a named risk |
| 3 | `G-LOOPBACK` run, blocked, or out of scope by a `DEC` with a named residual gap | **out of scope** (`DEC-103`), residual gap `R-043` |
| 4 | `bringup_log/` reads as a coherent narrative | 125 `DEC` entries, 58 risks, every judgement reconstructable — **but see the caveat below** |
| 5 | `README.md` carries measured PCC, the `models/common/` answer, and "not implemented" | complete, plus a "known-imperfect" section (`DEC-124`) |
| 6 | `07_RISKS.md` lists every gap with an owner, and its table agrees with its body | **58 rows / 58 sections**, checked mechanically; 4 deliberate supersede sections; 0 dangling `DEC-` or `R-` references anywhere in the logs or the README |
| 7 | `verify_citations.py` reports 0 mismatched and 0 unresolved | **it does — and item 10 is the reason that is not sufficient.** 247 refs were wrong while this item was satisfied at every phase boundary. **Appendix C item 7 is a check on the verifier, not on the citations**, and a package can meet it with a fifth of its references pointing at unrelated text |

**The caveat on item 4, stated plainly because this is the last review.** The narrative is complete
and it is *long*: `05_DECISIONS.md` is 4,320 lines, `06_GATES.md` 2,859, `07_RISKS.md` 1,270 — **10,352
lines of log** against **4,681 lines of `tt/`** and 11,142 of `tests/`, with no index and no summary of what a reader
should read first. Every individual entry earns its place; the *set* is past the size where a new engineer
will read it. That is a finding about the method, not about this run: the recipe mandates the entries
and says nothing about navigability.

**STOPPED HERE. P9 is the last phase, and the run is complete.** Every gate in Appendix A has a
verdict, a number and a raw log. `G-CLEAN` is **PASS** on all eleven items. The suite is
**247 passed, 0 failed**; citations are **629/629 · 1225/1225 · 146/146** with the recipe
fingerprinted. The two human gates this phase owns, `H6` and `H7`, **ran on their defaults** and are
`R-056` and `R-057`: nothing here is a sign-off, and 20 defects outside this package are logged with
0 filed.
