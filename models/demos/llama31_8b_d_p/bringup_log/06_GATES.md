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
  (`BRINGUP_RECIPE.md:857-863`).
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
  (`BRINGUP_RECIPE.md:280-281`) — but the two mechanical checks both fired before they passed, which
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
  *resolved* rather than obeyed (`03_OUTLINE.md` §5.1: `BRINGUP_RECIPE.md:1022-1024` requires both
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
  (`BRINGUP_RECIPE.md:975-981`).
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
    identical to `BRINGUP_RECIPE.md:925-927`.
  - All three BH-galaxy torus mesh-graph descriptors named in §7 exist (listed in the raw log).
  - `verify_citations.py`: `citations checked 253 / verified 253 / mismatched 0 / missing 0`;
    `doc refs scanned 404 / resolved 404 / unresolved 0`; exit 0. `CITES` grew by **28** this phase
    (225 → 253).
  - per-phase regression: `10 passed` in 13.77 s — unchanged.
- **Verdict:** **PASS**
- **Negative control:** doc gate, so §1.4's four numeric fields are waived
  (`BRINGUP_RECIPE.md:280-281`). The structure check is the control and it **failed its first run**,
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
  `mesh_device` fixture (`BRINGUP_RECIPE.md:1027`).
- **Input distribution:** n/a — this gate has no numeric input. Its inputs are mesh shapes:
  `(1,1)`, `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)`, `(8,4)`, and the four refused
  `(mesh, tp)` pairs below.
- **Reference dtype policy:** n/a — no reference tensor. The "reference" is arithmetic stated in
  `00_MODEL_CARD.md` §4 and re-derived from the bundled `config.json` in the test rather than
  restated as a literal.
- **Threshold:** exact asserts (`BRINGUP_RECIPE.md:1726`). No PCC, so §1.4's floor field does not
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
     configuration that must *refuse* as a control (`BRINGUP_RECIPE.md:277-278`).
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
  (`BRINGUP_RECIPE.md:1025-1026`) and writing it twice would let the two copies disagree
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
- **Threshold:** PCC >= 0.9999 (`BRINGUP_RECIPE.md:1728`). The error ratio is **recorded, not
  asserted**: a correct module sits right on §2.2's 3x stage bound, so asserting it would gate on
  the wrong side of the noise (`BRINGUP_RECIPE.md:1043-1046`).
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
- **Threshold:** PCC >= 0.999 (`BRINGUP_RECIPE.md:1729`), expecting ~0.99999.
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
     (`BRINGUP_RECIPE.md:1093-1095`), and measured, `max|cos_scaled - cos_unscaled|` is
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
   computed floor at each** (`BRINGUP_RECIPE.md:1730`). Unlike `G-RMS`, that ratio bound is
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
  dtype** (`BRINGUP_RECIPE.md:1769`). Unlike `G-RMS`, Appendix A states a ratio bound for this gate,
  so the ratio is **asserted**, not merely recorded.
- **Measured:** 14/14 tests pass.
  - bf8_b: **0.9999133 / 0.9999144 / 0.9999144** -> **1.10x / 1.10x / 1.10x** the floor
  - bf16: **0.9999851 / 0.9999852 / 0.9999852** -> **2.11x / 2.10x / 2.09x** the floor
  - Both dtypes run and both are recorded, as `BRINGUP_RECIPE.md:1769` requires. bf8_b clears its
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
    deployment never uses" caveat (`BRINGUP_RECIPE.md:576-578`) applied to a collective rather than
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
  (`BRINGUP_RECIPE.md:1770`). See **Deviations** for how the block budget is applied.
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
  **This is a discrepancy with the recipe, and in the safe direction:** `BRINGUP_RECIPE.md:1214`
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
  (`BRINGUP_RECIPE.md:1771`); the layout claims on **bit-equality** (`torch.equal`, `rtol=atol=0`),
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
    (`BRINGUP_RECIPE.md:1241-1243`).
  - **The dtype delta `DEC-021` owed:** bf8_b K **0.9999756** / V **0.9999757** versus bf16
    **0.9999986** / **0.9999986** — **17.8x** the error on K and **17.7x** on V, for half the bytes
    (128 vs 256 per token per head). Measured, not assumed; `DEC-021` stands.
- **Negative controls / refusals — five, all fired:**
  1. **The bit-exactness assertion itself is the control for the position map**, and it *caught a
     real failure*: the first version of the probe, built to `BRINGUP_RECIPE.md:1260-1262`'s stated
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
is `PASS-WITH-DEVIATION` (`DEC-042`), so `BRINGUP_RECIPE.md:1267-1269`'s "all of G-MESH, G-RMS,
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
- **Threshold:** PCC ≥ 0.999 and ≤ 8x the floor (`BRINGUP_RECIPE.md:1372`, Appendix A `:1854`),
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
  (`BRINGUP_RECIPE.md:1375`): it cannot localise a sublayer fault, and it may not substitute for
  `G-RMS`/`G-ROPE`/`G-MLP`/`G-ATTN`, all of which are met on their own. It also runs at TP=1, so
  neither module's TP collective has ever executed (P8), and it writes no KV cache — at TP=1 the
  packed cache refuses the model's 8 local KV heads outright (`00_MODEL_CARD.md` §4.1, `R-001`).

### G-WEIGHTS — real-checkpoint weight loading, bit-exact
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py -x -q`
- **Mesh / device:** one card, `(1,1)`. Cache-only at TP > 1 is the P8 extension
  (`BRINGUP_RECIPE.md:1411`).
- **Inputs (distribution):** not a synthetic distribution — the inputs **are** the real
  Llama-3.1-8B-Instruct checkpoint tensors at their stored dtype, `torch.bfloat16` (measured and
  logged by `torch_dtype_of`, matching `00_MODEL_CARD.md` §2). Stated rather than omitted because
  §1.4 requires it.
- **Reference dtype policy:** the **same torch tensor object** drives both sides — the device path
  and `quantize_like_device` — with **no fp32 detour**, so a bf16→bf8_b vs fp32→bf8_b double-rounding
  difference cannot be mistaken for a loader fault. Every comparison is `torch.equal`
  (`rtol = atol = 0`), never PCC: recipe §2.5 measured a completely wrong mapping still scoring
  PCC 0.99890, and a transpose or swizzle applied twice is that class of bug
  (`BRINGUP_RECIPE.md:1408-1409`).
- **Noise floor:** not applicable, and that is the point — a bit-exactness gate has no floor because
  the tolerance is zero. The nearest equivalent, recorded instead: `max|Δ| = 0.000e+00` on all
  twelve tensors.
- **Threshold:** exact, in three parts (`BRINGUP_RECIPE.md:1404-1411`).
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
  both make the verifier exit non-zero (`BRINGUP_RECIPE.md:1588-1590`).
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
  and the same `head_dim = 128` a chip performs at TP=8 (`BRINGUP_RECIPE.md:1574`).
- **Inputs:** the golden trace's own 512 `token_ids` — real prompt tokens, real checkpoint weights.
  The hidden states the producers see are the ones the assembled model actually produces, i.e. the
  real-scale arm §2.2.2 and `R-018` say predicts model behaviour, not a synthetic one.
- **Reference dtype policy:** the fp32 golden (`G-GOLDEN`, `DEC-059`). The device's K is
  Meta-swizzled over `head_dim` (the loader `reverse_permute`s Q/K), so **the golden and the floor
  are permuted HF → Meta before comparison, and the permutation is applied *before* quantising** —
  `bfloat8_b` shares one exponent per 16-element block of the last dim, so the block boundaries move
  with the permutation. V is not swizzled and is compared as-is.
- **Thresholds (`BRINGUP_RECIPE.md:1578-1584`):** mutual PCC ≥ **0.999 per layer** (expected exact);
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
- **Mesh / device:** **none** (`BRINGUP_RECIPE.md:1954` gives this gate device "none"). 37 tests,
  8.2 s, no mesh opened. The two `__init__` refusals are reached with a `_MeshStub` exposing only
  `.shape` — both run before any `ttnn` object is constructed — and the per-chunk refusals with an
  `object.__new__` instance carrying only the three attributes those checks read (`DEC-068`). The
  code under test is the real method on the real class with the real messages; only the state it
  reads is supplied directly.
- **Inputs:** the engine's own source, parsed with `ast`. **Not the contract doc** — that is the
  whole point of the gate (`BRINGUP_RECIPE.md:1593-1596`).
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
  golden (`BRINGUP_RECIPE.md:1958`).
- **Verdict:** **BLOCKED** — `07_RISKS.md` `R-023`, which names **P8** as the owner, as
  `BRINGUP_RECIPE.md:1598-1600` requires.
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
`BRINGUP_RECIPE.md:1598-1600` requires by construction, not as a concession.

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
`BRINGUP_RECIPE.md:1732-1734` requires — and it is the reason the rest of the phase is configured the
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
  (`BRINGUP_RECIPE.md:1735-1737`). `Topology.Linear`, `FABRIC_1D`, `num_links=1`.
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
- **Threshold, and it names a depth** (`BRINGUP_RECIPE.md:1745-1749`): **>= 0.999 at layer 1** — one
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
  `BRINGUP_RECIPE.md:1725-1728`), which is the shape of thing §0.3's H5 says to investigate rather
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
BRINGUP_RECIPE.md:1768-1770), DEC-085 (meta_head_index duplicated between the script and the tests),
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
  not happen", `BRINGUP_RECIPE.md:199`; Appendix C item 2), so a ledger row citing a file that is not
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
