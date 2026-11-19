# Stage5 anomaly ledger

| Observed anomaly | Evidence and control | Affected path / investigation | Resolution |
|---|---|---|---|
| First indexer candidate had unchanged latency | `parallel-indexer_l3` versus `parallel_indexer_v2_l3`; exact component control | Wrong fused constructor binding; corrected actual MeshQSA binding and explicitly enabled valid-row/copy-once behavior | Fixed; first run excluded |
| Expanded DRAM split-QSA block40 exceeded L1 | `dram_geometry_qsa_v2` log: 1,998,784 requested vs1,572,864 available; `dram_geometry_v3_qsa` | TTNN program scratch at one reader; adapted smaller legal blocks and swept8/16 shards,1/2/3 readers; GR block40 separately legal and measured | Controlled physical limit; family retained |
| DRAM GDN was slower despite faster component matmuls | `dram_ablation_gdn_l0` versus `dram_compact_gdn_l0` | Per-output padding and runtime slice/concat; packed logical Z/A/B together before upload | Fixed; compact implementation selected |
| Hidden QSA adapter raised host Shape slice TypeError | First hidden candidate log versus `hidden_qsa_rs/agmm` | Convert Shape to tuple before host shape slicing; repeat both coherent whole-decoder families | Fixed; families then rejected by B32 latency, not API failure |
| Legacy B32 probe rejected a diagnostic row despite whole-output pass | `real_dram_gdn-compact_LoFi_native`: whole.99907094, minrow.99447459 | Old per-row.995 check contradicted explicit whole-output contract; acceptance now uses whole-output.995 and retains every row diagnostic | Fixed harness contract; historical nonzero preserved; not a whole-output accuracy rejection |
| Real BFP4 attention fails quality | `real_qsa_bfp4`: short minimum.96450335, B32whole.98230147; native BF8 controls pass | Real checkpoint weights and real embedding fixture; also earlier seeded-input failure | Controlled; candidate rejected, no new BF4 default |
| BFP8 KV passes but slows decode | `real_bfp8_kv` versus `hidden_stream_control`; exact selected-row unpack component | B1+12.1%, B32+0.5%, selected BF8 page-reader overhead included | Controlled; BF16 KV retained |
| Shared packed/split phases retain two weight representations | `final_context_memory_plan.json` versus single-representation alternative | Weight-cache key includes packing; counted both prepared buffers | Controlled; final same packed policy shares one representation |
| Probe import failed during concurrent CPU edit | `dram_packed_shared_l0.log`, successful `_v2` | Syntax error occurred before device open; queue stopped, source compiled/formatted, CPU batch-branch checks passed before restart | Fixed; no hardware dispatch occurred |
| Real fixture PLE reference had wrong token rank | Initial real layer1 reference log versus `real_baseline_before_l1_v2` | Public token words must be[1,1,16]; corrected preparation | Fixed before candidate timing |
| Global BFP8 async payload lowered diagnostic row PCC and was slower | `async-bfp8_l0` versus `before_l0`; subgroup real transport trials | Conversion cost included, ordered FP32 reduction preserved | Controlled; BF16 synchronous transport retained |
| Tracy warns that its optional WASM-viewer copy source is missing | `profile_final_mesh_l0.log`; original `.tracy` and generated ops CSV exist under the explicitly selected external profile root | Upstream viewer-copy uses the default generated/profiler path although capture uses `-o`; no web server was requested | Controlled; actual capture/report retained, no data-loss claim |
| Profiler segmentation adds drain gaps | Capture wrapper source and final profile provenance | Device op attribution uses explicit drains; headline latency comes from clean warmed full traces | Controlled accounting; no drain-gap speedup claim |

All failed device runs were closed and reset/list/collective-smoked as recorded
in work_log.md. There was no hang, unresolved watcher assertion, or failed
hardware recovery at selection freeze. Final gate outcomes and independent
review are recorded separately; this ledger does not replace them. Routine
host fixes and measured candidate rejections did not require AutoFix. An
unresolved final gate would require repair, not a waiver from this ledger.
